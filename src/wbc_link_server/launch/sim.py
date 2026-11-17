"""Start a complete local simulator stack and keep it alive.

Processes (each in its own process group, all tied to this launcher's life;
on SIGTERM the launcher stops their groups in order, the controller with SIGINT
so it runs its damping and saves its log):
  bridge   MuJoCo robot speaking the controller's UDP protocol, on a scene of wbc_link_server.scenes
  vision   ego RGB-D from the simulator (unless --no-vision)
  policy   the controller (deploy.py) tracking the motion channel
  control  the robot link server; restarted on the same port if it exits (the controller keeps running)
  viewer   with --viewer: an interactive window on the simulation; the stack does not depend on it

When everything is up, writes {"control": URL, "vision": URL|null} to --endpoint-file. ``--heading-deg`` turns the
robot's start.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from wbc.core import logs
from wbc.core.jsonhttp import write_json_atomic
from wbc.core.process import die_with_parent, signal_group, wait_for_file

log = logging.getLogger("wbc_link_server.launch.sim")

STARTUP_TIMEOUT = 90.0
CONTROL_RESTARTS = 3  # within CONTROL_RESTART_WINDOW_S: give up and stop the stack
CONTROL_RESTART_WINDOW_S = 30.0
STOP_GRACE_S = 10.0  # for each process group to exit on its signal before it is killed
# Processes that never render import MuJoCo without a GL backend: loading EGL initializes the GPU driver,
# which serializes across processes and slows the start of concurrent stacks.
NO_GL = {"MUJOCO_GL": "disable"}
UDP_PORTS = (20000, 32000)  # simulator/controller UDP numbers: below Linux's ephemeral range (32768-60999)
CAMERA = "wbc_link_server.sensors.vision"  # the module run as the camera process
VIEWER = "wbc_link_server.sensors.viewer"  # the module run as the window, with --viewer


def module(name: str, *arguments, python: str | None = None) -> list[str]:
    """Run module ``name`` with the selected interpreter (this Python by default)."""
    return [python or sys.executable, "-m", name, *map(str, arguments)]


def lease_directory(name: str) -> Path:
    directory = Path(tempfile.gettempdir()) / f"{name}-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


def try_lease(path: Path):
    """An exclusive lock on ``path``, not waiting: the open lock file (closing it releases the lock), or None while
    another launcher holds it. Lock files stay in place, so all launchers lock the same inode."""
    lease = path.open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    return lease


class Stack:
    def __init__(self, directory: Path):
        self.directory = directory
        self.processes: dict[str, subprocess.Popen] = {}
        self.leases = []  # open lock files (the UDP numbers), held for the stack's lifetime
        self.stopping = False

    def udp_ports(self) -> tuple[int, int]:
        """Lease both UDP numbers through startup and the stack's lifetime.

        Children bind the ports themselves. The short bind/close gap therefore
        needs a cooperative lease: concurrent launchers must not reuse a state
        port as another robot's command port before its receiver has started.
        Keep lock files in place so all launchers lock the same inode.
        Numbers come from below the kernel's ephemeral range: a port the kernel
        hands out to any other socket in the gap would otherwise collide.
        """
        directory = lease_directory("wbc-sim-udp")
        rng = random.Random()
        ports = []
        while len(ports) < 2:
            port = rng.randrange(*UDP_PORTS)
            if port in ports:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                lease = try_lease(directory / str(port))
                if lease is None:
                    continue
                self.leases.append(lease)
                ports.append(port)
        return tuple(ports)

    def spawn(self, name: str, argv: list[str], env: dict | None = None) -> subprocess.Popen:
        """Start ``argv`` as process ``name`` in its own process group (``stop`` signals it), with ``env`` over this
        process's environment; its output goes to ``name.log``."""
        with (self.directory / f"{name}.log").open("ab") as output:
            process = subprocess.Popen(
                [str(a) for a in argv],
                env={**os.environ, **env} if env else None,
                stdout=output,
                stderr=subprocess.STDOUT,
                preexec_fn=die_with_parent,
                start_new_session=True,  # its own group: a signal to the launcher's group reaches only the launcher
            )
        self.processes[name] = process
        write_json_atomic(self.directory / "processes.json", {n: p.pid for n, p in self.processes.items()})
        return process

    def wait_for(self, path: Path, name: str) -> dict:
        """Wait for the file process ``name`` writes when it is ready (STARTUP_TIMEOUT s); its JSON."""

        def running():
            if self.stopping:
                raise RuntimeError("stopped during startup")
            if self.processes[name].poll() is not None:
                raise RuntimeError(f"{name} exited during startup; see {self.directory / (name + '.log')}")

        if not wait_for_file(path, STARTUP_TIMEOUT, running):
            raise TimeoutError(f"{name} did not start; see {self.directory / (name + '.log')}")
        return json.loads(path.read_text())

    def check(self, *names: str) -> None:
        for name in names:
            if self.exited(name):
                raise RuntimeError(f"{name} exited; see {self.directory / (name + '.log')}")

    def exited(self, name: str) -> bool:
        return self.processes[name].poll() is not None

    def stop(self) -> None:
        """Signal each running process's group in order and wait for its leader; a group still running after
        STOP_GRACE_S is killed. The policy gets SIGINT so deploy.py runs its damping/shutdown path."""
        for name, sig in (
            ("viewer", signal.SIGTERM),
            ("control", signal.SIGTERM),
            ("policy", signal.SIGINT),
            ("vision", signal.SIGTERM),
            ("bridge", signal.SIGTERM),
        ):
            process = self.processes.get(name)
            if process is None or process.poll() is not None:
                continue
            signal_group(process.pid, sig)
            try:
                process.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        write_json_atomic(self.directory / "exit.json", {n: p.returncode for n, p in self.processes.items()})
        for lease in self.leases:
            lease.close()
        self.leases.clear()


def tend_control(stack, restart, restarts: deque) -> None:
    """Restart an exited link server; it holds no state the controller needs. Give up if it keeps exiting."""
    if not stack.exited("control"):
        return
    restarts.append(time.monotonic())
    if len(restarts) == restarts.maxlen and restarts[-1] - restarts[0] < CONTROL_RESTART_WINDOW_S:
        raise RuntimeError(f"control keeps exiting; see {stack.directory / 'control.log'}")
    log.warning("control exited (code %s); restarting", stack.processes["control"].returncode)
    restart()


def bridge_arguments(args, directory, state_port, command_port) -> list:
    """The bridge's simulator directory, the controller's channel, the UDP numbers and ``--paused``."""
    return [
        "--directory",
        directory / "sim",
        "--channel",
        directory / "controller" / "channel",
        "--state-port",
        state_port,
        "--command-port",
        command_port,
        *(["--paused"] if args.paused else []),
    ]


def start_camera(stack, args, directory, *, python=None):
    """The camera process on the bridge's MuJoCo scene (unless --no-vision); the vision URL or None."""
    if args.no_vision:
        return None
    stack.spawn(
        "vision",
        module(CAMERA, "--directory", directory / "sim", "--endpoint-file", directory / "vision.json", python=python),
    )
    return stack.wait_for(directory / "vision.json", "vision")["vision"]


def start_mujoco(stack, args, directory, state_port, command_port):
    """The MuJoCo bridge on one of our scenes and (unless --no-vision) its camera process; the vision URL or None."""
    if args.scene == "locomanilab":
        return start_locomanilab(stack, args, directory, state_port, command_port)
    stack.spawn(
        "bridge",
        module(
            "wbc_link_server.bridges.sim_bridge",
            *bridge_arguments(args, directory, state_port, command_port),
            "--scene",
            args.scene,
            "--seed",
            args.seed,
            "--heading-deg",
            args.heading_deg,
        ),
        env=NO_GL,
    )
    stack.wait_for(directory / "sim" / "bridge.json", "bridge")
    if args.viewer:  # a window on the bridge's scene; it needs a display, and nothing waits for it
        stack.spawn("viewer", module(VIEWER, "--directory", directory / "sim"), env={"MUJOCO_GL": "glfw"})
    return start_camera(stack, args, directory)


def start_locomanilab(stack, args, directory, state_port, command_port):
    """The native benchmark owns physics; HITCH policy/control keep their environment."""
    python = os.environ.get("WBC_BENCHMARK_PYTHON")
    if not python or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError("set WBC_BENCHMARK_PYTHON to the executable native benchmark Python")
    if not args.benchmark_task:
        raise ValueError("locomanilab requires --benchmark-task")
    if args.heading_deg != 0:
        raise ValueError("locomanilab uses the native task reset heading; omit --heading-deg")
    stack.spawn(
        "bridge",
        module(
            "wbc_link_server.bridges.locomanilab_bridge",
            *bridge_arguments(args, directory, state_port, command_port),
            "--benchmark-task",
            args.benchmark_task,
            "--seed",
            args.seed,
            python=python,
        ),
        env=NO_GL,
    )
    stack.wait_for(directory / "sim" / "bridge.json", "bridge")
    if args.viewer:
        stack.spawn(
            "viewer", module(VIEWER, "--directory", directory / "sim", python=python), env={"MUJOCO_GL": "glfw"}
        )
    return start_camera(stack, args, directory, python=python)


def start_policy(stack, args, directory, state_port, command_port) -> None:
    """The controller (launch.policy), on the start state the bridge wrote."""
    stack.spawn(
        "policy",
        module(
            "wbc_link_server.launch.policy",
            "--backend",
            "sim",
            "--directory",
            directory / "controller",
            "--initial",
            directory / "sim" / "initial.npz",
            "--state-port",
            state_port,
            "--command-port",
            command_port,
            *(["--paused"] if args.paused else []),
            *(["--checkpoint", args.checkpoint.resolve()] if args.checkpoint else []),
        ),
        env=NO_GL,
    )


def start_control(stack, directory, port) -> str:
    """The link server on ``port`` (0: any free one); its URL."""
    (directory / "control.json").unlink(missing_ok=True)
    stack.spawn(
        "control",
        module(
            "wbc_link_server.launch.control",
            "--attach",
            directory / "controller",
            "--port",
            port,
            "--wait",
            STARTUP_TIMEOUT,
            "--endpoint-file",
            directory / "control.json",
        ),
        env=NO_GL,
    )
    return stack.wait_for(directory / "control.json", "control")["control"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--scene", default="flat")
    p.add_argument("--benchmark-task", help="exact native LocoManiLab task ID")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heading-deg", type=float, default=0.0, help="the robot's start heading")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--viewer", action="store_true", help="open an interactive window (needs a display)")
    p.add_argument("--checkpoint", type=Path, help="controller bundle with deploy.yaml and ONNX")
    p.add_argument("--port", type=int, default=0, help="link server port (default: any free port)")
    p.add_argument("--endpoint-file", type=Path, required=True)
    args = p.parse_args()
    directory = args.directory.resolve()
    logs.setup("sim", logs.session_of(directory))
    directory.mkdir(parents=True, exist_ok=True)
    stack = Stack(directory)

    def stop(*_):
        stack.stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        state_port, command_port = stack.udp_ports()
        vision = start_mujoco(stack, args, directory, state_port, command_port)
        start_policy(stack, args, directory, state_port, command_port)
        control = start_control(stack, directory, args.port)
        write_json_atomic(args.endpoint_file, {"control": control, "vision": vision})
        restarts: deque[float] = deque(maxlen=CONTROL_RESTARTS)
        port = urlparse(control).port
        viewer_gone = False
        while not stack.stopping:
            stack.check("bridge", "policy")
            if args.viewer and not viewer_gone and stack.exited("viewer"):
                viewer_gone = True
                log.warning("the viewer window is gone (see %s); the simulation goes on", directory / "viewer.log")
            tend_control(stack, lambda: start_control(stack, directory, port), restarts)
            time.sleep(0.2)
    finally:
        stack.stop()


if __name__ == "__main__":
    main()

