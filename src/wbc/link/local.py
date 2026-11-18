"""Processes this machine starts for a session (the local simulator stack)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from wbc.core import target as targets
from wbc.core.errors import ErrorCode, WbcError
from wbc.core.process import signal_group, wait_for_file


def start_simulator(directory: Path, target: str, timeout: float):
    """Launch ``wbc_link_server.launch.sim`` for a
    ``sim://SCENE?seed=&heading=&paused=&vision=&viewer=&checkpoint=`` target (``wbc.core.target``).

    The launcher leads its own process group; returns it and its endpoints.
    """
    options = targets.options(target)
    scene = targets.scene(target) or "flat"
    endpoint_file = directory / "endpoints.json"
    command = [
        sys.executable,
        "-m",
        "wbc_link_server.launch.sim",
        "--directory",
        str(directory),
        "--scene",
        scene,
        "--seed",
        options.get("seed", "0"),
        "--heading-deg",
        options.get("heading", "0"),
        "--endpoint-file",
        str(endpoint_file),
    ]
    if targets.paused(target):
        command.append("--paused")
    if not targets.vision(target):
        command.append("--no-vision")
    if targets.viewer(target):
        command.append("--viewer")
    if options.get("checkpoint"):
        command += ["--checkpoint", options["checkpoint"]]
    if scene == "locomanilab":
        if not options.get("task"):
            raise WbcError(ErrorCode.INVALID_REQUEST, "sim://locomanilab requires a task query parameter")
        command += ["--benchmark-task", options["task"]]
    directory.mkdir(parents=True)
    with (directory / "launcher.log").open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)

    def running():
        if process.poll() is not None:
            raise WbcError(ErrorCode.UNREACHABLE, f"simulator exited; see {directory}")

    try:
        if not wait_for_file(endpoint_file, timeout, running, poll=0.1):
            raise WbcError(ErrorCode.UNREACHABLE, "simulator did not start in time")
        return process, json.loads(endpoint_file.read_text())
    except BaseException:
        # Session.connect only receives ownership after this function returns.
        # Signals and a malformed endpoint therefore must unwind it here. A
        # simulator that is still starting needs no long grace, and a failure to
        # stop it must not hide the original error.
        with suppress(Exception):
            stop_process_group(process.pid, grace=5.0)
        raise


def stop_process_group(pid: int, grace: float = 60.0) -> None:
    """Stop a process group we started, escalating to SIGKILL."""
    if not signal_group(pid, signal.SIGTERM):
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not signal_group(pid, 0):
            break
        with suppress(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        time.sleep(0.1)
    else:
        signal_group(pid, signal.SIGKILL)
