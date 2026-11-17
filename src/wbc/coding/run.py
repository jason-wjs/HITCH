"""Bounded Codex episodes sharing HITCH's persistent Tools and simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from wbc.agent import prompts
from wbc.agent.loop import _check, _robot_down, _scene, _scene_task
from wbc.coding.broker import RobotBroker
from wbc.coding.gateway import Gateway
from wbc.session import client

HERE = Path(__file__).parent

GUIDE = """# Python robot interface

Use `import robot` in saved Python programs. Functions return JSON-compatible dictionaries:
- `robot.observe()` obtains the latest body state and head-camera images.
- `robot.perceive(points, observation_id=obs['observation_id'])` measures image points.
- `robot.execute(request, observation_id=obs['observation_id'])` plans, executes and observes.
- `robot.finish(success, summary)` records your final self-assessment.

The result includes text, structured data, observation_id and local image paths. Read returned
images with the image-view tool; printing their paths does not show them to the model.
Use the most recent observation_id. Never reuse coordinates after walking: observe and measure
again. Use the motion request and perception formats documented below. Every call executes in the
same persistent robot session, even when separate Python processes make the calls.

Develop the task solution as Python files in /workspace. Execute a short, inspectable stage at a
time, inspect images and structured feedback, and revise the program based on observed results.
Preserve useful measurements and reasoning in files. Do not blindly repeat an entire script that
already moved the robot. Your programs may calculate geometry and branch on returned values.
Before any contact-sensitive move, use current measurements. Call finish honestly, including
failure if appropriate; do not infer a grasp solely from a closed hand.

You can use Python standard library, NumPy and Pillow. There is no internet access or simulator
state access. Only the robot API supplies task observations. No package installation is needed.
The model and robot-call budgets are finite: prioritize physical progress over scaffolding.

## Shared HITCH motion and perception reference
"""


def reference(hand):
    text = prompts.system_prompt(hand)
    start = text.index("One reply may call several tools;")
    end = text.index("The first message shows", start)
    text = text[:start] + text[end:]
    return GUIDE + "\n" + text + "\n\n" + prompts.template("tools")


def sandbox(workspace: Path, bridge: Path, codex: Path) -> list[str]:
    command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--clearenv"]
    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ld.so.cache"):
        command += ["--ro-bind", path, path]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", str(Path.home())]
    command += ["--ro-bind", str(codex), "/app/codex"]
    for name in ("robot.py", "relay.py", "models.json"):
        command += ["--ro-bind", str(HERE / name), f"/app/{name}"]
    command += ["--ro-bind", str(bridge), "/bridge.sock", "--bind", str(workspace), "/workspace"]
    for name, value in {
        "HOME": str(Path.home()),
        "PATH": "/usr/bin:/bin",
        "SHELL": "/bin/bash",
        "PYTHONPATH": "/app",
        "HITCH_BRIDGE_SOCKET": "/bridge.sock",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        command += ["--setenv", name, value]
    return command + ["--chdir", "/workspace"]


def codex_args(prompt: str, image: str | None) -> list[str]:
    args = [
        "/usr/bin/python3",
        "/app/relay.py",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--ephemeral",
        "-s",
        "danger-full-access",
        "-m",
        "deepseek-flash",
        "-C",
        "/workspace",
        "-o",
        "/workspace/final.txt",
    ]
    settings = {
        "approval_policy": "never",
        "model_provider": "deepseek",
        "model_providers.deepseek.name": "DeepSeek",
        "model_providers.deepseek.wire_api": "responses",
        "model_providers.deepseek.requires_openai_auth": False,
        "model_providers.deepseek.supports_websockets": False,
        "model_providers.deepseek.request_max_retries": 0,
        "model_providers.deepseek.stream_max_retries": 0,
        "model_catalog_json": "/app/models.json",
        "model_reasoning_effort": "high",
        "web_search": "disabled",
        "features.multi_agent": False,
    }
    for key, value in settings.items():
        args += ["-c", f"{key}={json.dumps(value)}"]
    if image:
        args += ["-i", image]
    return args + ["--", prompt]


def flat_metrics(sim: Path) -> dict:
    from wbc_link_server.scenes.success import states

    frames = iter(states(sim))
    first = next(frames)
    last, minimum = first, float(first[2])
    for last in frames:
        minimum = min(minimum, float(last[2]))
    return {
        "root_start_xyz": first[:3].tolist(),
        "root_final_xyz": last[:3].tolist(),
        "root_displacement_m": (last[:3] - first[:3]).tolist(),
        "minimum_pelvis_height_m": minimum,
        "final_pelvis_height_m": float(last[2]),
        "note": "Independent physics measurement, not an upstream task checker.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("flat_walk", "tabletop_pick"))
    parser.add_argument("--root", type=Path, default=Path("data/coding"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--max-calls", type=int, default=60)
    parser.add_argument("--wall-seconds", type=int, default=1200)
    parser.add_argument("--probe", action="store_true", help="check runtime/model without starting a simulator")
    args = parser.parse_args()
    if min(args.max_decisions, args.max_calls, args.wall_seconds) <= 0:
        parser.error("budgets must be positive")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        key = Path(os.environ["DEEPSEEK_API_KEY_FILE"]).read_text().strip()
    if not key:
        parser.error("empty DeepSeek key")
    codex = shutil.which("codex")
    if codex is None:
        parser.error("codex must be installed and on PATH")
    codex = Path(codex).resolve()
    task = prompts.Task.load(args.task)
    os.environ["WBC_HAND"] = task.hand
    root = args.root.resolve()
    record_dir = root / "episodes" / datetime.now().strftime("coding-%Y%m%d-%H%M%S-%f")
    workspace = record_dir / "workspace"
    workspace.mkdir(parents=True)
    record = {
        "task": task.name,
        "seed": args.seed,
        "started": time.time(),
        "probe": args.probe,
        "model": "deepseek-flash",
        "reasoning": "high",
        "max_decisions": args.max_decisions,
        "max_calls": args.max_calls,
        "wall_seconds": args.wall_seconds,
        "codex_version": subprocess.check_output([str(codex), "--version"], text=True).strip(),
    }
    record["hitch_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE.parents[2], text=True
    ).strip()
    record["coding_source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(HERE.iterdir())
        if p.suffix in (".py", ".json")
    }
    session = sim = process = server = broker = server_thread = None
    print(record_dir, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="hitch-coding-") as temp:
            bridge = Path(temp) / "bridge.sock"
            # Bindable socket exists before checking the actual namespace layout.
            server = Gateway(bridge, None, record_dir, key, os.environ.get("HITCH_MODEL_PROXY"), args.max_decisions)
            base = sandbox(workspace, bridge, codex)
            check = subprocess.run(
                base
                + [
                    "/usr/bin/python3",
                    "-c",
                    "import socket,pathlib,numpy,PIL; "
                    "assert not pathlib.Path('/home/zhang/projects').exists(); "
                    "s=socket.socket(); assert s.connect_ex(('127.0.0.1',7890)) != 0; "
                    "print('filesystem and network isolation checked')",
                ],
                capture_output=True,
                text=True,
            )
            if check.returncode:
                raise RuntimeError(check.stderr)
            record["isolation"] = check.stdout.strip()
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            if args.probe:
                prompt = "Write /workspace/probe.py that prints 6*7. Run it using python3 and report the result."
                initial = None
            else:
                target = task.target + (f"&seed={args.seed}" if task.name == "tabletop_pick" else "")
                record["target"] = target
                connected = client.connect(target, root=root)
                session = connected["session"]
                record["session_id"] = connected["session_id"]
                sim = root / "sessions" / connected["session_id"] / "runtime" / "sim"
                broker = server.broker = RobotBroker(session, workspace, record_dir, args.max_calls)
                observation = broker.dispatch({"op": "observe", "args": {}})
                if fault := _robot_down({"calls": observation["calls"]}):
                    raise RuntimeError(fault)
                instruction = _scene_task(task, sim) if prompts.SCENE_TASK in task.text else task.text
                (workspace / "ROBOT.md").write_text(reference(task.hand))
                (workspace / "initial.json").write_text(json.dumps(observation, indent=2))
                images = observation.get("images", [])
                initial = images[0] if images else None
                if isinstance(initial, dict):
                    initial = initial["path"]
                prompt = (
                    f"Operate this simulated G1 through Python programs. Task: {instruction}\n"
                    "Read ROBOT.md and initial.json. The attached image is the initial head view. "
                    "Write and run a task program, inspect feedback and images, revise it if necessary, "
                    "and execute the revised stage in this same session. Finish with robot.finish. "
                    f"Budget: {args.max_decisions} model requests, {args.max_calls} robot calls, "
                    f"{args.wall_seconds} wall seconds. Simulated time pauses while idle."
                )
            (record_dir / "prompt.txt").write_text(prompt)
            with (record_dir / "codex.jsonl").open("w") as out, (record_dir / "codex.stderr").open("w") as err:
                process = subprocess.Popen(
                    base + codex_args(prompt, initial),
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                deadline = time.monotonic() + args.wall_seconds
                while process.poll() is None:
                    if time.monotonic() >= deadline or server.exhausted.is_set():
                        record["stop"] = "wall_budget" if time.monotonic() >= deadline else "model_budget"
                        break
                    time.sleep(0.2)
                else:
                    record["stop"] = "agent_exit"
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                record["returncode"] = process.returncode
                if record["stop"] == "agent_exit" and process.returncode:
                    record["stop"] = "agent_error"
    except BaseException as error:
        record["stop"] = "infrastructure_error"
        record["error"] = repr(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if server_thread:
            server.shutdown()
            server_thread.join()
        if server:
            server.server_close()
            record["decisions"] = server.decisions
        if broker:
            record["robot_calls"] = broker.calls
            record["self_report"] = broker.finished
        if session:
            try:
                client.call(session, "close")
            except Exception as error:
                record["close_error"] = repr(error)
            else:
                if "task_spec" in _scene(sim):
                    record["check"] = _check(sim)
                elif task.name == "flat_walk":
                    record["flat_physics"] = flat_metrics(sim)
        record["ended"] = time.time()
        (record_dir / "episode.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
