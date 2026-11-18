"""Route the native Codex client to a Bubblewrap execution worker."""

import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

CHATGPT_MODEL = "gpt-6.2-sol"
CHATGPT_REASONING = "medium"


@contextmanager
def subscription_env(codex: Path, temporary: Path, worker: list[str], record: Path):
    """Bind the worker's loopback exec-server to a private Unix socket."""
    source_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    auth = source_home / "auth.json"
    if not auth.is_file():
        raise RuntimeError("ChatGPT subscription requires Codex file-based login on the host")
    home = temporary / "codex-home"
    home.mkdir(mode=0o700)
    shutil.copyfile(auth, home / "auth.json")
    (home / "auth.json").chmod(0o600)
    env = {key: os.environ[key] for key in ("HOME", "PATH", "LANG") if key in os.environ}
    env["CODEX_HOME"] = str(home)
    if proxy := os.environ.get("HITCH_MODEL_PROXY"):
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
    result = subprocess.run([str(codex), "login", "status"], env=env, text=True, capture_output=True, timeout=15)
    if result.returncode or "Logged in using ChatGPT" not in result.stdout + result.stderr:
        raise RuntimeError("Codex requires a valid host ChatGPT login")
    transport = temporary / "executor"
    transport.mkdir()
    command = [
        *worker,
        "--bind",
        str(transport),
        "/executor",
        "/bin/sh",
        "-c",
        "/usr/bin/socat UNIX-LISTEN:/executor/worker.sock,fork TCP:127.0.0.1:18765 & "
        "exec /app/codex exec-server --listen ws://127.0.0.1:18765",
    ]
    processes = []
    try:
        with (record / "exec-server.log").open("w") as log:
            isolated = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            processes.append(isolated)
            wait_ready(isolated, lambda: (transport / "worker.sock").is_socket(), "isolated exec-server socket")
        relay_log = record / "exec-relay.log"
        with relay_log.open("w") as log:
            relay = subprocess.Popen(
                [
                    "socat",
                    "-d",
                    "-d",
                    "TCP-LISTEN:0,bind=127.0.0.1,reuseaddr,fork",
                    f"UNIX-CONNECT:{transport / 'worker.sock'}",
                ],
                stderr=log,
                start_new_session=True,
            )
            processes.append(relay)
            match = wait_ready(
                relay,
                lambda: re.search(r"listening on AF=2 127\.0\.0\.1:(\d+)", relay_log.read_text()),
                "exec-server loopback relay",
            )
        env["CODEX_EXEC_SERVER_URL"] = f"ws://127.0.0.1:{match[1]}"
        env["NO_PROXY"] = "localhost,127.0.0.1"
        yield env
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)


def wait_ready(process, check, description):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process.poll() is None:
        if result := check():
            return result
        time.sleep(0.05)
    raise RuntimeError(f"could not start {description}")
