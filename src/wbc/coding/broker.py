"""Trusted robot boundary for programs in the coding workspace.

One broker owns one HITCH Tools instance throughout an episode. Generated code
receives observations and explicit robot operations, never the session URL or
its file service. Program snapshots and original tool results remain outside
the coding workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
from dataclasses import asdict
from pathlib import Path

from wbc.agent.conversation import ToolCall
from wbc.agent.tools import Tools


class RobotBroker:
    def __init__(self, session: str, workspace: Path, record_dir: Path, max_calls: int = 60):
        self.workspace = Path(workspace).resolve()
        self.record_dir = Path(record_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.record_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.record_dir / "programs").mkdir(exist_ok=True)
        self.tools = Tools(session, self.record_dir / "images")
        self.max_calls = max_calls
        self.calls = 0
        self.finished: dict | None = None
        self.last_result: dict | None = None
        self._observation: dict | None = None
        self._image_paths: dict[str, str] = {}
        self._sequence = 0
        self._lock = threading.Lock()

    def dispatch(self, payload: dict) -> dict:
        """Run one explicit operation and immediately record its source and result.

        observe/perceive/execute consume the robot-call budget. finish remains
        available after the budget is exhausted, and makes the broker terminal.
        """
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            programs = self._snapshot(sequence)
            self._record({"event": "request", "sequence": sequence, "payload": payload, "programs": programs})
            started = time.monotonic()
            try:
                op, args = self._validate(payload)
                if op == "observe":
                    self.calls += 1
                    result = self.tools.observe()
                else:
                    if op != "finish":
                        self.calls += 1
                    result = self.tools.run(ToolCall(f"robot-{sequence}", op, json.dumps(args, allow_nan=False)))
                self._record(
                    {
                        "event": "result",
                        "sequence": sequence,
                        "seconds": time.monotonic() - started,
                        "result": asdict(result),
                    }
                )
                for call in (result.data or {}).get("calls", ()):
                    if call["op"] == "observe" and call.get("result") is not None:
                        self._observation = call["result"]
                images = self._publish_images(result.images)
                response = {
                    "call_id": sequence,
                    "text": result.text,
                    "images": images,
                    "observation_id": self.tools.seen,
                    "observation": self._observation,
                    "moved": result.moved,
                    "finished": result.finished,
                    "calls": (result.data or {}).get("calls", []),
                }
                if (result.data or {}).get("refused"):
                    response["refused"] = result.data["refused"]
                if op == "perceive":
                    for call in (result.data or {}).get("calls", ()):
                        if call["op"] == "perceive" and call.get("result") is not None:
                            response["points"] = call["result"]["points"]
                self.finished = result.finished or self.finished
                self.last_result = self._public(response)
                return self.last_result
            except Exception as error:
                self._record(
                    {"event": "error", "sequence": sequence, "type": type(error).__name__, "error": str(error)}
                )
                raise ValueError(self._public(str(error))) from None

    def _validate(self, payload: dict) -> tuple[str, dict]:
        if not isinstance(payload, dict) or set(payload) - {"op", "args"}:
            raise ValueError("Give a robot operation as {op, args}.")
        op, args = payload.get("op"), payload.get("args", {})
        fields = {
            "observe": set(),
            "perceive": {"points", "observation_id"},
            "execute": {"request", "observation_id"},
            "finish": {"success", "summary"},
        }
        if not isinstance(op, str) or op not in fields:
            raise ValueError("Robot operation must be observe, perceive, execute, or finish.")
        if not isinstance(args, dict) or set(args) != fields[op]:
            raise ValueError(f"{op} requires exactly {sorted(fields[op])}.")
        if self.finished is not None:
            raise ValueError("This episode has already finished.")
        if op != "finish" and self.calls >= self.max_calls:
            raise ValueError(f"Robot-call budget exhausted ({self.max_calls}); finish the episode.")
        if op in ("perceive", "execute"):
            observation_id = args["observation_id"]
            if not isinstance(observation_id, str) or observation_id != self.tools.seen:
                raise ValueError(f"Observation is stale; current observation_id is {self.tools.seen!r}. Observe again.")
        if op == "execute":
            if not isinstance(args["request"], dict):
                raise ValueError("execute request must be a dictionary.")
            return op, args["request"]
        if op == "perceive":
            return op, {"points": args["points"]}
        if op == "finish" and (not isinstance(args["success"], bool) or not isinstance(args["summary"], str)):
            raise ValueError("finish requires a boolean success and a string summary.")
        return op, args

    def _snapshot(self, sequence: int) -> list[dict]:
        """Copy regular Python files through directory descriptors, never symlinks."""
        destination = self.record_dir / "programs" / f"{sequence:06d}"
        destination.mkdir()
        files = []
        for directory, _, names, directory_fd in os.fwalk(self.workspace, follow_symlinks=False):
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                with os.fdopen(descriptor, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        continue
                    content = source.read()
                relative = Path(directory).relative_to(self.workspace) / name
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                files.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "snapshot": target.relative_to(self.record_dir).as_posix(),
                    }
                )
        return files

    def _publish_images(self, images) -> list[str]:
        directory = self.workspace / "images"
        directory.mkdir(exist_ok=True)
        if directory.is_symlink():
            raise ValueError("workspace/images must be a real directory, not a symlink.")
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        paths = []
        try:
            for image in images:
                name = image.path.name
                target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644, dir_fd=descriptor)
                with os.fdopen(target, "wb") as stream:
                    stream.write(image.path.read_bytes())
                public = f"/workspace/images/{name}"
                self._image_paths[image.label] = public
                paths.append(public)
        finally:
            os.close(descriptor)
        return paths

    def _public(self, value):
        if isinstance(value, str):
            return self._image_paths.get(value, value.replace(self.tools.session, "<robot-session>"))
        if isinstance(value, list):
            return [self._public(item) for item in value]
        if isinstance(value, dict):
            return {key: self._public(item) for key, item in value.items()}
        return value

    def _record(self, value: dict) -> None:
        with (self.record_dir / "robot.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), **value}, default=str, allow_nan=False) + "\n")
