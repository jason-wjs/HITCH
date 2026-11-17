"""Robot API available to programs inside the coding workspace."""

from __future__ import annotations

import http.client
import json
import os
import socket


class _Connection(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("localhost", timeout=900)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def _call(op: str, **args) -> dict:
    connection = _Connection(os.environ["HITCH_BRIDGE_SOCKET"])
    try:
        connection.request(
            "POST",
            "/robot",
            body=json.dumps({"op": op, "args": args}, allow_nan=False),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(result.get("error", f"Robot bridge returned HTTP {response.status}"))
        return result
    finally:
        connection.close()


def observe() -> dict:
    """Observe the robot and return its current observation_id and image paths."""
    return _call("observe")


def perceive(points: list, *, observation_id: str) -> dict:
    """Measure normalized image points (0–1000) in the named observation."""
    return _call("perceive", points=points, observation_id=observation_id)


def execute(request: dict, *, observation_id: str) -> dict:
    """Plan and execute a HITCH body request, then return fresh feedback and images."""
    return _call("execute", request=request, observation_id=observation_id)


def finish(success: bool, summary: str) -> dict:
    """Record the agent's outcome claim; the trusted runner scores the task separately."""
    return _call("finish", success=success, summary=summary)
