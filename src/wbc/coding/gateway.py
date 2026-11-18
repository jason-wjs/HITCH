"""Private Unix bridge: robot tools and one fixed model endpoint."""

from __future__ import annotations

import json
import socketserver
import threading
import urllib.error
import urllib.request
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from wbc.coding.providers import PROVIDERS, Provider


class Gateway(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False

    def __init__(
        self,
        socket_path: Path,
        broker,
        record: Path,
        key: str,
        proxy: str | None,
        max_decisions: int,
        provider: Provider | None = PROVIDERS["deepseek"],
    ):
        self.provider = provider
        self.broker = broker
        self.record = record
        self.key = key
        self.proxy = proxy
        self.max_decisions = max_decisions
        self.decisions = 0
        self.lock = threading.Lock()
        self.exhausted = threading.Event()
        super().__init__(str(socket_path), Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def send_json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= 32 * 1024 * 1024:
            self.send_json(400, {"error": "invalid body size"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if self.path == "/robot" and self.server.broker is not None:
                self.send_json(200, self.server.broker.dispatch(payload))
            elif self.path == "/responses" and self.server.provider is not None:
                self.model(payload)
            else:
                self.send_json(404, {"error": "unknown endpoint"})
        except (ValueError, TypeError, KeyError) as error:
            self.send_json(400, {"error": str(error)})

    def model(self, payload):
        server = self.server
        if payload.get("model") != server.provider.model:
            raise ValueError(f"this experiment uses {server.provider.model}")
        with server.lock:
            if server.decisions >= server.max_decisions:
                server.exhausted.set()
                self.send_json(429, {"error": "model decision budget exhausted"})
                return
            server.decisions += 1
            number = server.decisions
        payload["max_output_tokens"] = 32768
        payload["reasoning"] = {"effort": "high"}
        stem = server.record / f"model-{number:03d}"
        stem.with_suffix(".request.json").write_text(json.dumps(payload))
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": server.proxy} if server.proxy else {})
        )
        request = urllib.request.Request(
            server.provider.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {server.key}", "Content-Type": "application/json"},
        )
        try:
            try:
                response = opener.open(request, timeout=180)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.end_headers()
                with stem.with_suffix(".response").open("wb") as log:
                    while chunk := response.read1(65536):
                        log.write(chunk)
                        log.flush()
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except (OSError, HTTPException) as error:
            stem.with_suffix(".error.txt").write_text(str(error))
            self.close_connection = True
