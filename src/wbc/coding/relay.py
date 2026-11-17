"""Run inside the network namespace; adapt Codex TCP HTTP to the private Unix bridge."""

import http.client
import os
import socket
import subprocess
import sys
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/bridge.sock")


class Relay(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/responses":
            self.send_error(404)
            return
        with closing(UnixConnection("localhost", timeout=900)) as connection:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            connection.request("POST", self.path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type"))
            self.end_headers()
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    command = [
        "/app/codex",
        "exec",
        "-c",
        f'model_providers.deepseek.base_url="http://127.0.0.1:{server.server_port}"',
        *sys.argv[1:],
    ]
    result = subprocess.run(command, env=os.environ)
    server.shutdown()
    sys.exit(result.returncode)
