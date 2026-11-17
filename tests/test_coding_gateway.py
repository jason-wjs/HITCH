from __future__ import annotations

import http.client
import io
import json
import socket
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from wbc.coding.gateway import Gateway


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=3)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Upstream(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.socket = self.root / "bridge.sock"
        self.broker = Mock()
        self.gateway = Gateway(self.socket, self.broker, self.root, "test-key", None, 2)
        self.thread = threading.Thread(target=self.gateway.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.stop_gateway)
        self.opener = Mock()
        self.upstream = patch("wbc.coding.gateway.urllib.request.build_opener", return_value=self.opener)
        self.upstream.start()
        self.addCleanup(self.upstream.stop)

    def stop_gateway(self):
        self.gateway.shutdown()
        self.thread.join()
        self.gateway.server_close()

    def post(self, endpoint, payload):
        connection = UnixConnection(self.socket)
        try:
            connection.request("POST", endpoint, json.dumps(payload), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_fixed_model_endpoint_and_parameters_are_recorded_without_key(self):
        content = b'data: {"type":"response.completed"}\n\n'
        self.opener.open.return_value = Upstream(content)
        status, body = self.post(
            "/responses",
            {"model": "deepseek-flash", "max_output_tokens": 1, "reasoning": {"effort": "low"}, "input": "test"},
        )
        self.assertEqual((status, body), (200, content))
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["max_output_tokens"], 32768)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(json.loads((self.root / "model-001.request.json").read_text()), payload)
        self.assertEqual((self.root / "model-001.response").read_bytes(), content)
        self.assertNotIn("test-key", (self.root / "model-001.request.json").read_text())

    def test_other_model_is_rejected_before_spending_budget(self):
        status, body = self.post("/responses", {"model": "another-model"})
        self.assertEqual(status, 400)
        self.assertIn("deepseek-flash", json.loads(body)["error"])
        self.assertEqual(self.gateway.decisions, 0)
        self.opener.open.assert_not_called()

    def test_model_budget_limits_upstream_requests(self):
        self.opener.open.side_effect = [Upstream(b"first"), Upstream(b"second")]
        self.assertEqual(self.post("/responses", {"model": "deepseek-flash"}), (200, b"first"))
        self.assertEqual(self.post("/responses", {"model": "deepseek-flash"}), (200, b"second"))
        status, body = self.post("/responses", {"model": "deepseek-flash"})
        self.assertEqual(status, 429)
        self.assertIn("budget exhausted", json.loads(body)["error"])
        self.assertEqual(self.gateway.decisions, 2)
        self.assertEqual(self.opener.open.call_count, 2)
        self.assertTrue(self.gateway.exhausted.is_set())
        self.assertFalse((self.root / "model-003.request.json").exists())

    def test_unknown_endpoint_cannot_reach_model_or_robot(self):
        for endpoint in ("/files/private.json", "/status", "/responses/extra", "https://other.example/responses"):
            with self.subTest(endpoint=endpoint):
                status, body = self.post(endpoint, {"model": "deepseek-flash"})
                self.assertEqual(status, 404)
                self.assertEqual(json.loads(body), {"error": "unknown endpoint"})
        self.assertEqual(self.gateway.decisions, 0)
        self.opener.open.assert_not_called()
        self.broker.dispatch.assert_not_called()

    def test_robot_value_error_returns_json_400_without_using_model(self):
        self.broker.dispatch.side_effect = ValueError("Observation is stale")
        payload = {"op": "execute", "args": {"observation_id": "old", "request": {"wait_s": 1}}}
        status, body = self.post("/robot", payload)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "Observation is stale"})
        self.broker.dispatch.assert_called_once_with(payload)
        self.opener.open.assert_not_called()
        self.assertEqual(self.gateway.decisions, 0)

    def test_upstream_connection_failure_retains_request_and_error(self):
        self.opener.open.side_effect = urllib.error.URLError("upstream unavailable")
        with self.assertRaises(http.client.RemoteDisconnected):
            self.post("/responses", {"model": "deepseek-flash"})
        self.assertTrue((self.root / "model-001.request.json").exists())
        self.assertIn("upstream unavailable", (self.root / "model-001.error.txt").read_text())
        self.assertEqual(self.gateway.decisions, 1)

    def test_upstream_stream_failure_preserves_partial_response_and_error(self):
        partial = b'data: {"type":"response.output_text.delta","delta":"first"}\n\n'
        upstream = Upstream(partial)
        upstream.read1 = Mock(side_effect=[partial, OSError("upstream stream interrupted")])
        self.opener.open.return_value = upstream
        status, body = self.post("/responses", {"model": "deepseek-flash"})
        self.assertEqual((status, body), (200, partial))
        self.assertEqual((self.root / "model-001.response").read_bytes(), partial)
        self.assertIn("upstream stream interrupted", (self.root / "model-001.error.txt").read_text())
        self.assertTrue((self.root / "model-001.request.json").exists())
        self.assertTrue(upstream.closed)


if __name__ == "__main__":
    unittest.main()
