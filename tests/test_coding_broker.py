from __future__ import annotations

import hashlib
import json
import os
import socketserver
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

from wbc.agent.conversation import Image
from wbc.agent.tools import Result
from wbc.coding import robot
from wbc.coding.broker import RobotBroker


class FakeTools:
    def __init__(self, session, images):
        self.session = session
        self.images = images
        self.seen = None
        self.walked = False
        self.fingers = True
        self.operations = []
        self.observations = 0

    def _observe(self):
        self.observations += 1
        self.seen = f"obs-{self.observations}"
        label = f"observations/{self.seen}/rgb.png"
        path = self.images / f"{self.seen}_rgb.png"
        path.write_bytes(b"test image")
        observation = {"observation_id": self.seen, "body": {"pelvis_height": 0.75}, "images": [label]}
        return observation, Image(path, label)

    def observe(self):
        self.operations.append("observe")
        observation, image = self._observe()
        return Result("Standing.", (image,), {"calls": [{"op": "observe", "result": observation}]})

    def run(self, call):
        self.operations.append(call.name)
        args = json.loads(call.arguments)
        if call.name == "finish":
            return Result("Finished.", data={"calls": []}, finished=args)
        if call.name == "perceive":
            points = [{"point": args["points"][0], "position": [0.3, 0.0, 0.9]}]
            return Result("Measured.", data={"calls": [{"op": "perceive", "result": {"points": points}}]})
        self.walked = "walk" in args
        observation, image = self._observe()
        return Result(
            "Executed.",
            (image,),
            {"calls": [{"op": "execute", "result": {"state": "done"}}, {"op": "observe", "result": observation}]},
            moved=True,
        )


class BrokerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.records = self.root / "records"
        self.patch = patch("wbc.coding.broker.Tools", FakeTools)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.broker = RobotBroker("http://127.0.0.1:1234", self.workspace, self.records)

    def records_of(self, event):
        records = [json.loads(line) for line in (self.records / "robot.jsonl").read_text().splitlines()]
        return [record for record in records if record["event"] == event]

    def test_state_survives_calls_and_old_observation_is_rejected(self):
        first = self.broker.dispatch({"op": "observe"})
        result = self.broker.dispatch(
            {
                "op": "execute",
                "args": {"request": {"walk": {"target": [0.3, 0]}}, "observation_id": first["observation_id"]},
            }
        )
        self.assertEqual(result["observation_id"], "obs-2")
        self.assertTrue(self.broker.tools.walked)
        self.assertEqual(result["observation"]["body"]["pelvis_height"], 0.75)
        self.assertEqual(result["images"], ["/workspace/images/obs-2_rgb.png"])
        self.assertEqual(result["observation"]["images"], result["images"])
        self.assertEqual((self.workspace / "images/obs-2_rgb.png").read_bytes(), b"test image")
        for op, args in (("perceive", {"points": [[500, 500]]}), ("execute", {"request": {"wait_s": 1}})):
            with self.subTest(op=op), self.assertRaisesRegex(ValueError, "stale"):
                self.broker.dispatch({"op": op, "args": {**args, "observation_id": "obs-1"}})
        self.assertEqual(self.broker.tools.operations, ["observe", "execute"])
        measured = self.broker.dispatch({"op": "perceive", "args": {"points": [[500, 500]], "observation_id": "obs-2"}})
        self.assertEqual(measured["points"][0]["position"], [0.3, 0.0, 0.9])
        self.assertEqual(measured["observation_id"], "obs-2")

    def test_budget_stops_robot_calls_but_allows_finish(self):
        self.broker.max_calls = 1
        self.broker.dispatch({"op": "observe"})
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.broker.dispatch({"op": "observe"})
        self.assertEqual(self.broker.tools.operations, ["observe"])
        done = self.broker.dispatch({"op": "finish", "args": {"success": False, "summary": "Budget reached."}})
        self.assertEqual(done["finished"], {"success": False, "summary": "Budget reached."})
        self.assertEqual(self.broker.calls, 1)
        with self.assertRaisesRegex(ValueError, "already finished"):
            self.broker.dispatch({"op": "observe"})

    def test_snapshots_capture_revisions_and_skip_external_symlinks(self):
        program = self.workspace / "attempt.py"
        program.write_text("print('first revision')\n")
        external = self.root / "outside"
        external.mkdir()
        (external / "private.py").write_text("PRIVATE_CONTENT\n")
        (self.workspace / "linked.py").symlink_to(external / "private.py")
        (self.workspace / "linked_directory").symlink_to(external, target_is_directory=True)
        self.broker.dispatch({"op": "observe"})
        program.write_text("print('second revision')\n")
        self.broker.dispatch({"op": "observe"})
        requests = self.records_of("request")
        self.assertEqual([p["path"] for p in requests[0]["programs"]], ["attempt.py"])
        old, new = (request["programs"][0] for request in requests)
        self.assertNotEqual(old["sha256"], new["sha256"])
        self.assertEqual((self.records / old["snapshot"]).read_text(), "print('first revision')\n")
        self.assertEqual(new["sha256"], hashlib.sha256(program.read_bytes()).hexdigest())

    def test_action_record_survives_a_later_program_error(self):
        first = self.broker.dispatch({"op": "observe"})
        with self.assertRaisesRegex(RuntimeError, "program failed"):
            self.broker.dispatch(
                {"op": "execute", "args": {"request": {"wait_s": 1}, "observation_id": first["observation_id"]}}
            )
            raise RuntimeError("program failed after the robot moved")
        results = self.records_of("result")
        self.assertEqual(len(results), 2)
        self.assertTrue(results[-1]["result"]["moved"])
        self.assertEqual(self.broker.dispatch({"op": "observe"})["observation_id"], "obs-3")

    def test_image_directory_cannot_redirect_writes_outside_workspace(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.workspace / "images").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.broker.dispatch({"op": "observe"})
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(len(self.records_of("result")), 1)

    def test_only_robot_operations_are_public(self):
        for op in ("status", "show", "files", "close"):
            with self.subTest(op=op), self.assertRaisesRegex(ValueError, "operation must be"):
                self.broker.dispatch({"op": op})
        self.assertEqual(self.broker.tools.operations, [])

    def test_session_url_is_removed_from_results_and_exceptions(self):
        message = self.broker.tools.session + "/call: connection refused"
        failed = Result(message, data={"calls": [{"op": "observe", "error": {"message": message}}]})
        with patch.object(self.broker.tools, "observe", return_value=failed):
            result = self.broker.dispatch({"op": "observe"})
        self.assertNotIn(self.broker.tools.session, json.dumps(result))
        self.assertIn("<robot-session>/call", result["text"])
        self.assertIn(self.broker.tools.session, json.dumps(self.records_of("result")))
        with (
            patch.object(self.broker.tools, "observe", side_effect=RuntimeError(message)),
            self.assertRaisesRegex(ValueError, "<robot-session>/call") as caught,
        ):
            self.broker.dispatch({"op": "observe"})
        self.assertNotIn(self.broker.tools.session, str(caught.exception))
        self.assertIn(self.broker.tools.session, self.records_of("error")[-1]["error"])

    def test_standard_library_sdk_uses_the_unix_robot_gateway(self):
        broker = self.broker

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/robot":
                    self.send_error(404)
                    return
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    result, status = broker.dispatch(payload), 200
                except ValueError as error:
                    result, status = {"error": str(error)}, 400
                body = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        path = self.root / "bridge.sock"
        with socketserver.UnixStreamServer(str(path), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch.dict(os.environ, {"HITCH_BRIDGE_SOCKET": str(path)}):
                    observation = robot.observe()
                    moved = robot.execute({"wait_s": 1}, observation_id=observation["observation_id"])
                    measured = robot.perceive([[500, 500]], observation_id=moved["observation_id"])
                    self.assertEqual(measured["points"][0]["position"], [0.3, 0.0, 0.9])
                    with self.assertRaisesRegex(RuntimeError, "stale"):
                        robot.execute({"wait_s": 1}, observation_id=observation["observation_id"])
                    self.assertTrue(robot.finish(True, "Verified by observations.")["finished"]["success"])
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
