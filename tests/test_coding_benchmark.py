from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from wbc.coding import run


class BenchmarkRunnerTests(unittest.TestCase):
    def test_native_terminal_stops_agent_and_keeps_score_private(self):
        for reason, success in (("native_success", True), ("step_budget", False)):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                sim = root / "sessions/test/runtime/sim"
                sim.mkdir(parents=True)
                score = {
                    "terminal": True,
                    "success": success,
                    "genuine": success,
                    "reason": reason,
                    "steps": 100,
                    "stage": {"private_stage": 9},
                }
                (sim / "benchmark_result.json").write_text(json.dumps(score))
                (sim / "native_manifest.json").write_text(
                    json.dumps({"private_reset_truth": [1, 2, 3], "max_steps": 100, "control_freq": 50})
                )
                gateway = Mock(decisions=1)
                process = Mock(pid=12345, returncode=-15)
                process.poll.side_effect = [None, None, -15]
                broker = Mock(calls=1, finished=None)
                broker.dispatch.return_value = {"calls": [], "images": [], "observation_id": "obs-1"}
                stack.enter_context(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}))
                stack.enter_context(
                    patch(
                        "sys.argv",
                        ["coding", "locomanilab", "--benchmark-task", "NativeComposite_scene13", "--root", str(root)],
                    )
                )
                stack.enter_context(patch.object(run.shutil, "which", return_value="/bin/true"))
                stack.enter_context(patch.object(run.subprocess, "check_output", return_value="test-version"))
                stack.enter_context(
                    patch.object(run.subprocess, "run", return_value=Mock(returncode=0, stdout="isolated"))
                )
                stack.enter_context(patch.object(run.subprocess, "Popen", return_value=process))
                kill = stack.enter_context(patch.object(run.os, "killpg"))
                stack.enter_context(patch.object(run, "sandbox", return_value=["sandbox"]))
                stack.enter_context(patch.object(run, "Gateway", return_value=gateway))
                stack.enter_context(patch.object(run, "RobotBroker", return_value=broker))
                stack.enter_context(patch.object(run, "reference", return_value="public robot reference"))
                stack.enter_context(patch.object(run, "_scene_task", return_value="Put the can in the basket."))
                stack.enter_context(patch.object(run, "_robot_down", return_value=None))
                connect = stack.enter_context(
                    patch.object(run.client, "connect", return_value={"session": "test", "session_id": "test"})
                )
                close = stack.enter_context(patch.object(run.client, "call"))
                stack.enter_context(patch("builtins.print"))
                run.main()
                episode = next((root / "episodes").iterdir())
                record = json.loads((episode / "episode.json").read_text())
                self.assertEqual(record["stop"], reason)
                self.assertEqual(record["check"], score)
                self.assertEqual(record["benchmark_task"], "NativeComposite_scene13")
                kill.assert_called_once_with(12345, run.signal.SIGTERM)
                close.assert_called_once_with("test", "close")
                gateway.server_close.assert_called_once()
                self.assertIn("task=NativeComposite_scene13", connect.call_args.args[0])
                self.assertIn("paused=1", connect.call_args.args[0])
                agent_text = (episode / "prompt.txt").read_text() + "".join(
                    path.read_text() for path in (episode / "workspace").iterdir()
                )
                self.assertIn("Put the can in the basket.", agent_text)
                self.assertIn("100 control steps (2.00 simulated seconds)", agent_text)
                self.assertNotIn("private_stage", agent_text)
                self.assertNotIn("private_reset_truth", agent_text)


if __name__ == "__main__":
    unittest.main()
