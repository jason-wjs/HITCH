from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from wbc.link.local import start_simulator
from wbc_link_server.launch.sim import start_control, start_mujoco, start_policy


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.python = self.root / "native-python"
        self.python.write_text("")
        self.python.chmod(0o700)
        self.stack = Mock()
        self.stack.wait_for.return_value = {"vision": "http://camera", "control": "http://control"}
        self.args = SimpleNamespace(
            scene="flat",
            seed=0,
            heading_deg=0,
            paused=True,
            no_vision=False,
            viewer=False,
            checkpoint=None,
            benchmark_task="ExampleComposite_scene0",
        )

    def test_existing_scenes_keep_original_module_and_interpreter(self):
        for scene in ("flat", "tabletop_pick"):
            with self.subTest(scene=scene), patch.dict("os.environ", {"WBC_BENCHMARK_PYTHON": str(self.python)}):
                self.stack.reset_mock()
                self.args.scene = scene
                start_mujoco(self.stack, self.args, self.root, 20001, 20002)
                commands = {call.args[0]: call.args[1] for call in self.stack.spawn.call_args_list}
                self.assertEqual(commands["bridge"][:3], [sys.executable, "-m", "wbc_link_server.bridges.sim_bridge"])
                self.assertEqual(commands["bridge"][-6:], ["--scene", scene, "--seed", "0", "--heading-deg", "0"])
                self.assertEqual(commands["vision"][0], sys.executable)

    def test_native_python_only_runs_physics_and_camera(self):
        self.args.scene = "locomanilab"
        with patch.dict("os.environ", {"WBC_BENCHMARK_PYTHON": str(self.python)}):
            start_mujoco(self.stack, self.args, self.root, 20001, 20002)
            start_policy(self.stack, self.args, self.root, 20001, 20002)
            start_control(self.stack, self.root, 0)
        commands = {call.args[0]: call.args[1] for call in self.stack.spawn.call_args_list}
        self.assertEqual(commands["bridge"][:3], [str(self.python), "-m", "wbc_link_server.bridges.locomanilab_bridge"])
        self.assertEqual(commands["vision"][0], str(self.python))
        self.assertEqual(commands["policy"][0], sys.executable)
        self.assertEqual(commands["control"][0], sys.executable)
        self.assertIn("--benchmark-task", commands["bridge"])
        self.assertIn("--paused", commands["bridge"])

    def test_missing_native_python_fails_before_starting_any_process(self):
        self.args.scene = "locomanilab"
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "WBC_BENCHMARK_PYTHON"):
            start_mujoco(self.stack, self.args, self.root, 20001, 20002)
        self.stack.spawn.assert_not_called()

    def test_session_target_forwards_exact_task_id(self):
        directory = self.root / "runtime"
        process = Mock()
        process.poll.return_value = None

        def ready(path, *args, **kwargs):
            path.write_text(json.dumps({"control": "http://control", "vision": "http://camera"}))
            return True

        with (
            patch("wbc.link.local.subprocess.Popen", return_value=process) as launch,
            patch("wbc.link.local.wait_for_file", side_effect=ready),
        ):
            start_simulator(directory, "sim://locomanilab?task=ExampleComposite_scene0&seed=2&paused=1", 30)
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--benchmark-task") + 1], "ExampleComposite_scene0")
        self.assertEqual(command[command.index("--seed") + 1], "2")
        self.assertEqual(command[0], sys.executable)


if __name__ == "__main__":
    unittest.main()
