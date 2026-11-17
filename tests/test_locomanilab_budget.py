"""Terminal protocol checks without assets, a controller model, or an API."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wbc_link_server.benchmarks.native import NativeTask


class CountingEnv:
    def __init__(self, data, success_at=None):
        self.data = data
        self.calls = 0
        self.success_at = success_at

    def step(self, action):
        self.calls += 1
        self.data.time += 0.02

    def _check_success(self):
        return self.success_at is not None and self.calls >= self.success_at


class TerminalProtocolTests(unittest.TestCase):
    def task(self, directory, success_at=None):
        # A counted native-step boundary makes unintended post-terminal physics
        # observable without requiring a long rollout or changing real task specs.
        task = NativeTask.__new__(NativeTask)
        task.directory = Path(directory)
        task.task_id = "protocol-test"
        task.spec = {"max_steps": 2, "genuine_min_steps": 2}
        task.steps = 0
        task.started_at = 0.0
        task._terminal = None
        task.controller = SimpleNamespace(command=None)
        task.data = SimpleNamespace(time=0.0, qpos=np.zeros(1))
        task.env = CountingEnv(task.data, success_at)
        return task

    def test_exhausted_budget_keeps_replying_without_advancing_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory)
            command = (np.zeros(29),) * 3
            self.assertFalse(task.step(*command)["terminal"])
            final = task.step(*command)
            self.assertEqual(final["reason"], "step_budget")
            for _ in range(10):
                self.assertEqual(task.step(*command), final)
            self.assertEqual(task.env.calls, 2)
            self.assertAlmostEqual(task.data.time, .04)
            self.assertTrue((Path(directory) / "benchmark_result.json").exists())

    def test_early_raw_success_is_terminal_but_not_genuine(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self.task(directory, success_at=1)
            command = (np.zeros(29),) * 3
            final = task.step(*command)
            self.assertTrue(final["success"])
            self.assertTrue(final["terminal"])
            self.assertFalse(final["genuine"])
            self.assertEqual(task.step(*command), final)
            self.assertEqual(task.env.calls, 1)


if __name__ == "__main__":
    unittest.main()
