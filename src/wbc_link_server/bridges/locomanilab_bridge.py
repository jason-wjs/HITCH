"""Native LocoManiLab physics behind the unchanged HITCH controller UDP protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from wbc.core import logs
from wbc.core.layout import JOINTS
from wbc_link_server.benchmarks.native import NativeTask
from wbc_link_server.bridges.sim_bridge import MujocoBridge


class Bridge(MujocoBridge):
    def build_scene(self) -> None:
        self.task = NativeTask(self.args.benchmark_task, self.directory, seed=self.args.seed)
        self.model, self.data, self.initial = self.task.model, self.task.data, self.task.initial

    def configure_robot(self) -> None:
        self.qadr, self.vadr = self.task.qadr, self.task.vadr
        joints = [self.model.joint("robot0_" + name).id for name in JOINTS]
        self.bounds = self.model.jnt_range[joints]

    def measured(self):
        return self.task.measured()

    def set_fingers(self, side, *, closed, joints=None) -> None:
        self.task.set_fingers(side, closed, joints=joints)

    def advance(self, target, kp, kd) -> None:
        if self.started:
            self.task.step(target, kp, kd)
        else:
            self.task.hold(target)

    def bridge_info(self) -> dict:
        return {
            "source_id": self.source,
            "scene": "locomanilab",
            "benchmark_task": self.args.benchmark_task,
            "seed": self.args.seed,
            "state_port": self.args.state_port,
            "command_port": self.args.command_port,
            "physics_dt": self.model.opt.timestep,
        }

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.task.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--channel", type=Path, required=True)
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--command-port", type=int, required=True)
    parser.add_argument("--benchmark-task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paused", action="store_true")
    args = parser.parse_args()
    logs.setup("bridge", logs.session_of(args.directory))
    Bridge(args).run()


if __name__ == "__main__":
    main()
