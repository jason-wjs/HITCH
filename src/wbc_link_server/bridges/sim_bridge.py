"""MuJoCo stand-in for the robot, behind the controller's UDP protocol (``protocol.ProtocolBridge``): the physics step
is torque-clipped joint PD, as on the robot. ``MujocoBridge`` is the MuJoCo robot; ``Bridge`` builds it on a scene of
``wbc_link_server.scenes``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbc.core import logs
from wbc.core.jsonhttp import write_json_atomic
from wbc.core.layout import FPS, JOINTS
from wbc.paths import INITIAL_REFERENCE
from wbc.robot.hands import current_hand
from wbc.robot.model import body_joint_addresses, hold_fingers, robot_extent, set_fingers, set_robot_pose
from wbc_link_server import scenes
from wbc_link_server.bridges.fingers import FingerCommands
from wbc_link_server.bridges.protocol import CONTROL_DT, ProtocolBridge
from wbc_link_server.sensors.frame_ring import FrameRing

# WBC_SIM_UNPACED=1: step as soon as the controller's command for the latest state arrives, instead of
# holding each step to CONTROL_DT of wall time. The exchange is lockstep (a command answers one state stamp), so the
# physics is the same; only the wall time shrinks.
UNPACED = os.environ.get("WBC_SIM_UNPACED") == "1"
FINGER_POLL_FRAMES = 2  # control frames between reads of the link's hand commands (40 ms)


def initialize_scene(model, data, spec, initial):
    """Set the robot pose and settle spawned scenery before the first observation.

    The robot remains held during this settle, just as during bridge startup. No object pose is edited after
    settling.
    """
    _, robot_nv = robot_extent(model)
    set_robot_pose(model, data, initial)
    hold_fingers(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(round(spec.get("settle_seconds", 0.0) / model.opt.timestep)):
        mujoco.mj_step(model, data)
        set_robot_pose(model, data, initial)
        data.qvel[:robot_nv] = 0
    data.time = 0.0
    mujoco.mj_forward(model, data)


def save_chunk(path: Path, rows: list) -> None:
    np.savez_compressed(
        path,
        qpos=[r[0] for r in rows],
        qvel=[r[1] for r in rows],
        time=[r[2] for r in rows],
        target=[r[3] for r in rows],
    )


class MujocoBridge(ProtocolBridge):
    """A robot on MuJoCo physics: torque-clipped joint PD at the model's physics step, the fingers the link commands,
    the physics log (``physics_*.npz``) and the camera process's frame ring. A subclass builds the model, the start
    state and the scene's records (``build_scene``) and says what ``bridge.json`` holds (``bridge_info``).

    ``finger_commands`` reads the link's hand commands (``FingerCommands`` on the controller runtime); ``changes()``
    answers the sides that changed at a poll.
    """

    paced = not UNPACED

    def __init__(self, args):
        super().__init__(args)
        self.build_scene()
        self.configure_robot()
        self.finger_commands = FingerCommands(args.channel.parent)
        self.finger_stages = {}  # side -> [(frame, joints)] still to open
        self.substeps = round(CONTROL_DT / self.model.opt.timestep)
        self.ring = FrameRing(self.directory / "camera_frames.npy", nq=self.model.nq, nv=self.model.nv)
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-log")
        self.pending_write = None
        self.write_error = None  # the first chunk the log thread failed to write (raised by finish)
        self.dropped_log_frames = 0
        # last: the launcher polls for it, and everything above is in place by then
        write_json_atomic(self.directory / "bridge.json", self.bridge_info())

    def configure_robot(self) -> None:
        actuated = self.model.actuator_trnid[: len(JOINTS), 0]  # fingers follow the 29 body motors
        if tuple(self.model.joint(int(j)).name for j in actuated) != JOINTS:
            raise ValueError("model actuators are not in the canonical joint order")
        self.qadr, self.vadr = body_joint_addresses(self.model)
        _, self.robot_nv = robot_extent(self.model)
        self.bounds = self.model.jnt_range[actuated]
        hold_fingers(self.model, self.data)

    def set_fingers(self, side, *, closed, joints=None) -> None:
        set_fingers(self.model, self.data, side, closed=closed, joints=joints)

    def advance(self, target, kp, kd) -> None:
        m, d = self.model, self.data
        if self.started:
            for _ in range(self.substeps):
                torque = kp * (target - d.qpos[self.qadr]) - kd * d.qvel[self.vadr]
                d.ctrl[: len(JOINTS)] = np.clip(
                    torque, m.actuator_ctrlrange[: len(JOINTS), 0], m.actuator_ctrlrange[: len(JOINTS), 1]
                )
                mujoco.mj_step(m, d)
        else:
            set_robot_pose(m, d, np.r_[self.initial[:7], target])
            d.qvel[: self.robot_nv] = 0.0

    def build_scene(self) -> None:
        """Set ``model``, ``data`` (the settled start state) and ``initial``, and write
        the scene's records (``scene.xml``, ``initial.npz``, ``scene_spec.json``, ``scene_reset.npz``)."""
        raise NotImplementedError

    def bridge_info(self) -> dict:
        """``bridge.json``, written last in the constructor."""
        raise NotImplementedError

    def measured(self):
        d = self.data
        return d.qpos[self.qadr], d.qvel[self.vadr], d.qpos[3:7], d.qvel[3:6]

    def publish(self) -> None:
        self.ring.write(self.frame, self.stamp, self.data)
        super().publish()

    def step(self, command) -> None:
        target, kp, kd = self.take_command(command)
        m, d = self.model, self.data
        if self.frame % FINGER_POLL_FRAMES == 0:
            changes = self.finger_commands.changes()
            for side, state in changes.items():
                self.finger_stages.pop(side, None)
                if state == "closed":
                    self.set_fingers(side, closed=True)
                else:  # thumb first (Hand.release)
                    self.finger_stages[side] = [
                        (self.frame + round(delay * FPS), joints) for delay, joints in current_hand().release()
                    ]
            for side, stages in list(self.finger_stages.items()):
                while stages and stages[0][0] <= self.frame:
                    self.set_fingers(side, closed=False, joints=stages.pop(0)[1])
                if not stages:
                    del self.finger_stages[side]
            if changes:
                with (self.directory / "fingers.jsonl").open("a") as log:
                    log.write(json.dumps({"frame": self.frame, "time": float(d.time), **changes}) + "\n")
        self.advance(target, kp, kd)
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all():
            raise FloatingPointError("simulation diverged")
        self.log((d.qpos.copy(), d.qvel.copy(), float(d.time), target))
        self.frame += 1

    # ------------------------------------------------------------ the physics log, on its own thread
    def write_chunk(self, number, rows, final) -> bool:
        """``physics_NNNN.npz``, written on the log thread. A chunk that fills while the one before is still being
        written is dropped (counted in ``dropped_log_frames``): the sim clock never waits on the disk. The final one
        waits for it and is always written. A write that fails is kept (``write_error``, the first) and raised when
        the bridge stops (``finish``): the run goes on, its other chunks are written, and the error is never lost."""
        if final:
            self.written()
        elif self.pending_write is not None:
            if not self.pending_write.done():
                self.dropped_log_frames += len(rows)
                return False
            self.written()
        self.pending_write = self.writer.submit(save_chunk, self.directory / f"physics_{number:04d}.npz", rows)
        return True

    def written(self) -> None:
        """Wait until the chunk being written is on disk; its error, if any, becomes ``write_error`` (the first)."""
        pending, self.pending_write = self.pending_write, None
        if pending is not None and pending.exception() is not None and self.write_error is None:
            self.write_error = pending.exception()

    # ------------------------------------------------------------ protocol hooks: the log writer
    def finish(self) -> None:
        self.written()  # the final chunk on disk
        self.writer.shutdown(wait=True)
        if self.write_error is not None:  # ProtocolBridge.end records it in shutdown.json, then raises it
            raise self.write_error

    def shutdown_record(self) -> dict:
        return {"frames": self.frame, "sim_time": float(self.data.time), "dropped_log_frames": self.dropped_log_frames}


class Bridge(MujocoBridge):
    """A robot on a scene of ``wbc_link_server.scenes``."""

    def build_scene(self) -> None:
        args = self.args
        spec = scenes.scene(args.scene, args.seed)
        self.model = scenes.build_scene(spec, xml_path=self.directory / "scene.xml")
        self.data = mujoco.MjData(self.model)
        initial = current_hand().rest(np.load(args.initial, allow_pickle=False)["qpos"][0])
        offset = spec.get("initial_offset", {})
        initial[:2] += offset.get("xy", [0.0, 0.0])
        initial[3:7] = (
            Rotation.from_euler("z", args.heading_deg + offset.get("heading_deg", 0.0), degrees=True)
            * Rotation.from_quat(initial[3:7], scalar_first=True)
        ).as_quat(scalar_first=True)
        self.initial = initial
        initialize_scene(self.model, self.data, spec, initial)
        np.savez(self.directory / "initial.npz", qpos=initial[None])
        # The scene's own evidence; not part of observations or link state.
        (self.directory / "scene_spec.json").write_text(json.dumps(spec, indent=2))
        np.savez(self.directory / "scene_reset.npz", qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy())

    def bridge_info(self) -> dict:
        return {
            "source_id": self.source,
            "scene": self.args.scene,
            "seed": self.args.seed,
            "state_port": self.args.state_port,
            "command_port": self.args.command_port,
            "physics_dt": self.model.opt.timestep,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--channel", type=Path, required=True, help="the controller's motion channel directory")
    p.add_argument("--state-port", type=int, required=True)
    p.add_argument("--command-port", type=int, required=True)
    p.add_argument("--scene", default="flat")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heading-deg", type=float, default=0.0)
    p.add_argument("--initial", type=Path, default=INITIAL_REFERENCE)
    p.add_argument("--paused", action="store_true")
    args = p.parse_args()
    logs.setup("bridge", logs.session_of(args.directory))
    Bridge(args).run()


if __name__ == "__main__":
    main()
