"""Native LocoManiLab tasks driven by the unchanged HITCH PD/hand protocol.

Run in the isolated benchmark interpreter. The native environment owns reset,
physics advancement, stage latches and success; HITCH owns every motor command.
No benchmark object state or success information belongs in agent observations.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from wbc.core.jsonhttp import write_json_atomic
from wbc.core.layout import JOINTS
from wbc.paths import INITIAL_REFERENCE, PROJECT
from wbc.robot.hands import current_hand
from wbc.robot.model import _model_xml
from wbc_link_server.benchmarks.mjcf import camera_xml, inline_defaults

SOURCES = PROJECT / "data/locomanilab/src"
PREFIX = "robot0_"
CONTROL_FREQ = 50
REVISIONS = {
    "LocoManiLab": "306d62dc9a271f717d56a41be52fd777b244cb6d",
    "humanoid_benchmark": "2f96b798aa85e32444db582f4ca9bfdefe79c7f3",
    "g1_backend": "25d262a56a6f034e679c0f4bc6be63585233ad20",
}


def robot_xml(directory: Path) -> tuple[Path, np.ndarray]:
    """Adapt names / metadata, retaining HITCH's physical robot and Dex3 drives."""
    hand = current_hand()
    if hand.name != "dex3":
        raise ValueError("LocoManiLab integration requires WBC_HAND=dex3")
    initial = hand.rest(np.load(INITIAL_REFERENCE, allow_pickle=False)["qpos"][0])
    root = ET.fromstring(_model_xml(hand=hand, simulate=True))
    reference = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    inline_defaults(root)
    world = root.find("worldbody")
    for element in list(world):
        if element.tag != "body":
            world.remove(element)  # Native scene owns floor, lights, and scenery.
    for index, geom in enumerate(world.iter("geom")):
        if "name" not in geom.attrib:
            geom.set("name", f"hitch_geom_{index}")
    # RobotModel otherwise inserts its own damping/friction/armature defaults.
    for joint in world.iter("joint"):
        jid = reference.joint(joint.get("name")).id
        address = reference.jnt_dofadr[jid]
        for name, values in (("damping", reference.dof_damping),
                             ("armature", reference.dof_armature),
                             ("frictionloss", reference.dof_frictionloss)):
            joint.set(name, str(float(values[address])))
    path = directory / "hitch_robot.xml"
    path.write_text(ET.tostring(root, encoding="unicode"))
    return path, initial


def register_robot(path: Path, initial: np.ndarray) -> None:
    """Process-local replacement of the native robot constructor, not task logic.

    Native reset and camera code select G1FloatingBody by its registered name.
    The manifest identifies the actual robot as HITCH; no upstream source is edited.
    """
    from robosuite.models.robots.manipulators.g1_robot import G1FloatingBody as NativeG1

    hand = current_hand()
    positions = dict(zip(JOINTS, initial[7:], strict=True))
    for joint in ET.parse(path).iter("joint"):
        name = joint.get("name")
        if "_hand_" in name:
            side, hand_joint = name.split("_", 1)
            positions[name] = hand.hold_target(side, hand_joint)

    class G1FloatingBody(NativeG1):
        xml_path = str(path)

        @property
        def init_qpos(self):
            return np.array([positions[name.removeprefix(self.naming_prefix)] for name in self.joints])

        @property
        def bottom_offset(self):
            # Native task XYZ denotes the floor; HITCH's root is the physical pelvis.
            return np.array([0.0, 0.0, -initial[2]])

        def add_mobile_base(self, mobile_base):
            # Keep pelvis inertia and kinematics intact. The stock implementation
            # moves physical bodies into an elevated manipulator_mount frame.
            self.base = mobile_base
            anchor = copy.deepcopy(mobile_base.worldbody[0])
            anchor.set("pos", "0 0 0")
            for inertial in list(anchor.iter("inertial")):
                anchor.remove(inertial)
            self._elements["root_body"].append(anchor)
            self.merge_assets(mobile_base)
            self.cameras = self.get_element_names(self.worldbody, "camera")

        @property
        def contact_geoms(self):
            # HITCH uses groups 3/4 for collisions, while robosuite lists group 0.
            # Keep the original groups (camera semantics) and expose their names.
            return [geom.get("name") for geom in self.worldbody.iter("geom")
                    if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1"))]


def register_controller(initial: np.ndarray) -> dict:
    from robosuite.controllers import load_composite_controller_config
    from robosuite.controllers.composite.composite_controller import (
        CompositeController,
        register_composite_controller,
    )

    @register_composite_controller
    class HitchPD(CompositeController):
        name = "HITCH_PD"

        def load_controller_config(self, part_controller_config, composite_controller_specific_config=None):
            self.part_controller_config = part_controller_config
            model = self.sim.model
            self.qadr = np.array([model.get_joint_qpos_addr(PREFIX + name) for name in JOINTS])
            self.vadr = np.array([model.get_joint_qvel_addr(PREFIX + name) for name in JOINTS])
            self.actuators = np.array([model.actuator_name2id(PREFIX + name.removesuffix("_joint"))
                                       for name in JOINTS])
            self.command = None

        def spawn_joint_targets(self):
            return list(zip(self.qadr, self.vadr, initial[7:], strict=True))

        def set_goal(self, action):
            pass

        def update_state(self):
            pass

        @property
        def action_limits(self):
            return np.zeros(1), np.zeros(1)

        def create_action_vector(self, action_dict):
            return np.zeros(1)

        def run_controller(self, enabled_parts):
            if self.command is None:
                raise RuntimeError("native physics step requires an external HITCH command")
            target, kp, kd = self.command
            data, model = self.sim.data, self.sim.model
            torque = kp * (target - data.qpos[self.qadr]) - kd * data.qvel[self.vadr]
            limits = model.actuator_ctrlrange[self.actuators]
            data.ctrl[self.actuators] = np.clip(torque, limits[:, 0], limits[:, 1])
            return {}

    config = load_composite_controller_config(robot="G1FloatingBody")
    config["type"] = "HITCH_PD"
    config["composite_controller_specific_configs"] = {}
    return config


def make_native_env(env_name: str, directory: Path, *, scene_id=None, scene_seed=0, seed=0):
    """Construct the native host. EmptyEnv is useful for asset-free contract checks."""
    import robocasa  # noqa: F401 -- registers native tasks and deferred controllers
    import robosuite

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path, initial = robot_xml(directory)
    register_robot(path, initial)
    config = register_controller(initial)
    kwargs = {}
    if scene_id is not None:
        kwargs.update(scene_ids=scene_id, fixed_scene_seed=scene_seed)
    env = robosuite.make(
        env_name=env_name, robots="G1FloatingBody", controller_configs=config,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=CONTROL_FREQ, initialization_noise=None, ignore_done=True,
        renderer="mujoco", seed=seed, **kwargs,
    )
    return env


class NativeTask:
    """One native benchmark episode with private scoring and external HITCH PD."""

    def __init__(self, task_id: str, directory: Path, seed: int = 0, episode: int = 0):
        import robocasa  # noqa: F401
        from locomanilab.runtime.sim_client import apply_task_overrides

        self.directory = Path(directory)
        self.task_id, self.seed, self.episode = task_id, seed, episode
        table = yaml.safe_load((SOURCES / "LocoManiLab/src/locomanilab/benchmark/tasks.yaml").read_text())
        self.spec = {**table["defaults"], **table["tasks"][task_id]}
        self.overrides = apply_task_overrides(self.spec["env"], self.spec["scene_id"])
        self.env = make_native_env(
            self.spec["env"], self.directory, scene_id=self.spec["scene_id"],
            scene_seed=self.spec["scene_seed"], seed=seed,
        )
        self.reset()

    def reset(self):
        from locomanilab.runtime.sim_client import _apply_reset_jitter, _initial_state_digest

        obs = self.env.reset()
        _apply_reset_jitter(self.env, self.env, SimpleNamespace(**self.spec), obs, self.episode)
        self.model = self.env.sim.model._model
        self.data = self.env.sim.data._data
        if not np.isclose(self.model.opt.timestep, self.env.model_timestep):
            raise ValueError("native environment and MuJoCo physics clocks disagree")
        self.controller = self.env.robots[0].composite_controller
        self.qadr, self.vadr = self.controller.qadr, self.controller.vadr
        root = self.model.joint(PREFIX + "floating_base_joint").id
        self.root_qadr = int(self.model.jnt_qposadr[root])
        self.root_vadr = int(self.model.jnt_dofadr[root])
        self.initial = np.r_[self.data.qpos[self.root_qadr:self.root_qadr + 7], self.data.qpos[self.qadr]]
        self.started_at = float(self.data.time)
        self.steps = 0
        self._terminal = None
        for side in ("left", "right"):
            self.set_fingers(side, False)
        self.save_scene(self.directory / "scene.xml")
        np.savez(self.directory / "initial.npz", qpos=self.initial[None])
        np.savez(self.directory / "scene_reset.npz", qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy())
        self.manifest = {
            "task": self.task_id, "native_env": self.spec["env"], "scene_id": self.spec["scene_id"],
            "robot_variant": "HITCH G1 + Dex3", "controller": "external HITCH HEFT PD",
            "robot_initialization": "HITCH initial reference; native spawn XY/yaw and floor calibration",
            "physics_dt": float(self.model.opt.timestep), "control_freq": CONTROL_FREQ,
            "physics_configuration": "native benchmark", "hitch_original_physics_dt": 0.005,
            "max_steps": self.spec["max_steps"], "genuine_min_steps": self.spec["genuine_min_steps"],
            "scene_seed": self.spec["scene_seed"], "seed": self.seed,
            "reset_jitter": self.spec["reset_jitter"], "reset_object_jitter": self.spec["reset_object_jitter"],
            "jitter_seed": self.spec["jitter_seed"] + self.episode,
            "task_overrides": self.overrides, "initial_state": _initial_state_digest(self.env),
            "mujoco_version": mujoco.__version__, "scoring": "native _post_action and _check_success unchanged",
            "source_revisions": REVISIONS,
        }
        write_json_atomic(self.directory / "native_manifest.json", self.manifest)
        write_json_atomic(self.directory / "scene_spec.json", {
            "kind": "locomanilab", "benchmark_task": self.task_id,
            "task": {"instruction": self.spec["prompt"]}, "seed": self.seed,
        })
        self._write_score()

    def measured(self):
        q, v = self.root_qadr, self.root_vadr
        return (self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy(),
                self.data.qpos[q + 3:q + 7].copy(), self.data.qvel[v + 3:v + 6].copy())

    def set_fingers(self, side, closed, joints=None):
        hand = current_hand()
        names = joints if joints is not None else {**hand.hold, **(hand.grip or {})}
        for joint in names:
            actuator = self.model.actuator(PREFIX + f"{side}_{joint.removesuffix('_joint')}").id
            self.data.ctrl[actuator] = hand.finger_target(side, joint, closed)

    def hold(self, target):
        """HITCH's pre-start hold; no episode time or task state is advanced."""
        self.data.qpos[self.root_qadr:self.root_qadr + 7] = self.initial[:7]
        self.data.qpos[self.qadr] = target
        for joint in self.env.robots[0].robot_model.joints:
            address = self.env.sim.model.get_joint_qvel_addr(joint)
            self.data.qvel[address] = 0
        self.data.qvel[self.root_vadr:self.root_vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, target, kp, kd):
        if self._terminal is not None:
            return self._terminal.copy()
        command = tuple(np.asarray(value, dtype=float) for value in (target, kp, kd))
        if any(value.shape != (len(JOINTS),) or not np.isfinite(value).all() for value in command):
            raise ValueError("PD command must contain three finite canonical 29-vectors")
        self.controller.command = command
        self.env.step(np.zeros(1))
        self.steps += 1
        if not np.isfinite(self.data.qpos).all():
            raise FloatingPointError("native simulation diverged")
        result = self.score()
        if result["terminal"]:
            self._terminal = result
        if self.steps % CONTROL_FREQ == 0 or result["terminal"]:
            self._write_score()
        return result

    def score(self):
        if self._terminal is not None:
            return self._terminal.copy()
        success = bool(self.env._check_success())
        exhausted = self.steps >= self.spec["max_steps"]
        stages = {name: int(getattr(self.env, name)) for name in
                  ("_walk_stage", "_walk_approach_hold", "_walk_operation_hold", "_approach_stage",
                   "_approach_hold", "_operation_hold", "_routine_stage", "_routine_hold")
                  if hasattr(self.env, name)}
        return {
            "task": self.task_id, "success": success,
            "genuine": success and self.steps >= self.spec["genuine_min_steps"],
            "terminal": success or exhausted,
            "reason": "native_success" if success else "step_budget" if exhausted else "running",
            "steps": self.steps, "max_steps": self.spec["max_steps"], "stage": stages,
            "sim_time": float(self.data.time) - self.started_at,
        }

    def _write_score(self):
        write_json_atomic(self.directory / "benchmark_result.json", self.score())

    def save_scene(self, path: Path):
        """Camera-only name translation; task evaluation retains prefixed native XML."""
        # mj_saveLastXML rounds the native scene's 1e-8 visual geom masses to
        # zero, making some static bodies impossible to recompile. The source
        # model retains their precision; live free-joint poses come via FrameRing.
        Path(path).write_text(camera_xml(self.env.model.get_xml()))

    def close(self):
        self._write_score()
        self.env.close()
