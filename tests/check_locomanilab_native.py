"""Explicit integration check in the isolated native interpreter (no API / HEFT).

Run from the HITCH root with WBC_HAND=dex3 and the benchmark PYTHONPATH.
This file is deliberately outside default unit-test discovery: it needs the
pinned optional native simulator, and exercises that real dependency.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

from wbc.core.layout import JOINTS
from wbc.robot.model import _model_xml
from wbc_link_server.benchmarks.mjcf import camera_xml
from wbc_link_server.benchmarks.native import PREFIX, make_native_env


def main():
    directory = Path("data/locomanilab/probe/native_contract")
    reference = mujoco.MjModel.from_xml_string(_model_xml(simulate=True))
    env = make_native_env("EmptyEnv", directory)
    model, data = env.sim.model._model, env.sim.data._data
    try:
        # Dynamics and geometry are compared per named physical body, excluding
        # native world/attachment frames. Root placement is owned by task reset.
        for index in range(1, reference.nbody):
            name = reference.body(index).name
            actual = model.body(PREFIX + name).id
            for attr in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
                np.testing.assert_allclose(getattr(model, attr)[actual], getattr(reference, attr)[index],
                                           atol=1e-10, rtol=1e-8, err_msg=f"{name}: {attr}")
            if name != "pelvis":
                for attr in ("body_pos", "body_quat"):
                    np.testing.assert_allclose(getattr(model, attr)[actual], getattr(reference, attr)[index],
                                               atol=1e-10, rtol=1e-8, err_msg=f"{name}: {attr}")
            expected_geoms = np.flatnonzero(reference.geom_bodyid == index)
            actual_geoms = np.flatnonzero(model.geom_bodyid == actual)
            assert len(actual_geoms) == len(expected_geoms), name
            for attr in ("geom_type", "geom_pos", "geom_quat", "geom_size", "geom_friction",
                         "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity",
                         "geom_condim", "geom_margin", "geom_gap"):
                np.testing.assert_allclose(getattr(model, attr)[actual_geoms],
                                           getattr(reference, attr)[expected_geoms], atol=1e-8, rtol=1e-7,
                                           err_msg=f"{name}: {attr}")
        for index in range(reference.njnt):
            name = reference.joint(index).name
            actual = model.joint(PREFIX + name).id
            for attr in ("jnt_type", "jnt_range", "jnt_axis"):
                np.testing.assert_allclose(getattr(model, attr)[actual], getattr(reference, attr)[index],
                                           atol=1e-10, rtol=1e-8, err_msg=f"{name}: {attr}")
            if reference.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE:
                expected_dof, actual_dof = reference.jnt_dofadr[index], model.jnt_dofadr[actual]
                for attr in ("dof_damping", "dof_armature", "dof_frictionloss"):
                    np.testing.assert_allclose(getattr(model, attr)[actual_dof],
                                               getattr(reference, attr)[expected_dof], atol=1e-10,
                                               err_msg=f"{name}: {attr}")
        for index in range(reference.nu):
            name = reference.actuator(index).name
            actual = model.actuator(PREFIX + name).id
            for attr in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gear",
                         "actuator_gainprm", "actuator_biasprm", "actuator_dynprm"):
                np.testing.assert_allclose(getattr(model, attr)[actual], getattr(reference, attr)[index],
                                           atol=1e-10, rtol=1e-8, err_msg=f"{name}: {attr}")
        # Compile the camera translation and check exact state / actuator order.
        camera = mujoco.MjModel.from_xml_string(camera_xml(env.model.get_xml()))
        assert (camera.nq, camera.nv, camera.nu) == (model.nq, model.nv, model.nu)
        for index in range(model.njnt):
            assert camera.joint(index).name == model.joint(index).name.removeprefix(PREFIX)
        assert camera.camera("ego").id >= 0
        np.testing.assert_array_equal(camera.jnt_qposadr, model.jnt_qposadr)
        np.testing.assert_array_equal(camera.actuator_trnid, model.actuator_trnid)

        controller = env.robots[0].composite_controller
        assert len(controller.qadr) == len(JOINTS) == 29
        # Extreme positive targets make the clipping result exact at every
        # physics substep, independently of the integrator's final qpos.
        controller.command = (np.full(29, 1e6), np.ones(29), np.zeros(29))
        before = float(data.time)
        env.step(np.zeros(1))
        np.testing.assert_allclose(data.ctrl[controller.actuators],
                                   model.actuator_ctrlrange[controller.actuators, 1])
        np.testing.assert_allclose(float(data.time) - before, 1 / 50, atol=1e-12)
        assert env.timestep == 1
        (directory / "result.json").write_text(json.dumps({
            "passed": True, "mujoco": mujoco.__version__,
            "checks": ["body_dynamics", "collision_geometry", "joint_parameters", "actuator_parameters",
                       "camera_state_order", "torque_clipping", "control_clock"],
            "physics_dt": float(model.opt.timestep), "control_dt": float(data.time) - before,
        }, indent=2) + "\n")
        print("NATIVE_CONTRACT_OK: robot dynamics, joint mapping, camera state order, torque clip, 50Hz clock")
    finally:
        env.close()


if __name__ == "__main__":
    main()
