"""Full native-scene camera export contract; no rendering, model, or API."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from wbc_link_server.benchmarks.native import PREFIX, NativeTask


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "IThorKitchenApproachTableAndLiftApple_scene13"
    directory = Path("data/locomanilab/probe/camera_contract") / task_id
    task = NativeTask(task_id, directory)
    try:
        camera = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        host = task.model
        assert (camera.nq, camera.nv, camera.nu, camera.nbody, camera.ngeom) == (
            host.nq, host.nv, host.nu, host.nbody, host.ngeom)
        for kind, count in (("joint", host.njnt), ("body", host.nbody), ("geom", host.ngeom)):
            for index in range(count):
                assert getattr(camera, kind)(index).name == getattr(host, kind)(index).name.removeprefix(PREFIX)
        for name in ("jnt_qposadr", "jnt_dofadr", "actuator_trnid", "body_parentid"):
            np.testing.assert_array_equal(getattr(camera, name), getattr(host, name), err_msg=name)
        for name in ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_pos", "geom_quat",
                     "geom_size", "geom_rgba", "geom_contype", "geom_conaffinity", "cam_fovy"):
            np.testing.assert_allclose(getattr(camera, name), getattr(host, name),
                                       atol=1e-10, rtol=1e-8, err_msg=name)
        assert camera.camera("ego").id >= 0
        data = mujoco.MjData(camera)
        data.qpos[:] = task.data.qpos
        data.qvel[:] = task.data.qvel
        mujoco.mj_forward(camera, data)
        np.testing.assert_allclose(data.xpos, task.data.xpos, atol=1e-9)
        (directory / "result.json").write_text(json.dumps({
            "passed": True, "task": task_id, "mujoco": mujoco.__version__,
            "checks": ["full_scene_recompile", "state_order", "static_geometry", "world_poses"],
        }, indent=2) + "\n")
        print("FULL_SCENE_CAMERA_CONTRACT_OK", task_id, flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()
