# Native LocoManiLab environment

This setup keeps the original HITCH environment intact and installs the native
LocoManiLab task and asset dependencies in `data/locomanilab/.venv`. Run commands
from the HITCH repository root on Linux, with `uv` and Git available. If network
access requires a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` before downloading.

Verified: the three task classes register, all 4,511 selected asset files pass
upstream SHA-256 checks, and both native scene XMLs compile and run 200 physics
steps with finite qpos and qvel. All three task resets and camera-export contracts
passed. A kitchen smoke check obtained RGB images and executed a small torso
motion through the original HITCH planner and HEFT, remained standing, and
closed all four runtime processes cleanly. This is infrastructure validation;
formal Codex task success has not been established. The scene checks below do
not load an agent, a controller, or robot hardware.

## Pinned sources

| Checkout | Repository | Commit |
|---|---|---|
| `src/LocoManiLab` | `Gamma-Robot/LocoManiLab` | `306d62dc9a271f717d56a41be52fd777b244cb6d` |
| `src/humanoid_benchmark` | `LocoManiLab/humanoid_benchmark` | `2f96b798aa85e32444db582f4ca9bfdefe79c7f3` |
| `src/g1_backend` | `LocoManiLab/g1_backend` | `25d262a56a6f034e679c0f4bc6be63585233ad20` |

Paths in this table are relative to `data/locomanilab`. The second and third
commits are the gitlinks at `external/humanoid_benchmark` and
`external/g1_backend` in the pinned LocoManiLab commit. We check them out as
siblings and pass the asset path explicitly; no recursive submodule update is
needed. The remaining upstream submodules, training stacks, datasets, and
controller checkpoints are not part of this setup.

```sh
mkdir -p data/locomanilab/src data/locomanilab/manifests
git clone --no-checkout --filter=blob:none https://github.com/Gamma-Robot/LocoManiLab.git data/locomanilab/src/LocoManiLab
git -C data/locomanilab/src/LocoManiLab checkout --detach 306d62dc9a271f717d56a41be52fd777b244cb6d
git clone --no-checkout --filter=blob:none https://github.com/LocoManiLab/humanoid_benchmark.git data/locomanilab/src/humanoid_benchmark
git -C data/locomanilab/src/humanoid_benchmark checkout --detach 2f96b798aa85e32444db582f4ca9bfdefe79c7f3
git clone --no-checkout --filter=blob:none https://github.com/LocoManiLab/g1_backend.git data/locomanilab/src/g1_backend
git -C data/locomanilab/src/g1_backend checkout --detach 25d262a56a6f034e679c0f4bc6be63585233ad20
```

## Isolated dependencies

| Environment | Python | MuJoCo | NumPy |
|---|---|---|---|
| Existing HITCH `.venv` | 3.10 | 3.10.0 | 1.26.4 |
| Native `data/locomanilab/.venv` | 3.11.15 | 3.3.1 | 2.2.5 |

Use the explicit interpreter paths. Installing native dependencies into the
original HITCH environment would change the existing control pipeline's runtime.
The pinned requirements file captures the verified Linux environment; editable
source installs are separate so no machine-specific paths enter the requirements.

```sh
uv venv --python 3.11.15 data/locomanilab/.venv
uv pip install --python data/locomanilab/.venv/bin/python \
  --default-index https://pypi.org/simple -r requirements/locomanilab.txt
uv pip install --python data/locomanilab/.venv/bin/python \
  --default-index https://pypi.org/simple --no-deps \
  -e data/locomanilab/src/humanoid_benchmark/robosuite_src \
  -e data/locomanilab/src/humanoid_benchmark \
  -e data/locomanilab/src/LocoManiLab
uv pip freeze --python data/locomanilab/.venv/bin/python \
  > data/locomanilab/manifests/environment.freeze.txt
```

The generated freeze is a local diagnostic record and contains absolute editable
paths; do not commit it. `g1_backend` is retained as a pinned source reference,
not installed as a package: its legacy metadata requires Python 3.10. The HITCH
adapter does not import that control backend; only HITCH's `src` is needed on
`PYTHONPATH`. Importing the native task registry does not instantiate its
controllers or require new weights.

## Three selected tasks and assets

- `IThorKitchenApproachTableAndLiftApple_scene13`
- `IThorBedroomApproachDeskAndPutCanInBasket_scene329`
- `IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket_scene329`

Assets come from the public Hugging Face dataset `GammaRobotics/LocoManiLab` at
revision `c046046d93583255fdd4643cb094f76434e4ae2f`. Their deduplicated output size
is 268,850,417 bytes across 4,511 files; network archive overhead may be larger.
The official downloader verifies file hashes and skips verified files on rerun.
Check for an existing downloader before resuming an interrupted connection.

```sh
HF_HOME="$PWD/data/locomanilab/hf-cache" HF_HUB_DISABLE_XET=1 \
data/locomanilab/.venv/bin/python data/locomanilab/src/LocoManiLab/scripts/hf/download.py \
  --revision c046046d93583255fdd4643cb094f76434e4ae2f assets \
  --task IThorKitchenApproachTableAndLiftApple_scene13 \
  --task IThorBedroomApproachDeskAndPutCanInBasket_scene329 \
  --task IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket_scene329 \
  --assets-dir data/locomanilab/src/humanoid_benchmark/robocasa/models/assets
```

## Verification

```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy \
PYTHONPATH=src \
data/locomanilab/.venv/bin/python - <<'PY'
import robocasa
from robosuite.environments.base import REGISTERED_ENVS
for name in (
    "IThorKitchenApproachTableAndLiftApple",
    "IThorBedroomApproachDeskAndPutCanInBasket",
    "IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket",
):
    assert name in REGISTERED_ENVS, name
    print(name, "registered")
PY

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl data/locomanilab/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import mujoco
import numpy as np
root = Path("data/locomanilab")
assets = root / "src/humanoid_benchmark/robocasa/models/assets"
checks = []
for scene in (13, 329):
    xml = assets / f"scenes/ithor/FloorPlan{scene}_physics.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    result = {"scene": scene, "steps": 200, "finite_qpos_qvel": True,
              "bodies": model.nbody, "geoms": model.ngeom,
              "meshes": model.nmesh, "textures": model.ntex}
    checks.append(result)
    print(result)
(root / "manifests/scene_checks.json").write_text(
    json.dumps({"mujoco": mujoco.__version__, "checks": checks}, indent=2) + "\n")
PY
```

Verified scene counts are 330 bodies / 2,865 geoms for scene 13, and 101 bodies /
230 geoms for scene 329. These native scene checks are separate from constructing
the task with HITCH's G1 model and HEFT controller.

The provisioned environment also retains `manifests/setup_manifest.json`,
`manifests/{hf-revision.txt,manifest.tsv.gz,tasks.json.gz}` and
`logs/assets-download.log` under `data/locomanilab`. They record the source and
asset pins, selected tasks, and setup checks. Generated data, caches, source
checkouts, and the virtual environment stay outside Git.

## Adapter contracts

Run the real optional-dependency contracts in the isolated interpreter, without
an API key or policy inference:

```sh
export WBC_HAND=dex3 MUJOCO_GL=disable PYNPUT_BACKEND=dummy PYTHONPATH=src
data/locomanilab/.venv/bin/python tests/check_locomanilab_native.py
data/locomanilab/.venv/bin/python tests/check_locomanilab_reset.py
for task in \
  IThorKitchenApproachTableAndLiftApple_scene13 \
  IThorBedroomApproachDeskAndPutCanInBasket_scene329 \
  IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket_scene329
do
  data/locomanilab/.venv/bin/python tests/check_locomanilab_camera.py "$task"
done
```

The camera contract checks a full scene, not just the robot: state and actuator
indexing, static geometry, and synchronized world poses must match the native
host. Export uses the original construction XML. MuJoCo's compiled-model XML
export rounded tiny visual-geom masses from `1e-8` to zero in this asset set,
preventing camera recompilation; preserving original precision fixes that
without changing native physics.

The full kitchen clock probe also verified that three seconds of wall-clock
waiting before and after a motion did not advance simulation time once paused.
HITCH's existing startup stabilization consumed 30 native steps (0.60 s) in that
probe. Those steps remain included in the native task budget; no extra allowance
was added. This fixed settling cost must be distinguished from provider latency.

To repeat the full camera/control/paused-clock check without a model API:

```sh
WBC_HAND=dex3 PYNPUT_BACKEND=dummy MUJOCO_GL=egl \
WBC_BENCHMARK_PYTHON="$PWD/data/locomanilab/.venv/bin/python" \
scripts/run tests/check_locomanilab_stack.py
```

This launches one simulated kitchen session, makes a small torso motion, checks
idle time before and after it, closes the session, and records results under
`data/locomanilab/probe/pause_clock/`. It is an infrastructure check, not a
successful benchmark episode.
