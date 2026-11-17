# Entry B: Codex programs using the HITCH robot tools

Entry B runs the installed Codex CLI with a selected DeepSeek Responses provider. The model writes and executes
Python programs, reads head-camera images, inspects structured results, and revises its
program. Every program talks to one persistent `wbc.agent.tools.Tools` instance. Motion
planning, perception, grip sequencing, HEFT tracking and MuJoCo are unchanged. Entry A
(`scripts/run-deepseek`) remains available.

## Run

The existing HITCH setup and checkpoint are required, together with Codex CLI 0.166.0,
Linux bubblewrap, and system Python 3 with NumPy and Pillow. The sandbox checks these
before starting a session. This first entry supports the two simulation tasks only;
it does not launch a real robot or add another simulator.

```bash
cd HITCH
export PATH="$HOME/.local/bin:$PATH"
export DEEPSEEK_API_KEY_FILE=/path/to/key-file
export HITCH_MODEL_PROXY=http://127.0.0.1:7890

# Optional API/runtime check; does not start a simulator.
scripts/coding flat_walk --probe --max-decisions 5 --wall-seconds 180

scripts/coding flat_walk --root data/coding_flat
scripts/coding tabletop_pick --seed 0 --root data/coding_tabletop_seed0
```

The default `--provider deepseek` preserves the official API baseline
(`https://api.deepseek.com/responses`, `deepseek-flash`). Select the tested
third-party provider explicitly:

```bash
export DEEPSEEK_API_KEY_FILE=/path/to/provider-key.md
scripts/coding flat_walk --provider dpsk --probe --max-decisions 5 --wall-seconds 180
scripts/coding tabletop_pick --provider dpsk --seed 0 --root data/dpsk_tabletop_seed0
```

`dpsk` uses `https://ai.cnjsbc.com/v1/responses` and `deepseek-v4.1-flash`.
Both providers retain high reasoning effort and the 32,768 output-token cap.
The gateway and Codex model catalog use the same selected model. The runtime
catalog is a read-only per-run copy; the checked-in template and user configuration
are unchanged. Provider, endpoint and requested model are recorded in `episode.json`;
a provider's model label alone does not prove equivalence to another service.

The key file may contain a plain `sk-` key or Markdown with exactly one such
credential; zero or multiple credentials are rejected without printing the contents.
`DEEPSEEK_API_KEY` may supply the key directly instead of the file. The parent reads
the key; it is neither passed to generated programs nor written into episode logs.
`HITCH_MODEL_PROXY` affects model traffic only. Omit it when a direct connection works.
No user Codex configuration is changed. Provider settings and the project model catalog
are passed to this invocation only.

Defaults are 40 model requests, 60 robot calls and 1,200 wall seconds per episode.
The wall limit begins when Codex starts. On stopping, the runner terminates the agent,
drains in-flight requests, closes the HITCH session and runs independent scoring.
Cleanup can take additional time if a robot action or model request is already running.
`observe`, `perceive` and `execute` consume the robot budget; initial observation counts.
`finish` remains available after the robot budget is exhausted.

## Native LocoManiLab tasks

Prepare the pinned sources, independent Python environment and selected assets
using [LOCOMANILAB_SETUP.md](LOCOMANILAB_SETUP.md). Then run an exact task ID
from the pinned LocoManiLab `tasks.yaml`:

```bash
export WBC_BENCHMARK_PYTHON="$PWD/data/locomanilab/.venv/bin/python"
scripts/coding locomanilab --provider dpsk \
  --benchmark-task IThorKitchenApproachTableAndLiftApple_scene13 \
  --seed 0 --root data/coding_native_apple \
  --max-decisions 120 --max-calls 180 --wall-seconds 1800
```

Codex remains online throughout the episode and uses the same robot API. The
native interpreter runs physics and cameras; the original HITCH environment
still runs planning and HEFT control. Native task reset, stage updates, success
checks and simulation-step limits are retained. The physical robot is HITCH's
G1 with Dex3, rather than the benchmark's default robot/controller combination;
this is recorded as an embodiment variant, not an official baseline reproduction.

The private simulator directory contains `native_manifest.json` and
`benchmark_result.json`. The runner stops the agent when the native task becomes
terminal and copies its final score into `episode.json` under `check`. These
files are outside the agent workspace. A valid task pass requires both native
`success` and `genuine`; an early raw success below the native minimum-step
threshold is reported separately. Agent self-assessment is never the task score.
The native step budget applies in addition to the model, robot-call and wall
budgets; paused model deliberation does not consume simulated task steps.

## Program interface

The agent receives `ROBOT.md`, `initial.json` and the initial RGB image. `ROBOT.md` embeds
the current HITCH motion and hand reference, plus the Python usage contract.

```python
import robot

obs = robot.observe()
result = robot.execute(
    {"walk": {"target": [0.30, 0.0], "turn_deg": 0}, "wait_s": 1.0},
    observation_id=obs["observation_id"],
)
print(result["text"])
print(result["images"])  # The model must open these with its image tool.
# Inspect the returned view and body state before making a success claim.
```

Other methods are `robot.perceive(points, observation_id=...)` and
`robot.finish(success, summary)`. Perception uses the same image-point format as A.
Results contain `text`, `images`, `observation_id`, `observation`, `calls`, `moved`
and `finished`; perception also returns `points`. An outdated observation ID is
rejected before any action. Every observation establishes a new coordinate frame:
after walking, measure again. Separate Python processes retain the same robot session.

The coding process has a writable `/workspace`, a read-only SDK and system runtime,
and isolated PID/network namespaces. A private Unix socket offers only robot calls
and the selected fixed Responses endpoint. The agent cannot read the repository,
simulator state, scoring data, real home directory or model key. Codex's inner shell
uses `danger-full-access` only inside this external bubblewrap sandbox. The simulator
and API credentials stay in the trusted parent process. This is an experiment boundary,
not a claim of protection against kernel exploits or all local resource exhaustion.

## Records and evaluation

Each printed episode directory contains:

- `workspace/`: generated programs, notes, model-visible images and final answer.
- `robot.jsonl`: ordered request/result/error events. Every request links Python
  source snapshots and SHA-256 hashes under `programs/`.
- `images/`: original observations outside the agent's writable workspace.
- `codex.jsonl` and `codex.stderr`: runtime tool events and diagnostics.
- `model-*.request.json` and `model-*.response`: model bodies/SSE, with no auth headers.
- `episode.json`: budgets, stop reason, session ID, self-report and independent score.

Sessions and physics logs are under the same data root's `sessions/`. Tabletop uses
the existing checker unchanged: for this pick scene the object must finish at least
3 cm above its initial position and the pelvis at least 45 cm high. That checker
does not independently prove a sustained stable grasp. Flat has no upstream task
checker; the runner records root displacement and pelvis height as `flat_physics`
without inventing an upstream success criterion.

Engineering acceptance is a functioning write/run/observe/revise/run loop. It does not
imply the grasp succeeds. Seed 0 is a development scene. A single A/B pilot is not a
model ranking or a generalization result; model requests in a coding runtime include
file inspection and editing, so their counts do not equal A's robot-planning turns.

## Checks

```bash
scripts/run -m unittest discover -s tests -p 'test_coding_*.py' -v
.venv/bin/ruff check src/wbc/coding tests/test_coding_*.py
.venv/bin/ruff format --check src/wbc/coding tests/test_coding_*.py
git diff --check
```

The provider catalog follows the official DeepSeek Codex integration metadata, with
a short project-owned instruction template:
<https://api-docs.deepseek.com/quick_start/agent_integrations/codex/>.

