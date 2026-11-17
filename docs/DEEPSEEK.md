# DeepSeek simulation tests

This checkout adds a DeepSeek Responses backend to the existing HITCH agent loop.
Planner, tracker, simulation, tool semantics, and task scoring are unchanged.

## Setup

The original pilot used uv from $HOME/.local/bin. If a machine's global package
mirror differs from the upstream lockfile, setup can select the original index:

```bash
PATH="$HOME/.local/bin:$PATH" \
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
NO_PROXY=127.0.0.1,localhost \
scripts/setup --default-index https://pypi.org/simple
```

The original uv.lock is unchanged. The proxy shown above was needed on the pilot host; omit it when direct access works.

## Run

```bash
export DEEPSEEK_API_KEY_FILE=/path/to/key-file
export HITCH_MODEL_PROXY=http://127.0.0.1:7890
WBC_DATA="$PWD/data/deepseek_flat" scripts/run-deepseek flat_walk
WBC_DATA="$PWD/data/deepseek_tabletop_seed0" \
  scripts/run-deepseek tabletop_pick --seed 0 --max-turns 40
```

An existing DEEPSEEK_API_KEY takes precedence over the file. The launcher does not
print or store the key. HITCH_MODEL_PROXY is optional; unset it on machines that
can reach the model API directly. Robot/session traffic retains its direct
loopback connection. The proxy_env profile field stores only the environment
variable name, not its value.

The DeepSeek profile uses deepseek-flash, high reasoning effort, 32768 maximum
output tokens per reply, a 180-second API timeout, and one retry.

## Scope and interpretation

- flat is an upstream scene without an upstream agent task or success checker.
  flat_walk is our added smoke task: walk forward 0.30 m and remain standing.
  Actual displacement can be inspected from the saved physics logs.
- tabletop_pick is the unchanged upstream task, with Dex3 hands and the original
  success checker. This pilot uses seed 0 and a 40-reply budget; upstream defaults
  to 100 replies.
- Both use paused=1, matching the upstream task's idle-time simulation behavior.
- The fake backend only measures, walks, and measures again. Its false task
  success is expected and does not indicate an infrastructure failure.
- Network-aborted attempts are retained in the data directories and must not be
  counted as task-performance trials.
- The user excluded HITCH-27B deployment/testing. These runs do not establish a
  performance comparison with HITCH-27B or the paper's benchmarks.

## Records

HITCH writes episode.json, messages.jsonl, requests, images, and transcript.md
under each WBC_DATA/episodes directory. Session directories contain plans,
execution records, and runtime/sim/physics_*.npz. DeepSeek's success claim and the
tabletop scene's check.success are separate fields.
