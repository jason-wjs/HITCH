# Third-party DeepSeek tabletop evaluation

Date: 2026-11-18. Provider: `dpsk`, endpoint
`https://ai.cnjsbc.com/v1/responses`, requested model `deepseek-v4.1-flash`.
This is the provider's model label; it is a separate experiment from the official
DeepSeek Flash pilot in [CODING_RESULTS.md](CODING_RESULTS.md).

## Result

Three bounded online Codex episodes on `tabletop_pick`, seed 0, all failed the
unchanged independent HITCH checker. The initial instruction was to pick the
cyan cylinder off the round table and hold it with one hand. All three objects
ended on the floor; maximum lift was approximately 7 mm.
The checker requires final lift of at least 3 cm and pelvis height of at least
45 cm. Its `upright` field describes that pelvis-height threshold, not cylinder
orientation; a deliberate deep crouch can make it false without a robot fall.

| Attempt | Model requests | Robot calls | Episode wall time | Maximum lift | Final lift | Checker |
|---|---:|---:|---:|---:|---:|---|
| 1 | 120 | 118 | 1348.07 s | 0.007153 m | -0.882565 m | false |
| 2 | 120 | 173 | 1465.48 s | 0.007109 m | -0.882565 m | false |
| 3 | 120 | 180 | 1620.96 s | 0.007108 m | -0.882565 m | false |

Each episode allowed 120 model requests, 180 robot calls, and 1800 model-runtime
wall seconds. Episode wall time above includes startup and cleanup. All stopped
at the model-request budget; the third also reached the robot-call limit. Codex
CLI was 0.166.0, reasoning effort high, and maximum response output 32,768 tokens.
The same robot API, task seed, HITCH planner, and HEFT checkpoint were retained.
No prior successful task program or failure analysis was supplied to these agents.

All three simulator shutdown records report no error and no dropped physics-log
frames. API protocol checks, image/tool round trips, and an isolated Codex
write-and-run probe passed separately. Protocol compatibility therefore does not
establish successful task execution.

## First-attempt causal evidence

The first drop occurred during robot request sequence 32, `stage_reach.py`,
HITCH execution `run-5`. The model requested a palm pose near the cylinder with
downward-pointing fingers. HITCH reported that the planned orientation differed
from the requested orientation by about 48 degrees, after executing the plan.

Recorded joint states align observation 21 with simulation time 11.02 s and
observation 22 with 13.42 s, bracketing sequence 32. Reconstructed contacts show
the cylinder touching `right_hand_index_0_contact` at approximately 12.00 s;
it crossed below tabletop height at 12.54 s and contacted the floor at 12.88 s.
Before 11.90 s its world displacement was only about 5.8 micrometres. This
rules out the earlier hypothesis that the first pregrasp moves had already
pushed it off, and shows that sequence 34 began after the drop.

No closed-hand command had executed before the drop. A late closed-hand command
did execute during the subsequent floor-recovery attempt at 37.54 s; the whole
episode must not be described as having never closed the hand. The final
failure was not a confirmed grasp followed by slipping from a closed Dex3.

The harness enabled saved programs, measurement calculations, visual inspection,
and iterative recovery. Its shared `execute` operation combined planning and
execution, so the model learned of the large orientation deviation only after
the physical motion. Exposing the existing HITCH plan result before execution
is a concrete follow-up hypothesis, not a proven fix. This experiment did not
change planning tolerances, add an automatic grasp controller, or test that
hypothesis.

The third attempt did close the hand while the object was still near the table,
but did not establish a valid lifted hold before the object fell. Thus the first
attempt's sequence is not a sufficient explanation for all three failures;
grasp approach and contact geometry also need examination. A plan-preview tool
alone is not known to solve those failures.

This is three attempts at one fixed scene, not a statistical estimate of model
capability or proof that the new service is globally weaker than the official
model. The earlier official-provider success was one development run with a
different request budget. Native LocoManiLab agent trials remain gated on the
agreed tabletop acceptance decision; infrastructure checks are separate.

## Records

The ignored data directories on the experiment host contain full model, program,
robot, and physics logs:

- `data/dpsk_tabletop_seed0/episodes/coding-20261118-000326-633283`
- `data/dpsk_tabletop_seed0_attempt2/episodes/coding-20261118-005920-201743`
- `data/dpsk_tabletop_seed0_attempt3/episodes/coding-20261118-023818-315598`

Each `episode.json` contains the requested model, budgets, Codex version, Git
revision, coding-source hashes, session ID, independent score, and stop reason.
The first two runner source hashes match; the third includes the new native-task
entry and its private terminal monitor, with the tabletop path unchanged.
The HEFT checkpoint SHA-256 is
`8594fa2739036da4a57602f47f3be9560ca9c0c42b2a448add7d9bb133792aae`.
