# GPT-6.2 Sol Medium on HITCH

2026-11-18. Runs used a ChatGPT Plus login on tom, Codex CLI 0.166.0, the
built-in OpenAI provider, model `gpt-6.2-sol`, medium reasoning and seed 0.
The HITCH planner, HEFT checkpoint, Dex3, robot interface, task text, and
independent scoring were retained. HEFT checkpoint SHA-256:
`8594fa2739036da4a57602f47f3be9560ca9c0c42b2a448add7d9bb133792aae`.

| Task | Independent result | Robot calls | Episode wall time |
|---|---|---:|---:|
| HITCH tabletop pick | Pass. Final lift 0.13620 m. Pelvis height passed the upright criterion. | 14 | 191.0 s |
| LocoManiLab kitchen approach and lift apple, scene 13 | Fail. Approach complete. Maximum target rise 0.13660 m against the required 0.30 m. | 15 | 171.3 s |

The tabletop agent used fresh depth measurements after walking, approached
through several separate waypoints with an inward palm, closed the right
hand after positioning it, and lifted it in a separate motion. The final
image and measured points supported its success report. The physics trace
records the right-hand closure at 10.98 s and a final rise of 13.62 cm.

On the kitchen task, the agent walked to the counter, closed the left hand
at 12.50 s, and generated a lift waypoint 0.14 m above the measured palm.
The model reported success and ended the episode at 762 of 1308 native steps.
The native score records stage 1 with zero operation hold steps.

Offline comparison with the recorded physics identifies the exact native
target body `apple_main`. Its center started at 0.955380 m, peaked at
1.091981 m, and ended at 1.091247 m. The native criterion requires a center
height of at least 1.255380 m for 1 second. No recorded frame reached that
height. The final score is consistent with the trajectory and no
repeat was needed to resolve the scoring discrepancy. The standard task
prompt was “walk to the table, then lift the apple” and omitted both the
height and hold thresholds. This is an observed failure of the task under
the standard prompt and scoring, with a confirmed grasp and insufficient
lift. It does not establish that the model or tracker cannot execute a
higher lift. Evaluation stopped at this first failed composite task.

Both episodes used 180 robot calls and 1800 model-runtime wall seconds as
limits. The subscription uses native Codex requests, so the 120-request
DeepSeek gateway cap is not available. This is a comparison of model and
service combinations with matching scene seeds and control, rather than
a matched request-budget or causal provider comparison. Third-party
DeepSeek recorded 0/3 on the same tabletop seed. GPT-6.2 recorded 1/1.

Runs on tom:

- `data/gpt62_tabletop_seed0/episodes/coding-20261118-111820-987517`
- `data/gpt62_native_apple/episodes/coding-20261118-112927-486451`

Both retained `episode.json`, `robot.jsonl`, `codex.jsonl`, the generated
program snapshots, and the simulator's physics and shutdown records.
Both physics shutdown records show zero dropped frames and no error.
The runner source hashes in the records identify the subscription changes
used during the runs. The code was still uncommitted at runtime, so the
recorded Git base is `9142c1d`; the source is published in `e9da26f902a19e5eba23068bd85f1442c69386a5`.
