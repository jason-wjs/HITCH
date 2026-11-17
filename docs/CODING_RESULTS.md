# Initial A/B development pilot

Date: 2026-11-17. HITCH base: 275afdf06c8b9ac6e3a3e17afeeec0d2ae0b2e88.
Model: official DeepSeek Flash, high reasoning. Codex CLI: 0.166.0.
HEFT checkpoint SHA-256: 8594fa2739036da4a57602f47f3be9560ca9c0c42b2a448add7d9bb133792aae.

| Task | Entry A | Entry B |
|---|---|---|
| flat_walk | 0.2950 m forward, standing | 0.2950 m forward, standing |
| tabletop_pick, seed 0 | original checker false; object fell off table | original checker true; final lift 0.18385 m, upright |

B's tabletop run used 40 model requests and 37 robot calls (19 observe, 7 perceive,
11 execute), plus finish. It submitted success before the model budget stopped
the next request. The robot held the lifted object for a 2 s simulation wait.

The model revised the same program after invalid camera-angle and orientation
arguments, then reran it. It also used image crops and pixel inspection. The grasp
program used fixed waypoints and model decisions between stages; it was not an
automatically optimized grasp controller.

This is one development seed, not a statistical comparison. A and B chose different
stances and grasp approaches. The unchanged pick checker tests final object lift
(at least 3 cm) and pelvis height (at least 45 cm), not long-term grasp stability.
Flat has no upstream task checker; its displacement comes from recorded physics.

Local episode identifiers:
- A tabletop: ep-20261117-174945
- B flat: coding-20261117-182840-397051
- B tabletop: coding-20261117-183046-268926

Generated logs remain under ignored data/. No HITCH-27B or real robot test was run.
