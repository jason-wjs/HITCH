"""Reset / one-frame check for the first two bedroom tasks; no model or API."""

import json
from pathlib import Path

import numpy as np

from wbc_link_server.benchmarks.native import NativeTask


def main():
    tasks = (
        "IThorBedroomApproachDeskAndPutCanInBasket_scene329",
        "IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket_scene329",
    )
    root = Path("data/locomanilab/probe/native_reset")
    for name in tasks:
        task = NativeTask(name, root / name)
        try:
            before = task.score()
            assert before["steps"] == 0 and not before["success"] and not before["terminal"], before
            task.hold(task.initial[7:])
            assert task.score() == before
            after = task.step(task.initial[7:], np.zeros(29), np.zeros(29))
            assert after["steps"] == 1 and abs(after["sim_time"] - .02) < 1e-12, after
            assert np.isfinite(task.data.qpos).all()
            (root / name / "integration_check.json").write_text(json.dumps({
                "passed": True, "purpose": "reset_hold_one_native_frame",
                "agent_trial": False, "result": after,
            }, indent=2) + "\n")
            print("TASK_RESET_ONE_STEP_OK", name, json.dumps(after), flush=True)
        finally:
            task.close()


if __name__ == "__main__":
    main()
