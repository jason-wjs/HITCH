"""Explicit full-stack/paused-clock check without API requests.

Run with scripts/run, WBC_HAND=dex3 and WBC_BENCHMARK_PYTHON set.
Published frames lag the actual paused physics by one step; final native
scoring and recorded physics remain authoritative.
"""
import json
import time
from pathlib import Path

from wbc.coding.broker import RobotBroker
from wbc.session import client
from wbc_link_server.sensors.frame_ring import FrameRing

root = Path("data/locomanilab/probe/pause_clock").resolve()
root.mkdir(parents=True, exist_ok=True)
connected = client.connect(
    "sim://locomanilab?task=IThorKitchenApproachTableAndLiftApple_scene13&seed=0&paused=1",
    root=root,
)
session = connected["session"]
runtime = root / "sessions" / connected["session_id"] / "runtime"
sim = runtime / "sim"
ring = FrameRing(sim / "camera_frames.npy")
record = {"session_id": connected["session_id"], "kind": "native_pause_clock_no_model"}


def sample():
    state = ring.latest()
    return {
        "wall_time": time.time(),
        "frame": int(state["frame"]),
        "sim_time": float(state["time"]),
        "native_steps_from_time": round(float(state["time"]) * 50),
        "persisted_score": json.loads((sim / "benchmark_result.json").read_text()),
    }


try:
    broker = RobotBroker(session, root / "workspace", root / "record")
    observation = broker.dispatch({"op": "observe", "args": {}})
    record["initial"] = observation
    assert observation["images"]
    time.sleep(1)
    record["before_idle"] = sample()
    time.sleep(3)
    record["after_idle"] = sample()
    for key in ("frame", "sim_time", "persisted_score"):
        assert record["before_idle"][key] == record["after_idle"][key], key
    result = broker.dispatch({"op": "execute", "args": {
        "observation_id": observation["observation_id"],
        "request": {"torso": {"pitch_deg": 5, "yaw_deg": 0}, "wait_s": 1},
    }})
    record["action"] = result
    waits = [call["result"] for call in result["calls"] if call["op"] == "wait"]
    assert waits and waits[-1]["state"] == "done"
    assert result["observation"]["body"]["posture"] == "standing"
    time.sleep(1)
    record["before_post_action_idle"] = sample()
    time.sleep(3)
    record["after_post_action_idle"] = sample()
    for key in ("frame", "sim_time", "persisted_score"):
        assert record["before_post_action_idle"][key] == record["after_post_action_idle"][key], key
    assert record["after_post_action_idle"]["sim_time"] > record["after_idle"]["sim_time"]
    record["passed"] = True
finally:
    client.call(session, "close")
    record["native_final"] = json.loads((sim / "benchmark_result.json").read_text())
    record["process_exit"] = json.loads((runtime / "exit.json").read_text())
    (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")

print(json.dumps({key: record[key] for key in (
    "session_id", "passed", "before_idle", "after_idle",
    "before_post_action_idle", "after_post_action_idle", "process_exit",
)}))
