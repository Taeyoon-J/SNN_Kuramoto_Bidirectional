"""Recovery-scoped prediction dispatch after SW0163 training-arm recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
from collaborative_test.SW_0163_relation_context_assignment import recovery_training_queue as recovery
from collaborative_test.SW_0163_relation_context_assignment import prediction_queue as prediction
from collaborative_test.SW_0163_relation_context_assignment import train
if _visible is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible

RUNTIME = HERE / "results_archive" / "recovery_prediction_queue_20261014"
STATE = RUNTIME / "state.json"
MODULE = "collaborative_test.SW_0163_relation_context_assignment.recovery_prediction_queue"
BASE_PREDICTION_IMPLEMENTATION = prediction.implementation_fingerprint


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    result = dict(BASE_PREDICTION_IMPLEMENTATION())
    result.update({"recovery_prediction_queue.py": sha256_file(HERE / "recovery_prediction_queue.py"),
            "recovery_training_queue.py": sha256_file(HERE / "recovery_training_queue.py"),
            "recovery_packet_sha256": recovery.EXPECTED_FAILURE_RECEIPT_SHA256,
            "recovery_protocol.json": sha256_file(HERE / "protocol.json")})
    return result


def training_dependency(*, state_path=recovery.RUNTIME / "state.json", process_live=recovery._pid_live):
    path = Path(state_path)
    if not path.is_file():
        raise ValueError("SW0163 recovery training queue state is missing")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") == "complete" and state.get("recovery_status") == "complete_recovered":
        recovery.verify_recovery_fingerprint(state.get("recovery_fingerprint"))
        original_fp = prediction.training_queue.implementation_fingerprint()
        if state.get("queue_fingerprint") != original_fp:
            raise ValueError("SW0163 recovered training state is not bound to the sealed base queue")
        if set(state.get("results", {})) != set(train.ARMS) or not all(state["results"].values()):
            raise ValueError("SW0163 recovery does not attest all four arms")
        if set(state.get("final_arms", {})) != set(train.ARMS):
            raise ValueError("SW0163 recovery final arm inventory is incomplete")
        return {"state": state, "state_sha256": sha256_file(path),
                "recovery_fingerprint": state["recovery_fingerprint"]}
    status = state.get("status")
    if status in ("failed_recovery", "failed_training", "failed_preflight_dependency"):
        raise ValueError(f"SW0163 recovery training is terminal: {status}")
    if not process_live(state.get("supervisor_pid")):
        raise ValueError("SW0163 recovery training is pending without its registered live supervisor")
    return None


def run():
    # The original predictor implementation is reused unchanged, in a new runtime.
    # Its training-state observation is redirected only within this recovery process.
    prediction.training_queue.STATE = recovery.RUNTIME / "state.json"
    original_fingerprint = prediction.implementation_fingerprint
    prediction.implementation_fingerprint = implementation_fingerprint
    try:
        queue = prediction.PredictionQueue(runtime=RUNTIME,
                                           dependency=lambda: training_dependency())
        original_update = queue._update
        def recovery_update(status, **fields):
            current = json.loads((recovery.RUNTIME / "state.json").read_text(encoding="utf-8"))
            recovery_fp = current.get("recovery_fingerprint")
            if recovery_fp is not None:
                fields.setdefault("training_recovery_fingerprint", recovery_fp)
            return original_update(status, **fields)
        queue._update = recovery_update
        return queue.run()
    finally:
        prediction.implementation_fingerprint = original_fingerprint


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
