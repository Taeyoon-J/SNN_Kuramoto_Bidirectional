"""Recovery-scoped GT-last score dispatch after SW0163 recovered predictions."""
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
from collaborative_test.SW_0163_relation_context_assignment import recovery_prediction_queue as recovery_prediction
from collaborative_test.SW_0163_relation_context_assignment import recovery_training_queue as recovery_training
from collaborative_test.SW_0163_relation_context_assignment import score_queue as score
from collaborative_test.SW_0163_relation_context_assignment import prediction_queue as prediction
from collaborative_test.SW_0163_relation_context_assignment import predict_evaluate as evaluation
if _visible is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible
BASE_SCORE_IMPLEMENTATION = score.implementation_fingerprint

RUNTIME = HERE / "results_archive" / "recovery_score_queue_20261014"
STATE = RUNTIME / "state.json"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    result = dict(BASE_SCORE_IMPLEMENTATION())
    result.update({"recovery_score_queue.py": sha256_file(HERE / "recovery_score_queue.py"),
            "recovery_prediction_queue.py": sha256_file(HERE / "recovery_prediction_queue.py"),
            "recovery_training_queue.py": sha256_file(HERE / "recovery_training_queue.py"),
            "recovery_packet_sha256": recovery_training.EXPECTED_FAILURE_RECEIPT_SHA256,
            "recovery_protocol.json": sha256_file(HERE / "protocol.json")})
    return result


def prediction_dependency(*, state_path=recovery_prediction.RUNTIME / "state.json",
                         process_live=recovery_training._pid_live):
    path = Path(state_path)
    if not path.is_file():
        raise ValueError("SW0163 recovery prediction queue state is missing")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") == "complete":
        if state.get("queue_fingerprint") != recovery_prediction.implementation_fingerprint():
            raise ValueError("SW0163 recovered prediction code fingerprint changed")
        recovery_training.verify_recovery_fingerprint(state.get("training_recovery_fingerprint"))
        expected = set((*evaluation.ARMS, "source_T_control"))
        if set(state.get("prediction_hashes", {})) != expected:
            raise ValueError("SW0163 recovered prediction arm set is incomplete")
        for arm, hashes in state["prediction_hashes"].items():
            out = evaluation._out(arm)
            if sha256_file(out / "predictions.npz") != hashes.get("prediction_sha256"):
                raise ValueError(f"SW0163 recovered prediction archive changed: {arm}")
            if sha256_file(out / "protocol.json") != hashes.get("protocol_sha256"):
                raise ValueError(f"SW0163 recovered prediction protocol changed: {arm}")
        return {"state": state, "state_sha256": sha256_file(path)}
    if state.get("status") in ("failed", "cancelled"):
        raise ValueError(f"SW0163 recovered prediction is terminal: {state['status']}")
    if not process_live(state.get("supervisor_pid")):
        raise ValueError("SW0163 recovered predictions are pending without their registered live owner")
    return None


def run():
    prediction.STATE = recovery_prediction.RUNTIME / "state.json"
    old_fp = score.implementation_fingerprint
    score.implementation_fingerprint = implementation_fingerprint
    try:
        queue = score.ScoreQueue(runtime=RUNTIME,
                                 dependency=lambda: prediction_dependency())
        original_update = queue._update
        def recovery_update(status, **fields):
            training_state = json.loads((recovery_training.RUNTIME / "state.json").read_text(encoding="utf-8"))
            prediction_state = json.loads((recovery_prediction.RUNTIME / "state.json").read_text(encoding="utf-8"))
            if training_state.get("recovery_fingerprint") is not None:
                fields.setdefault("training_recovery_fingerprint", training_state["recovery_fingerprint"])
            if prediction_state.get("queue_fingerprint") is not None:
                fields.setdefault("prediction_recovery_fingerprint", prediction_state["queue_fingerprint"])
            return original_update(status, **fields)
        queue._update = recovery_update
        queue._update("waiting_for_predictions", recovery_handoff=True)
        return queue.run()
    finally:
        score.implementation_fingerprint = old_fp


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
