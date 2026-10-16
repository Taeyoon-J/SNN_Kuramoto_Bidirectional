"""GT-last coordinator; opens scoring only after six prediction artifacts freeze."""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

_WORKER = "--worker" in sys.argv
if not _WORKER: os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1"); os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

from collaborative_test.SW_0183_shared_prefix_unique_scaling import score

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
PRED_MODULE = "collaborative_test.SW_0183_shared_prefix_unique_scaling.prediction_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "score_queue_20261016"
STOP = threading.Event(); SEEDS = score.SEEDS


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def state_path(): return RUNTIME / "state.json"


def _write_state(status, **fields):
    row = {"experiment": "SW0183_score_queue", "status": status,
        "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}
    _atomic_json(state_path(), row); return row


def _prediction_terminal(owner, queue, seed, state):
    if not isinstance(state, dict): return False
    if (state.get("experiment") != "SW0183_prediction_queue" or state.get("seed") != seed):
        raise ValueError("SW0183 prediction queue state identity mismatch")
    if state.get("status") == "complete_training_rejected":
        raise RuntimeError(f"SW0183 seed{seed} training was scientifically rejected; full score unavailable")
    if state.get("status") == "complete":
        if (state.get("ground_truth_used_for_prediction") is not False
                or state.get("validation_data_read") is not False
                or set(state.get("predictions", {})) != set(score.ARMS)):
            raise ValueError("SW0183 prediction state lacks both GT-free arm archives")
        return True
    active = {"waiting_for_training", "waiting_for_exclusive_gpu", "waiting_for_stable_gpu",
        "running_worker", "yielding_foreign_owner"}
    if state.get("status") in active and queue._module_live(
            owner, state.get("supervisor_pid"), PRED_MODULE, seed, worker=False):
        return False
    raise RuntimeError(f"SW0183 seed{seed} prediction process is not live or complete")


def predictions_ready(owner=None, queue=None, states=None):
    if owner is None:
        import importlib
        owner = importlib.import_module(OWNER_MODULE)
    if queue is None:
        import importlib
        queue = importlib.import_module(PRED_MODULE)
    if states is None:
        states = {}
        for seed in SEEDS:
            path = queue.state_path(seed)
            if path.is_file(): states[seed] = json.loads(path.read_text(encoding="utf-8"))
    pending = [seed for seed in SEEDS if not _prediction_terminal(owner, queue, seed, states.get(seed))]
    return len(pending) == 0, pending


def run(*, poll_seconds=5):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise RuntimeError("SW0183 score queue must hide CUDA")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set()); signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if state_path().exists(): raise FileExistsError("preserving existing SW0183 score queue state")
    _write_state("waiting_for_predictions", validation_data_read=False)
    import importlib
    owner = importlib.import_module(OWNER_MODULE); queue = importlib.import_module(PRED_MODULE)
    while not STOP.is_set():
        ready, pending = predictions_ready(owner, queue)
        if ready: break
        _write_state("waiting_for_predictions", pending_seeds=pending, validation_data_read=False)
        STOP.wait(poll_seconds)
    if STOP.is_set(): raise RuntimeError("SW0183 score queue stopped before prediction freeze")
    prepared = score.prepare_score_inputs_cpu()
    frozen_states = {str(seed): prepared["frozen"][seed]["state_sha256"] for seed in SEEDS}
    frozen_predictions = {str(seed): {arm: {
        "prediction_sha256": prepared["frozen"][seed]["results"][arm]["prediction_sha256"],
        "protocol_sha256": prepared["frozen"][seed]["results"][arm]["protocol_sha256"]}
        for arm in score.ARMS} for seed in SEEDS}
    _write_state("scoring_gt_last", prediction_state_sha256=frozen_states,
        predictions=frozen_predictions, validation_data_read=False)
    report = score.score_all(prepared=prepared)
    return _write_state("complete", score_sha256=report["score_sha256"], output=str(score.SCORE_PATH),
        prediction_state_sha256=frozen_states, predictions=frozen_predictions,
        ground_truth_used_for_prediction=False, ground_truth_used_for_scoring=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if not args.worker: parser.error("score coordinator runs as an owned worker")
    result = run(); print(json.dumps({"status": result["status"], "score_sha256": result["score_sha256"]}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
