"""GT-last coordinator; scoring starts only after all three predictions freeze."""
from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

_WORKER = "--worker" in sys.argv
if not _WORKER: os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np

from collaborative_test.SW_0181_actual_descent_qce import score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRED_MODULE = "collaborative_test.SW_0181_actual_descent_qce.prediction_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "score_queue_20261016"
STOP = threading.Event()
SEEDS = score.SEEDS


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


def _state_path(): return RUNTIME / "state.json"


def _write_state(status, **fields):
    state = {"experiment": "SW0181_score_queue", "status": status,
        "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}
    _atomic_json(_state_path(), state)
    return state


def _module_live(owner, pid, seed):
    if type(pid) is not int or pid <= 0: return False
    proc = owner._proc_record(pid)
    if not proc or proc.get("state") == "Z": return False
    argv = list(proc.get("argv") or [])
    if argv.count("-m") != 1: return False
    index = argv.index("-m")
    if index + 1 >= len(argv) or argv[index + 1] != PRED_MODULE or "--worker" in argv:
        return False
    return argv.count("--source-seed") == 1 and argv[argv.index("--source-seed") + 1:][:1] == [str(seed)]


def prediction_terminal_set(owner, states):
    """Return readiness and exact pending seeds; fail closed on stale processes."""
    pending = []
    for seed in SEEDS:
        state = states.get(seed)
        if not isinstance(state, dict):
            pending.append(seed); continue
        if state.get("experiment") != "SW0181_prediction_queue" or state.get("seed") != seed \
                or state.get("arm") != "product_q_ce":
            raise ValueError("SW0181 prediction state identity mismatch")
        if state.get("status") == "complete_training_rejected":
            raise RuntimeError(f"SW0181 seed{seed} is scientifically rejected; three-seed score unavailable")
        if state.get("status") == "complete":
            if state.get("ground_truth_used_for_prediction") is not False \
                    or state.get("validation_data_read") is not False:
                raise ValueError("SW0181 prediction state violates GT-last protocol")
            continue
        if state.get("status") in {"waiting_for_training", "waiting_for_exclusive_gpu",
                "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner"} \
                and _module_live(owner, state.get("supervisor_pid"), seed):
            pending.append(seed); continue
        raise RuntimeError(f"SW0181 prediction seed{seed} is not live or complete")
    return not pending, pending


def run(*, poll_seconds=5):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0181 score queue must hide CUDA")
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path()
    if state_path.exists(): raise FileExistsError("preserving existing SW0181 score queue state")
    _write_state("waiting_for_predictions", validation_data_read=False)
    owner = score.pe.train._module(OWNER_MODULE)
    while not STOP.is_set():
        prediction_queue = score.pe.train._module(PRED_MODULE)
        states = {}
        for seed in SEEDS:
            path = prediction_queue._state_path(seed)
            if path.is_file(): states[seed] = json.loads(path.read_text(encoding="utf-8"))
        ready, pending = prediction_terminal_set(owner, states)
        if ready: break
        _write_state("waiting_for_predictions", pending_seeds=pending,
            prediction_state_sha256={str(seed): prediction_queue._sha(prediction_queue._state_path(seed))
                for seed in states})
        STOP.wait(poll_seconds)
    if STOP.is_set(): raise RuntimeError("SW0181 score queue stopped before predictions froze")
    prepared = score.prepare_score_inputs_cpu()
    state_hashes = {str(seed): prepared["frozen"][seed]["state_sha256"] for seed in SEEDS}
    _write_state("scoring_gt_last", prediction_state_sha256=state_hashes,
        prediction_sha256={str(seed): prepared["frozen"][seed]["result"]["prediction_sha256"]
            for seed in SEEDS}, validation_data_read=False)
    report = score.score_all(prepared=prepared)
    result = _write_state("complete", score_sha256=report["score_sha256"],
        output=str(score.SCORE_PATH), prediction_state_sha256=state_hashes,
        ground_truth_used_for_prediction=False, ground_truth_used_for_scoring=True)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if not args.worker: parser.error("score coordinator is an owned worker")
    run(); return 0


if __name__ == "__main__": raise SystemExit(main())
