"""CPU-only GT-last scorer that waits for both frozen SW0175 predictions."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from collaborative_test.SW_0175_critical_component_binding import predict_evaluate_v2 as evaluate, train_v2 as train

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "results_archive" / "score_queue_v3_20261015"
PREDICTION_STATE_ROOT = HERE / "results_archive" / "prediction_queue_v3_20261015"
_stop = threading.Event()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if path.exists(): raise FileExistsError(path)
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _proc_live(owner, pid, arm):
    if type(pid) is not int or pid <= 0: return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z": return False
    argv = record.get("argv", [])
    if "-m" not in argv: return False
    pos = argv.index("-m")
    return (pos + 1 < len(argv)
            and argv[pos + 1] == "collaborative_test.SW_0175_critical_component_binding.prediction_queue_v3"
            and argv.count("--arm") == 1
            and argv.index("--arm") + 1 < len(argv)
            and argv[argv.index("--arm") + 1] == arm)


def _prediction_complete(arm, owner):
    path = PREDICTION_STATE_ROOT / f"{arm}_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"SW0175 prediction queue state missing for {arm}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != "SW0175_prediction_queue_v3" or state.get("arm") != arm:
        raise ValueError("SW0175 prediction terminal state identity mismatch")
    if state.get("status") == "complete":
        out = Path(state.get("output", ""))
        if (not out.is_dir() or _sha(out / "predictions.npz") != state.get("prediction_sha256")
                or _sha(out / "protocol.json") != state.get("protocol_sha256")):
            raise ValueError("SW0175 completed prediction state hashes do not match disk")
        return state
    if state.get("status") in {"waiting_for_training", "waiting_for_stable_gpu",
            "waiting_for_exclusive_gpu", "running_prediction", "yielding_foreign_gpu"}:
        if _proc_live(owner, state.get("supervisor_pid"), arm): return False
    raise RuntimeError(f"SW0175 prediction dependency for {arm} failed/orphaned: {state.get('status')}")


def run(*, timeout_seconds=8 * 3600):
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0175 score coordinator must remain CUDA-hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = RUNTIME / "state.json"; lock_path = RUNTIME / "queue.lock"
    if state_path.exists() or (RUNTIME / "score.log").exists():
        raise FileExistsError("preserving existing SW0175 score queue artifacts")
    lock = lock_path.open("a+"); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        import importlib
        owner = importlib.import_module("collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4")
        # Owner helper import must not alter parent visibility.
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("owner import changed score-process CUDA visibility")
        deadline = time.monotonic() + timeout_seconds
        state = {"experiment": "SW0175_score_queue_v3", "status": "waiting_for_both_predictions",
                 "supervisor_pid": os.getpid(), "updated": time.time()}
        _atomic_json(state_path, state)
        while not _stop.is_set():
            if time.monotonic() > deadline: raise TimeoutError("SW0175 score wait timed out")
            done = {arm: _prediction_complete(arm, owner) for arm in train.ARMS}
            if all(done.values()): break
            state.update({"updated": time.time(), "prediction_status": {
                arm: ("complete" if done[arm] else "waiting") for arm in train.ARMS}})
            # Atomic replace for mutable supervisor state.
            tmp = state_path.with_name(state_path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
            with tmp.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(state, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, state_path)
            time.sleep(5)
        else: raise RuntimeError("SW0175 score queue stopped before predictions completed")
        # Only now perform deep artifact validation and GT-last scoring.
        prepared = evaluate.prepare_prediction_inputs_cpu()
        validated = {arm: evaluate.validate_prediction(arm, prepared=prepared) for arm in train.ARMS}
        for arm in train.ARMS:
            if (validated[arm]["prediction_sha256"] != done[arm]["prediction_sha256"]
                    or validated[arm]["protocol_sha256"] != done[arm]["protocol_sha256"]):
                raise ValueError("SW0175 prediction changed between terminal wait and scoring")
        update = {"experiment": "SW0175_score_queue_v3", "status": "scoring_gt_last",
                  "supervisor_pid": os.getpid(), "updated": time.time(),
                  "prediction_sha256": {arm: validated[arm]["prediction_sha256"] for arm in train.ARMS}}
        tmp = state_path.with_name(state_path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(update, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, state_path)
        report = evaluate.score_all(prepared=prepared)
        score_sha = _sha(evaluate.SCORE_PATH)
        final = {**update, "status": "complete", "score_sha256": score_sha,
                 "score_path": str(evaluate.SCORE_PATH),
                 "implementation_fingerprint": evaluate.implementation_fingerprint(),
                 "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
                 "gate": report["gate"], "finished": time.time()}
        tmp = state_path.with_name(state_path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(final, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, state_path)
        return final
    except BaseException as exc:
        current = json.loads((RUNTIME / "state.json").read_text(encoding="utf-8")) if (RUNTIME / "state.json").exists() else {}
        current.update({"status": "failed", "error": repr(exc), "finished": time.time()})
        tmp = (RUNTIME / "state.json").with_name(f"state.json.tmp-{os.getpid()}-{time.time_ns()}")
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(current, f, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, RUNTIME / "state.json")
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker: raise SystemExit("SW0175 score queue has no GPU worker mode")
    run()


if __name__ == "__main__": main()
