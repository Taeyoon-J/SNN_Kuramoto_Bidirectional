"""CPU-only GT-last SW0176 scorer, gated by the frozen fixed320 prediction state."""
from __future__ import annotations

import hashlib
import importlib
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

import numpy as np  # Keep NumPy/OpenMP before legacy torch imports.

from collaborative_test.SW_0176_local_phase_credit import predict_evaluate as evaluate
from collaborative_test.SW_0176_local_phase_credit import train

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "results_archive" / "score_queue_v1_20261016"
PREDICTION_STATE = HERE / "results_archive" / "prediction_queue_v1_20261016" / "state.json"
PREDICTION_MODULE = "collaborative_test.SW_0176_local_phase_credit.prediction_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
STOP = threading.Event()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value, *, replace=False):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_state(status, **fields):
    _atomic_json(RUNTIME / "state.json", {"experiment": "SW0176_score_queue_v1",
        "status": status, "supervisor_pid": os.getpid(), "updated_unix": time.time(),
        **fields}, replace=True)


def _module_live(owner, pid):
    if type(pid) is not int or pid <= 0:
        return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z":
        return False
    argv = record.get("argv", [])
    if argv.count("-m") != 1:
        return False
    index = argv.index("-m")
    return index + 1 < len(argv) and argv[index + 1] == PREDICTION_MODULE


def _prediction_terminal(owner):
    if not PREDICTION_STATE.is_file():
        return None
    state = json.loads(PREDICTION_STATE.read_text(encoding="utf-8"))
    if state.get("experiment") != "SW0176_prediction_queue_v1":
        raise ValueError("SW0176 prediction terminal state schema/identity mismatch")
    if state.get("status") == "complete":
        directory = Path(state.get("output", ""))
        return _validate_prediction_state(state, owner,
            output_is_dir=directory.is_dir(),
            prediction_sha256=_sha(directory / "predictions.npz") if directory.is_dir() else None,
            protocol_sha256=_sha(directory / "protocol.json") if directory.is_dir() else None)
    return _validate_prediction_state(state, owner)


def _validate_prediction_state(state, owner, *, output_is_dir=None,
                               prediction_sha256=None, protocol_sha256=None):
    """Consume the producer's terminal/live-state contract without opening GT."""
    if state.get("experiment") != "SW0176_prediction_queue_v1":
        raise ValueError("SW0176 prediction terminal state schema/identity mismatch")
    if state.get("status") == "complete":
        if (output_is_dir is not True or not _is_sha(state.get("prediction_sha256"))
                or not _is_sha(state.get("protocol_sha256"))
                or state.get("prediction_sha256") != prediction_sha256
                or state.get("protocol_sha256") != protocol_sha256
                or state.get("ground_truth_used_for_prediction") is not False):
            raise ValueError("SW0176 prediction queue state artifact hashes differ")
        return state
    if state.get("status") in {"waiting_for_training", "waiting_for_exclusive_gpu",
            "waiting_for_stable_gpu", "running_prediction", "yielding_foreign_gpu"}:
        if _module_live(owner, state.get("supervisor_pid")):
            return False
    raise RuntimeError(f"SW0176 prediction dependency failed/orphaned: {state.get('status')}")


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _score_gt_last(prediction, *, evaluator, arm, mark_scoring):
    """Validate the frozen prediction first, then make the scoring transition explicit."""
    prepared = evaluator.prepare_prediction_inputs_cpu()
    validated = evaluator.validate_prediction(arm, prepared=prepared)
    if (validated["prediction_sha256"] != prediction["prediction_sha256"]
            or validated["protocol_sha256"] != prediction["protocol_sha256"]):
        raise ValueError("SW0176 prediction changed after terminal queue receipt")
    mark_scoring(validated)
    report = evaluator.score_all(prepared=prepared)
    return report, validated


def run(*, timeout_seconds=8 * 3600):
    import fcntl
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0176 score coordinator must remain CUDA-hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = RUNTIME / "state.json"
    if state_path.exists() or (RUNTIME / "score.log").exists():
        raise FileExistsError("preserving existing SW0176 score queue state/log")
    lock = (RUNTIME / "queue.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        owner = importlib.import_module(OWNER_MODULE)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
            raise RuntimeError("SW0176 owner import changed CPU scorer CUDA visibility")
        _write_state("waiting_for_prediction")
        deadline = time.monotonic() + timeout_seconds
        prediction = None
        while not STOP.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("SW0176 score queue timed out waiting for frozen predictions")
            prediction = _prediction_terminal(owner)
            if prediction is not False and prediction is not None:
                break
            _write_state("waiting_for_prediction", prediction_status="running")
            time.sleep(5)
        if prediction is None or prediction is False:
            raise RuntimeError("SW0176 score queue stopped before prediction completion")
        # Deeply validate the complete prediction and references before any GT read.
        def mark_scoring(validated):
            _write_state("scoring_gt_last", prediction_sha256=validated["prediction_sha256"],
                protocol_sha256=validated["protocol_sha256"], optimizer_updates=0,
                ground_truth_used_for_prediction=False)
        report, validated = _score_gt_last(prediction, evaluator=evaluate,
            arm=train.ARM, mark_scoring=mark_scoring)
        score_sha = _sha(evaluate.SCORE_PATH)
        # score_all writes then invokes validate_score, which independently
        # reloads the frozen prediction and recomputes every metric/bootstrap.
        final = {"experiment": "SW0176_score_queue_v1", "status": "complete",
            "supervisor_pid": os.getpid(), "prediction_sha256": validated["prediction_sha256"],
            "protocol_sha256": validated["protocol_sha256"], "score_sha256": score_sha,
            "score_path": str(evaluate.SCORE_PATH),
            "implementation_fingerprint": evaluate.implementation_fingerprint(),
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
            "gate": report["gate"], "finished_unix": time.time()}
        _write_state("complete", **{key: value for key, value in final.items()
                                     if key not in {"experiment", "status", "supervisor_pid"}})
        return final
    except BaseException as exc:
        state_path = RUNTIME / "state.json"
        current = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        current.update({"experiment": "SW0176_score_queue_v1", "status": "failed",
                       "error": repr(exc), "finished_unix": time.time()})
        _atomic_json(state_path, current, replace=True)
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        parser.error("SW0176 score queue does not have a GPU worker mode")
    result = run()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
