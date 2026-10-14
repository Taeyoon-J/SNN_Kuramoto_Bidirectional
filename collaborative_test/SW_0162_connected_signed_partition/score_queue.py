"""CPU-only GT-last score coordinator for both frozen SW0162 prediction artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0162_connected_signed_partition import predict_evaluate as evaluation
from collaborative_test.SW_0162_connected_signed_partition import prediction_queue

try:
    import fcntl
except ImportError:
    fcntl = None

RUNTIME = HERE / "results_archive" / "score_queue_20261014"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "score.log"
POLL_SECONDS = 5.0
ACTIVE_PREDICTION_STATES = {"waiting_for_source_preflight", "waiting_stable_gpu",
    "waiting_for_exclusive_gpu", "waiting_for_prediction_owner", "running"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "score_queue.py", HERE / "prediction_queue.py", HERE / "predict_evaluate.py",
        HERE / "preflight.py", HERE / "preflight_queue.py", HERE / "connected_signed.py",
        HERE / "protocol.json", HERE / "DESIGN_KO.md")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("SW0162 score queue dependency missing")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def _atomic_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2, allow_nan=False); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def _prediction_owner_live(state_path=prediction_queue.STATE):
    if not Path(state_path).is_file():
        return False
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if (state.get("experiment") != "SW0162_connected_signed_partition"
            or state.get("status") not in ACTIVE_PREDICTION_STATES):
        return False
    pid = state.get("supervisor_pid")
    if type(pid) is int and pid > 0:
        try:
            row = prediction_queue.owner._proc_record(pid)
            if (row and row.get("state") != "Z"
                    and "collaborative_test.SW_0162_connected_signed_partition.prediction_queue" in row.get("argv", [])):
                return True
        except (OSError, ValueError, RuntimeError):
            pass
    pid = state.get("child_pid")
    if type(pid) is int and pid > 0:
        try:
            row = prediction_queue.owner._proc_record(pid)
            return bool(row and row.get("state") != "Z"
                and "collaborative_test.SW_0162_connected_signed_partition.predict_evaluate" in row.get("argv", []))
        except (OSError, ValueError, RuntimeError):
            return False
    return False


class ScoreQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if prediction_queue.fcntl is None:
            raise RuntimeError("SW0162 score coordinator requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0162 score coordinator owns the queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.score_fp = evaluation.implementation_fingerprint()
        self.existing_complete = False
        try:
            if evaluation.SCORE_PATH.is_file():
                evaluation.validate_score(evaluation.SCORE_PATH)
                self.existing_complete = True
            elif STATE.exists() or LOG.exists():
                raise FileExistsError("preserving existing SW0162 score queue state/log")
            else:
                self._update("waiting_for_both_predictions", supervisor_pid=os.getpid(),
                    queue_fingerprint=self.queue_fp, score_fingerprint=self.score_fp,
                    output=str(evaluation.SCORE_PATH), ground_truth_used_for_prediction=False,
                    ground_truth_used_for_scoring=True)
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try: fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally: self.lock_stream.close()

    def _update(self, status, **fields):
        with self.state_lock:
            current = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
            current.update(experiment="SW0162_connected_signed_partition", status=status,
                updated=time.time(), **fields)
            _atomic_json(STATE, current)

    def run(self):
        try:
            if self.existing_complete:
                return 0
            while not self.stop_event.is_set():
                state = json.loads(prediction_queue.STATE.read_text(encoding="utf-8")) if prediction_queue.STATE.is_file() else None
                dirs = [evaluation._output_dir(arm) for arm in evaluation.ARMS]
                present = [all((d / name).is_file() for name in ("predictions.npz", "protocol.json")) for d in dirs]
                if state is None:
                    raise RuntimeError("SW0162 prediction owner has not been registered")
                if state.get("status") != "complete":
                    if state.get("status") in ACTIVE_PREDICTION_STATES and _prediction_owner_live():
                        self._update("waiting_for_prediction_owner", prediction_state=state.get("status"))
                        self.stop_event.wait(POLL_SECONDS); continue
                    raise RuntimeError(f"SW0162 prediction queue is terminal/nonlive: {state.get('status')}")
                if not all(present):
                    raise RuntimeError("SW0162 prediction queue completed without both complete artifacts")
                if state.get("queue_fingerprint") != prediction_queue.implementation_fingerprint():
                    raise RuntimeError("SW0162 prediction queue terminal state has a stale code fingerprint")
                state_hashes = state.get("output_hashes")
                if not isinstance(state_hashes, dict) or set(state_hashes) != set(evaluation.ARMS):
                    raise ValueError("SW0162 prediction queue has no per-arm output receipts")
                for arm in evaluation.ARMS:
                    protocol = json.loads((evaluation._output_dir(arm) / "protocol.json").read_text(encoding="utf-8"))
                    if state_hashes[arm] != protocol.get("prediction_sha256"):
                        raise ValueError("SW0162 queue output receipt differs from prediction protocol")
                if (implementation_fingerprint() != self.queue_fp
                        or evaluation.implementation_fingerprint() != self.score_fp):
                    raise RuntimeError("SW0162 scoring implementation changed after registration")
                prepared = evaluation.prepare_prediction_cpu()
                for arm in evaluation.ARMS:
                    evaluation.validate_prediction(arm, prepared=prepared)
                self._update("validating_all_predictions_before_gt", prediction_queue_sha256=sha256_file(prediction_queue.STATE))
                report = evaluation.score_all(prepared=prepared)
                if (report.get("status") != "complete"
                        or report.get("ground_truth_opened_after_all_predictions_validated") is not True
                        or report.get("ground_truth_used_for_prediction") is not False
                        or report.get("ground_truth_used_for_scoring") is not True):
                    raise ValueError("SW0162 scoring producer violated GT-last schema")
                checked = evaluation.validate_score(evaluation.SCORE_PATH, prepared=prepared)
                self._update("complete", score_sha256=checked["score_sha256"],
                    seed1_expansion_gate=checked["seed1_expansion_gate"], finished=time.time(),
                    ground_truth_used_for_prediction=False, ground_truth_used_for_scoring=True)
                return 0
            self._update("cancelled")
            return 1
        except BaseException as exc:
            self._update("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "wait_for": str(prediction_queue.STATE),
            "score_path": str(evaluation.SCORE_PATH), "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_scoring": True, "queue_fingerprint": implementation_fingerprint(),
            "score_fingerprint": evaluation.implementation_fingerprint()}, allow_nan=False), flush=True)
        return 0
    return ScoreQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
