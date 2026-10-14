"""CPU-only SW0159 paired score coordinator; masks are opened after both predictions validate."""
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

try:
    import fcntl
except ImportError:
    fcntl = None

from collaborative_test.SW_0159_one_shot_event_regime import predict_evaluate as evaluation
from collaborative_test.SW_0159_one_shot_event_regime import train

RUNTIME = HERE / "results_archive" / "score_queue_20261014"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "score.log"
PREDICTION_QUEUE_ROOT = HERE / "results_archive" / "prediction_queues_20261014"
POLL_SECONDS = 5.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "score_queue.py", HERE / "predict_evaluate.py", HERE / "train.py",
             HERE / "preflight.py", HERE / "protocol.json")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0159 score queue dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _process_has_args(pid, module, arm):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        row = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        state = row[row.rfind(")") + 2:].split()[0]
        argv = [value.decode("utf-8", errors="replace") for value in
                Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]
    except (OSError, ValueError):
        return False
    return state != "Z" and module in argv and "--arm" in argv and arm in argv


def _prediction_queue_is_live(arm):
    path = PREDICTION_QUEUE_ROOT / arm / "state.json"
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != train.EXPERIMENT or state.get("arm") != arm:
        return False
    active = {"waiting_for_training_artifact", "waiting_stable_gpu", "waiting_for_exclusive_gpu",
              "running", "launching"}
    if state.get("status") not in active:
        return False
    return (_process_has_args(state.get("supervisor_pid"),
                "collaborative_test.SW_0159_one_shot_event_regime.prediction_queue", arm)
        or _process_has_args(state.get("child_pid"),
                "collaborative_test.SW_0159_one_shot_event_regime.predict_evaluate", arm))


def _atomic_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class ScoreQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0159 CPU score queue requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0159 score supervisor owns the queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.queue_fp = implementation_fingerprint()
        self.score_fp = evaluation.implementation_fingerprint()
        self.output = evaluation.SCORE_PATH
        self.existing_complete = False
        try:
            if self.output.exists():
                evaluation.validate_score(self.output)
                self.existing_complete = True
            elif STATE.exists() or LOG.exists():
                raise FileExistsError("preserve existing SW0159 score state/log")
            else:
                _atomic_json(STATE, {"experiment": train.EXPERIMENT,
                    "status": "waiting_for_both_predictions", "supervisor_pid": os.getpid(),
                    "queue_fingerprint": self.queue_fp, "score_fingerprint": self.score_fp,
                    "ground_truth_used_for_prediction": False})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _update(self, status, **fields):
        with self.state_lock:
            state = json.loads(STATE.read_text(encoding="utf-8"))
            state.update(status=status, updated=time.time(), **fields)
            _atomic_json(STATE, state)

    def run(self):
        try:
            if self.existing_complete:
                return 0
            while not self.stop_event.is_set():
                waiting = None
                for arm in train.ARMS:
                    folder = evaluation.PREDICTION_ROOT / f"{arm}_seed1"
                    required = (folder / "predictions.npz", folder / "protocol.json")
                    if all(path.is_file() for path in required):
                        continue
                    if folder.exists():
                        raise RuntimeError(f"present SW0159 {arm} prediction is incomplete")
                    if not _prediction_queue_is_live(arm):
                        raise RuntimeError(f"SW0159 {arm} prediction is absent and has no live queue owner")
                    waiting = arm
                    break
                if waiting is not None:
                    self._update("waiting_for_prediction_owner", arm=waiting)
                    self.stop_event.wait(POLL_SECONDS)
                    continue
                if (implementation_fingerprint() != self.queue_fp
                        or evaluation.implementation_fingerprint() != self.score_fp):
                    raise RuntimeError("SW0159 score dependencies changed while waiting")
                # score_all independently validates both prediction archives and source/T
                # references before it accesses a single ground-truth mask.
                report = evaluation.score_all(output=self.output)
                if (report.get("status") != "complete"
                        or report.get("ground_truth_opened_after_all_predictions_validated") is not True
                        or report.get("ground_truth_used_for_prediction") is not False
                        or report.get("ground_truth_used_for_scoring") is not True):
                    raise ValueError("SW0159 score producer returned an invalid GT-last report")
                evaluation.validate_score(self.output)
                self._update("complete", score_sha256=sha256_file(self.output),
                    promotion_gate=report.get("promotion_gate"), finished=time.time(),
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
        print(json.dumps({"status": "planned", "wait_for_predictions": list(train.ARMS),
            "output": str(evaluation.SCORE_PATH), "queue_fingerprint": implementation_fingerprint(),
            "score_fingerprint": evaluation.implementation_fingerprint(),
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_scoring": True}, allow_nan=False), flush=True)
        return 0
    return ScoreQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
