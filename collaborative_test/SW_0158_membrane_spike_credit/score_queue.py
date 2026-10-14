"""CPU-only coordinator for SW0158's GT-last paired score."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

try:
    import fcntl
except ImportError:
    fcntl = None

if __name__ == "__main__":
    # CLI parent must become CPU-only before importing Torch through evaluation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (ROOT, ROOT / "collaborative_test"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0158_membrane_spike_credit import predict_evaluate as evaluation
from collaborative_test.SW_0158_membrane_spike_credit import train

RUNTIME = HERE / "results_archive" / "score_queue_20261014"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "score.log"
POLL_SECONDS = 5.0
PREDICTION_QUEUE_ROOT = HERE / "results_archive" / "prediction_queues_20261014"


def _process_has_args(pid, module_or_script, arm):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        state = stat[stat.rfind(")") + 2:].split()[0]
        argv = [part.decode("utf-8", errors="replace") for part in
                Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False
    return state != "Z" and module_or_script in argv and "--arm" in argv and arm in argv


def _prediction_queue_is_live(arm):
    path = PREDICTION_QUEUE_ROOT / arm / "state.json"
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    active = {"waiting_for_training_artifact", "waiting_stable_gpu",
              "waiting_for_exclusive_gpu", "launching", "running"}
    if (state.get("experiment") != train.EXPERIMENT or state.get("arm") != arm
            or state.get("status") not in active):
        return False
    supervisor_live = _process_has_args(state.get("supervisor_pid"),
        "collaborative_test.SW_0158_membrane_spike_credit.prediction_queue", arm)
    child_live = _process_has_args(state.get("child_pid"),
        "collaborative_test.SW_0158_membrane_spike_credit.predict_evaluate", arm)
    return supervisor_live or child_live


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = (HERE / "score_queue.py", HERE / "predict_evaluate.py", HERE / "train.py",
             HERE / "spike_credit.py", HERE / "protocol.json")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


class ScoreQueue:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0158 score coordinator requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0158 score coordinator owns the queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.fingerprint = implementation_fingerprint()
        self.output = evaluation.SCORE_PATH
        try:
            if self.output.exists():
                evaluation.validate_score(self.output)
                self.complete = True
            else:
                self.complete = False
                if STATE.exists() or LOG.exists():
                    raise FileExistsError("preserve existing SW0158 score queue state/log")
                _atomic_json(STATE, {"experiment": train.EXPERIMENT,
                    "status": "waiting_for_both_predictions", "supervisor_pid": os.getpid(),
                    "queue_fingerprint": self.fingerprint,
                    "score_fingerprint": evaluation.implementation_fingerprint(),
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
            if self.complete:
                return 0
            while not self.stop_event.is_set():
                self._update("waiting_for_both_predictions")
                waiting_on = None
                for arm in train.ARMS:
                    folder = evaluation.PREDICTION_ROOT / f"{arm}_seed1"
                    required = (folder / "predictions.npz", folder / "protocol.json")
                    if not all(path.is_file() for path in required):
                        if folder.exists():
                            raise RuntimeError(f"present SW0158 prediction for {arm} is incomplete")
                        if not _prediction_queue_is_live(arm):
                            raise RuntimeError(f"SW0158 prediction dependency for {arm} is missing and no live owner remains")
                        waiting_on = arm
                        break
                if waiting_on is not None:
                    self._update("waiting_for_prediction_owner", arm=waiting_on)
                    self.stop_event.wait(POLL_SECONDS)
                    continue
                ready = {}
                for arm in train.ARMS:
                    # Expensive row/RGB/training lineage validation runs only once
                    # both immutable final archives are present.
                    ready[arm] = evaluation.validate_prediction(arm)
                # A present but invalid prediction is terminal; never score around it.
                self._update("both_predictions_validated",
                    prediction_sha256={arm: row["prediction_sha256"] for arm, row in ready.items()},
                    protocol_sha256={arm: row["protocol_sha256"] for arm, row in ready.items()})
                if implementation_fingerprint() != self.fingerprint:
                    raise RuntimeError("SW0158 score dependencies changed while waiting")
                with LOG.open("x", encoding="utf-8") as log:
                    log.write("starting GT-last score after both arm predictions validated\n")
                    log.flush(); os.fsync(log.fileno())
                    report = evaluation.score_all(output=self.output)
                if (report.get("status") != "complete"
                        or report.get("ground_truth_opened_after_all_predictions_validated") is not True
                        or report.get("ground_truth_used_for_prediction") is not False
                        or report.get("ground_truth_used_for_scoring") is not True):
                    raise ValueError("SW0158 scorer returned invalid GT-last contract")
                evaluation.validate_score(self.output)
                self._update("complete", score_sha256=sha256_file(self.output),
                    prediction_sha256={arm: row["prediction_sha256"] for arm, row in ready.items()},
                    finished=time.time(), promotion_gate=report["promotion_gate"])
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
        print(json.dumps({"status": "planned", "dependencies": implementation_fingerprint(),
            "wait_for_predictions": list(train.ARMS), "output": str(evaluation.SCORE_PATH),
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_scoring": True}, allow_nan=False), flush=True)
        return 0
    return ScoreQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
