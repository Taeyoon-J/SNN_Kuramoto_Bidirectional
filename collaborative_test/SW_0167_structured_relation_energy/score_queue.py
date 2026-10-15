"""CPU GT-last score coordinator after validated SW0167 predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
try:
    import fcntl
except ImportError:
    fcntl = None

_MISSING = object()
_VISIBLE = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _VISIBLE is _MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _VISIBLE

from collaborative_test.SW_0167_structured_relation_energy import predict_evaluate as evaluation
from collaborative_test.SW_0167_structured_relation_energy import prediction_queue

RUNTIME = HERE / "results_archive" / "score_queue_20261015"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "score.log"
POLL_SECONDS = 5.0


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    guard = Path(owner.OWNER_GUARD)
    if not guard.is_file() or sha256_file(guard) != owner.OWNER_GUARD_SHA256:
        raise ValueError("SW0148 owner guard changed")
    paths = [HERE / name for name in ("score_queue.py", "prediction_queue.py",
             "predict_evaluate.py", "train.py", "training_queue.py", "DESIGN_KO.md")]
    paths.extend([Path(owner.__file__), guard])
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0167 score queue dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def prediction_dependency(*, state_path=prediction_queue.STATE):
    path = Path(state_path)
    if not path.is_file():
        raise ValueError("SW0167 prediction queue state missing")
    state = json.loads(path.read_text(encoding="utf-8"))
    status = state.get("status")
    if status == "complete":
        if state.get("queue_fingerprint") != prediction_queue.implementation_fingerprint():
            raise ValueError("SW0167 prediction queue fingerprint changed")
        expected = set(evaluation.ARMS)
        if set(state.get("prediction_hashes", {})) != expected:
            raise ValueError("SW0167 prediction terminal state has wrong arm set")
        return {"state": state, "state_sha256": sha256_file(path)}
    if status == "failed":
        raise ValueError(f"SW0167 prediction prerequisite failed: {state.get('error')}")
    if not prediction_queue._proc_live(
            state.get("supervisor_pid"),
            module="collaborative_test.SW_0167_structured_relation_energy.prediction_queue"):
        raise ValueError("SW0167 prediction is nonterminal but its registered supervisor is not live")
    return None


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


class ScoreQueue:
    def __init__(self, *, runtime=RUNTIME, dependency=prediction_dependency,
                 poll_seconds=POLL_SECONDS):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0167 score supervisor requires Linux flock")
        self.runtime = Path(runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime / "state.json"
        self.lock_path = self.runtime / "queue.lock"
        self.log_path = self.runtime / "score.log"
        self.lock = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("another SW0167 score supervisor owns this queue")
        if self.state_path.exists() or self.log_path.exists():
            self._unlock()
            raise FileExistsError("preserving existing SW0167 score attempt")
        self.dependency = dependency
        self.poll_seconds = float(poll_seconds)
        self.stop = threading.Event()
        self.fp = implementation_fingerprint()
        self._update("waiting_for_predictions")

    def _update(self, status, **fields):
        _atomic(self.state_path, {"experiment": "SW0167_score_queue", "status": status,
            "supervisor_pid": os.getpid(), "queue_fingerprint": self.fp,
            "updated": time.time(), **fields})

    def _unlock(self):
        try: fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
        finally: self.lock.close()

    def run(self):
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("SW0167 score coordinator must hide CUDA")
            while not self.stop.is_set():
                dep = self.dependency()
                if dep is not None:
                    break
                self._update("waiting_for_predictions")
                self.stop.wait(self.poll_seconds)
            else:
                return 2
            # Check cheap producer publication first, avoiding repeated deep
            # hashing while the prediction owner is still writing artifacts.
            if not all((evaluation._out(arm) / "protocol.json").is_file()
                       and (evaluation._out(arm) / "predictions.npz").is_file()
                       for arm in evaluation.ARMS):
                raise ValueError("prediction queue completed without all head outputs")
            prepared = evaluation.prepare_prediction_inputs_cpu()
            preds = {arm: evaluation.validate_prediction(arm, prepared=prepared)
                     for arm in evaluation.ARMS}
            actual = {arm: {"prediction_sha256": row["prediction_sha256"],
                            "protocol_sha256": row["protocol_sha256"]}
                      for arm, row in preds.items()}
            if dep["state"].get("prediction_hashes") != actual:
                raise ValueError("queue prediction hashes differ from validated immutable outputs")
            if self.log_path.exists():
                raise FileExistsError("preserving existing SW0167 score log")
            self._update("scoring_gt_last", prediction_state_sha256=dep["state_sha256"],
                         prediction_hashes=actual)
            report = evaluation.score_all(prepared=prepared)
            valid = evaluation.validate_score(prepared=prepared)
            score_sha = sha256_file(evaluation.SCORE_PATH)
            self._update("complete", finished=time.time(), score_sha256=score_sha,
                         score_status=valid["status"], prediction_state_sha256=dep["state_sha256"],
                         seed_expansion_gate=report["seed1_expansion_gate"])
            return 0
        except BaseException as exc:
            self._update("failed", error=repr(exc), finished=time.time(), artifacts_preserved=True)
            return 1
        finally:
            self.stop.set()
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"runtime": str(RUNTIME), "score_path": str(evaluation.SCORE_PATH),
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")},
                         allow_nan=False))
        return 0
    return ScoreQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
