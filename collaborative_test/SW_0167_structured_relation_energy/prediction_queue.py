"""Owner-safe fixed320 inference after both SW0167 heads are frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
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
from collaborative_test.SW_0167_structured_relation_energy import train, training_queue

RUNTIME = HERE / "results_archive" / "prediction_queue_20261015"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "worker.log"
MODULE = "collaborative_test.SW_0167_structured_relation_energy.training_queue"
DEVICE = "cuda:0"
MAX_MEMORY_MIB = 512
POLL_SECONDS = 5.0
WATCHDOG_SECONDS = 8 * 60 * 60
_stop = threading.Event()


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
    paths = [HERE / name for name in ("prediction_queue.py", "predict_evaluate.py",
             "train.py", "training_queue.py", "feature_modes.py", "structured_energy.py",
             "DESIGN_KO.md")]
    paths.extend([Path(owner.__file__), guard])
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0167 prediction queue dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def command(gpu):
    return [sys.executable, "-m", "collaborative_test.SW_0167_structured_relation_energy.predict_evaluate",
            "--stage", "predict", "--device", "cuda:0"]


def _proc_live(pid, *, module=MODULE):
    if type(pid) is not int or pid <= 1 or not sys.platform.startswith("linux"):
        return False
    try:
        argv = [item.decode(errors="replace") for item in
                (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0") if item]
        return "-m" in argv and module in argv
    except OSError:
        return False


def training_dependency(*, process_live=_proc_live):
    state_records = {}
    pending = False
    for arm in train.ARMS:
        path = training_queue._state_path(arm)
        if not path.is_file():
            raise ValueError(f"SW0167 {arm} training queue state missing")
        state = json.loads(path.read_text(encoding="utf-8"))
        status = state.get("status")
        if status == "failed":
            raise ValueError(f"SW0167 {arm} training prerequisite failed: {state.get('error')}")
        if status != "complete":
            if not process_live(state.get("supervisor_pid")):
                raise ValueError(f"SW0167 {arm} training is nonterminal but its registered supervisor is not live")
            pending = True
        state_records[arm] = (state, path)
    if pending:
        return None
    states = {}
    for arm, (state, path) in state_records.items():
        if state.get("arm") != arm:
            raise ValueError(f"SW0167 {arm} training completion state has the wrong arm")
        artifact = train.validate_training_artifact(arm)
        manifest_sha = sha256_file(train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json")
        if state.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"SW0167 {arm} training manifest differs from queue completion")
        states[arm] = {"state": state, "state_sha256": sha256_file(path),
                       "manifest_sha256": manifest_sha,
                       "checkpoint_sha256": artifact["checkpoint_sha256"]}
    return {"arms": states,
            "state_sha256": hashlib.sha256(json.dumps(
                {arm: states[arm]["state_sha256"] for arm in train.ARMS},
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


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


class PredictionQueue:
    def __init__(self, *, runtime=RUNTIME, dependency=training_dependency,
                 poll_seconds=POLL_SECONDS):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if fcntl is None:
            raise RuntimeError("SW0167 prediction supervisor requires Linux flock")
        self.runtime = Path(runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime / "state.json"
        self.lock_path = self.runtime / "queue.lock"
        self.log_path = self.runtime / "worker.log"
        self.lock = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("another SW0167 prediction supervisor owns this queue")
        if self.state_path.exists() or self.log_path.exists():
            self._unlock()
            raise FileExistsError("preserving existing SW0167 prediction attempt")
        self.dependency = dependency
        self.poll_seconds = float(poll_seconds)
        self.stop = threading.Event()
        self.child = None
        self.lease = None
        self.fp = implementation_fingerprint()
        self._update("waiting_for_training")

    def _update(self, status, **fields):
        _atomic(self.state_path, {"experiment": "SW0167_prediction_queue",
            "status": status, "supervisor_pid": os.getpid(),
            "queue_fingerprint": self.fp, "updated": time.time(), **fields})

    def _verify(self):
        if implementation_fingerprint() != self.fp:
            raise RuntimeError("SW0167 prediction implementation changed after queue registration")

    def _release(self):
        lease, self.lease = self.lease, None
        if lease is not None:
            try: fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally: lease.close()

    def _unlock(self):
        try: fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
        finally: self.lock.close()

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("SW0167 CPU supervisor must hide CUDA")
            while not self.stop.is_set():
                self._verify()
                dependency = self.dependency()
                if dependency is not None:
                    break
                self._update("waiting_for_training")
                self.stop.wait(self.poll_seconds)
            else:
                return 2
            prepared = evaluation.prepare_prediction_inputs_cpu()
            self._update("waiting_for_idle_gpu", training_state_sha256=dependency["state_sha256"])
            while not self.stop.is_set():
                gpu, gpu_uuid, memory, lease = owner._acquire(self.stop)
                self.lease = lease
                proc = None
                try:
                    stable = owner._stable_idle(gpu, gpu_uuid, self.stop,
                        lambda elapsed, used: self._update("waiting_for_stable_gpu", gpu=gpu,
                            gpu_uuid=gpu_uuid, stable_seconds=elapsed, memory_used_mib=int(used)))
                    if not stable:
                        self._release()
                        continue
                    self._verify()
                    current_dependency = self.dependency()
                    current_inputs = evaluation.prepare_prediction_inputs_cpu()
                    if (current_dependency is None
                            or current_dependency["state_sha256"] != dependency["state_sha256"]
                            or current_inputs["source_checkpoint_sha256"] != prepared["source_checkpoint_sha256"]):
                        raise RuntimeError("SW0167 upstream training/source changed before inference")
                    actual_uuid, used, owners = owner.gpu_guard._gpu_info(gpu)
                    if actual_uuid != gpu_uuid or owners or int(used) > MAX_MEMORY_MIB:
                        self._update("waiting_for_idle_gpu", gpu=gpu, final_gpu_sample={
                            "uuid": actual_uuid, "memory_mib": int(used), "owners": list(owners)})
                        self._release()
                        continue
                    cmd = command(gpu)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
                    with self.log_path.open("x", encoding="utf-8", newline="\n") as log:
                        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lease.fileno(),))
                        self.child = proc
                        self._update("running_prediction", child_pid=proc.pid,
                            child_pgid=proc.pid, child_argv=cmd, physical_gpu=gpu,
                            gpu_uuid=gpu_uuid, training_state_sha256=dependency["state_sha256"])
                        started = time.monotonic()
                        while proc.poll() is None and not self.stop.is_set():
                            if time.monotonic() - started > WATCHDOG_SECONDS:
                                owner._stop_owned(proc)
                                raise TimeoutError("SW0167 prediction worker timed out")
                            now_uuid, now_memory, gpu_owners = owner.gpu_guard._gpu_info(gpu)
                            # NVML may briefly retain a PID after a successful
                            # worker exit. Re-poll our captured child before
                            # treating that stale sample as foreign ownership.
                            if proc.poll() is not None:
                                break
                            if now_uuid != gpu_uuid:
                                if proc.poll() is None:
                                    owner._stop_owned(proc)
                                    raise RuntimeError("leased GPU UUID changed during inference")
                                break
                            foreign = owner.gpu_guard._foreign_owners(
                                gpu_owners, owner.gpu_guard._process_tree(proc.pid))
                            if foreign and proc.poll() is None:
                                owner._stop_owned(proc)
                                raise RuntimeError(f"foreign GPU owner detected: {foreign}")
                            time.sleep(2.0)
                        if self.stop.is_set() and proc.poll() is None:
                            owner._stop_owned(proc)
                            raise RuntimeError("prediction cancelled; owned child stopped")
                        rc = proc.wait()
                    if rc != 0:
                        raise RuntimeError(f"SW0167 prediction child exited {rc}; preserving work rows")
                    final_inputs = evaluation.prepare_prediction_inputs_cpu()
                    predictions = {arm: evaluation.validate_prediction(
                        arm, prepared=final_inputs) for arm in evaluation.ARMS}
                    actual = {arm: {"prediction_sha256": row["prediction_sha256"],
                                    "protocol_sha256": row["protocol_sha256"]}
                              for arm, row in predictions.items()}
                    self._update("complete", prediction_hashes=actual,
                        training_state_sha256=dependency["state_sha256"], finished=time.time())
                    return 0
                finally:
                    try:
                        if proc is not None and proc.poll() is None:
                            owner._stop_owned(proc)
                    finally:
                        self.child = None
                        self._release()
            return 2
        except BaseException as exc:
            self._update("failed", error=repr(exc), finished=time.time(), artifacts_preserved=True)
            return 1
        finally:
            self.stop.set()
            if self.child is not None and self.child.poll() is None:
                owner._stop_owned(self.child)
            self._release()
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"runtime": str(RUNTIME), "arms": list(train.ARMS),
            "count": evaluation.COUNT, "gpu_candidates": [0, 1, 2, 3],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}, allow_nan=False))
        return 0
    return PredictionQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
