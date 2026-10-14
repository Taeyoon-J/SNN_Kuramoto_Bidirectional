"""Owner-reserved fixed320 source-only inference for the SW0162 candidate/control pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
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

_MISSING = object()
_saved_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", _MISSING)
from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
if _saved_visibility is _MISSING:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _saved_visibility

from collaborative_test.SW_0162_connected_signed_partition import predict_evaluate as evaluation
from collaborative_test.SW_0162_connected_signed_partition import preflight, preflight_queue

RUNTIME = HERE / "results_archive" / "prediction_queue_20261014"
STATE, LOCK, LOG = RUNTIME / "state.json", RUNTIME / "queue.lock", RUNTIME / "attempt01.log"
MAX_MEMORY_MIB, STABLE_SECONDS = 512, 5.0
SAMPLE_SECONDS, POLL_SECONDS, WATCHDOG_SECONDS = 2.0, 5.0, 7200.0
DEVICE = "cuda:0"
ACTIVE_PREFLIGHT_STATES = {"waiting_for_idle_gpu", "waiting_stable_gpu",
    "waiting_for_exclusive_gpu", "waiting_gpu_probe_error", "running"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def command():
    return [sys.executable, "-m", "collaborative_test.SW_0162_connected_signed_partition.predict_evaluate",
        "--stage", "predict", "--device", DEVICE]


def implementation_fingerprint():
    guard = Path(owner.OWNER_GUARD)
    if sha256_file(guard) != owner.OWNER_GUARD_SHA256:
        raise ValueError("shared GPU owner guard checksum changed")
    paths = (HERE / "prediction_queue.py", HERE / "predict_evaluate.py", HERE / "preflight.py",
        HERE / "preflight_queue.py", HERE / "connected_signed.py", HERE / "protocol.json",
        HERE / "DESIGN_KO.md", Path(owner.__file__), guard)
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("SW0162 prediction queue dependency missing")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def _atomic_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2, allow_nan=False); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def _proc_has_args(pid, module):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        row = owner._proc_record(pid)
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(row and row.get("state") != "Z" and module in row.get("argv", []))


class PredictionQueue:
    def __init__(self, *, prepared_fn=None, owner_api=None):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.owner = owner if owner_api is None else owner_api
        if fcntl is None:
            raise RuntimeError("SW0162 GPU reservation requires Linux flock")
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.lock_stream = LOCK.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise RuntimeError("another SW0162 prediction supervisor owns the queue") from exc
        self.state_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.fingerprint = implementation_fingerprint()
        self.report_fingerprint = evaluation.implementation_fingerprint()
        self.prepared_fn = prepared_fn or evaluation.prepare_prediction_cpu
        self.child = None
        self.lease = None
        try:
            done = all(evaluation._output_dir(a).is_dir() for a in evaluation.ARMS)
            if done:
                prepared = self.prepared_fn()
                for arm in evaluation.ARMS:
                    evaluation.validate_prediction(arm, prepared=prepared)
                if STATE.exists():
                    prior = json.loads(STATE.read_text(encoding="utf-8"))
                    if (prior.get("status") != "complete"
                            or prior.get("experiment") != "SW0162_connected_signed_partition"):
                        raise FileExistsError("preserving noncomplete SW0162 prediction queue state")
                elif LOG.exists():
                    raise FileExistsError("preserving SW0162 prediction log without terminal state")
                self._write_state("complete", adopted=True,
                    prediction_fingerprint=self.report_fingerprint, finished=time.time())
                self.complete = True
            else:
                if STATE.exists() or LOG.exists():
                    raise FileExistsError("preserving existing SW0162 prediction queue state/log")
                self.complete = False
                self._write_state("waiting_for_source_preflight", supervisor_pid=os.getpid(),
                    queue_fingerprint=self.fingerprint, prediction_fingerprint=self.report_fingerprint,
                    task={"arms": list(evaluation.ARMS), "image_ids": [1320, 1639], "count": 320,
                        "steps": 1024, "settle": 512, "optimizer_updates": 0,
                        "ground_truth_used_for_prediction": False},
                    outputs={arm: str(evaluation._output_dir(arm)) for arm in evaluation.ARMS})
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_stream.close()

    def _write_state(self, status, **fields):
        with self.state_lock:
            current = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
            current.update(experiment="SW0162_connected_signed_partition", status=status,
                updated=time.time(), **fields)
            _atomic_json(STATE, current)

    def _verify(self):
        if (implementation_fingerprint() != self.fingerprint
                or evaluation.implementation_fingerprint() != self.report_fingerprint):
            raise RuntimeError("SW0162 prediction implementation changed after reservation")

    def _preflight_owner_live(self, state):
        for key, module in (("supervisor_pid", "collaborative_test.SW_0162_connected_signed_partition.preflight_queue"),
                            ("child_pid", "collaborative_test.SW_0162_connected_signed_partition.preflight")):
            pid = state.get(key)
            if type(pid) is not int or pid <= 0:
                continue
            try:
                proc = owner._proc_record(pid)
            except (OSError, RuntimeError, ValueError):
                continue
            if proc and proc.get("state") != "Z" and module in proc.get("argv", []):
                return True
        return False

    def _preflight(self):
        path = preflight_queue.STATE
        while not self.stop_event.is_set():
            state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            if state is None:
                raise RuntimeError("SW0162 source preflight has not been registered")
            if state.get("experiment") != "SW0162_connected_signed_partition":
                raise ValueError("unexpected SW0162 preflight queue state")
            if state.get("status") == "complete":
                report = json.loads(preflight.OUTPUT.read_text(encoding="utf-8"))
                if not preflight_queue.validate_report(preflight.OUTPUT,
                        expected_fingerprint=preflight.implementation_fingerprint(),
                        expected_source=preflight_queue.source_identity()):
                    raise ValueError("SW0162 actual eight-TRAIN preflight report is invalid")
                if report.get("status") != "source8_preflight_complete":
                    raise ValueError("SW0162 source preflight status is not complete")
                return self.prepared_fn()
            if state.get("status") not in ACTIVE_PREFLIGHT_STATES or not self._preflight_owner_live(state):
                raise RuntimeError(f"SW0162 source preflight is terminal/nonlive: {state.get('status')}")
            self._write_state("waiting_for_source_preflight", preflight_status=state.get("status"),
                preflight_supervisor_pid=state.get("supervisor_pid"))
            self.stop_event.wait(POLL_SECONDS)
        raise RuntimeError("SW0162 prediction queue stopped while awaiting source preflight")

    def _final_idle_check(self, gpu, gpu_uuid):
        try:
            actual_uuid, memory, owners = self.owner.gpu_guard._gpu_info(gpu)
        except Exception as exc:
            return False, {"reason": "gpu_query_error", "detail": repr(exc)}
        info = {"gpu_uuid": actual_uuid, "memory_used_mib": int(memory), "owner_pids": list(owners)}
        if actual_uuid != gpu_uuid: info["reason"] = "uuid_changed"
        elif owners: info["reason"] = "foreign_owner"
        elif int(memory) > MAX_MEMORY_MIB: info["reason"] = "memory_over_limit"
        else: info["reason"] = "idle"
        return info["reason"] == "idle", info

    def run(self):
        if self.complete:
            self._unlock(); return 0
        try:
            prepared = self._preflight()  # CPU source/reference/RGB/schema checks before a GPU lease.
            self._verify()
            # If all 320 committed rows already exist, finish/adopt without model allocation.
            work = evaluation._work_dir()
            if all(evaluation._row_dir(work, i).is_dir() for i in range(evaluation.COUNT)):
                evaluation.predict_candidate(device=DEVICE)
                for arm in evaluation.ARMS:
                    evaluation.validate_prediction(arm, prepared=prepared)
                self._write_state("complete", adopted_from_committed_rows=True, finished=time.time())
                return 0
            while not self.stop_event.is_set():
                gpu, gpu_uuid, memory, lease = self.owner._acquire(self.stop_event)
                self.lease = lease
                self._write_state("waiting_stable_gpu", gpu=gpu, gpu_uuid=gpu_uuid,
                    initial_memory_mib=memory, output_paths={a: str(evaluation._output_dir(a)) for a in evaluation.ARMS})
                stable = self.owner._stable_idle(gpu, gpu_uuid, self.stop_event,
                    lambda elapsed, used: self._write_state("waiting_stable_gpu", gpu=gpu, gpu_uuid=gpu_uuid,
                        stable_seconds=elapsed, memory_used_mib=used))
                if not stable:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); self.lease = None
                    if self.stop_event.is_set():
                        self._write_state("cancelled_before_child", reservation_released=True)
                        return 1
                    self._write_state("waiting_for_exclusive_gpu", reservation_released=True)
                    continue
                prepared = self._preflight()  # Revalidate exact dependencies immediately before child start.
                self._verify()
                idle, detail = self._final_idle_check(gpu, gpu_uuid)
                if idle:
                    break
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); self.lease = None
                self._write_state("waiting_for_exclusive_gpu", reservation_released=True, gpu_check=detail)
            if self.stop_event.is_set():
                self._write_state("cancelled_before_child"); return 1
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("OMP_NUM_THREADS", "1"); env.setdefault("MKL_NUM_THREADS", "1")
            with LOG.open("x", encoding="utf-8", newline="\n") as log:
                self.child = subprocess.Popen(command(), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                self._write_state("running", child_pid=self.child.pid, child_pgid=self.child.pid,
                    gpu=gpu, gpu_uuid=gpu_uuid, started=time.time())
                deadline = time.monotonic() + WATCHDOG_SECONDS
                while self.child.poll() is None:
                    if time.monotonic() >= deadline:
                        self.owner._stop_owned(self.child)
                        raise TimeoutError("SW0162 fixed320 source rollout exceeded watchdog")
                    try:
                        now_uuid, now_memory, owners = self.owner.gpu_guard._gpu_info(gpu)
                    except Exception:
                        now_uuid, now_memory, owners = gpu_uuid, 0, []
                    foreign = [pid for pid in owners if pid != self.child.pid]
                    if now_uuid != gpu_uuid or foreign:
                        self.owner._stop_owned(self.child)
                        raise RuntimeError("foreign GPU owner/collision; only the owned child was stopped")
                    self._write_state("running", child_pid=self.child.pid, gpu=gpu, gpu_uuid=gpu_uuid,
                        memory_used_mib=int(now_memory), foreign_owner_pids=foreign)
                    time.sleep(1.0)
                code = self.child.wait()
            if code != 0:
                raise RuntimeError(f"SW0162 prediction child exited {code}; see {LOG}")
            self._verify()
            for arm in evaluation.ARMS:
                evaluation.validate_prediction(arm, prepared=prepared)
            self._write_state("complete", child_returncode=code, prediction_fingerprint=self.report_fingerprint,
                output_hashes={arm: evaluation.validate_prediction(arm, prepared=prepared)["prediction_sha256"]
                    for arm in evaluation.ARMS}, finished=time.time())
            return 0
        except BaseException as exc:
            if self.child is not None and self.child.poll() is None:
                self.owner._stop_owned(self.child)
            self._write_state("failed", error=repr(exc), finished=time.time())
            raise
        finally:
            if self.lease is not None:
                try: fcntl.flock(self.lease.fileno(), fcntl.LOCK_UN)
                finally: self.lease.close(); self.lease = None
            self._unlock()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "planned", "arms": list(evaluation.ARMS), "count": 320,
            "image_ids": [1320, 1639], "steps": 1024, "optimizer_updates": 0,
            "outputs": {arm: str(evaluation._output_dir(arm)) for arm in evaluation.ARMS},
            "queue_fingerprint": implementation_fingerprint(),
            "prediction_fingerprint": evaluation.implementation_fingerprint(),
            "ground_truth_used_for_prediction": False}, allow_nan=False), flush=True)
        return 0
    return PredictionQueue().run()


if __name__ == "__main__":
    raise SystemExit(main())
