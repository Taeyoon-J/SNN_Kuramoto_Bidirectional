"""No-retry exclusive-GPU queue for the registered SW0133 seed-1 pilot."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

try:
    import fcntl
except ImportError:  # Windows supports dry-run and validation tests only.
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0133_soft_partition_rgb import coordinator, evaluate, preflight_queue, run

ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "pilot_queue_state.json"
LOCK = ARCHIVE / "pilot_queue.lock"
LEASES = pathlib.Path("/tmp/kevinswk_sw0113_gpu_leases")
MAX_MEMORY_MIB = 512


def task_plan():
    tasks = []
    for arm in run.ARMS:
        tasks.append({"task_id": f"sw0133_train_s1_{arm}", "stage": "train", "seed": 1,
                      "arm": arm})
        tasks.append({"task_id": f"sw0133_eval_s1_{arm}", "stage": "evaluate", "seed": 1,
                      "arm": arm})
    return tasks


def _paths(task):
    arm, stage = task["arm"], task["stage"]
    if task.get("seed") != 1 or arm not in run.ARMS or stage not in ("train", "evaluate"):
        raise ValueError("unregistered SW0133 pilot task")
    train_dir = ROOT / "trained_models" / "SW0133_soft_partition_rgb" / f"seed1_{arm}"
    eval_dir = ARCHIVE / f"evaluation_seed1_{arm}"
    return train_dir, eval_dir


def command(task, device="cuda:0"):
    train_dir, eval_dir = _paths(task)
    if task["stage"] == "train":
        return [sys.executable, str(HERE / "train.py"), "--seed", "1", "--arm", task["arm"],
                "--device", device, "--output", str(train_dir)]
    return [sys.executable, str(HERE / "evaluate.py"), "--seed", "1", "--arm", task["arm"],
            "--device", device, "--training-dir", str(train_dir), "--output", str(eval_dir)]


def valid_result(task):
    train_dir, eval_dir = _paths(task)
    try:
        if task["stage"] == "train":
            from collaborative_test.SW_0133_soft_partition_rgb import train
            assets = run.sw130.validate_rgb_assets()
            manifest_path, manifest, _hashes, _source_sha = evaluate._training_artifacts(
                1, task["arm"], train_dir, assets)
            return (manifest_path.is_file() and manifest.get("status") == "training_complete"
                    and (train_dir / "TRAINING_COMPLETED").is_file()
                    and manifest.get("trainer_sha256") == run.sha(HERE / "train.py")
                    and manifest.get("preflight_queue_sha256") == run.sha(HERE / "preflight_queue.py"))
        return coordinator.evaluation_valid(1, task["arm"], eval_dir / "evaluation.json", train_dir)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError):
        return False


def _atomic(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


class PilotQueue:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError("SW0133 GPU pilot queue requires Linux flock")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        self.queue_lock = LOCK.open("a+")
        try:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.queue_lock.close()
            raise RuntimeError("another SW0133 pilot queue owns its lock") from exc
        try:
            if STATE.exists():
                raise FileExistsError(f"preserve existing SW0133 pilot state: {STATE}")
            assets = run.sw130.validate_rgb_assets()
            for seed in run.SEEDS:
                if not preflight_queue.valid_result({"stage": "preflight", "seed": seed}, assets):
                    raise RuntimeError(f"seed{seed} source preflight is missing/invalid; pilot blocked")
        except BaseException:
            fcntl.flock(self.queue_lock.fileno(), fcntl.LOCK_UN)
            self.queue_lock.close()
            raise
        self.tasks = task_plan()
        self.state = {"experiment": "SW0133_soft_partition_rgb", "status": "running",
                      "stage": "paired_seed1_pilot", "supervisor_pid": os.getpid(),
                      "runner_fingerprint": run.implementation_fingerprint(),
                      "queue_sha256": run.sha(HERE / "pilot_queue.py"),
                      "dispatcher_sha256": run.sha(owner.__file__),
                      "evaluation_sha256": run.sha(HERE / "evaluate.py"),
                      "trainer_sha256": run.sha(HERE / "train.py"),
                      "tasks": {t["task_id"]: {"task": t, "status": "queued"}
                                for t in self.tasks}, "started": time.time()}
        _atomic(STATE, self.state)

    def _update(self, task_id, **fields):
        self.state["tasks"][task_id].update(fields)
        _atomic(STATE, self.state)

    def _reserve(self, task_id):
        while True:
            for gpu in range(4):
                lease_path = LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close(); continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and memory <= MAX_MEMORY_MIB:
                        self._update(task_id, status="reserved", gpu=gpu, gpu_uuid=uuid,
                                     memory_before_mib=memory, lease_path=str(lease_path))
                        return gpu, lease, uuid
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); raise
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
            self._update(task_id, status="waiting_for_exclusive_gpu", last_wait=time.time())
            time.sleep(5)

    @staticmethod
    def _stop_owned(proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

    def _execute(self, task):
        key = task["task_id"]
        train_dir, eval_dir = _paths(task)
        artifact = train_dir if task["stage"] == "train" else eval_dir
        log = ARCHIVE / f"{key}.log"
        if artifact.exists() or log.exists():
            self._update(key, status="failed_existing_artifact", preserved=str(artifact))
            return False
        lease = proc = None
        try:
            gpu, lease, uuid = self._reserve(key)
            triton = pathlib.Path(f"/tmp/kevinswk_sw0133_{key}_{int(time.time()*1000)}")
            triton.mkdir(parents=True, exist_ok=False)
            argv = command(task)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TRITON_CACHE_DIR=str(triton))
            self._update(key, status="launching", argv=argv, gpu=gpu, gpu_uuid=uuid,
                         log_path=str(log), triton_cache=str(triton))
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(key, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None:
                        break
                    _uuid, _memory, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._stop_owned(proc)
                        self._update(key, status="interrupted_foreign_gpu_owner",
                                     foreign_pids=foreign, finished=time.time())
                        return False
            rc = int(proc.returncode)
            passed = rc == 0 and valid_result(task)
            self._update(key, status="passed" if passed else "failed",
                         returncode=rc, artifact_valid=bool(passed), finished=time.time())
            return passed
        except BaseException as exc:
            if proc is not None: self._stop_owned(proc)
            self._update(key, status="failed", error=repr(exc), finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        for task in self.tasks:
            if not self._execute(task):
                for later in self.tasks[self.tasks.index(task) + 1:]:
                    self._update(later["task_id"], status="blocked_prior_task_failure")
                self.state.update(status="pilot_pipeline_failed", finished=time.time())
                _atomic(STATE, self.state)
                return False
        reports = {}
        for arm in run.ARMS:
            path = _paths({"stage": "evaluate", "seed": 1, "arm": arm})[1] / "evaluation.json"
            if not coordinator.evaluation_valid(1, arm, path):
                self.state.update(status="pilot_artifact_validation_failed", finished=time.time())
                _atomic(STATE, self.state)
                return False
            reports[arm] = json.loads(path.read_text(encoding="utf-8"))
        _source_path, source_sha, source = evaluate._source_eval_reference(1)
        gate = coordinator.pilot_gate(source, reports)
        gate.update({"source97_evaluation_path": str(_source_path.resolve()),
                     "source97_evaluation_sha256": source_sha,
                     "evaluation_sha256": {
                         arm: run.sha(_paths({"stage": "evaluate", "seed": 1,
                                              "arm": arm})[1] / "evaluation.json")
                         for arm in run.ARMS},
                     "training_manifest_sha256": {
                         arm: run.sha(_paths({"stage": "train", "seed": 1,
                                              "arm": arm})[0] / "manifest.json")
                         for arm in run.ARMS}})
        gate_path = ARCHIVE / "pilot_gate_seed1.json"
        run.write_once(gate_path, gate)
        self.state.update(status=gate["status"], pilot_gate_path=str(gate_path.resolve()),
                          pilot_gate_sha256=run.sha(gate_path), finished=time.time())
        _atomic(STATE, self.state)
        return gate["status"] == "pilot_gate_passed"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"tasks": [{**task, "argv": command(task)} for task in task_plan()],
                          "serial": True, "no_retries": True}, indent=2))
        return
    PilotQueue().run()


if __name__ == "__main__":
    main()
