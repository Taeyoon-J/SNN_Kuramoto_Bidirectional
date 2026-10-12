"""Owner-aware SW0146 seven-task queue. CPU-only scoring; no automatic retries."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import fcntl
except ImportError:
    fcntl = None

HERE = Path(__file__).resolve().parent
ACMLAB = HERE.parents[1]
ROOT = Path(os.environ.get("PATCH_V2_REPO", ACMLAB / "patch_v2_sw")).resolve()
EXP = ROOT / "collaborative_test/SW_0146_native32_boundary_q_pilot"
ARCHIVE = EXP / "results_archive"
RUNTIME = ARCHIVE / "queue_runtime_20261012"
STATE = RUNTIME / "queue_state.json"
LOGS = RUNTIME / "logs"
ADMISSION = EXP / "admission.json"
HELPER_SHA = "b32c1aa5ce8417359aaab40bb9606e6fdee2c8d8314281f365616c24805dbe00"
SERVER_HELPER = ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/results_archive/recovery_code_v3_20261012/recover_queue.py"
LOCAL_HELPER = HERE.parent / "SW0143_recovery_v2_20261012/recover_queue.py"
HELPER = SERVER_HELPER if SERVER_HELPER.is_file() else LOCAL_HELPER
SAMPLE_SECONDS = 5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_helper():
    if not HELPER.is_file() or sha(HELPER) != HELPER_SHA:
        raise RuntimeError("reviewed SW0143 stable-idle helper missing or changed")
    spec = importlib.util.spec_from_file_location("sw0143_idle_for_sw0146", HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load owner helper {HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = load_helper()
frozen = helper.frozen


def task_plan():
    from collaborative_test.SW_0146_native32_boundary_q_pilot import run
    return [
        {"task_id": f"preflight_{arm}", "stage": "preflight", "arm": arm,
         "output": str(run.preflight_path(arm))}
        for arm in run.ARMS
    ] + [
        {"task_id": f"train_{arm}", "stage": "train", "arm": arm,
         "output": str(run.arm_dir(arm) / "manifest.json")}
        for arm in run.ARMS
    ] + [
        {"task_id": f"predict_{arm}", "stage": "predict", "arm": arm,
         "output": str(run._prediction_dir(arm) / "protocol.json")}
        for arm in run.ARMS
    ] + [{"task_id": "score", "stage": "score", "arm": None,
          "output": str(run.ARCHIVE / "evaluation_seed1.json")}]


def _result_valid(task, run):
    try:
        stage, arm = task["stage"], task["arm"]
        if stage == "preflight":
            ctx = run._admission_context(ADMISSION)
            return run._valid_preflight(arm, task["output"], ctx)
        if stage == "train":
            return run.validate_training(arm)["manifest"].get("status") == "training_complete"
        if stage == "predict":
            return run.validate_prediction(arm).get("protocol", {}).get("status") == "complete"
        if stage == "score":
            record = json.loads(Path(task["output"]).read_text(encoding="utf-8"))
            return (record.get("status") == "complete" and record.get("experiment") == run.NAME
                    and record.get("seed") == 1 and record.get("ground_truth_opened_after_all_predictions_validated") is True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
    return False


def _dependencies():
    return {
        "preflight_encoder_graph": (), "preflight_joint": (),
        "train_encoder_graph": ("preflight_encoder_graph", "preflight_joint"),
        "train_joint": ("preflight_encoder_graph", "preflight_joint"),
        "predict_encoder_graph": ("train_encoder_graph",),
        "predict_joint": ("train_joint",),
        "score": ("predict_encoder_graph", "predict_joint"),
    }


def child_environment(stage, gpu=None, base=None):
    """Return explicit child environment; CPU score must never initialize CUDA."""
    env = dict(os.environ if base is None else base)
    env.update(PATCH_V2_REPO=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if stage == "score":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        if gpu not in range(4):
            raise ValueError("GPU stages require a mapped physical GPU0..3")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _record_outcome(stop_event, valid):
    if not valid:
        stop_event.set()
    return bool(valid)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class PilotQueue:
    def __init__(self, admission=ADMISSION, *, sample=None, sleep=None, monotonic=None,
                 popen=None, max_workers=4):
        if fcntl is None:
            raise RuntimeError("SW0146 GPU queue requires Linux flock")
        from collaborative_test.SW_0146_native32_boundary_q_pilot import run
        self.run_module = run
        self.admission = Path(admission).resolve()
        if self.admission != run.ADMISSION.resolve():
            raise ValueError("SW0146 admission path is fixed by protocol")
        if not self.admission.is_file():
            raise FileNotFoundError("sealed SW0146 admission is not available yet")
        self.context = run._admission_context(self.admission)
        self.plan = task_plan()
        self.deps = _dependencies()
        if {row["task_id"] for row in self.plan} != set(self.deps):
            raise AssertionError("SW0146 task plan and dependency graph differ")
        if RUNTIME.exists() or STATE.exists() or any(Path(row["output"]).exists() for row in self.plan):
            raise FileExistsError("preserve existing SW0146 queue or stage artifacts")
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        lockpath = ARCHIVE / "pilot_queue.lock"
        self.lock = lockpath.open("x+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise RuntimeError("another SW0146 queue owns the experiment lock")
        self.runtime = RUNTIME
        self.logs = LOGS
        self.runtime.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir(exist_ok=False)
        self.stop = threading.Event()
        self.sample = sample or (lambda gpu: frozen.owner._nvidia_gpu_info(gpu))
        self.sleep = sleep or time.sleep
        self.monotonic = monotonic or time.monotonic
        self.popen = popen or subprocess.Popen
        self.max_workers = max_workers
        self.state = {
            "experiment": run.NAME, "status": "queued", "supervisor_pid": os.getpid(),
            "run_sha256": sha(EXP / "run.py"), "protocol_sha256": sha(EXP / "protocol.json"),
            "queue_sha256": sha(Path(__file__)), "owner_helper_sha256": sha(HELPER),
            "owner_dispatcher_sha256": sha(Path(frozen.owner.__file__)),
            "admission_sha256": self.context["admission_sha256"],
            "sw0145_report_sha256": self.context["sw0145_report_sha256"],
            "sw0143_score_sha256": self.context["sw0143_score_sha256"],
            "dependencies": self.deps,
            "tasks": {row["task_id"]: {"status": "queued", "task": row} for row in self.plan},
        }
        self._save()

    def _save(self):
        atomic_json(STATE, self.state)

    def _update(self, task_id, **fields):
        self.state["tasks"][task_id].update(fields)
        self._save()

    def _reserve(self, task_id):
        while not self.stop.is_set():
            for gpu in range(4):
                lease_path = frozen.GPU_LEASES / f"gpu{gpu}.lock"
                lease_path.parent.mkdir(parents=True, exist_ok=True)
                lease = lease_path.open("a+")
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    sample = lambda: self.sample(gpu)
                    first = sample()
                    if not frozen.gpu_candidate(first[1], first[2]):
                        fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); continue
                    stable, detail = helper.stable_idle_window(
                        sample, stop_event=self.stop, monotonic=self.monotonic, sleep=self.sleep,
                        on_sample=lambda elapsed, uid, mem: self._update(
                            task_id, status="waiting_for_stable_idle_window", physical_gpu=gpu,
                            gpu_uuid=uid, stable_seconds=elapsed, memory_mib=mem))
                    if stable:
                        return gpu, lease, detail["gpu_uuid"]
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
                except BaseException:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
                    raise
            self._update(task_id, status="waiting_for_exclusive_gpu")
            self.sleep(5)
        raise RuntimeError("queue stopped before GPU reservation")

    def _execute(self, task):
        from collaborative_test.SW_0146_native32_boundary_q_pilot import run
        task_id, stage, arm = task["task_id"], task["stage"], task["arm"]
        gpu = lease = proc = None
        try:
            for dep in self.deps[task_id]:
                if self.state["tasks"][dep].get("status") != "passed":
                    raise RuntimeError(f"dependency not passed: {dep}")
            argv = [sys.executable, "-m", "collaborative_test.SW_0146_native32_boundary_q_pilot.run",
                    "--stage", stage, "--admission", str(self.admission)]
            if arm is not None:
                argv += ["--arm", arm]
            log_path = self.logs / f"{task_id}.log"
            if log_path.exists() or Path(task["output"]).exists():
                raise FileExistsError("preserve preexisting child log/output")
            if stage != "score":
                argv += ["--device", "cuda:0"]
                gpu, lease, uuid = self._reserve(task_id)
                now, memory, owners = self.sample(gpu)
                if now != uuid or not frozen.gpu_candidate(memory, owners):
                    raise RuntimeError("GPU changed after stable idle window")
            env = child_environment(stage, gpu, os.environ)
            if stage == "score":
                pass
            else:
                cache_dir = Path("/tmp") / f"kevinswk_sw0146_{os.getpid()}_{task_id}_{int(time.time())}"
                cache_dir.mkdir(exist_ok=False)
                env["TRITON_CACHE_DIR"] = str(cache_dir)
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                self._update(task_id, status="launching", argv=argv, log_path=str(log_path),
                             physical_gpu=gpu, cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"])
                pass_fds = (lease.fileno(),) if lease is not None else ()
                proc = self.popen(argv, cwd=str(ROOT), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
                self._update(task_id, status="running", child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    self.sleep(5)
                    if proc.poll() is not None or stage == "score":
                        continue
                    _, _, owners = self.sample(gpu)
                    foreign = frozen.owner.foreign_owner_pids(owners, frozen.owner._process_tree(proc.pid))
                    if foreign:
                        frozen.SelectiveGeodesicQueue._terminate_owned(proc)
                        raise RuntimeError(f"foreign GPU owner appeared: {foreign}")
            valid = proc.returncode == 0 and _result_valid(task, run)
            _record_outcome(self.stop, valid)
            self._update(task_id, status="passed" if valid else "failed",
                         returncode=proc.returncode, artifact_valid=bool(valid), finished=time.time())
            return valid
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                frozen.SelectiveGeodesicQueue._terminate_owned(proc)
            self.stop.set()
            self._update(task_id, status="failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())
            return False
        finally:
            if lease is not None:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()

    def _run_group(self, rows):
        if self.state["status"] == "queued":
            self.state["status"] = "running"
            self._save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(rows))) as pool:
            results = list(pool.map(self._execute, rows))
        return all(results)

    def run(self):
        plan = {row["task_id"]: row for row in self.plan}
        groups = [
            [plan["preflight_encoder_graph"], plan["preflight_joint"]],
            [plan["train_encoder_graph"], plan["train_joint"]],
            [plan["predict_encoder_graph"], plan["predict_joint"]],
            [plan["score"]],
        ]
        try:
            for group in groups:
                if not self._run_group(group):
                    for later in groups[groups.index(group) + 1:]:
                        for row in later:
                            if self.state["tasks"][row["task_id"]]["status"] == "queued":
                                self._update(row["task_id"], status="blocked_by_prior_failure")
                    self.state["status"] = "failed"
                    self._save()
                    return 1
            self.state["status"] = "complete"
            self._save()
            return 0
        finally:
            if not self.lock.closed:
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
                self.lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, default=ADMISSION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "tasks": task_plan(), "dependencies": _dependencies(),
                          "max_parallel": 4, "stable_idle_seconds": 60}, indent=2))
        return 0
    if not args.admission.is_file():
        print(json.dumps({"status": "WAITING_UPSTREAM", "reason": "sealed admission file is absent",
                          "admission": str(args.admission)}), flush=True)
        return 3
    return PilotQueue(args.admission).run()


if __name__ == "__main__":
    raise SystemExit(main())
