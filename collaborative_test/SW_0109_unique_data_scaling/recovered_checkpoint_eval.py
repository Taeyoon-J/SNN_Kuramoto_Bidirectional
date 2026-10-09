"""Checkpoint-only evaluation of SW0109 seed-0 outputs with incomplete manifests.

This queue never marks training complete or repairs historical training metadata.
It evaluates only the three already-produced final core.pt files into fresh,
separate diagnostic directories.
"""
try:
    import fcntl
except ImportError:  # The GPU supervisor runs on Linux; local contract tests may run on Windows.
    fcntl = None
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import run

ROOT = run.ROOT
HERE = run.HERE
OUT = run.OUT
ARCHIVE = run.ARCHIVE
QUEUE_DIR = ARCHIVE / "recovered_checkpoint_eval_20261009"
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
TRAIN_GAMMA = ROOT / "data/SW_0090_large_unique_scale/gamma_train_70000.pt"
TRAIN_MANIFEST = ROOT / "data/SW_0090_large_unique_scale/manifest.json"
VAL_GAMMA = ROOT / "data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
VAL_MANIFEST = ROOT / "data/SW_0042_hdf5_aligned/manifest.json"
PYTHON = Path("/Data0/kevinswk/envs/snn/bin/python")
LEASE_ROOT = Path("/tmp/kevinswk_sw0113_gpu_leases")
EXPECTED_RUNNER_SHA256 = "500c30573cb43e6070d45ee8db919aff8838b1ce3aef0efebc5d42feb5aa83ab"
EVALUATOR = ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"
EXPECTED_EVALUATOR_SHA256 = "b3d7b2344329940c05adcd0139108feacea4dd2eeb495295bf0e7db5595feab1"
EXPECTED_CORE_SHA256 = {
    2500: "1ec3929ea32fa6fb435686b7fc45fc2f8f12d3ca1d6d6f85b1033de7bbd67ab3",
    10000: "09150ec6cb8c2a644ea36fb2fa6049fd416fc8afd092ce02ea14f38e9a6966b9",
    70000: "8ada8fe58fd87b18576ba3c33df5f2bf50aff5bfa4584c951cac047de2b41c34",
}
METRICS = run.METRICS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_plan():
    tasks = []
    for size in (2500, 10000, 70000):
        core = OUT / f"seed0_N{size}" / "core.pt"
        tasks.append({
            "task_id": f"checkpoint_only_eval_seed0_N{size}",
            "stage": "checkpoint-only-eval",
            "seed": 0,
            "size": size,
            "checkpoint": str(core),
            "checkpoint_sha256": EXPECTED_CORE_SHA256[size],
            "output": str(QUEUE_DIR / f"N{size}"),
            "training_manifest_status": "missing_historical_manifest; training not certified complete",
        })
    return tasks


def output_path(task):
    return Path(task["output"])


def command(task, device):
    return [str(PYTHON), str(HERE / "run.py"), "--stage", "eval",
            "--seed", "0", "--size", str(task["size"]),
            "--checkpoint", task["checkpoint"], "--output", task["output"],
            "--train-gamma", str(TRAIN_GAMMA), "--train-manifest", str(TRAIN_MANIFEST),
            "--val-gamma", str(VAL_GAMMA), "--val-manifest", str(VAL_MANIFEST),
            "--dataset", str(DATASET), "--device", device]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_state(path, state):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def valid_result(task):
    out = output_path(task)
    report = out / "evaluation.json"
    manifest = out / "evaluation_manifest.json"
    if not (out / "COMPLETED").is_file() or not report.is_file() or not manifest.is_file():
        return False
    if not Path(task["checkpoint"]).is_file() or sha256(task["checkpoint"]) != task["checkpoint_sha256"]:
        return False
    try:
        m, d = _read(manifest), _read(report)
        if (m.get("status") != "complete" or m.get("seed") != 0
                or m.get("pool_size") != task["size"]
                or m.get("checkpoint_sha256") != task["checkpoint_sha256"]
                or Path(m.get("checkpoint", "")).resolve() != Path(task["checkpoint"]).resolve()
                or m.get("runner_sha256") != EXPECTED_RUNNER_SHA256
                or m.get("validation_gamma_sha256") != sha256(VAL_GAMMA)
                or m.get("validation_gamma_manifest_sha256") != sha256(VAL_MANIFEST)
                or Path(m.get("dataset_path", "")).resolve() != DATASET.resolve()
                or m.get("ids") != [1320, 1639] or m.get("images") != 320
                or m.get("ground_truth_used_for_prediction") is not False):
            return False
        if (d.get("ids") != [1320, 1639] or d.get("images") != 320
                or d.get("ground_truth_used_for_prediction") is not False):
            return False
        score = d["sweep"][0]["scored_targets"]["our_hdf5"]
        for name in METRICS:
            values = score["per_image"][name]
            mean = float(score["metrics"][name])
            if (score["valid_count"][name] != 320 or len(values) != 320
                    or not all(math.isfinite(float(x)) for x in values)
                    or not math.isfinite(mean)
                    or abs(sum(float(x) for x in values) / 320 - mean) > 1e-12):
                return False
            if (m.get("valid_count", {}).get(name) != 320
                    or abs(float(m.get("metrics", {}).get(name, float("nan"))) - mean) > 1e-12):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _gpu_uuid(gpu):
    return subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()


def _gpu_owners(gpu):
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                      "--format=csv,noheader"], text=True)
    uuid = _gpu_uuid(gpu)
    return [int(row.split(",")[1].strip()) for row in output.splitlines()
            if len(row.split(",")) == 2 and row.split(",")[0].strip() == uuid]


def _try_lease(gpu):
    if fcntl is None:
        raise RuntimeError("GPU lease supervisor requires Linux flock")
    path = LEASE_ROOT / f"gpu{gpu}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    try:
        used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                                           "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                          text=True).strip())
        owners = _gpu_owners(gpu)
    except Exception:
        stream.close()
        raise
    if owners or used > 512:
        stream.close()
        return None
    return gpu, stream, used


def is_descendant(pid, ancestor, parent_lookup):
    """Return true only for a confirmed live parent chain; missing PIDs are not owned."""
    seen = set()
    current = pid
    for _ in range(64):
        if current == ancestor:
            return True
        if current in seen:
            return False
        seen.add(current)
        parent = parent_lookup(current)
        if parent is None or parent <= 0:
            return False
        current = parent
    return False


def _parent_pid(pid):
    try:
        value = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return int(value) if value else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def _pid_exists(pid):
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        return True
    # This supervisor is Linux-only; without /proc, fail conservatively.
    return os.name != "posix"


def foreign_owners(owner_pids, parent_pid, parent_lookup, process_exists):
    foreign = []
    for pid in owner_pids:
        if is_descendant(pid, parent_pid, parent_lookup):
            continue
        # nvidia-smi can briefly report a PID that exited between GPU and ps
        # queries. Ignore only a positively confirmed vanished process.
        if process_exists(pid):
            foreign.append(pid)
    return foreign


def _foreign_owners(gpu, parent_pid):
    return foreign_owners(_gpu_owners(gpu), parent_pid, _parent_pid, _pid_exists)


def _stop_owned(children):
    stopped = {}
    for task_id, rec in children.items():
        proc = rec["proc"]
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    os.killpg(proc.pid, 9)
                    proc.wait()
        stopped[task_id] = proc.poll()
        rec["lease"].close()
    return stopped


def run_queue(poll_seconds=5):
    if fcntl is None:
        raise RuntimeError("GPU lease supervisor requires Linux flock")
    if sha256(HERE / "run.py") != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("SW0109 runner SHA changed; do not use the checkpoint-only queue")
    if sha256(EVALUATOR) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("shared evaluation implementation SHA changed; do not start")
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    for path in (TRAIN_GAMMA, TRAIN_MANIFEST, VAL_GAMMA, VAL_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    run.gamma_contract(TRAIN_GAMMA, TRAIN_MANIFEST)
    run.gamma_contract(VAL_GAMMA, VAL_MANIFEST, validation=True)
    tasks = task_plan()
    for task in tasks:
        checkpoint = Path(task["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != task["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint missing or SHA mismatch: {checkpoint}")
        if output_path(task).exists():
            raise FileExistsError(f"preserve existing diagnostic output: {output_path(task)}")
    QUEUE_DIR.mkdir(parents=True, exist_ok=False)
    queue_lock = (QUEUE_DIR / "queue.lock").open("x", encoding="utf-8")
    queue_lock.write(str(os.getpid()))
    queue_lock.flush()
    if fcntl is not None:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = QUEUE_DIR / "queue_state.json"
    logs = QUEUE_DIR / "logs"
    logs.mkdir()
    state = {"status": "running", "supervisor_pid": os.getpid(),
             "scope": "checkpoint-only evaluation; historical training incomplete",
             "runner_sha256": EXPECTED_RUNNER_SHA256, "tasks": {}, "started": time.time(),
             "shared_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
             "validation_gamma_sha256": sha256(VAL_GAMMA),
             "validation_gamma_manifest_sha256": sha256(VAL_MANIFEST),
             "training_gamma_sha256": sha256(TRAIN_GAMMA),
             "training_gamma_manifest_sha256": sha256(TRAIN_MANIFEST),
             "dataset_path": str(DATASET.resolve()),
             "dataset_identity": {"size_bytes": DATASET.stat().st_size,
                                  "mtime_ns": DATASET.stat().st_mtime_ns},
             "training_completion_certified": False}
    state["tasks"] = {task["task_id"]: {"status": "queued", "checkpoint_sha256": task["checkpoint_sha256"],
                                       "output": task["output"]} for task in tasks}
    _write_state(state_path, state)
    pending = list(tasks)
    children = {}
    try:
        while pending or children:
            progress = False
            for task_id, rec in list(children.items()):
                proc = rec["proc"]
                rc = proc.poll()
                if rc is None:
                    if any(pid not in rec["baseline_owners"] for pid in _foreign_owners(rec["gpu"], proc.pid)):
                        raise RuntimeError(f"foreign process appeared on leased GPU{rec['gpu']}; stop queue safely")
                    continue
                ok = rc == 0 and valid_result(rec["task"])
                state["tasks"][task_id].update({"status": "passed" if ok else "failed",
                                                  "return_code": rc, "finished": time.time()})
                rec["lease"].close()
                del children[task_id]
                if not ok:
                    raise RuntimeError(f"evaluation failed; preserve log/output for {task_id}")
                progress = True
            for task in list(pending):
                leased = None
                for gpu in (0, 1, 2, 3):
                    leased = _try_lease(gpu)
                    if leased is not None:
                        break
                if leased is None:
                    break
                gpu, lease, used = leased
                # Recheck after taking the shared lease, immediately before launch.
                baseline_owners = _gpu_owners(gpu)
                if baseline_owners:
                    lease.close()
                    continue
                task_id = task["task_id"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           TRITON_CACHE_DIR=f"/tmp/kevinswk_sw0109_recovered_eval_triton/gpu{gpu}")
                log_path = logs / f"{task_id}.log"
                argv = command(task, "cuda:0")
                try:
                    with log_path.open("x", encoding="utf-8") as log:
                        proc = subprocess.Popen(argv, cwd=ROOT, env=env,
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                except BaseException:
                    lease.close()
                    raise
                children[task_id] = {"proc": proc, "gpu": gpu, "lease": lease, "task": task,
                                     "baseline_owners": baseline_owners, "log": str(log_path), "argv": argv}
                state["tasks"][task_id] = {"status": "running", "gpu": gpu, "pid": proc.pid,
                                           "checkpoint_sha256": task["checkpoint_sha256"], "log": str(log_path),
                                           "memory_before_mib": used, "argv": argv}
                pending.remove(task)
                progress = True
                state["pending"] = [t["task_id"] for t in pending]
                _write_state(state_path, state)
            state["pending"] = [t["task_id"] for t in pending]
            _write_state(state_path, state)
            if not progress:
                time.sleep(poll_seconds)
        state.update({"status": "complete", "finished": time.time(),
                      "training_completion_certified": False})
        _write_state(state_path, state)
        return state
    except Exception as exc:
        stopped = _stop_owned(children)
        for task_id, rc in stopped.items():
            state["tasks"][task_id].update({"status": "stopped_after_queue_failure",
                                             "return_code": rc, "output_preserved": True,
                                             "finished": time.time()})
        children.clear()
        state.update({"status": "failed", "failure": f"{type(exc).__name__}: {exc}",
                      "pending": [t["task_id"] for t in pending], "failed_at": time.time()})
        _write_state(state_path, state)
        raise
    finally:
        _stop_owned(children)
        queue_lock.close()


if __name__ == "__main__":
    if "--run" not in sys.argv:
        raise SystemExit("Readiness only; pass --run after review to start checkpoint-only evaluations.")
    sys.argv.remove("--run")
    run_queue()
