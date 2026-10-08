"""Single owner-aware dispatcher for registered SW0108/109/110 tasks.

Adapters are loaded only on the server at queue start; this module's scheduler
primitives remain importable for lightweight tests without importing torch.
"""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Windows permits import/unit tests; real dispatch is Linux-only.
    fcntl = None
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
STATE = ARCHIVE / "queue_state.json"
GPUS = (0, 1, 2, 3)
MAX_USED_MIB = 512


def ready_tasks(tasks, completed, failed=()):
    """Return tasks with all prerequisites completed and none failed."""
    done, bad = set(completed), set(failed)
    return [t for t in tasks if t.get("task_id") not in bad
            and set(t.get("depends_on", ())) <= done
            and not (set(t.get("depends_on", ())) & bad)]


def gpu_reserve(gpu, reservations, owner_fn, memory_fn, max_used=MAX_USED_MIB):
    """Reserve locally, then acquire a cross-process flock and recheck GPU."""
    if gpu in reservations or owner_fn(gpu) or memory_fn(gpu) > max_used:
        return None
    reservations.add(gpu)
    return gpu


def handoff_valid(record_path):
    """Require an explicit parent-audited safe handoff before one master starts."""
    try:
        record = json.loads(Path(record_path).read_text())
        archived = Path(record.get("archived_state_path", ""))
        recorded_sha = record.get("archived_state_sha256")
        archive_matches = archived.is_file() and sha(archived) == recorded_sha
        return (record.get("status") == "verified_superseded"
                and record.get("active_worker_count") == 0
                and record.get("old_coordinator_alive") is False
                and archive_matches)
    except (OSError, ValueError, TypeError):
        return False


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_adapter(name, folder):
    """Load coordinator and its sibling `run` without cross-experiment aliasing."""
    folder = ROOT / folder
    original_path = list(sys.path)
    missing = object()
    old_run = sys.modules.pop("run", missing)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(
            f"sw0113_{name}_coordinator", folder / "coordinator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        sys.modules.pop("run", None)
        if old_run is not missing:
            sys.modules["run"] = old_run


def active_public(record):
    """Drop OS objects from durable JSON while retaining the lease identity."""
    clean = {k: v for k, v in record.items() if k not in ("proc", "lease_stream")}
    if "lease_stream" in record:
        clean["lease_path"] = record.get("lease_path")
    return clean


def registered_tasks(args):
    """Build exact experiment tasks through their reviewed local adapters."""
    a108 = _load_adapter("sw0108", "collaborative_test/SW_0108_gate_contribution_screen")
    a109 = _load_adapter("sw0109", "collaborative_test/SW_0109_unique_data_scaling")
    a110 = _load_adapter("sw0110", "collaborative_test/SW_0110_xy_graph_route")

    tasks = []
    for index, item in enumerate(a108.screen_tasks()):
        arm = item["arm"]
        task_id = f"sw0108_{item['stage']}_{arm}"
        deps = []
        if item["stage"] == "evaluate" and arm == "baseline":
            deps = ["sw0108_preflight_all"]
        elif item["stage"] == "evaluate" and arm != "baseline":
            deps = ["sw0108_evaluate_baseline"]
        task = {"task_id": task_id, "experiment": "SW0108", "stage": item["stage"],
                "arm": arm, "seed": 0, "depends_on": deps,
                "priority": 0 if arm in ("all", "baseline") else 2}
        if item["stage"] == "preflight":
            task["depends_on"] = ["external_sw0107_terminal"]
        task["command"] = a108.worker_cmd(item, args.source108, args.gamma108,
                                           args.dataset, "cuda:0")
        output = (a108.ARCHIVE / "preflight_seed0.json" if item["stage"] == "preflight"
                  else a108.OUT / arm)
        task["output"] = str(output)
        task["validate"] = lambda t=task, m=a108: m.valid_result(t["arm"], Path(t["output"]))
        task["adapter_index"] = index
        tasks.append(task)

    args109 = argparse.Namespace(train_gamma=args.train_gamma109,
        train_manifest=args.train_manifest109, val_gamma=args.val_gamma109,
        val_manifest=args.val_manifest109, dataset=args.dataset)
    for item in a109.task_plan():
        task = dict(item)
        task["experiment"] = "SW0109"
        if task["stage"] == "preflight":
            task["depends_on"] = ["external_sw0106_complete"]
        task["priority"] = 50 if task["stage"] == "train" else 1
        task["output"] = str(a109.output_for(task))
        task["command"] = a109.command(task, args109)
        task["validate"] = lambda t=task, m=a109: m.valid(t)
        tasks.append(task)

    for item in a110.task_plan():
        task = dict(item)
        task["output"] = str(a110.artifact_path(task))
        task["command"] = a110.command(task, "cuda:0")
        task["validate"] = lambda t=task, m=a110: m.valid_result(t)
        tasks.append(task)

    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs in central experiment registry")
    return tasks


def appended_tasks(experiments):
    """Load reviewed append-only task plans for later scientific runners."""
    folders = {"SW0111": "collaborative_test/SW_0111_frozen_partition_rgb",
               "SW0112": "collaborative_test/SW_0112_learned_gate_residual",
               "SW0114": "collaborative_test/SW_0114_resolution32_pilot"}
    result = []
    for experiment in experiments:
        if experiment not in folders:
            raise ValueError(f"unregistered append experiment {experiment}")
        folder = ROOT / folders[experiment]
        protocol_path = folder / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        if protocol.get("status") not in ("preregistered_pending_implementation_review_and_real_preflight",
                                            "preregistered"):
            raise ValueError(f"{experiment} protocol is not in preregistered state")
        adapter = _load_adapter(experiment.lower(), folders[experiment])
        if not all(hasattr(adapter, name) for name in ("task_plan", "artifact_path", "command", "valid_result")):
            raise ValueError(f"{experiment} coordinator does not expose the required task adapter API")
        for item in adapter.task_plan():
            task = dict(item)
            task.setdefault("experiment", experiment)
            task["output"] = str(adapter.artifact_path(task))
            task["command"] = adapter.command(task, "cuda:0")
            task["validate"] = lambda t=task, m=adapter: m.valid_result(t)
            result.append(task)
    return result


def read_append_manifest(path, previous_version, previous_experiments):
    """Accept strictly versioned append-only experiment names, never task edits."""
    path = Path(path)
    if not path.is_file():
        return previous_version, list(previous_experiments), None
    payload = json.loads(path.read_text())
    version = payload.get("version")
    experiments = payload.get("experiments")
    if not isinstance(version, int) or not isinstance(experiments, list):
        raise ValueError("append registry requires integer version and experiments list")
    if version == previous_version:
        if experiments != list(previous_experiments):
            raise ValueError("append registry contents changed without a version increment")
        return version, list(experiments), sha(path)
    if version != previous_version + 1 or experiments[:len(previous_experiments)] != list(previous_experiments):
        raise ValueError("append registry must increase by one and preserve the prior experiment prefix")
    if any(not isinstance(name, str) or name not in ("SW0111", "SW0112", "SW0114")
           for name in experiments[len(previous_experiments):]):
        raise ValueError("append registry contains an unapproved experiment name")
    return version, list(experiments), sha(path)


def owners(gpu):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def memory(gpu):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(result.stdout.strip().splitlines()[0])


def external_gate(path, gate):
    """Check registered upstream state without imposing unrelated queue barriers."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return "waiting"
    if gate == "sw0106":
        if data.get("status") == "complete" and not data.get("active"):
            return "complete"
        if data.get("status") in ("failed", "error", "complete_with_branch_failures"):
            return "failed"
        return "waiting"
    if data.get("status") == "complete" and not data.get("active"):
        return "complete"
    if data.get("sw0107_status") == "complete":
        active = data.get("active", {}).values()
        if not any(str(row.get("task", {}).get("stage", "")).startswith("sw107-")
                   for row in active):
            return "complete"
    if data.get("status") in ("failed", "error", "complete_with_branch_failures"):
        return "failed"
    return "waiting"


def descendants(root_pid):
    result = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True)
    parents = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                parents[int(fields[0])] = int(fields[1])
            except ValueError:
                pass
    found, changed = {root_pid}, True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def stop_owned(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def execute(args):
    if fcntl is None:
        raise RuntimeError("SW0113 dispatch requires Linux fcntl GPU leases")
    if not handoff_valid(args.handoff108) or not handoff_valid(args.handoff109):
        raise RuntimeError("parent-verified SW0108 and SW0109 handoff records are required")
    if STATE.exists():
        raise FileExistsError(f"preserve existing SW0113 state: {STATE}")
    tasks = registered_tasks(args)
    external_dependencies = {
        "external_sw0107_terminal": (args.sw0107_state, "sw0107"),
        "external_sw0106_complete": (args.sw0106_state, "sw0106"),
    }
    append_path = Path(args.append_registry)
    append_version, appended_experiments, append_sha = read_append_manifest(
        append_path, 0, [])
    if appended_experiments:
        tasks.extend(appended_tasks(appended_experiments))
    by_id = {task["task_id"]: task for task in tasks}
    # Adopt only already complete, contract-valid outputs. Existing invalid or
    # partial outputs are immutable failures; never overwrite or retry them.
    complete, failed = set(), set()
    for task in tasks:
        out = Path(task["output"])
        if out.exists():
            if task["validate"]():
                complete.add(task["task_id"])
            else:
                failed.add(task["task_id"])
    for task in tasks:
        missing = set(task.get("depends_on", ())) - set(by_id) - set(external_dependencies)
        if missing:
            raise ValueError(f"unregistered dependencies for {task['task_id']}: {sorted(missing)}")

    lease_dir = Path(args.lease_dir)
    lease_dir.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "dispatcher_pid": os.getpid(), "tasks_total": len(tasks),
             "tasks": [{k: v for k, v in t.items() if k not in ("command", "validate")}
                       for t in tasks],
             "completed": sorted(complete), "failed": sorted(failed), "pending": [],
             "active": {}, "attempts": [], "eligible_gpus": list(GPUS),
             "max_used_mib": MAX_USED_MIB, "append_registry": str(append_path),
             "append_version": append_version, "append_experiments": appended_experiments,
             "append_registry_sha256": append_sha}
    workers, reservations = {}, set()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    write_json(STATE, state)

    def save():
        state["completed"] = sorted(complete)
        state["failed"] = sorted(failed)
        state["pending"] = [t["task_id"] for t in tasks
                             if t["task_id"] not in complete | failed
                             and t["task_id"] not in workers]
        state["active"] = {key: active_public(value) for key, value in workers.items()}
        write_json(STATE, state)

    try:
        while True:
            progress = False
            for dependency, (state_path, gate_name) in external_dependencies.items():
                result = external_gate(state_path, gate_name)
                if result == "complete" and dependency not in complete:
                    complete.add(dependency)
                    progress = True
                elif result == "failed" and dependency not in failed:
                    failed.add(dependency)
                    progress = True
            next_version, next_experiments, next_sha = read_append_manifest(
                append_path, append_version, appended_experiments)
            if next_version != append_version:
                additions = appended_tasks(next_experiments[len(appended_experiments):])
                existing_ids = set(by_id)
                new_ids = [t["task_id"] for t in additions]
                if len(new_ids) != len(set(new_ids)) or existing_ids.intersection(new_ids):
                    raise ValueError("append registry would duplicate an existing task ID")
                by_id.update({t["task_id"]: t for t in additions})
                for task in additions:
                    unknown = set(task.get("depends_on", ())) - set(by_id)
                    if unknown:
                        raise ValueError(f"appended task {task['task_id']} has unknown dependencies {sorted(unknown)}")
                    output = Path(task["output"])
                    if output.exists():
                        if task["validate"]():
                            complete.add(task["task_id"])
                        else:
                            raise FileExistsError(f"preserve invalid existing appended output {output}")
                    tasks.append(task)
                append_version, appended_experiments, append_sha = next_version, next_experiments, next_sha
                state.update({"tasks_total": len(tasks), "append_version": append_version,
                    "append_experiments": appended_experiments, "append_registry_sha256": append_sha})
                state["tasks"] = [{k: v for k, v in t.items() if k not in ("command", "validate")}
                                   for t in tasks]
                progress = True
            # Validate completed workers before releasing global GPU leases.
            for task_id, worker in list(workers.items()):
                proc, gpu = worker["proc"], worker["gpu"]
                rc = proc.poll()
                if rc is None:
                    foreign = [pid for pid in owners(gpu) if pid not in descendants(proc.pid)]
                    if foreign:
                        stop_owned(proc)
                        task = by_id[task_id]
                        failed.add(task_id)
                        state["attempts"].append({"task_id": task_id, "gpu": gpu,
                            "pid": proc.pid, "return_code": proc.poll(), "validated": False,
                            "log": worker["log"], "foreign_owners_during": foreign,
                            "failure": "foreign_owner_appeared", "ended_unix": time.time()})
                        workers.pop(task_id)
                        reservations.remove(gpu)
                        worker["lease_stream"].close()
                        progress = True
                        continue
                    continue
                task = by_id[task_id]
                passed = rc == 0 and task["validate"]()
                state["attempts"].append({"task_id": task_id, "gpu": gpu, "pid": proc.pid,
                    "return_code": rc, "validated": passed, "log": worker["log"],
                    "ended_unix": time.time()})
                if passed:
                    complete.add(task_id)
                else:
                    failed.add(task_id)
                workers.pop(task_id)
                reservations.remove(gpu)
                worker["lease_stream"].close()
                progress = True

            # Propagate dependency failures without suppressing unrelated branches.
            for task in tasks:
                tid = task["task_id"]
                if tid in complete or tid in failed:
                    continue
                if set(task.get("depends_on", ())) & failed:
                    failed.add(tid)
                    state.setdefault("blocked_by_failure", {})[tid] = sorted(set(task["depends_on"]) & failed)
                    progress = True

            ready = sorted(ready_tasks([t for t in tasks
                if t["task_id"] not in complete | failed | set(workers)], complete, failed),
                key=lambda t: (t.get("priority", 5), t["task_id"]))
            for task in ready:
                if not ready:
                    break
                gpu = None
                lease_stream = None
                for candidate in GPUS:
                    if candidate in reservations:
                        continue
                    lockpath = lease_dir / f"gpu{candidate}.lock"
                    stream = lockpath.open("a+")
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        stream.close()
                        continue
                    reservations.add(candidate)
                    # Cross-process lease plus immediate physical ownership recheck.
                    if owners(candidate) or memory(candidate) > MAX_USED_MIB:
                        reservations.remove(candidate)
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                        stream.close()
                        continue
                    gpu, lease_stream = candidate, stream
                    break
                if gpu is None:
                    break
                task_id = task["task_id"]
                output = Path(task["output"])
                if output.exists():
                    reservations.remove(gpu)
                    fcntl.flock(lease_stream.fileno(), fcntl.LOCK_UN)
                    lease_stream.close()
                    raise FileExistsError(f"preserve task output before launch: {output}")
                log = HERE / "logs" / f"{task_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", TRITON_CACHE_DIR=f"/tmp/kevinswk_sw0113_triton/gpu{gpu}")
                owners_before = {str(g): owners(g) for g in GPUS}
                command = list(task["command"])
                # Evaluators/training scripts run inside the one-visible-GPU namespace.
                for index, part in enumerate(command):
                    if part == "cuda:0":
                        command[index] = "cuda:0"
                with log.open("x", encoding="utf-8") as log_stream:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=log_stream,
                        stderr=subprocess.STDOUT, env=env, start_new_session=True)
                workers[task_id] = {"proc": proc, "pid": proc.pid, "gpu": gpu,
                    "task_id": task_id, "stage": task["stage"], "experiment": task["experiment"],
                    "log": str(log), "started_unix": time.time(), "owners_before": owners_before,
                    "lease_stream": lease_stream, "lease_path": str(lease_dir / f"gpu{gpu}.lock")}
                progress = True
                # With a serialized 108 screen, do not schedule its next task
                # until its predecessor validates; other branches remain parallel.
                if task["experiment"] == "SW0108":
                    break

            if len(complete | failed) == len(tasks) and not workers:
                state["status"] = "complete_with_branch_failures" if failed else "complete"
                state["finished_unix"] = time.time()
                save()
                return state
            if not progress:
                state["status"] = "waiting_for_free_gpu_or_dependency"
                save()
                time.sleep(max(2, min(args.poll_seconds, 5)))
            else:
                save()
    except Exception as exc:
        for tid, worker in list(workers.items()):
            proc = worker["proc"]
            if proc.poll() is None:
                stop_owned(proc)
            worker["lease_stream"].close()
            reservations.discard(worker["gpu"])
            failed.add(tid)
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_unix"] = time.time()
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--lease-dir", default="/tmp/kevinswk_sw0113_gpu_leases")
    parser.add_argument("--handoff108", type=Path, required=True)
    parser.add_argument("--handoff109", type=Path, required=True)
    parser.add_argument("--sw0107-state", type=Path, required=True)
    parser.add_argument("--sw0106-state", type=Path, required=True)
    parser.add_argument("--append-registry", default=str(HERE / "append_registry.json"))
    parser.add_argument("--source108", type=Path, required=True)
    parser.add_argument("--gamma108", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-gamma109", type=Path, required=True)
    parser.add_argument("--train-manifest109", type=Path, required=True)
    parser.add_argument("--val-gamma109", type=Path, required=True)
    parser.add_argument("--val-manifest109", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Readiness only. Use --run after reviewed safe handoff of prior coordinators.")
    if args.poll_seconds < 2:
        raise ValueError("poll-seconds must be >=2")
    result = execute(args)
    print(json.dumps({"status": result["status"], "completed": len(result["completed"]),
                      "failed": len(result["failed"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
