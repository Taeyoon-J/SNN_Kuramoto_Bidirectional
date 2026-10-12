"""One-shot recovery adapter for the SW0143 trace-metadata preflight bug."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PATCH_V2_REPO", HERE.parents[1] / "patch_v2_sw")).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

CODE_V2 = ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/results_archive/recovery_code_v3_20261012/recover_queue.py"
LOCAL_V2 = HERE.parent / "SW0143_recovery_v2_20261012" / "recover_queue.py"
V2_HELPER_SHA256 = "b32c1aa5ce8417359aaab40bb9606e6fdee2c8d8314281f365616c24805dbe00"
OLD_RUN_SHA256 = "4f83687346e2fea78a16e694a3a22ed621e8cd41139cb585d27fdfc122ec276a"
NEW_RUN_SHA256 = "C004674618D3A714A3093BEAFD03B27F12B94CF3E16CE775FC36DA954DA7AA5D".lower()
RUN_REL = "collaborative_test/SW_0143_native32_selective_geodesic/run.py"
ARCHIVE = ROOT / "collaborative_test/SW_0143_native32_selective_geodesic/results_archive"
V2_STATE_PATH = ARCHIVE / "recovery_v2_20261012/recovery_state.json"
V2_LOG_PATH = ARCHIVE / "recovery_v2_20261012/logs/preflight_s2.attempt01.log"
V2_ROOT = ARCHIVE / "recovery_v2_20261012"
RECOVERY_ROOT = ARCHIVE / "recovery_v3_20261012"
CODE_ROOT = ARCHIVE / "recovery_code_v4_20261012"
EXPECTED_SUPERVISOR = 390000
EXPECTED_FAILED_CHILD = 400821
EXPECTED_ORIGINAL_STATE_SHA = "9ef6a919ed029f8b432e2ccebb419e9c01bea834809f1a8a85af23c6162178ce"
EXPECTED_V1_STATE_SHA = "3889eda3ecbec37b23a87d2007d0cc089cab6b4ca1cb0756534ab6800472d9bc"
EXPECTED_V1_HELPER_SHA = "4ad493180d63a89e44bb9f6a27cd79aaaf9e520907a339958d16b4cec38499ed"
EXPECTED_V2_QUEUE_SHA = "47dd7d94c7d1059dfb396ea2431925680b7dc357669e73f8b7305091f25ca6fe"
EXPECTED_V2_STATE_SHA = "b63e45f05def81d4c5feb6aab912fccaf19002b2f8614eef95335b50b92256a9"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_v2():
    path = CODE_V2 if CODE_V2.is_file() else LOCAL_V2
    if not path.is_file() or sha256_file(path) != V2_HELPER_SHA256:
        raise RuntimeError("reviewed stable recovery helper v2 is absent or changed")
    spec = importlib.util.spec_from_file_location("sw0143_stable_recovery_v2_frozen", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen v2 recovery adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
frozen = v2.frozen
run = v2.run


def _proc_alive(pid):
    return Path(f"/proc/{int(pid)}").exists()


def _proc_cmdlines():
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("repair admission requires process ownership inspection via /proc")
    rows = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError(f"cannot audit argv for pid {entry.name}") from exc
        if raw:
            rows.append((int(entry.name), raw.replace(b"\0", b" ").decode(errors="replace")))
    return rows


def _old_fingerprint_matches(recorded, current):
    if not isinstance(recorded, dict) or set(recorded) != set(current):
        return False
    for rel, actual in current.items():
        expected = OLD_RUN_SHA256 if rel == RUN_REL else actual
        if recorded.get(rel) != expected:
            return False
    return recorded.get(RUN_REL) == OLD_RUN_SHA256 and current.get(RUN_REL) == NEW_RUN_SHA256


def validate_failed_v2(state_bytes, state, log_text, *, pid_alive=_proc_alive,
                       proc_cmdlines=_proc_cmdlines, current_fingerprint=None,
                       state_sha256=None):
    """Validate the exact terminal v2 failure and the sole registered code delta."""
    actual_state_sha = hashlib.sha256(state_bytes).hexdigest()
    if state_sha256 is not None and actual_state_sha != state_sha256:
        raise ValueError("terminal v2 recovery state SHA differs from observed failure")
    if (state.get("experiment") != "SW0143_native32_selective_geodesic_recovery_v2"
            or state.get("status") != "failed"
            or state.get("supervisor_pid") != EXPECTED_SUPERVISOR
            or state.get("original_state_sha256") != EXPECTED_ORIGINAL_STATE_SHA
            or state.get("v1_state_sha256") != EXPECTED_V1_STATE_SHA
            or state.get("v1_helper_sha256") != EXPECTED_V1_HELPER_SHA
            or state.get("queue_sha256") != EXPECTED_V2_QUEUE_SHA
            or state.get("recovery_helper_sha256") != V2_HELPER_SHA256):
        raise ValueError("terminal v2 state is not the registered failed SW0143 recovery")
    current = current_fingerprint or run.implementation_fingerprint()
    if not _old_fingerprint_matches(state.get("implementation_fingerprint"), current):
        raise ValueError("frozen SW0143 dependency fingerprint changed beyond the run.py repair")
    tasks = state.get("tasks")
    plan = frozen.task_plan()
    if not isinstance(tasks, dict) or set(tasks) != {task["task_id"] for task in plan}:
        raise ValueError("terminal v2 task set differs from the frozen 13-stage DAG")
    failed = tasks["preflight_s2"]
    if (failed.get("status") != "failed" or failed.get("child_pid") != EXPECTED_FAILED_CHILD
            or failed.get("returncode") != 1 or failed.get("artifact_valid") is not False
            or str(failed.get("log_path", "")).replace("\\", "/").endswith(
                "/recovery_v2_20261012/logs/preflight_s2.attempt01.log") is False
            or "TypeError: 'int' object is not subscriptable" not in log_text
            or "run.py\", line 191" not in log_text):
        raise ValueError("preflight_s2 failure does not match the known trace-metadata TypeError")
    for seed in (0, 1):
        row = tasks[f"preflight_s{seed}"]
        if (row.get("status") != "failed"
                or "recovery cancelled while waiting for a stable exclusive GPU" not in row.get("error", "")):
            raise ValueError(f"preflight_s{seed} is not the expected fail-fast cancellation")
    for task_id, row in tasks.items():
        pid = row.get("child_pid")
        if pid and pid_alive(pid):
            raise RuntimeError(f"old v2 child still running: {task_id} pid={pid}")
    if pid_alive(EXPECTED_SUPERVISOR) or pid_alive(EXPECTED_FAILED_CHILD):
        raise RuntimeError("old v2 supervisor or failed child remains alive")
    for pid, argv in proc_cmdlines():
        if pid == os.getpid():
            continue
        normalized = str(argv).replace("\\", "/").lower()
        if ("/recovery_code_v3_20261012/recover_queue.py" in normalized
                or "/sw_0143_native32_selective_geodesic/run.py" in normalized):
            raise RuntimeError(f"SW0143 scientific/recovery process still live pid={pid}")
    return {"v2_state_sha256": actual_state_sha,
            "prior_implementation_fingerprint": state["implementation_fingerprint"],
            "current_implementation_fingerprint": current,
            "run_py_previous_sha256": OLD_RUN_SHA256,
            "run_py_repaired_sha256": NEW_RUN_SHA256}


def prepare(*, state_path=V2_STATE_PATH, log_path=V2_LOG_PATH,
            recovery_root=RECOVERY_ROOT, pid_alive=_proc_alive,
            proc_cmdlines=_proc_cmdlines, state_sha256=EXPECTED_V2_STATE_SHA):
    if Path(recovery_root).exists():
        raise FileExistsError(f"preserve existing SW0143 repair output: {recovery_root}")
    state_bytes = Path(state_path).read_bytes()
    actual_sha = hashlib.sha256(state_bytes).hexdigest()
    if state_sha256 is not None and actual_sha != state_sha256:
        raise ValueError("terminal v2 state changed after read-only observation")
    state = json.loads(state_bytes.decode("utf-8"))
    log_text = Path(log_path).read_text(encoding="utf-8")
    lineage = validate_failed_v2(state_bytes, state, log_text, pid_alive=pid_alive,
                                 proc_cmdlines=proc_cmdlines, state_sha256=actual_sha)
    tasks = frozen.task_plan()
    present = [str(frozen.output_path(task)) for task in tasks if frozen.output_path(task).exists()]
    if present:
        raise FileExistsError("preserve any output from failed v2; unexpected artifacts: " + ", ".join(present))
    return state, state_bytes, lineage, {"tasks": tasks, "preserved": [],
                                         "missing": [task["task_id"] for task in tasks]}


class RepairQueue(v2.StableRecoveryQueue):
    def __init__(self, lineage, plan, *, recovery_root=RECOVERY_ROOT):
        if Path(recovery_root).exists():
            raise FileExistsError(f"preserve existing SW0143 v3 repair path: {recovery_root}")
        v2.V2_ROOT = Path(recovery_root)
        super().__init__(lineage, plan)
        self.state.update({"experiment": "SW0143_native32_selective_geodesic_recovery_v3",
                           "previous_recovery": "recovery_v2_20261012",
                           "recovery_reason": "known mixed tensor/scalar trace slicing TypeError",
                           "repair_helper_sha256": sha256_file(Path(__file__)),
                           "repaired_run_sha256": NEW_RUN_SHA256})
        self._save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state, state_bytes, lineage, plan = prepare()
    if args.dry_run:
        print(json.dumps({"status": "repair_prechecked", **lineage,
                          "missing": plan["missing"], "preserved": plan["preserved"],
                          "max_parallel": frozen.MAX_PARALLEL,
                          "quiet_window_seconds": v2.STABLE_IDLE_SECONDS}, indent=2), flush=True)
        return 0
    return RepairQueue(lineage, plan).run()


if __name__ == "__main__":
    raise SystemExit(main())
