"""Create-only SW0163 recovery: adopt the two valid arms, train only two missing arms.

The original four-arm queue and its terminal failure receipt remain immutable.
This recovery namespace is narrowly bound to that receipt and the two audited
512-update artifacts.  Parent-side Torch model validation is serialized because
its legacy RNG isolation uses process-global state and is not thread-safe.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

from collaborative_test.SW_0163_relation_context_assignment import training_queue as base
from collaborative_test.SW_0163_relation_context_assignment import train

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "results_archive" / "recovery_training_queue_20261014"
FAILURE_RECEIPT = HERE / "results_archive" / "terminal_parent_validation_failure_20261014" / "server_records.json"
EXPECTED_FAILURE_RECEIPT_SHA256 = "70d73268155e021234d63a28218f5f0eab98cd81d72955c3cc19ffd8eba55976"
OLD_TRAINING_STATE_SHA256 = "719c4d763c4625d295099a0381eda0d296a04f1bff14f526ab0922431885f03d"
OLD_PREDICTION_STATE_SHA256 = "8bc78099fd5dbd4cbe2f20ad37380204984fb0b847c981cb7517e85da58ff746"
OLD_SCORE_STATE_SHA256 = "f5e4e06235a3895e24662c4a3e896f26d2a3acaf099b011bbc4ae892adf215d6"
ADOPTED_MANIFESTS = {
    "Q_only": "5784f30b65046336b6c60b5351ea4261718cf6290c2de8f55772d9cbb8f12e68",
    "A_only": "8dc2fdcb7b5cfecfe46e4d0df4bacbae313d213fd6172ef72e43ae4462a8affb",
}
EXPECTED_INITIAL_MODEL_SHA256 = "c9baacaa239b9ae41d913adc5ed5bee8f505df213c8b4580b24dd89ff9d15470"
ADOPTED = ("Q_only", "A_only")
TRAIN_ONLY = ("raw_only", "Q_A")
_PARENT_VALIDATION_LOCK = threading.RLock()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recovery_fingerprint():
    return {
        "recovery_training_queue.py": sha256_file(HERE / "recovery_training_queue.py"),
        "base_training_queue.py": sha256_file(HERE / "training_queue.py"),
        "train.py": sha256_file(HERE / "train.py"),
        "model.py": sha256_file(HERE / "model.py"),
        "protocol.json": sha256_file(HERE / "protocol.json"),
        "failure_snapshot_sha256": EXPECTED_FAILURE_RECEIPT_SHA256,
        "old_training_state_sha256": OLD_TRAINING_STATE_SHA256,
        "old_prediction_state_sha256": OLD_PREDICTION_STATE_SHA256,
        "old_score_state_sha256": OLD_SCORE_STATE_SHA256,
        "adopted_manifests": dict(ADOPTED_MANIFESTS),
    }


def verify_recovery_fingerprint(expected):
    if recovery_fingerprint() != expected:
        raise RuntimeError("SW0163 recovery code or pinned evidence changed after registration")


def validate_failure_snapshot(*, receipt_path=FAILURE_RECEIPT, base_dir=HERE,
                              process_live=None, acquire_locks=False, require_live_states=True):
    """Validate the exact known failure and prove its old owners no longer run."""
    receipt_path = Path(receipt_path)
    if not receipt_path.is_file() or sha256_file(receipt_path) != EXPECTED_FAILURE_RECEIPT_SHA256:
        raise ValueError("SW0163 recovery failure snapshot is absent or changed")
    packet = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    queues = packet.get("queues", {})
    expected = {
        "training_queue_20261014": (OLD_TRAINING_STATE_SHA256, "failed_training"),
        "prediction_queue_20261014": (OLD_PREDICTION_STATE_SHA256, "failed"),
        "score_queue_20261014": (OLD_SCORE_STATE_SHA256, "failed"),
    }
    for key, (pinned_sha, status) in expected.items():
        item = queues.get(key)
        if not isinstance(item, dict) or item.get("state_sha256") != pinned_sha:
            raise ValueError(f"SW0163 old {key} snapshot SHA differs")
        state = item.get("state")
        if not isinstance(state, dict) or state.get("status") != status:
            raise ValueError(f"SW0163 old {key} is not the pinned terminal failure")
        if require_live_states:
            live_state = Path(base_dir) / "results_archive" / key / "state.json"
            if not live_state.is_file() or sha256_file(live_state) != pinned_sha:
                raise ValueError(f"SW0163 live old {key} state is not the captured failed state")
    if queues["prediction_queue_20261014"]["state"].get("error") != \
            "ValueError('SW0163 training prerequisite terminal: failed_training')":
        raise ValueError("SW0163 old prediction failure cause differs from captured dependency failure")
    if queues["score_queue_20261014"]["state"].get("error") != \
            "ValueError('prediction prerequisite terminal: failed')":
        raise ValueError("SW0163 old score failure cause differs from captured prediction dependency failure")
    train_state = queues["training_queue_20261014"]["state"]
    if train_state.get("experiment") != base.EXPERIMENT:
        raise ValueError("SW0163 old training failure belongs to another experiment")
    arms = train_state.get("arms", {})
    for arm in ADOPTED:
        captured = arms.get(arm, {})
        if captured.get("status") != "failed" or captured.get("error") != \
                "ValueError('SW0163 training manifest/source/order binding is invalid')":
            raise ValueError(f"SW0163 expected captured {arm} parent-validation failure")
    for arm in TRAIN_ONLY:
        captured = arms.get(arm, {})
        if captured.get("status") != "failed" or captured.get("error") != \
                "RuntimeError('queue stopped while waiting for an exclusive GPU')":
            raise ValueError(f"SW0163 expected captured missing-arm terminal failure for {arm}")
    pids = {train_state.get("supervisor_pid")}
    for key in ("prediction_queue_20261014", "score_queue_20261014"):
        pids.add(queues[key]["state"].get("supervisor_pid"))
    pids.update(arms.get(arm, {}).get("child_pid") for arm in train.ARMS)
    pids.discard(None)
    live_check = process_live or _pid_live
    for pid in pids:
        if live_check(pid):
            raise RuntimeError(f"pinned SW0163 failed queue process is still live: {pid}")
    if acquire_locks:
        _prove_old_locks_free(Path(base_dir))
    old_logs = {}
    for name in expected:
        runtime = Path(base_dir) / "results_archive" / name
        old_logs[name] = {log.name: sha256_file(log) for log in sorted(runtime.glob("*.log")) if log.is_file()}
    return {"snapshot_sha256": EXPECTED_FAILURE_RECEIPT_SHA256,
            "old_training_state_sha256": OLD_TRAINING_STATE_SHA256,
            "old_prediction_state_sha256": OLD_PREDICTION_STATE_SHA256,
            "old_score_state_sha256": OLD_SCORE_STATE_SHA256,
            "dead_pids": sorted(int(pid) for pid in pids),
            "old_log_sha256": old_logs,
            "captured_failure_statuses": {key: value[1] for key, value in expected.items()}}


def _pid_live(pid, *, proc_root="/proc"):
    modules = ("collaborative_test.SW_0163_relation_context_assignment.training_queue",
               "collaborative_test.SW_0163_relation_context_assignment.prediction_queue",
               "collaborative_test.SW_0163_relation_context_assignment.score_queue",
               "collaborative_test.SW_0163_relation_context_assignment.train",
               "collaborative_test.SW_0163_relation_context_assignment.recovery_training_queue",
               "collaborative_test.SW_0163_relation_context_assignment.recovery_prediction_queue",
               "collaborative_test.SW_0163_relation_context_assignment.recovery_score_queue")
    return any(process_module_live(pid, module, proc_root=proc_root) for module in modules)


def process_module_live(pid, module, *, proc_root="/proc"):
    """Strict /proc command-line predicate, separately testable on CPU fixtures."""
    if not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path(proc_root) / str(pid)
    try:
        args = [value.decode("utf-8", errors="replace") for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
        return proc.exists() and "-m" in args and module in args
    except OSError:
        return False


def sys_platform_linux():
    import sys
    return sys.platform.startswith("linux")


def _prove_old_locks_free(base_dir):
    if base.fcntl is None:
        raise RuntimeError("SW0163 recovery requires Linux flock")
    roots = ("training_queue_20261014", "prediction_queue_20261014", "score_queue_20261014")
    streams = []
    try:
        for name in roots:
            lock = Path(base_dir) / "results_archive" / name / "queue.lock"
            if not lock.is_file():
                raise ValueError(f"old SW0163 queue lock is missing: {lock}")
            stream = lock.open("a+")
            streams.append(stream)
            try:
                base.fcntl.flock(stream.fileno(), base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"old SW0163 queue still owns its lock: {name}") from exc
        return True
    finally:
        for stream in reversed(streams):
            try:
                base.fcntl.flock(stream.fileno(), base.fcntl.LOCK_UN)
            finally:
                stream.close()


def validate_adoptable_arms(prepared, *, train_root=None):
    """Sequentially validate and pin only Q_only/A_only; reject any missing-arm output."""
    root = Path(train_root or train.TRAIN_ROOT)
    for arm in TRAIN_ONLY:
        if (root / f"{arm}_seed1").exists():
            raise FileExistsError(f"preserving unexpected partial {arm} artifact")
    validated = {}
    with _PARENT_VALIDATION_LOCK:
        for arm in ADOPTED:
            out = root / f"{arm}_seed1"
            manifest = out / "manifest.json"
            if not manifest.is_file() or sha256_file(manifest) != ADOPTED_MANIFESTS[arm]:
                raise ValueError(f"SW0163 pinned {arm} manifest is absent or changed")
            result = train.validate_training_artifact(arm, directory=out, prepared=prepared)
            if result.get("completed_updates") != train.UPDATES:
                raise ValueError(f"SW0163 pinned {arm} is not a full completed run")
            validated[arm] = {"manifest_sha256": sha256_file(manifest),
                              "checkpoint_sha256": result.get("checkpoint_sha256"),
                              "completed_updates": result["completed_updates"],
                              "initial_model_state_sha256": result.get("initial_model_state_sha256")}
    if validated["Q_only"]["initial_model_state_sha256"] != validated["A_only"]["initial_model_state_sha256"]:
        raise ValueError("SW0163 adopted arms do not share the audited initialization")
    if validated["Q_only"]["initial_model_state_sha256"] != EXPECTED_INITIAL_MODEL_SHA256:
        raise ValueError("SW0163 adopted arms differ from the source-audited initialization")
    return validated


def _serialize_parent_validation():
    """Serialize model-construction validations invoked by base queue worker threads."""
    original_validate = train.validate_training_artifact
    original_prepare = train.prepare_training_inputs_cpu

    def locked_validate(*args, **kwargs):
        with _PARENT_VALIDATION_LOCK:
            return original_validate(*args, **kwargs)

    def locked_prepare(*args, **kwargs):
        with _PARENT_VALIDATION_LOCK:
            return original_prepare(*args, **kwargs)

    train.validate_training_artifact = locked_validate
    train.prepare_training_inputs_cpu = locked_prepare
    return original_validate, original_prepare


def run_recovery(*, runtime=RUNTIME, process_live=None, acquire_old_locks=True,
                 queue_factory=None):
    """Adopt Q/A, run raw/Q_A only, and publish a separate recovery receipt/state."""
    runtime = Path(runtime)
    q = (queue_factory or base.TrainingQueue)(runtime=runtime)
    fp = recovery_fingerprint()
    q._update("validating_recovery", recovery_fingerprint=fp, original_failure_snapshot_sha256=EXPECTED_FAILURE_RECEIPT_SHA256,
              adopted_arms_expected=list(ADOPTED), arms_to_train=list(TRAIN_ONLY))
    old_validate, old_prepare = _serialize_parent_validation()
    try:
        q._verify_code()
        verify_recovery_fingerprint(fp)
        evidence = validate_failure_snapshot(process_live=process_live, acquire_locks=acquire_old_locks)
        dependency = base.preflight_dependency()
        if dependency is None:
            raise RuntimeError("SW0163 admitted preflight is not complete yet")
        prepared = dependency["prepared"]
        adopted = validate_adoptable_arms(prepared)
        q._update("recovery_admitted", recovery_fingerprint=fp,
                  original_failure_snapshot=evidence, adopted_arms=adopted,
                  arms_to_train=list(TRAIN_ONLY), preflight_report_sha256=dependency["report_sha256"])
        for arm in ADOPTED:
            # Revalidate once more under the parent lock before marking an arm adopted.
            result = train.validate_training_artifact(arm, directory=train.TRAIN_ROOT / f"{arm}_seed1",
                                                       prepared=prepared)
            q._arm_update(arm, "adopted_complete", manifest_sha256=ADOPTED_MANIFESTS[arm],
                          completed_updates=result["completed_updates"], recovery_source="pinned_terminal_run")
        q._update("training_arms_queued", recovery_fingerprint=fp,
                  recovery_train_arms=list(TRAIN_ONLY), adopted_arms=adopted,
                  original_failure_snapshot=evidence)
        results = {arm: True for arm in ADOPTED}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=2) as pool:
            def run_missing(arm):
                verify_recovery_fingerprint(fp)
                return q._run_arm(arm, prepared)
            futures = {pool.submit(run_missing, arm): arm for arm in TRAIN_ONLY}
            for future in as_completed(futures):
                arm = futures[future]
                try:
                    results[arm] = bool(future.result())
                except BaseException as exc:
                    results[arm] = False
                    q._arm_update(arm, "failed_recovery", error=repr(exc), artifacts_preserved=True)
                    q.stop_event.set()
        verify_recovery_fingerprint(fp)
        if all(results.values()):
            # Final parent-side validation is deliberately sequential.
            final = {}
            for arm in train.ARMS:
                result = train.validate_training_artifact(arm, prepared=prepared)
                final[arm] = {"manifest_sha256": sha256_file(train.TRAIN_ROOT / f"{arm}_seed1" / "manifest.json"),
                              "completed_updates": result["completed_updates"],
                              "initial_model_state_sha256": result.get("initial_model_state_sha256")}
            if len({item["initial_model_state_sha256"] for item in final.values()}) != 1 or \
                    final["raw_only"]["initial_model_state_sha256"] != EXPECTED_INITIAL_MODEL_SHA256:
                raise ValueError("SW0163 recovered arms do not share the registered initialization")
            final_evidence = validate_failure_snapshot(process_live=process_live,
                acquire_locks=acquire_old_locks)
            if final_evidence != evidence:
                raise ValueError("SW0163 original terminal failure evidence/logs changed during recovery")
            q._update("complete", recovery_status="complete_recovered", results=results, recovery_fingerprint=fp,
                      recovery_evidence=evidence, adopted_arms=adopted,
                      final_arms=final, preflight_report_sha256=dependency["report_sha256"],
                      finished=time.time(), artifacts_preserved=True)
            return 0
        q._update("failed_recovery", results=results, recovery_fingerprint=fp,
                  recovery_evidence=evidence, adopted_arms=adopted,
                  finished=time.time(), artifacts_preserved=True)
        return 1
    except BaseException as exc:
        q._update("failed_recovery", recovery_fingerprint=fp, error=repr(exc),
                  original_failure_snapshot_sha256=EXPECTED_FAILURE_RECEIPT_SHA256,
                  finished=time.time(), artifacts_preserved=True)
        return 2
    finally:
        train.validate_training_artifact = old_validate
        train.prepare_training_inputs_cpu = old_prepare
        q.stop_event.set()
        for arm, proc in list(q.children.items()):
            if proc.poll() is None:
                base.owner._stop_owned(proc)
        for arm in list(q.leases):
            q._release_lease(arm)
        q._unlock()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return run_recovery()


if __name__ == "__main__":
    raise SystemExit(main())
