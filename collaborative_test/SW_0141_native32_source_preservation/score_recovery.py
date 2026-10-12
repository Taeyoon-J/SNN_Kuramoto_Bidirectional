"""CPU-only recovery for the completed SW0141 score stage.

This loads the frozen pilot source in memory and applies one exact repair:
consume the digest returned by the validated Slot contract, because the actual
legacy Slot protocol does not store a prediction_sha256 field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

EXPECTED_PILOT_SHA256 = "f6b483d0ce840272e0d1ca0c0f183a2502c4851b36900a5b4501f9eba034f0ce"
EXPECTED_FINGERPRINT = "062e47d41ec96ab3f690575a3dcb1ba6db35014e97ca600fbfebcbe5ae070aff"
EXPECTED_QUEUE_SHA256 = "73d565b3090101456c9aa70035cc222c212cb86ff95330eccd980c14eb121a69"
PILOT_REL = Path("collaborative_test/SW_0141_native32_source_preservation/source_preservation_pilot/pilot.py")
PATCH_CALL = "    contract32.validate_slot_protocol(slot_protocol, seed=SEED, prediction_path=slot_npz)"
PATCH_USE = '                        "npz_sha": slot_protocol["prediction_sha256"],'
FIX_CALL = "    slot_prediction_sha = contract32.validate_slot_protocol(slot_protocol, seed=SEED, prediction_path=slot_npz)"
FIX_USE = '                        "npz_sha": slot_prediction_sha,'
EXPECTED_TASKS = {
    "source_q_cache_s0", "source_q_cache_s1", "anchor_lambda_s0",
    "preflight_c_only_s1", "preflight_c_plus_anchor_s1",
    "train_c_only_s1", "train_c_plus_anchor_s1",
    "predict_c_only_s1", "predict_c_plus_anchor_s1", "score_seed1",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def patch_frozen_source(source: str) -> str:
    """Apply only the validated Slot digest return-value handoff."""
    if source.count(PATCH_CALL) != 1 or source.count(PATCH_USE) != 1:
        raise ValueError("frozen scorer no longer matches the reviewed one-line Slot digest defect")
    result = source.replace(PATCH_CALL, FIX_CALL, 1).replace(PATCH_USE, FIX_USE, 1)
    if result.count(FIX_CALL) != 1 or result.count(FIX_USE) != 1:
        raise AssertionError("in-memory score repair did not apply exactly once")
    return result


def validate_failed_attempt(state_path: Path, log_path: Path) -> tuple[dict, str, str]:
    state_bytes = state_path.read_bytes()
    log_bytes = log_path.read_bytes()
    state = json.loads(state_bytes.decode("utf-8-sig"))
    log = log_bytes.decode("utf-8", errors="replace")
    if state.get("status") != "failed_or_blocked" or state.get("pilot") != "SW0141_native32_source_preservation":
        raise ValueError("expected the preserved terminal SW0141 failure state")
    fingerprint = state.get("implementation_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("queue state lacks the implementation fingerprint map")
    actual_fp = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual_fp != EXPECTED_FINGERPRINT:
        raise ValueError("queue state implementation fingerprint is not the reviewed SW0141 build")
    if state.get("implementation_fingerprint_sha256", actual_fp) != actual_fp:
        raise ValueError("queue fingerprint digest is inconsistent")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != EXPECTED_TASKS or state.get("task_count") != 10:
        raise ValueError("failed attempt does not contain the expected ten-stage DAG")
    for name in sorted(EXPECTED_TASKS - {"score_seed1"}):
        row = tasks[name]
        if row.get("status") != "passed" or row.get("returncode") != 0 or row.get("artifact_valid") is not True:
            raise ValueError(f"preserved prerequisite task is not passed: {name}")
    score = tasks["score_seed1"]
    if score.get("status") != "failed" or score.get("returncode") != 1:
        raise ValueError("the score stage is not the expected terminal failure")
    if "KeyError: 'prediction_sha256'" not in log or 'pilot.py", line 1073' not in log:
        raise ValueError("score log does not contain the known pre-GT Slot protocol KeyError")
    if state.get("pilot_sha256") != EXPECTED_PILOT_SHA256:
        raise ValueError("failed run was produced by an unexpected pilot source")
    return state, hashlib.sha256(state_bytes).hexdigest(), hashlib.sha256(log_bytes).hexdigest()


def run_recovery(*, repo_root: Path, state_path: Path, failure_log: Path,
                 output: Path, receipt: Path) -> dict:
    repo_root = repo_root.resolve()
    state_path, failure_log = state_path.resolve(), failure_log.resolve()
    output, receipt = output.resolve(), receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError("preserve existing recovery output/receipt; no overwrite or rerun")
    state, state_sha, log_sha = validate_failed_attempt(state_path, failure_log)
    pilot_path = (repo_root / PILOT_REL).resolve()
    queue_path = pilot_path.parent / "source_preservation_queue.py"
    queue_sha = sha256(queue_path)
    if queue_sha != EXPECTED_QUEUE_SHA256:
        raise ValueError(f"frozen SW0141 queue SHA mismatch: {queue_sha}")
    pilot_bytes = pilot_path.read_bytes()
    original_sha = hashlib.sha256(pilot_bytes).hexdigest()
    if original_sha != EXPECTED_PILOT_SHA256:
        raise ValueError(f"frozen SW0141 pilot SHA mismatch: {original_sha}")
    patched_source = patch_frozen_source(pilot_bytes.decode("utf-8"))
    patched_sha = hashlib.sha256(patched_source.encode("utf-8")).hexdigest()

    # Import the exact frozen scorer while CUDA is hidden. __file__ remains the
    # actual repo file so its source/reference/assets resolve normally.
    os.environ["PATCH_V2_REPO"] = str(repo_root)
    module_name = "_sw141_cpu_score_recovery"
    namespace = {"__name__": module_name, "__file__": str(pilot_path)}
    exec(compile(patched_source, str(pilot_path), "exec"), namespace)
    loaded_fp = namespace["implementation_fingerprint"]()
    loaded_fp_sha = hashlib.sha256(json.dumps(loaded_fp, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if loaded_fp_sha != EXPECTED_FINGERPRINT:
        raise ValueError("loaded scorer implementation fingerprint changed")
    if namespace["torch"].cuda.is_initialized():
        raise RuntimeError("CPU-only recovery unexpectedly initialized CUDA before scoring")
    score_fn = namespace["score_pilot"]
    kwargs = {"prediction_root": pilot_path.parent / "results",
              "training_root": pilot_path.parent / "training",
              "output": output}
    result = score_fn(**kwargs)
    if namespace["torch"].cuda.is_initialized():
        raise RuntimeError("CPU-only recovery unexpectedly initialized CUDA during scoring")
    if result.get("status") != "complete_local_unregistered" or result.get("ground_truth_opened_after_all_prediction_hashes") is not True:
        raise RuntimeError("score did not finish the registered GT-last scoring contract")
    if not output.is_file():
        raise RuntimeError("frozen score function returned without writing its create-only result")
    # Confirm the preserved failed-run evidence was not rewritten while scoring.
    if sha256(state_path) != state_sha or sha256(failure_log) != log_sha:
        raise RuntimeError("original queue state or failure log changed during recovery")
    receipt_record = {
        "status": "score_recovery_complete",
        "pilot": "SW0141_native32_source_preservation",
        "original_queue_status": state["status"],
        "original_task_count": state["task_count"],
        "original_passed_prerequisites": 9,
        "original_score_failure": "KeyError: 'prediction_sha256' before GT access",
        "original_state_path": str(state_path), "original_state_sha256": state_sha,
        "original_failure_log_path": str(failure_log), "original_failure_log_sha256": log_sha,
        "original_queue_script_path": str(queue_path), "original_queue_script_sha256": queue_sha,
        "frozen_pilot_path": str(pilot_path), "frozen_pilot_sha256": original_sha,
        "loaded_implementation_fingerprint_sha256": loaded_fp_sha,
        "implementation_fingerprint_sha256": EXPECTED_FINGERPRINT,
        "recovery_patch": "use digest returned by contract32.validate_slot_protocol; actual Slot protocol has no prediction_sha256 key",
        "patched_in_memory_source_sha256": patched_sha,
        "output_path": str(output), "output_sha256": sha256(output),
        "five_prediction_artifacts": result.get("artifacts"),
        "result_status": result.get("status"),
        "ground_truth_opened_after_all_prediction_hashes": result.get("ground_truth_opened_after_all_prediction_hashes"),
        "training_cache_predictions_rerun": False,
        "original_failed_state_preserved": True,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(receipt_record, f, indent=2, allow_nan=False)
        f.write("\n")
    return receipt_record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_recovery(repo_root=args.repo_root, state_path=args.state,
                          failure_log=args.failure_log, output=args.output,
                          receipt=args.receipt)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
