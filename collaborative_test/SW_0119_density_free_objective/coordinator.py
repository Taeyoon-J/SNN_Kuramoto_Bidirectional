"""Seed-0 SW0119 task adapter for SW0113's exclusive GPU dispatcher."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0119_density_free_objective"
RUNNER = HERE / "run.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0119_density_free_objective import run


def task_plan():
    validate = {"experiment": "SW0119", "task_id": "sw0119_validate_sw0117_control_s0",
                "stage": "validate-control", "seed": 0, "arm": run.ARM,
                "depends_on": [], "priority": 0}
    preflight = {"experiment": "SW0119", "task_id": "sw0119_preflight_s0_density_free",
                 "stage": "preflight", "seed": 0, "arm": run.ARM,
                 "depends_on": [validate["task_id"]], "priority": 0}
    train = {"experiment": "SW0119", "task_id": "sw0119_train_s0_density_free",
             "stage": "train", "seed": 0, "arm": run.ARM,
             "depends_on": [preflight["task_id"]], "priority": 1}
    evaluate = {"experiment": "SW0119", "task_id": "sw0119_eval_s0_density_free",
                "stage": "evaluate", "seed": 0, "arm": run.ARM,
                "depends_on": [train["task_id"]], "priority": 1}
    return [validate, preflight, train, evaluate]


def artifact_path(task):
    stage = task["stage"]
    if stage == "validate-control":
        return ARCHIVE / "sw0117_control_validation.json"
    if stage == "preflight":
        return ARCHIVE / "preflight_seed0_density_free.json"
    folder = OUT / "seed0_density_free_candidate"
    return folder / ("manifest.json" if stage == "train" else "evaluation.json")


def command(task, device="cuda"):
    output = artifact_path(task)
    if task["stage"] == "train":
        output = output.parent
    args = [sys.executable, str(RUNNER), task["stage"], "--output", str(output),
            "--device", device]
    if task["stage"] == "evaluate":
        args += ["--checkpoint", str(OUT / "seed0_density_free_candidate/core.pt")]
    return args


def _finite_score(score):
    names = {"fg_ari", "foreground_iou", "matched_object_iou"}
    if set(score.get("metrics", {})) != names or set(score.get("valid_count", {})) != names \
            or set(score.get("per_image", {})) != names:
        return False
    try:
        return all(
            score["valid_count"][name] == 320
            and len(score["per_image"][name]) == 320
            and all(math.isfinite(float(value)) for value in score["per_image"][name])
            and math.isfinite(float(score["metrics"][name]))
            and abs(sum(float(value) for value in score["per_image"][name]) / 320.0
                     - float(score["metrics"][name])) <= 1e-12
            for name in names)
    except (KeyError, TypeError, ValueError):
        return False


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if task["stage"] == "validate-control":
            return (record.get("status") == "passed" and record.get("experiment") == "SW0119"
                    and record.get("seed") == 0 and record.get("arm") == run.ARM
                    and record.get("lambda_joint") == run.LAMBDA
                    and record.get("control_manifest_sha256") == run.sha(
                        run.sw117.OUT / "seed0_analytic_candidate/manifest.json")
                    and record.get("control_evaluation_sha256") == run.sha(
                        run.sw117.OUT / "seed0_analytic_candidate/evaluation.json"))
        if task["stage"] == "preflight":
            if not (record.get("status") == "passed" and record.get("experiment") == "SW0119"
                    and record.get("seed") == 0 and record.get("arm") == run.ARM
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("lambda_joint") == run.LAMBDA
                    and record.get("lambda_sha256") == run.sha(run.sw117.LAMBDA_PATH)
                    and record.get("ground_truth_used") is False
                    and len(record.get("batches", [])) == 4
                    and record.get("throwaway_parameter_groups_changed") == {
                        "graph": True, "core": True, "encoder": True}
                    and record.get("throwaway_b16_adam_update") is True
                    and len(record.get("training_ids", [])) == 4096
                    and record.get("matched_shuffle_seed") == 117
                    and float(record.get("scrambled_minus_real_rgb_loss", float("nan"))) > 0):
                return False
            for row in record["batches"]:
                if (len(row.get("global_ids", [])) != 16
                        or row.get("cached_labels_H_exact") is not True
                        or row.get("same_input_reference_identity_exact") is not True
                        or not math.isfinite(float(row.get("cache_gamma_max_abs_diff", float("nan"))))
                        or float(row["cache_gamma_max_abs_diff"]) > 2e-5
                        or not math.isfinite(float(row.get("cached_baseline_old_loss_abs_diff", float("nan"))))
                        or float(row["cached_baseline_old_loss_abs_diff"]) > 2e-5
                        or not math.isfinite(float(row.get("legacy_minus_density_free", float("nan"))))
                        or not math.isfinite(float(row.get("removed_balance_terms", float("nan"))))
                        or abs(float(row["legacy_minus_density_free"])
                               - float(row["removed_balance_terms"]))
                           > 2e-6 + 1e-6 * abs(float(row["removed_balance_terms"]))):
                    return False
                for family in ("encoder", "graph", "core", "joint"):
                    for group in ("old_gradient_norms", "rgb_gradient_norms"):
                        norm = float(row.get(group, {}).get(family, float("nan")))
                        if not math.isfinite(norm) or norm <= 0:
                            return False
                    if family != "joint":
                        diff = float(row.get("removed_balance_gradient_decomposition", {})
                                     .get(family, {}).get("relative_max_abs_difference", float("nan")))
                        if not math.isfinite(diff) or diff > 1e-5:
                            return False
                qnorm = float(row.get("rgb_to_Q_gradient_norm", float("nan")))
                if not math.isfinite(qnorm) or qnorm <= 0:
                    return False
        update_norm = float(record.get("throwaway_gradient_norm_preclip", float("nan")))
        if not math.isfinite(update_norm) or update_norm <= 0:
            return False
        source_sha = record.get("source_core_sha256", "")
        if (len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha)
                or len(record.get("training_ids", [])) != 4096):
            return False
        return True
        if task["stage"] == "train":
            folder = path.parent
            preflight = ARCHIVE / "preflight_seed0_density_free.json"
            history = folder / "history.json"
            rows = json.loads(history.read_text(encoding="utf-8"))
            return (record.get("status") == "training_complete"
                    and record.get("experiment") == "SW0119" and record.get("seed") == 0
                    and record.get("arm") == run.ARM and record.get("updates") == 256
                    and record.get("batch_size") == 16 and record.get("matched_shuffle_seed") == 117
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("preflight_sha256") == run.sha(preflight)
                    and record.get("source_core_sha256") == json.loads(preflight.read_text()).get("source_core_sha256")
                    and record.get("source_manifest_sha256") == json.loads(preflight.read_text()).get("source_manifest_sha256")
                    and record.get("training_ids_sha256") == json.loads(preflight.read_text()).get("training_ids_sha256")
                    and record.get("lambda_joint") == run.LAMBDA
                    and record.get("lambda_artifact_sha256") == run.sha(run.sw117.LAMBDA_PATH)
                    and record.get("core_sha256") == run.sha(folder / "core.pt")
                    and record.get("encoder_sha256") == run.sha(folder / "encoder.pt")
                    and record.get("optimizer_sha256") == run.sha(folder / "optimizer.pt")
                    and record.get("history_sha256") == run.sha(history)
                    and (folder / "TRAINING_COMPLETED").is_file()
                    and len(rows) == 256
                    and [row.get("update") for row in rows] == list(range(1, 257))
                    and all(math.isfinite(float(row[key])) for row in rows
                            for key in ("total", "density_free_old", "primary",
                                        "positive_actual_spike_product", "rgb",
                                        "gradient_norm_preclip")))
        if task["stage"] == "evaluate":
            folder = path.parent
            manifest_path = folder / "manifest.json"
            core_path, encoder_path = folder / "core.pt", folder / "encoder.pt"
            sidecar_path = folder / "evaluation_manifest.json"
            if not sidecar_path.is_file():
                return False
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            expected_contract = {"ids": [1320, 1639], "images": 320, "batch_size": 8,
                "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
                "readout_threshold": 0.50, "min_group_size": 2,
                "background": "largest_component", "ground_truth_used_for_prediction": False}
            return (record.get("experiment") == "SW0119" and record.get("seed") == 0
                    and record.get("arm") == run.ARM
                    and record.get("ground_truth_used_for_prediction") is False
                    and record.get("evaluation_runner_sha256") == run.sha(RUNNER)
                    and record.get("shared_evaluator_sha256") == run.sha(
                        run.ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py")
                    and record.get("training_manifest_sha256") == run.sha(manifest_path)
                    and record.get("checkpoint_sha256") == run.sha(core_path)
                    and record.get("encoder_checkpoint_sha256") == run.sha(encoder_path)
                    and sidecar.get("experiment") == "SW0119"
                    and sidecar.get("seed") == 0 and sidecar.get("arm") == run.ARM
                    and sidecar.get("evaluation_sha256") == run.sha(path)
                    and sidecar.get("training_manifest_sha256") == run.sha(manifest_path)
                    and sidecar.get("checkpoint_sha256") == run.sha(core_path)
                    and sidecar.get("encoder_checkpoint_sha256") == run.sha(encoder_path)
                    and sidecar.get("evaluation_runner_sha256") == record.get("evaluation_runner_sha256")
                    and sidecar.get("shared_evaluator_sha256") == record.get("shared_evaluator_sha256")
                    and sidecar.get("evaluation_contract") == expected_contract
                    and _finite_score(record["sweep"][0]["scored_targets"]["our_hdf5"]))
        return False
    except (OSError, KeyError, IndexError, TypeError, ValueError, AssertionError):
        return False


if __name__ == "__main__":
    raise SystemExit("SW0119 tasks are dispatched by the reviewed SW0113 queue.")
