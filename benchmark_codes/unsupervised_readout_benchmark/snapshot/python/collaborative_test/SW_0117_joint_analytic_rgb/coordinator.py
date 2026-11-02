"""Seed-0 paired-task adapter for the SW0113 central dispatcher."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0117_joint_analytic_rgb.run import (
    ARMS, ARCHIVE, OUT, RUNNER, SOURCE_ENDPOINT_REVIEW, implementation_fingerprint,
    sha, validate_source_endpoint_review,
)

SOURCE_ENDPOINT_REVIEW_ID = "sw0117_source_endpoint_reviewed"


def task_plan():
    tasks = []
    for arm in ARMS:
        pre = {"experiment": "SW0117", "task_id": f"sw0117_preflight_s0_{arm}",
               "stage": "preflight", "seed": 0, "arm": arm,
               "depends_on": [], "priority": 0}
        train = {"experiment": "SW0117", "task_id": f"sw0117_train_s0_{arm}",
                 "stage": "train", "seed": 0, "arm": arm,
                 "depends_on": [pre["task_id"], SOURCE_ENDPOINT_REVIEW_ID], "priority": 1}
        evaluate = {"experiment": "SW0117", "task_id": f"sw0117_eval_s0_{arm}",
                    "stage": "evaluate", "seed": 0, "arm": arm,
                    "depends_on": [train["task_id"]], "priority": 1}
        tasks.extend((pre, train, evaluate))
    return tasks


def artifact_path(task):
    if task["stage"] == "preflight":
        return ARCHIVE / f"preflight_seed0_{task['arm']}.json"
    folder = OUT / f"seed0_{task['arm']}"
    return folder / ("manifest.json" if task["stage"] == "train" else "evaluation.json")


def command(task, device="cuda"):
    cmd = [sys.executable, str(RUNNER), task["stage"], "--seed", "0",
           "--arm", task["arm"], "--device", device]
    if task["stage"] in ("preflight", "train"):
        output = artifact_path(task)
        if task["stage"] == "train":
            output = output.parent
        return cmd + ["--output", str(output)]
    folder = OUT / f"seed0_{task['arm']}"
    return cmd + ["--checkpoint", str(folder / "core.pt"),
                  "--output", str(artifact_path(task))]


def _finite_score(score):
    metrics = ("fg_ari", "foreground_iou", "matched_object_iou")
    if set(score.get("metrics", {})) != set(metrics):
        return False
    if set(score.get("valid_count", {})) != set(metrics) or set(score.get("per_image", {})) != set(metrics):
        return False
    try:
        for name in metrics:
            values = score["per_image"][name]
            mean = float(score["metrics"][name])
            if (len(values) != 320 or score["valid_count"][name] != 320
                    or not all(math.isfinite(float(v)) for v in values)
                    or not math.isfinite(mean)
                    or abs(sum(float(v) for v in values) / 320.0 - mean) > 1e-12):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
        if task["stage"] == "preflight":
            if (record.get("status") != "passed" or record.get("experiment") != "SW0117"
                    or record.get("seed") != 0 or record.get("arm") != task["arm"]
                    or record.get("implementation_fingerprint") != implementation_fingerprint()
                    or record.get("ground_truth_used") is not False
                    or len(record.get("batches", [])) != 4
                    or record.get("throwaway_b16_adam_update") is not True
                    or record.get("throwaway_parameter_groups_changed") != {
                        "encoder": True, "graph": True, "core": True}
                    or record.get("same_input_reference_identity_exact") is not True
                    or len(record.get("training_ids", [])) != 4096
                    or record.get("matched_shuffle_seed") != 117
                    or not record.get("source_core_sha256")
                    or not record.get("encoder_sha256")
                    or not record.get("preprocessing_sha256")
                    or not record.get("gamma_train_sha256")
                    or not record.get("rgb_cache_sha256")):
                return False
            for row in record["batches"]:
                parity = row.get("gamma_rollout_parity", {})
                trace_diffs = parity.get("cache_vs_live_trace_max_abs_diff", {})
                if (not math.isfinite(float(parity.get("max_gamma_cache_abs_diff", float("nan"))))
                        or float(parity["max_gamma_cache_abs_diff"]) > 2e-5
                        or not math.isfinite(float(parity.get("cached_baseline_old_loss_abs_diff", float("nan"))))
                        or float(parity["cached_baseline_old_loss_abs_diff"]) > 2e-5
                        or parity.get("cached_baseline_labels_H_exact") is not True
                        or parity.get("same_input_reference_loss_exact") is not True
                        or parity.get("same_input_reference_traces_exact") is not True
                        or parity.get("same_input_reference_labels_H_exact") is not True
                        or set(trace_diffs) != {"spikes", "core_out", "theta", "component_spikes"}
                        or not all(math.isfinite(float(value)) for value in trace_diffs.values())):
                    return False
            for row in record["batches"]:
                for family in ("encoder", "graph", "core", "joint"):
                    old = float(row["old_gradient_norms"][family])
                    if not math.isfinite(old) or old <= 0:
                        return False
                    if task["arm"] == "analytic_candidate":
                        rgb = float(row["rgb_gradient_norms"][family])
                        if not math.isfinite(rgb) or rgb <= 0:
                            return False
                if task["arm"] == "analytic_candidate":
                    qnorm = float(row.get("rgb_to_q_gradient_norm", float("nan")))
                    if not math.isfinite(qnorm) or qnorm <= 0:
                        return False
            if task["arm"] == "analytic_candidate":
                lam = float(record.get("lambda_joint", float("nan")))
                return (math.isfinite(lam) and lam > 0
                        and math.isfinite(float(record.get("scrambled_minus_real_rgb_loss", float("nan"))))
                        and float(record["scrambled_minus_real_rgb_loss"]) > 0
                        and all(math.isfinite(float(row.get("rgb_to_q_gradient_norm", float("nan"))))
                                and float(row["rgb_to_q_gradient_norm"]) > 0
                                for row in record["batches"]))
            return True
        if task["stage"] == "train":
            core, encoder = path.parent / "core.pt", path.parent / "encoder.pt"
            optimizer, history = path.parent / "optimizer.pt", path.parent / "history.json"
            rows = json.loads(history.read_text())
            preflight_path = ARCHIVE / f"preflight_seed0_{task['arm']}.json"
            preflight = json.loads(preflight_path.read_text())
            endpoint_review = validate_source_endpoint_review()
            return (record.get("status") == "training_complete"
                    and record.get("experiment") == "SW0117"
                    and record.get("seed") == 0 and record.get("arm") == task["arm"]
                    and record.get("updates") == 256 and record.get("batch_size") == 16
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("implementation_fingerprint") == implementation_fingerprint()
                    and record.get("source_endpoint_review_sha256") == sha(SOURCE_ENDPOINT_REVIEW)
                    and record.get("source_endpoint_report_sha256") == endpoint_review["report_sha256"]
                    and len(record.get("training_ids", [])) == 4096
                    and record.get("preflight_sha256") == sha(preflight_path)
                    and record.get("source_core_sha256") == preflight.get("source_core_sha256")
                    and record.get("source_manifest_sha256") == preflight.get("source_manifest_sha256")
                    and record.get("encoder_source_sha256") == preflight.get("encoder_sha256")
                    and record.get("preprocessing_sha256") == preflight.get("preprocessing_sha256")
                    and record.get("gamma_train_sha256") == preflight.get("gamma_train_sha256")
                    and record.get("gamma_train_manifest_sha256") == preflight.get("gamma_train_manifest_sha256")
                    and record.get("rgb_cache_sha256") == preflight.get("rgb_cache_sha256")
                    and record.get("rgb_cache_manifest_sha256") == preflight.get("rgb_cache_manifest_sha256")
                    and record.get("training_ids_sha256") == preflight.get("training_ids_sha256")
                    and record.get("lambda_joint") == preflight.get("lambda_joint")
                    and record.get("core_sha256") == sha(core)
                    and record.get("encoder_sha256") == sha(encoder)
                    and record.get("optimizer_sha256") == sha(optimizer)
                    and record.get("history_sha256") == sha(history)
                    and (path.parent / "TRAINING_COMPLETED").is_file()
                    and len(rows) == 256
                    and all(math.isfinite(float(row[k])) for row in rows
                            for k in ("total", "old", "primary", "positive_actual_spike_product",
                                      "rgb", "gradient_norm_preclip")))
        if (record.get("experiment") != "SW0117" or record.get("seed") != 0
                or record.get("arm") != task["arm"] or record.get("ids") != [1320, 1639]
                or record.get("images") != 320
                or record.get("ground_truth_used_for_prediction") is not False):
            return False
        folder = OUT / f"seed0_{task['arm']}"
        manifest_path, checkpoint, encoder = folder / "manifest.json", folder / "core.pt", folder / "encoder.pt"
        training = json.loads(manifest_path.read_text())
        sidecar_path = folder / "evaluation_manifest.json"
        if not sidecar_path.is_file():
            return False
        sidecar = json.loads(sidecar_path.read_text())
        expected_contract = {"ids": [1320, 1639], "images": 320, "batch_size": 8,
            "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
            "readout_threshold": 0.50, "min_group_size": 2,
            "background": "largest_component", "ground_truth_used_for_prediction": False}
        if (training.get("status") != "training_complete" or training.get("arm") != task["arm"]
                or training.get("core_sha256") != sha(checkpoint)
                or training.get("encoder_sha256") != sha(encoder)
                or record.get("training_manifest_sha256") != sha(manifest_path)
                or record.get("checkpoint_sha256") != sha(checkpoint)
                or record.get("encoder_checkpoint_sha256") != sha(encoder)
                or sidecar.get("experiment") != "SW0117"
                or sidecar.get("seed") != 0 or sidecar.get("arm") != task["arm"]
                or sidecar.get("evaluation_file") != path.name
                or sidecar.get("evaluation_sha256") != sha(path)
                or sidecar.get("checkpoint_sha256") != sha(checkpoint)
                or sidecar.get("encoder_checkpoint_sha256") != sha(encoder)
                or sidecar.get("training_manifest_sha256") != sha(manifest_path)
                or sidecar.get("evaluation_runner_sha256") != record.get("evaluation_runner_sha256")
                or sidecar.get("shared_evaluator_sha256") != record.get("shared_evaluator_sha256")
                or sidecar.get("trained_encoder_gamma_sha256") != record.get("trained_encoder_gamma_sha256")
                or sidecar.get("trained_encoder_gamma_manifest_sha256") != record.get("trained_encoder_gamma_manifest_sha256")
                or sidecar.get("evaluation_contract") != expected_contract):
            return False
        return _finite_score(record["sweep"][0]["scored_targets"]["our_hdf5"])
    except (OSError, KeyError, IndexError, TypeError, ValueError, AssertionError):
        return False


def main():
    raise SystemExit("SW0117 is dispatched only by the reviewed central SW0113 queue.")


if __name__ == "__main__":
    main()
