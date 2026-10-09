"""SW0121 seed-0 task adapter for the central SW0113 dispatcher."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0117_joint_analytic_rgb import run as prior
from collaborative_test.SW_0121_competitive_assignment import run


CONTROL_ID = "sw0121_registered_control"


def task_plan():
    control = {"experiment": "SW0121", "task_id": CONTROL_ID,
               "stage": "control_validation", "seed": 0,
               "depends_on": [], "priority": 0}
    preflight = {"experiment": "SW0121", "task_id": "sw0121_preflight_seed0",
                 "stage": "preflight", "seed": 0,
                 "depends_on": [CONTROL_ID], "priority": 1}
    train = {"experiment": "SW0121", "task_id": "sw0121_train_seed0",
             "stage": "train", "seed": 0,
             "depends_on": [preflight["task_id"]], "priority": 1}
    evaluate = {"experiment": "SW0121", "task_id": "sw0121_evaluate_seed0",
                "stage": "evaluate", "seed": 0,
                "depends_on": [train["task_id"]], "priority": 1}
    return [control, preflight, train, evaluate]


def artifact_path(task):
    stage = task["stage"]
    if stage == "control_validation":
        return prior.OUT / "seed0_analytic_candidate" / "evaluation.json"
    if stage == "preflight":
        return run.ARCHIVE / "preflight_seed0.json"
    folder = run.OUT / "seed0_analytic_candidate"
    return folder / ("manifest.json" if stage == "train" else "evaluation.json")


def command(task, device="cuda"):
    if task["stage"] == "control_validation":
        return []
    path = artifact_path(task)
    cmd = [sys.executable, str(run.RUNNER), task["stage"], "--seed", "0",
           "--device", device]
    if task["stage"] in ("preflight", "train"):
        if task["stage"] == "train":
            path = path.parent
        return cmd + ["--output", str(path)]
    return cmd + ["--checkpoint", str(run.OUT / "seed0_analytic_candidate" / "core.pt"),
                  "--output", str(path)]


def _finite_score(score):
    metrics = {"fg_ari", "foreground_iou", "matched_object_iou"}
    try:
        if (set(score["metrics"]) != metrics or set(score["valid_count"]) != metrics
                or set(score["per_image"]) != metrics):
            return False
        for key in metrics:
            values = score["per_image"][key]
            mean = float(score["metrics"][key])
            if (score["valid_count"][key] != 320 or len(values) != 320
                    or not all(math.isfinite(float(x)) for x in values)
                    or not math.isfinite(mean)
                    or abs(sum(map(float, values)) / 320.0 - mean) > 1e-12):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _valid_registered_control():
    return run.validate_registered_control()


def valid_result(task):
    if task["stage"] == "control_validation":
        return _valid_registered_control()
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if task["stage"] == "preflight":
            if (record.get("status") != "passed" or record.get("experiment") != "SW0121"
                    or record.get("seed") != 0
                    or record.get("implementation_fingerprint") != run.implementation_fingerprint()
                    or record.get("ground_truth_used") is not False
                    or record.get("matched_shuffle_seed") != 117
                    or len(record.get("training_ids", [])) != 4096
                    or len(record.get("calibration_batches", [])) != 4
                    or not (run.LAMBDA_MIN <= float(record.get("lambda", float("nan"))) <= run.LAMBDA_MAX)
                    or not (float(record.get("warmup_R_mean", float("nan")))
                            < float(record.get("global_mean_R_baseline", float("nan"))))
                    or float(record.get("fixed_scramble_R_delta", 0)) <= 0
                    or record.get("throwaway_update", {}).get("passed") is not True
                    or record.get("throwaway_update", {}).get("changed") != {
                        "encoder": True, "graph": True, "core": True, "head": True}):
                return False
            for row in record["calibration_batches"]:
                if (not math.isfinite(float(row.get("old_norm", float("nan"))))
                        or not math.isfinite(float(row.get("aux_norm", float("nan"))))
                        or float(row["old_norm"]) <= 0 or float(row["aux_norm"]) <= 0
                        or not math.isfinite(float(row.get("R_to_head_norm", float("nan"))))
                        or not math.isfinite(float(row.get("R_to_component_spikes_norm", float("nan"))))
                        or not math.isfinite(float(row.get("C_to_Q_norm", float("nan"))))
                        or not math.isfinite(float(row.get("assignment_patch_std", float("nan"))))
                        or float(row.get("R_to_head_norm", 0)) <= 0
                        or float(row.get("R_to_component_spikes_norm", 0)) <= 0
                        or float(row.get("C_to_Q_norm", 0)) <= 0
                        or float(row.get("assignment_patch_std", 0)) <= 0):
                    return False
            for artifact, key in ((run.WARMUP_HEAD, "head_sha256"),
                                  (run.WARMUP_OPTIMIZER, "optimizer_sha256"),
                                  (run.WARMUP_FEATURES, "features_sha256"),
                                  (run.WARMUP_METADATA, "warmup_metadata_sha256")):
                if not artifact.is_file() or record.get("artifacts", {}).get(key) != run.sha(artifact):
                    return False
            lambda_record = json.loads(run.LAMBDA_PATH.read_text(encoding="utf-8"))
            if (lambda_record.get("status") != "passed"
                    or lambda_record.get("preflight_sha256") != run.sha(path)
                    or float(lambda_record.get("lambda", float("nan"))) != float(record["lambda"])
                    or lambda_record.get("implementation_fingerprint") != run.implementation_fingerprint()):
                return False
            return True
        if task["stage"] == "train":
            folder = path.parent
            history_path = folder / "history.json"
            if not all(p.is_file() for p in (folder / "TRAINING_COMPLETED", folder / "core.pt",
                                             folder / "encoder.pt", folder / "head.pt",
                                             folder / "core_optimizer.pt", folder / "head_optimizer.pt",
                                             folder / "optimizer.pt",
                                             history_path)):
                return False
            history = json.loads(history_path.read_text(encoding="utf-8"))
            pf_path = run.ARCHIVE / "preflight_seed0.json"
            pf = json.loads(pf_path.read_text(encoding="utf-8"))
            return (record.get("status") == "training_complete"
                    and record.get("experiment") == "SW0121" and record.get("seed") == 0
                    and record.get("arm") == "analytic_candidate"
                    and record.get("updates") == run.UPDATES and record.get("batch_size") == run.BATCH
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("preflight_sha256") == run.sha(pf_path)
                    and record.get("lambda") == pf.get("lambda")
                    and record.get("training_ids_sha256") == pf.get("training_ids_sha256")
                    and record.get("source_core_sha256") == pf.get("source_core_sha256")
                    and record.get("source_manifest_sha256") == pf.get("source_manifest_sha256")
                    and record.get("encoder_source_sha256") == pf.get("encoder_sha256")
                    and record.get("preprocessing_sha256") == pf.get("preprocessing_sha256")
                    and record.get("gamma_train_sha256") == pf.get("gamma_train_sha256")
                    and record.get("gamma_train_manifest_sha256") == pf.get("gamma_train_manifest_sha256")
                    and record.get("rgb_cache_sha256") == pf.get("rgb_cache_sha256")
                    and record.get("rgb_cache_manifest_sha256") == pf.get("rgb_cache_manifest_sha256")
                    and record.get("head_warmup_sha256") == run.sha(run.WARMUP_HEAD)
                    and record.get("head_warmup_optimizer_sha256") == run.sha(run.WARMUP_OPTIMIZER)
                    and record.get("head_warmup_features_sha256") == run.sha(run.WARMUP_FEATURES)
                    and record.get("core_sha256") == run.sha(folder / "core.pt")
                    and record.get("encoder_sha256") == run.sha(folder / "encoder.pt")
                    and record.get("head_sha256") == run.sha(folder / "head.pt")
                    and record.get("core_optimizer_sha256") == run.sha(folder / "core_optimizer.pt")
                    and record.get("head_optimizer_sha256") == run.sha(folder / "head_optimizer.pt")
                    and record.get("optimizer_sha256") == run.sha(folder / "optimizer.pt")
                    and record.get("history_sha256") == run.sha(history_path)
                    and len(history) == run.UPDATES
                    and all(math.isfinite(float(row[k])) for row in history
                            for k in ("old", "primary", "positive_actual_Q", "R", "C", "total",
                                      "core_gradient_norm_preclip", "head_gradient_norm_preclip")))
        if (record.get("experiment") != "SW0121" or record.get("seed") != 0
                or record.get("arm") != "analytic_candidate"
                or record.get("ids") != [1320, 1639] or record.get("images") != 320
                or record.get("ground_truth_used_for_prediction") is not False
                or record.get("assignment_head_used_for_prediction") is not False):
            return False
        folder = path.parent
        manifest_path = artifact_path({"stage": "train", "seed": 0})
        sidecar_path = folder / "evaluation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        checkpoint = folder / "core.pt"
        return (manifest.get("status") == "training_complete"
                and record.get("training_manifest_sha256") == run.sha(manifest_path)
                and record.get("checkpoint_sha256") == run.sha(checkpoint)
                and record.get("encoder_checkpoint_sha256") == run.sha(folder / "encoder.pt")
                and record.get("head_checkpoint_sha256") == run.sha(folder / "head.pt")
                and sidecar.get("evaluation_sha256") == run.sha(path)
                and sidecar.get("training_manifest_sha256") == run.sha(manifest_path)
                and sidecar.get("checkpoint_sha256") == run.sha(checkpoint)
                and sidecar.get("experiment") == "SW0121"
                and sidecar.get("evaluation_file") == path.name
                and sidecar.get("evaluation_runner_sha256") == record.get("evaluation_runner_sha256")
                and sidecar.get("shared_evaluator_sha256") == record.get("shared_evaluator_sha256")
                and sidecar.get("evaluation_contract") == {
                    "ids": [1320, 1639], "images": 320, "batch_size": 8,
                    "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
                    "readout_threshold": 0.50, "min_group_size": 2,
                    "background": "largest_component", "ground_truth_used_for_prediction": False}
                and _finite_score(record["sweep"][0]["scored_targets"]["our_hdf5"]))
    except (OSError, KeyError, IndexError, TypeError, ValueError, AssertionError):
        return False


def main():
    raise SystemExit("SW0121 tasks are dispatched by SW0113 after their reviewed preregistration.")


if __name__ == "__main__":
    main()
