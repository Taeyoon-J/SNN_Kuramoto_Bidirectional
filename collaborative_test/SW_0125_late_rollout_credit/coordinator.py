"""Nine-task dependency adapter for the SW0125 three-seed late-tail pilot."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0125_late_rollout_credit import evaluate, run

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def task_plan():
    preflights = []
    trains = []
    evaluations = []
    for seed in run.SEEDS:
        preflights.append({"experiment": "SW0125", "task_id": f"sw0125_preflight_s{seed}",
                           "stage": "preflight", "seed": seed, "arm": run.ARM,
                           "depends_on": [], "priority": 0})
    all_preflight_ids = [task["task_id"] for task in preflights]
    for seed in run.SEEDS:
        train = {"experiment": "SW0125", "task_id": f"sw0125_train_s{seed}",
                 "stage": "train", "seed": seed, "arm": run.ARM,
                 "depends_on": list(all_preflight_ids), "priority": 1}
        trains.append(train)
        evaluations.append({"experiment": "SW0125", "task_id": f"sw0125_eval_s{seed}",
                            "stage": "evaluate", "seed": seed, "arm": run.ARM,
                            "depends_on": [train["task_id"]], "priority": 1})
    return preflights + trains + evaluations


def _folder(seed):
    return run.OUT / f"seed{seed}_{run.ARM}"


def artifact_path(task):
    seed, stage = int(task["seed"]), task["stage"]
    if task["arm"] != run.ARM:
        raise ValueError(f"unsupported SW0125 arm {task['arm']!r}")
    if stage == "preflight":
        return run.ARCHIVE / f"preflight_seed{seed}_{run.ARM}.json"
    if stage == "train":
        return _folder(seed) / "manifest.json"
    if stage == "evaluate":
        return _folder(seed) / "evaluation" / "evaluation.json"
    raise ValueError(f"unknown SW0125 task stage: {stage}")


def command(task, device="cuda:0"):
    seed, stage = int(task["seed"]), task["stage"]
    if stage in ("preflight", "train"):
        output = artifact_path(task)
        if stage == "train":
            output = output.parent
        return [sys.executable, str(run.RUNNER), stage, "--seed", str(seed),
                "--device", device, "--output", str(output)]
    folder = _folder(seed)
    return [sys.executable, str(HERE / "evaluate.py"), "--seed", str(seed),
            "--device", device, "--checkpoint", str(folder / "core.pt"),
            "--output", str(artifact_path(task))]


def _valid_score(score):
    try:
        if (set(score["metrics"]) != set(METRICS)
                or set(score["valid_count"]) != set(METRICS)
                or set(score["per_image"]) != set(METRICS)):
            return False
        for metric in METRICS:
            values = score["per_image"][metric]
            mean = float(score["metrics"][metric])
            if (len(values) != 320 or int(score["valid_count"][metric]) != 320
                    or not all(math.isfinite(float(value)) for value in values)
                    or not math.isfinite(mean)
                    or abs(sum(float(value) for value in values) / 320.0 - mean) > 1e-12):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        seed, stage = int(task["seed"]), task["stage"]
        if stage == "preflight":
            foundation_path = (run.ARCHIVE /
                f"foundation_layoutfix_seed{seed}_source_20261009.json")
            if (row.get("status") != "passed" or row.get("experiment") != "SW0125"
                    or row.get("seed") != seed or row.get("arm") != run.ARM
                    or row.get("implementation_fingerprint") != run.implementation_fingerprint()
                    or row.get("ground_truth_used") is not False
                    or row.get("updates") != 256 or row.get("batch_size") != 16
                    or row.get("total_time_steps") != 1024 or row.get("settle") != 512
                    or row.get("tail_steps") != 64 or len(row.get("training_ids", [])) != 4096
                    or len(row.get("batches", [])) != 4
                    or row.get("throwaway_b16_adam_update") is not True
                    or row.get("throwaway_parameter_groups_changed") != {
                        "graph": True, "core": True, "encoder": True}
                    or row.get("throwaway_encoder_changed") is not True
                    or row.get("throwaway_named_core_groups_changed") != {
                        "graph": True, "oscillator_drive": True, "kuramoto": True,
                        "dendritic": True, "membrane": True}
                    or not foundation_path.is_file()
                    or row.get("foundation_report", {}).get("sha256") != run.sha(foundation_path)
                    or row.get("lambda_reference", {}).get("lambda_artifact_sha256") != run.sha(run.LAMBDA_PATH)
                    or row.get("lambda_reference", {}).get("lambda_joint") != run.LAMBDA):
                return False
            for batch in row["batches"]:
                parity = batch.get("same_input_parity", {})
                if (not all(parity.get(key) is True for key in (
                        "q_exact", "production_labels_h_exact"))
                        or not all(value is True for value in parity.get("full_trace_exact", {}).values())
                        or not all(value is True for value in parity.get("objective_values_exact", {}).values())
                        or not math.isfinite(float(parity.get("gamma_cache_max_abs_diff", float("nan"))))
                        or float(parity["gamma_cache_max_abs_diff"]) > 2e-5):
                    return False
                for family in ("encoder", "graph", "core", "joint"):
                    for key in ("old_gradient_norms", "rgb_gradient_norms"):
                        value = float(batch.get(key, {}).get(family, float("nan")))
                        if not math.isfinite(value) or value <= 0:
                            return False
                qnorm = float(batch.get("rgb_to_q_gradient_norm", float("nan")))
                if not math.isfinite(qnorm) or qnorm <= 0:
                    return False
                for family in ("encoder", "graph", "oscillator_drive", "kuramoto", "dendritic", "membrane"):
                    value = float(batch.get("rgb_gradient_norms_by_named_family", {}).get(
                        family, float("nan")))
                    if not math.isfinite(value) or value <= 0:
                        return False
            for value in row.get("throwaway_gradient_norms_by_named_family", {}).values():
                if not math.isfinite(float(value)) or float(value) <= 0:
                    return False
            return True

        if stage == "train":
            folder = path.parent
            history_path = folder / "history.json"
            files = {"core": folder / "core.pt", "encoder": folder / "encoder.pt",
                     "optimizer": folder / "optimizer.pt", "history": history_path}
            if any(not value.is_file() for value in files.values()):
                return False
            history = json.loads(history_path.read_text(encoding="utf-8"))
            pf_path = run.ARCHIVE / f"preflight_seed{seed}_{run.ARM}.json"
            pf = json.loads(pf_path.read_text(encoding="utf-8"))
            return (row.get("status") == "training_complete" and row.get("experiment") == "SW0125"
                    and row.get("seed") == seed and row.get("arm") == run.ARM
                    and row.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and row.get("preflight_sha256") == run.sha(pf_path)
                    and row.get("source_core_sha256") == pf.get("source_core_sha256")
                    and row.get("foundation_report_sha256") == pf.get("foundation_report", {}).get("sha256")
                    and row.get("lambda_artifact_sha256") == run.sha(run.LAMBDA_PATH)
                    and row.get("training_ids_sha256") == pf.get("training_ids_sha256")
                    and row.get("lambda_joint") == run.LAMBDA
                    and row.get("updates") == 256 and row.get("batch_size") == 16
                    and row.get("ground_truth_used_for_training") is False
                    and len(history) == 256
                    and [int(item.get("update", -1)) for item in history] == list(range(1, 257))
                    and all(math.isfinite(float(item[key])) for item in history
                            for key in ("total", "primary_phase_last512",
                                        "positive_actual_q_last512", "rgb_last512",
                                        "gradient_norm_preclip"))
                    and all(row.get(name + "_sha256") == run.sha(file)
                            for name, file in files.items())
                    and (folder / "TRAINING_COMPLETED").is_file())

        folder = path.parent
        sidecar_path = folder / "evaluation_manifest.json"
        gamma_path = folder / "trained_encoder_gamma_validation.pt"
        gamma_manifest_path = folder / "trained_encoder_gamma_validation_manifest.json"
        if not all(file.is_file() for file in (sidecar_path, gamma_path, gamma_manifest_path)):
            return False
        manifest_path = _folder(seed) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        score = row["sweep"][0]["scored_targets"]["our_hdf5"]
        contract = {"ids": [1320, 1639], "images": 320, "batch_size": 8,
                    "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
                    "readout_threshold": 0.50, "min_group_size": 2,
                    "background": "largest_component", "ground_truth_used_for_prediction": False}
        return (row.get("status") == "complete" and row.get("experiment") == "SW0125"
                and row.get("seed") == seed and row.get("arm") == run.ARM
                and row.get("ids") == [1320, 1639] and row.get("images") == 320
                and row.get("ground_truth_used_for_prediction") is False
                and row.get("evaluation_contract") == contract and _valid_score(score)
                and row.get("training_manifest_sha256") == run.sha(manifest_path)
                and row.get("checkpoint_sha256") == run.sha(_folder(seed) / "core.pt")
                and row.get("encoder_checkpoint_sha256") == run.sha(_folder(seed) / "encoder.pt")
                and row.get("trained_encoder_gamma_sha256") == run.sha(gamma_path)
                and row.get("trained_encoder_gamma_manifest_sha256") == run.sha(gamma_manifest_path)
                and row.get("evaluation_runner_sha256") == run.sha(HERE / "evaluate.py")
                and row.get("training_runner_sha256") == run.sha(run.RUNNER)
                and sidecar.get("experiment") == "SW0125" and sidecar.get("seed") == seed
                and sidecar.get("arm") == run.ARM and sidecar.get("evaluation_sha256") == run.sha(path)
                and sidecar.get("evaluation_contract") == contract
                and sidecar.get("training_manifest_sha256") == run.sha(manifest_path))
    except (OSError, KeyError, IndexError, TypeError, ValueError, AssertionError, RuntimeError):
        return False


if __name__ == "__main__":
    raise SystemExit("SW0125 is run only by the owner-aware dispatcher.")
