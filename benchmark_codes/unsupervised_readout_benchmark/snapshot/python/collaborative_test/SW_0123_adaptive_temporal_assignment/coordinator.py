"""Dependency-aware task adapter for the nine SW0123 seed/arm trajectories."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import evaluate, run

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def task_plan():
    tasks = []
    for seed in run.SEEDS:
        for arm in run.ARMS:
            preflight = {"experiment": "SW0123", "task_id": f"sw0123_preflight_s{seed}_{arm}",
                         "stage": "preflight", "seed": seed, "arm": arm,
                         "depends_on": [], "priority": 0}
            train = {"experiment": "SW0123", "task_id": f"sw0123_train_s{seed}_{arm}",
                     "stage": "train", "seed": seed, "arm": arm,
                     "depends_on": [preflight["task_id"]], "priority": 1}
            endpoint = {"experiment": "SW0123", "task_id": f"sw0123_eval_s{seed}_{arm}",
                        "stage": "evaluate", "seed": seed, "arm": arm,
                        "depends_on": [train["task_id"]], "priority": 1}
            tasks.extend((preflight, train, endpoint))
    return tasks


def _seed_arm_folder(seed, arm):
    return run.OUT / f"seed{seed}_{arm}"


def artifact_path(task):
    seed, arm = int(task["seed"]), task["arm"]
    if task["stage"] == "preflight":
        return run.ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    folder = _seed_arm_folder(seed, arm)
    if task["stage"] == "train":
        return folder / "manifest.json"
    if task["stage"] == "evaluate":
        return folder / "evaluation" / "evaluation.json"
    raise ValueError(f"unknown SW0123 task stage {task['stage']!r}")


def command(task, device="cuda"):
    seed, arm, stage = int(task["seed"]), task["arm"], task["stage"]
    if stage in ("preflight", "train"):
        output = artifact_path(task)
        if stage == "train":
            output = output.parent
        return [sys.executable, str(run.HERE / "run.py"), stage,
                "--seed", str(seed), "--arm", arm, "--device", device,
                "--output", str(output)]
    folder = _seed_arm_folder(seed, arm)
    return [sys.executable, str(evaluate.HERE / "evaluate.py"),
            "--seed", str(seed), "--arm", arm, "--device", device,
            "--checkpoint", str(folder / "core.pt"),
            "--output", str(artifact_path(task))]


def _score_valid(score):
    try:
        if (set(score["metrics"]) != set(METRICS)
                or set(score["valid_count"]) != set(METRICS)
                or set(score["per_image"]) != set(METRICS)):
            return False
        for metric in METRICS:
            values = score["per_image"][metric]
            mean = float(score["metrics"][metric])
            if (len(values) != 320 or int(score["valid_count"][metric]) != 320
                    or not all(math.isfinite(float(v)) for v in values)
                    or not math.isfinite(mean)
                    or abs(sum(float(v) for v in values) / 320 - mean) > 1e-12):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def valid_result(task):
    """Validate durable stage artifacts without converting scientific guards to success."""
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        seed, arm, stage = int(task["seed"]), task["arm"], task["stage"]
        if stage == "preflight":
            warm_path = run._warmup_path(seed, arm)
            if not warm_path.is_file():
                return False
            warm = __import__("torch").load(warm_path, map_location="cpu", weights_only=False)
            expected = record.get("expected_parameter_groups_changed", {})
            changed = record.get("throwaway_parameter_groups_changed", {})
            batches = record.get("gradient_batches", [])
            if not (record.get("status") == "passed" and record.get("experiment") == "SW0123"
                    and record.get("seed") == seed and record.get("arm") == arm
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("ground_truth_used") is False
                    and record.get("throwaway_b16_two_adam_update") is True
                    and record.get("warmup_artifact_sha256") == run.sha(warm_path)
                    and warm.get("status") == "passed" and warm.get("seed") == seed
                    and warm.get("arm") == arm and len(batches) == 4
                    and len(record.get("training_ids", [])) == 4096
                    and expected == changed
                    and all(math.isfinite(float(row["loss"])) for row in batches)):
                return False
            for row in batches:
                gradients = row.get("gradients", {})
                required = {"encoder", "graph", "head", "decoder"}
                if arm != "gate_only_control":
                    required.update(("dendrite", "membrane"))
                if record.get("throwaway_parameter_groups_changed", {}).get("oscillator_drive") is True:
                    required.add("oscillator_drive")
                if any(name not in gradients
                       or not math.isfinite(float(gradients[name].get("norm", float("nan"))))
                       or float(gradients[name]["norm"]) <= 0 for name in required):
                    return False
            return True

        if stage == "train":
            folder = path.parent
            files = {"core": folder / "core.pt", "encoder": folder / "encoder.pt",
                     "assignment_head": folder / "assignment_head.pt", "rgb_decoder": folder / "rgb_decoder.pt",
                     "core_optimizer": folder / "core_optimizer.pt", "head_optimizer": folder / "head_optimizer.pt",
                     "history": folder / "history.json"}
            if any(not p.is_file() for p in files.values()) or not (folder / "TRAINING_COMPLETED").is_file():
                return False
            history = json.loads(files["history"].read_text(encoding="utf-8"))
            preflight_path = run.ARCHIVE / f"preflight_seed{seed}_{arm}.json"
            warmup_path = run._warmup_path(seed, arm)
            if not preflight_path.is_file() or not warmup_path.is_file():
                return False
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            return (record.get("status") == "training_complete" and record.get("experiment") == "SW0123"
                    and record.get("seed") == seed and record.get("arm") == arm
                    and record.get("updates") == 256 and record.get("batch_size") == 16
                    and record.get("time_steps") == 64 and record.get("settle") == 32
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("preflight_sha256") == run.sha(preflight_path)
                    and record.get("warmup_artifact_sha256") == run.sha(warmup_path)
                    and preflight.get("status") == "passed" and preflight.get("arm") == arm
                    and record.get("source_core_sha256") == preflight.get("source_core_sha256")
                    and record.get("training_ids_sha256") == preflight.get("training_ids_sha256")
                    and len(history) == 256
                    and [int(x.get("update", -1)) for x in history] == list(range(1, 257))
                    and all(record.get(name + "_sha256") == run.sha(file)
                            for name, file in files.items()))

        folder = path.parent
        sidecar_path = folder / "evaluation_manifest.json"
        pred_path = folder / "frozen_predictions.pt"
        gamma_path = folder / "trained_encoder_gamma.pt"
        if not all(p.is_file() for p in (sidecar_path, pred_path, gamma_path)):
            return False
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        train_manifest_path = _seed_arm_folder(seed, arm) / "manifest.json"
        train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
        training_files = {"core": train_manifest_path.parent / "core.pt",
                          "encoder": train_manifest_path.parent / "encoder.pt",
                          "assignment_head": train_manifest_path.parent / "assignment_head.pt",
                          "rgb_decoder": train_manifest_path.parent / "rgb_decoder.pt",
                          "core_optimizer": train_manifest_path.parent / "core_optimizer.pt",
                          "head_optimizer": train_manifest_path.parent / "head_optimizer.pt",
                          "history": train_manifest_path.parent / "history.json"}
        if (any(not file.is_file() for file in training_files.values())
                or any(train_manifest.get(name + "_sha256") != run.sha(file)
                       for name, file in training_files.items())):
            return False
        contract = {"batch_size": 8, "time_steps": 1024, "settle": 512,
                    "readout_threshold": 0.50, "min_group_size": 2,
                    "background": "largest_component", "ground_truth_used_for_prediction": False}
        return (record.get("status") == "complete" and record.get("experiment") == "SW0123"
                and record.get("seed") == seed and record.get("arm") == arm
                and record.get("ids") == [1320, 1639] and record.get("images") == 320
                and record.get("ground_truth_used_for_prediction") is False
                and record.get("evaluation_contract") == contract
                and record.get("evaluation_implementation_fingerprint") == evaluate.evaluation_fingerprint()
                and record.get("training_manifest_sha256") == run.sha(train_manifest_path)
                and train_manifest.get("status") == "training_complete"
                and record.get("prediction_sha256") == run.sha(pred_path)
                and record.get("trained_encoder_gamma_sha256") == run.sha(gamma_path)
                and record.get("training_artifact_sha256") == {
                    name: run.sha(file) for name, file in training_files.items()}
                and record.get("metrics_valid") is True
                and _score_valid(record.get("primary_assignment", {}))
                and _score_valid(record.get("secondary_actual_qcc", {}))
                and sidecar.get("experiment") == "SW0123" and sidecar.get("seed") == seed
                and sidecar.get("arm") == arm and sidecar.get("evaluation_sha256") == run.sha(path)
                and sidecar.get("prediction_sha256") == run.sha(pred_path)
                and sidecar.get("trained_encoder_gamma_sha256") == run.sha(gamma_path)
                and sidecar.get("training_manifest_sha256") == run.sha(train_manifest_path)
                and sidecar.get("evaluation_implementation_fingerprint") == evaluate.evaluation_fingerprint()
                and sidecar.get("contract") == contract)
    except (OSError, KeyError, TypeError, ValueError, AssertionError, RuntimeError):
        return False


def main():
    raise SystemExit("SW0123 tasks are run by a reviewed owner-aware dispatcher.")


if __name__ == "__main__":
    main()
