"""Task adapter for the reviewed central SW0113 dispatcher; no GPU process here."""
from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(HERE)]
import run as experiment
import numpy as np


def _cmd(*args):
    return [sys.executable, str(HERE / "run.py"), *map(str, args)]


def artifact_path(task):
    return Path(task["output"])


def command(task, device):
    return list(task["command"]) + ["--device", str(device)]


def task_plan():
    """Create one shared cache task, then paired source-matched tasks per seed."""
    tasks = []
    val_cache = "sw0114_cache_val"
    tasks.append({"task_id": val_cache, "experiment": "SW0114", "stage": "cache_validation",
                  "seed": None, "depends_on": [], "priority": 0,
                  "command": _cmd("--build-validation-cache"),
                  "output": str(experiment.ARCHIVE / "gamma32_validation_1320_1639.pt")})
    for seed in experiment.SEEDS:
        cache = f"sw0114_cache_train_s{seed}"
        tasks.append({"task_id": cache, "experiment": "SW0114", "stage": "cache_training",
                      "seed": seed, "depends_on": [], "priority": 0,
                      "command": _cmd("--build-train-cache", seed),
                      "output": str(experiment.ARCHIVE / f"gamma32_train_seed{seed}.pt")})
    for grid in (16, 32):
        for seed in experiment.SEEDS:
            deps = [] if grid == 16 else [f"sw0114_cache_train_s{seed}"]
            task_id = f"sw0114_preflight_g{grid}_s{seed}"
            tasks.append({"task_id": task_id, "experiment": "SW0114", "stage": "preflight",
                          "seed": seed, "grid_size": grid, "depends_on": deps,
                          "priority": 0,
                          "command": _cmd("--preflight", seed, "--grid", grid),
                          "output": str(experiment.ARCHIVE / f"preflight_seed{seed}_grid{grid}.json")})
    microaudit = "sw0114_preflight_g16_s0"
    for grid in (16, 32):
        for seed in experiment.SEEDS:
            task_id = f"sw0114_train_g{grid}_s{seed}"
            dependencies = [f"sw0114_preflight_g{grid}_s{seed}", microaudit]
            if grid == 32:
                dependencies.append(f"sw0114_cache_train_s{seed}")
            output = ROOT / "trained_models/SW0114_resolution32_pilot" / f"seed{seed}_grid{grid}"
            tasks.append({"task_id": task_id, "experiment": "SW0114", "stage": "train",
                          "seed": seed, "grid_size": grid, "depends_on": dependencies,
                          "priority": 2, "command": _cmd("--train", seed, "--grid", grid,
                                                               "--output", output),
                          "output": str(output)})
    for seed in experiment.SEEDS:
        task_id = f"sw0114_evaluate_s{seed}"
        control = ROOT / "trained_models/SW0114_resolution32_pilot" / f"seed{seed}_grid16/core.pt"
        candidate = ROOT / "trained_models/SW0114_resolution32_pilot" / f"seed{seed}_grid32/core.pt"
        tasks.append({"task_id": task_id, "experiment": "SW0114", "stage": "evaluate",
                      "seed": seed, "depends_on": [f"sw0114_train_g16_s{seed}",
                                                       f"sw0114_train_g32_s{seed}", val_cache],
                      "priority": 1,
                      "command": _cmd("--evaluate", seed, "--control-checkpoint", control,
                                       "--candidate-checkpoint", candidate),
                      "output": str(experiment.ARCHIVE / f"evaluation_seed{seed}/evaluation.json")})
    summary_output = experiment.ARCHIVE / "summary.json"
    tasks.append({"task_id": "sw0114_summary", "experiment": "SW0114", "stage": "summarize",
                  "seed": None, "depends_on": [f"sw0114_evaluate_s{s}" for s in experiment.SEEDS],
                  "priority": 1, "command": _cmd("--summarize"),
                  "output": str(summary_output)})
    return tasks


def valid_result(task):
    try:
        task_id = task["task_id"]
        stage = task["stage"]
        if stage == "cache_training":
            p = Path(task["output"])
            meta = json.loads(p.with_suffix(".json").read_text())
            from collaborative_test.SW_0110_xy_graph_route import run as common
            source, source_manifest, source_record = common.source_paths(task["seed"])
            return (p.is_file() and experiment.sha(p) == meta.get("gamma32_sha256")
                    and meta.get("seed") == task["seed"] and meta.get("count") == 4096
                    and meta.get("ids") == source_record.get("training_ids")
                    and meta.get("source_core_sha256") == experiment.sha(source)
                    and meta.get("source_manifest_sha256") == experiment.sha(source_manifest)
                    and meta.get("encoder_sha256") == common.EXPECTED_ENCODER_SHA256
                    and meta.get("preprocessing_sha256") == common.EXPECTED_PREPROCESSING_SHA256
                    and bool(meta.get("selected_rgb_sha256"))
                    and meta.get("dataset_identity", {}).get("image_shape") == [100000, 128, 128, 3]
                    and meta.get("registered_gamma16_max_abs_diff", 1.) <= 2e-5)
        if stage == "cache_validation":
            p = Path(task["output"])
            meta = json.loads(p.with_suffix(".json").read_text())
            from collaborative_test.SW_0110_xy_graph_route import run as common
            return (p.is_file() and experiment.sha(p) == meta.get("gamma32_sha256")
                    and meta.get("ids") == list(range(1320, 1640))
                    and meta.get("count") == 320 and meta.get("ground_truth_read") is False
                    and meta.get("encoder_sha256") == common.EXPECTED_ENCODER_SHA256
                    and meta.get("preprocessing_sha256") == common.EXPECTED_PREPROCESSING_SHA256
                    and meta.get("registered_gamma16_sha256") == experiment.sha(experiment.VAL_GAMMA)
                    and bool(meta.get("selected_rgb_sha256")))
        if stage == "preflight":
            p = Path(task["output"])
            item = json.loads(p.read_text())
            micro = item.get("microbatch_equivalence")
            if task["grid_size"] == 16 and task["seed"] == 0:
                if not isinstance(micro, dict) or micro.get("pass") is not True:
                    return False
            memory = item.get("full_horizon_inference", {})
            from collaborative_test.SW_0110_xy_graph_route import run as common
            source, source_manifest, source_record = common.source_paths(task["seed"])
            if any(not isinstance(row.get("loss"), (int, float)) or not math.isfinite(row["loss"])
                   or not isinstance(row.get("gradient_norm"), (int, float))
                   or not math.isfinite(row["gradient_norm"]) or row["gradient_norm"] <= 0
                   for row in item.get("batches", [])):
                return False
            return (item.get("status") == "passed" and item.get("seed") == task["seed"]
                    and item.get("grid_size") == task["grid_size"]
                    and item.get("source_core_sha256") == experiment.sha(source)
                    and item.get("source_manifest_sha256") == experiment.sha(source_manifest)
                    and item.get("source_manifest_ids_sha256") == hashlib.sha256(
                        np.asarray(source_record["training_ids"], dtype="<i8").tobytes()).hexdigest()
                    and item.get("implementation_sha256") == experiment.implementation_fingerprint()
                    and item.get("ground_truth_used") is False
                    and item.get("throwaway_logical_adam_update") is True
                    and memory.get("time_steps") == 1024
                    and math.isfinite(memory.get("seconds", float("nan")))
                    and str(item.get("device", "")).startswith("cuda")
                    and memory.get("peak_allocated_bytes") is not None
                    and memory.get("peak_reserved_bytes") is not None
                    and len(item.get("batches", [])) == 4)
        if stage == "train":
            folder = Path(task["output"])
            m = json.loads((folder / "manifest.json").read_text())
            p = folder / "core.pt"
            from collaborative_test.SW_0110_xy_graph_route import run as common
            source, source_manifest, source_record = common.source_paths(task["seed"])
            return (p.is_file() and m.get("status") == "complete"
                    and m.get("seed") == task["seed"] and m.get("grid_size") == task["grid_size"]
                    and m.get("ground_truth_used_for_training") is False
                    and m.get("updates") == 256 and m.get("core_sha256") == experiment.sha(p)
                    and m.get("source_core_sha256") == experiment.sha(source)
                    and m.get("source_manifest_sha256") == experiment.sha(source_manifest)
                    and m.get("training_ids") == source_record.get("training_ids")
                    and m.get("implementation_sha256") == experiment.implementation_fingerprint())
        if stage == "evaluate":
            p = Path(task["output"])
            report = json.loads(p.read_text())
            from collaborative_test.SW_0110_xy_graph_route import run as common
            source = common.source_paths(task["seed"])[0]
            out_root = ROOT / "trained_models/SW0114_resolution32_pilot"
            control = out_root / f"seed{task['seed']}_grid16/core.pt"
            candidate = out_root / f"seed{task['seed']}_grid32/core.pt"
            prediction_file = p.parent / "frozen_predictions.pt"
            if (report.get("status") != "complete" or report.get("count") != 320
                    or report.get("seed") != task["seed"]
                    or report.get("ground_truth_used_during_prediction") is not False
                    or report.get("source_baseline_equivalence", {}).get("pass") is not True
                    or report.get("source_baseline_equivalence", {}).get("reference_sha256")
                       != experiment.SOURCE_EVAL_SHAS.get(task["seed"])
                    or report.get("implementation_fingerprint") != experiment.implementation_fingerprint()
                    or not prediction_file.is_file()
                    or report.get("frozen_predictions_sha256") != experiment.sha(prediction_file)
                    or report.get("source_checkpoint_sha256") != experiment.sha(source)
                    or report.get("control_checkpoint_sha256") != experiment.sha(control)
                    or report.get("candidate_checkpoint_sha256") != experiment.sha(candidate)):
                return False
            expected_metrics = {"fg_ari", "foreground_iou", "matched_object_iou"}
            if set(report.get("scores", {})) != {"source16_vs_modal8", "control16_vs_modal8",
                    "control_repeated32_vs_modal4", "candidate_pooled16_vs_modal8",
                    "candidate_native32_vs_modal4"}:
                return False
            for section in report["scores"].values():
                if (set(section.get("valid_count", {})) != expected_metrics
                        or set(section.get("per_image", {})) != expected_metrics
                        or set(section.get("mean", {})) != expected_metrics):
                    return False
                for metric, count in section["valid_count"].items():
                    values = section["per_image"][metric]
                    finite = [v for v in values if v is not None]
                    if (count != 320 or len(values) != 320 or len(finite) != 320
                            or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in finite)
                            or not isinstance(section["mean"][metric], (int, float))
                            or not math.isfinite(section["mean"][metric])):
                        return False
                    mean = sum(finite) / 320
                    if abs(mean - section["mean"][metric]) > 1e-12:
                        return False
            return True
        if stage == "summarize":
            from summarize import summarize
            p = Path(task["output"])
            saved = json.loads(p.read_text())
            recomputed = summarize()
            return saved == recomputed and saved.get("experiment") == "SW0114"
        return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
