"""Dependency plan and artifact validation for SW0111."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(HERE)]
import run as experiment


def command(task, device):
    return list(task["command"]) + ["--device", str(device)]


def artifact_path(task):
    return Path(task["output"])


def _cmd(*args):
    return [sys.executable, str(HERE / "run.py"), *map(str, args)]


def task_plan():
    tasks = []
    for seed in experiment.SEEDS:
        deps = [] if seed == 0 else ["sw0111_preflight_s0"]
        tasks.append({"task_id": f"sw0111_preflight_s{seed}", "experiment": "SW0111",
                      "stage": "preflight", "seed": seed, "depends_on": deps,
                      "priority": 0 if seed == 0 else 1,
                      "command": _cmd("--preflight", seed),
                      "output": str(experiment.ARCHIVE / f"preflight_seed{seed}.json")})
    for seed in experiment.SEEDS:
        for arm in experiment.ARMS:
            task_id = f"sw0111_train_{arm}_s{seed}"
            tasks.append({"task_id": task_id, "experiment": "SW0111", "stage": "train",
                          "seed": seed, "arm": arm, "depends_on": [f"sw0111_preflight_s{seed}"],
                          "priority": 2, "command": _cmd("--train", seed, "--arm", arm),
                          "output": str(experiment.OUT / f"{arm}_seed{seed}")})
    for seed in experiment.SEEDS:
        tasks.append({"task_id": f"sw0111_evaluate_s{seed}", "experiment": "SW0111",
                      "stage": "evaluate", "seed": seed,
                      "depends_on": [f"sw0111_train_control_s{seed}",
                                     f"sw0111_train_candidate_s{seed}"],
                      "priority": 1, "command": _cmd("--evaluate", seed),
                      "output": str(experiment.ARCHIVE / f"evaluation_seed{seed}/evaluation.json")})
    tasks.append({"task_id": "sw0111_summary", "experiment": "SW0111", "stage": "summarize",
                  "seed": None, "depends_on": [f"sw0111_evaluate_s{s}" for s in experiment.SEEDS],
                  "priority": 1, "command": _cmd("--summarize"),
                  "output": str(experiment.ARCHIVE / "summary.json")})
    return tasks


def _finite_metric_block(block):
    if set(block.get("mean", {})) != set(experiment.METRICS):
        return False
    if set(block.get("valid_count", {})) != set(experiment.METRICS):
        return False
    if set(block.get("per_image", {})) != set(experiment.METRICS):
        return False
    for metric in experiment.METRICS:
        vals = block["per_image"][metric]
        if (not isinstance(vals, list) or len(vals) != 320
                or block["valid_count"][metric] != 320
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vals)
                or not isinstance(block["mean"][metric], (int, float))
                or not math.isfinite(block["mean"][metric])
                or abs(sum(vals) / 320 - block["mean"][metric]) > 1e-12):
            return False
    return True


def valid_result(task):
    try:
        stage, seed = task["stage"], task.get("seed")
        if stage == "preflight":
            p = Path(task["output"])
            pf = json.loads(p.read_text())
            _, ids, source, source_sha, control_sha, source_manifest = experiment.verify_contract(seed)
            accepted = (pf.get("status") == "passed" and pf.get("seed") == seed
                    and pf.get("source_core_sha256") == experiment.sha(source)
                    and pf.get("matched_control_core_sha256") == control_sha
                    and pf.get("source_manifest_sha256") == experiment.sha(source_manifest)
                    and pf.get("training_ids") == ids.tolist()
                    and pf.get("implementation_sha256") == experiment.implementation_fingerprint()
                    and pf.get("ground_truth_used") is False
                    and pf.get("throwaway_split_optimizer_update") is True
                    and math.isfinite(float(pf.get("lambda"))) and float(pf["lambda"]) > 0
                    and Path(pf.get("warmup_artifact", "")).is_file()
                    and experiment.sha(pf["warmup_artifact"]) == pf.get("warmup_artifact_sha256"))
            if accepted and seed > 0:
                pf0 = json.loads((experiment.ARCHIVE / "preflight_seed0.json").read_text())
                accepted = (pf0.get("status") == "passed"
                            and pf.get("lambda") == pf0.get("lambda"))
            return accepted
        if stage == "train":
            folder = Path(task["output"])
            manifest = json.loads((folder / "manifest.json").read_text())
            core_path = folder / "core.pt"
            _, ids, source, source_sha, control_sha, source_manifest = experiment.verify_contract(seed)
            return ((folder / "TRAINING_COMPLETED").is_file() and core_path.is_file()
                    and manifest.get("status") == "complete" and manifest.get("seed") == seed
                    and manifest.get("arm") == task["arm"] and manifest.get("updates") == 256
                    and manifest.get("training_ids") == ids.tolist()
                    and manifest.get("source_core_sha256") == source_sha
                    and manifest.get("source_manifest_sha256") == experiment.sha(source_manifest)
                    and manifest.get("matched_control_core_sha256") == control_sha
                    and manifest.get("core_sha256") == experiment.sha(core_path)
                    and manifest.get("ground_truth_used_for_training") is False
                    and manifest.get("runner_sha256") == experiment.sha(HERE / "run.py"))
        if stage == "evaluate":
            report_path = Path(task["output"])
            report = json.loads(report_path.read_text())
            frozen_path = report_path.parent / "frozen_predictions.pt"
            reference_path, reference_sha, _ = experiment.registered_source_evaluation(seed)
            from collaborative_test.SW_0110_xy_graph_route import run as common
            source = common.source_paths(seed)[0]
            control = experiment.OUT / f"control_seed{seed}/core.pt"
            candidate = experiment.OUT / f"candidate_seed{seed}/core.pt"
            recorded_checkpoints = report.get("checkpoint_sha256", {})
            return (report.get("status") == "complete" and report.get("seed") == seed
                    and report.get("ids") == [1320, 1639] and report.get("count") == 320
                    and report.get("ground_truth_used_during_prediction") is False
                    and report.get("source_reference_sha256") == reference_sha
                    and report.get("source_reference_check") is not None
                    and all(v.get("pass") is True for v in report["source_reference_check"].values())
                    and report.get("implementation_sha256") == experiment.implementation_fingerprint()
                    and recorded_checkpoints == {"source": experiment.sha(source),
                                                 "control": experiment.sha(control),
                                                 "candidate": experiment.sha(candidate)}
                    and frozen_path.is_file()
                    and report.get("frozen_predictions_sha256") == experiment.sha(frozen_path)
                    and report.get("scores") is not None
                    and set(report["scores"]) == {"source", "control", "candidate"}
                    and all(_finite_metric_block(b) for b in report["scores"].values()))
        if stage == "summarize":
            p = Path(task["output"])
            saved = json.loads(p.read_text())
            from summarize import summarize
            return saved == summarize()
        return False
    except (OSError, ValueError, KeyError, TypeError, AssertionError, OverflowError):
        return False
