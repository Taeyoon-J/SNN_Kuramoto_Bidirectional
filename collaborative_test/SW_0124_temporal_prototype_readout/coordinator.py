"""Four-stage-adapter plan for three read-only source preflights and scoring."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0124_temporal_prototype_readout import evaluate

SEEDS = (0, 1, 2)
METRICS = evaluate.METRICS
DEFAULT_ATTEMPT = HERE / "results_archive" / "stage1_attempt_20261009"


def task_plan(attempt_root=DEFAULT_ATTEMPT):
    root = Path(attempt_root)
    tasks = []
    for seed in SEEDS:
        tasks.append({"task_id": f"sw0124_preflight_s{seed}", "stage": "preflight",
                      "seed": seed, "depends_on": [], "priority": 0,
                      "attempt_root": str(root)})
    tasks.append({"task_id": "sw0124_evaluate_three_seed", "stage": "evaluate",
                  "seed": None, "depends_on": [f"sw0124_preflight_s{s}" for s in SEEDS],
                  "priority": 1, "attempt_root": str(root)})
    return tasks


def artifact_path(task):
    root = Path(task["attempt_root"])
    if task["stage"] == "preflight":
        return root / "preflight" / f"preflight_seed{int(task['seed'])}.json"
    if task["stage"] == "evaluate":
        return root / "evaluation" / "summary.json"
    raise ValueError(f"unknown stage {task['stage']!r}")


def command(task, device="cuda:0"):
    root = Path(task["attempt_root"])
    if task["stage"] == "preflight":
        return [sys.executable, str(evaluate.HERE / "evaluate.py"), "--stage", "preflight",
                "--seed", str(task["seed"]), "--device", device,
                "--output", str(artifact_path(task))]
    if task["stage"] == "evaluate":
        return [sys.executable, str(evaluate.HERE / "evaluate.py"), "--stage", "evaluate",
                "--device", device, "--output-dir", str(root / "evaluation"),
                "--preflight-dir", str(root / "preflight")]
    raise ValueError(f"unknown stage {task['stage']!r}")


def _valid_score(score):
    try:
        if (set(score["metrics"]) != set(METRICS)
                or set(score["valid_count"]) != set(METRICS)
                or set(score["per_image"]) != set(METRICS)):
            return False
        for metric in METRICS:
            values = [float(v) for v in score["per_image"][metric]]
            mean = float(score["metrics"][metric])
            if (len(values) != 320 or score["valid_count"][metric] != 320
                    or not all(math.isfinite(value) for value in values)
                    or not math.isfinite(mean)
                    or abs(sum(values) / 320 - mean) > 1e-12):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if task["stage"] == "preflight":
            return (record.get("experiment") == "SW0124"
                    and record.get("stage") == "source_prototype_preflight"
                    and record.get("status") == "passed"
                    and record.get("seed") == task["seed"]
                    and record.get("count") == 8
                    and record.get("image_ids") == [1320, 1327]
                    and record.get("ground_truth_used") is False
                    and record.get("optimizer_updates") == 0
                    and isinstance(record.get("implementation_fingerprint"), dict)
                    and record.get("source_core_sha256") == evaluate.base.EXPECTED_SOURCE_SHAS[task["seed"]]
                    and record.get("source_evaluation_sha256") == evaluate.SOURCE_EVALUATION_SHAS[task["seed"]])
        if task["stage"] == "evaluate":
            if (record.get("experiment") != "SW0124"
                    or record.get("status") not in ("promotion_passed", "promotion_failed")
                    or record.get("seeds") != list(SEEDS)
                    or record.get("gates", {}).get("all_source_qcc_baselines_reproduced") is not True):
                return False
            outdir = path.parent
            for seed in SEEDS:
                report_path = outdir / f"evaluation_seed{seed}.json"
                if not report_path.is_file():
                    return False
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (report.get("status") != "complete" or report.get("experiment") != "SW0124"
                        or report.get("seed") != seed or report.get("ids") != [1320, 1639]
                        or report.get("images") != 320
                        or report.get("source_qcc_baseline_reproduced") is not True
                        or report.get("ground_truth_used_for_prediction") is not False
                        or not _valid_score(report.get("temporal_prototype", {}))
                        or not _valid_score(report.get("production_qcc", {}))):
                    return False
            return True
        return False
    except (OSError, ValueError, TypeError, KeyError):
        return False
