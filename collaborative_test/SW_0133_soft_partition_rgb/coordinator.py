"""Artifact validation and preregistered seed-1 pilot gate for SW0133."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from collaborative_test.SW_0133_soft_partition_rgb import evaluate, run, train

BOOTSTRAP_SEED = 133
BOOTSTRAP_REPLICATES = 10_000
def evaluation_valid(seed, arm, path, training_dir=None):
    """Require a completed, finite, fixed-320 report bound to its own sidecar."""
    path = Path(path)
    sidecar_path = path.parent / "evaluation_manifest.json"
    marker = path.parent / "COMPLETED"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (not marker.is_file() or report.get("status") != "complete"
                or report.get("experiment") != "SW0133_soft_partition_rgb"
                or report.get("seed") != seed or report.get("arm") != arm
                or report.get("images") != 320 or report.get("ids") != [1320, 1639]
                or report.get("time_steps") != 1024 or report.get("settle") != 512
                or report.get("batch_size") != 8
                or report.get("ground_truth_used_for_prediction") is not False
                or report.get("ground_truth_used_for_scoring") is not True
                or sidecar.get("status") != "complete"
                or sidecar.get("experiment") != "SW0133_soft_partition_rgb"
                or sidecar.get("seed") != seed or sidecar.get("arm") != arm
                or sidecar.get("evaluation_sha256") != run.sha(path)
                or sidecar.get("evaluation_fingerprint") != evaluate.evaluation_fingerprint()
                or sidecar.get("frozen_predictions_sha256") != report.get("frozen_predictions_sha256")
                or sidecar.get("gamma_sha256") != report.get("gamma_sha256")):
            return False
        prediction_path = Path(report["frozen_predictions_path"])
        gamma_path = Path(report["gamma_path"])
        if (not prediction_path.is_file() or not gamma_path.is_file()
                or run.sha(prediction_path) != report["frozen_predictions_sha256"]
                or run.sha(gamma_path) != report["gamma_sha256"]):
            return False
        if report.get("readout") != {"affinity_mode": "spike", "threshold": 0.50,
                                     "minimum_group_size": 2,
                                     "background": "largest_component"}:
            return False
        metrics = report["scores"]
        for metric in evaluate.METRICS:
            vals = np.asarray(metrics["per_image"][metric], dtype=np.float64)
            mean = float(metrics["metrics"][metric])
            if (vals.shape != (320,) or not np.isfinite(vals).all()
                    or not math.isfinite(mean) or metrics["valid_count"][metric] != 320
                    or abs(float(vals.mean()) - mean) > 1e-10):
                return False
        manifest = json.loads(Path(report["training_manifest_path"]).read_text(encoding="utf-8"))
        if (run.sha(Path(report["training_manifest_path"])) != report["training_manifest_sha256"]
                or manifest.get("seed") != seed or manifest.get("arm") != arm
                or manifest.get("artifact_sha256") != report.get("training_artifact_sha256")
                or manifest.get("source_core_sha256") != report.get("source_core_sha256")):
            return False
        if training_dir is not None and Path(report["training_manifest_path"]).parent.resolve() != Path(training_dir).resolve():
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return False


def _paired_bootstrap_lower(candidate, reference, *, rng, indices):
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if difference.shape != (320,) or not np.isfinite(difference).all():
        raise ValueError("paired bootstrap requires finite matched 320-image arrays")
    sampled = difference[indices].mean(axis=1)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def pilot_gate(source_metrics, arm_reports):
    """Evaluate only the registered seed-1 comparisons; never imply three-seed promotion."""
    expected_arms = set(run.ARMS)
    if set(arm_reports) != expected_arms:
        raise ValueError("all three completed SW0133 seed-1 arms are required")
    arrays = {arm: arm_reports[arm]["scores"]["per_image"] for arm in expected_arms}
    source_arrays = {}
    for metric in evaluate.METRICS:
        row = source_metrics[metric]
        if isinstance(row, dict):
            values = row.get("per_image")
            if row.get("valid_count") != 320:
                raise ValueError("source97 must provide 320 valid per-image scores")
            mean = float(row.get("mean", float("nan")))
            if not math.isfinite(mean):
                raise ValueError("source97 metric mean must be finite")
        else:  # compact CPU callers may pass already-extracted per-image arrays.
            values = row
            mean = None
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (320,) or not np.isfinite(values).all():
            raise ValueError("source97 needs finite matched fixed-320 metrics")
        if mean is not None and abs(float(values.mean()) - mean) > 1e-10:
            raise ValueError("source97 mean does not match its per-image scores")
        source_arrays[metric] = values
    for item in arrays.values():
        for metric in evaluate.METRICS:
            values = np.asarray(item[metric], dtype=np.float64)
            if values.shape != (320,) or not np.isfinite(values).all():
                raise ValueError("pilot comparison needs matched finite fixed-320 metrics")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, 320, size=(BOOTSTRAP_REPLICATES, 320))
    phase = arrays["phase_live"]
    comparisons = {}
    for reference in ("source97", "constant_live", "phase_detached"):
        ref = source_arrays if reference == "source97" else arrays[reference]
        comparisons[reference] = {}
        for metric in evaluate.METRICS:
            cand = np.asarray(phase[metric], dtype=np.float64)
            base = np.asarray(ref[metric], dtype=np.float64)
            lo, hi = _paired_bootstrap_lower(cand, base, rng=rng, indices=indices)
            comparisons[reference][metric] = {
                "candidate_mean": float(cand.mean()), "reference_mean": float(base.mean()),
                "mean_delta": float((cand - base).mean()),
                "paired_bootstrap_95ci": [lo, hi], "lower_bound_positive": lo > 0,
            }
    checks = {
        "phase_fg_exceeds_source97": comparisons["source97"]["fg_ari"]["candidate_mean"] >
                                      comparisons["source97"]["fg_ari"]["reference_mean"],
        "phase_fg_exceeds_constant_live": comparisons["constant_live"]["fg_ari"]["mean_delta"] > 0,
        "phase_fg_exceeds_phase_detached": comparisons["phase_detached"]["fg_ari"]["mean_delta"] > 0,
        "paired_ci_lower_positive_all_three": all(
            comparisons[ref]["fg_ari"]["lower_bound_positive"]
            for ref in ("source97", "constant_live", "phase_detached")),
    }
    return {"status": "pilot_gate_passed" if all(checks.values()) else "pilot_gate_failed",
            "seed": 1, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "phase_live_means": arm_reports["phase_live"]["scores"]["metrics"],
            "all_arm_means": {arm: report["scores"]["metrics"]
                              for arm, report in arm_reports.items()},
            "comparisons": comparisons, "checks": checks,
            "scope": "single-seed pilot only; no all-seed promotion, gating, or scaling claim"}
