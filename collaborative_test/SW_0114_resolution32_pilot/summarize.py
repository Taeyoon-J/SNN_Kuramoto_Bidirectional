"""Preregistered paired three-seed summary for SW0114 frozen endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import sha

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
SEEDS = (0, 1, 2)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def _section(report, key):
    section = report["scores"][key]
    arrays = {}
    for metric in METRICS:
        if section["valid_count"].get(metric) != 320:
            raise ValueError(f"{key}.{metric} is not valid on all320 IDs")
        values = np.asarray(section["per_image"][metric], dtype=np.float64)
        if values.shape != (320,) or not np.isfinite(values).all():
            raise ValueError(f"{key}.{metric} has invalid per-image values")
        mean = float(values.mean())
        if abs(mean - float(section["mean"][metric])) > 1e-12:
            raise ValueError(f"{key}.{metric} mean does not recompute")
        arrays[metric] = values
    return arrays


def paired_bootstrap(delta, *, n=10000, seed=114):
    """Resample the same image indices jointly across the three trained seeds."""
    values = np.asarray(delta, dtype=np.float64)
    if values.shape != (3, 320) or not np.isfinite(values).all():
        raise ValueError("bootstrap input must be finite [3 seeds,320 shared images]")
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=np.float64)
    for j in range(n):
        ix = rng.integers(0, 320, size=320)
        samples[j] = values[:, ix].mean()
    return {"resamples": n, "seed": seed, "sampling_unit": "shared_image_ids_across_fixed_seeds",
            "mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(samples, [.025, .975])],
            "lower_bound_gt_zero": bool(np.quantile(samples, .025) > 0)}


def summarize(paths=None, output=None):
    paths = paths or {seed: ARCHIVE / f"evaluation_seed{seed}/evaluation.json" for seed in SEEDS}
    reports = {}
    for seed in SEEDS:
        report = json.loads(Path(paths[seed]).read_text())
        if (report.get("status") != "complete" or report.get("seed") != seed
                or report.get("count") != 320 or report.get("ground_truth_used_during_prediction") is not False):
            raise ValueError(f"seed{seed} endpoint/provenance is incomplete")
        reports[seed] = report
    sections = ("source16_vs_modal8", "control16_vs_modal8", "candidate_pooled16_vs_modal8",
                "control_repeated32_vs_modal4", "candidate_native32_vs_modal4")
    arrays = {key: {metric: np.stack([_section(reports[s], key)[metric] for s in SEEDS])
                    for metric in METRICS} for key in sections}
    mean_by_section = {key: {metric: float(values.mean()) for metric, values in items.items()}
                       for key, items in arrays.items()}
    deltas = {}
    for baseline_key, label in (("control16_vs_modal8", "matched_control16"),
                                ("source16_vs_modal8", "original_source16")):
        deltas[label] = {}
        for metric in METRICS:
            delta = arrays["candidate_pooled16_vs_modal8"][metric] - arrays[baseline_key][metric]
            deltas[label][metric] = {
                "per_seed": {str(seed): float(delta[i].mean()) for i, seed in enumerate(SEEDS)},
                "mean": float(delta.mean()),
                "positive_seeds": sum(float(delta[i].mean()) > 0 for i in range(3)),
                "paired_image_bootstrap": paired_bootstrap(delta, seed=114)}
    gate = {
        "candidate_fg_ari_delta_vs_control_at_least_0p01":
            deltas["matched_control16"]["fg_ari"]["mean"] >= .01,
        "candidate_fg_ari_delta_vs_source_at_least_0p01":
            deltas["original_source16"]["fg_ari"]["mean"] >= .01,
        "at_least_two_positive_fg_ari_seeds_vs_both":
            min(deltas["matched_control16"]["fg_ari"]["positive_seeds"],
                deltas["original_source16"]["fg_ari"]["positive_seeds"]) >= 2,
        "candidate_pooled16_mean_foreground_iou_floor":
            mean_by_section["candidate_pooled16_vs_modal8"]["foreground_iou"] >= .25358913,
        "candidate_pooled16_mean_object_iou_floor":
            mean_by_section["candidate_pooled16_vs_modal8"]["matched_object_iou"] >= .25693698,
        "paired_bootstrap_ci_lower_gt_zero_vs_both": all(
            deltas[ref]["fg_ari"]["paired_image_bootstrap"]["lower_bound_gt_zero"]
            for ref in ("matched_control16", "original_source16")),
    }
    result = {"status": "complete", "experiment": "SW0114", "seeds": list(SEEDS),
              "ids": [1320, 1639], "count_per_seed": 320,
              "all_predictions_frozen_before_gt": True,
              "evaluation_sha256": {str(seed): sha(paths[seed]) for seed in SEEDS},
              "metrics": mean_by_section, "candidate_primary_deltas": deltas,
              "promotion_gates": gate, "promotion_passed": all(gate.values()),
              "bootstrap_interpretation": "conditional on three trained cores; not a training-seed population claim"}
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
