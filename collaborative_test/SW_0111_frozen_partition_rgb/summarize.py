"""Predeclared SW0111 three-seed endpoint and paired-image bootstrap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ARCHIVE, METRICS, SEEDS, sha, write


def _read(seed):
    path = ARCHIVE / f"evaluation_seed{seed}/evaluation.json"
    report = json.loads(path.read_text())
    if (report.get("status") != "complete" or report.get("seed") != seed
            or report.get("count") != 320 or report.get("ids") != [1320, 1639]
            or report.get("ground_truth_used_during_prediction") is not False):
        raise AssertionError(f"invalid evaluation report: {path}")
    scores = report["scores"]
    if set(scores) != {"source", "control", "candidate"}:
        raise AssertionError(f"missing paired arm scores: {path}")
    for arm in scores.values():
        for metric in METRICS:
            vals = np.asarray(arm["per_image"][metric], dtype=np.float64)
            if (vals.shape != (320,) or not np.isfinite(vals).all()
                    or arm["valid_count"][metric] != 320
                    or abs(float(vals.mean()) - float(arm["mean"][metric])) > 1e-12):
                raise AssertionError(f"invalid {metric} values in {path}")
    return report


def _bootstrap_lower(deltas, seed=113, samples=10000):
    # Rows are fixed training seeds; columns are the same 320 paired images.
    matrix = np.stack(deltas, axis=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, matrix.shape[1], size=(samples, matrix.shape[1]))
    boot = matrix[:, draws].mean(axis=(0, 2))
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def summarize():
    reports = {seed: _read(seed) for seed in SEEDS}
    means = {arm: {metric: float(np.mean([reports[s]["scores"][arm]["mean"][metric]
                                          for s in SEEDS]))
                   for metric in METRICS}
             for arm in ("source", "control", "candidate")}
    seed_deltas = {}
    bootstrap = {}
    fg_matrices = {}
    for ref in ("control", "source"):
        seed_deltas[ref] = {}
        for seed in SEEDS:
            seed_deltas[ref][str(seed)] = {
                metric: float(reports[seed]["scores"]["candidate"]["mean"][metric]
                              - reports[seed]["scores"][ref]["mean"][metric])
                for metric in METRICS}
        fg_matrices[ref] = [
            np.asarray(reports[s]["scores"]["candidate"]["per_image"]["fg_ari"], dtype=np.float64)
            - np.asarray(reports[s]["scores"][ref]["per_image"]["fg_ari"], dtype=np.float64)
            for s in SEEDS]
        low, high = _bootstrap_lower(fg_matrices[ref])
        bootstrap[ref] = {"lower_95": low, "upper_95": high, "samples": 10000,
                          "rng_seed": 113, "same_image_indices_across_seeds": True,
                          "conditional_on_fixed_training_seeds": True}
    control_delta = means["candidate"]["fg_ari"] - means["control"]["fg_ari"]
    source_delta = means["candidate"]["fg_ari"] - means["source"]["fg_ari"]
    positive_control = sum(seed_deltas["control"][str(s)]["fg_ari"] > 0 for s in SEEDS)
    positive_source = sum(seed_deltas["source"][str(s)]["fg_ari"] > 0 for s in SEEDS)
    gates = {
        "fg_mean_delta_vs_control_at_least_0p01": control_delta >= .01,
        "fg_mean_delta_vs_source_at_least_0p01": source_delta >= .01,
        "at_least_two_positive_seed_deltas_vs_control": positive_control >= 2,
        "at_least_two_positive_seed_deltas_vs_source": positive_source >= 2,
        "foreground_iou_floor": means["candidate"]["foreground_iou"] >= .25358913,
        "matched_object_iou_floor": means["candidate"]["matched_object_iou"] >= .25693698,
        "bootstrap_lower_bound_vs_control_positive": bootstrap["control"]["lower_95"] > 0,
        "bootstrap_lower_bound_vs_source_positive": bootstrap["source"]["lower_95"] > 0,
    }
    result = {"experiment": "SW0111", "status": "complete", "seeds": list(SEEDS),
              "metrics_mean_across_fixed_seeds": means,
              "fg_ari_mean_deltas": {"vs_control": control_delta, "vs_source": source_delta},
              "positive_seed_counts": {"vs_control": positive_control, "vs_source": positive_source},
              "per_seed_deltas": seed_deltas, "paired_image_bootstrap": bootstrap,
              "promotion_gates": gates, "promotion_passed": all(gates.values()),
              "evaluation_sha256": {str(s): sha(ARCHIVE / f"evaluation_seed{s}/evaluation.json")
                                    for s in SEEDS},
              "ground_truth_used_for_training": False}
    return result


def main():
    result = summarize()
    out = ARCHIVE / "summary.json"
    if out.exists():
        raise FileExistsError(f"preserve existing summary: {out}")
    write(out, result)
    print(json.dumps({"status": "complete", "summary": str(out),
                      "promotion_passed": result["promotion_passed"]}), flush=True)


if __name__ == "__main__":
    main()
