#!/usr/bin/env python3
"""Validate and summarize one SW0054 count-32 pilot without causal overclaim."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
AUC_SIGNALS = ("phase", "gate", "carrier", "h_wave", "membrane", "spike")
PILOT_CONDITIONS = (
    "normal", "gate_perm_s0", "carrier_perm_s0", "gate_mean", "carrier_mean", "K0"
)
CHECKPOINT_SUFFIX = "SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt"


def _finite(value: Any, where: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {where}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{where}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _finite(child, f"{where}[{i}]")


def validate_report(r: dict[str, Any]) -> None:
    if r.get("experiment") != "SW0054 causal mechanism ablation":
        raise ValueError("unexpected SW0054 experiment/schema")
    if not str(r.get("checkpoint", "")).replace("\\", "/").endswith(CHECKPOINT_SUFFIX):
        raise ValueError("unexpected SW0054 checkpoint")
    digest = r.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("missing/invalid checkpoint SHA-256")
    if r.get("count") != 32 or r.get("ids") != [1320, 1351]:
        raise ValueError("expected pilot count=32 over IDs 1320-1351")
    if r.get("core", {}).get("steps") != 256 or r.get("core", {}).get("settle") != 64:
        raise ValueError("expected T256/settle64 pilot")
    if r.get("interventions", {}).get("condition_set") != "pilot":
        raise ValueError("expected condition_set=pilot")
    if r.get("target", {}).get("ground_truth_used_for_prediction") is not False:
        raise ValueError("target-use protocol missing or violated")
    if r.get("target", {}).get("ids") != [1320, 1351]:
        raise ValueError("target IDs do not match pilot range")
    conditions = r.get("conditions", {})
    if set(conditions) != set(PILOT_CONDITIONS):
        raise ValueError("pilot condition set mismatch")
    normal = conditions["normal"]
    for name in PILOT_CONDITIONS:
        row = conditions[name]
        for readout in ("spike_cc", "membrane_spatial"):
            metrics = row.get("fixed_readouts", {}).get(readout, {}).get("metrics", {})
            if not set(METRICS).issubset(metrics):
                raise ValueError(f"{name}/{readout} lacks required metrics")
        auc = row.get("distance_controlled_macro_auc", {})
        if not set(AUC_SIGNALS).issubset(auc):
            raise ValueError(f"{name} lacks AUC signals")
        activity = row.get("activity_scale", {})
        for field in ("spike_event_rate_mean", "membrane_temporal_variance_mean", "membrane_abs_mean"):
            if field not in activity:
                raise ValueError(f"{name} lacks activity statistic {field}")
        if name.startswith("gate_") or name.startswith("carrier_"):
            invariant_rows = row.get("gate_invariants", [])
            if not invariant_rows or not all(x.get("graph_bitwise_equal") and x.get("theta_allclose") for x in invariant_rows):
                raise ValueError(f"{name} did not preserve graph/theta invariants")
    if conditions["K0"].get("kuramoto_K_during_rollout") != 0.0:
        raise ValueError("K0 arm did not report K=0")
    _finite(r)


def summarize(r: dict[str, Any]) -> dict[str, Any]:
    validate_report(r)
    normal = r["conditions"]["normal"]
    deltas: dict[str, Any] = {}
    for name, row in r["conditions"].items():
        if name == "normal":
            continue
        fixed = {}
        for readout in ("spike_cc", "membrane_spatial"):
            base = normal["fixed_readouts"][readout]["metrics"]
            value = row["fixed_readouts"][readout]["metrics"]
            fixed[readout] = {m: float(value[m] - base[m]) for m in METRICS}
        auc = {}
        for signal in AUC_SIGNALS:
            a = normal["distance_controlled_macro_auc"][signal]["macro_distance_auc"]
            b = row["distance_controlled_macro_auc"][signal]["macro_distance_auc"]
            auc[signal] = None if a is None or b is None else float(b - a)
        base_activity, activity = normal["activity_scale"], row["activity_scale"]
        ratios = {}
        for key, out_key in (("spike_event_rate_mean", "spike_event_rate_ratio"),
                             ("membrane_temporal_variance_mean", "membrane_variance_ratio"),
                             ("membrane_abs_mean", "membrane_abs_mean_ratio")):
            denominator = float(base_activity[key])
            ratios[out_key] = None if denominator == 0 else float(activity[key] / denominator)
        deltas[name] = {
            "fixed_readout_metric_delta_vs_normal": fixed,
            "distance_controlled_macro_auc_delta_vs_normal": auc,
            "activity_scale_ratios_vs_normal": ratios,
        }
    return {
        "experiment": "SW0054 causal mechanism ablation summary",
        "source_checkpoint": r["checkpoint"],
        "checkpoint_sha256": r["checkpoint_sha256"],
        "count": 32, "ids": [1320, 1351], "steps": 256, "settle": 64,
        "condition_set": "pilot", "ground_truth_used_for_prediction": False,
        "normal_reference": normal,
        "intervention_deltas": deltas,
        "interpretation": "Numerical deltas only; this summary does not infer whether an effect is explained by activity-scale collapse.",
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# SW0054 count-32 pilot summary", "",
             f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`  ",
             "Protocol: IDs 1320-1351, T256/settle64, pilot condition set, GT-free prediction.", "",
             "Deltas are intervention minus normal. Activity values are ratios to normal.", "",
             "| Condition | Readout | FG-ARI delta | FG IoU delta | Matched IoU delta | Spike-rate ratio | Membrane-variance ratio | Membrane-abs ratio |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition, row in summary["intervention_deltas"].items():
        ratios = row["activity_scale_ratios_vs_normal"]
        for readout, metrics in row["fixed_readout_metric_delta_vs_normal"].items():
            lines.append(f"| {condition} | {readout} | {metrics['fg_ari']:.6f} | {metrics['foreground_iou']:.6f} | {metrics['matched_object_iou']:.6f} | {ratios['spike_event_rate_ratio']} | {ratios['membrane_variance_ratio']} | {ratios['membrane_abs_mean_ratio']} |")
    lines += ["", "## Distance-controlled AUC delta", "", "| Condition | Phase | Gate | Carrier | H-wave | Membrane | Spike |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition, row in summary["intervention_deltas"].items():
        auc = row["distance_controlled_macro_auc_delta_vs_normal"]
        vals = ["n/a" if auc[s] is None else f"{auc[s]:.6f}" for s in AUC_SIGNALS]
        lines.append(f"| {condition} | " + " | ".join(vals) + " |")
    lines += ["", summary["interpretation"], ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, help="JSON output path; Markdown uses the same stem")
    p.add_argument("--overwrite", action="store_true", help="allow replacing both outputs")
    args = p.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    result = summarize(raw)
    out = args.output or args.input.with_name(args.input.stem + "_summary.json")
    md = out.with_suffix(".md")
    if not args.overwrite and (out.exists() or md.exists()):
        raise FileExistsError(f"refusing to overwrite {out} or {md}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md.write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
