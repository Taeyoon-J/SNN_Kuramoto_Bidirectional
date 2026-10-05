#!/usr/bin/env python3
"""Validate and summarize the preregistered SW0058 seed0/1 stopping screen."""
import json
import math
from pathlib import Path

from summarize import METRICS, fixed_multireadout_metrics, fixed_spike_metrics, load_baseline

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    output = RESULTS / "stage1_seed0_seed1_stop.json"
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {markdown}")
    candidate, baseline, secondary = {}, {}, {}
    for seed in (0, 1):
        folder = RESULTS / f"seed{seed}"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["seed"] == seed
        assert manifest["gamma"]["train_ids"] == [0, 999]
        assert manifest["intervention"]["primary_loss_weight"] == 0.0
        assert manifest["intervention"]["spike_plv_weight"] == 5.0
        report = json.loads((folder / "fixed_spike_long_full_n320_T1024_settle512.json").read_text(encoding="utf-8"))
        provenance = json.loads((folder / "fixed_spike_long_full_n320_T1024_settle512.provenance.json").read_text(encoding="utf-8"))
        candidate[seed] = fixed_spike_metrics(report)
        assert provenance["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        multi = json.loads((folder / "sw0057_fixed_multireadout_long_full_n320.json").read_text(encoding="utf-8"))
        secondary[seed] = fixed_multireadout_metrics(multi, seed)
        baseline_path = (HERE.parent / "SW_0053_lr_early_stop_matrix" / "results" /
                         f"seed{seed}_low_lr" / "validation_long_T1024_settle512.json")
        baseline[seed], _ = load_baseline(baseline_path, seed)

    summary = {
        "schema_version": 1,
        "experiment": "SW0058 component-spike-only stage1 stopping screen",
        "seeds_completed": [0, 1],
        "seed2_launched": False,
        "candidate": {"primary_loss_weight": 0.0, "spike_plv_weight": 5.0,
                      "train_ids": [0, 999], "epochs": 25, "lr": 0.0003},
        "baseline": {"experiment": "SW0053 mixed loss", "primary_loss_weight": 1.0,
                     "spike_plv_weight": 5.0},
        "endpoint": {"ids": [1320, 1639], "steps": 1024, "settle": 512,
                     "spike_threshold": 0.50, "postselection": False},
        "primary": {},
        "secondary_candidate": secondary,
        "decision": "stop before seed2; candidate is lower on all three primary metrics for both completed seeds",
    }
    for metric in METRICS:
        cand = [candidate[seed][metric] for seed in (0, 1)]
        base = [baseline[seed][metric] for seed in (0, 1)]
        delta = [c - b for c, b in zip(cand, base)]
        if not all(math.isfinite(value) for value in cand + base + delta):
            raise ValueError(f"non-finite {metric}")
        if not all(value < 0 for value in delta):
            raise ValueError(f"stopping rule not satisfied for {metric}: {delta}")
        summary["primary"][metric] = {
            "candidate_seed0_seed1": cand,
            "baseline_seed0_seed1": base,
            "paired_delta_seed0_seed1": delta,
            "candidate_mean": sum(cand) / 2,
            "baseline_mean": sum(base) / 2,
            "paired_delta_mean": sum(delta) / 2,
        }
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# SW0058 stage-1 stopping result", "",
        "Fixed validation IDs 1320-1639; T1024/settle512; spike CC threshold .50.",
        "The candidate and baseline use the same 1,000-scene gamma, LR, epochs, and optimizer recipe.", "",
        "| Metric | Candidate s0 | Baseline s0 | Delta s0 | Candidate s1 | Baseline s1 | Delta s1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, row in summary["primary"].items():
        c, b, d = row["candidate_seed0_seed1"], row["baseline_seed0_seed1"], row["paired_delta_seed0_seed1"]
        lines.append(f"| {metric} | {c[0]:.6f} | {b[0]:.6f} | {d[0]:.6f} | {c[1]:.6f} | {b[1]:.6f} | {d[1]:.6f} |")
    lines += ["", "Seed2 was not launched because both completed seeds were lower on every primary metric.",
              "This rejects complete removal of the phase-primary term; it does not reject smaller reweighting or a structural gate-transduction change."]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output} and {markdown}")


if __name__ == "__main__":
    main()
