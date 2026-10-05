#!/usr/bin/env python3
"""Fixed-threshold three-seed SW0058 versus SW0053 mixed-loss summary."""
import argparse
import json
import math
import statistics
from pathlib import Path

SEEDS = (0, 1, 2)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
HERE = Path(__file__).resolve().parent


def fixed_spike_metrics(report):
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("prediction must be target-independent")
    if report.get("ids") != [1320, 1639] or report.get("images") != 320:
        raise ValueError("expected IDs1320-1639/count320")
    if report.get("inference", {}).get("steps") != 1024 or report.get("inference", {}).get("settle") != 512:
        raise ValueError("primary endpoint must be long T1024/settle512")
    rows = [row for row in report.get("sweep", [])
            if row.get("affinity_mode") == "spike" and row.get("synchrony_threshold") == 0.50]
    if len(rows) != 1:
        raise ValueError("primary threshold .50 must appear exactly once")
    metrics = rows[0].get("scored_targets", {}).get("our_hdf5", {}).get("metrics", {})
    values = {metric: float(metrics[metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite primary metric")
    return values


def fixed_multireadout_metrics(report, seed):
    if report.get("experiment") != "SW0057 fixed multi-readout" or report.get("seed") != seed:
        raise ValueError("unexpected SW0057 report/seed")
    if report.get("ids") != [1320, 1639] or report.get("images") != 320:
        raise ValueError("SW0057 report split mismatch")
    if report.get("inference", {}).get("steps") != 1024 or report["inference"].get("settle") != 512:
        raise ValueError("SW0057 report window mismatch")
    if report.get("readout_contract", {}).get("ground_truth_used_for_prediction") is not False:
        raise ValueError("SW0057 predictions must be GT-free")
    rows = {row.get("readout"): row for row in report.get("rows", [])}
    expected = {"spike_cc_threshold_0p50", "membrane_spatial_sigma1p5_k10"}
    if set(rows) != expected:
        raise ValueError("SW0057 fixed readout set mismatch")
    out = {}
    for name, row in rows.items():
        out[name] = {metric: float(row["metrics"][metric]) for metric in METRICS}
        if not all(math.isfinite(value) for value in out[name].values()):
            raise ValueError("non-finite SW0057 readout metric")
    return out


def load_baseline(path, seed):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("ids") != [1320, 1639] or report.get("images") != 320:
        raise ValueError(f"SW0053 baseline split mismatch: {path}")
    if report.get("inference", {}).get("steps") != 1024 or report["inference"].get("settle") != 512:
        raise ValueError(f"SW0053 baseline window mismatch: {path}")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError(f"SW0053 baseline prediction provenance mismatch: {path}")
    # SW0052 holds the matched low-LR seed2 epoch-25 mixed-loss checkpoint;
    # SW0053 seed2_baseline_lr is the separate LR=1e-3 stopping control.
    matches = [row for row in report.get("sweep", [])
               if row.get("affinity_mode") == "spike" and row.get("synchrony_threshold") == 0.50]
    if len(matches) != 1:
        raise ValueError(f"missing fixed .50 mixed-loss baseline row: {path}")
    metrics = matches[0].get("scored_targets", {}).get("our_hdf5", {}).get("metrics", {})
    result = {metric: float(metrics[metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"non-finite mixed-loss baseline: {path}")
    if seed == 2 and "SW_0052_checkpoint_trajectory" not in str(path):
        raise ValueError("seed2 baseline must be SW0052 low-LR epoch25, not LR1e-3 control")
    return result, report


def canonical_argv(argv):
    """Ignore only seed and output path; compare every other CLI token exactly."""
    normalized = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--seed", "--save-path"):
            if index + 1 >= len(argv):
                raise ValueError(f"missing value for {token} in training argv")
            index += 2
            continue
        normalized.append(token)
        index += 1
    return tuple(normalized)


def summarize(model_root: Path):
    candidate_spike = {}
    candidate_readouts = {}
    baseline = {}
    recipe_signature = None
    gamma_sha = None
    code_sha = None
    for seed in SEEDS:
        out = model_root / f"SW_0058_component_spike_only_s{seed}_lr0p0003_epoch25"
        if not (out / "TRAINING_COMPLETED").is_file() or not (out / "EVALUATION_COMPLETED").is_file():
            raise FileNotFoundError(f"seed{seed} lacks verified train/evaluation completion")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("seed") != seed or manifest.get("experiment") != "SW0058 component-spike-only loss":
            raise ValueError(f"seed{seed} manifest identity mismatch")
        if manifest.get("gamma", {}).get("train_ids") != [0, 999] or manifest.get("validation_ids") != [1320, 1639]:
            raise ValueError(f"seed{seed} data split mismatch")
        intervention = manifest.get("intervention", {})
        if intervention.get("primary_loss_weight") != 0.0 or intervention.get("spike_plv_weight") != 5.0:
            raise ValueError(f"seed{seed} loss intervention mismatch")
        recipe = manifest.get("baseline", {})
        if recipe.get("lr") != 0.0003 or recipe.get("epochs") != 25 or recipe.get("spike_plv_weight") != 5.0:
            raise ValueError(f"seed{seed} recipe is not matched to SW0053 low-LR epoch25")
        signature = (manifest["gamma"]["sha256"], manifest.get("code_sha256"),
                    canonical_argv(manifest.get("exact_training_argv", [])))
        if gamma_sha is None:
            gamma_sha, code_sha, recipe_signature = signature
        elif signature != (gamma_sha, code_sha, recipe_signature):
            raise ValueError("training data/code/recipe differ across seeds")
        spike_report = json.loads((out / "fixed_spike_long_full_n320_T1024_settle512.json").read_text(encoding="utf-8"))
        provenance = json.loads((out / "fixed_spike_long_full_n320_T1024_settle512.provenance.json").read_text(encoding="utf-8"))
        candidate_spike[seed] = fixed_spike_metrics(spike_report)
        checkpoint_sha = provenance.get("checkpoint_sha256")
        if checkpoint_sha != manifest.get("checkpoint_sha256"):
            raise ValueError(f"seed{seed} checkpoint SHA mismatch across manifest/evaluation")
        if provenance.get("gamma_sha256") == "" or provenance.get("evaluator_code_sha256") == "":
            raise ValueError(f"seed{seed} missing evaluation provenance")
        multi = json.loads((out / "sw0057_fixed_multireadout_long_full_n320.json").read_text(encoding="utf-8"))
        candidate_readouts[seed] = fixed_multireadout_metrics(multi, seed)

        if seed == 0:
            baseline_path = HERE.parents[1] / "SW_0053_lr_early_stop_matrix" / "results" / "seed0_low_lr" / "validation_long_T1024_settle512.json"
        elif seed == 1:
            baseline_path = HERE.parents[1] / "SW_0053_lr_early_stop_matrix" / "results" / "seed1_low_lr" / "validation_long_T1024_settle512.json"
        else:
            baseline_path = HERE.parents[1] / "SW_0052_checkpoint_trajectory" / "results" / "low_lr_epoch25_long_T1024_settle512.json"
        baseline[seed], baseline_report = load_baseline(baseline_path, seed)
        if candidate_spike[seed].keys() != baseline[seed].keys():
            raise ValueError("candidate/baseline metric schema differs")

    result = {
        "schema_version": 1,
        "experiment": "SW0058 component-spike-only versus SW0053 mixed-loss",
        "candidate_recipe": {"primary_loss_weight": 0.0, "spike_plv_weight": 5.0,
                             "lr": 0.0003, "epochs": 25, "train_ids": [0, 999]},
        "matched_baseline": {"experiment": "SW0053 mixed loss (seed2 low-LR epoch25 source from SW0052)",
                             "primary_loss_weight": 1.0, "spike_plv_weight": 5.0},
        "validation": {"ids": [1320, 1639], "count": 320, "steps": 1024,
                       "settle": 512, "fixed_spike_threshold": 0.50,
                       "threshold_postselection": False,
                       "ground_truth_used_for_prediction": False},
        "seeds": list(SEEDS), "gamma_sha256": gamma_sha, "training_code_sha256": code_sha,
        "primary_spike_cc": {}, "fixed_secondary_readouts": {},
    }
    for metric in METRICS:
        cand = [candidate_spike[seed][metric] for seed in SEEDS]
        base = [baseline[seed][metric] for seed in SEEDS]
        delta = [a - b for a, b in zip(cand, base)]
        result["primary_spike_cc"][metric] = {
            "candidate_per_seed": cand, "candidate_mean": statistics.mean(cand),
            "candidate_sample_sd": statistics.stdev(cand),
            "mixed_baseline_per_seed": base, "mixed_baseline_mean": statistics.mean(base),
            "mixed_baseline_sample_sd": statistics.stdev(base),
            "paired_delta_per_seed": delta, "paired_delta_mean": statistics.mean(delta),
        }
    readouts = sorted({name for seed in SEEDS for name in candidate_readouts[seed]})
    for readout in readouts:
        result["fixed_secondary_readouts"][readout] = {}
        for metric in METRICS:
            vals = [candidate_readouts[s][readout][metric] for s in SEEDS]
            result["fixed_secondary_readouts"][readout][metric] = {
                "per_seed": vals, "mean": statistics.mean(vals),
                "sample_sd": statistics.stdev(vals),
            }
    return result


def markdown(result):
    lines = ["# SW0058 three-seed fixed-endpoint comparison", "",
             "Primary: long T1024/settle512, spike connected components at the preregistered threshold .50.",
             "Candidate and SW0053 mixed-loss baseline use the same 1,000 training scenes, low LR, and 25 epochs.",
             "No validation threshold selection was performed.", "",
             "| Metric | Candidate mean (SD) | SW0053 mixed mean (SD) | Paired delta mean | Candidate seeds 0/1/2 | Baseline seeds 0/1/2 |",
             "|---|---:|---:|---:|---|---|"]
    for metric, row in result["primary_spike_cc"].items():
        lines.append(f"| {metric} | {row['candidate_mean']:.6f} ({row['candidate_sample_sd']:.6f}) | "
                     f"{row['mixed_baseline_mean']:.6f} ({row['mixed_baseline_sample_sd']:.6f}) | "
                     f"{row['paired_delta_mean']:.6f} | "
                     + ", ".join(f"{v:.6f}" for v in row["candidate_per_seed"]) + " | "
                     + ", ".join(f"{v:.6f}" for v in row["mixed_baseline_per_seed"]) + " |")
    lines += ["", "## Fixed secondary readouts", "",
              "These are fixed SW0057 readouts on the same long-window validation inputs; they do not alter the primary endpoint.", "",
              "| Readout | Metric | Mean (sample SD) | Seed0 | Seed1 | Seed2 |", "|---|---|---:|---:|---:|---:|"]
    for readout, metrics in result["fixed_secondary_readouts"].items():
        for metric, row in metrics.items():
            lines.append(f"| {readout} | {metric} | {row['mean']:.6f} ({row['sample_sd']:.6f}) | "
                         + " | ".join(f"{x:.6f}" for x in row["per_seed"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=HERE / "results" / "sw0058_three_seed_comparison.json")
    args = p.parse_args()
    md = args.output.with_suffix(".md")
    if args.output.exists() or md.exists():
        raise FileExistsError(f"refusing overwrite: {args.output} or {md}")
    result = summarize(args.model_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    with md.open("x", encoding="utf-8") as f:
        f.write(markdown(result))
    print(f"wrote {args.output} and {md}")


if __name__ == "__main__":
    main()
