"""Validate and summarize preregistered SW0110 paired full320 evaluations."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0110_xy_graph_route"
ARCHIVE = Path(__file__).resolve().parent / "results_archive"
SEEDS = (0, 1, 2)
ARMS = ("control", "xy_candidate")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SLOT = {"fg_ari": 0.7749334666743514,
        "foreground_iou": 0.2035891311522426,
        "matched_object_iou": 0.2069369791521436}
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 113


def read(path):
    return json.loads(Path(path).read_text())


def report_data(path):
    data = read(path)
    if data.get("ids") != [1320, 1639] or data.get("images") != 320:
        raise ValueError(f"not the registered full320 split: {path}")
    if data.get("ground_truth_used_for_prediction") is not False:
        raise ValueError(f"prediction used GT or lacks a false marker: {path}")
    if len(data.get("sweep", [])) != 1 or data["sweep"][0].get("synchrony_threshold") != 0.5:
        raise ValueError(f"wrong readout or threshold: {path}")
    score = data["sweep"][0]["scored_targets"]["our_hdf5"]
    values = {}
    for metric in METRICS:
        arr = np.asarray(score["per_image"][metric], dtype=np.float64)
        mean = float(score["metrics"][metric])
        if (len(arr) != 320 or not np.isfinite(arr).all()
                or score["valid_count"].get(metric) != 320 or not math.isfinite(mean)):
            raise ValueError(f"invalid per-image metric {metric}: {path}")
        actual = float(arr.mean())
        if abs(actual - mean) > 1e-12:
            raise ValueError(f"reported mean does not recompute for {metric}: {path}")
        values[metric] = {"mean": mean, "per_image": arr}
    return data, values


def paired_bootstrap(candidate, reference, seed=BOOTSTRAP_SEED, count=N_BOOTSTRAP):
    # Shape [3,320]: the same sampled image indices are used for every seed.
    delta = np.stack([candidate[s] - reference[s] for s in SEEDS], axis=0)
    image_delta = delta.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(image_delta), size=(count, len(image_delta)))
    sample_means = image_delta[draws].mean(axis=1)
    lo, hi = np.percentile(sample_means, [2.5, 97.5])
    return {"mean_delta": float(image_delta.mean()), "ci95": [float(lo), float(hi)],
            "lower_bound_positive": bool(lo > 0), "bootstrap_count": count,
            "bootstrap_seed": seed, "paired_ids": [1320, 1639],
            "same_image_samples_across_seeds": True,
            "interpretation": "conditional on these three trained source cores"}


def summarize():
    source_reports, source_scores = {}, {}
    control_reports, controls = {}, {}
    candidate_reports, candidates = {}, {}
    record_refs = {}
    for seed in SEEDS:
        source_path = ROOT / f"collaborative_test/SW_0097_graph_adaptation/results/seed{seed}_positive_frozen/evaluation.json"
        source_reports[seed], source_scores[seed] = report_data(source_path)
        record_refs[str(seed)] = {"source_evaluation": str(source_path)}
        control_reports[seed], controls[seed] = {}, {}
        candidate_reports[seed], candidates[seed] = {}, {}
        for arm in ARMS:
            folder = OUT / f"seed{seed}_{arm}"
            train_manifest = read(folder / "manifest.json")
            if (train_manifest.get("status") != "training_complete"
                    or train_manifest.get("seed") != seed
                    or train_manifest.get("arm") != arm
                    or train_manifest.get("updates") != 256
                    or train_manifest.get("batch_size") != 16
                    or train_manifest.get("ground_truth_used_for_training") is not False):
                raise ValueError(f"training manifest fails registered contract: {folder}")
            if len(train_manifest.get("training_ids", [])) != 4096:
                raise ValueError(f"training ID record has wrong length: {folder}")
            if arm == "control":
                control_reports[seed], controls[seed] = report_data(folder / "evaluation.json")
            else:
                candidate_reports[seed], candidates[seed] = report_data(folder / "evaluation.json")
            record_refs[str(seed)][arm] = {
                "training_manifest": str(folder / "manifest.json"),
                "evaluation": str(folder / "evaluation.json"),
                "training_ids_sha256": train_manifest.get("training_ids_sha256"),
                "source_core_sha256": train_manifest.get("source_core_sha256")}
        if (record_refs[str(seed)]["control"]["training_ids_sha256"] !=
                record_refs[str(seed)]["xy_candidate"]["training_ids_sha256"]):
            raise ValueError(f"paired control/candidate training IDs differ for seed {seed}")
        if (record_refs[str(seed)]["control"]["source_core_sha256"] !=
                record_refs[str(seed)]["xy_candidate"]["source_core_sha256"]):
            raise ValueError(f"paired source SHA differs for seed {seed}")

    means = {"control": {}, "xy_candidate": {}, "SW0097_source": {}, "Slot": SLOT}
    seed_rows = {}
    for metric in METRICS:
        means["control"][metric] = float(np.mean([controls[s][metric]["mean"] for s in SEEDS]))
        means["xy_candidate"][metric] = float(np.mean([candidates[s][metric]["mean"] for s in SEEDS]))
        means["SW0097_source"][metric] = float(np.mean([source_scores[s][metric]["mean"] for s in SEEDS]))
    for seed in SEEDS:
        seed_rows[str(seed)] = {reference: {
            metric: candidates[seed][metric]["mean"] - tables[seed][metric]["mean"]
            for metric in METRICS}
            for reference, tables in (("matched_control", controls), ("initial_SW0097", source_scores))}
    bootstrap = {}
    for reference, table in (("matched_control", controls), ("initial_SW0097", source_scores)):
        bootstrap[reference] = {}
        for metric in METRICS:
            bootstrap[reference][metric] = paired_bootstrap(
                {s: candidates[s][metric]["per_image"] for s in SEEDS},
                {s: table[s][metric]["per_image"] for s in SEEDS})

    mean_fg_gain_control = means["xy_candidate"]["fg_ari"] - means["control"]["fg_ari"]
    mean_fg_gain_source = means["xy_candidate"]["fg_ari"] - means["SW0097_source"]["fg_ari"]
    count_control = sum(seed_rows[str(s)]["matched_control"]["fg_ari"] > 0 for s in SEEDS)
    count_source = sum(seed_rows[str(s)]["initial_SW0097"]["fg_ari"] > 0 for s in SEEDS)
    gates = {
        "mean_fg_gain_vs_matched_control_ge_0_01": mean_fg_gain_control >= 0.01,
        "mean_fg_gain_vs_SW0097_ge_0_01": mean_fg_gain_source >= 0.01,
        "at_least_two_seed_gains_vs_matched_control": count_control >= 2,
        "at_least_two_seed_gains_vs_SW0097": count_source >= 2,
        "paired_ci_lower_positive_vs_matched_control": bootstrap["matched_control"]["fg_ari"]["lower_bound_positive"],
        "paired_ci_lower_positive_vs_SW0097": bootstrap["initial_SW0097"]["fg_ari"]["lower_bound_positive"],
        "foreground_iou_at_least_slot_plus_0_05": means["xy_candidate"]["foreground_iou"] >= SLOT["foreground_iou"] + 0.05,
        "matched_object_iou_at_least_slot_plus_0_05": means["xy_candidate"]["matched_object_iou"] >= SLOT["matched_object_iou"] + 0.05}
    result = {"status": "complete", "experiment": "SW0110_xy_graph_route",
              "ground_truth_used_for_prediction": False,
              "ids": [1320, 1639], "images": 320, "per_seed_delta": seed_rows,
              "means": means, "paired_image_bootstrap": bootstrap,
              "promotion_gates": gates, "promotion_passed": all(gates.values()),
              "bootstrap_draws_same_image_indices_across_all_seeds": True,
              "bootstrap_conditional_on_fixed_three_source_cores": True,
              "source_and_output_paths": record_refs}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARCHIVE / "summary.json")
    args = parser.parse_args()
    result = summarize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    print(json.dumps({"status": result["status"], "promotion_passed": result["promotion_passed"],
                      "means": result["means"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
