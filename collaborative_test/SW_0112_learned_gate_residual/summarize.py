"""Strict paired validation summary for all SW0112 seeds and arms."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0112_learned_gate_residual"
ARCHIVE = Path(__file__).resolve().parent / "results_archive"
SEEDS = (0, 1, 2)
ARMS = ("control", "gate_candidate")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SLOT = {"fg_ari": 0.7749334666743514,
        "foreground_iou": 0.2035891311522426,
        "matched_object_iou": 0.2069369791521436}


def scores(path):
    report = json.loads(Path(path).read_text())
    if report.get("ids") != [1320, 1639] or report.get("images") != 320:
        raise ValueError(f"invalid full320 evaluation: {path}")
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError(f"GT used for prediction: {path}")
    row = report["sweep"][0]
    if row.get("synchrony_threshold") != 0.5:
        raise ValueError(f"wrong threshold in evaluation: {path}")
    result = {}
    target = row["scored_targets"]["our_hdf5"]
    for metric in METRICS:
        values = np.asarray(target["per_image"][metric], dtype=np.float64)
        mean = float(target["metrics"][metric])
        if (len(values) != 320 or target["valid_count"].get(metric) != 320
                or not np.isfinite(values).all() or not math.isfinite(mean)
                or abs(float(values.mean()) - mean) > 1e-12):
            raise ValueError(f"invalid metric vector/mean for {metric}: {path}")
        result[metric] = {"mean": mean, "per_image": values}
    return result


def paired_bootstrap(cand, ref, rng):
    delta = np.stack([cand[s] - ref[s] for s in SEEDS])
    shared_image_delta = delta.mean(axis=0)
    indices = rng.integers(0, len(shared_image_delta), size=(10000, len(shared_image_delta)))
    means = shared_image_delta[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_delta": float(shared_image_delta.mean()), "ci95": [float(lo), float(hi)],
            "lower_bound_positive": bool(lo > 0), "bootstrap_draws": 10000,
            "rng_seed": 113, "same_image_indices_across_seeds": True,
            "conditional_on_three_trained_source_cores": True}


def summarize():
    reference = {s: scores(ROOT / f"collaborative_test/SW_0097_graph_adaptation/results/seed{s}_positive_frozen/evaluation.json")
                 for s in SEEDS}
    results = {arm: {} for arm in ARMS}
    refs = {}
    for seed in SEEDS:
        refs[str(seed)] = {}
        for arm in ARMS:
            folder = OUT / f"seed{seed}_{arm}"
            train = json.loads((folder / "manifest.json").read_text())
            if (train.get("status") != "training_complete" or train.get("seed") != seed
                    or train.get("arm") != arm or train.get("updates") != 256
                    or train.get("batch_size") != 16 or train.get("ground_truth_used_for_training") is not False
                    or len(train.get("training_ids", [])) != 4096):
                raise ValueError(f"training manifest failed contract: {folder}")
            sidecar = json.loads((folder / "evaluation_manifest.json").read_text())
            if (sidecar.get("status") != "complete" or sidecar.get("seed") != seed
                    or sidecar.get("arm") != arm or sidecar.get("ground_truth_used_for_prediction") is not False):
                raise ValueError(f"evaluation sidecar failed contract: {folder}")
            results[arm][seed] = scores(folder / "evaluation.json")
            refs[str(seed)][arm] = {"train_manifest": str(folder / "manifest.json"),
                "evaluation": str(folder / "evaluation.json"),
                "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
                "source_core_sha256": train.get("source_core_sha256"),
                "training_ids_sha256": train.get("training_ids_sha256")}
        if (refs[str(seed)]["control"]["source_core_sha256"] !=
                refs[str(seed)]["gate_candidate"]["source_core_sha256"]
                or refs[str(seed)]["control"]["training_ids_sha256"] !=
                refs[str(seed)]["gate_candidate"]["training_ids_sha256"]):
            raise ValueError(f"matched SW0112 arms differ in source or training IDs for seed{seed}")
    mean = {arm: {metric: float(np.mean([results[arm][s][metric]["mean"] for s in SEEDS]))
                  for metric in METRICS} for arm in ARMS}
    mean["SW0097_source"] = {metric: float(np.mean([reference[s][metric]["mean"] for s in SEEDS]))
                             for metric in METRICS}
    per_seed = {str(s): {refname: {metric: results["gate_candidate"][s][metric]["mean"] - ref[s][metric]["mean"]
                                   for metric in METRICS}
                         for refname, ref in (("matched_control", results["control"]),
                                              ("initial_SW0097", reference))}
                for s in SEEDS}
    boot = {}
    for refname, ref in (("matched_control", results["control"]), ("initial_SW0097", reference)):
        boot[refname] = {}
        for metric in METRICS:
            rng = np.random.default_rng(113)
            boot[refname][metric] = paired_bootstrap(
                {s: results["gate_candidate"][s][metric]["per_image"] for s in SEEDS},
                {s: ref[s][metric]["per_image"] for s in SEEDS}, rng)
    gates = {"mean_fg_gain_vs_control_ge_0_01": mean["gate_candidate"]["fg_ari"] - mean["control"]["fg_ari"] >= 0.01,
             "mean_fg_gain_vs_SW0097_ge_0_01": mean["gate_candidate"]["fg_ari"] - mean["SW0097_source"]["fg_ari"] >= 0.01,
             "at_least_two_seed_gains_vs_control": sum(per_seed[str(s)]["matched_control"]["fg_ari"] > 0 for s in SEEDS) >= 2,
             "at_least_two_seed_gains_vs_SW0097": sum(per_seed[str(s)]["initial_SW0097"]["fg_ari"] > 0 for s in SEEDS) >= 2,
             "paired_ci_positive_vs_control": boot["matched_control"]["fg_ari"]["lower_bound_positive"],
             "paired_ci_positive_vs_SW0097": boot["initial_SW0097"]["fg_ari"]["lower_bound_positive"],
             "foreground_iou_at_least_slot_plus_0_05": mean["gate_candidate"]["foreground_iou"] >= SLOT["foreground_iou"] + 0.05,
             "matched_object_iou_at_least_slot_plus_0_05": mean["gate_candidate"]["matched_object_iou"] >= SLOT["matched_object_iou"] + 0.05}
    return {"status": "complete", "experiment": "SW0112_learned_gate_residual",
            "ids": [1320, 1639], "images": 320, "ground_truth_used_for_prediction": False,
            "means": {**mean, "Slot": SLOT}, "per_seed_delta": per_seed,
            "paired_image_bootstrap": boot, "promotion_gates": gates,
            "promotion_passed": all(gates.values()), "source_paths": refs}


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
