"""Audit matched SW0102/SW0097 full-320 metrics from actual evaluator schema."""

import json
import hashlib
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "collaborative_test/SW_0102_cannot_link_draft/results_archive"
CONTROL = (ROOT / "collaborative_test/SW_0100_phasor_imag_raw_gate/results_archive"
           / "trained_models/SW0097_graph_adaptation")
CALIBRATION = ROOT / "collaborative_test/SW_0102_cannot_link_draft/calibration_result.json"
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SLOT_MEAN = {"fg_ari": .7749334667, "foreground_iou": .2035891312,
             "matched_object_iou": .2069369792}


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}: {value}")
    return result


def score_record(path):
    evaluation = read(path)
    if evaluation.get("images") != 320 or evaluation.get("ids") != [1320, 1639]:
        raise AssertionError(f"not the registered 320-image evaluation: {path}")
    # Use the evaluator's authoritative nested score path, not guessed root keys.
    scored = evaluation["sweep"][0]["scored_targets"]["our_hdf5"]
    metric_values = {}
    per_image = {}
    for metric in METRICS:
        if scored["valid_count"][metric] != 320:
            raise AssertionError(f"{path}: {metric} valid_count is not 320")
        metric_values[metric] = finite(scored["metrics"][metric], metric)
        values = scored["per_image"][metric]
        if len(values) != 320:
            raise AssertionError(f"{path}: {metric} has {len(values)} per-image values")
        per_image[metric] = [finite(value, f"{metric} per-image") for value in values]
    return metric_values, per_image


def validate_history(path, manifest, seed):
    history = read(path)
    if len(history) != 256:
        raise AssertionError(f"seed{seed} history length {len(history)} != 256")
    keys = ("loss", "primary", "spike", "cannot_link_cut_loss")
    for row in history:
        for key in keys:
            finite(row[key], f"seed{seed} step{row['step']} {key}")
        if not math.isfinite(row["gradient_norm"]["core"]) or row["gradient_norm"]["core"] <= 0:
            raise AssertionError(f"seed{seed} step{row['step']} has invalid core gradient norm")
        if row["gradient_norm"]["graph"] != 0.0:
            raise AssertionError(f"seed{seed} step{row['step']} updated/used frozen graph gradient")
    if manifest.get("status") != "complete" or manifest.get("steps") != 256:
        raise AssertionError(f"seed{seed} manifest not complete at 256 updates")
    if manifest.get("batch") != 16 or manifest.get("unique_images_seen") != 4096:
        raise AssertionError(f"seed{seed} training data contract mismatch")
    if manifest.get("train_steps") != 64 or manifest.get("train_settle") != 32:
        raise AssertionError(f"seed{seed} training time-window mismatch")
    if any(key.startswith("graph_generator.") for key in manifest.get("changed_core_keys", [])):
        raise AssertionError(f"seed{seed} frozen graph changed")
    return history


def main():
    calibration = read(CALIBRATION)
    if calibration.get("status") != "calibration_complete":
        raise AssertionError("no completed pre-training calibration")
    result = {"status": "complete", "experiment": "SW0102_cannot_link_draft",
              "training": {"source": "SW0095 whole-core seed matched to SW0097 frozen control",
                           "steps": 256, "batch": 16, "unique_ids_per_seed": 4096,
                           "shuffle_seeds": {"0": 117, "1": 118, "2": 119},
                           "train_steps": 64, "settle": 32, "lr": 3e-5,
                           "optimizer": "Adam", "gate_mode": "raw",
                           "encoder_frozen": True, "graph_frozen": True,
                           "ground_truth_used_for_training": False,
                           "loss": "old primary + 5x old positive-product spike criterion + shared-lambda*Lcut",
                           "shared_lambda": calibration["shared_lambda"],
                           "calibration_batches": 12,
                           "calibration_samecc_q_ge_050_by_seed": {
                               seed: row["same_cc_negative_decisions_at_q_ge_0_50"]
                               for seed, row in calibration["seeds"].items()},
                           "calibration_indirect_samecc_q_le_040_by_seed": {
                               seed: row["indirect_same_cc_q_le_0_40_decisions"]
                               for seed, row in calibration["seeds"].items()}},
              "evaluation": {"split": "validation", "ids": [1320, 1639],
                             "images": 320, "batch": 8, "steps": 1024,
                             "settle": 512, "membrane_threshold": .06,
                             "classifier_threshold": .50,
                             "background": "largest_component", "min_group_size": 2,
                             "authoritative_metric_path": "sweep[0].scored_targets.our_hdf5.metrics",
                             "valid_count_required": 320},
              "baseline": {"name": "matched SW0097_positive_frozen controls",
                           "slot_mean": SLOT_MEAN},
              "per_seed": {}, "notes": []}
    launch = read(CANDIDATE / "queue_launch.json")
    result["execution_provenance"] = {
        "queue_started_unix": launch["started_unix"],
        "queue_completed_unix": launch["completed_unix"],
        "workers": [{"seed": row["seed"], "gpu": row["gpu"], "pid": row["pid"],
                     "status": row["status"], "exit_code": row["exit_code"],
                     "source_sha256": row["source_sha256"],
                     "control_source_sha256": row["control_source_sha256"]}
                    for row in launch["workers"]],
        "gpu_owners_before_launch": launch["owners_before_launch"],
        "code_sha256": {name: sha256(Path(__file__).with_name(name))
                        for name in ("forest_loss.py", "run.py", "coordinator.py",
                                     "calibration_preflight.py", "summarize_results.py")},
        "cycle_peer_check": {"time_utc": "2026-10-07 09:37", "head": "e2389f5",
                              "result": "no newer peer commit observed"}}
    candidate_rows = {metric: [] for metric in METRICS}
    control_rows = {metric: [] for metric in METRICS}
    candidate_images = {metric: [] for metric in METRICS}
    control_images = {metric: [] for metric in METRICS}
    for seed in range(3):
        cand_dir = CANDIDATE / f"seed{seed}"
        control_dir = CONTROL / f"seed{seed}_positive_frozen"
        if not (cand_dir / "COMPLETED").is_file():
            raise FileNotFoundError(f"seed{seed} candidate completion marker missing")
        manifest = read(cand_dir / "manifest.json")
        if manifest.get("source_model_seed") != seed or manifest.get("seed") != 117 + seed:
            raise AssertionError(f"seed{seed} candidate seed metadata mismatch")
        if manifest.get("ground_truth_used_for_training") is not False:
            raise AssertionError(f"seed{seed} did not preserve no-GT training")
        if manifest.get("cannot_link_lambda") != calibration["shared_lambda"]:
            raise AssertionError(f"seed{seed} lambda differs from calibrated immutable value")
        history = validate_history(cand_dir / "history.json", manifest, seed)
        cand_metrics, cand_per_image = score_record(cand_dir / "evaluation.json")
        control_metrics, control_per_image = score_record(control_dir / "evaluation.json")
        if len(manifest.get("training_ids", [])) != 4096:
            raise AssertionError(f"seed{seed} candidate manifest does not contain 4096 IDs")
        control_manifest = read(control_dir / "manifest.json")
        if manifest["training_ids"] != control_manifest["training_ids"]:
            raise AssertionError(f"seed{seed} candidate training IDs/order differ from SW0097")
        if manifest["source_sha256"] != control_manifest["source_sha256"]:
            raise AssertionError(f"seed{seed} candidate source SHA differs from matched control")
        deltas = {metric: cand_metrics[metric] - control_metrics[metric]
                  for metric in METRICS}
        result["per_seed"][str(seed)] = {
            "source_sha256": manifest["source_sha256"],
            "matched_control_source_sha256": control_manifest["source_sha256"],
            "training_ids_and_order_match": True,
            "updates": len(history), "lambda": manifest["cannot_link_lambda"],
            "candidate_metrics": cand_metrics, "matched_control_metrics": control_metrics,
            "paired_metric_deltas": deltas,
            "mean_cut_loss": statistics.fmean(row["cannot_link_cut_loss"] for row in history),
            "cut_loss_first4_mean": statistics.fmean(row["cannot_link_cut_loss"] for row in history[:4]),
            "cut_loss_last4_mean": statistics.fmean(row["cannot_link_cut_loss"] for row in history[-4:]),
            "cut_loss_min": min(row["cannot_link_cut_loss"] for row in history),
            "cut_loss_max": max(row["cannot_link_cut_loss"] for row in history),
            "mean_primary_loss": statistics.fmean(row["primary"] for row in history),
            "mean_unweighted_spike_loss": statistics.fmean(row["spike"] for row in history),
            "mean_total_loss": statistics.fmean(row["loss"] for row in history),
            "evaluation_images": 320}
        count_rows = [row["cannot_link_counts"] for row in history]
        selected = sum(int(row["selected_directed_negatives"]) for row in count_rows)
        connected = sum(int(row["connected_directed_negatives"]) for row in count_rows)
        result["per_seed"][str(seed)]["training_conflicts"] = {
            "selected_directed_negative_decisions_total": selected,
            "connected_at_forest_margin_directed_negative_total": connected,
            "connected_fraction": connected / selected if selected else 0.0,
            "mean_selected_per_update": selected / len(count_rows),
            "mean_connected_per_update": connected / len(count_rows),
            "definition": "Forest connectivity uses detached symmetrized classifier q=max(qij,qji) with strict edge margin q>.40; this is not the classifier .50 conflict count."}
        for metric in METRICS:
            candidate_rows[metric].append(cand_metrics[metric])
            control_rows[metric].append(control_metrics[metric])
            candidate_images[metric].append(cand_per_image[metric])
            control_images[metric].append(control_per_image[metric])

    means = {metric: statistics.fmean(candidate_rows[metric]) for metric in METRICS}
    control_means = {metric: statistics.fmean(control_rows[metric]) for metric in METRICS}
    deltas = {metric: means[metric] - control_means[metric] for metric in METRICS}
    seed_gains = {metric: sum(c > b for c, b in zip(candidate_rows[metric], control_rows[metric]))
                  for metric in METRICS}
    promotion = {"mean_fg_ari_gain_minimum": .01,
                 "seed_fg_ari_gains_minimum": 2,
                 "iou_margin_over_slot_minimum": .05,
                 "mean_fg_ari_delta": deltas["fg_ari"],
                 "seed_fg_ari_gains": seed_gains["fg_ari"],
                 "mean_foreground_iou_delta": deltas["foreground_iou"],
                 "mean_object_iou_delta": deltas["matched_object_iou"],
                 "foreground_iou_over_slot_margin": means["foreground_iou"] - SLOT_MEAN["foreground_iou"],
                 "object_iou_over_slot_margin": means["matched_object_iou"] - SLOT_MEAN["matched_object_iou"],
                 "fg_ari_gate_pass": deltas["fg_ari"] >= .01 and seed_gains["fg_ari"] >= 2,
                 "iou_gate_pass": (means["foreground_iou"] - SLOT_MEAN["foreground_iou"] >= .05
                                   and means["matched_object_iou"] - SLOT_MEAN["matched_object_iou"] >= .05)}
    promotion["candidate_for_70k"] = promotion["fg_ari_gate_pass"] and promotion["iou_gate_pass"]
    result["means"] = means
    result["matched_control_means"] = control_means
    result["paired_mean_deltas"] = deltas
    result["seed_gains_by_metric"] = seed_gains
    result["promotion_gate"] = promotion
    if promotion["candidate_for_70k"]:
        # Resample the same held-out image indices jointly across seeds and
        # metrics only when the preregistered aggregate/seed gates make promotion possible.
        import numpy as np
        rng = np.random.default_rng(102)
        boot = {}
        reps = 10000
        for metric in METRICS:
            per_seed_delta = np.asarray(candidate_images[metric]) - np.asarray(control_images[metric])
            sampled = rng.integers(0, 320, size=(reps, 320))
            per_rep = np.stack([per_seed_delta[seed, sampled].mean(axis=1)
                                for seed in range(3)]).mean(axis=0)
            boot[metric] = {"repetitions": reps,
                            "lower_95": float(np.quantile(per_rep, .025)),
                            "upper_95": float(np.quantile(per_rep, .975)),
                            "probability_positive": float((per_rep > 0).mean())}
        result["paired_image_bootstrap"] = boot
        promotion["bootstrap_fg_ari_positive"] = boot["fg_ari"]["lower_95"] > 0
        promotion["candidate_for_70k"] = (promotion["candidate_for_70k"]
                                           and promotion["bootstrap_fg_ari_positive"])
    else:
        result["paired_image_bootstrap"] = None
        promotion["bootstrap_fg_ari_positive"] = None
        result["notes"].append("No paired bootstrap was run because the required aggregate FG-ARI improvement and at least two per-seed FG-ARI gains did not pass.")
    result["notes"].append("The SW0097 matched-control history does not contain cannot-link mask/conflict counts; candidate connected-negative totals use the strict q>.40 forest margin and must not be interpreted as q>=.50 control counts.")
    result["decision"] = "candidate_for_70k_review" if promotion["candidate_for_70k"] else "stop_recipe_no_70k"
    path = CANDIDATE / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"means": means, "matched_control_means": control_means,
                      "paired_mean_deltas": deltas, "seed_gains": seed_gains,
                      "promotion": promotion, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
