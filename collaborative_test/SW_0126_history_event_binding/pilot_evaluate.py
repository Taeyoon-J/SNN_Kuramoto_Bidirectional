"""Conditional seed-1 SW0126 paired endpoint evaluation.

Both complete prediction arrays are written and SHA-verified before this
program opens the validation mask dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0125_late_rollout_credit import summarize as sw125_summary
from collaborative_test.SW_0118_allowed_patch_labels import run as sw118
from collaborative_test.SW_0126_history_event_binding import pilot_run
from collaborative_test.SW_0126_history_event_binding.history_event import attach_history_event_membrane
from collaborative_test.SW_0126_history_event_binding.pilot_rollout import rollout
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch

IDS = tuple(range(1320, 1640))
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SLOT_IOU_REFERENCE = {"foreground_iou": 0.20358913115224261,
                      "matched_object_iou": 0.20693697915214362}
DRAWS, BOOTSTRAP_SEED = 10_000, 126
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite_metric_rows(score):
    result = {}
    if "per_image" in score:
        per_image = score["per_image"]
        means = score.get("mean", score.get("metrics", {}))
        counts = score.get("valid_count", {})
    else:
        per_image = {metric: score[metric]["per_image"] for metric in METRICS}
        means = {metric: score[metric]["mean"] for metric in METRICS}
        counts = {metric: score[metric].get("valid_count", 320) for metric in METRICS}
    for metric in METRICS:
        values = np.asarray(per_image[metric], dtype=np.float64)
        mean = float(means[metric])
        if values.shape != (320,) or not np.isfinite(values).all() or not math.isfinite(mean):
            raise ValueError(f"invalid 320-image {metric} values")
        if int(counts.get(metric, 320)) != 320 or abs(float(values.mean()) - mean) > 1e-10:
            raise ValueError(f"{metric} mean/count do not match the per-image scores")
        result[metric] = values
    return result


def slot_labels_from_assignments(assignment):
    """Apply registered hard slot readout: largest group BG, groups <2 omitted."""
    values = torch.as_tensor(assignment).detach().cpu()
    if values.ndim != 3 or tuple(values.shape[1:]) != (11, 256):
        raise ValueError("assignment must be [B,11,256]")
    if not torch.isfinite(values).all():
        raise ValueError("assignment contains nonfinite values")
    slots = values.argmax(dim=1).reshape(values.shape[0], 16, 16)
    labels = torch.zeros_like(slots, dtype=torch.int64)
    for i in range(slots.shape[0]):
        counts = torch.bincount(slots[i].reshape(-1), minlength=11)
        # argmax deterministically selects the lowest slot ID on a tie.
        background = int(counts.argmax())
        for slot in range(11):
            if slot != background and int(counts[slot]) >= 2:
                labels[i][slots[i] == slot] = slot + 1
    return labels


def paired_bootstrap(candidate, reference, draws=DRAWS, seed=BOOTSTRAP_SEED):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (candidate.shape != (320,) or reference.shape != (320,)
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()
            or draws != DRAWS or seed != BOOTSTRAP_SEED):
        raise ValueError("bootstrap requires the preregistered paired 320-image seed-126 procedure")
    delta = candidate - reference
    samples = np.random.default_rng(seed).integers(0, 320, size=(draws, 320))
    boot = delta[samples].mean(axis=1)
    return {"mean_delta": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "draws": draws, "seed": seed,
            "unit": "paired validation image"}


def _source_per_image():
    score, provenance = sw125_summary._source_score(1)
    normalized = {"mean": score["metrics"], "per_image": score["per_image"],
                  "valid_count": {metric: 320 for metric in METRICS}}
    arrays = _finite_metric_rows(normalized)
    return arrays, provenance


def _slot_anchor():
    _, metadata, summary = sw118._load_slot(1)
    means = summary["mean"]
    for metric in METRICS:
        if int(summary["valid_count"][metric]) != 320 or not math.isfinite(float(means[metric])):
            raise ValueError(f"matched Slot seed1 {metric} anchor is invalid")
    metadata["matched_seed1_means"] = {metric: float(means[metric]) for metric in METRICS}
    metadata["registered_three_seed_iou_threshold_means"] = SLOT_IOU_REFERENCE
    # The registered pilot gate uses the previously fixed three-seed Slot
    # means; the seed-1 report is provenance only, not a changed threshold.
    return dict(SLOT_IOU_REFERENCE), metadata


def _prediction_model(arm, device):
    source, source_manifest, _source_record, _ids, _rows = sw122.source_contract(1)
    core = base.make_core(device, steps=64)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    attach_history_event_membrane(core, capture_gate_trace=True)
    folder = pilot_run.training_dir(arm)
    if not pilot_run.valid_training(folder, arm):
        raise AssertionError(f"seed1 {arm} training artifacts are not valid and complete")
    core.load_state_dict(torch.load(folder / "core.pt", map_location=device, weights_only=True), strict=True)
    binder = pilot_run.TemporalSlotRGBBinder(init_seed=1261).to(device)
    binder.load_state_dict(torch.load(folder / "binder.pt", map_location=device, weights_only=True), strict=True)
    core.eval(); binder.eval()
    return core, binder, folder, source, source_manifest


def _predict_arm(arm, device):
    core, binder, folder, source, source_manifest = _prediction_model(arm, device)
    rgb_cache, rgb_meta, rgb_cache_sha = sw117.validate_rgb_validation_cache()
    cached_gamma, gamma_meta = base.validate_gamma_cache(base.GAMMA_VAL,
                                                          base.GAMMA_VAL_MANIFEST,
                                                          validation=True)
    encoder = sw117.load_input_encoder(str(sw117.ENCODER_PATH), num_kernels=8,
        kernel_size=3, channels=3, device=device)
    encoder.load_state_dict(torch.load(sw117.ENCODER_PATH, map_location=device,
                                      weights_only=True), strict=True)
    encoder.eval().requires_grad_(False)
    patcher = sw117.FeaturePatchGammaInitializer(grid_size=16).to(device).eval()
    stats = torch.load(sw117.STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = sw117.preprocessing_tensors(stats, device)
    labels = []
    max_gamma_delta = 0.0
    rows = np.arange(320, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, 320, 8):
            image = sw117.read_rgb(rgb_cache, rows[start:start+8], device)
            gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, image)
            cached = cached_gamma[torch.as_tensor(rows[start:start+8], dtype=torch.long)].to(device)
            delta = float((gamma - cached).abs().max().cpu())
            if not math.isfinite(delta) or delta > 2e-5:
                raise AssertionError(f"validation gamma/cache mismatch: {delta}")
            max_gamma_delta = max(max_gamma_delta, delta)
            trace = rollout(core, gamma, total_steps=1024, settle=512, live_tail_steps=0)
            event = core.membrane_layer.component_event_trace(8)[..., -512:]
            gate = core.membrane_layer.component_gate_trace(8)[..., -512:]
            spikes = trace["component_spikes"][..., -512:]
            if not torch.equal(spikes, gate * event):
                raise AssertionError("prediction did not preserve actual gate-times-event spike traces")
            signal = pilot_run.binder_trace_for_arm(spikes, gate, arm)
            output = binder(signal)
            labels.append(slot_labels_from_assignments(output["assignment"]))
    artifact = torch.cat(labels, dim=0)
    if tuple(artifact.shape) != (320, 16, 16):
        raise AssertionError("prediction must contain all fixed 320 16x16 label grids")
    train_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return artifact, {"arm": arm, "training_dir": str(folder),
        "training_manifest_sha256": sha(folder / "manifest.json"),
        "core_sha256": sha(folder / "core.pt"), "binder_sha256": sha(folder / "binder.pt"),
        "history_sha256": sha(folder / "history.json"),
        "implementation_fingerprint": train_manifest["implementation_fingerprint"],
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "rgb_validation_cache_sha256": rgb_cache_sha,
        "rgb_validation_manifest_sha256": sha(sw117.VAL_RGB_MANIFEST),
        "gamma_validation_cache_sha256": sha(base.GAMMA_VAL),
        "gamma_validation_manifest_sha256": sha(base.GAMMA_VAL_MANIFEST),
        "max_live_gamma_cache_abs_diff": max_gamma_delta,
        "ids": [IDS[0], IDS[-1]], "count": len(IDS),
        "ground_truth_used_for_prediction": False,
        "prediction_rule": "argmax binder assignment; largest slot background; nonbackground groups <2 patches omitted"}


def _standard_score(labels, target):
    return sw118._score_standard(labels, target)


def _promotion(candidate, gate, source, slot):
    candidate_arrays = _finite_metric_rows(candidate)
    gate_arrays = _finite_metric_rows(gate)
    source_arrays = source
    means = {arm: {m: float(arrays[m].mean()) for m in METRICS}
             for arm, arrays in (("history_event", candidate_arrays),
                                 ("gate_only", gate_arrays), ("source97", source_arrays))}
    deltas = {anchor: paired_bootstrap(candidate_arrays["fg_ari"], arrays["fg_ari"])
              for anchor, arrays in (("gate_only", gate_arrays), ("source97", source_arrays))}
    passed = (means["history_event"]["fg_ari"] > means["gate_only"]["fg_ari"]
        and means["history_event"]["fg_ari"] > means["source97"]["fg_ari"]
        and all(delta["ci95"][0] > 0 for delta in deltas.values())
        and means["history_event"]["foreground_iou"] > SLOT_IOU_REFERENCE["foreground_iou"] + 0.05
        and means["history_event"]["matched_object_iou"] > SLOT_IOU_REFERENCE["matched_object_iou"] + 0.05)
    return {"means": means, "paired_fg_ari_differences": deltas,
        "registered_slot_three_seed_iou_reference": SLOT_IOU_REFERENCE,
            "gates": {"candidate_fg_exceeds_gate": means["history_event"]["fg_ari"] > means["gate_only"]["fg_ari"],
                      "candidate_fg_exceeds_source97": means["history_event"]["fg_ari"] > means["source97"]["fg_ari"],
                      "paired_ci_lower_gt_zero_vs_gate": deltas["gate_only"]["ci95"][0] > 0,
                      "paired_ci_lower_gt_zero_vs_source97": deltas["source97"]["ci95"][0] > 0,
                      "foreground_iou_gt_slot_plus_005": means["history_event"]["foreground_iou"] > SLOT_IOU_REFERENCE["foreground_iou"] + 0.05,
                      "matched_object_iou_gt_slot_plus_005": means["history_event"]["matched_object_iou"] > SLOT_IOU_REFERENCE["matched_object_iou"] + 0.05,
                      "pilot_pass": bool(passed)},
            "interpretation": "conditional seed-1 pilot only; no multi-seed, gate-causality, or data-scaling claim"}


def evaluate(output, dataset=DATASET, device="cuda:0"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing evaluation output {output}")
    output.mkdir(parents=True, exist_ok=False)
    # Produce and persist both complete, provenance-bound prediction grids first.
    predictions, provenance = {}, {}
    for arm in pilot_run.ARM_NAMES:
        predictions[arm], provenance[arm] = _predict_arm(arm, device)
    pred_path = output / "paired_predictions.pt"
    if pred_path.exists():
        raise FileExistsError(pred_path)
    tmp = pred_path.with_suffix(".pt.tmp")
    torch.save({"ids": list(IDS), "predictions": predictions, "provenance": provenance}, tmp)
    os.replace(tmp, pred_path)
    pred_sha = sha(pred_path)
    pred_manifest = {"status": "predictions_complete", "experiment": "SW0126",
        "seed": 1, "ids": [IDS[0], IDS[-1]], "count": len(IDS),
        "prediction_sha256": pred_sha, "arm_provenance": provenance,
        "ground_truth_used_for_prediction": False,
        "evaluation_code_sha256": sha(Path(__file__))}
    manifest_path = output / "prediction_manifest.json"
    manifest_path.write_text(json.dumps(pred_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Re-read and verify before opening any mask dataset.
    if sha(pred_path) != pred_sha or json.loads(manifest_path.read_text())["prediction_sha256"] != pred_sha:
        raise AssertionError("saved paired predictions failed SHA verification")
    frozen = torch.load(pred_path, map_location="cpu", weights_only=False)
    if frozen.get("ids") != list(IDS) or set(frozen.get("predictions", {})) != set(pilot_run.ARM_NAMES):
        raise AssertionError("saved prediction IDs or arm set are invalid")
    for arm in pilot_run.ARM_NAMES:
        value = frozen["predictions"][arm]
        if tuple(value.shape) != (320, 16, 16) or value.dtype != torch.int64:
            raise AssertionError(f"saved {arm} prediction has invalid shape/dtype")

    # Resolve independent anchors before reading GT, so missing provenance fails closed.
    source_score, source_prov = _source_per_image()
    slot_mean, slot_prov = _slot_anchor()
    import h5py
    with h5py.File(dataset, "r") as h5:
        masks = np.asarray(h5["mask"][1320:1640])
    if masks.shape not in ((320, 128, 128), (320, 128, 128, 1)):
        raise ValueError(f"unexpected validation mask geometry {masks.shape}")
    target = clevr_mask_patch(torch.as_tensor(masks, dtype=torch.int64), 8)["patch_labels"]
    arm_scores = {}
    for arm in pilot_run.ARM_NAMES:
        arm_scores[arm] = _standard_score(frozen["predictions"][arm], target)
        _finite_metric_rows(arm_scores[arm])
    promotion = _promotion(arm_scores["history_event"], arm_scores["gate_only"],
                           source_score, slot_mean)
    result = {"status": "complete", "experiment": "SW0126", "seed": 1,
        "ids": [IDS[0], IDS[-1]], "count": 320,
        "prediction_manifest_sha256": sha(manifest_path), "prediction_blob_sha256": pred_sha,
        "prediction_provenance": provenance,
        "scores": arm_scores,
        "anchors": {"source97": source_prov, "matched_slot70k_seed1_provenance": slot_prov,
                    "registered_slot_three_seed_iou_threshold_means": slot_mean},
        "promotion": promotion,
        "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True,
        "model_training_or_weights_changed_during_evaluation": False}
    final_path = output / "evaluation.json"
    final_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = evaluate(args.output, args.dataset, args.device)
    print(json.dumps({"status": result["status"], "experiment": result["experiment"],
                      "seed": 1, "pilot_pass": result["promotion"]["gates"]["pilot_pass"],
                      "output": str(args.output / "evaluation.json")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
