"""Read-only first-four-TRAIN-batch diagnostics for completed SW0123 arms.

No masks are loaded, no gradients or optimizer updates are performed, and no
checkpoint is modified. Each invocation creates one SHA-bound JSON receipt.
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

from collaborative_test.SW_0123_adaptive_temporal_assignment import coordinator, run
from collaborative_test.SW_0123_adaptive_temporal_assignment.model import (
    AdaptiveTemporalRGBModel, normalized_patch_centers,
)
from collaborative_test.SW_0123_adaptive_temporal_assignment.readout import assignment_to_labels

SEEDS = (0, 1, 2)
ARMS = ("legacy_full", "adaptive_full", "gate_only_control")
DIAGNOSTIC_BATCHES = 4
DIAGNOSTIC_IMAGES = DIAGNOSTIC_BATCHES * run.BATCH
PERMUTATION_SEED = 12301


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic quantiles require nonempty finite values")
    q = np.quantile(array, [0.0, .1, .5, .9, 1.0])
    return {"min": float(q[0]), "p10": float(q[1]), "median": float(q[2]),
            "p90": float(q[3]), "max": float(q[4])}


@torch.inference_mode()
def batch_diagnostics(out, rgb, permutation):
    """Summarize assignment, slot, decoder, and RGB reconstruction behavior."""
    probability = out["assignment"]
    slot_embedding = out["slot_embedding"]
    decoded = out["decoded_slot_patches"]
    reconstructed = out["reconstructed_image"]
    if (probability.ndim != 3 or tuple(probability.shape[1:]) != (256, 11)
            or slot_embedding.shape != (probability.shape[0], 11, 64)
            or decoded.shape != (probability.shape[0], 11, 256, 8, 8, 3)
            or reconstructed.shape != rgb.shape or rgb.shape[1:] != (3, 128, 128)
            or permutation.shape != (256,)):
        raise ValueError("SW0123 diagnostic tensors have an unexpected shape")
    if not all(torch.isfinite(value).all() for value in
               (probability, slot_embedding, decoded, reconstructed, rgb)):
        raise FloatingPointError("SW0123 diagnostic tensors must be finite")

    tiny = torch.finfo(probability.dtype).tiny
    entropy = -(probability * probability.clamp_min(tiny).log()).sum(dim=-1)
    slot_mean = probability.mean(dim=(0, 1))
    slot_spatial_variance = probability.var(dim=1, unbiased=False).mean(dim=0)
    hard = probability.argmax(dim=-1)
    counts = torch.stack([(hard == slot).sum(dim=1) for slot in range(11)], dim=1)
    empty_fraction = (counts == 0).float().mean(dim=1)
    largest_fraction = counts.max(dim=1).values.float() / 256.0
    readout = assignment_to_labels(probability)
    readout_fg_fraction = (readout > 0).float().mean(dim=(1, 2))
    readout_counts = torch.bincount(readout.reshape(-1).to(torch.int64), minlength=12)

    slot_pairs = torch.triu_indices(11, 11, offset=1, device=slot_embedding.device)
    pairwise_embedding_distance = torch.linalg.vector_norm(
        slot_embedding[:, slot_pairs[0]] - slot_embedding[:, slot_pairs[1]], dim=-1)

    decoded_pair_rms = []
    for left in range(11):
        for right in range(left + 1, 11):
            decoded_pair_rms.append((decoded[:, left] - decoded[:, right]).square()
                                    .mean(dim=(1, 2, 3, 4)).sqrt())
    decoded_pair_rms = torch.stack(decoded_pair_rms, dim=1)
    rgb_scale = rgb.float().std(dim=(1, 2, 3), unbiased=False).clamp_min(1e-6)
    decoded_pair_normalized = decoded_pair_rms / rgb_scale[:, None]

    shuffled_probability = probability.index_select(1, permutation.to(probability.device))
    shuffled_patches = torch.einsum("bnk,bknhwc->bnhwc", shuffled_probability, decoded)
    mean_probability = probability.mean(dim=1, keepdim=True).expand_as(probability)
    mean_patches = torch.einsum("bnk,bknhwc->bnhwc", mean_probability, decoded)
    real_loss, _ = run.normalized_rgb_loss(reconstructed, rgb)
    shuffled_loss, _ = run.normalized_rgb_loss(
        run.image_from_patches(shuffled_patches), rgb)
    mean_loss, _ = run.normalized_rgb_loss(run.image_from_patches(mean_patches), rgb)

    return {
        "assignment_entropy_mean": float(entropy.mean()),
        "slot_mean_probability": [float(x) for x in slot_mean],
        "slot_spatial_probability_variance": [float(x) for x in slot_spatial_variance],
        "empty_slot_fraction_per_image": [float(x) for x in empty_fraction],
        "largest_slot_background_fraction_per_image": [float(x) for x in largest_fraction],
        "readout_foreground_patch_fraction_per_image": [float(x) for x in readout_fg_fraction],
        "hard_argmax_slot_patch_counts_per_image": counts.cpu().tolist(),
        "readout_label_counts_batch": {str(i): int(value) for i, value in enumerate(readout_counts)},
        "pairwise_slot_latent_l2": _quantiles(pairwise_embedding_distance.cpu().numpy()),
        "pairwise_decoded_slot_patch_rms": _quantiles(decoded_pair_rms.cpu().numpy()),
        "pairwise_decoded_slot_patch_rms_over_rgb_std": _quantiles(decoded_pair_normalized.cpu().numpy()),
        "normalized_rgb_reconstruction_loss": float(real_loss),
        "fixed_row_shuffle_reconstruction_loss_seed12301": float(shuffled_loss),
        "image_mean_assignment_reconstruction_loss": float(mean_loss),
        "shuffle_minus_real_loss": float(shuffled_loss - real_loss),
        "mean_assignment_minus_real_loss": float(mean_loss - real_loss),
    }


def _load_completed(seed, arm, device):
    train_task = next(task for task in coordinator.task_plan()
                      if task["stage"] == "train" and task["seed"] == seed and task["arm"] == arm)
    if not coordinator.valid_result(train_task):
        raise RuntimeError(f"seed{seed}/{arm} training artifact is not valid and complete")
    folder = run.OUT / f"seed{seed}_{arm}"
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source, source_manifest, _record, ids, rows = run.source_contract(seed)
    if (manifest.get("source_core_sha256") != _sha(source)
            or manifest.get("source_manifest_sha256") != _sha(source_manifest)
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("gamma_rows") != rows.tolist()
            or manifest.get("matched_shuffle_seed") != 117 + seed):
        raise AssertionError("diagnostic training artifact has wrong source or ordered IDs")

    calibration_path, calibration, kappa, rms = run.load_calibration(seed)
    if manifest.get("calibration_sha256") != _sha(calibration_path):
        raise AssertionError("diagnostic checkpoint calibration provenance mismatch")
    core, encoder, patcher, mean, std, clip, *_ = run.load_training_assets(seed, device)
    run.configure_arm(core, arm, kappa)
    core.load_state_dict(torch.load(folder / "core.pt", map_location=device, weights_only=True), strict=True)
    encoder.load_state_dict(torch.load(folder / "encoder.pt", map_location=device, weights_only=True), strict=True)
    model = AdaptiveTemporalRGBModel().to(device)
    model.assignment_head.load_state_dict(
        torch.load(folder / "assignment_head.pt", map_location=device, weights_only=True), strict=True)
    model.rgb_decoder.load_state_dict(
        torch.load(folder / "rgb_decoder.pt", map_location=device, weights_only=True), strict=True)
    if not torch.equal(model.assignment_head.spike_rms.cpu(), rms.cpu()):
        raise AssertionError("diagnostic assignment head differs from calibrated shared RMS")
    core.eval(); core.graph_generator.eval(); encoder.eval(); patcher.eval(); model.eval()
    return {"folder": folder, "manifest_path": manifest_path, "manifest": manifest,
            "source": source, "source_manifest": source_manifest, "ids": ids, "rows": rows,
            "calibration_path": calibration_path, "calibration": calibration,
            "core": core, "encoder": encoder, "patcher": patcher,
            "mean": mean, "std": std, "clip": clip, "model": model}


def _diagnostic_training_batch(seed, rows, start, gamma_cache, rgb_cache,
                               encoder, patcher, mean, std, clip, device):
    """Use live trained-encoder gamma without enforcing source-cache equality."""
    return run._batch_data(seed, rows, start, gamma_cache, rgb_cache, encoder,
                           patcher, mean, std, clip, device, check_source_cache=False)


def _endpoint_counts(seed, arm, train_manifest, hashes):
    task = next(task for task in coordinator.task_plan()
                if task["stage"] == "evaluate" and task["seed"] == seed and task["arm"] == arm)
    if not coordinator.valid_result(task):
        return {"status": "pending", "reason": "endpoint artifact is not yet valid"}
    report_path = coordinator.artifact_path(task)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pred_path = report_path.parent / report["prediction_file"]
    if (report.get("training_manifest_sha256") != _sha(train_manifest)
            or report.get("training_artifact_sha256") != hashes
            or report.get("ground_truth_used_for_prediction") is not False
            or not pred_path.is_file() or _sha(pred_path) != report.get("prediction_sha256")):
        raise AssertionError("endpoint hard labels do not bind the completed training artifact")
    frozen = torch.load(pred_path, map_location="cpu", weights_only=False)
    labels = frozen.get("primary_labels")
    if (frozen.get("experiment") != "SW0123" or frozen.get("seed") != seed
            or frozen.get("arm") != arm or frozen.get("image_ids") != [1320, 1639]
            or frozen.get("ground_truth_used_for_prediction") is not False
            or tuple(labels.shape) != (320, 16, 16)):
        raise AssertionError("frozen endpoint hard-label artifact has wrong identity/shape")
    histogram = torch.bincount(labels.to(torch.int64).reshape(-1), minlength=12)
    per_image_fg = (labels > 0).float().mean(dim=(1, 2))
    group_counts = torch.tensor([
        torch.unique(image[image > 0]).numel() for image in labels], dtype=torch.float32)
    background_fraction = (labels == 0).float().mean(dim=(1, 2))
    return {"status": "available", "evaluation_sha256": _sha(report_path),
            "prediction_sha256": _sha(pred_path),
            "label_counts_across_320_images": {str(i): int(count) for i, count in enumerate(histogram)},
            "foreground_patch_fraction_per_image": _quantiles(per_image_fg.numpy()),
            "foreground_group_count_per_image": _quantiles(group_counts.numpy()),
            "background_patch_fraction_per_image": _quantiles(background_fraction.numpy()),
            "all_background_image_count": int((per_image_fg == 0).sum()),
            "note": "fixed endpoint IDs differ from the first-four TRAIN diagnostic IDs"}


def diagnose(seed: int, arm: str, device="cuda"):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("unregistered SW0123 seed/arm")
    loaded = _load_completed(seed, arm, device)
    gamma_cache, gamma_manifest, rgb_cache, rgb_meta, rgb_sha = run.load_registered_data()
    permutation = torch.as_tensor(np.random.default_rng(PERMUTATION_SEED).permutation(256),
                                  dtype=torch.long, device=device)
    batch_reports = []
    actual_ids = []
    actual_rows = []
    gamma_cache_drift = []
    with torch.inference_mode():
        for batch in range(DIAGNOSTIC_BATCHES):
            start = batch * run.BATCH
            batch_rows, images, rgb, gamma = _diagnostic_training_batch(
                seed, loaded["rows"], start, gamma_cache, rgb_cache,
                loaded["encoder"], loaded["patcher"], loaded["mean"],
                loaded["std"], loaded["clip"], device)
            cached_gamma = gamma_cache[torch.as_tensor(batch_rows, dtype=torch.long)].to(device)
            drift = float((gamma.detach() - cached_gamma).abs().max())
            if not math.isfinite(drift):
                raise FloatingPointError("trained-encoder gamma/cache descriptive difference is nonfinite")
            gamma_cache_drift.append(drift)
            out, components, _theta, _events = run.rollout(
                loaded["core"], gamma, loaded["model"], arm,
                normalized_patch_centers(device=device))
            if tuple(components.shape) != (run.BATCH, 4, 256, run.TIME_STEPS):
                raise AssertionError("diagnostic rollout did not preserve actual four-component traces")
            batch_reports.append(batch_diagnostics(out, rgb, permutation))
            actual_ids.extend(loaded["ids"][start:start + run.BATCH].tolist())
            actual_rows.extend(batch_rows.tolist())

    manifest_path = loaded["manifest_path"]
    folder = loaded["folder"]
    train_files = {name: folder / filename for name, filename in {
        "core": "core.pt", "encoder": "encoder.pt", "assignment_head": "assignment_head.pt",
        "rgb_decoder": "rgb_decoder.pt", "history": "history.json"}.items()}
    checkpoint_hashes = {name: _sha(path) for name, path in train_files.items()}
    if any(loaded["manifest"].get(name + "_sha256") != digest
           for name, digest in checkpoint_hashes.items()):
        raise AssertionError("training checkpoint changed during the read-only diagnostic")
    endpoint = _endpoint_counts(seed, arm, manifest_path, {
        **checkpoint_hashes,
        "core_optimizer": _sha(folder / "core_optimizer.pt"),
        "head_optimizer": _sha(folder / "head_optimizer.pt"),
    })
    train_readout_counts = {str(label): sum(row["readout_label_counts_batch"][str(label)]
                                            for row in batch_reports) for label in range(12)}
    endpoint_train_l1 = None
    if endpoint["status"] == "available":
        train_total = sum(train_readout_counts.values())
        endpoint_counts = endpoint["label_counts_across_320_images"]
        endpoint_total = sum(endpoint_counts.values())
        endpoint_train_l1 = sum(abs(train_readout_counts[str(label)] / train_total
                                    - endpoint_counts[str(label)] / endpoint_total)
                                for label in range(12))
    calibration_path = loaded["calibration_path"]
    return {
        "experiment": "SW0123", "stage": "read_only_train_assignment_diagnostic",
        "status": "complete", "seed": seed, "arm": arm,
        "ground_truth_used": False, "optimizer_updates": 0,
        "diagnostic_contract": {"batches": DIAGNOSTIC_BATCHES, "batch_size": run.BATCH,
                                "time_steps": run.TIME_STEPS, "settle": run.SETTLE,
                                "training_ids": "first64 in the registered seed-specific ordered TRAIN vector",
                                "permutation_seed": PERMUTATION_SEED},
        "training_image_ids": actual_ids, "gamma_cache_rows": actual_rows,
        "gamma_cache_manifest_status": gamma_manifest.get("status"),
        "trained_encoder_vs_frozen_gamma_cache_max_abs_per_batch": gamma_cache_drift,
        "rgb_cache_sha256": rgb_sha, "rgb_cache_manifest_sha256": _sha(run.RGB_CACHE_MANIFEST),
        "source_core_sha256": _sha(loaded["source"]),
        "source_manifest_sha256": _sha(loaded["source_manifest"]),
        "training_manifest_sha256": _sha(manifest_path),
        "calibration_sha256": _sha(calibration_path),
        "training_checkpoint_sha256": checkpoint_hashes,
        "implementation_sha256": {
            "diagnostic": _sha(Path(__file__)), "runner": _sha(run.HERE / "run.py"),
            "model": _sha(HERE / "model.py"), "adaptive_dynamics": _sha(HERE / "adaptive_dynamics.py"),
            "readout": _sha(HERE / "readout.py"),
        },
        "first_four_batch_diagnostics": batch_reports,
        "training_readout_label_counts_first64": train_readout_counts,
        "means_over_first64_training_images": {
            key: float(np.mean([row[key] for row in batch_reports]))
            for key in ("assignment_entropy_mean", "mean_assignment_minus_real_loss",
                        "shuffle_minus_real_loss", "normalized_rgb_reconstruction_loss",
                        "fixed_row_shuffle_reconstruction_loss_seed12301",
                        "image_mean_assignment_reconstruction_loss")},
        "fixed_endpoint_hard_label_counts": endpoint,
        "endpoint_vs_training_label_distribution_l1": endpoint_train_l1,
        "note": "Diagnostic uses trained checkpoints and native RGB only; it neither reads masks nor changes any model state on disk.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or HERE / "results_archive" / f"assignment_diagnostic_seed{args.seed}_{args.arm}_20261009.json"
    if output.exists():
        raise FileExistsError(f"preserving existing diagnostic report: {output}")
    result = diagnose(args.seed, args.arm, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "seed": args.seed, "arm": args.arm,
                      "means_over_first64_training_images": result["means_over_first64_training_images"],
                      "endpoint_status": result["fixed_endpoint_hard_label_counts"]["status"],
                      "output": str(output)}, allow_nan=False))


if __name__ == "__main__":
    main()
