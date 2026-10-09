"""Frozen SW0097 source/QCC pair-relation QA; no training or threshold tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np  # Must precede torch in the shared Windows OpenMP runtime.
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0118_allowed_patch_labels import run as sw118
from collaborative_test.SW_0126_history_event_binding import screen as source_rollout
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels, clevr_mask_patch
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity, spike_synchrony_components,
)

SEEDS = (0, 1, 2)
IDS = tuple(range(1320, 1336))
BATCH = 8
STEPS = 1024
SETTLE = 512
GRID = 16
BINS = (("distance_0_2", 0.0, 2.0), ("distance_2_5", 2.0, 5.0))
POS_Q = 0.9
NEG_Q = 0.1
MIN_PRECISION = 0.95
MIN_COVERAGE_IMAGES = 12
MIN_PAIRS_PER_CLASS = 50
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _source_contract(seed: int):
    core_path, manifest_path, manifest = base.source_paths(seed)
    source_ids = manifest.get("training_ids")
    if (sha256_file(core_path) != base.EXPECTED_SOURCE_SHAS[seed]
            or manifest.get("source_model_seed") != seed
            or manifest.get("status") != "complete"
            or manifest.get("ground_truth_used_for_training") is not False
            or not isinstance(source_ids, list) or len(source_ids) != 4096):
        raise ValueError(f"seed{seed}: registered SW0097 source contract mismatch")
    return core_path, manifest_path, manifest


def _implementation_fingerprint() -> dict:
    paths = (HERE / "run.py", HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0126_history_event_binding/screen.py",
             ROOT / "collaborative_test/SW_0118_allowed_patch_labels/run.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
             ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py")
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in paths}


def _load_core(seed: int, device):
    source, manifest_path, manifest = _source_contract(seed)
    core = base.make_core(device, steps=STEPS)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    core.eval().requires_grad_(False)
    if (core.osc_dim != 4 or core.gamma_drive_mode != "static"
            or core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None):
        raise ValueError("source checkpoint is outside the registered static D4 QCC contract")
    return core, source, manifest_path, manifest


def _labels_from_actual_spikes(component_spikes: torch.Tensor) -> torch.Tensor:
    if (component_spikes.ndim != 4 or tuple(component_spikes.shape[1:3]) != (4, 256)
            or component_spikes.shape[-1] != STEPS
            or not torch.isfinite(component_spikes).all()):
        raise ValueError("QCC requires finite actual [B,4,256,1024] component spikes")
    groups = spike_synchrony_components(
        component_spikes.mean(dim=1).detach().cpu(),
        foreground_threshold=0.15, synchrony_threshold=0.50, min_group_size=2,
        settle=SETTLE, components=component_spikes.detach().cpu(),
        background="largest_component", synchrony_quantile=None,
        target_foreground=None, spatial_sigma=None, spatial_grid_size=GRID,
        affinity_mode="spike")
    return spatial_components_to_patch_labels(groups, GRID, device="cpu").reshape(-1, GRID, GRID).long()


def _q_matrices(component_spikes: torch.Tensor) -> torch.Tensor:
    q = spike_synchrony_affinity(
        component_spikes.mean(dim=1), components=component_spikes,
        settle=SETTLE, affinity_mode="spike")
    if tuple(q.shape) != (component_spikes.shape[0], 256, 256) or not torch.isfinite(q).all():
        raise ValueError("actual-spike Q matrix has invalid shape or values")
    return q.detach().cpu().to(torch.float32)


def _native_parity(core, gamma) -> dict:
    manual = source_rollout._full_rollout(core, gamma, live_tail=0)
    parity = source_rollout.assert_native_source_rollout_parity(core, gamma, manual)
    manual_labels = _labels_from_actual_spikes(manual["component_spikes"])
    native_labels = _labels_from_actual_spikes(core.last_component_spikes)
    if not torch.equal(manual_labels, native_labels):
        raise AssertionError("manual and native source QCC labels differ")
    return {"trace_exact": parity, "qcc_labels_exact": True}


def _predict_all_seeds(device) -> tuple[dict, dict, dict]:
    gamma, _meta = base.validate_gamma_cache(base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST,
                                              validation=True)
    if tuple(gamma.shape) != (320, 8, 256):
        raise ValueError("registered validation gamma cache geometry changed")
    row_ids = np.arange(IDS[0] - 1320, IDS[-1] - 1320 + 1, dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(16, dtype=np.int64)):
        raise AssertionError("fixed validation gamma rows are not 1320..1335")
    predictions, provenance, parity = {}, {}, {}
    for seed in SEEDS:
        core, source, manifest_path, manifest = _load_core(seed, device)
        pieces_labels, pieces_q = [], []
        parity[str(seed)] = _native_parity(
            core, gamma[:2].to(device))
        for start in range(0, len(IDS), BATCH):
            rows = row_ids[start:start + BATCH]
            batch_gamma = gamma[torch.as_tensor(rows, dtype=torch.long)].to(device)
            with torch.no_grad():
                _, _, _, _theta = core(batch_gamma, num_time_steps=STEPS,
                                       return_theta=True, return_core_out=True)
                components = core.last_component_spikes
            pieces_labels.append(_labels_from_actual_spikes(components))
            pieces_q.append(_q_matrices(components))
        labels = torch.cat(pieces_labels)
        q = torch.cat(pieces_q)
        if tuple(labels.shape) != (16, GRID, GRID) or tuple(q.shape) != (16, 256, 256):
            raise AssertionError(f"seed{seed}: incomplete frozen QA predictions")
        predictions[str(seed)] = {"labels": labels, "q": q}
        provenance[str(seed)] = {"source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_model_seed": seed, "shuffle_seed": manifest.get("seed"),
            "source_training_ids_sha256": hashlib.sha256(
                np.asarray(manifest["training_ids"], dtype="<i8").tobytes()).hexdigest(),
            "image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
            "ground_truth_used_for_prediction": False}
    return predictions, provenance, parity


def _relation_mask_hash(mask: torch.Tensor) -> str:
    value = mask.detach().cpu().contiguous().to(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _build_relation_masks(labels: torch.Tensor, q: torch.Tensor) -> dict:
    """Freeze directed foreground-anchor Q decisions and GT-free eligibility."""
    labels = labels.reshape(256).long().cpu()
    q = q.detach().cpu().float()
    if tuple(q.shape) != (256, 256) or not torch.isfinite(q).all():
        raise ValueError("relation mask requires finite 256x256 actual-spike Q")
    coords = torch.stack(torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij"), dim=-1).reshape(256, 2)
    dist = torch.cdist(coords.float(), coords.float())
    no_self = ~torch.eye(256, dtype=torch.bool)
    anchor_fg = labels[:, None] > 0
    same_pred_fg = anchor_fg & (labels[:, None] == labels[None, :])
    different_pred = labels[:, None] != labels[None, :]
    frozen = {}
    for name, low, high in BINS:
        in_bin = no_self & (dist > low) & (dist <= high)
        positive = same_pred_fg & (q >= POS_Q) & in_bin
        negative = anchor_fg & different_pred & (q <= NEG_Q) & in_bin
        eligible = positive.any(dim=1) & negative.any(dim=1)
        frozen[name] = {"positive": positive & eligible[:, None],
                         "negative": negative & eligible[:, None],
                         "eligible_anchor": eligible}
    return frozen


def _freeze_predictions(output: Path, predictions: dict, provenance: dict,
                        parity: dict, gamma_sha: str, gamma_manifest_sha: str) -> tuple[Path, str]:
    mask_hashes = {}
    for seed in SEEDS:
        seed_key = str(seed)
        seed_data = predictions[seed_key]
        masks_per_image = []
        for labels, q in zip(seed_data["labels"], seed_data["q"]):
            masks_per_image.append(_build_relation_masks(labels, q))
        seed_data["relation_masks"] = masks_per_image
        mask_hashes[seed_key] = [
            {bin_name: {kind: _relation_mask_hash(masks_per_image[i][bin_name][kind])
                        for kind in ("positive", "negative", "eligible_anchor")}
             for bin_name, _, _ in BINS}
            for i in range(len(IDS))]
    path = output / "frozen_predictions.pt"
    with path.open("xb") as stream:
        torch.save({"image_ids": list(IDS), "predictions": predictions}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256_file(path)
    manifest = {"status": "predictions_complete", "experiment": "SW0128",
        "image_ids": [IDS[0], IDS[-1]], "count": len(IDS), "seeds": list(SEEDS),
        "prediction_sha256": digest, "ground_truth_used_for_prediction": False,
        "optimizer_updates": 0, "ids": list(IDS), "batch_size": BATCH,
        "steps": STEPS, "settle": SETTLE,
        "readout": {"threshold": 0.50, "foreground_threshold": 0.15,
            "min_group_size": 2, "background": "largest_component",
            "synchrony_quantile": None, "target_foreground": None,
            "spatial_sigma": None, "affinity_mode": "spike", "grid_size": GRID},
        "source_provenance": provenance, "native_rollout_parity": parity,
        "relation_mask_sha256_by_seed_image_bin": mask_hashes,
        "validation_gamma_sha256": gamma_sha,
        "validation_gamma_manifest_sha256": gamma_manifest_sha,
        "implementation_fingerprint": _implementation_fingerprint()}
    _write_once(output / "prediction_manifest.json", manifest)
    if sha256_file(path) != digest:
        raise AssertionError("prediction bytes changed during freeze")
    frozen = torch.load(path, map_location="cpu", weights_only=True)
    if (frozen.get("image_ids") != list(IDS)
            or set(frozen.get("predictions", {})) != {str(seed) for seed in SEEDS}):
        raise ValueError("frozen prediction bundle failed its contract")
    for seed in SEEDS:
        row = frozen["predictions"][str(seed)]
        if tuple(row["labels"].shape) != (16, GRID, GRID) or tuple(row["q"].shape) != (16, 256, 256):
            raise ValueError("frozen prediction tensor shape mismatch")
        if len(row.get("relation_masks", [])) != 16 or not torch.isfinite(row["q"]).all():
            raise ValueError("frozen Q matrix contains nonfinite values")
        for image_index, frozen_masks in enumerate(row["relation_masks"]):
            rebuilt = _build_relation_masks(row["labels"][image_index], row["q"][image_index])
            for bin_name, _, _ in BINS:
                for kind in ("positive", "negative", "eligible_anchor"):
                    if not torch.equal(frozen_masks[bin_name][kind], rebuilt[bin_name][kind]):
                        raise ValueError("relation masks differ from frozen labels/Q")
    return path, digest


def _pair_audit(frozen_masks: dict, gt: torch.Tensor):
    gt = gt.reshape(256).long().cpu()
    same_gt_fg = (gt[:, None] > 0) & (gt[:, None] == gt[None, :])
    gt_fg_relevant = (gt[:, None] > 0) | (gt[None, :] > 0)
    result = {}
    for name, _, _ in BINS:
        pos = frozen_masks[name]["positive"]
        neg = frozen_masks[name]["negative"]
        pos_rel = pos & gt_fg_relevant
        neg_rel = neg & gt_fg_relevant
        result[name] = {
            "eligible_anchors": int(frozen_masks[name]["eligible_anchor"].sum()),
            "positive_candidates": int(pos.sum()),
            "negative_candidates": int(neg.sum()),
            "positive_fg_relevant": int(pos_rel.sum()),
            "negative_fg_relevant": int(neg_rel.sum()),
            "positive_correct": int((pos_rel & same_gt_fg).sum()),
            "negative_correct": int((neg_rel & ~same_gt_fg).sum()),
            "positive_fg_fg_same": int((pos & (gt[:, None] > 0) & (gt[None, :] > 0) & same_gt_fg).sum()),
            "positive_fg_fg_different": int((pos & (gt[:, None] > 0) & (gt[None, :] > 0) & ~same_gt_fg).sum()),
            "positive_fg_bg_bridge": int((pos & ((gt[:, None] > 0) ^ (gt[None, :] > 0))).sum()),
            "negative_fg_fg_different": int((neg & (gt[:, None] > 0) & (gt[None, :] > 0) & ~same_gt_fg).sum()),
            "negative_fg_bg_separated": int((neg & ((gt[:, None] > 0) ^ (gt[None, :] > 0))).sum()),
            "bg_bg_excluded": int(((pos | neg) & (gt[:, None] == 0)
                                    & (gt[None, :] == 0)).sum()),
        }
    return result


def _precision(correct: int, count: int):
    return float(correct / count) if count else None


def _score_frozen(output: Path, target_loader) -> dict:
    pred_path, manifest_path = output / "frozen_predictions.pt", output / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("prediction_sha256")
    if (manifest.get("status") != "predictions_complete"
            or manifest.get("experiment") != "SW0128"
            or manifest.get("image_ids") != [IDS[0], IDS[-1]]
            or manifest.get("ground_truth_used_for_prediction") is not False
            or manifest.get("implementation_fingerprint") != _implementation_fingerprint()
            or manifest.get("validation_gamma_sha256") != sha256_file(base.GAMMA_VAL)
            or manifest.get("validation_gamma_manifest_sha256") != sha256_file(base.GAMMA_VAL_MANIFEST)
            or not isinstance(expected, str) or sha256_file(pred_path) != expected):
        raise ValueError("frozen predictions invalid; refusing GT access")
    bundle = torch.load(pred_path, map_location="cpu", weights_only=True)
    if (bundle.get("image_ids") != list(IDS)
            or set(bundle.get("predictions", {})) != {str(seed) for seed in SEEDS}):
        raise ValueError("all-seed prediction bundle incomplete; refusing GT access")
    # Re-hash and validate every seed before the callback can open the dataset.
    for seed in SEEDS:
        item = bundle["predictions"][str(seed)]
        source, source_manifest, _ = _source_contract(seed)
        provenance = manifest.get("source_provenance", {}).get(str(seed), {})
        frozen_masks = item.get("relation_masks")
        mask_hashes = manifest.get("relation_mask_sha256_by_seed_image_bin", {}).get(str(seed))
        if (tuple(item["labels"].shape) != (16, GRID, GRID)
                or tuple(item["q"].shape) != (16, 256, 256)
                or not torch.isfinite(item["q"]).all()
                or not isinstance(frozen_masks, list) or len(frozen_masks) != 16
                or not isinstance(mask_hashes, list) or len(mask_hashes) != 16
                or provenance.get("source_checkpoint_sha256") != sha256_file(source)
                or provenance.get("source_manifest_sha256") != sha256_file(source_manifest)):
            raise ValueError("invalid frozen seed prediction; refusing GT access")
        for image_index in range(16):
            rebuilt = _build_relation_masks(item["labels"][image_index], item["q"][image_index])
            for bin_name, _, _ in BINS:
                for kind in ("positive", "negative", "eligible_anchor"):
                    stored = frozen_masks[image_index][bin_name][kind]
                    if (not torch.equal(stored, rebuilt[bin_name][kind])
                            or mask_hashes[image_index][bin_name][kind] != _relation_mask_hash(stored)):
                        raise ValueError("frozen relation mask hash/content mismatch; refusing GT access")
    if sha256_file(pred_path) != expected:
        raise ValueError("prediction hash changed before GT access")
    target = target_loader()
    if tuple(target.shape) != (16, GRID, GRID):
        raise ValueError("modal-8x8 GT labels have invalid shape")
    per_seed = {}
    for seed in SEEDS:
        pred = bundle["predictions"][str(seed)]
        per_image = []
        for i in range(16):
            per_image.append(_pair_audit(pred["relation_masks"][i], target[i]))
        bins = {}
        for name, _, _ in BINS:
            rows = [row[name] for row in per_image]
            eligible = [i for i, mask_row in enumerate(pred["relation_masks"])
                        if bool(mask_row[name]["eligible_anchor"].any())]
            pos_n = sum(row["positive_fg_relevant"] for row in rows)
            pos_ok = sum(row["positive_correct"] for row in rows)
            neg_n = sum(row["negative_fg_relevant"] for row in rows)
            neg_ok = sum(row["negative_correct"] for row in rows)
            pprec, nprec = _precision(pos_ok, pos_n), _precision(neg_ok, neg_n)
            bins[name] = {
                "eligible_images": len(eligible), "eligible_image_ids": [IDS[i] for i in eligible],
                "total_images": 16, "positive_fg_relevant_pairs": pos_n,
                "positive_correct_same_fg_object": pos_ok, "positive_precision": pprec,
                "negative_fg_relevant_pairs": neg_n,
                "negative_correct_not_same_fg_object": neg_ok, "negative_precision": nprec,
                "positive_fg_fg_same": sum(row["positive_fg_fg_same"] for row in rows),
                "positive_fg_fg_different": sum(row["positive_fg_fg_different"] for row in rows),
                "positive_fg_bg_bridge": sum(row["positive_fg_bg_bridge"] for row in rows),
                "negative_fg_fg_different": sum(row["negative_fg_fg_different"] for row in rows),
                "negative_fg_bg_separated": sum(row["negative_fg_bg_separated"] for row in rows),
                "bg_bg_excluded": sum(row["bg_bg_excluded"] for row in rows),
                "pass": (len(eligible) >= MIN_COVERAGE_IMAGES and pprec is not None
                         and nprec is not None and pprec >= MIN_PRECISION and nprec >= MIN_PRECISION
                         and pos_n >= MIN_PAIRS_PER_CLASS and neg_n >= MIN_PAIRS_PER_CLASS),
                "per_image": [{"image_id": IDS[i], **rows[i]} for i in range(16)],
            }
        # Count predicted connected foreground groups that merge distinct GT objects.
        merges = []
        for i in range(16):
            labels = pred["labels"][i].reshape(-1)
            gt = target[i].reshape(-1)
            for label in torch.unique(labels[labels > 0]).tolist():
                objects = torch.unique(gt[(labels == label) & (gt > 0)])
                if objects.numel() > 1:
                    merges.append({"image_id": IDS[i], "predicted_label": int(label),
                                   "ground_truth_object_ids": [int(v) for v in objects.tolist()]})
        per_seed[str(seed)] = {"bins": bins,
            "predicted_groups_merging_multiple_gt_objects": merges,
            "merge_count": len(merges)}
    passed = all(per_seed[str(seed)]["bins"][name]["pass"] for seed in SEEDS for name, _, _ in BINS)
    return {"status": "complete", "experiment": "SW0128", "image_ids": [IDS[0], IDS[-1]],
        "count": 16, "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True, "optimizer_updates": 0,
        "all_three_prediction_bundles_sha_verified_before_ground_truth": True,
        "prediction_sha256": expected, "prediction_manifest_sha256": sha256_file(manifest_path),
        "acceptance": {"precision_each_class_each_bin": MIN_PRECISION,
            "eligible_images_each_seed_bin": MIN_COVERAGE_IMAGES,
                "minimum_fg_relevant_pairs_per_class_aggregate": MIN_PAIRS_PER_CLASS,
            "all_seed_bins_pass": passed},
        "per_seed": per_seed, "interpretation": "Source QCC pair-relation diagnostic only; not a training or causal-gate claim."}


def _load_targets(dataset=DATASET):
    if Path(dataset).resolve() != DATASET.resolve():
        raise ValueError(f"only canonical CLEVR dataset is permitted: {DATASET}")
    import h5py
    with h5py.File(dataset, "r") as handle:
        masks = np.asarray(handle["mask"][IDS[0]:IDS[-1] + 1])
    return clevr_mask_patch(torch.from_numpy(masks), 8)["patch_labels"].long().cpu()


def generate(output: Path, device="cuda:0") -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0128 output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    gamma, _ = base.validate_gamma_cache(base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, validation=True)
    if tuple(gamma.shape) != (320, 8, 256):
        raise ValueError("validation cache does not match registered 320 image set")
    # _predict_all_seeds checks native/manual parity before each seed's outputs.
    predictions, provenance, parity = _predict_all_seeds(device)
    if set(parity) != {str(seed) for seed in SEEDS}:
        raise AssertionError("missing native parity check for a source seed")
    path, digest = _freeze_predictions(output, predictions, provenance, parity,
        sha256_file(base.GAMMA_VAL), sha256_file(base.GAMMA_VAL_MANIFEST))
    return {"status": "predictions_complete", "path": str(path), "sha256": digest,
            "seed_count": 3, "image_count": 16, "ground_truth_used": False}


def score(output: Path, dataset=DATASET) -> dict:
    output = Path(output)
    report_path = output / "qa_results.json"
    if report_path.exists():
        raise FileExistsError(f"preserving completed or failed SW0128 result: {report_path}")
    result = _score_frozen(output, lambda: _load_targets(dataset))
    result.update({"implementation_fingerprint": _implementation_fingerprint(),
                   "completed_unix": time.time()})
    _write_once(report_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "score"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args(argv)
    result = generate(args.output, args.device) if args.stage == "generate" else score(args.output, args.dataset)
    print(json.dumps({"status": result["status"], "experiment": "SW0128",
        "output": str(args.output), "stage": args.stage,
        "all_seed_bins_pass": result.get("acceptance", {}).get("all_seed_bins_pass")},
        allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
