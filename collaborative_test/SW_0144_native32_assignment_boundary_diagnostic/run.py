"""GT-free first64-train diagnostic of warm P fragmentation and spike Q at RGB edges."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0142_native32_masked_completion import run as sw142
from collaborative_test.SW_0135_native32_spike_binding.binder import (
    NativeSpikeSlotBinder, canonical_hard_labels,
)
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

NAME = "SW0144_native32_assignment_boundary_diagnostic"
ROWS = 64
GRID = 32
PATCH = 4
IMAGE = 128


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def four_connected_summary(labels):
    """Return foreground component masses and mass-weighted split fraction."""
    lab = np.asarray(labels)
    if lab.ndim != 2 or not np.issubdtype(lab.dtype, np.integer) or np.any(lab < 0):
        raise ValueError("hard labels must be a nonnegative 2-D integer grid")
    height, width = lab.shape
    total_fg = int(np.count_nonzero(lab))
    split_mass = 0
    by_label = []
    for value in sorted(int(v) for v in np.unique(lab) if v != 0):
        remaining = lab == value
        masses = []
        while remaining.any():
            y, x = np.argwhere(remaining)[0]
            stack = [(int(y), int(x))]
            remaining[y, x] = False
            mass = 0
            while stack:
                cy, cx = stack.pop()
                mass += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < height and 0 <= nx < width and remaining[ny, nx]):
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
            masses.append(mass)
        mass = int(sum(masses))
        largest = int(max(masses))
        split_mass += mass - largest
        by_label.append({"label": value, "mass": mass, "component_masses": masses,
                         "disconnected_fraction": (mass - largest) / mass})
    return {"foreground_patch_mass": total_fg, "labels": by_label,
            "mass_weighted_disconnected_fraction": (split_mass / total_fg if total_fg else 0.0)}


def visible_patch_mask(tiles):
    tile_rows = np.asarray(tiles)
    if tile_rows.ndim != 1 or not np.issubdtype(tile_rows.dtype, np.integer):
        raise ValueError("tiles must be a vector of integer 8x8 tile IDs")
    if np.any(tile_rows < 0) or np.any(tile_rows >= 64) or len(np.unique(tile_rows)) != len(tile_rows):
        raise ValueError("tile IDs must be unique values in [0,64)")
    visible = np.ones((GRID, GRID), dtype=bool)
    for tile in tile_rows:
        row, col = divmod(int(tile), 8)
        visible[row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = False
    return visible


def adjacent_boundary_edges(rgb, tiles):
    """Return visible 4-neighbor patch pairs and their true RGB seam MSE."""
    image = np.asarray(rgb)
    if image.shape != (IMAGE, IMAGE, 3) or image.dtype != np.uint8:
        raise ValueError("RGB must be one native128 uint8 image")
    visible = visible_patch_mask(tiles)
    pairs, errors = [], []
    # Horizontal patch neighbors: compare the two pixel columns sharing seam.
    for row in range(GRID):
        for col in range(GRID - 1):
            if visible[row, col] and visible[row, col + 1]:
                left = image[row * PATCH:(row + 1) * PATCH, (col + 1) * PATCH - 1].astype(np.float64)
                right = image[row * PATCH:(row + 1) * PATCH, (col + 1) * PATCH].astype(np.float64)
                pairs.append((row * GRID + col, row * GRID + col + 1))
                errors.append(float(np.mean(((left - right) / 255.0) ** 2)))
    # Vertical patch neighbors.
    for row in range(GRID - 1):
        for col in range(GRID):
            if visible[row, col] and visible[row + 1, col]:
                top = image[(row + 1) * PATCH - 1, col * PATCH:(col + 1) * PATCH].astype(np.float64)
                bottom = image[(row + 1) * PATCH, col * PATCH:(col + 1) * PATCH].astype(np.float64)
                pairs.append((row * GRID + col, (row + 1) * GRID + col))
                errors.append(float(np.mean(((top - bottom) / 255.0) ** 2)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), np.asarray(errors, dtype=np.float64)


def _correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def batch_observations(spikes, rgbs, tiles, assignments):
    """Build per-image no-GT observations from actual cached S and binder P."""
    if not torch.is_tensor(spikes) or spikes.ndim != 4 or tuple(spikes.shape[1:]) != (4, 1024, 512):
        raise ValueError("cached emitted spikes must be [B,4,1024,512]")
    if not torch.isfinite(spikes).all():
        raise ValueError("cached emitted spikes contain nonfinite values")
    b = spikes.shape[0]
    images = np.asarray(rgbs)
    tile_batch = np.asarray(tiles)
    if images.shape != (b, 128, 128, 3) or images.dtype != np.uint8 or tile_batch.shape[0] != b:
        raise ValueError("cache RGB/tiles do not align with spike batch")
    if tuple(assignments.shape) != (b, 1024, 11) or not torch.isfinite(assignments).all():
        raise ValueError("warm binder assignments must be finite [B,1024,11]")
    labels = canonical_hard_labels(assignments).cpu().numpy()
    # Production QCC definition: mean component activity plus componentwise
    # positive-clamped synchrony product, directly from emitted cached spikes.
    q = spike_synchrony_affinity(spikes.mean(dim=1), components=spikes,
                                 settle=0, affinity_mode="spike")
    if q.shape != (b, 1024, 1024) or not torch.isfinite(q).all():
        raise ValueError("actual-spike Q affinity has invalid shape or values")
    records = []
    edge_errors = []
    for i in range(b):
        pairs, errors = adjacent_boundary_edges(images[i], tile_batch[i])
        if len(pairs):
            edge_errors.extend(errors.tolist())
        q_cpu = q[i].detach().cpu().numpy()
        edge_q = q_cpu[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.empty(0, dtype=np.float32)
        records.append({"assignment": four_connected_summary(labels[i]),
                        "edges": {"pairs": pairs.tolist(), "rgb_boundary_mse": errors.tolist(),
                                  "q_affinity": np.asarray(edge_q, dtype=np.float64).tolist()}})
    return records, np.asarray(edge_errors, dtype=np.float64)


def _edge_summary(records, scale):
    per_image = []
    all_q, all_target = [], []
    for index, row in enumerate(records):
        edges = row["edges"]
        error = np.asarray(edges["rgb_boundary_mse"], dtype=np.float64)
        q = np.asarray(edges["q_affinity"], dtype=np.float64)
        target = np.exp(-error / scale)
        if not np.isfinite(q).all() or not np.isfinite(target).all():
            raise FloatingPointError("nonfinite edge relation diagnostic")
        per_image.append({"row": index, "visible_adjacent_edges": int(error.size),
                          "mean_rgb_boundary_mse": float(error.mean()) if error.size else None,
                          "mean_actual_spike_q": float(q.mean()) if q.size else None,
                          "mean_exp_boundary_target": float(target.mean()) if target.size else None,
                          "q_vs_exp_boundary_pearson": _correlation(q, target),
                          "mean_absolute_difference": float(np.abs(q - target).mean()) if error.size else None})
        all_q.extend(q.tolist()); all_target.extend(target.tolist())
    return {"per_image": per_image, "pooled_visible_edge_count": len(all_q),
            "pooled_q_vs_exp_boundary_pearson": _correlation(all_q, all_target),
            "pooled_mean_absolute_difference": (float(np.mean(np.abs(np.asarray(all_q) - np.asarray(all_target))))
                                                 if all_q else None)}


def run_diagnostic(*, output, rows=ROWS):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"preserve existing diagnostic output: {destination}")
    if rows != ROWS:
        raise ValueError("diagnostic scope is fixed to the first64 registered TRAIN rows")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU diagnostic must start without an initialized CUDA context")
    torch.set_num_threads(1)
    foundation = sw142.source_foundation(device="cpu", verify_assets=True)
    cache = sw142.MaskCache(sw142.CACHE, foundation)
    warm_manifest, warm_state = sw142._warm_state("cpu")
    binder = NativeSpikeSlotBinder(seed=142)
    binder.load_state_dict(warm_state["binder"], strict=True)
    binder.eval()
    observations, boundary = [], []
    with torch.inference_mode():
        for start in range(0, rows, sw142.MICRO):
            spikes, rgb, tiles = cache.batch(start, "cpu")
            assignments, _slots, _features = binder(spikes)
            block, errors = batch_observations(spikes, rgb.numpy(), tiles, assignments)
            observations.extend(block)
            boundary.extend(errors.tolist())
    scale = float(np.mean(boundary)) if boundary else 0.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("first64 descriptive RGB boundary scale is not positive and finite")
    ordered_ids = list(map(int, cache.manifest["image_ids"][:rows]))
    source = {k: foundation.provenance[k] for k in (
        "source_core_sha256", "source_manifest_sha256", "source_training_ids_sha256",
        "encoder_sha256", "feature_preprocessing_sha256", "rgb_asset_validation")}
    record = {"status": "complete", "experiment": NAME, "seed": 1,
              "scope": "first64 registered source1 TRAIN rows from SW0142 masked cache; descriptive only",
              "image_ids": ordered_ids,
              "training_ids_sha256": cache.manifest["training_ids_sha256"],
              "cache_manifest_sha256": sha256_file(sw142.CACHE / "manifest.json"),
              "diagnostic_protocol_sha256": sha256_file(HERE / "protocol.json"),
              "cache_implementation_fingerprint": cache.manifest["implementation_fingerprint"],
              "warm_manifest_sha256": sha256_file(sw142.WARM / "manifest.json"),
              "warm_checkpoint_sha256": warm_manifest["checkpoint_sha256"],
              "warm_optimizer_sha256": warm_manifest["optimizer_sha256"],
              "source": source,
              "boundary_scale": {"value": scale,
                                  "definition": "mean visible adjacent-patch RGB boundary pixel-pair MSE over these same64 rows",
                                  "scope": "descriptive scale only; not training calibration"},
              "assignment_fragmentation": [x["assignment"] for x in observations],
              "q_vs_rgb_boundary": _edge_summary(observations, scale),
              "ground_truth_used": False, "optimizer_updates": 0,
              "cuda_initialized": torch.cuda.is_initialized(),
              "model_parameters_changed": False,
              "limitations": ["Masked-cache float16 actual spikes; no clean-input parity or gradient claim.",
                              "First64-only descriptive association; no threshold or training decision.",
                              "No GT, training, optimizer step, prediction, or model update."]}
    record["implementation_sha256"] = sha256_file(HERE / "run.py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results_archive" / "first64_assignment_boundary.json")
    args = parser.parse_args(argv)
    record = run_diagnostic(output=args.output)
    print(json.dumps({"status": record["status"], "rows": len(record["image_ids"]),
                      "output": str(args.output), "ground_truth_used": False}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
