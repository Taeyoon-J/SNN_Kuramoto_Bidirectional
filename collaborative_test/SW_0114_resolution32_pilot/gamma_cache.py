"""On-demand exact-RGB gamma regeneration for SW0114 selected IDs only."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")


def cache_rows(global_ids):
    ids = np.asarray(global_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("IDs must be a unique vector")
    if np.any((ids < 0) | ((ids >= 1000) & (ids < 1640)) | (ids >= 70640)):
        raise ValueError("requested image IDs leave the registered train/validation pools")
    return np.where(ids < 1000, ids, ids - 640).astype(np.int64)


def hdf5_indices(global_ids, *, allow_validation=False):
    ids = np.asarray(global_ids, dtype=np.int64)
    if ids.ndim != 1 or np.any((ids < 0) | (ids >= 70640)):
        raise ValueError("HDF5 image IDs are outside the available non-reserved range")
    train_ids = (ids < 1000) | (ids >= 1640)
    val_ids = (ids >= 1320) & (ids < 1640)
    if not np.all(train_ids | (allow_validation & val_ids)):
        raise ValueError("reserved validation gap must not be read as train data")
    # Source IDs are literal HDF5 image indices. The packed gamma-cache row
    # mapping (IDs >=1640 -> ID-640) is separate and must never affect RGB I/O.
    return ids.copy()


def ordered_rgb_batch(dataset, global_ids, *, allow_validation=False):
    ids = hdf5_indices(global_ids, allow_validation=allow_validation)
    order = np.argsort(ids)
    sorted_images = dataset["image"][ids[order].tolist()]
    images = sorted_images[np.argsort(order)]
    if images.shape != (len(ids), 128, 128, 3) or images.dtype != np.uint8:
        raise AssertionError(f"expected native uint8 RGB batch, got {images.shape}/{images.dtype}")
    return torch.from_numpy(images.copy()).permute(0, 3, 1, 2)


def encode_gamma(images, encoder, stats, grid_size, device):
    if grid_size not in (16, 32):
        raise ValueError("registered gamma grids are 16 and 32")
    x = images.to(device=device, dtype=torch.float32) / 255.
    features = encoder(x)
    mean = stats["mean"].to(device=device, dtype=features.dtype)
    std = stats["std"].to(device=device, dtype=features.dtype)
    clip = float(stats.get("clip", 3.))
    if stats.get("mode") != "standardize" or not torch.isfinite(std).all() or (std <= 0).any():
        raise AssertionError("registered feature standardization contract is invalid")
    normalized = ((features - mean) / std).clamp(-clip, clip)
    pooled = F.adaptive_avg_pool2d(normalized, (grid_size, grid_size))
    gamma = pooled.flatten(2)
    if tuple(gamma.shape[1:]) != (8, grid_size * grid_size) or not torch.isfinite(gamma).all():
        raise AssertionError("native gamma tensor has invalid shape or values")
    return gamma


def ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def dataset_identity():
    stat = DATASET.stat()
    with h5py.File(DATASET, "r") as handle:
        shape = list(handle["image"].shape)
    return {"path": str(DATASET), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "image_shape": shape}
