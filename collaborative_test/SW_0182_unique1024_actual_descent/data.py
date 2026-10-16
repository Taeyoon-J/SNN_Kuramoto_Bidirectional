"""Immutable canonical TRAIN1024 RGB/modal-mask input extension for SW0182."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "results_archive" / "training_inputs_20261016"
PREFIX_COUNT = 512
COUNT = 1024
GRID = 32
PIXEL = 128
FORBIDDEN = tuple(range(1320, 1640)) + tuple(range(90640, 90960))


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(name: str):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def extend_arrays(prefix: dict, *, full_ids, next_rgb, next_modal) -> dict:
    """Append the next canonical 512 rows without touching the frozen prefix."""
    ids = np.asarray(full_ids, dtype=np.int64)
    old_rgb = np.asarray(prefix["rgb"])
    old_modal = np.asarray(prefix["modal_gt"])
    rgb = np.asarray(next_rgb)
    modal = np.asarray(next_modal)
    if ids.shape != (COUNT,) or len(np.unique(ids)) != COUNT:
        raise ValueError("source contract must provide 1024 unique ordered TRAIN IDs")
    if old_rgb.dtype != np.uint8 or old_rgb.shape != (PREFIX_COUNT, PIXEL, PIXEL, 3):
        raise ValueError("frozen SW0171 RGB prefix has invalid schema")
    if old_modal.dtype != np.int32 or old_modal.shape != (PREFIX_COUNT, GRID, GRID):
        raise ValueError("frozen SW0171 full-modal prefix has invalid schema")
    if rgb.dtype != np.uint8 or rgb.shape != (PREFIX_COUNT, PIXEL, PIXEL, 3):
        raise ValueError("new canonical TRAIN RGB rows have invalid schema")
    if modal.dtype != np.int32 or modal.shape != (PREFIX_COUNT, GRID, GRID):
        raise ValueError("new canonical TRAIN modal rows have invalid schema")
    if np.intersect1d(ids, np.asarray(FORBIDDEN, dtype=np.int64)).size:
        raise ValueError("1024 TRAIN IDs include registered validation or reserve rows")
    return {
        "rgb": np.concatenate((old_rgb, rgb), axis=0),
        "modal_gt": np.concatenate((old_modal, modal), axis=0),
        "image_ids": ids.copy(),
    }


def _validate_arrays(arrays: dict, expected_ids) -> None:
    if set(arrays) != {"rgb", "modal_gt", "image_ids"}:
        raise ValueError("SW0182 cache may contain only RGB, canonical modal GT, and IDs")
    if arrays["rgb"].dtype != np.uint8 or arrays["rgb"].shape != (COUNT, PIXEL, PIXEL, 3):
        raise ValueError("SW0182 RGB array schema mismatch")
    if arrays["modal_gt"].dtype != np.int32 or arrays["modal_gt"].shape != (COUNT, GRID, GRID):
        raise ValueError("SW0182 modal GT array schema mismatch")
    if arrays["image_ids"].dtype != np.int64 or arrays["image_ids"].shape != (COUNT,):
        raise ValueError("SW0182 image ID array schema mismatch")
    actual = list(map(int, arrays["image_ids"].tolist()))
    if actual != list(map(int, expected_ids)) or len(set(actual)) != COUNT:
        raise ValueError("SW0182 IDs differ from exact canonical TRAIN1024 order")
    if np.intersect1d(np.asarray(actual), np.asarray(FORBIDDEN, dtype=np.int64)).size:
        raise ValueError("SW0182 cache includes validation/reserve IDs")
    if np.any(arrays["modal_gt"] < 0):
        raise ValueError("modal GT labels must be nonnegative integer IDs")


def _write_cache(output: Path, arrays: dict, manifest: dict) -> tuple[str, str]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"immutable SW0182 input cache already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".stage-" + uuid.uuid4().hex)
    if not stage.resolve().is_relative_to(output.parent.resolve()):
        raise ValueError("SW0182 staging path escaped results archive")
    stage.mkdir(parents=False, exist_ok=False)
    arrays_path = stage / "arrays.npz"
    with arrays_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    arrays_sha = _sha_file(arrays_path)
    payload = dict(manifest)
    payload["arrays_sha256"] = arrays_sha
    manifest_path = stage / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    manifest_sha = _sha_file(manifest_path)
    os.rename(stage, output)
    return manifest_sha, arrays_sha


def _read_cache(output: Path, *, ids, prefix, expected_binding) -> dict:
    """Reuse a completed cache only when its canonical binding still matches."""
    manifest_path, arrays_path = output / "manifest.json", output / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise ValueError("existing SW0182 cache is incomplete; refusing replacement")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    if (manifest.get("experiment") != "SW0182_canonical_source_TRAIN1024_inputs"
            or manifest.get("status") != "complete"
            or manifest.get("count") != COUNT
            or manifest.get("image_ids") != list(map(int, ids))):
        raise ValueError("existing SW0182 cache manifest does not match TRAIN1024 contract")
    for key, value in expected_binding.items():
        if manifest.get(key) != value:
            raise ValueError(f"existing SW0182 cache lineage changed: {key}")
    arrays_sha = _sha_file(arrays_path)
    if manifest.get("arrays_sha256") != arrays_sha:
        raise ValueError("existing SW0182 cache arrays hash does not match its manifest")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    _validate_arrays(arrays, ids)
    if (not np.array_equal(arrays["rgb"][:PREFIX_COUNT], prefix["rgb"])
            or not np.array_equal(arrays["modal_gt"][:PREFIX_COUNT], prefix["modal_gt"])):
        raise ValueError("existing SW0182 cache changed byte-exact TRAIN512 prefix")
    return {"arrays": arrays, "manifest": manifest,
            "manifest_sha256": _sha_bytes(manifest_bytes), "arrays_sha256": arrays_sha}


def prepare_training_inputs_cpu(*, output_dir: Path | None = None) -> dict:
    """Validate SW0171 prefix and append canonical TRAIN rows 512:1024.

    Only TRAIN RGB and full modal masks are read. No spike, feasibility, VAL,
    or reserve arrays are produced or consumed.
    """
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0182 data preparation must precede CUDA initialization")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    old = _module("collaborative_test.SW_0179_qce_seed_replication.train")
    source = _module("collaborative_test.SW_0130_phase_state_integration.run")
    aligned = _module("collaborative_test.SW_0155_supervised_spike_relations.train")
    objective = _module("collaborative_test.SW_0155_supervised_spike_relations.objective")
    sw171 = _module("collaborative_test.SW_0171_contextual_instance_graph.preflight")
    prefix = old.prepare_training_inputs_cpu()
    prefix_arrays = prefix["arrays"]
    contract = source.source_contract(1)
    if len(contract) != 6:
        raise ValueError("source_contract(1) return shape changed")
    _checkpoint, source_manifest_path, _source_manifest, pool_rows, ordered_ids, checkpoint_sha = contract
    ids = list(map(int, ordered_ids[:COUNT]))
    if len(ids) != COUNT or len(set(ids)) != COUNT:
        raise ValueError("registered seed-1 source TRAIN prefix lacks 1024 unique IDs")
    if ids[:PREFIX_COUNT] != prefix["ordered_train_ids"]:
        raise ValueError("SW0171 frozen 512 IDs are not the exact SW0182 prefix")
    if np.intersect1d(np.asarray(ids), np.asarray(FORBIDDEN, dtype=np.int64)).size:
        raise ValueError("canonical ordered TRAIN1024 overlaps fixed validation/reserve")
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    if pool_rows.ndim != 1 or len(pool_rows) < COUNT or np.any(pool_rows[:COUNT] < 0):
        raise ValueError("seed-1 source contract has invalid RGB row mapping")

    sw130 = _module("collaborative_test.SW_0130_phase_state_integration.run")
    rgb_path = Path(sw130.TRAIN_RGB)
    rgb_meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    if not rgb_meta_path.is_file():
        raise FileNotFoundError("canonical TRAIN RGB completion manifest is missing")
    rgb_manifest_sha = _sha_file(rgb_meta_path)
    source_manifest_path = Path(source_manifest_path)
    source_manifest_sha = _sha_file(source_manifest_path)
    hdf_path = Path(aligned.HDF5_PATH).resolve()
    hdf_stat = hdf_path.stat()
    prefix_manifest_sha = prefix["dataset_manifest_sha256"]
    prefix_arrays_sha = prefix["dataset_arrays_sha256"]
    binding = {
        "prefix_512_manifest_sha256": prefix_manifest_sha,
        "prefix_512_arrays_sha256": prefix_arrays_sha,
        "source_seed": 1,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_manifest_sha256": source_manifest_sha,
        "train_rgb_manifest_sha256": rgb_manifest_sha,
        "source_pool_rows_sha256": _sha_bytes(np.asarray(pool_rows[:COUNT], dtype="<i8").tobytes()),
        "canonical_hdf5_path": str(hdf_path),
        "canonical_hdf5_size_bytes": hdf_stat.st_size,
        "canonical_hdf5_mtime_ns": hdf_stat.st_mtime_ns,
        "new_rows_start": PREFIX_COUNT,
        "new_rows_count": PREFIX_COUNT,
        "new_image_ids": ids[PREFIX_COUNT:],
        "modal_builder": "SW0155 objective.modal_train_targets32 on canonical HDF5 TRAIN masks",
        "ground_truth_scope": "TRAIN modal32 only; no VAL or reserve data",
        "gamma_or_spike_cache_created": False,
        "feasible_projected_targets_created": False,
        "optimizer_updates": 0,
        "validation_data_read": False,
        "cuda_initialized": False,
    }
    output = Path(output_dir) if output_dir is not None else OUTPUT
    if output.exists():
        reused = _read_cache(output, ids=ids, prefix=prefix_arrays, expected_binding=binding)
        if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
            raise RuntimeError("CPU cache reuse changed CUDA state/visibility")
        return {"arrays": reused["arrays"], "ordered_train_ids": ids,
                "cache_dir": str(output.resolve()), "manifest_sha256": reused["manifest_sha256"],
                "arrays_sha256": reused["arrays_sha256"],
                "prefix_manifest_sha256": prefix_manifest_sha,
                "prefix_arrays_sha256": prefix_arrays_sha,
                "source_checkpoint_sha256": checkpoint_sha,
                "validation_data_read": False, "cuda_initialized": False,
                "manifest": reused["manifest"], "cache_reused": True}
    rgb_memmap = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if rgb_memmap.dtype != np.uint8 or tuple(rgb_memmap.shape) != (70000, 128, 128, 3):
        raise ValueError("canonical TRAIN RGB asset schema changed")
    all_rgb = np.asarray(rgb_memmap[pool_rows[:COUNT]]).copy()
    del rgb_memmap
    if not np.array_equal(all_rgb[:PREFIX_COUNT], prefix_arrays["rgb"]):
        raise ValueError("frozen SW0171 RGB prefix differs from canonical TRAIN asset")
    new_rgb = all_rgb[PREFIX_COUNT:]
    new_ids = ids[PREFIX_COUNT:]
    aligned_rgb, masks = aligned.load_aligned_train_rows(
        aligned.HDF5_PATH, new_ids, new_rgb, 0, PREFIX_COUNT)
    if not np.array_equal(aligned_rgb.cpu().numpy(), new_rgb):
        raise ValueError("HDF5 aligned RGB differs from canonical TRAIN cache rows")
    modal_t = objective.modal_train_targets32(masks)
    new_modal = modal_t.reshape(PREFIX_COUNT, GRID, GRID).cpu().numpy().astype(np.int32, copy=False)
    del aligned_rgb, masks, modal_t
    arrays = extend_arrays(prefix_arrays, full_ids=ids, next_rgb=new_rgb, next_modal=new_modal)
    _validate_arrays(arrays, ids)

    manifest = {
        "experiment": "SW0182_canonical_source_TRAIN1024_inputs",
        "status": "complete",
        "count": COUNT,
        "image_ids": ids,
        "schema": {
            "rgb": {"shape": [COUNT, PIXEL, PIXEL, 3], "dtype": "uint8"},
            "modal_gt": {"shape": [COUNT, GRID, GRID], "dtype": "int32"},
            "image_ids": {"shape": [COUNT], "dtype": "int64"},
        },
        **binding,
    }
    manifest_sha, arrays_sha = _write_cache(output, arrays, manifest)
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise RuntimeError("CPU data preparation changed CUDA state/visibility")
    return {"arrays": arrays, "ordered_train_ids": ids, "cache_dir": str(output.resolve()),
            "manifest_sha256": manifest_sha, "arrays_sha256": arrays_sha,
            "prefix_manifest_sha256": prefix_manifest_sha,
            "prefix_arrays_sha256": prefix_arrays_sha,
            "source_checkpoint_sha256": checkpoint_sha, "validation_data_read": False,
            "cuda_initialized": False, "manifest": manifest}
