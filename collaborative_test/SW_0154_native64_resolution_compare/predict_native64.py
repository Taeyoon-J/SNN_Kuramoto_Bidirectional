"""Read-only fixed320 native64 source predictions with per-image resumable ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PATCH_V2_REPO", HERE.parents[1])).resolve()
if not (ROOT / "collaborative_test").is_dir():
    raise RuntimeError(f"invalid PATCH_V2_REPO: {ROOT}")
for path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150
from collaborative_test.SW_0154_native64_resolution_compare import resource_probe
from collaborative_test.SW_0154_native64_resolution_compare.native64_model import GRID, NODES
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components

SEEDS = (0, 1, 2)
IMAGE_IDS = tuple(contract.IMAGE_IDS)
COUNT = contract.COUNT
DATASET = Path(sw130.source97.DATASET)
PREDICTION_ROOT = HERE / "predictions_native64"
WORK_ROOT = HERE / "prediction_work_native64"
STEPS, SETTLE, MODEL_INIT_SEED, INFERENCE_SEED = 1024, 512, 154002, 154001


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def implementation_fingerprint():
    paths = (HERE / "predict_native64.py", HERE / "resource_probe.py",
        HERE / "native64_model.py", HERE / "resolution64.py", HERE / "score_slot64.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/preflight.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("native64 prediction dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _model_state_digest(wrapped, encoder, patcher):
    digest = hashlib.sha256()
    for group_name, module in (("wrapped", wrapped), ("encoder", encoder),
                               ("patcher", patcher)):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(group_name.encode())
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserve existing native64 artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temp.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"concurrent native64 artifact appeared: {path}")
    os.replace(temp, path)


def _write_protocol_once(path, record):
    path = Path(path)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"preserve existing native64 protocol {path}")
    os.replace(temp, path)


def _publish_prediction(output_dir, image_ids, labels, protocol_builder):
    """Expose NPZ+protocol together via a create-once sibling-directory rename."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"preserve existing native64 prediction directory {output_dir}")
    stage_dir = output_dir.with_name(output_dir.name + ".finalizing")
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_npz, staged_protocol = stage_dir / "predictions.npz", stage_dir / "protocol.json"
    unexpected = {p.name for p in stage_dir.iterdir()} - {staged_npz.name, staged_protocol.name}
    if unexpected:
        raise FileExistsError(f"preserve unexpected native64 finalization files: {sorted(unexpected)}")
    if staged_protocol.exists() and not staged_npz.exists():
        raise ValueError("preserve incomplete native64 finalization protocol without NPZ")
    if staged_npz.exists():
        with np.load(staged_npz, allow_pickle=False) as existing:
            if (not {"image_ids", "labels"}.issubset(existing.files)
                    or not np.array_equal(existing["image_ids"], image_ids)
                    or not np.array_equal(existing["labels"], labels)):
                raise ValueError("staged native64 NPZ differs from completed immutable image ledger")
    else:
        _atomic_npz(staged_npz, image_ids=image_ids, labels=labels)
    prediction_sha = sha256_file(staged_npz)
    protocol = protocol_builder(prediction_sha)
    if staged_protocol.exists():
        if _read_protocol(staged_protocol) != protocol:
            raise ValueError("staged native64 protocol differs from rebuilt ledger provenance")
    else:
        _write_protocol_once(staged_protocol, protocol)
    os.replace(stage_dir, output_dir)
    return output_dir / "predictions.npz", output_dir / "protocol.json", protocol


def _read_protocol(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_resume_row(path, *, seed, image_id, source_sha, manifest_sha, assets,
                     fingerprint, rgb_sha, parity_sha, model_seed=MODEL_INIT_SEED):
    with np.load(path, allow_pickle=False) as saved:
        required = {"labels", "record_json"}
        if not required.issubset(saved.files):
            raise ValueError("preserving malformed native64 per-image resume record")
        labels = np.asarray(saved["labels"])
        record = json.loads(str(saved["record_json"].item()))
    if (labels.shape != (GRID, GRID) or labels.dtype != np.int64 or np.any(labels < 0)
            or record.get("seed") != seed or record.get("image_id") != image_id
            or record.get("source_checkpoint_sha256") != source_sha
            or record.get("source_manifest_sha256") != manifest_sha
            or record.get("rgb_asset_validation") != assets
            or record.get("implementation_fingerprint") != fingerprint
            or record.get("input_rgb_sha256") != rgb_sha
            or record.get("model_initialization_seed") != model_seed
            or record.get("saved_native32_parity_sha256") != parity_sha
            or record.get("inference_seed") != INFERENCE_SEED
            or record.get("steps") != STEPS or record.get("settle") != SETTLE
            or record.get("live_tail_steps") != 0
            or record.get("grid_size") != [GRID, GRID]
            or record.get("ground_truth_used_for_prediction") is not False):
        raise ValueError("native64 per-image resume lineage/schema mismatch")
    return labels, record


def _predict_one(wrapped, encoder, patcher, mean, std, clip, image, device):
    rgb = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    tensor = rgb.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        feature = encoder(tensor)
        gamma = patcher(((feature - mean) / std).clamp(-clip, clip))
        if tuple(gamma.shape) != (1, 8, NODES) or not torch.isfinite(gamma).all():
            raise FloatingPointError("native64 gamma must be finite [1,8,4096]")
        trace = late_rollout(wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
        for name in ("component_spikes", "component_membrane", "component_gates",
                     "spikes", "membrane", "theta"):
            value = trace[name]
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"native64 {name} is nonfinite")
        groups = spike_synchrony_components(
            trace["spikes"].detach().cpu(), synchrony_threshold=.50, min_group_size=32,
            settle=SETTLE, components=trace["component_spikes"].detach().cpu(),
            background="largest_component", affinity_mode="spike", spatial_grid_size=GRID)
        labels = spatial_components_to_patch_labels(groups, GRID, device="cpu")
    result = labels[0].numpy().astype(np.int64, copy=False)
    if result.shape != (GRID, GRID) or np.any(result < 0):
        raise ValueError("native64 QCC did not produce nonnegative64x64 labels")
    return result


def predict_seed(seed, *, device="cuda:0", dataset=DATASET,
                 prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT):
    if seed not in SEEDS:
        raise ValueError("native64 predictions are registered for seeds0/1/2 only")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native64 fixed320 inference requires an exclusively leased CUDA device")
    torch.cuda.set_device(dev)
    checkpoint, source_manifest_path, manifest, pool, train_ids, source_sha = sw130.source_contract(seed)
    if len(train_ids) != 4096 or len(pool) != 4096:
        raise ValueError("registered source seed TRAIN order must contain exactly4096 IDs")
    assets = sw130.validate_rgb_assets()
    parity_path = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive" / (
        f"native32_zero_adapter_parity_seed{seed}_b4.json")
    parity, parity_sha = sw150.load_and_validate_parity(
        parity_path, source_sha=source_sha,
        source_manifest_sha=sha256_file(source_manifest_path),
        training_ids=train_ids, asset_validation=assets)
    del parity
    cache_path = Path(sw130.VAL_RGB)
    if not cache_path.is_file():
        raise FileNotFoundError("registered fixed320 validation RGB cache is missing")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("registered validation RGB cache does not match fixed320 native128 schema")
    if not np.array_equal(np.asarray(IMAGE_IDS, dtype=np.int64), np.asarray(contract.IMAGE_IDS)):
        raise AssertionError("fixed320 native64 inference order drifted")
    dataset = Path(dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    dataset_stat = dataset.stat()
    with h5py.File(dataset, "r") as h5:
        images = h5["image"]
        if images.shape != (100000, 128, 128, 3):
            raise ValueError("canonical RGB HDF5 schema changed")
        sample = np.asarray(images[IMAGE_IDS[0]:IMAGE_IDS[0] + 1])
    if sample.dtype == np.uint8:
        sample_u8 = sample
    elif sample.dtype in (np.float16, np.float32):
        sample_u8 = np.rint(np.clip(sample.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    else:
        raise ValueError("canonical evaluation RGB dtype changed")
    if not np.array_equal(sample_u8[0], np.asarray(cache[0])):
        raise ValueError("canonical fixed320 HDF5 RGB differs from registered VAL RGB cache")
    fingerprint = implementation_fingerprint()
    output_dir = Path(prediction_root) / f"seed{seed}"
    output_npz, protocol_path = output_dir / "predictions.npz", output_dir / "protocol.json"
    if output_dir.exists():
        raise FileExistsError("preserve existing native64 completed prediction directory")
    work_dir = Path(work_root) / f"seed{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)
    source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rng_device = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[rng_device]):
        torch.manual_seed(MODEL_INIT_SEED)
        torch.cuda.manual_seed(MODEL_INIT_SEED)
        wrapped, encoder, patcher, mean, std, clip = resource_probe._build_native64(
            seed, dev, source_state)
    wrapped.eval()
    encoder.eval()
    patcher.eval()
    model_before = _model_state_digest(wrapped, encoder, patcher)
    manifest_sha = sha256_file(source_manifest_path)
    image_records = []
    for index, image_id in enumerate(IMAGE_IDS):
        image = np.asarray(cache[index]).copy()
        input_sha = sha256_bytes(image.tobytes())
        row_path = work_dir / f"image_{image_id}.npz"
        if row_path.exists():
            labels, record = _read_resume_row(row_path, seed=seed, image_id=image_id,
                source_sha=source_sha, manifest_sha=manifest_sha, assets=assets,
                fingerprint=fingerprint, rgb_sha=input_sha, parity_sha=parity_sha,
                model_seed=MODEL_INIT_SEED)
            image_records.append(record)
            continue
        with torch.random.fork_rng(devices=[rng_device]):
            torch.manual_seed(INFERENCE_SEED)
            torch.cuda.manual_seed(INFERENCE_SEED)
            labels = _predict_one(wrapped, encoder, patcher, mean, std, clip, image, dev)
        record = {"seed": seed, "image_id": image_id,
            "input_rgb_sha256": input_sha, "source_checkpoint_sha256": source_sha,
            "source_manifest_sha256": manifest_sha, "rgb_asset_validation": assets,
            "saved_native32_parity_sha256": parity_sha,
            "model_initialization_seed": MODEL_INIT_SEED,
            "inference_seed": INFERENCE_SEED, "grid_size": [GRID, GRID],
            "steps": STEPS, "settle": SETTLE, "live_tail_steps": 0,
            "ground_truth_used_for_prediction": False,
            "implementation_fingerprint": fingerprint}
        _atomic_npz(row_path, labels=labels, record_json=np.asarray(json.dumps(record, sort_keys=True)))
        image_records.append(record)
        if (index + 1) % 16 == 0:
            print(json.dumps({"seed": seed, "completed_images": index + 1, "total_images": COUNT}),
                  flush=True)
    if len(image_records) != COUNT or [row["image_id"] for row in image_records] != list(IMAGE_IDS):
        raise AssertionError("native64 fixed320 resume ledger has missing or reordered images")
    if _model_state_digest(wrapped, encoder, patcher) != model_before:
        raise AssertionError("read-only native64 prediction mutated model or encoder state")
    labels_all = np.stack([_read_resume_row(work_dir / f"image_{image_id}.npz",
        seed=seed, image_id=image_id, source_sha=source_sha, manifest_sha=manifest_sha,
        assets=assets, fingerprint=fingerprint,
        rgb_sha=image_records[index]["input_rgb_sha256"], parity_sha=parity_sha,
        model_seed=MODEL_INIT_SEED)[0]
        for index, image_id in enumerate(IMAGE_IDS)])
    def protocol_builder(prediction_sha):
        return {"experiment": "SW0154_native64_resolution_compare", "status": "prediction_complete",
            "stage": "native64_fixed320_source_prediction", "seed": seed,
            "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
            "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
            "prediction_sha256": prediction_sha, "ground_truth_used_for_prediction": False,
            "source_checkpoint_sha256": source_sha, "source_manifest_sha256": manifest_sha,
            "source_training_ids_sha256": hashlib.sha256(np.asarray(train_ids, dtype="<i8").tobytes()).hexdigest(),
            "rgb_asset_validation": assets, "validation_rgb_cache_sha256": sha256_file(cache_path),
            "canonical_dataset": {"path": str(dataset.resolve()), "size": dataset_stat.st_size,
                "mtime_ns": dataset_stat.st_mtime_ns},
            "saved_native32_parity_sha256": parity_sha,
            "model_initialization_seed": MODEL_INIT_SEED, "inference_seed": INFERENCE_SEED,
            "readout": {"kind": "actual emitted-spike QCC", "synchrony_threshold": .50,
                        "min_group_size": 32, "background": "largest_component",
                        "affinity_mode": "spike"},
            "steps": STEPS, "settle": SETTLE, "live_tail_steps": 0,
            "model_state_sha256_before_after": model_before,
            "implementation_fingerprint": fingerprint,
            "work_ledger": {"directory": str(work_dir.resolve()),
                            "completed_image_records": COUNT, "sha256": [sha256_file(
                work_dir / f"image_{image_id}.npz") for image_id in IMAGE_IDS]}}
    output_npz, protocol_path, protocol = _publish_prediction(
        output_dir, np.asarray(IMAGE_IDS, dtype=np.int64), labels_all, protocol_builder)
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    args = parser.parse_args(argv)
    report = predict_seed(args.seed, device=args.device, dataset=args.dataset,
        prediction_root=args.prediction_root, work_root=args.work_root)
    print(json.dumps({"status": report["status"], "seed": args.seed,
        "count": COUNT, "ground_truth_used_for_prediction": False}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
