"""Fixed-budget SW0155 TRAIN-mask source continuation runner."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0152_source_preserving_partition_equivariance.deep_encoder import SourcePreservingDeepEncoder
from collaborative_test.SW_0155_supervised_spike_relations import objective

EXPERIMENT = "SW0155_supervised_spike_relations"
ARMS = ("encoder_graph", "joint")
SEED, CORE_INIT_SEED, RESIDUAL_INIT_SEED = 1, 149001, 152001
BATCH, MICRO, UPDATES, STEPS, SETTLE, TAIL = 16, 4, 128, 1024, 512, 64
HDF5_PATH = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
TRAIN_ROOT = HERE / "training"
WORK_ROOT = HERE / "work"
PREFLIGHT_PATH = HERE / "results_archive" / "preflight_seed1.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def implementation_fingerprint():
    paths = [HERE / name for name in ("protocol.json", "objective.py", "train.py", "preflight.py")]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/integration.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/spatial_diffusion.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/seed_selection.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/deep_encoder.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "snn_kuramoto_bidirectional/evaluation.py",
        "snn_kuramoto_bidirectional/spike_classifier.py"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SW0155 dependency: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def implementation_sha256():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def source_contract():
    checkpoint, manifest_path, manifest, rows, image_ids, source_sha = sw130.source_contract(SEED)
    rows = np.asarray(rows, dtype=np.int64)
    image_ids = [int(value) for value in image_ids]
    if (len(image_ids) != 4096 or len(set(image_ids)) != 4096 or rows.shape != (4096,)
            or len(np.unique(rows)) != 4096):
        raise ValueError("SW0155 requires the immutable ordered 4096-image source TRAIN contract")
    objective.validate_training_prefix(image_ids[:2048], image_ids)
    return {"checkpoint": Path(checkpoint), "manifest_path": Path(manifest_path),
            "manifest": manifest, "rows": rows, "image_ids": image_ids,
            "source_sha256": source_sha}


def _load_foundation(device):
    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.random.default_generator.manual_seed(CORE_INIT_SEED)
        torch.cuda.manual_seed(CORE_INIT_SEED)
        return sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)


def configure_trainability(foundation, arm):
    if arm not in ARMS:
        raise ValueError("unregistered SW0155 arm")
    if not isinstance(foundation.encoder, SourcePreservingDeepEncoder):
        raise ValueError("SW0155 residual encoder must be installed")
    encoder, core = foundation.encoder, foundation.wrapped.core
    encoder.requires_grad_(False)
    encoder.residual.requires_grad_(True)
    encoder.train()
    encoder.base_encoder.eval()
    foundation.wrapped.train()
    foundation.wrapped.requires_grad_(True)
    core.graph_generator.requires_grad_(True)
    native_core = [p for name, p in core.named_parameters()
                   if not name.startswith("graph_generator.")]
    integration = [foundation.wrapped.a_d, foundation.wrapped.a_m, foundation.wrapped.b]
    joint = arm == "joint"
    for parameter in native_core + integration:
        parameter.requires_grad_(joint)
    families = {"spatial_residual": list(encoder.residual.parameters()),
                "graph": list(core.graph_generator.parameters())}
    if joint:
        families.update(native_core=native_core, integration_tensors=integration)
    selected = [parameter for values in families.values() for parameter in values]
    if (not selected or len({id(parameter) for parameter in selected}) != len(selected)
            or any(not parameter.requires_grad for parameter in selected)):
        raise ValueError("SW0155 trainable parameter selection is invalid")
    if any(p.requires_grad for p in encoder.base_encoder.parameters()):
        raise AssertionError("the pretrained RGB base encoder must remain frozen")
    if not joint and any(p.requires_grad for p in native_core + integration):
        raise AssertionError("encoder_graph must freeze native core and integration tensors")
    return families


def make_optimizer(families):
    base = {"spatial_residual", "graph"}
    if set(families) == base:
        rows = (("spatial_residual", 3e-6), ("graph", 3e-5))
    elif set(families) == base | {"native_core", "integration_tensors"}:
        rows = (("spatial_residual", 3e-6), ("graph", 3e-5),
                ("native_core", 3e-5), ("integration_tensors", 1e-3))
    else:
        raise ValueError("optimizer groups do not match a registered SW0155 arm")
    return torch.optim.Adam([{"name": name, "params": families[name], "lr": rate}
                             for name, rate in rows])


def load_aligned_train_rows(h5_path, image_ids, rgb_cache_rows, start, count, *,
                            hdf5_handle=None, expected_rows=100000):
    """Read canonical TRAIN image/mask pairs by global ID and restore source order."""
    ids = [int(value) for value in image_ids[start:start + count]]
    cache = np.asarray(rgb_cache_rows[start:start + count]).copy()
    if len(ids) != count or cache.dtype != np.uint8 or cache.shape != (count, 128, 128, 3):
        raise ValueError("ordered RGB batch has an invalid source-cache schema")
    if any(1000 <= value <= 1639 or 90640 <= value <= 90959 for value in ids):
        raise ValueError("reserved validation/test IDs cannot enter the TRAIN mask loader")
    order = np.argsort(np.asarray(ids, dtype=np.int64))
    sorted_ids = np.asarray(ids, dtype=np.int64)[order]
    owns_handle = hdf5_handle is None
    handle = h5py.File(h5_path, "r") if owns_handle else hdf5_handle
    try:
        images, masks = handle["image"], handle["mask"]
        if images.shape != (expected_rows, 128, 128, 3) or masks.shape not in (
                (expected_rows, 128, 128), (expected_rows, 128, 128, 1)):
            raise ValueError("canonical supervised HDF5 schema changed")
        sorted_images, sorted_masks = np.asarray(images[sorted_ids]), np.asarray(masks[sorted_ids])
    finally:
        if owns_handle:
            handle.close()
    if sorted_masks.dtype.kind not in "ui" or np.any(sorted_masks < 0):
        raise ValueError("canonical TRAIN masks must contain nonnegative integer IDs")
    return restore_source_order(ids, sorted_ids, sorted_images, sorted_masks, cache)


def restore_source_order(ids, sorted_ids, sorted_images, sorted_masks, cache):
    """Restore sorted HDF5 rows to requested IDs and enforce image/cache identity."""
    ids = np.asarray(ids, dtype=np.int64)
    sorted_ids = np.asarray(sorted_ids, dtype=np.int64)
    order = np.argsort(ids)
    if (ids.ndim != 1 or sorted_ids.shape != ids.shape
            or not np.array_equal(sorted_ids, ids[order])
            or sorted_images.shape[0] != len(ids) or sorted_masks.shape[0] != len(ids)):
        raise ValueError("sorted HDF5 rows do not match the requested global IDs")
    inverse = np.argsort(order)
    aligned_images, aligned_masks = sorted_images[inverse], sorted_masks[inverse]
    if aligned_masks.ndim == 4:
        aligned_masks = aligned_masks[..., 0]
    if aligned_images.dtype == np.uint8:
        aligned_rgb = aligned_images
    elif aligned_images.dtype in (np.float16, np.float32):
        aligned_rgb = np.rint(np.clip(aligned_images.astype(np.float32), 0, 1) * 255).astype(np.uint8)
    else:
        raise ValueError("canonical TRAIN image dtype changed")
    if not np.array_equal(aligned_rgb, cache):
        raise ValueError("canonical HDF5 images do not match registered RGB cache/global IDs")
    return torch.from_numpy(np.ascontiguousarray(cache)), torch.from_numpy(np.ascontiguousarray(aligned_masks))


def logical_update(*, optimizer, family_map, compute_micro, wrapped, micro_count=4):
    if micro_count != 4:
        raise ValueError("SW0155 uses four microbatches per logical update")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for micro in range(micro_count):
        loss = compute_micro(micro)
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("SW0155 loss must be a finite scalar")
        (loss / micro_count).backward()
        losses.append(float(loss.detach().cpu()))
        del loss
    norms = {}
    for name, parameters in family_map.items():
        squared = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite SW0155 gradient in {name}")
                squared += float(parameter.grad.detach().double().square().sum().cpu())
        norms[name] = squared ** 0.5
    active = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not active or any(norms[name] <= 0 for name in family_map):
        raise FloatingPointError("SW0155 selected family lacks finite positive gradient")
    torch.nn.utils.clip_grad_norm_(active, 1.0)
    optimizer.step()
    with torch.no_grad():
        wrapped.project_integrations_()
    return {"mean_loss": sum(losses) / micro_count, "family_gradient_norms": norms}


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT_PATH,
              hdf5_path=HDF5_PATH, train_root=TRAIN_ROOT, work_root=WORK_ROOT):
    from collaborative_test.SW_0155_supervised_spike_relations import preflight as preflight_validator
    if arm not in ARMS:
        raise ValueError("unregistered SW0155 arm")
    preflight_path, hdf5_path = Path(preflight_path), Path(hdf5_path)
    if not preflight_path.is_file():
        raise FileNotFoundError("SW0155 training requires an actual-source preflight receipt")
    pre = json.loads(preflight_path.read_text(encoding="utf-8-sig"))
    src = source_contract()
    assets = sw130.validate_rgb_assets()
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    data_stat = hdf5_path.stat()
    if not preflight_validator.validate_admission(pre, source=src, assets=assets,
            hdf5_path=hdf5_path, hdf5_size=data_stat.st_size, hdf5_mtime_ns=data_stat.st_mtime_ns):
        raise ValueError("SW0155 preflight is not admitted or has stale provenance")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0155 training requires assigned CUDA")
    torch.cuda.set_device(dev)
    src, assets = source_contract(), sw130.validate_rgb_assets()
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source RGB cache schema changed")
    ids, rows = src["image_ids"][:2048], src["rows"][:2048]
    objective.validate_training_prefix(ids, src["image_ids"])
    dataset_stat = hdf5_path.stat()
    foundation = _load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != src["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha256_file(src["manifest_path"])
            or foundation.provenance.get("rgb_asset_validation") != assets
            or list(map(int, foundation.image_ids)) != src["image_ids"]
            or not np.array_equal(foundation.pool_indices, src["rows"])):
        raise ValueError("fresh source foundation differs from registered assets/order")
    rgb0 = torch.from_numpy(np.asarray(cache[rows[:MICRO]]).copy()).to(dev)
    source_result = forward_assignment(foundation, rgb0, live_tail_steps=TAIL)
    source_values = {key: source_result[key].detach().cpu() for key in ("q", "qcc", "gamma")}
    trace_names = ("theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates")
    source_values.update({key: source_result["trace"][key].detach().cpu() for key in trace_names})
    del source_result
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(RESIDUAL_INIT_SEED)
        foundation.encoder = SourcePreservingDeepEncoder(
            foundation.encoder, freeze_base=True).to(dev)
    student_result = forward_assignment(foundation, rgb0, live_tail_steps=TAIL)
    parity = all(torch.equal(source_values[key], student_result[key].detach().cpu())
                 for key in ("q", "qcc", "gamma"))
    parity &= all(torch.equal(source_values[key], student_result["trace"][key].detach().cpu())
                  for key in trace_names)
    del student_result, rgb0, source_values
    if not parity:
        raise AssertionError("SW0155 zero-residual source parity failed")
    families = configure_trainability(foundation, arm)
    optimizer = make_optimizer(families)
    initial_hashes = {name: source_training.parameter_hash(parameters)
                      for name, parameters in families.items()}
    frozen_base = source_training.state_sha(foundation.encoder.base_encoder)
    dataset_stat = hdf5_path.stat()
    bindings = {"experiment": EXPERIMENT, "arm": arm, "seed": SEED,
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": sha256_file(src["manifest_path"]),
        "source_training_ids_sha256": ids_sha(src["image_ids"]),
        "ordered_train_ids_sha256": ids_sha(ids), "rgb_asset_validation": assets,
        "preflight_sha256": sha256_file(preflight_path), "implementation_fingerprint": implementation_fingerprint(),
        "hdf5_path": str(hdf5_path.resolve()), "hdf5_size": dataset_stat.st_size,
        "hdf5_mtime_ns": dataset_stat.st_mtime_ns, "batch": BATCH, "microbatch": MICRO,
        "updates": UPDATES, "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
        "initial_parameter_sha256": initial_hashes, "ground_truth_used": True,
        "ground_truth_scope": "TRAIN masks only"}
    runtime = durable.runtime_signature(dev)
    work_dir = Path(work_root) / f"{arm}_seed1" / "checkpoints"
    output_dir = Path(train_root) / f"{arm}_seed1"
    if output_dir.exists():
        raise FileExistsError("preserving existing SW0155 completed training output")
    if work_dir.exists() and (work_dir / "boundary.ptx").is_file():
        metadata = durable.inspect_committed_metadata(work_dir)
        if metadata.get("bindings") != bindings or metadata.get("runtime") != runtime:
            raise ValueError("SW0155 checkpoint lineage/runtime mismatch")
        loaded = durable.load_latest(work_dir, wrapped=foundation.wrapped, encoder=foundation.encoder,
            optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
            device=dev, max_updates=UPDATES)
        completed, history = loaded["completed_updates"], loaded["history"]
    else:
        if work_dir.exists() and not durable.is_only_uncommitted_temp(work_dir):
            raise ValueError("preserving unknown/incomplete SW0155 work artifacts")
        work_dir.mkdir(parents=True, exist_ok=True)
        completed, history = 0, []
        durable.save_boundary(work_dir, completed_updates=0, wrapped=foundation.wrapped,
            encoder=foundation.encoder, optimizer=optimizer, history=history, bindings=bindings,
            runtime=runtime, device=dev)
    if completed > UPDATES or len(history) != completed:
        raise ValueError("SW0155 checkpoint update/history cursor is invalid")
    train_cache_rows = np.asarray(cache[rows]).copy()
    with h5py.File(hdf5_path, "r") as hdf5_handle:
        if hdf5_handle["image"].shape != (100000, 128, 128, 3):
            raise ValueError("canonical TRAIN HDF5 image dataset changed")
        if hdf5_handle["mask"].shape not in ((100000, 128, 128), (100000, 128, 128, 1)):
            raise ValueError("canonical TRAIN HDF5 mask dataset changed")
        for update in range(completed, UPDATES):
            batch_ids = ids[update * BATCH:(update + 1) * BATCH]
            def compute_micro(micro):
                start = update * BATCH + micro * MICRO
                rgb_cpu, masks_cpu = load_aligned_train_rows(
                    hdf5_path, ids, train_cache_rows, start, MICRO, hdf5_handle=hdf5_handle)
                labels = objective.modal_train_targets32(masks_cpu).to(dev)
                rgb = rgb_cpu.to(dev)
                gamma = foundation.encode(rgb)
                trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
                loss, _ = objective.supervised_relation_loss(trace, labels, settle=SETTLE)
                return loss
            row = logical_update(optimizer=optimizer, family_map=families,
                compute_micro=compute_micro, wrapped=foundation.wrapped)
            row.update(update=update + 1, image_ids=batch_ids)
            history.append(row)
            durable.save_boundary(work_dir, completed_updates=update + 1,
                wrapped=foundation.wrapped, encoder=foundation.encoder, optimizer=optimizer,
                history=history, bindings=bindings, runtime=runtime, device=dev)
            if (update + 1) % 16 == 0:
                print(json.dumps({"arm": arm, "completed_updates": update + 1,
                                  "total_updates": UPDATES}, allow_nan=False), flush=True)
    if len(history) != UPDATES or source_training.state_sha(foundation.encoder.base_encoder) != frozen_base:
        raise AssertionError("SW0155 completion/frozen-base validation failed")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    checkpoint, optimizer_path, history_path = staging / "checkpoint.pt", staging / "optimizer.pt", staging / "history.json"
    with checkpoint.open("xb") as stream:
        torch.save({"wrapped": foundation.wrapped.state_dict(), "encoder": foundation.encoder.state_dict()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    with optimizer_path.open("xb") as stream:
        torch.save(optimizer.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
    with history_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(history, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    final_hashes = {name: source_training.parameter_hash(parameters) for name, parameters in families.items()}
    manifest = {"status": "training_complete", "experiment": EXPERIMENT, "arm": arm,
        "seed": SEED, "completed_updates": UPDATES, "logical_batch": BATCH, "microbatch": MICRO,
        "exposures": BATCH * UPDATES, "train_ids": ids, "bindings": bindings,
        "initial_parameter_sha256": initial_hashes, "final_parameter_sha256": final_hashes,
        "frozen_base_encoder_sha256_before_after": frozen_base,
        "checkpoint_sha256": sha256_file(checkpoint), "optimizer_sha256": sha256_file(optimizer_path),
        "history_sha256": sha256_file(history_path),
        "implementation_fingerprint": bindings["implementation_fingerprint"],
        "optimizer_step_summary": {group["name"]: max(
            (int(optimizer.state[p]["step"].item()) for p in group["params"] if p in optimizer.state),
            default=0) for group in optimizer.param_groups},
        "ground_truth_used": True, "ground_truth_scope": "TRAIN masks only"}
    with (staging / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.rename(staging, output_dir)
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--hdf5", type=Path, default=HDF5_PATH)
    args = parser.parse_args(argv)
    result = train_arm(args.arm, device=args.device, preflight_path=args.preflight, hdf5_path=args.hdf5)
    print(json.dumps({"status": result["status"], "arm": args.arm,
                      "completed_updates": UPDATES, "ground_truth_used": True}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
