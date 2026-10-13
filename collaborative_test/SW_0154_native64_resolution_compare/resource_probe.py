"""One-image, read-only native4096 source resource probe; never fans out validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PATCH_V2_REPO", HERE.parents[1])).resolve()
if not (ROOT / "collaborative_test").is_dir():
    raise RuntimeError(f"PATCH_V2_REPO is not the patch_v2_sw repository: {ROOT}")
for item in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import numpy as np
import torch
try:
    import resource
except ImportError:  # Windows supports importing this module for CPU contract tests.
    resource = None

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0135_native32_spike_binding.resolution import attach_checkpointed_geodesic
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150_preflight
from collaborative_test.SW_0154_native64_resolution_compare.native64_model import (
    GRID, NODES, configure_grid_hparams, convert_state_dict16_to64,
)
from SW_0094_aligned_joint_pilot.run import hparams
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity, spike_synchrony_components,
)
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder

SEED = 1
TOTAL_STEPS = 1024
SETTLE = 512
GEODESIC_CHUNK = 1
REPORT_NAME = "source1_first_train_native64_resource.json"
TRACE_FIELDS = ("component_spikes", "component_membrane", "component_gates",
                "spikes", "membrane", "theta")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    files = dict(sw135.implementation_fingerprint())
    extra = (
        HERE / "resolution64.py", HERE / "native64_model.py", HERE / "resource_probe.py", HERE / "design.json",
        HERE / "score_slot64.py", ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/parity.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/preflight.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
        ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    for path in extra:
        if not path.is_file():
            raise FileNotFoundError(f"native64 probe dependency missing: {path}")
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return files


def _build_native64(seed, device, source_state):
    hp = hparams("raw")
    configure_grid_hparams(hp, grid_size=GRID, graph_top_k=512)
    core = S2NetCore(hp, device=str(device)).to(device)
    mapped = convert_state_dict16_to64(source_state, core.state_dict())
    core.load_state_dict(mapped, strict=True)
    if core.graph_generator is None or core.graph_generator.top_k != 512:
        raise AssertionError("native64 source graph must use the frozen 512-neighbor setting")
    attach_checkpointed_geodesic(core.graph_generator, chunk=GEODESIC_CHUNK, log4=True)
    wrapped = PhaseStateIntegration(core, "phase").to(device)
    encoder = load_input_encoder(str(sw130.ENCODER_SOURCE), num_kernels=8,
                                 kernel_size=3, channels=3, device=device)
    encoder.requires_grad_(False)
    encoder.train()  # match the registered SW0135 source foundation's encoder mode
    stats = torch.load(sw130.FEATURE_STATS, map_location=device, weights_only=True)
    if stats.get("mode") != "standardize":
        raise ValueError("source feature statistics must use registered standardization")
    mean = torch.as_tensor(stats["mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(stats["std"], device=device, dtype=torch.float32)
    clip = float(stats.get("clip", 3.0))
    patcher = FeaturePatchGammaInitializer(grid_size=(GRID, GRID)).to(device)
    core.requires_grad_(False)
    wrapped.requires_grad_(False)
    wrapped.eval()
    return wrapped, encoder, patcher, mean, std, clip


def _encode_once(encoder, patcher, mean, std, clip, rgb_uint8, device):
    if rgb_uint8.dtype != np.uint8 or rgb_uint8.shape != (128, 128, 3):
        raise ValueError("source probe input must be one registered uint8 [128,128,3] TRAIN image")
    rgb = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(device)
    image = rgb.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    prior = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    encoder.train()
    with torch.no_grad():
        feature = encoder(image)
        gamma = patcher(((feature - mean) / std).clamp(-clip, clip))
    # BatchNorm may update training buffers. Restore them so the audit has no
    # lasting in-memory side effect while keeping the registered source mode.
    encoder.load_state_dict(prior, strict=True)
    if tuple(gamma.shape) != (1, 8, NODES) or not torch.isfinite(gamma).all():
        raise FloatingPointError("native64 adaptive-pool gamma must be finite [1,8,4096]")
    return gamma.detach(), rgb


def _state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _trace_shapes_and_finiteness(trace):
    expected = {
        "component_spikes": (1, 4, NODES, TOTAL_STEPS),
        "component_membrane": (1, 4, NODES, TOTAL_STEPS),
        "component_gates": (1, 4, NODES, TOTAL_STEPS),
        "spikes": (1, NODES, TOTAL_STEPS),
        "membrane": (1, NODES, TOTAL_STEPS),
        "theta": (1, TOTAL_STEPS, NODES, 4),
    }
    records = {}
    for key in TRACE_FIELDS:
        tensor = trace[key]
        if tuple(tensor.shape) != expected[key] or not torch.isfinite(tensor).all():
            raise FloatingPointError(f"native64 {key} shape/finiteness mismatch: {tuple(tensor.shape)}")
        records[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "finite": True}
    return records


def audit_source(*, output: Path, device="cuda:0"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing native64 resource output: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native64 resource probe requires one explicitly leased CUDA device")
    torch.cuda.set_device(dev)
    source_path, manifest_path, source_manifest, pool, ids, source_sha = sw130.source_contract(SEED)
    parity_path = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/native32_zero_adapter_parity_seed1_b4.json"
    if not parity_path.is_file():
        raise FileNotFoundError("the registered seed1 native32 parity record is required before native64 resource work")
    rgb_assets = sw130.validate_rgb_assets()
    parity, parity_sha = sw150_preflight.load_and_validate_parity(
        parity_path, source_sha=source_sha, source_manifest_sha=sha256_file(manifest_path),
        training_ids=ids, asset_validation=rgb_assets)
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if rgb_cache.dtype != np.uint8 or tuple(rgb_cache.shape) != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    row, image_id = int(pool[0]), int(ids[0])
    if not 0 <= row < 70000 or image_id in range(1320, 1640):
        raise ValueError("native64 resource probe must use a non-validation TRAIN row")
    image = np.asarray(rgb_cache[row]).copy()
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError("first registered TRAIN RGB row is not uint8 128x128")
    rgb_sample_sha = hashlib.sha256(image.tobytes()).hexdigest()
    source_state = torch.load(source_path, map_location="cpu", weights_only=True)
    wrapped, encoder, patcher, mean, std, clip = _build_native64(SEED, dev, source_state)
    model_before = {"wrapped": _state_digest(wrapped), "encoder": _state_digest(encoder)}
    gamma, rgb_t = _encode_once(encoder, patcher, mean, std, clip, image, dev)
    rng_devices = [dev.index if dev.index is not None else torch.cuda.current_device()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(rng_devices[0]).clone()
    trace = q = labels = rgb_t = None
    graph_seconds = rollout_seconds = qcc_seconds = None
    trace_records = None
    try:
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(154001)
            torch.cuda.manual_seed(154001)
            torch.cuda.synchronize(dev)
            allocated_after_load = int(torch.cuda.memory_allocated(dev))
            reserved_after_load = int(torch.cuda.memory_reserved(dev))
            torch.cuda.reset_peak_memory_stats(dev)
            start = time.perf_counter()
            with torch.no_grad():
                _ = wrapped.core.graph_generator(gamma)
            torch.cuda.synchronize(dev)
            graph_seconds = time.perf_counter() - start
            del _
            torch.cuda.empty_cache()
            start = time.perf_counter()
            with torch.no_grad():
                trace = late_rollout(wrapped, gamma, total_steps=TOTAL_STEPS, live_tail_steps=0)
            torch.cuda.synchronize(dev)
            rollout_seconds = time.perf_counter() - start
            trace_records = _trace_shapes_and_finiteness(trace)
            start = time.perf_counter()
            with torch.no_grad():
                q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                             settle=SETTLE, affinity_mode="spike")
                groups = spike_synchrony_components(
                    trace["spikes"].detach().cpu(), synchrony_threshold=.50,
                    min_group_size=32, settle=SETTLE,
                    components=trace["component_spikes"].detach().cpu(),
                    background="largest_component", affinity_mode="spike",
                    spatial_grid_size=GRID)
                labels = spatial_components_to_patch_labels(groups, GRID, device="cpu")
            torch.cuda.synchronize(dev)
            qcc_seconds = time.perf_counter() - start
            if tuple(q.shape) != (1, NODES, NODES) or not torch.isfinite(q).all():
                raise FloatingPointError("actual-spike affinity did not emit finite native4096 Q")
            if tuple(labels.shape) != (1, GRID, GRID):
                raise ValueError("production QCC did not emit native64 labels")
            q_offdiag = q[0][~torch.eye(NODES, dtype=torch.bool, device=dev)]
            if not torch.isfinite(q_offdiag).all():
                raise FloatingPointError("Q off-diagonal contains nonfinite values")
            report = {
                "status": "complete", "experiment": "SW0154_native64_resolution_compare",
                "stage": "source_only_resource_probe", "seed": SEED,
                "source_image_id": image_id, "source_pool_row": row,
                "source_training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                "source_core_sha256": source_sha,
                "native32_parity_path": str(parity_path),
                "native32_parity_sha256": parity_sha,
                "source_manifest_sha256": sha256_file(manifest_path),
                "source_manifest_status": source_manifest["status"],
                "rgb_asset_validation": rgb_assets, "sample_rgb_sha256": rgb_sample_sha,
                "model_state_sha256_before": model_before,
                "model_state_sha256_after": {"wrapped": _state_digest(wrapped),
                                              "encoder": _state_digest(encoder)},
                "source_model_seed": SEED, "native_grid": [GRID, GRID], "node_count": NODES,
                "gamma_shape": list(gamma.shape), "gamma_finite": bool(torch.isfinite(gamma).all().item()),
                "graph_settings": {"top_k": 512, "k": 4096.0,
                    "geodesic_steps": wrapped.core.graph_generator.geodesic_steps,
                    "geodesic_radius": wrapped.core.graph_generator.geodesic_radius,
                    "geodesic_temperature": wrapped.core.graph_generator.geodesic_temperature,
                    "geodesic_cap": wrapped.core.graph_generator.geodesic_cap,
                    "distance_units_per_cell": 0.25, "full_intermediate_axis": True,
                    "row_chunk": GEODESIC_CHUNK, "log4_correction": True},
                "rollout": {"total_steps": TOTAL_STEPS, "settle": SETTLE,
                    "live_tail_steps": 0, "batch_size": 1, "optimizer_updates": 0,
                    "trace_fields": trace_records},
                "readout": {"q": "actual emitted component-spike affinity",
                    "q_shape": list(q.shape), "q_finite": True,
                    "q_offdiag_mean": float(q_offdiag.mean().item()),
                    "q_offdiag_std": float(q_offdiag.std(unbiased=False).item()),
                    "qcc_threshold": 0.50, "qcc_min_group": 32,
                    "qcc_background": "largest connected component", "labels_shape": list(labels.shape),
                    "label_count": int(torch.unique(labels).numel())},
                "resource": {"graph_seconds": graph_seconds, "rollout_seconds": rollout_seconds,
                    "q_qcc_seconds": qcc_seconds,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
                    "allocated_after_model_load_bytes": allocated_after_load,
                    "reserved_after_model_load_bytes": reserved_after_load,
                    "host_maxrss": (int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                                    if resource is not None else None),
                    "dense_float32_square_bytes": NODES * NODES * 4,
                    "row_chunk_candidate_intermediate_bytes": NODES * NODES * 4},
                "implementation_fingerprint": implementation_fingerprint(),
                "ground_truth_used": False, "masks_read": False,
                "training_admission": False,
                "interpretation_limit": "One source TRAIN image resource/finite-state measurement only; no full320 fanout, model-quality claim, training admission, or native64 scaling claim.",
            }
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, rng_devices[0])
        del trace, q, labels, gamma, rgb_t, source_state, rgb_cache
        torch.cuda.empty_cache()
    if report["model_state_sha256_before"] != report["model_state_sha256_after"]:
        raise AssertionError("read-only probe changed source model/encoder parameters or buffers")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_source(output=args.output, device=args.device)
    print(json.dumps({"status": result["status"], "stage": result["stage"],
                      "source_image_id": result["source_image_id"],
                      "training_admission": False, "output": str(args.output)},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
