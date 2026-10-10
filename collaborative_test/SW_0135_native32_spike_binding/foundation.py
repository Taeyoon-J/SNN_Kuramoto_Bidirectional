"""Strict SW0097 source loading and native128-RGB to native32 gamma foundation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import resolution
from collaborative_test.SW_0135_native32_spike_binding.binder import (
    NativeSpikeSlotBinder, RelativeSlotRGBDecoder, reconstruct_from_spikes,
)
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss, phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder

GRID = 32
PATCHES = GRID * GRID
FEATURES = 8
IMAGE = 128


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    """Bind foundation and every registered implementation used to construct it."""
    paths = (
        HERE / "foundation.py", HERE / "protocol.json", HERE / "resolution.py",
        HERE / "binder.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
        ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
        ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
        ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
        ROOT / "snn_kuramoto_bidirectional/loss_function.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "snn_kuramoto_bidirectional/hyperparameter.py",
        ROOT / "snn_kuramoto_bidirectional/gamma_initializer.py",
        ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py",
        ROOT / "snn_kuramoto_bidirectional/training/train_input_layer_generator.py",
        ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
        ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
        ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
        ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
    )
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"foundation implementation dependency missing: {missing[0]}")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def encode_native32(encoder, patcher, mean, std, clip, rgb_uint8):
    """Encode NHWC uint8 native128 RGB once into [B,8,1024] gamma.

    This function intentionally accepts the registered uint8 cache representation,
    making the single /255 conversion explicit and preventing reuse of a 16-grid cache.
    Gradients remain live through the encoder and feature normalization.
    """
    if not torch.is_tensor(rgb_uint8):
        rgb_uint8 = torch.as_tensor(rgb_uint8)
    if (rgb_uint8.dtype != torch.uint8 or rgb_uint8.ndim != 4
            or tuple(rgb_uint8.shape[1:]) != (IMAGE, IMAGE, 3)):
        raise ValueError("native32 foundation requires uint8 RGB [B,128,128,3]")
    if not np.isfinite(float(clip)) or float(clip) <= 0:
        raise ValueError("registered feature clipping value must be finite and positive")
    device = next(encoder.parameters()).device
    images = rgb_uint8.to(device=device).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    feature_maps = encoder(images)
    if (feature_maps.ndim != 4 or feature_maps.shape[0] != rgb_uint8.shape[0]
            or feature_maps.shape[1] != FEATURES):
        raise ValueError("registered encoder must return [B,8,H,W] feature maps")
    mean = torch.as_tensor(mean, device=device, dtype=feature_maps.dtype)
    std = torch.as_tensor(std, device=device, dtype=feature_maps.dtype)
    if mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("registered feature normalization statistics are invalid")
    try:
        normalized = ((feature_maps - mean) / std).clamp(-float(clip), float(clip))
    except RuntimeError as exc:
        raise ValueError("registered feature statistics do not broadcast over [B,8,H,W]") from exc
    gamma = patcher(normalized)
    if tuple(gamma.shape) != (rgb_uint8.shape[0], FEATURES, PATCHES):
        raise AssertionError(f"native32 gamma shape mismatch: {tuple(gamma.shape)}")
    if not torch.isfinite(gamma).all():
        raise FloatingPointError("native32 gamma contains nonfinite values")
    return gamma


def make_criterion32():
    """SW0130 phase-primary coefficients with prospectively registered coherence1.0."""
    return UnsupervisedS2NetLoss(
        spike_rate_weight=0.0, spike_smooth_weight=0.0,
        spike_diversity_weight=0.0, structural_weight=0.0,
        plv_bimodality_weight=6.0, plv_balance_weight=10.0,
        plv_coherence_weight=1.0, plv_collapse_weight=1.0,
        plv_target_density=0.867, patch_grid_size=(32, 32))


def wrap_mapped_source(source_state, device="cpu"):
    """Map and strict-load source state before adding SW0130 integration scalars."""
    core = resolution.build_core(source_state, device=device)
    wrapped = PhaseStateIntegration(core, "phase")
    if (core.in_dim != PATCHES or core.spike_spatial_grid_size != GRID
            or core.graph_generator is None or core.graph_generator.top_k != 128
            or not any(parameter.requires_grad for parameter in core.graph_generator.parameters())):
        raise AssertionError("mapped source does not satisfy the trainable native32 core contract")
    return wrapped


@dataclass
class Native32Foundation:
    seed: int
    wrapped: PhaseStateIntegration
    encoder: torch.nn.Module
    patcher: FeaturePatchGammaInitializer
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_clip: float
    pool_indices: np.ndarray
    image_ids: list[int]
    provenance: dict

    def encode(self, rgb_uint8):
        return encode_native32(self.encoder, self.patcher, self.feature_mean,
                               self.feature_std, self.feature_clip, rgb_uint8)

    def rollout(self, gamma, *, total_steps=1024, live_tail_steps=64):
        return late_rollout(self.wrapped, gamma, total_steps=total_steps,
                            live_tail_steps=live_tail_steps)


def load_native32_foundation(seed: int, device="cpu", *, verify_rgb_assets=True) -> Native32Foundation:
    """Load a same-seed immutable SW0097 checkpoint, then map and wrap native32."""
    seed = int(seed)
    if seed not in sw130.SEEDS:
        raise ValueError(f"seed must be one of {sw130.SEEDS}")
    device = torch.device(device)
    checkpoint, manifest_path, manifest, pool, ids, source_sha = sw130.source_contract(seed)
    asset_record = sw130.validate_rgb_assets() if verify_rgb_assets else None
    source_state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(source_state, dict):
        raise TypeError("registered SW0097 core checkpoint must be a state dictionary")
    # Strict source load and 16->32 mapping occur before SW0130 parameters.
    wrapped = wrap_mapped_source(source_state, device=device)
    core = wrapped.core
    encoder = load_input_encoder(
        str(sw130.ENCODER_SOURCE), num_kernels=8, kernel_size=3, channels=3,
        device=device)
    encoder.requires_grad_(True)
    encoder.train()
    stats = torch.load(sw130.FEATURE_STATS, map_location=device, weights_only=True)
    if stats.get("mode") != "standardize":
        raise AssertionError("registered encoder preprocessing mode must be standardize")
    mean = torch.as_tensor(stats["mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(stats["std"], device=device, dtype=torch.float32)
    clip = float(stats.get("clip", 3.0))
    if not np.isfinite(clip) or clip <= 0:
        raise AssertionError("registered feature clipping value must be finite and positive")
    patcher = FeaturePatchGammaInitializer(grid_size=GRID).to(device)
    if (core.in_dim != PATCHES or core.spike_spatial_grid_size != GRID
            or patcher.grid_size != (GRID, GRID) or core.graph_generator.top_k != 128):
        raise AssertionError("constructed core is not the registered native32 model")
    if core.graph_generator is None or not any(p.requires_grad for p in core.graph_generator.parameters()):
        raise AssertionError("native32 learned graph must remain trainable")

    provenance = {
        "experiment": "SW0135_native32_spike_binding",
        "status": "foundation_loaded_no_optimizer_updates",
        "seed": seed,
        "source_core_path": str(checkpoint),
        "source_manifest_path": str(manifest_path),
        "source_core_sha256": source_sha,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_steps": int(manifest["steps"]),
        "source_training_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "encoder_sha256": sha256_file(sw130.ENCODER_SOURCE),
        "feature_preprocessing_sha256": sha256_file(sw130.FEATURE_STATS),
        "rgb_asset_validation": asset_record,
        "implementation_fingerprint": implementation_fingerprint(),
        "native32": {"grid": [GRID, GRID], "patches": PATCHES,
                     "gamma_shape": [None, FEATURES, PATCHES],
                     "source_mapping": "SW0135 strict spatial2x2 map from same-seed SW0097"},
        "optimizer_updates": 0,
        "ground_truth_used": False,
    }
    return Native32Foundation(seed, wrapped, encoder, patcher, mean, std, clip,
                              np.asarray(pool, dtype=np.int64), ids, provenance)


def load_models(seed, device="cpu", *, verify_rgb_assets=True):
    """Return the SW0134-compatible tuple with native32 foundation modules."""
    foundation = load_native32_foundation(seed, device, verify_rgb_assets=verify_rgb_assets)
    binder = NativeSpikeSlotBinder(seed=135).to(device)
    decoder = RelativeSlotRGBDecoder(seed=106).to(device)
    return (foundation.wrapped, foundation.encoder, foundation.patcher,
            foundation.feature_mean, foundation.feature_std, foundation.feature_clip,
            binder, decoder, foundation.pool_indices, foundation.image_ids,
            foundation.provenance["source_core_sha256"])


def forward_batch(wrapped, encoder, patcher, mean, std, clip, binder, decoder,
                  rgb_uint8, *, criterion=None, total_steps=1024, settle=512,
                  live_tail_steps=64):
    """Native32 gamma, late rollout, binder RGB output, and no-GT objective parts."""
    if total_steps != 1024 or settle != 512:
        raise ValueError("SW0135 forward is registered for T1024 and settle512")
    gamma = encode_native32(encoder, patcher, mean, std, clip, rgb_uint8)
    trace = late_rollout(wrapped, gamma, total_steps=total_steps,
                         live_tail_steps=live_tail_steps)
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                 settle=settle, affinity_mode="spike")
    criterion = make_criterion32() if criterion is None else criterion
    theta_band = trace["theta"][:, settle:]
    phase_plv = phase_locking_value(theta_band, combine="mean")
    phase, phase_parts = criterion(plv=phase_plv, theta=theta_band)
    positive_q, q_parts = criterion(plv=q)
    old = phase + 5.0 * positive_q
    prediction, assignment, slots, patch_features, labels = reconstruct_from_spikes(
        trace["component_spikes"][..., settle:], binder, decoder)
    target = rgb_uint8.to(device=prediction.device, dtype=prediction.dtype) / 255.0
    rgb = (prediction - target).square().mean()
    return {
        "old": old,
        "old_parts": {"phase": phase_parts, "q": q_parts},
        "rgb": rgb,
        "trace": trace,
        "prediction": prediction,
        "assignment": assignment,
        "slots": slots,
        "patch_features": patch_features,
        "labels": labels,
        "gamma": gamma,
        "q": q,
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=sw130.SEEDS, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-rgb-assets", action="store_true")
    args = parser.parse_args(argv)
    foundation = load_native32_foundation(args.seed, args.device,
                                          verify_rgb_assets=args.verify_rgb_assets)
    print(json.dumps(foundation.provenance, indent=2, allow_nan=False))
    return foundation


if __name__ == "__main__":
    main()
