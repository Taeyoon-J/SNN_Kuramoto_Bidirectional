"""Controlled continuation on the existing 70k training pool."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
from snn_kuramoto_bidirectional.training.train_s2net_core import _component_spike_synchrony, _forward_with_plv
from SW_0102_cannot_link_draft.calibration_preflight import actual_101_negative_masks
from SW_0102_cannot_link_draft.forest_loss import cannot_link_hinge

ARMS = ("absolute_frozen", "positive_frozen", "positive_graph", "positive_joint")
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
ASSETS = Path("/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002")
SOURCE = ROOT / "trained_models/SW0090_unique70000_s0_e10/checkpoints/epoch_01.pt"
GAMMA = ROOT / "data/SW_0090_large_unique_scale/gamma_train_70000.pt"
VAL_GAMMA = ROOT / "data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hparams(gate_mode="raw"):
    return S2NetHyperparameters(
        num_feature_maps=8, num_regions=256, osc_dim=4,
        gamma_drive_mode="static", num_time_steps=64,
        gamma_phase_mode="standardize_tanh", theta_init="gamma",
        graph_mode="learned", graph_top_k=32, graph_spatial_decay=.35,
        geodesic_steps=3, geodesic_radius=1.5, geodesic_contrast=2.,
        geodesic_temperature=.5, geodesic_cap=16., kuramoto_backend="factorized",
        k=256., freq_gain=2., membrane_vth=.06, membrane_low_m=-4.,
        membrane_high_m=0., low_n=-4., high_n=0., branch=4,
        gate_mode=gate_mode, spike_per_component=True, dendritic_projection="shared",
        spike_spatial_grid_size=16,
    ).validate()


def aligned_affinity(core, settle=32):
    components = core.last_component_spikes
    # Reuse the prediction formula directly; no second definition can drift.
    return spike_synchrony_affinity(components.mean(dim=1), components, settle=settle)


def rgb_batch(dataset, indices):
    ids = np.where(indices < 1000, indices, indices + 640)
    order = np.argsort(ids)
    images = dataset["image"][ids[order].tolist()][np.argsort(order)]
    return torch.from_numpy(images.copy()).permute(0, 3, 1, 2)


def grad_norm(module):
    total = 0.
    for p in module.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError("nonfinite gradient")
            total += float(p.grad.detach().square().sum())
    return total ** .5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--source-seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--source-checkpoint", type=Path, default=None)
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--train-time-steps", type=int, default=64)
    p.add_argument("--train-settle", type=int, default=32)
    p.add_argument("--validation-count", type=int, choices=[80, 320], default=80)
    p.add_argument("--gate-mode", choices=["raw", "phasor_imag_raw"], default="raw")
    p.add_argument("--cannot-link-lambda", type=float, default=None)
    args = p.parse_args()
    if args.steps < 1 or args.batch < 1 or args.steps * args.batch > 70000:
        raise ValueError("pilot must use a positive, bounded without-replacement budget")
    if not 0 <= args.train_settle < args.train_time_steps:
        raise ValueError("training settle must be inside the temporal window")
    if args.cannot_link_lambda is not None:
        if (not np.isfinite(args.cannot_link_lambda) or args.cannot_link_lambda <= 0
                or args.arm != "positive_frozen" or args.gate_mode != "raw"
                or args.batch != 16 or args.train_time_steps != 64
                or args.train_settle != 32 or args.steps not in (1, 256)):
            raise ValueError("SW0102 cannot-link mode requires positive_frozen/raw, B16, 64/32, 1-step preflight or 256-step matched run")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    shuffle_seed = 17 + args.source_seed if args.shuffle_seed is None else args.shuffle_seed
    torch.manual_seed(shuffle_seed)
    source = args.source_checkpoint or ROOT / f"trained_models/SW0090_unique70000_s{args.source_seed}_e10/checkpoints/epoch_01.pt"
    hp = hparams(args.gate_mode)
    hp.num_time_steps = args.train_time_steps
    core = S2NetCore(hp.validate(), device=args.device).to(args.device)
    initial = torch.load(source, map_location=args.device, weights_only=True)
    core.load_state_dict(initial, strict=True)
    if args.arm.endswith("frozen"):
        core.graph_generator.requires_grad_(False)
    # Object-overlap loss is disabled; omit unused exponential clique detection.
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    core.train()
    gamma_cache = torch.load(GAMMA, map_location="cpu", weights_only=True, mmap=True)
    assert tuple(gamma_cache.shape) == (70000, 8, 256)
    stats = torch.load(ASSETS / "feature_preprocessing.pt", weights_only=True)
    mean, std = stats["mean"].to(args.device), stats["std"].to(args.device)
    clip = float(stats.get("clip", 3.))
    assert stats["mode"] == "standardize" and float(std) > 0
    encoder = load_input_encoder(str(ASSETS / "input_encoder/input_layer_encoder.pt"),
                                 num_kernels=8, kernel_size=3, channels=3, device=args.device)
    encoder.train(args.arm == "positive_joint")
    encoder.requires_grad_(args.arm == "positive_joint")
    initial_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    patcher = FeaturePatchGammaInitializer(grid_size=16).to(args.device)

    def encode(images):
        features = encoder(images.to(args.device).float() / 255.)
        return patcher(((features - mean) / std).clamp(-clip, clip))

    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0., spike_smooth_weight=0., spike_diversity_weight=0.,
        structural_weight=0., plv_bimodality_weight=6., plv_balance_weight=10.,
        plv_coherence_weight=.5, plv_collapse_weight=1., plv_target_density=.867,
        patch_grid_size=(16, 16))
    params = [p for p in core.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": 3e-5}]
    if args.arm == "positive_joint":
        groups.append({"params": encoder.parameters(), "lr": 3e-6})
    optimizer = torch.optim.Adam(groups)
    indices = torch.randperm(70000, generator=torch.Generator().manual_seed(shuffle_seed))[:args.steps * args.batch]
    manifest = {"arm": args.arm, "status": "training", "pilot": True,
                "training_pool": 70000, "unique_images_seen": len(indices),
                "training_ids": [int(i) if i < 1000 else int(i) + 640 for i in indices],
                "source": str(source), "source_sha256": sha(source),
                "seed": shuffle_seed, "source_model_seed": args.source_seed, "steps": args.steps,
                "batch": args.batch, "core_lr": 3e-5, "encoder_lr": 3e-6,
                "train_steps": args.train_time_steps, "train_settle": args.train_settle, "started": time.time(),
                "gate_mode": args.gate_mode,
                "cannot_link_lambda": args.cannot_link_lambda,
                "cannot_link_mask_spec": "SW0101 frozen definition; per-current-image masks constructed GT-free" if args.cannot_link_lambda is not None else None,
                "ground_truth_used_for_training": False,
                "preflight": args.preflight, "device": args.device}
    assert not set(manifest["training_ids"]).intersection(range(1000, 1640))
    write(args.output / "manifest.json", manifest)
    history = []
    with h5py.File(DATASET, "r") as dataset:
        for step in range(args.steps):
            ix = indices[step * args.batch:(step + 1) * args.batch]
            cached = gamma_cache[ix].to(args.device)
            if step == 0 or args.arm == "positive_joint":
                rgb = rgb_batch(dataset, ix.numpy())
                if step == 0:
                    with torch.no_grad():
                        diff = float((encode(rgb) - cached).abs().max())
                    if diff > 2e-5:
                        raise AssertionError(f"RGB/cached gamma mismatch: {diff}")
                    manifest["initial_encoder_gamma_max_diff"] = diff
            gamma = encode(rgb) if args.arm == "positive_joint" else cached
            projected, adjacency = [], []
            graph_hook = projection_hook = None
            if args.cannot_link_lambda is not None:
                graph_hook = core.graph_generator.register_forward_hook(
                    lambda _module, _inputs, output: adjacency.append(output))
                projection_hook = core.graph_generator.projection.register_forward_hook(
                    lambda _module, _inputs, output: projected.append(output))
            try:
                _, spikes, out, plv, theta = _forward_with_plv(core, gamma, criterion, args.train_settle, "phase", "mean")
            finally:
                if graph_hook is not None:
                    graph_hook.remove()
                if projection_hook is not None:
                    projection_hook.remove()
            assert core.last_component_spikes.shape[-1] == args.train_time_steps
            primary, _ = criterion(plv=plv, theta=theta)
            absolute = _component_spike_synchrony(core, args.train_settle)
            positive = aligned_affinity(core, settle=args.train_settle)
            affinity = absolute if args.arm == "absolute_frozen" else positive
            spike_loss, _ = criterion(plv=affinity)
            loss = primary + 5. * spike_loss
            cut_loss = None
            cut_counts = None
            if args.cannot_link_lambda is not None:
                if len(adjacency) != 1 or len(projected) != 1:
                    raise AssertionError("cannot-link mode expects one graph/projection call")
                negative_masks, _ = actual_101_negative_masks(adjacency[0], projected[0])
                cut_loss, cut_counts = cannot_link_hinge(positive, negative_masks)
                if not torch.isfinite(cut_loss):
                    raise FloatingPointError("nonfinite cannot-link cut loss")
                loss = loss + args.cannot_link_lambda * cut_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms = {"core": grad_norm(core), "graph": grad_norm(core.graph_generator),
                     "encoder": grad_norm(encoder)}
            if norms["core"] == 0 or (args.arm == "positive_joint" and norms["encoder"] == 0):
                raise AssertionError(f"missing gradient: {norms}")
            if args.arm in ("positive_graph", "positive_joint") and norms["graph"] == 0:
                raise AssertionError("no graph gradient")
            torch.nn.utils.clip_grad_norm_(params + list(encoder.parameters()), 1.)
            optimizer.step()
            history.append({"step": step + 1, "loss": float(loss.detach()),
                            "primary": float(primary.detach()), "spike": float(spike_loss.detach()),
                            "cannot_link_cut_loss": float(cut_loss.detach()) if cut_loss is not None else None,
                            "cannot_link_counts": cut_counts,
                            "gradient_norm": norms,
                            "absolute_positive_edge_disagreement": float(((absolute >= .5) != (positive >= .5)).float().mean())})
    changed = {k: not torch.equal(v.detach(), initial[k]) for k, v in core.state_dict().items()}
    if args.arm.endswith("frozen") and any(v for k, v in changed.items() if k.startswith("graph_generator.")):
        raise AssertionError("frozen graph changed")
    if not any(v for k, v in changed.items() if not k.startswith("graph_generator.")):
        raise AssertionError("downstream core did not update")
    if args.arm.endswith("frozen") and any(
            not torch.equal(v.detach(), initial_encoder[k])
            for k, v in encoder.state_dict().items()):
        raise AssertionError("frozen encoder changed")
    torch.save(core.state_dict(), args.output / "core.pt")
    torch.save(encoder.state_dict(), args.output / "encoder.pt")
    torch.save(stats, args.output / "feature_preprocessing.pt")
    write(args.output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(),
                    changed_core_keys=[k for k, v in changed.items() if v],
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(args.device).startswith("cuda") else None)
    write(args.output / "manifest.json", manifest)
    if args.preflight:
        (args.output / "PREFLIGHT_COMPLETED").write_text("finite real-data backward and update\n")
        return
    gamma_path = VAL_GAMMA
    gamma_manifest = ROOT / "data/SW_0042_hdf5_aligned/manifest.json"
    if args.arm == "positive_joint":
        encoder.eval()
        with h5py.File(DATASET, "r") as dataset, torch.no_grad():
            validation = []
            for start in range(1320, 1640, 16):
                rgb = torch.from_numpy(dataset["image"][start:start + 16]).permute(0, 3, 1, 2)
                validation.append(encode(rgb).cpu())
        gamma_path = args.output / "gamma_validation.pt"
        torch.save(torch.cat(validation), gamma_path)
        gamma_manifest = args.output / "gamma_manifest.json"
        write(gamma_manifest, {"dataset": str(DATASET), "image_ids": [1320, 1639],
                              "encoder": str(args.output / "encoder.pt"),
                              "encoder_sha256": sha(args.output / "encoder.pt"),
                              "gamma_sha256": sha(gamma_path), "shape": [320, 8, 256]})
    command = [sys.executable, str(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
               "--checkpoint", str(args.output / "core.pt"), "--gamma-path", str(gamma_path),
               "--gamma-global-start", "1320", "--gamma-manifest", str(gamma_manifest),
               "--dataset-path", str(DATASET), "--output-path", str(args.output / "evaluation.json"),
               "--start", "1320", "--count", str(args.validation_count), "--steps", "1024", "--settle", "512",
               "--membrane-vth", ".06", "--min-group-size", "2", "--background", "largest_component",
               "--thresholds", ".50", "--dendritic-projection", "shared", "--graph-spatial-decay", ".35",
               "--geodesic-steps", "3", "--geodesic-radius", "1.5", "--geodesic-contrast", "2",
               "--geodesic-temperature", ".5", "--geodesic-cap", "16", "--kuramoto-backend", "factorized",
               "--gate-mode", args.gate_mode, "--device", args.device]
    with (args.output / "evaluation.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest.update(status="complete", evaluation_images=args.validation_count,
                    evaluation_ids=[1320, 1320 + args.validation_count - 1])
    write(args.output / "manifest.json", manifest)
    (args.output / "COMPLETED").write_text("training and pilot evaluation complete\n")


if __name__ == "__main__":
    main()
