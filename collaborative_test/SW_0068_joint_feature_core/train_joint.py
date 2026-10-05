#!/usr/bin/env python3
"""Jointly tune the native RGB feature generator and the S2Net core."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]

from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
from snn_kuramoto_bidirectional.loss_function import (
    UnsupervisedS2NetLoss,
    patch_pool_rgb,
)
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.train_s2net_core import (
    _component_spike_synchrony,
    _forward_with_plv,
    _select_loss_signal,
    load_graph_checkpoint,
    reset_graph_initialization,
    save_s2net_core,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_ids(count):
    if count < 1 or count > 2500:
        raise ValueError("count must lie in [1, 2500]")
    full = list(range(1000)) + list(range(1640, 3140))
    return full[:count]


def build_hparams():
    return S2NetHyperparameters(
        num_feature_maps=8, num_regions=256, osc_dim=4,
        gamma_drive_mode="static", num_time_steps=64,
        gamma_phase_mode="standardize_tanh", theta_init="gamma",
        theta_init_noise=0.0, graph_mode="learned", graph_top_k=32,
        graph_spatial_decay=0.35, geodesic_steps=3, geodesic_radius=1.5,
        geodesic_contrast=2.0, geodesic_temperature=0.5,
        geodesic_cap=16.0, kuramoto_backend="factorized", k=256.0,
        freq_gain=2.0, membrane_vth=0.06, membrane_low_m=-4.0,
        membrane_high_m=0.0, low_n=-4.0, high_n=0.0, branch=4,
        gate_mode="raw", spike_per_component=True,
        dendritic_projection="shared", spike_spatial_grid_size=16,
    ).validate()


def load_encoder(path, device):
    encoder = CNNFeatureEncoder(8, 3, in_channels=3, bias=True).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    encoder.load_state_dict(state, strict=True)
    return encoder


def gamma_from_images(encoder, patcher, images, mean, std, clip):
    features = encoder(images.float().div(255.0))
    normalized = ((features - mean) / std).clamp(-clip, clip)
    return patcher(normalized)


def compute_graph_output_anchor_loss(core, gamma, anchor_gamma):
    """Match current image-conditioned adjacency to the frozen anchor-gamma adjacency."""
    if getattr(core, "graph_generator", None) is None:
        raise ValueError("graph-output anchor requires a learned graph generator")
    with torch.no_grad():
        reference = core.graph_generator(anchor_gamma)
    current = core.graph_generator(gamma)
    if current.shape != reference.shape:
        raise ValueError(f"graph output shape mismatch: {tuple(current.shape)} vs {tuple(reference.shape)}")
    return F.mse_loss(current, reference)


def validate_graph_output_anchor(weight, freeze_core):
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("graph-output anchor weight must be finite and nonnegative")
    if weight > 0 and not freeze_core:
        raise ValueError("graph-output anchor requires a frozen core")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--anchor-gamma", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--graph-init-seed", type=int, default=0)
    p.add_argument("--graph-checkpoint", default=None)
    p.add_argument("--freeze-graph", action="store_true")
    p.add_argument("--core-checkpoint", default=None)
    p.add_argument("--freeze-core", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--core-lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr", type=float, required=True)
    p.add_argument("--anchor-weight", type=float, default=0.0)
    p.add_argument("--activity-anchor-weight", type=float, default=0.0)
    p.add_argument("--graph-output-anchor-weight", type=float, default=0.0,
                   help="MSE anchor on frozen core ImageConditionedGraph output versus anchor-gamma graph.")
    p.add_argument("--slot-reconstruction-weight", type=float, default=0.0)
    p.add_argument("--slot-num-slots", type=int, default=7)
    p.add_argument("--slot-temperature", type=float, default=0.3)
    p.add_argument("--max-samples", type=int, default=2500)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("require epochs >= 1 and 0 <= warmup < epochs")
    if args.freeze_core and (args.core_checkpoint is None or args.warmup_epochs != 0):
        raise ValueError("freeze-core requires a checkpoint and zero warmup epochs")
    if args.freeze_graph and args.graph_checkpoint is None:
        raise ValueError("freeze-graph requires graph-checkpoint")
    if args.graph_checkpoint is not None and args.core_checkpoint is not None:
        raise ValueError("graph-checkpoint and core-checkpoint are mutually exclusive")
    if (args.activity_anchor_weight > 0 or args.graph_output_anchor_weight > 0) and not args.freeze_core:
        raise ValueError("activity/graph-output anchors require a frozen core")
    validate_graph_output_anchor(args.graph_output_anchor_weight, args.freeze_core)
    if (args.encoder_lr <= 0 or args.core_lr <= 0 or args.anchor_weight < 0
            or args.activity_anchor_weight < 0 or args.graph_output_anchor_weight < 0
            or args.slot_reconstruction_weight < 0):
        raise ValueError("learning rates must be positive and loss weights nonnegative")
    if args.slot_num_slots < 2 or args.slot_temperature <= 0:
        raise ValueError("slot count must be >=2 and temperature positive")
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    anchors = torch.load(args.anchor_gamma, map_location="cpu", weights_only=True).float()
    ids = training_ids(args.max_samples)
    if tuple(anchors.shape) != (2500, 8, 256):
        raise ValueError(f"unexpected anchor gamma shape {tuple(anchors.shape)}")
    anchors = anchors[:len(ids)]
    with h5py.File(args.dataset, "r") as dataset:
        images = torch.from_numpy(dataset["image"][ids]).permute(0, 3, 1, 2)
    if images.dtype != torch.uint8 or tuple(images.shape[1:]) != (3, 128, 128):
        raise ValueError(f"unexpected RGB tensor {images.dtype} {tuple(images.shape)}")

    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    if stats.get("mode") != "standardize":
        raise ValueError("expected standardize preprocessing stats")
    mean = stats["mean"].float().to(device)
    std = stats["std"].float().to(device)
    clip = float(stats.get("clip", 3.0))
    if mean.numel() != 1 or std.numel() != 1 or float(std) <= 0:
        raise ValueError("invalid scalar feature statistics")

    hp = build_hparams()
    core = S2NetCore(hp, device=device).to(device)
    if args.graph_checkpoint is None:
        reset_graph_initialization(core, hp, device, args.graph_init_seed)
    else:
        load_graph_checkpoint(core, args.graph_checkpoint, device,
                              freeze=args.freeze_graph)
    if args.core_checkpoint is not None:
        core.load_state_dict(torch.load(args.core_checkpoint, map_location=device,
                                        weights_only=True), strict=True)
    initial_core = {key: value.detach().cpu().clone()
                    for key, value in core.state_dict().items()}
    initial_graph = {key: value.detach().cpu().clone()
                     for key, value in core.graph_generator.state_dict().items()}
    if args.freeze_core:
        core.requires_grad_(False)
    # The registered recipe has object_overlap_weight=0. Computing the legacy
    # maximal-clique classifier inside every training forward is therefore
    # unused and can become exponential on dense early-training affinities.
    # Evaluation still uses the full classifier from an unmodified core.
    core._detect_object_groups = (
        lambda core_out, spikes: [[] for _ in range(spikes.size(0))]
    )
    # Loading the auxiliary encoder must not perturb the baseline core RNG
    # sequence or the subsequent baseline-equivalent DataLoader shuffle.
    with torch.random.fork_rng(devices=[]):
        encoder = load_encoder(args.encoder, device)
    patcher = FeaturePatchGammaInitializer(grid_size=16).to(device)

    encoder.eval()
    with torch.no_grad():
        reproduced = gamma_from_images(
            encoder, patcher, images[:4].to(device), mean, std, clip).cpu()
    initial_max_diff = float((reproduced - anchors[:4]).abs().max())
    if initial_max_diff > 2e-6:
        raise AssertionError(f"encoder path does not reproduce cached gamma: {initial_max_diff}")
    initial_encoder = {key: value.detach().cpu().clone()
                       for key, value in encoder.state_dict().items()}

    loader = DataLoader(TensorDataset(images, anchors), batch_size=args.batch_size,
                        shuffle=True)
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0.0, spike_smooth_weight=0.0,
        spike_diversity_weight=0.0, structural_weight=0.0,
        sample_diversity_weight=0.0, plv_bimodality_weight=6.0,
        plv_balance_weight=10.0, plv_coherence_weight=0.5,
        plv_collapse_weight=1.0, plv_target_density=0.867,
        slot_reconstruction_weight=args.slot_reconstruction_weight,
        slot_num_slots=args.slot_num_slots,
        slot_temperature=args.slot_temperature,
        patch_grid_size=(16, 16),
    )
    groups = [{"params": encoder.parameters(), "lr": args.encoder_lr}]
    if not args.freeze_core:
        groups.insert(0, {"params": core.parameters(), "lr": args.core_lr})
    optimizer = torch.optim.Adam(groups)
    history = []
    for epoch in range(1, args.epochs + 1):
        core.train()
        encoder.train(epoch > args.warmup_epochs)
        totals = {"loss": 0.0, "primary": 0.0, "spike": 0.0,
                  "anchor": 0.0, "activity_anchor": 0.0,
                  "graph_output_anchor": 0.0,
                  "slot_reconstruction": 0.0}
        seen = 0
        for image_batch, anchor_batch in loader:
            image_batch = image_batch.to(device)
            anchor_batch = anchor_batch.to(device)
            if epoch <= args.warmup_epochs:
                gamma = anchor_batch
            else:
                gamma = gamma_from_images(encoder, patcher, image_batch,
                                          mean, std, clip)
            if args.activity_anchor_weight > 0:
                with torch.no_grad():
                    _, _, baseline_core_out, _, _ = _forward_with_plv(
                        core, anchor_batch, criterion, 32, "phase", "mean")
                    baseline_activity = torch.sigmoid(
                        baseline_core_out[..., 32:]).mean(dim=-1)
            groups, spikes, core_out, plv, theta = _forward_with_plv(
                core, gamma, criterion, 32, "phase", "mean")
            values = _select_loss_signal(spikes, core_out, "sigmoid_membrane")
            recon_target = patch_pool_rgb(image_batch.float().div(255.0), (16, 16))
            primary, primary_parts = criterion(
                spikes=values, object_groups=groups, sc=core.sc, plv=plv,
                theta=theta, plv_settle=32, recon_target=recon_target)
            component_plv = _component_spike_synchrony(core, 32)
            if component_plv is None:
                raise RuntimeError("component spike synchrony missing")
            spike_loss, _ = criterion(plv=component_plv, plv_settle=32)
            anchor_loss = (F.mse_loss(gamma, anchor_batch)
                           if epoch > args.warmup_epochs else gamma.new_zeros(()))
            activity_anchor_loss = (
                F.mse_loss(values[..., 32:].mean(dim=-1), baseline_activity)
                if args.activity_anchor_weight > 0 else gamma.new_zeros(())
            )
            if args.graph_output_anchor_weight > 0:
                if core.graph_generator is None:
                    raise RuntimeError("graph-output anchor requires learned graph mode")
                graph_anchor_loss = compute_graph_output_anchor_loss(
                    core, gamma, anchor_batch
                )
            else:
                graph_anchor_loss = gamma.new_zeros(())
            loss = (primary + 5.0 * spike_loss + args.anchor_weight * anchor_loss
                    + args.activity_anchor_weight * activity_anchor_loss
                    + args.graph_output_anchor_weight * graph_anchor_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            trainable = [p for p in list(core.parameters()) + list(encoder.parameters())
                         if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            batch = len(image_batch)
            seen += batch
            totals["loss"] += float(loss.detach()) * batch
            totals["primary"] += float(primary.detach()) * batch
            totals["spike"] += float(spike_loss.detach()) * batch
            totals["anchor"] += float(anchor_loss.detach()) * batch
            totals["activity_anchor"] += float(activity_anchor_loss.detach()) * batch
            totals["graph_output_anchor"] += float(graph_anchor_loss.detach()) * batch
            totals["slot_reconstruction"] += float(
                primary_parts.get("slot_reconstruction", gamma.new_zeros(())).detach()
            ) * batch
        record = {"epoch": epoch, **{key: value / seen for key, value in totals.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)

    save_s2net_core(core, out / "core.pt")
    torch.save(encoder.state_dict(), out / "encoder.pt")
    with torch.no_grad():
        final_gamma = gamma_from_images(
            encoder.eval(), patcher, images[:32].to(device), mean, std, clip).cpu()
    encoder_change = max(
        float((value.detach().cpu() - initial_encoder[key]).abs().max())
        for key, value in encoder.state_dict().items())
    gamma_drift = float((final_gamma - anchors[:32]).square().mean().sqrt())
    core_change = max(
        float((value.detach().cpu() - initial_core[key]).abs().max())
        for key, value in core.state_dict().items())
    graph_change = max(
        float((value.detach().cpu() - initial_graph[key]).abs().max())
        for key, value in core.graph_generator.state_dict().items())
    if encoder_change <= 0 or not math.isfinite(gamma_drift):
        raise AssertionError("joint path did not update the encoder finitely")
    if args.freeze_core and core_change != 0.0:
        raise AssertionError("frozen core changed")
    if args.freeze_graph and graph_change != 0.0:
        raise AssertionError("frozen graph changed")
    manifest = {
        "experiment": "SW0068 joint native feature generator and core",
        "seed": args.seed, "graph_init_seed": args.graph_init_seed,
        "graph_checkpoint": args.graph_checkpoint, "freeze_graph": args.freeze_graph,
        "core_checkpoint": args.core_checkpoint, "freeze_core": args.freeze_core,
        "source_core_sha256": sha256(args.core_checkpoint) if args.core_checkpoint else None,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "samples": len(ids),
        "training_ids": {"first": ids[0], "last": ids[-1],
                         "full_segments": [[0, 999], [1640, 3139]]
                         if len(ids) == 2500 else None},
        "core_lr": args.core_lr, "encoder_lr": args.encoder_lr,
        "data_order": "global RNG after baseline-equivalent core initialization",
        "anchor_weight": args.anchor_weight,
        "activity_anchor_weight": args.activity_anchor_weight,
        "graph_output_anchor_weight": args.graph_output_anchor_weight,
        "slot_reconstruction_weight": args.slot_reconstruction_weight,
        "slot_num_slots": args.slot_num_slots,
        "slot_temperature": args.slot_temperature,
        "initial_gamma_max_abs_diff": initial_max_diff,
        "final_gamma_rms_drift_n32": gamma_drift,
        "encoder_max_parameter_change": encoder_change,
        "core_max_parameter_change": core_change,
        "graph_max_parameter_change": graph_change,
        "source_encoder_sha256": sha256(args.encoder),
        "stats_sha256": sha256(args.stats),
        "anchor_gamma_sha256": sha256(args.anchor_gamma),
        "history": history,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
