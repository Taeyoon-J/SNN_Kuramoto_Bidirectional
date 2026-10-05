#!/usr/bin/env python3
"""Tune only the frozen SW0072 core's image-conditioned graph generator."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]

from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss, patch_pool_rgb
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.train_s2net_core import (
    _component_spike_synchrony,
    _forward_with_plv,
    _select_loss_signal,
    save_s2net_core,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_only_parameters(core):
    if getattr(core, "graph_generator", None) is None:
        raise ValueError("graph-only training requires a learned graph generator")
    core.requires_grad_(False)
    core.graph_generator.requires_grad_(True)
    return [parameter for parameter in core.graph_generator.parameters() if parameter.requires_grad]


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


def run_training(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0:
        raise ValueError("epochs/batch size must be positive and max_steps nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("graph learning rate must be finite and positive")
    if not math.isfinite(args.reconstruction_weight) or args.reconstruction_weight < 0:
        raise ValueError("reconstruction weight must be finite and nonnegative")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    if args.count != 2500 or args.start != 0:
        raise ValueError("SW0080 is fixed to the registered 2500-row training gamma")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    hp = build_hparams()
    core = S2NetCore(hp, device=device).to(device)
    source_state = torch.load(args.core_checkpoint, map_location=device, weights_only=True)
    core.load_state_dict(source_state, strict=True)
    original_core = {key: value.detach().cpu().clone() for key, value in core.state_dict().items()}
    original_graph = {key: value.detach().cpu().clone()
                      for key, value in core.graph_generator.state_dict().items()}
    trainable = graph_only_parameters(core)
    if not trainable or any(parameter.requires_grad for name, parameter in core.named_parameters()
                            if not name.startswith("graph_generator.")):
        raise AssertionError("only graph_generator parameters may be trainable")

    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True).float()
    if tuple(gamma.shape) != (2500, 8, 256) or not torch.isfinite(gamma).all():
        raise ValueError("training gamma must be finite [2500,8,256]")
    ids = list(range(1000)) + list(range(1640, 3140))
    with h5py.File(args.dataset_path, "r") as dataset:
        if ids[-1] >= len(dataset["image"]):
            raise ValueError("training image IDs exceed the HDF5 image dataset")
        images = torch.from_numpy(dataset["image"][ids]).permute(0, 3, 1, 2).contiguous()
    if images.dtype != torch.uint8 or tuple(images.shape) != (2500, 3, 128, 128):
        raise ValueError(f"expected 2500 RGB images, got {images.dtype} {tuple(images.shape)}")

    # Match data order and stochastic state across reconstruction arms after
    # architecture construction/checkpoint loading has completed.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    loader = DataLoader(TensorDataset(images, gamma), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0.0, spike_smooth_weight=0.0,
        spike_diversity_weight=0.0, structural_weight=0.0,
        sample_diversity_weight=0.0, plv_bimodality_weight=6.0,
        plv_balance_weight=10.0, plv_coherence_weight=0.5,
        plv_collapse_weight=1.0, plv_target_density=0.867,
        slot_reconstruction_weight=args.reconstruction_weight,
        slot_num_slots=7, slot_temperature=0.3,
        patch_grid_size=(16, 16),
    )
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    core._detect_object_groups = lambda core_out, spikes: [[] for _ in range(spikes.size(0))]

    step = 0
    history = []
    stop = False
    for epoch in range(1, args.epochs + 1):
        core.train()
        totals = {key: 0.0 for key in ("loss", "primary", "spike", "slot_reconstruction", "graph_grad_norm")}
        seen = batches = 0
        for image_batch, gamma_batch in loader:
            image_batch = image_batch.to(device)
            gamma_batch = gamma_batch.to(device)
            groups, spikes, core_out, plv, theta = _forward_with_plv(
                core, gamma_batch, criterion, 32, "phase", "mean"
            )
            values = _select_loss_signal(spikes, core_out, "sigmoid_membrane")
            recon_target = patch_pool_rgb(image_batch.float().div(255.0), (16, 16))
            primary, parts = criterion(
                spikes=values, object_groups=groups, sc=core.sc, plv=plv,
                theta=theta, plv_settle=32, recon_target=recon_target,
            )
            component_plv = _component_spike_synchrony(core, 32)
            if component_plv is None:
                raise RuntimeError("component spike synchrony missing")
            spike_loss, _ = criterion(plv=component_plv, plv_settle=32)
            loss = primary + 5.0 * spike_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite graph-only loss at epoch {epoch}, step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("nonfinite graph gradient")
            optimizer.step()
            batch = len(image_batch)
            seen += batch
            batches += 1
            step += 1
            totals["loss"] += float(loss.detach()) * batch
            totals["primary"] += float(primary.detach()) * batch
            totals["spike"] += float(spike_loss.detach()) * batch
            totals["slot_reconstruction"] += float(
                parts.get("slot_reconstruction", loss.new_zeros(())).detach()
            ) * batch
            totals["graph_grad_norm"] += float(grad_norm.detach()) * batch
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        record = {"epoch": epoch, "steps": step, "samples_seen": seen,
                  **{key: value / seen for key, value in totals.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)
        if stop:
            break

    output.mkdir(parents=True)
    save_s2net_core(core, output / "core.pt")
    graph_change = max(float((value.detach().cpu() - original_graph[key]).abs().max())
                       for key, value in core.graph_generator.state_dict().items())
    non_graph_changes = {
        key: float((value.detach().cpu() - original_core[key]).abs().max())
        for key, value in core.state_dict().items() if not key.startswith("graph_generator.")
    }
    if max(non_graph_changes.values(), default=0.0) != 0.0:
        raise AssertionError("a non-graph core tensor changed")
    if graph_change <= 0.0:
        raise AssertionError("graph generator did not update")
    manifest = {
        "experiment": "SW0080 graph-only phase/spike binding with RGB slot reconstruction",
        "seed": args.seed, "training_ids": {"segments_inclusive": [[0, 999], [1640, 3139]],
                                               "count": len(ids)},
        "gamma_path": str(Path(args.gamma_path).resolve()),
        "gamma_sha256": sha256(args.gamma_path),
        "source_core_checkpoint": str(Path(args.core_checkpoint).resolve()),
        "source_core_sha256": sha256(args.core_checkpoint),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "code_dependencies_sha256": {
            relative: sha256(Path(__file__).resolve().parents[2] / relative)
            for relative in (
                "snn_kuramoto_bidirectional/s2net_cls.py",
                "snn_kuramoto_bidirectional/graph_generator.py",
                "snn_kuramoto_bidirectional/kuramoto_layer.py",
                "snn_kuramoto_bidirectional/loss_function.py",
                "snn_kuramoto_bidirectional/training/train_s2net_core.py",
            )
        },
        "epochs_requested": args.epochs, "steps_completed": step,
        "batch_size": args.batch_size, "graph_learning_rate": args.lr,
        "reconstruction_weight": args.reconstruction_weight,
        "primary_objectives": {"phase_binding": "existing PLV criterion", "component_spike_weight": 5.0},
        "freeze_contract": {"encoder_frozen": True, "non_graph_core_frozen": True,
                             "non_graph_max_abs_change": max(non_graph_changes.values(), default=0.0),
                             "graph_max_abs_change": graph_change},
        "ground_truth_used_for_training": False,
        "training_protocol": "cached aligned gamma + RGB images only; masks and object counts are never opened",
        "history": history,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "TRAINING_COMPLETED").write_text("completed\n", encoding="ascii")
    print(json.dumps({"output": str(output), "manifest": str(output / "manifest.json"),
                      "graph_max_abs_change": graph_change}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core-checkpoint", required=True)
    p.add_argument("--gamma-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--reconstruction-weight", type=float, default=0.3)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Optional bounded real-asset smoke; zero runs the full epoch budget.")
    p.add_argument("--device", default="cuda")
    run_training(p.parse_args())


if __name__ == "__main__":
    main()
