#!/usr/bin/env python3
"""Train only the graph generator with horizontal-flip graph equivariance."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from equivariance import (freeze_graph_only, graph_equivariance_mse, grad_norm,
                          target_equivariance_weights)
from trajectory import assert_checkpoint_equal, save_epoch_checkpoint
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.train_s2net_core import (
    _component_spike_synchrony, _forward_with_plv, _select_loss_signal,
    save_s2net_core,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CODE_DEPENDENCIES = (
    "snn_kuramoto_bidirectional/s2net_cls.py",
    "snn_kuramoto_bidirectional/graph_generator.py",
    "snn_kuramoto_bidirectional/input_layer_generator.py",
    "snn_kuramoto_bidirectional/gamma_initializer.py",
    "snn_kuramoto_bidirectional/loss_function.py",
    "snn_kuramoto_bidirectional/training/train_s2net_core.py",
)


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


def graph_loss_terms(core, encoder, patcher, images, gamma, mean, std, clip):
    # Teacher is the current learned graph on registered cached gamma.
    with torch.no_grad():
        teacher = core.graph_generator(gamma)
        flipped_images = torch.flip(images, dims=(-1,))
        features = encoder(flipped_images.float().div(255.0))
        flipped_gamma = patcher(((features - mean) / std).clamp(-clip, clip))
    flipped_graph = core.graph_generator(flipped_gamma)
    loss, aligned = graph_equivariance_mse(teacher, flipped_graph, 16)
    return loss, teacher, aligned


def run(args):
    if args.seed != 1 or args.count != 2500 or args.start != 0:
        raise ValueError("SW0081 is fixed to seed1 and 2500 registered training rows")
    if args.epochs < 1 or args.batch_size < 1 or args.max_steps != 0:
        raise ValueError("SW0082 requires full epochs; epochs/batch size must be positive")
    if not math.isfinite(args.equivariance_weight) or args.equivariance_weight < 0:
        raise ValueError("equivariance weight must be finite and nonnegative")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    epoch_dir = Path(args.epoch_checkpoint_dir)
    if epoch_dir.exists():
        raise FileExistsError(f"refusing existing epoch checkpoint directory: {epoch_dir}")
    if not Path(args.preflight_marker).is_file():
        raise FileNotFoundError("SW0081 calibration marker is required")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    hp = build_hparams()
    core = S2NetCore(hp, device=device).to(device)
    core.load_state_dict(torch.load(args.core_checkpoint, map_location=device, weights_only=True), strict=True)
    encoder = CNNFeatureEncoder(8, 3, in_channels=3, bias=True).to(device)
    encoder.load_state_dict(torch.load(args.encoder, map_location=device, weights_only=True), strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    params = freeze_graph_only(core, encoder)

    initial_core = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    initial_encoder = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    initial_graph = {k: v.detach().cpu().clone() for k, v in core.graph_generator.state_dict().items()}

    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True).float()
    if tuple(gamma.shape) != (2500, 8, 256) or not torch.isfinite(gamma).all():
        raise ValueError("gamma must be finite with shape [2500,8,256]")
    ids = list(range(1000)) + list(range(1640, 3140))
    with h5py.File(args.dataset_path, "r") as data:
        images = torch.from_numpy(data["image"][ids]).permute(0, 3, 1, 2).contiguous()
    if images.dtype != torch.uint8 or tuple(images.shape) != (2500, 3, 128, 128):
        raise ValueError("expected RGB HDF5 training images [2500,3,128,128]")
    stats = torch.load(args.stats, map_location="cpu", weights_only=True)
    if stats.get("mode") != "standardize":
        raise ValueError("registered encoder stats must use standardize mode")
    mean, std = stats["mean"].float().to(device), stats["std"].float().to(device)
    clip = float(stats.get("clip", 3.0))
    if mean.numel() != 1 or std.numel() != 1 or float(std) <= 0:
        raise ValueError("encoder preprocessing stats must be finite scalar mean/std")
    patcher = FeaturePatchGammaInitializer(grid_size=(16, 16)).to(device)

    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0.0, spike_smooth_weight=0.0,
        spike_diversity_weight=0.0, structural_weight=0.0,
        sample_diversity_weight=0.0, plv_bimodality_weight=6.0,
        plv_balance_weight=10.0, plv_coherence_weight=0.5,
        plv_collapse_weight=1.0, plv_target_density=0.867,
        slot_reconstruction_weight=0.0, slot_num_slots=7,
        slot_temperature=0.3, patch_grid_size=(16, 16),
    )
    loader = DataLoader(TensorDataset(images, gamma), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    core._detect_object_groups = lambda core_out, spikes: [[] for _ in range(spikes.size(0))]

    # Independent component gradients quantify a reproducible scale for arms.
    train_iterator = iter(loader)
    first_images, first_gamma = next(train_iterator)
    first_images, first_gamma = first_images.to(device), first_gamma.to(device)
    groups, spikes, core_out, plv, theta = _forward_with_plv(
        core, first_gamma, criterion, 32, "phase", "mean")
    signal = _select_loss_signal(spikes, core_out, "sigmoid_membrane")
    binding, _ = criterion(spikes=signal, object_groups=groups, sc=core.sc,
                           plv=plv, theta=theta, plv_settle=32)
    component_plv = _component_spike_synchrony(core, 32)
    if component_plv is None:
        raise RuntimeError("component spike synchrony unavailable")
    spike_loss, _ = criterion(plv=component_plv, plv_settle=32)
    binding_total = binding + 5.0 * spike_loss
    eq_loss, teacher, aligned = graph_loss_terms(
        core, encoder, patcher, first_images, first_gamma, mean, std, clip)
    bind_grad = grad_norm(binding_total, params, retain_graph=True)
    eq_grad = grad_norm(eq_loss, params, retain_graph=True)
    proposed_weights = target_equivariance_weights(bind_grad, eq_grad)
    weight = args.equivariance_weight
    if args.propose_weight_0p1x:
        weight = proposed_weights["weight_0p1x"]
    loss = binding_total + weight * eq_loss
    if not torch.isfinite(loss) or not torch.isfinite(eq_loss):
        raise FloatingPointError("nonfinite pre-update loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(grad):
        raise FloatingPointError("nonfinite graph gradient")
    optimizer.step()
    preflight_record = {
        "initial_raw_equivariance_mse": float(eq_loss.detach()),
        "binding_plus_spike_loss": float(binding_total.detach()),
        "binding_gradient_norm": float(bind_grad.detach()),
        "raw_equivariance_gradient_norm": float(eq_grad.detach()),
        "proposed_weights_for_gradient_ratios": proposed_weights,
        "equivariance_weight_used": weight,
        "weighted_equivariance_gradient_ratio": float(weight * eq_grad / bind_grad.clamp_min(1e-30)),
        "aligned_teacher_mse_after_flip": float(F.mse_loss(aligned, teacher).detach()),
        "loss_finite": True,
    }

    step, history = 1, []
    totals = {key: 0.0 for key in ("total", "binding", "spike", "equivariance", "graph_grad_norm")}
    totals["total"] += float(loss.detach()) * len(first_images)
    totals["binding"] += float(binding.detach()) * len(first_images)
    totals["spike"] += float(spike_loss.detach()) * len(first_images)
    totals["equivariance"] += float(eq_loss.detach()) * len(first_images)
    totals["graph_grad_norm"] += float(grad.detach()) * len(first_images)
    seen = len(first_images)
    max_steps = args.max_steps
    if not max_steps:
        # Reuse first batch as the first update, then run remaining stream batches.
        iterator = train_iterator
        stop = False
        for epoch in range(1, args.epochs + 1):
            if epoch > 1:
                iterator = iter(loader)
            for image_batch, gamma_batch in iterator:
                image_batch, gamma_batch = image_batch.to(device), gamma_batch.to(device)
                groups, spikes, core_out, plv, theta = _forward_with_plv(
                    core, gamma_batch, criterion, 32, "phase", "mean")
                signal = _select_loss_signal(spikes, core_out, "sigmoid_membrane")
                bind, _ = criterion(spikes=signal, object_groups=groups, sc=core.sc,
                                    plv=plv, theta=theta, plv_settle=32)
                cp = _component_spike_synchrony(core, 32)
                sp, _ = criterion(plv=cp, plv_settle=32)
                eq, _, _ = graph_loss_terms(core, encoder, patcher, image_batch,
                                             gamma_batch, mean, std, clip)
                current_loss = bind + 5.0 * sp + weight * eq
                if not torch.isfinite(current_loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                current_loss.backward()
                current_grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                n = len(image_batch); step += 1; seen += n
                totals["total"] += float(current_loss.detach()) * n
                totals["binding"] += float(bind.detach()) * n
                totals["spike"] += float(sp.detach()) * n
                totals["equivariance"] += float(eq.detach()) * n
                totals["graph_grad_norm"] += float(current_grad.detach()) * n
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
            record = {"epoch": epoch, "steps": step,
                      **{k: v / seen for k, v in totals.items()}}
            history.append(record); print(json.dumps(record), flush=True)
            save_epoch_checkpoint(core, epoch_dir, epoch)
            if stop:
                break

    # For preflight max_steps=1, above one-update measurement is the whole run.
    output.mkdir(parents=True)
    save_s2net_core(core, output / "core.pt")
    final_epoch_path = epoch_dir / f"epoch_{len(history):02d}_core.pt"
    if len(history) != args.epochs or not final_epoch_path.is_file():
        raise AssertionError("expected a saved checkpoint at each requested epoch")
    assert_checkpoint_equal(output / "core.pt", final_epoch_path)
    non_graph = {key: float((value.detach().cpu() - initial_core[key]).abs().max())
                 for key, value in core.state_dict().items()
                 if not key.startswith("graph_generator.")}
    encoder_change = max(float((v.detach().cpu() - initial_encoder[k]).abs().max())
                         for k, v in encoder.state_dict().items())
    graph_change = max(float((v.detach().cpu() - initial_graph[k]).abs().max())
                       for k, v in core.graph_generator.state_dict().items())
    if max(non_graph.values(), default=0.0) != 0 or encoder_change != 0 or graph_change <= 0:
        raise AssertionError("freeze/update invariants failed")
    manifest = {
        "experiment": "SW0081 graph-only horizontal-flip equivariance",
        "seed": 1, "training_ids": {"segments_inclusive": [[0, 999], [1640, 3139]], "count": 2500},
        "gamma_path": str(Path(args.gamma_path).resolve()), "gamma_sha256": sha256(args.gamma_path),
        "source_core_checkpoint": str(Path(args.core_checkpoint).resolve()),
        "source_core_sha256": sha256(args.core_checkpoint),
        "encoder_path": str(Path(args.encoder).resolve()), "encoder_sha256": sha256(args.encoder),
        "stats_path": str(Path(args.stats).resolve()), "stats_sha256": sha256(args.stats),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "sw0081_preflight_marker": str(Path(args.preflight_marker).resolve()),
        "sw0081_preflight_sha256": sha256(args.preflight_marker),
        "equivariance_impl_sha256": sha256(Path(__file__).with_name("equivariance.py")),
        "trajectory_helper_sha256": sha256(Path(__file__).with_name("trajectory.py")),
        "trajectory_checkpointing": {"checkpoint_count": len(history),
                                     "final_matches_epoch_checkpoint_bitwise": True,
                                     "epoch_checkpoint_sha256": {
                                         str(epoch): sha256(epoch_dir / f"epoch_{epoch:02d}_core.pt")
                                         for epoch in range(1, args.epochs + 1)},
                                     "final_checkpoint_sha256": sha256(output / "core.pt")},
        "code_dependencies_sha256": {
            relative: sha256(ROOT / relative) for relative in CODE_DEPENDENCIES
        },
        "epochs_requested": args.epochs, "steps_completed": step, "batch_size": args.batch_size,
        "graph_learning_rate": args.lr, "equivariance_weight": weight,
        "objective": {"phase_binding": 1.0, "component_spike": 5.0,
                      "equivariance_target_gradient_ratio": "0.1x" if args.propose_weight_0p1x else "configured"},
        "preflight_scale_measurement": preflight_record,
        "freeze_contract": {"encoder_exact_unchanged": encoder_change == 0,
                            "non_graph_core_exact_unchanged": max(non_graph.values(), default=0) == 0,
                            "non_graph_max_abs_change": max(non_graph.values(), default=0),
                            "encoder_max_abs_change": encoder_change,
                            "graph_max_abs_change": graph_change},
        "ground_truth_used": False,
        "training_protocol": "cached gamma plus RGB horizontal-flip consistency; no masks/counts opened",
        "history": history,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "TRAINING_COMPLETED").write_text("completed\n", encoding="ascii")
    print(json.dumps({"output": str(output), "manifest": str(output / "manifest.json"),
                      "equivariance": preflight_record, "graph_change": graph_change}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core-checkpoint", required=True)
    p.add_argument("--gamma-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epoch-checkpoint-dir", required=True)
    p.add_argument("--preflight-marker", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--equivariance-weight", type=float, default=0.0)
    p.add_argument("--propose-weight-0p1x", action="store_true",
                   help="use measured weight making equivariance gradient 0.1x binding gradient")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()
