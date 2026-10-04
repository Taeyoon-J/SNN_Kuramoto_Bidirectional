"""Measure which core parameter groups receive gradients under the baseline loss."""

import argparse
import json
import sys
from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "snn_kuramoto_bidirectional"
sys.path.insert(0, str(PACKAGE_DIR))

from hyperparameter import S2NetHyperparameters
from loss_function import UnsupervisedS2NetLoss, graph_teacher_synchrony_loss
from s2net_cls import S2NetCore
from training.train_s2net_core import _forward_with_plv, _select_loss_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-teacher-weight", type=float, default=0.0)
    args = parser.parse_args()

    gamma = torch.load(args.gamma_path, map_location="cpu")[: args.num_samples]
    gamma = gamma.to(args.device)
    hp = S2NetHyperparameters(
        num_feature_maps=8,
        num_regions=256,
        sc=None,
        gamma_drive_mode="static",
        num_time_steps=64,
        theta_init="gamma",
        gamma_phase_mode="standardize_tanh",
        osc_dim=4,
        freq_gain=2.0,
        graph_mode="learned",
        graph_top_k=32,
        graph_spatial_decay=0.55,
        k=256.0,
        low_n=-4.0,
        high_n=0.0,
        membrane_vth=0.06,
        membrane_low_m=-4.0,
        membrane_high_m=0.0,
        gate_mode="raw",
        spike_classify_method="spatial_components",
        spike_spatial_grid_size=(16, 16),
        spike_per_component=True,
    ).validate()
    core = S2NetCore(hp, device=args.device).to(args.device)
    core.load_state_dict(torch.load(args.checkpoint, map_location=args.device), strict=True)
    core.train()
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0,
        spike_smooth_weight=0,
        spike_diversity_weight=0,
        structural_weight=0,
        plv_collapse_weight=1,
        plv_bimodality_weight=1,
        plv_balance_weight=10,
        plv_coherence_weight=0.5,
        plv_target_density=0.867,
        patch_grid_size=16,
    )
    groups, spikes, membrane, plv, theta = _forward_with_plv(
        core, gamma, criterion, 32, "phase", "mean"
    )
    activity = _select_loss_signal(spikes, membrane, "sigmoid_membrane")
    loss, parts = criterion(
        spikes=activity,
        object_groups=groups,
        sc=core.sc,
        plv=plv,
        theta=theta,
        plv_settle=32,
    )
    if args.graph_teacher_weight != 0.0:
        with torch.no_grad():
            graph = core.graph_generator(gamma)
        raw_teacher = graph_teacher_synchrony_loss(membrane, graph, settle=32)
        loss = loss + args.graph_teacher_weight * raw_teacher
        parts["graph_teacher_synchrony"] = raw_teacher.detach()
        parts["total"] = loss.detach()
    loss.backward()

    summary = {
        "loss": loss.detach().item(),
        "graph_teacher_weight": args.graph_teacher_weight,
        "parts": {k: v.detach().item() for k, v in parts.items()},
    }
    summary["gradients"] = {}
    for group in (
        "gamma_channel_proj",
        "graph_generator",
        "kuramoto",
        "dendric_layer",
        "membrane_layer",
    ):
        params = [(name, value) for name, value in core.named_parameters() if name.startswith(group)]
        summary["gradients"][group] = {
            "parameter_count": len(params),
            "connected_count": sum(value.grad is not None for _, value in params),
            "norms": {
                name: None if value.grad is None else float(value.grad.norm())
                for name, value in params
            },
        }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
