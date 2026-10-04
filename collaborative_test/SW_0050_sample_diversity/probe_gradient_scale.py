"""Measure raw sample-activity loss and gradient scale on a real gamma batch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0040_peer_transfer"))

from evaluate import _core
from snn_kuramoto_bidirectional.loss_function import (
    UnsupervisedS2NetLoss,
    phase_locking_value,
    sample_activity_diversity_loss,
    signal_synchrony,
)


def _grouped_norms(named_parameters, grads):
    sums = {name: [] for name in ("graph_generator", "upstream_kuramoto", "dendrite_membrane")}
    all_terms = []
    for (parameter_name, parameter), grad in zip(named_parameters, grads):
        if not parameter.requires_grad or grad is None:
            continue
        category = (
            "graph_generator" if parameter_name.startswith("graph_generator.")
            else "dendrite_membrane" if parameter_name.startswith(("dendric_layer.", "membrane_layer."))
            else "upstream_kuramoto"
        )
        term = grad.detach().float().square().sum()
        sums[category].append(term)
        all_terms.append(term)
    norms = {key: float(torch.stack(values).sum().sqrt()) if values else 0.0
             for key, values in sums.items()}
    norms["all_parameters"] = float(torch.stack(all_terms).sum().sqrt()) if all_terms else 0.0
    return norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 for sample diversity.")

    gamma_blob = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    probe_batches = min(4, int(gamma_blob.shape[0]) // args.batch_size)
    if probe_batches < 1:
        raise ValueError("Gamma training tensor is smaller than the requested real batch.")
    model = _core(args.device, args.checkpoint, 64, "shared", 3, 1.5, 2.0,
                  0.5, 16.0, 0.35, "factorized")
    model.train()
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=0.0, spike_smooth_weight=0.0,
        spike_diversity_weight=0.0, structural_weight=0.0,
        sample_diversity_weight=0.0, plv_collapse_weight=1.0,
        plv_bimodality_weight=6.0, plv_balance_weight=10.0,
        plv_coherence_weight=0.5, plv_target_density=0.867,
        patch_grid_size=(16, 16),
    )
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                        if parameter.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    batch_reports = []
    for batch_index in range(probe_batches):
        gamma = gamma_blob[batch_index * args.batch_size:(batch_index + 1) * args.batch_size]
        gamma = gamma.float().to(args.device)
        _, _, membrane, theta = model(gamma, return_core_out=True, return_theta=True)
        activity = torch.sigmoid(membrane)
        phase_plv = phase_locking_value(theta, settle=32, combine="mean")
        baseline_loss, baseline_parts = criterion(
            spikes=activity, sc=model.sc, plv=phase_plv, theta=theta, plv_settle=32)

        component_spikes = getattr(model, "last_component_spikes", None)
        if component_spikes is None:
            raise RuntimeError("Checkpoint has no component spikes; expected SW0042 BIM6.")
        component_plvs = [signal_synchrony(component_spikes[:, index], settle=32)
                          for index in range(component_spikes.size(1))]
        spike_plv = torch.stack(component_plvs).prod(dim=0)
        spike_loss, spike_parts = criterion(plv=spike_plv, plv_settle=32)
        baseline_total = baseline_loss + 5.0 * spike_loss
        diversity = sample_activity_diversity_loss(activity)
        base_grads = torch.autograd.grad(baseline_total, parameters, retain_graph=True,
                                         allow_unused=True)
        diversity_grads = torch.autograd.grad(diversity, parameters, allow_unused=True)
        base_norms = _grouped_norms(named_parameters, base_grads)
        diversity_norms = _grouped_norms(named_parameters, diversity_grads)
        if base_norms["all_parameters"] <= 0.0 or diversity_norms["all_parameters"] <= 0.0:
            raise RuntimeError("Could not form finite, nonzero baseline/diversity gradient norms.")
        batch_reports.append({
            "batch_rows": [batch_index * args.batch_size,
                           (batch_index + 1) * args.batch_size - 1],
            "raw_losses": {
                "baseline_phase_objective": float(baseline_loss.detach()),
                "baseline_spike_phase_objective_unweighted": float(spike_loss.detach()),
                "baseline_total_with_spike_plv_weight_5": float(baseline_total.detach()),
                "sample_activity_diversity_unweighted": float(diversity.detach()),
                "phase_parts": {key: float(value.detach()) for key, value in baseline_parts.items()},
                "spike_plv_parts": {key: float(value.detach()) for key, value in spike_parts.items()},
            },
            "gradient_norms": {
                "baseline_total": base_norms,
                "sample_activity_diversity_unweighted": diversity_norms,
            },
        })
    baseline_grad_norm = sum(item["gradient_norms"]["baseline_total"]["all_parameters"]
                             for item in batch_reports) / probe_batches
    diversity_grad_norm = sum(item["gradient_norms"]["sample_activity_diversity_unweighted"]["all_parameters"]
                              for item in batch_reports) / probe_batches
    if baseline_grad_norm <= 0.0 or diversity_grad_norm <= 0.0:
        raise RuntimeError("Mean gradient norms must be positive.")
    ratio = baseline_grad_norm / diversity_grad_norm
    report = {
        "checkpoint": args.checkpoint,
        "gamma_path": args.gamma_path,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "gamma_sha256": hashlib.sha256(Path(args.gamma_path).read_bytes()).hexdigest(),
        "gamma_train_rows": [0, probe_batches * args.batch_size - 1],
        "batch_size": args.batch_size,
        "probe_batch_count": probe_batches,
        "model_protocol": {
            "steps": 64, "plv_settle": 32, "membrane_vth": 0.06,
            "dendritic_projection": "shared", "graph_spatial_decay": 0.35,
            "geodesic_steps": 3, "kuramoto_backend": "factorized",
            "activity_source": "sigmoid(membrane)",
        },
        "per_batch": batch_reports,
        "gradient_norms": {
            "baseline_total": baseline_grad_norm,
            "sample_activity_diversity_unweighted": diversity_grad_norm,
            "unweighted_diversity_to_baseline": diversity_grad_norm / baseline_grad_norm,
        },
        "suggested_weights": {
            "target_0.1x_baseline_gradient": 0.1 * ratio,
            "target_1.0x_baseline_gradient": ratio,
            "interpretation": "weight * raw diversity gradient norm divided by baseline total gradient norm; suggestions for coarse pilot only",
        },
        "ground_truth_used": False,
    }
    for value in (baseline_grad_norm, diversity_grad_norm, ratio):
        if not math.isfinite(value):
            raise RuntimeError("Non-finite gradient-scale measurement.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["suggested_weights"], indent=2))


if __name__ == "__main__":
    main()
