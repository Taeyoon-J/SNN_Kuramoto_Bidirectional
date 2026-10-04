"""Compare pairwise/factorized dynamics on one cached-gamma batch and update."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT))

from evaluate_fixed_split import _core


def run_once(model, gamma, backend, learning_rate):
    model.zero_grad(set_to_none=True)
    _, spikes, membrane, theta = model(
        gamma, return_core_out=True, return_theta=True)
    # A diagnostic scalar only; this is not the experiment's training loss.
    loss = membrane.square().mean() + 0.01 * theta.square().mean()
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.step()
    update = {name: p.detach() - before[name]
              for name, p in model.named_parameters()}
    return {
        "backend": backend,
        "loss": float(loss.detach()),
        "spikes": spikes.detach(),
        "membrane": membrane.detach(),
        "theta": theta.detach(),
        "gradients": grads,
        "update": update,
    }


def max_abs(left, right):
    return float((left - right).abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", default="/work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt")
    parser.add_argument("--gamma-global-start", type=int, default=0)
    parser.add_argument("--image-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error("steps and learning-rate must be positive")

    rows = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    row = args.image_id - args.gamma_global_start
    if row < 0 or row >= len(rows):
        raise ValueError("image-id is outside the aligned gamma slice")
    gamma = rows[row:row + 1].float().to(args.device)
    shared_kwargs = dict(
        device=args.device, checkpoint=args.checkpoint, steps=args.steps,
        dendritic_projection="shared", geodesic_steps=3,
        geodesic_radius=1.5, geodesic_contrast=2.0,
        geodesic_temperature=0.5, geodesic_cap=16.0,
        graph_spatial_decay=0.35,
    )
    pairwise = _core(**shared_kwargs, kuramoto_backend="pairwise")
    factorized = _core(**shared_kwargs, kuramoto_backend="factorized")
    if set(pairwise.state_dict()) != set(factorized.state_dict()):
        raise RuntimeError("Backend changed checkpoint state_dict keys")
    pairwise.load_state_dict(factorized.state_dict(), strict=True)
    pairwise.membrane_layer.vth = 0.06
    factorized.membrane_layer.vth = 0.06

    result_pairwise = run_once(pairwise, gamma, "pairwise", args.learning_rate)
    result_factorized = run_once(factorized, gamma, "factorized", args.learning_rate)
    gradient_names = sorted(set(result_pairwise["gradients"])
                            & set(result_factorized["gradients"]))
    update_names = sorted(set(result_pairwise["update"]) & set(result_factorized["update"]))
    report = {
        "checkpoint": args.checkpoint,
        "gamma_path": args.gamma_path,
        "global_image_id": args.image_id,
        "rollout_steps": args.steps,
        "comparison": "same checkpoint, one aligned gamma batch, one diagnostic SGD update",
        "diagnostic_loss": "mean(membrane^2) + 0.01 * mean(theta^2); not the training objective",
        "state_dict_keys_identical": True,
        "max_abs_forward_difference": {
            key: max_abs(result_pairwise[key], result_factorized[key])
            for key in ("spikes", "membrane", "theta")
        },
        "max_abs_gradient_difference": max(
            (max_abs(result_pairwise["gradients"][name],
                     result_factorized["gradients"][name])
             for name in gradient_names), default=0.0),
        "all_compared_gradients_finite": all(
            torch.isfinite(result_pairwise["gradients"][name]).all()
            and torch.isfinite(result_factorized["gradients"][name]).all()
            for name in gradient_names),
        "max_abs_parameter_update_difference": max(
            (max_abs(result_pairwise["update"][name],
                     result_factorized["update"][name])
             for name in update_names), default=0.0),
        "backend_losses": {
            "pairwise": result_pairwise["loss"],
            "factorized": result_factorized["loss"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
