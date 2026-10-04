"""Same-checkpoint phase versus spike synchrony diagnostic, without labels."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.loss_function import phase_locking_value, signal_synchrony


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dendritic-projection", choices=["shared", "per_region"], required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--gamma-path", default="/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--settle", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(0)
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[:args.count].float()
    model = _core(args.device, args.checkpoint, args.steps, args.dendritic_projection)
    phase_rows, spike_rows = [], []
    with torch.no_grad():
        for start in range(0, len(gamma), 8):
            batch = gamma[start:start + 8].to(args.device)
            _, spikes, _, theta = model(batch, return_core_out=True, return_theta=True)
            phase_rows.append(phase_locking_value(theta, settle=args.settle).cpu())
            spike_rows.append(signal_synchrony(spikes, settle=args.settle).cpu())
    phase = torch.cat(phase_rows)
    spike = torch.cat(spike_rows)
    correlations = []
    for p, s in zip(phase, spike):
        mask = ~torch.eye(p.shape[-1], dtype=torch.bool)
        pv, sv = p[mask], s[mask]
        pv, sv = pv - pv.mean(), sv - sv.mean()
        correlations.append(float((pv @ sv) / (pv.norm() * sv.norm()).clamp_min(1e-8)))
    result = {
        "checkpoint": args.checkpoint,
        "dendritic_projection": args.dendritic_projection,
        "ids": [0, len(gamma) - 1],
        "steps": args.steps,
        "settle": args.settle,
        "mean_phase_spike_affinity_correlation": sum(correlations) / len(correlations),
        "phase_affinity_mean": float(phase.mean()),
        "spike_affinity_mean": float(spike.mean()),
        "label_free_diagnostic": True,
    }
    Path(args.output_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
