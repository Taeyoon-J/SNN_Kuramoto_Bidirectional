"""Check that graph teacher loss on actual spikes reaches SNN parameters."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.loss_function import graph_teacher_synchrony_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[1320:1322].float().to(args.device)
    core = _core(args.device, args.checkpoint, 64)
    _, spikes, membrane = core(gamma, return_core_out=True)
    with torch.no_grad():
        graph = core.graph_generator(gamma)
    names = [name for name, _ in core.named_parameters() if "dendric" in name or "membrane" in name]
    parameters = [dict(core.named_parameters())[name] for name in names]
    results = {}
    for source, signal in (("membrane", membrane), ("spikes", spikes)):
        loss = graph_teacher_synchrony_loss(signal, graph, settle=32, temperature=0.1)
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        results[source] = {
            "raw_loss": float(loss.detach()),
            "gradient_norms": {
                name: None if grad is None else float(grad.norm())
                for name, grad in zip(names, gradients)
            },
        }
    results.update({"ids": [1320, 1321], "checkpoint": args.checkpoint,
                    "spike_rate": float(spikes.detach().mean())})
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
