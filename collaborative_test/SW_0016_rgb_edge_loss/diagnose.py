"""Check RGB-edge loss magnitude and gradient reach on a fixed 8-image batch."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.loss_function import edge_membrane_separation_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ids = list(range(1320, 1328))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float().to(args.device)
    with h5py.File(args.dataset_path, "r") as dataset:
        image = torch.from_numpy(dataset["image"][ids]).permute(0, 3, 1, 2).float().to(args.device) / 255.0
    core = _core(args.device, args.checkpoint, 256)
    _, _, membrane = core(gamma, return_core_out=True)
    loss = edge_membrane_separation_loss(membrane, image, (16, 16), margin=0.3)
    named = [(name, value) for name, value in core.named_parameters()
             if "membrane" in name or "dendric" in name]
    gradients = torch.autograd.grad(loss, [value for _, value in named], allow_unused=True)
    result = {"ids": [1320, 1327], "checkpoint": args.checkpoint, "raw_loss": float(loss),
              "gradient_norms": {name: (None if grad is None else float(grad.norm()))
                                 for (name, _), grad in zip(named, gradients)}}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
