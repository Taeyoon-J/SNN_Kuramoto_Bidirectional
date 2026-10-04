"""Compare adapted edge loss against the existing patch_sw implementation."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from snn_kuramoto_bidirectional.loss_function import edge_membrane_separation_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-sw-loss", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    peer_path = Path(args.patch_sw_loss)
    sys.path.insert(0, str(peer_path.parent))
    spec = importlib.util.spec_from_file_location("patch_sw_loss_for_comparison", peer_path)
    peer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peer)
    torch.manual_seed(43)
    membrane = torch.randn(2, 256, 8, requires_grad=True)
    images = torch.rand(2, 3, 128, 128)
    ours = edge_membrane_separation_loss(membrane, images, (16, 16))
    theirs = peer.edge_membrane_separation_loss(membrane, images, (16, 16))
    our_grad = torch.autograd.grad(ours, membrane, retain_graph=True)[0]
    their_grad = torch.autograd.grad(theirs, membrane)[0]
    result = {"our_loss": float(ours.detach()), "patch_sw_loss": float(theirs.detach()),
              "absolute_loss_difference": float((ours - theirs).abs().detach()),
              "maximum_gradient_difference": float((our_grad - their_grad).abs().max())}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
