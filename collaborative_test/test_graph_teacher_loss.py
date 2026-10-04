"""Small CPU smoke test for the optional graph-to-membrane loss."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snn_kuramoto_bidirectional"))
from loss_function import graph_teacher_synchrony_loss


def main():
    torch.manual_seed(0)
    membrane = torch.randn(2, 6, 12, requires_grad=True)
    graph = torch.rand(2, 6, 6, requires_grad=True)
    value = graph_teacher_synchrony_loss(membrane, graph, settle=3)
    value.backward()
    assert torch.isfinite(value)
    assert membrane.grad is not None and torch.isfinite(membrane.grad).all()
    assert membrane.grad.norm() > 0
    assert graph.grad is None
    print(f"PASS loss={value.item():.6f} membrane_grad_norm={membrane.grad.norm().item():.6f}")


if __name__ == "__main__":
    main()
