"""Meaningful CPU checks for correlation semantics and differentiability."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _component_spike_synchrony


def main():
    opposite = torch.tensor([0., 1., 0., 1.])
    components = torch.stack((opposite, 1. - opposite)).reshape(1, 1, 2, 4).repeat(1, 4, 1, 1)
    old = _component_spike_synchrony(SimpleNamespace(last_component_spikes=components), 0)
    new = spike_synchrony_affinity(components.mean(1), components)
    assert old[0, 0, 1] > .999 and new[0, 0, 1] == 0
    generator = torch.Generator().manual_seed(9)
    # Mostly positively correlated, with noise: product has a useful derivative.
    base = torch.randn(1, 4, 1, 32, generator=generator)
    values = (base.expand(1, 4, 5, 32) + .3 * torch.randn(1, 4, 5, 32, generator=generator)).requires_grad_()
    result = spike_synchrony_affinity(values.mean(1), values)
    result.sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
    constant = torch.ones(1, 4, 2, 32, requires_grad=True)
    empty = spike_synchrony_affinity(constant.mean(1), constant)
    assert torch.isfinite(empty).all() and empty.abs().max() == 0
    empty.sum().backward()
    assert torch.isfinite(constant.grad).all()
    print("PASS sign mismatch counterexample, useful gradient, constant-trace stability")


if __name__ == "__main__":
    main()
