import unittest

import torch
from torch import nn

from train_graph import graph_only_parameters


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_like = nn.Linear(2, 2)
        self.dynamics = nn.Linear(2, 2)
        self.graph_generator = nn.Linear(2, 2, bias=False)


class GraphOnlyTrainingTests(unittest.TestCase):
    def test_only_graph_parameters_train_and_receive_reconstruction_gradient(self):
        core = ToyCore()
        params = graph_only_parameters(core)
        self.assertTrue(params)
        self.assertTrue(all(p.requires_grad for p in core.graph_generator.parameters()))
        self.assertFalse(any(p.requires_grad for n, p in core.named_parameters()
                             if not n.startswith("graph_generator.")))
        source = {n: p.detach().clone() for n, p in core.named_parameters()}
        x = torch.randn(4, 2)
        prediction = core.graph_generator(x)
        loss = (prediction - torch.ones_like(prediction)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(core.graph_generator.weight.grad.norm()), 0.0)
        self.assertTrue(all(p.grad is None for n, p in core.named_parameters()
                            if not n.startswith("graph_generator.")))
        torch.optim.Adam(params, lr=3e-5).step()
        self.assertTrue(torch.equal(source["encoder_like.weight"], core.encoder_like.weight))
        self.assertTrue(torch.equal(source["dynamics.weight"], core.dynamics.weight))
        self.assertFalse(torch.equal(source["graph_generator.weight"], core.graph_generator.weight))

    def test_requires_a_learned_graph_generator(self):
        core = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "learned graph"):
            graph_only_parameters(core)


if __name__ == "__main__":
    unittest.main()
