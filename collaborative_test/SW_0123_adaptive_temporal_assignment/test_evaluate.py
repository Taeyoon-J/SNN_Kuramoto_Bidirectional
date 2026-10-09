import unittest

import torch
from torch import nn

from evaluate import _predict_batch, _serialize_score


class Gate(nn.Module):
    def forward(self, h, gate):
        return h, gate


class HorizonCore(nn.Module):
    def __init__(self, batch, horizon):
        super().__init__()
        self.spike_per_component = True
        self.membrane_layer = Gate()
        self.batch = batch
        self.horizon = horizon
        self.num_time_steps = horizon
        self.patch_axis = torch.arange(256)
        self.actual_gates = []

    def forward(self, gamma, return_core_out=True, return_theta=True):
        batch = gamma.shape[0]
        device = gamma.device
        patch = self.patch_axis.to(device).view(1, 256)
        emitted = []
        for t in range(self.horizon):
            gate = ((patch + t) % 3 == 0).to(gamma.dtype).expand(batch, -1)
            folded = gate.repeat_interleave(4, dim=0)
            self.actual_gates.append(gate)
            _mem, spike = self.membrane_layer(torch.zeros_like(folded), folded)
            emitted.append(spike)
        self.last_component_spikes = torch.stack(emitted, dim=-1).reshape(batch, 4, 256, self.horizon)
        dummy = self.last_component_spikes.mean(dim=1)
        theta = torch.zeros(batch, self.horizon, 256, 4, device=device)
        return [], dummy, dummy, theta


class CaptureAssignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, traces):
        self.seen = traces.detach().clone()
        logits = torch.zeros(traces.shape[0], 256, 11, device=traces.device)
        logits[:, :, 1] = traces.mean(dim=(1, 3))
        return torch.softmax(logits, dim=-1), None, None


class EvaluationPathTests(unittest.TestCase):
    def test_gate_only_endpoint_uses_actual_folded_gate_and_variable_horizon(self):
        batch, horizon, settle = 2, 12, 4
        core = HorizonCore(batch, horizon)
        head = CaptureAssignment()
        gamma = torch.zeros(batch, 8, 256)
        primary, qcc, components = _predict_batch(core, head, gamma, "gate_only_control", settle)
        expected = torch.stack(core.actual_gates, dim=-1)[:, None].expand(-1, 4, -1, -1)[..., settle:]
        self.assertTrue(torch.equal(head.seen, expected))
        self.assertEqual(tuple(components.shape), (batch, 4, 256, horizon))
        self.assertEqual(tuple(primary.shape), (batch, 16, 16))
        self.assertEqual(tuple(qcc.shape), (batch, 16, 16))

    def test_assignment_arm_uses_actual_component_spikes_at_eval_horizon(self):
        batch, horizon, settle = 1, 10, 3
        core = HorizonCore(batch, horizon)
        head = CaptureAssignment()
        gamma = torch.zeros(batch, 8, 256)
        primary, qcc, components = _predict_batch(core, head, gamma, "adaptive_full", settle)
        self.assertTrue(torch.equal(head.seen, components[..., settle:]))
        self.assertEqual(tuple(primary.shape), (1, 16, 16))
        self.assertEqual(tuple(qcc.shape), (1, 16, 16))

    def test_serialized_score_preserves_three_metrics_and_valid_counts(self):
        prediction = torch.zeros(3, 16, 16, dtype=torch.int64)
        target = prediction.clone()
        target[:, :2, :2] = 1
        score = _serialize_score(prediction, target)
        self.assertEqual(set(score["metrics"]), {"fg_ari", "foreground_iou", "matched_object_iou"})
        self.assertEqual(score["valid_count"]["foreground_iou"], 3)
        self.assertEqual(len(score["per_image"]["fg_ari"]), 3)


if __name__ == "__main__":
    unittest.main()
