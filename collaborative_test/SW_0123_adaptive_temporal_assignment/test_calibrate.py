import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import calibrate
from calibrate import EligibilityFailure, activity_summary, calibration_arrays


class CalibrationMathTests(unittest.TestCase):
    def test_kappa_uses_settled_margin_quantile_and_rms_uses_same_traces(self):
        component_out = torch.zeros(64, 4, 256, 32)
        component_spikes = torch.zeros_like(component_out)
        margins = (0.10, 0.20, 0.30, 0.40)
        rms_values = (0.0, 0.5, 1.0, 2.0)
        for d in range(4):
            component_out[:, d] = 0.06 + margins[d]
            component_spikes[:, d] = rms_values[d]
        kappa, rms = calibration_arrays(component_out, component_spikes)
        torch.testing.assert_close(kappa, torch.tensor(margins) * 4, rtol=0, atol=1e-6)
        torch.testing.assert_close(rms, torch.tensor([1.0, 0.5, 1.0, 2.0]), rtol=0, atol=0)

    def test_activity_guard_accepts_mixed_submaximal_binary_events(self):
        events = torch.zeros(64, 4, 256, 32)
        events[..., :8] = 1
        summary = activity_summary(events)
        self.assertEqual(summary["occupancy_by_component"], [0.25] * 4)
        self.assertEqual(summary["mixed_image_patch_fraction_by_component"], [1.0] * 4)

    def test_activity_guard_fails_without_tuning_when_event_rate_is_extreme(self):
        events = torch.ones(64, 4, 256, 32)
        with self.assertRaises(EligibilityFailure):
            activity_summary(events)

    def test_rejects_fold_or_horizon_mismatch(self):
        with self.assertRaises(ValueError):
            calibration_arrays(torch.zeros(64, 16, 256, 32), torch.zeros(64, 16, 256, 32))
        with self.assertRaises(ValueError):
            activity_summary(torch.zeros(64, 4, 256, 31))

    def test_four_batch_forward_aggregates_legacy_traces_and_adaptive_events(self):
        gamma = torch.zeros(64, 8, 256)
        rows = np.arange(64, dtype=np.int64)
        core = SimpleNamespace(membrane_layer=SimpleNamespace(event_history=None))
        calls = {"n": 0}

        def fake_forward(model, batch, _loss, _settle, _primary, _reduction):
            index = calls["n"]
            calls["n"] += 1
            self.assertEqual(tuple(batch.shape), (16, 8, 256))
            model.last_component_out = torch.full((16, 4, 256, 64), float(index))
            model.last_component_spikes = torch.full((16, 4, 256, 64), float(index + 1))
            if model.membrane_layer.event_history is not None:
                model.membrane_layer.event_history = [
                    torch.full((64, 256), float(t % 2)) for t in range(64)]

        with patch.object(calibrate, "_forward_with_plv", fake_forward), \
                patch.object(calibrate.base, "criterion", return_value=object()):
            out, spikes = calibrate._forward_four(core, gamma, rows, "cpu", adaptive=False)
            self.assertEqual(tuple(out.shape), (64, 4, 256, 32))
            self.assertEqual(tuple(spikes.shape), (64, 4, 256, 32))
            self.assertTrue(torch.equal(out[:16], torch.zeros_like(out[:16])))
            self.assertTrue(torch.equal(out[16:32], torch.ones_like(out[16:32])))

            core.membrane_layer.event_history = []
            calls["n"] = 0
            events = calibrate._forward_four(core, gamma, rows, "cpu", adaptive=True)
            self.assertEqual(tuple(events.shape), (64, 4, 256, 32))
            self.assertTrue(torch.equal(events[0, 0, 0], torch.tensor([float(t % 2) for t in range(32, 64)])))

    def test_numpy_training_order_comparison(self):
        self.assertTrue(calibrate.same_ordered_indices(np.arange(4), np.arange(4)))
        self.assertFalse(calibrate.same_ordered_indices(np.arange(4), np.arange(4)[::-1]))


if __name__ == "__main__":
    unittest.main()
