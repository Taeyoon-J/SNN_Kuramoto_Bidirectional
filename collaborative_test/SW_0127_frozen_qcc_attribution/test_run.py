import unittest
from pathlib import Path
from unittest import mock
import uuid

import numpy as np
import torch

from collaborative_test.SW_0127_frozen_qcc_attribution import run
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def _prediction_fixture():
    labels = torch.zeros((16, 16), dtype=torch.int64)
    labels[:, :4] = 1
    labels[4:12, 7:10] = 2
    labels[12:, 12:] = 3
    target = torch.empty((128, 128), dtype=torch.int64)
    for row in range(16):
        for col in range(16):
            target[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] = labels[row, col]
    target = clevr_mask_patch(target.unsqueeze(0), 8)["patch_labels"]
    target = target.expand(320, -1, -1).clone()
    return labels, target


class FrozenQCCAttributionTests(unittest.TestCase):
    def _fixture_dir(self):
        root = run.ROOT / ("sw127_test_" + uuid.uuid4().hex[:10])
        root.mkdir()
        return root

    @staticmethod
    def _remove_fixture(root, output):
        for filename in ("frozen_predictions.pt", "prediction_manifest.json"):
            path = output / filename
            if path.exists():
                path.unlink()
        if output.exists():
            output.rmdir()
        root.rmdir()

    def test_score_waits_for_complete_hash_verified_four_arm_predictions(self):
        labels, target = _prediction_fixture()
        expected = run.sw118._score_standard(labels.unsqueeze(0).expand(320, -1, -1).clone(), target)
        source_values = {metric: expected[metric]["per_image"] for metric in run.METRICS}
        temp = self._fixture_dir()
        output = temp / "frozen"
        try:
            output.mkdir()
            predictions = {arm: labels.unsqueeze(0).expand(320, -1, -1).clone()
                           for arm in run.ARMS}
            provenance = {arm: {"ground_truth_used_for_prediction": False} for arm in run.ARMS}
            run._commit_predictions(output, predictions, provenance)
            loaded_after_commit = []

            def load_gt():
                self.assertTrue((output / "prediction_manifest.json").is_file())
                payload = torch.load(output / "frozen_predictions.pt", map_location="cpu",
                                     weights_only=True)
                self.assertEqual(set(payload["predictions"]), set(run.ARMS))
                loaded_after_commit.append(True)
                return target

            result = run._score_committed_predictions(output, load_gt, source_values)
            self.assertEqual(loaded_after_commit, [True])
            self.assertTrue(result["source97_per_image_reproduction"]["passed"])
            self.assertEqual(result["prediction_sha256"], run.sha256_file(output / "frozen_predictions.pt"))
        finally:
            self._remove_fixture(temp, output)

    def test_hash_mismatch_refuses_ground_truth_loader(self):
        labels, target = _prediction_fixture()
        source_values = {metric: [0.0] * 320 for metric in run.METRICS}
        temp = self._fixture_dir()
        output = temp / "frozen"
        try:
            output.mkdir()
            predictions = {arm: labels.unsqueeze(0).expand(320, -1, -1).clone()
                           for arm in run.ARMS}
            run._commit_predictions(output, predictions, {arm: {} for arm in run.ARMS})
            with (output / "frozen_predictions.pt").open("ab") as stream:
                stream.write(b"tamper")
            called = []
            with self.assertRaisesRegex(ValueError, "GT access refused"):
                run._score_committed_predictions(output, lambda: called.append(True) or target,
                                                 source_values)
            self.assertEqual(called, [])
        finally:
            self._remove_fixture(temp, output)

    def test_qcc_readout_uses_actual_components_and_fixed_registered_arguments(self):
        components = torch.zeros((2, 4, 256, 1024), dtype=torch.float32)
        captured = {}

        def fake_classifier(activity, **kwargs):
            captured["activity"] = activity
            captured.update(kwargs)
            return [[(0, 1)], []]

        with mock.patch.object(run, "spike_synchrony_components", side_effect=fake_classifier):
            labels = run._qcc_labels(components)
        self.assertEqual(tuple(labels.shape), (2, 16, 16))
        self.assertEqual(captured["foreground_threshold"], 0.15)
        self.assertEqual(captured["synchrony_threshold"], 0.50)
        self.assertEqual(captured["min_group_size"], 2)
        self.assertEqual(captured["background"], "largest_component")
        self.assertEqual(captured["affinity_mode"], "spike")
        self.assertEqual(captured["settle"], 512)
        self.assertIsNone(captured["synchrony_quantile"])
        self.assertIsNone(captured["target_foreground"])
        self.assertIsNone(captured["spatial_sigma"])
        self.assertTrue(torch.equal(captured["activity"], components.mean(dim=1)))

    def test_adapter_smoke_requires_actual_gate_times_event_spikes(self):
        core = mock.Mock()
        core.membrane_layer.component_event_trace.return_value = torch.ones(2, 4, 256, 1024)
        core.membrane_layer.component_gate_trace.return_value = torch.full((2, 4, 256, 1024), 0.25)
        trace = {"component_spikes": torch.full((2, 4, 256, 1024), 0.25)}
        self.assertTrue(run._assert_adapter_spikes(core, trace["component_spikes"], 2))
        trace["component_spikes"][0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(AssertionError, "gate-times-binary-event"):
            run._assert_adapter_spikes(core, trace["component_spikes"], 2)

    def test_source_reproduction_rejects_per_image_mismatch(self):
        scores = {metric: {"per_image": [0.5] * 320} for metric in run.METRICS}
        expected = {metric: [0.5] * 320 for metric in run.METRICS}
        self.assertTrue(run._compare_source_score(scores, expected)["passed"])
        expected["fg_ari"][0] += 1e-8
        result = run._compare_source_score(scores, expected)
        self.assertFalse(result["passed"])
        self.assertFalse(result["metrics"]["fg_ari"]["passed"])
        scores["fg_ari"]["per_image"] = [None] * 320
        invalid = run._compare_source_score(scores, {metric: [0.5] * 320 for metric in run.METRICS})
        self.assertFalse(invalid["passed"])
        self.assertIsNone(invalid["metrics"]["fg_ari"]["max_abs_diff"])


if __name__ == "__main__":
    unittest.main()
