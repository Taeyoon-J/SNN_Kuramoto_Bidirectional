"""Validation tests for frozen-source baseline and bootstrap gates."""
import numpy as np  # Import before Torch on the Windows development host.
from pathlib import Path
import unittest
import uuid
from unittest import mock

import torch

from collaborative_test.SW_0124_temporal_prototype_readout import evaluate


def valid_score(offset=0.0):
    score = {"metrics": {}, "valid_count": {}, "per_image": {}}
    for index, metric in enumerate(evaluate.METRICS):
        values = np.linspace(.1, .9, 320) + offset + index * .01
        score["metrics"][metric] = float(values.mean())
        score["valid_count"][metric] = 320
        score["per_image"][metric] = values.tolist()
    return score


class EvaluationValidationTests(unittest.TestCase):
    def test_static_gamma_contract_matches_core_api(self):
        gamma = torch.zeros((8, 8, 256))
        self.assertIs(evaluate._validate_gamma_batch(gamma), gamma)
        with self.assertRaisesRegex(ValueError, "static finite"):
            evaluate._validate_gamma_batch(torch.zeros((8, 8, 256, 1024)))

    def test_source_qcc_per_image_guard_accepts_exact_and_rejects_shift(self):
        source = valid_score()
        evaluate._assert_reproduces_source(source, {"score": {"per_image": source["per_image"]}})
        shifted = valid_score(1e-4)
        with self.assertRaisesRegex(AssertionError, "reproduction failed"):
            evaluate._assert_reproduces_source(
                shifted, {"score": {"per_image": source["per_image"]}})

    def test_three_seed_bootstrap_uses_shared_image_sampling(self):
        source = np.zeros((3, 320), dtype=np.float64)
        candidate = np.full((3, 320), .02, dtype=np.float64)
        record = evaluate.paired_bootstrap(candidate, source, draws=1000, seed=124)
        self.assertTrue(record["common_image_indices_across_seeds"])
        self.assertAlmostEqual(record["mean_delta"], .02, places=14)
        self.assertGreater(record["lower95"], 0)
        with self.assertRaisesRegex(ValueError, "three paired seeds"):
            evaluate.paired_bootstrap(candidate[:2], source[:2])

    def test_predict_seed_exercises_real_readout_with_mocked_assets(self):
        class FakeCore:
            num_time_steps = 1024
            spike_per_component = True
            kuramoto = type("Kuramoto", (), {"spike_pulse_gain": None})()
            graph_generator = type("Graph", (), {"uses_feedback": False, "eval": lambda self: self})()

            def load_state_dict(self, *_args, **_kwargs):
                return None

            def eval(self):
                return self

            def __call__(self, gamma, **_kwargs):
                self.last_component_spikes = (
                    torch.rand(gamma.shape[0], 4, 256, 1024, device=gamma.device) > .8
                ).to(gamma.dtype)
                return None, None, None, None

        fake_reference = {"source": Path("source.pt"), "source_manifest": Path("source.json"),
                          "manifest": {}, "evaluation": Path("eval.json"),
                          "evaluation_sha256": "e" * 64, "metrics": valid_score()["metrics"],
                          "score": valid_score()}
        fake_cal = {"status": "passed", "seed": 0, "spike_rms_sha256": "r" * 64}
        with (mock.patch.object(evaluate, "source_reference", return_value=fake_reference),
              mock.patch.object(evaluate.sw123, "load_calibration",
                                return_value=(Path("cal.json"), fake_cal, None, torch.ones(4))),
              mock.patch.object(evaluate.base, "validate_gamma_cache",
                                return_value=(torch.zeros(320, 8, 256), {"image_ids": [1320, 1639]})),
              mock.patch.object(evaluate.base, "make_core", return_value=FakeCore()),
              mock.patch.object(evaluate.torch, "load", return_value={}),
              mock.patch.object(evaluate, "sha", return_value="a" * 64),
              mock.patch.object(evaluate, "spike_synchrony_affinity",
                                side_effect=lambda activity, *_args, **_kw: torch.eye(256).expand(
                                    activity.shape[0], -1, -1).to(activity.device)),
              mock.patch.object(evaluate, "production_partition",
                                side_effect=lambda activity, _components, **_kw: (
                                    torch.zeros(activity.shape[0], 256, dtype=torch.long), None))):
            prediction, _reference = evaluate._predict_seed(0, "cpu", max_batches=1)
        self.assertEqual(prediction["count"], 8)
        self.assertEqual(prediction["image_ids"], [1320, 1327])
        self.assertEqual(tuple(prediction["prototype_labels"].shape), (8, 16, 16))
        self.assertEqual(tuple(prediction["qcc_labels"].shape), (8, 16, 16))
        self.assertFalse(prediction["ground_truth_used_for_prediction"])

    def test_preflight_creates_parent_and_persists_mocked_real_path_result(self):
        fake = {"experiment": "SW0124", "seed": 1, "image_ids": [1320, 1327], "count": 8,
                "prototype_labels": torch.zeros(8, 16, 16, dtype=torch.long),
                "qcc_labels": torch.zeros(8, 16, 16, dtype=torch.long),
                "anchors_per_image": [[0] for _ in range(8)],
                "ground_truth_used_for_prediction": False,
                "source_core_sha256": "a" * 64, "source_evaluation_sha256": "b" * 64,
                "calibration_sha256": "c" * 64, "gamma_sha256": "d" * 64,
                "gamma_manifest_sha256": "e" * 64}
        attempt = evaluate.HERE / f".test_preflight_{uuid.uuid4().hex}"
        attempt.mkdir()
        target = attempt / "not-created-yet" / "preflight.json"
        try:
            with mock.patch.object(evaluate, "_predict_seed", return_value=(fake, {})):
                record = evaluate.preflight_seed(1, target, "cpu")
            self.assertEqual(record["status"], "passed")
            self.assertTrue(target.is_file())
        finally:
            if target.exists():
                target.unlink()
            if target.parent.exists():
                target.parent.rmdir()
            attempt.rmdir()


if __name__ == "__main__":
    unittest.main()
