"""CPU checks for SW0186 paired fixed320 aggregation and registered gates."""
from __future__ import annotations

import unittest

import numpy as np

from collaborative_test.SW_0186_full_encoder_qce import score


class ScoreMathTests(unittest.TestCase):
    def test_source_encoder_learning_is_required_before_score_GT_access(self):
        manifest=dict(source_base_trainable=True,source_base_encoder_changed=True,
            source_base_encoder_initial_sha256='a'*64,source_base_encoder_final_sha256='b'*64,
            accepted_updates=3,optimizer_step_summary={'source_base':dict(
                unused_parameter_count=0,updated_parameter_count=2,steps=[3,3])})
        score._assert_source_encoder_learning(manifest)
        manifest['source_base_encoder_changed']=False
        with self.assertRaises(ValueError):score._assert_source_encoder_learning(manifest)
        manifest['source_base_encoder_changed']=True
        manifest['optimizer_step_summary']['source_base']['steps']=[0,0]
        with self.assertRaises(ValueError):score._assert_source_encoder_learning(manifest)

    def test_score_cli_help_is_wired(self):
        with self.assertRaises(SystemExit) as help_exit:
            score.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)

    def test_frozen_prediction_check_binds_both_archive_and_protocol(self):
        frozen = {"prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64}
        score._assert_prediction_unchanged(1, dict(frozen), frozen)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            score._assert_prediction_unchanged(1,
                {"prediction_sha256": "a" * 64, "protocol_sha256": "c" * 64}, frozen)

    def test_sw0181_comparison_resolves_only_registered_pinned_score_bytes(self):
        candidates = ("local/score.json", "server/evaluation_seedmean_20261016.json")
        hashes = {str(score.Path(candidates[0])): "wrong",
                  str(score.Path(candidates[1])): score.SW0181_SCORE_SHA256}
        selected = score._select_sw0181_score(candidates,
            is_file=lambda path: True, sha_file=lambda path: hashes[str(path)])
        self.assertEqual(selected, score.Path(candidates[1]))
        with self.assertRaisesRegex(ValueError, "missing, changed, or ambiguous"):
            score._select_sw0181_score(candidates, is_file=lambda path: True,
                sha_file=lambda _path: score.SW0181_SCORE_SHA256)

    def test_seed_average_is_paired_by_image_and_requires_all_three_seeds(self):
        vectors = {0: np.arange(score.COUNT, dtype=np.float64),
            1: np.arange(score.COUNT, dtype=np.float64) + 2,
            2: np.arange(score.COUNT, dtype=np.float64) + 4}
        result = score._average_seed(vectors)
        np.testing.assert_array_equal(result, np.arange(score.COUNT, dtype=np.float64) + 2)
        with self.assertRaises(ValueError): score._average_seed({0: vectors[0], 1: vectors[1]})

    def test_bootstrap_is_fixed_rng_and_resamples_shared_images(self):
        candidate = np.linspace(0.2, 0.8, score.COUNT)
        baseline = candidate - 0.1
        indices = np.random.RandomState(score.BOOTSTRAP_SEED).randint(
            0, score.COUNT, size=(score.BOOTSTRAP_REPLICATES, score.COUNT))
        left = score._bootstrap(candidate, baseline, indices)
        right = score._bootstrap(candidate, baseline, indices)
        self.assertEqual(left, right)
        self.assertEqual(left["resampling_unit"], "paired_image_after_three_seed_average")
        self.assertEqual(left["seed"], 178)

    def test_primary_gate_requires_ci_and_both_slot_iou_gates(self):
        scores = {"candidate_standalone": {"fg_ari": {"mean": 0.9},
            "foreground_iou": {"mean": 0.4}, "matched_object_iou": {"mean": 0.4}},
            "SW0165_foreground_induced_standalone": {"fg_ari": {"mean": score.SW165_FG_BASELINE}},
            "SW0181_standalone": {"fg_ari": {"mean": 0.8}},
            "Slot70k_standalone": {"fg_ari": {"mean": score.SLOT_FG_BASELINE},
                "foreground_iou": {"mean": score.SLOT_FOREGROUND_IOU_BASELINE},
                "matched_object_iou": {"mean": score.SLOT_MATCHED_IOU_BASELINE}}}
        paired = {"standalone_candidate_minus_SW0165_fg_ari": {"ci95": [0.01, 0.2]},
            "standalone_candidate_minus_Slot70k_fg_ari": {"ci95": [0.02, 0.3]},
            "standalone_candidate_minus_SW0181_fg_ari": {"ci95": [0.03, 0.3]}}
        self.assertTrue(score._gate(scores, paired)["primary_gate_passed"])
        # The registered primary gate uses the SW181 paired CI. Other CIs
        # remain reported diagnostics rather than extra unregistered gates.
        paired["standalone_candidate_minus_SW0165_fg_ari"]["ci95"][0] = -0.01
        paired["standalone_candidate_minus_Slot70k_fg_ari"]["ci95"][0] = -0.01
        self.assertTrue(score._gate(scores, paired)["primary_gate_passed"])
        paired["standalone_candidate_minus_SW0181_fg_ari"]["ci95"][0] = 0.0
        self.assertFalse(score._gate(scores, paired)["primary_gate_passed"])
        paired["standalone_candidate_minus_SW0181_fg_ari"]["ci95"][0] = 0.03
        scores["candidate_standalone"]["matched_object_iou"]["mean"] = 0.1
        self.assertFalse(score._gate(scores, paired)["primary_gate_passed"])

    def test_fixed320_aggregation_wires_sw0181_ci_into_primary_gate(self):
        # Exercise the same aggregation adapter used by score_all, including its
        # endpoint/group key construction and all three registered metrics.
        scores = {}
        levels = {
            "candidate": {"fg_ari": 0.90, "foreground_iou": 0.50, "matched_object_iou": 0.50},
            "SW0165_foreground_induced": {"fg_ari": score.SW165_FG_BASELINE,
                "foreground_iou": 0.40, "matched_object_iou": 0.40},
            "Slot70k": {"fg_ari": score.SLOT_FG_BASELINE,
                "foreground_iou": score.SLOT_FOREGROUND_IOU_BASELINE,
                "matched_object_iou": score.SLOT_MATCHED_IOU_BASELINE},
            "SW0181": {"fg_ari": 0.80, "foreground_iou": 0.30, "matched_object_iou": 0.30}}
        for endpoint in score.ENDPOINTS:
            for group, metrics in levels.items():
                scores[f"{group}_{endpoint}"] = {
                    name: {"per_image": np.full(score.COUNT, value, dtype=np.float64), "mean": value}
                    for name, value in metrics.items()}
        indices = np.random.RandomState(score.BOOTSTRAP_SEED).randint(
            0, score.COUNT, size=(score.BOOTSTRAP_REPLICATES, score.COUNT))
        paired, gate = score._aggregate_paired_and_gate(scores, indices)
        self.assertIn("standalone_candidate_minus_SW0181", paired)
        self.assertTrue(gate["fg_paired_ci95_lower_above_zero_vs_sw0181"])
        self.assertTrue(gate["primary_gate_passed"])
        scores["candidate_standalone"]["fg_ari"]["per_image"][:] = 0.80
        _, failed_gate = score._aggregate_paired_and_gate(scores, indices)
        self.assertFalse(failed_gate["fg_paired_ci95_lower_above_zero_vs_sw0181"])
        self.assertFalse(failed_gate["primary_gate_passed"])


if __name__ == "__main__": unittest.main()
