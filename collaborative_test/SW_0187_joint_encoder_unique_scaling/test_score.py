import unittest

import numpy as np

from collaborative_test.SW_0187_joint_encoder_unique_scaling import score


def _row(value):
    vector = np.full(score.COUNT, value, dtype=np.float64)
    return {metric: {"per_image": vector, "mean": float(value)} for metric in score.METRICS}


class ScoreAggregation(unittest.TestCase):
    def test_score_fingerprint_binds_training_and_queue_receipts(self):
        fingerprint = score.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0187_joint_encoder_unique_scaling/train.py", fingerprint)
        self.assertIn("collaborative_test/SW_0187_joint_encoder_unique_scaling/prediction_queue.py", fingerprint)
        self.assertEqual(fingerprint["SW0181_score_sha256"],
            "3f69a74a5729341b5fd11f2bd74f2d51ff245c0839f76053b8733af9523ddd54")

    def _fixture(self):
        values = {
            "candidate1024_standalone": .62, "control512_standalone": .55,
            "candidate1024_conditional": .31, "control512_conditional": .81,
            "SW0165_foreground_induced_standalone": .60,
            "Slot70k_standalone": .58, "SW0181_standalone": .59,
            "SW0165_foreground_induced_conditional": .60,
            "Slot70k_conditional": .58, "SW0181_conditional": .59,
            "SW0186_standalone": .61, "SW0186_conditional": .61,
        }
        return {name: _row(value) for name, value in values.items()}

    def test_actual_aggregation_wiring_uses_standalone_for_primary_gate(self):
        scores = self._fixture()
        indices = np.random.RandomState(score.BOOTSTRAP_SEED).randint(
            0, score.COUNT, size=(score.BOOTSTRAP_REPLICATES, score.COUNT))
        paired, gates = score.aggregate_and_gate(scores, indices)
        self.assertTrue(gates["unique_data_fg_mean_above_control"])
        self.assertTrue(gates["unique_data_fg_paired_ci95_lower_above_zero"])
        self.assertTrue(gates["unique_data_gate_passed"])
        self.assertLess(scores["candidate1024_conditional"]["fg_ari"]["mean"],
            scores["control512_conditional"]["fg_ari"]["mean"])
        self.assertEqual(paired["candidate1024_minus_control512_standalone_fg_ari"]["seed"], 178)
        self.assertEqual(paired["candidate1024_minus_control512_standalone_fg_ari"]["replicates"], 10_000)

    def test_gate_is_not_silently_tightened_to_external_benchmarks(self):
        scores = self._fixture()
        scores["candidate1024_standalone"]["fg_ari"]["per_image"][:] = .57
        scores["candidate1024_standalone"]["fg_ari"]["mean"] = .57
        indices = np.zeros((score.BOOTSTRAP_REPLICATES, score.COUNT), dtype=np.int64)
        _, gates = score.aggregate_and_gate(scores, indices)
        self.assertTrue(gates["unique_data_gate_passed"])
        self.assertFalse(gates["slot_fg_mean_strictly_exceeded"])

    def test_actual_pinned_joint_encoder_reference_is_validated(self):
        reference=score._read_sw0186_score()
        self.assertEqual(reference['sha256'],score.SW0186_SCORE_SHA256)
        self.assertAlmostEqual(reference['payload']['scores']['candidate_standalone']['fg_ari']['mean'],.8346805012913832)

    def test_joint_encoder_reference_is_separate_from_concurrent_data_gate(self):
        scores=self._fixture()
        scores['SW0186_standalone']=_row(.9)
        indices=np.zeros((score.BOOTSTRAP_REPLICATES,score.COUNT),dtype=np.int64)
        _,gates=score.aggregate_and_gate(scores,indices)
        self.assertTrue(gates['unique_data_gate_passed'])
        self.assertFalse(gates['sw0186_fg_mean_strictly_exceeded'])


if __name__ == "__main__": unittest.main()
