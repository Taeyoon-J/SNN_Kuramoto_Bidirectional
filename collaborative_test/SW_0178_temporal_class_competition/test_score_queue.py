import unittest

import numpy as np  # NumPy/OpenMP precedes legacy Torch imports.

from collaborative_test.SW_0178_temporal_class_competition import score_queue


class SW178ScoreReceiptContracts(unittest.TestCase):
    def test_prediction_protocol_is_bound_to_arm_and_no_gt_scope(self):
        protocol = {"experiment": "SW0178_temporal_class_competition_fixed320_prediction",
                    "arm": "product_q_ce", "status": "complete",
                    "ground_truth_used_for_prediction": False,
                    "prediction_sha256": "a" * 64}
        state = {"experiment": "SW0178_prediction_queue_v1", "arm": "product_q_ce",
                 "status": "complete", "ground_truth_used_for_prediction": False,
                 "prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64}
        receipt = score_queue._validate_prediction_receipt(
            "product_q_ce", state, prediction_path="predictions.npz",
            protocol_path="protocol.json", prediction_sha256="a" * 64,
            protocol_sha256="b" * 64, protocol_data=protocol)
        self.assertEqual(receipt["state"], state)
        self.assertEqual(receipt["prediction_path"], "predictions.npz")
        bad = dict(protocol, experiment="SW0177_other_experiment")
        with self.assertRaisesRegex(ValueError, "protocol does not match"):
            score_queue._validate_prediction_receipt(
                "product_q_ce", state, prediction_path="predictions.npz",
                protocol_path="protocol.json", prediction_sha256="a" * 64,
                protocol_sha256="b" * 64, protocol_data=bad)


if __name__ == "__main__":
    unittest.main()
