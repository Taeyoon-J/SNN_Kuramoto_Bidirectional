from __future__ import annotations

import unittest

import numpy as np

from collaborative_test.SW_0182_unique1024_actual_descent import schedule, train


class TrainingContractTests(unittest.TestCase):
    def test_cursor_binds_all_64_unique_exposures_and_proposals(self):
        batches = schedule.expected_batches_1024()
        history = []
        accepted = 0
        for index, indices in enumerate(batches):
            accepted_flag = index % 2 == 0
            accepted += int(accepted_flag)
            before = 1.0 + index / 100.0
            threshold = max(1e-7, 1e-6 * abs(before))
            history.append({"proposal": index + 1, "image_indices": indices,
                "proposal_seed": 171000 + 1 + index, "accepted": accepted_flag,
                "before_loss": before, "after_loss": before - 2 * threshold if accepted_flag else before,
                "acceptance_threshold": threshold, "accepted_update_count": accepted})
        train._validate_cursor(64, accepted, history, seed=1)
        self.assertEqual(set(i for row in history for i in row["image_indices"]), set(range(1024)))
        malformed = [dict(row) for row in history]
        malformed[32]["image_indices"] = history[31]["image_indices"]
        with self.assertRaisesRegex(ValueError, "registered order"):
            train._validate_cursor(64, accepted, malformed, seed=1)

    def test_first32_control_comparison_requires_same_state_and_decision(self):
        row = {"proposal": 1, "proposal_seed": 171001, "image_indices": list(range(16)),
            "accepted": True, "accepted_update_count": 1,
            "start_transaction_sha256": "a" * 64, "proposal_transaction_sha256": "b" * 64,
            "before_loss": 1.0, "after_loss": 0.9, "acceptance_threshold": 1e-6}
        control = dict(row)
        train._validate_prefix_control_row(row, control, index=0)
        near = dict(row, before_loss=1.0000005, after_loss=0.9000004,
                    acceptance_threshold=1.0000005e-6)
        train._validate_prefix_control_row(near, control, index=0)
        with self.assertRaisesRegex(ValueError, "accepted"):
            train._validate_prefix_control_row(dict(row, accepted=False), control, index=0)
        with self.assertRaisesRegex(ValueError, "transaction"):
            train._validate_prefix_control_row(dict(row, proposal_transaction_sha256="c" * 64),
                                               control, index=0)

    def test_bindings_mark_supervised_source_and_full_unique_dataset(self):
        parent = {"source_binding": {"seed": 2}, "source_state_sha256": "a" * 64,
            "base_encoder_sha256": "b" * 64, "initial_model_state_sha256": "c" * 64,
            "optimizer_groups": [{"name": "graph"}]}
        ids = list(range(30000, 31024))
        inputs = {"dataset_manifest_sha256": "d" * 64, "dataset_arrays_sha256": "e" * 64,
                  "ordered_train_ids": ids}
        control = {"manifest_sha256": "f" * 64, "history_sha256": "1" * 64,
                   "score_sha256": train.CONTROL_SCORE_SHA256,
                   "manifest_path": "control/manifest.json", "history_path": "control/history.json",
                   "score_path": "control/evaluation_seedmean_20261016.json"}
        binding = train._make_bindings("product_q_ce", 2, inputs,
                                       {"bindings": parent}, control)
        self.assertEqual(binding["ordered_train_ids"], ids)
        self.assertEqual(binding["control_manifest_sha256"], "f" * 64)
        self.assertTrue(binding["mask_supervised"])
        self.assertFalse(binding["unsupervised_leaderboard_eligible"])

    def test_control_score_resolver_accepts_only_one_registered_exact_hash(self):
        candidates = ("local/evaluation.json", "server/evaluation_seedmean_20261016.json")
        present = {str(train.Path(candidates[0])), str(train.Path(candidates[1]))}
        digests = {str(train.Path(candidates[0])): "wrong", str(train.Path(candidates[1])): "pinned"}
        selected = train._select_control_score(candidates, "pinned",
            is_file=lambda path: str(path) in present,
            sha_file=lambda path: digests[str(path)])
        self.assertEqual(selected, train.Path(candidates[1]))
        with self.assertRaisesRegex(ValueError, "missing, changed, or ambiguous"):
            train._select_control_score(candidates, "pinned",
                is_file=lambda path: str(path) in present,
                sha_file=lambda path: "pinned")


if __name__ == "__main__":
    unittest.main()
