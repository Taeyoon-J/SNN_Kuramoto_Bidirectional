"""CPU contract checks for the SW0167 preflight owner handoff."""
import numpy as np
import unittest

from collaborative_test.SW_0167_structured_relation_energy import preflight_queue as queue


def _proof():
    rows = []
    for index in range(8):
        rows.append({
            "index": index, "image_id": 100 + index,
            "rgb_sha256": f"{index:064x}", "source_file_sha256": f"{index+8:064x}",
            "trace_sha256": f"{index+16:064x}", "context_file_sha256": f"{index+24:064x}",
            "q_sha256": f"{index+32:064x}", "adjacency_sha256": f"{index+40:064x}",
        })
    prepared = {
        "selected_ids": list(range(100, 108)), "source_checkpoint_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64, "source_cache_manifest_sha256": "c" * 64,
        "context_cache_manifest_sha256": "d" * 64, "source_state_sha256": "e" * 64,
        "context_state_sha256": "e" * 64, "gain_receipt_sha256": "f" * 64,
        "rows": rows,
    }
    return queue._source_proof(prepared)


def _report(proof, fingerprint):
    rows = [{"cache_index": i, "image_id": proof["ordered_image_ids"][i],
             "q_sha256": proof["row_bindings"][i]["q_sha256"],
             "source_foreground_exact": True, "source_zero_head_labels_exact": True,
             "source_zero_head_merge_log_exact": True,
             "cpu_b4_source_F_sha256": f"{i:064x}",
             "feature_sha256": f"{i+16:064x}"} for i in range(8)]
    parity = [{"cache_index": i} for i in range(8)]
    arms = {}
    for arm in ("structured_hinge", "local_margin"):
        steps = [{"step": 1, "mean_loss": 1.0, "gradient_norms": {"output.weight": 1.0}},
                 {"step": 2, "mean_loss": 0.5,
                  "gradient_norms": {"output.weight": 0.5, "hidden.weight": 0.25}}]
        arms[arm] = {"initial_state_sha256": "8" * 64,
                     "both_steps_finite": True, "output_changed": True,
                     "hidden_changed_after_second_step": True, "steps": steps}
    return {
        "experiment": "SW0167_split_feature_two_update_preflight", "status": "complete_admitted",
        "optimizer_updates": 2,
        "feature_mode": "rho_from_cached_S_B1; context_Q_and_A_from_registered_original_B4",
        "source_F_mode": "production CPU float32 Q on original TRAIN B4 groups, then production QCC readout",
        "feature_sha256_by_row": [row["feature_sha256"] for row in rows],
        "same_feature_bytes_for_both_arms": True, "validation_data_read": False,
        "implementation_fingerprint_sha256": fingerprint,
        "source_proof": proof, "selected_image_ids": proof["ordered_image_ids"],
        "source_checkpoint_sha256": proof["source_checkpoint_sha256"],
        "source_manifest_sha256": proof["source_manifest_sha256"],
        "source_cache_manifest_sha256": proof["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": proof["context_cache_manifest_sha256"],
        "source_state_sha256": proof["source_state_sha256"],
        "context_state_sha256": proof["context_state_sha256"],
        "rows": rows, "source_foreground_and_zero_residual_parity": parity,
        "arms": arms,
    }


class PreflightQueueTests(unittest.TestCase):
    def test_source_proof_maps_actual_selected_ids_and_every_row_identity(self):
        proof = _proof()
        self.assertEqual(proof["ordered_image_ids"], list(range(100, 108)))
        self.assertEqual([row["index"] for row in proof["row_bindings"]], list(range(8)))
        self.assertEqual(proof["row_bindings"][7]["image_id"], 107)

    def test_report_validator_binds_source_fingerprint_and_both_arm_updates(self):
        proof = _proof(); fingerprint = "9" * 64
        self.assertTrue(queue.validate_report(_report(proof, fingerprint),
                                              source_proof=proof, fingerprint=fingerprint))
        bad = _report(proof, fingerprint)
        bad["rows"][3]["image_id"] += 1
        with self.assertRaises(ValueError):
            queue.validate_report(bad, source_proof=proof, fingerprint=fingerprint)
        bad = _report(proof, fingerprint)
        bad["arms"]["local_margin"]["steps"][1]["gradient_norms"]["hidden.weight"] = 0.0
        with self.assertRaises(ValueError):
            queue.validate_report(bad, source_proof=proof, fingerprint=fingerprint)

    def test_worker_command_is_the_registered_module_and_fixed_report(self):
        command = queue._worker_command(3)
        self.assertEqual(command[1:3], ["-m", "collaborative_test.SW_0167_structured_relation_energy.preflight_queue"])
        self.assertEqual(command[3:6], ["--worker", "--physical-gpu", "3"])
        self.assertEqual(command[-2:], ["--output", str(queue.REPORT)])

    def test_implementation_fingerprint_resolves_all_registered_dependencies(self):
        fingerprint = queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0167_structured_relation_energy/preflight_queue.py", fingerprint)
        self.assertIn("collaborative_test/SW_0167_structured_relation_energy/training_objective.py", fingerprint)
        self.assertIn("collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py", fingerprint)
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))


if __name__ == "__main__":
    unittest.main()
