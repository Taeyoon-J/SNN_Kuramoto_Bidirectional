import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0152_source_preserving_partition_equivariance import preflight, preflight_queue as queue


def report_fixture():
    ids = list(range(10000, 14096))
    names = ["spatial_residual", "graph", "native_core", "integration_scalars"]
    before = {n: ("1" if i % 2 else "2") * 64 for i, n in enumerate(names)}
    after = {n: ("3" if i % 2 else "4") * 64 for i, n in enumerate(names)}
    history = [{"mean_loss": 0.75, "family_gradient_norms": {n: 0.1 for n in names}}
               for _ in range(2)]
    return {"experiment": preflight.EXPERIMENT, "stage": "source1_two_update_preflight",
        "status": "source_preflight_complete", "seed": 1, "training_admission": False,
        "ground_truth_used": False, "masks_read": False,
        "implementation_fingerprint": preflight.implementation_fingerprint(),
        "source": {"training_ids": ids, "training_ids_sha256": preflight.ids_sha(ids),
            "checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64,
            "source_rgb_asset_validation": {"train_cache_sha256": "c" * 64},
            "source_encoder_sha256": "e" * 64},
        "saved_source_parity": {"path": str(queue.PARITY), "sha256": "d" * 64,
            "status": "native32_zero_adapter_parity_passed",
            "source1_actual_source_exact_T1024_settle512": True},
        "registered": {"logical_batch": 16, "microbatch": 4, "preflight_updates": 2,
            "total_steps": 1024, "settle": 512, "live_tail": 64},
        "within_color_permutation_diagnostic": {"training_ids": ids[:64],
            "baseline_alignment_js_mean": 0.0, "within_color_permuted_alignment_js_mean": 0.1,
            "eligible_patch_count": 1000, "changed_hard_assignment_count": 100,
            "moved_fraction": 0.1, "ground_truth_used": False, "diagnostic_only": True},
        "arms": {arm: {"status": "two_actual_updates_complete", "image_ids": ids[:32],
            "source_zero_head_parity_exact": True, "parameter_sha256_before": before,
            "parameter_sha256_after": after, "changed_parameter_families": {n: True for n in names},
            "frozen_base_encoder_sha256_before_after": "e" * 64, "history": history}
            for arm in ("equivariance", "mass_control")}}


def add_actual_diagnostic_contract(report):
    diag = report["within_color_permutation_diagnostic"]
    diag["augmentation_records"] = [
        {"image_id": image_id, "source_seed": 1,
         "gain": [1.0, 0.9, 1.1], "offset": [0.0, -0.05, 0.05]}
        for image_id in diag["training_ids"]]
    report["registered"]["optimizer_learning_rates"] = {
        "spatial_residual": 3e-6, "graph": 3e-5,
        "native_core": 3e-5, "integration_scalars": 1e-3}
    return report


class PreflightQueueTests(unittest.TestCase):
    def setUp(self):
        self.report = add_actual_diagnostic_contract(report_fixture())
        self.args = {"expected_ids": self.report["source"]["training_ids"],
            "expected_source": {"checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64},
            "expected_assets": {"train_cache_sha256": "c" * 64},
            "expected_parity_sha": "d" * 64,
            "expected_fingerprint": self.report["implementation_fingerprint"]}

    def test_actual_fingerprint_and_task_plan_bind_one_preflight_only(self):
        fingerprint = queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0152_source_preserving_partition_equivariance/preflight.py", fingerprint)
        tasks = queue.task_plan()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["stage"], "preflight")
        self.assertEqual(tasks[0]["depends_on"], [])
        self.assertNotIn("train", json.dumps(tasks))
        argv = queue.command(tasks[0], "cuda:2")
        self.assertIn("--device", argv)
        self.assertIn("cuda:2", argv)
        self.assertTrue(any("SW_0152_source_preserving_partition_equivariance.preflight" in x for x in argv))

    def test_report_with_actual_registered_contract_passes(self):
        self.assertTrue(queue.validate_report(self.report, **self.args))

    def test_negative_color_permutation_delta_is_preserved_but_not_gate_passed(self):
        self.report["within_color_permutation_diagnostic"]["within_color_permuted_alignment_js_mean"] = -0.1
        self.assertTrue(queue.validate_report(self.report, **self.args))
        delta = (self.report["within_color_permutation_diagnostic"]["within_color_permuted_alignment_js_mean"]
                 - self.report["within_color_permutation_diagnostic"]["baseline_alignment_js_mean"])
        self.assertLess(delta, 0)

    def test_lineage_and_exact_image_order_are_fail_closed(self):
        bad = json.loads(json.dumps(self.report)); bad["source"]["training_ids"][0] += 1
        self.assertFalse(queue.validate_report(bad, **self.args))
        bad = json.loads(json.dumps(self.report)); bad["source"]["manifest_sha256"] = "f" * 64
        self.assertFalse(queue.validate_report(bad, **self.args))
        bad = json.loads(json.dumps(self.report)); bad["implementation_fingerprint"] = {}
        self.assertFalse(queue.validate_report(bad, **self.args))

    def test_selected_family_gradient_and_parameter_updates_are_required(self):
        bad = json.loads(json.dumps(self.report))
        bad["arms"]["equivariance"]["history"][1]["family_gradient_norms"]["native_core"] = 0.0
        self.assertFalse(queue.validate_report(bad, **self.args))
        bad = json.loads(json.dumps(self.report))
        bad["arms"]["equivariance"]["changed_parameter_families"]["graph"] = False
        self.assertFalse(queue.validate_report(bad, **self.args))

    def test_learning_rates_and_per_image_augmentation_provenance_are_bound(self):
        bad = json.loads(json.dumps(self.report))
        bad["registered"]["optimizer_learning_rates"]["native_core"] = 1e-4
        self.assertFalse(queue.validate_report(bad, **self.args))
        bad = json.loads(json.dumps(self.report))
        bad["within_color_permutation_diagnostic"]["augmentation_records"][17]["image_id"] += 1
        self.assertFalse(queue.validate_report(bad, **self.args))
        bad = json.loads(json.dumps(self.report))
        bad["within_color_permutation_diagnostic"]["augmentation_records"][0]["gain"][0] = 2.0
        self.assertFalse(queue.validate_report(bad, **self.args))

    def test_failed_child_and_foreign_cleanup_use_audited_owner_helpers(self):
        self.assertTrue(callable(queue.owner._stop_owned))
        proc = object()
        original = queue.owner._stop_owned
        calls = []
        try:
            queue.owner._stop_owned = lambda child: calls.append(child)
            queue.stop_owned_child(proc)
        finally:
            queue.owner._stop_owned = original
        self.assertEqual(calls, [proc])
        failure = queue.child_failure_record(17, Path("missing-preflight-output.json"))
        self.assertEqual(failure, {"status": "failed_child", "returncode": 17,
                                   "output_sha256": None})

    def test_dry_run_is_serialized_and_never_adds_a_training_stage(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(queue.main(["--dry-run"]), 0)
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["training_stages"], 0)
        self.assertEqual(len(plan["tasks"]), 1)


if __name__ == "__main__":
    unittest.main()
