from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from collaborative_test.SW_0162_connected_signed_partition import preflight, preflight_queue as queue


class PreflightQueueTests(unittest.TestCase):
    def test_child_command_binds_actual_preflight_and_fixed_no_gt_contract(self):
        argv = queue.command()
        self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0162_connected_signed_partition.preflight"])
        self.assertEqual(argv[3:], ["--device", "cuda:0", "--output", str(preflight.OUTPUT)])
        plan = queue.task_plan()
        self.assertEqual(plan["positions"], list(preflight.POSITIONS))
        self.assertEqual(plan["optimizer_updates"], 0)
        self.assertIs(plan["ground_truth_used"], False)

    def test_report_validator_requires_all_eight_exact_parity_records(self):
        fp = {"preflight.py": "a" * 64}
        image_rows = list(preflight.POSITIONS)
        image_ids = [100 + index for index in range(8)]
        rgb_hashes = [str(index) * 64 for index in range(8)]
        report = Path(__file__).with_name(f".queue-report-fixture-{os.getpid()}.json")
        manifest = report.with_suffix(".manifest.json")
        try:
            manifest.write_text("registered source manifest fixture", encoding="utf-8")
            prepared = {"checkpoint": Path("source.pt"), "source_sha": "c" * 64,
                "manifest_path": manifest, "ordered_ids": list(range(4096)),
                "selected_rows": image_rows, "selected_ids": image_ids,
                "rgb_sha": "d" * 64, "cache_size_bytes": 123,
                "cache_mtime_ns": 456, "assets": {"train_manifest_sha256": "e" * 64,
                    "encoder_source_sha256": "f" * 64, "feature_stats_sha256": "a" * 64,
                    "feature_stats_mode": "standardize", "feature_clip": 3.0,
                    "scope": "registered TRAIN RGB only; no validation images or masks opened"}}
            source = preflight.source_report_fields(prepared, "b" * 64, "b" * 64)
            source["rgb_image_sha256"] = rgb_hashes
            record = {"experiment": "SW0162_connected_signed_partition",
                "status": "source8_preflight_complete", "seed": 1,
                "implementation": fp, "ground_truth_used": False, "optimizer_updates": 0,
                "source_operator": {"optimizer_updates": 0, "arm": "T", "steps": 1024,
                    "settle": 512, "live_tail": 0,
                    "A_origin": "untouched GraphVectorKuramoto.prepare_coupling tuple[0]",
                    "backend": "factorized"},
                "source": source,
                "images": [{"pool_row": row, "image_id": image_id, "rgb_sha256": rgb_sha,
                    "trace_parity": True, "Q_parity": True, "QCC_parity": True,
                    "actual_A_symmetric_positive_finite": True,
                    "deterministic_solver_replay": True,
                    "trace_sha256": {field: "1" * 64 for field in preflight.TRACE_FIELDS},
                    "Q_sha256": "2" * 64, "actual_A_sha256": "3" * 64,
                    "qcc_labels_sha256": "4" * 64, "gain": 2.0}
                    for row, image_id, rgb_sha in zip(image_rows, image_ids, rgb_hashes)]}
            expected_source = {key: value for key, value in source.items()
                               if key != "source_state_sha256_before" and key != "source_state_sha256_after"
                               and key != "rgb_asset_validation"}
            expected_source["rgb_image_sha256"] = rgb_hashes
            report.write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(queue.validate_report(report, expected_fingerprint=fp,
                                                  expected_source=expected_source))
            record["images"][7]["QCC_parity"] = False
            report.write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(queue.validate_report(report, expected_fingerprint=fp))
        finally:
            report.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)

    def test_fingerprint_and_child_path_are_local_registered_files(self):
        fp = queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0162_connected_signed_partition/preflight_queue.py", fp)
        self.assertIn("collaborative_test/SW_0162_connected_signed_partition/connected_signed.py", fp)
        self.assertTrue(all(len(value) == 64 for value in fp.values()))


if __name__ == "__main__":
    unittest.main()
