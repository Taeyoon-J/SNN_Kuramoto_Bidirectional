from __future__ import annotations

import numpy as np
import json
import unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0159_one_shot_event_regime import train, training_queue as queue


class SW0159TrainingQueueTests(unittest.TestCase):
    def test_registered_arm_routes_to_distinct_outputs_and_fixed_recipe(self):
        control = queue.task_plan("control")
        candidate = queue.task_plan("candidate")
        self.assertNotEqual(control["output"], candidate["output"])
        self.assertEqual(control["updates"], train.UPDATES)
        self.assertEqual(control["logical_batch"], train.BATCH)
        self.assertEqual(control["microbatch"], train.MICRO)
        self.assertEqual(control["ordered_TRAIN_images"], 2048)
        self.assertIn("--arm", queue.command("candidate"))
        self.assertEqual(queue.command("candidate")[queue.command("candidate").index("--arm") + 1],
                         "candidate")
        with self.assertRaisesRegex(ValueError, "unregistered"):
            queue.task_plan("source")

    def test_upstream_binding_contains_real_admission_lineage_shape(self):
        source = {"source_sha256": "a" * 64, "image_ids": list(range(4096))}
        assets = {"train_cache_sha256": "b" * 64}
        admitted = {"training_admission": True, "implementation_fingerprint": {"preflight": "c" * 64}}
        with mock.patch.object(train, "validate_preflight_admission", return_value=admitted), \
             mock.patch.object(train.sw155, "source_contract", return_value=source), \
             mock.patch.object(train.sw130, "validate_rgb_assets", return_value=assets), \
             mock.patch.object(queue, "sha256_file", return_value="d" * 64):
            upstream = queue._upstream_ready("control", preflight_path=queue.HERE / "fake.json")
        self.assertEqual(upstream["arm"], "control")
        self.assertEqual(upstream["source_sha256"], source["source_sha256"])
        self.assertEqual(upstream["source_training_ids_sha256"], train.sw155.ids_sha(source["image_ids"]))
        self.assertTrue(upstream["preflight_training_admission"])
        self.assertEqual(upstream["rgb_asset_validation"], assets)

    def test_final_under_lease_gpu_check_fails_closed_for_owner_or_uuid_change(self):
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", return_value=("uuid-A", 64, [])):
            self.assertTrue(queue._final_gpu_admission(2, "uuid-A")[0])
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", return_value=("uuid-B", 64, [])):
            ok, state = queue._final_gpu_admission(2, "uuid-A")
            self.assertFalse(ok); self.assertEqual(state["reason"], "gpu_uuid_changed")
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", return_value=("uuid-A", 64, [999])):
            ok, state = queue._final_gpu_admission(2, "uuid-A")
            self.assertFalse(ok); self.assertEqual(state["reason"], "gpu_has_owners")

    def test_queue_fingerprint_resolves_current_owner_guard_and_trainer_files(self):
        fp = queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0159_one_shot_event_regime/training_queue.py", fp)
        self.assertIn("collaborative_test/SW_0159_one_shot_event_regime/train.py", fp)
        self.assertIn(queue.owner.OWNER_GUARD.relative_to(queue.ROOT).as_posix(), fp)

    def test_terminal_preflight_queue_without_report_does_not_wait_forever(self):
        path = Path("preflight-state-fixture.json")
        states = [
            {"experiment": train.EXPERIMENT, "status": "complete_admitted", "training_admission": True},
            {"experiment": train.EXPERIMENT, "status": "complete_rejected", "training_admission": False},
            {"experiment": train.EXPERIMENT, "status": "waiting_for_exclusive_gpu"},
            {"experiment": "unexpected", "status": "waiting_for_exclusive_gpu"},
        ]
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "read_text", side_effect=[json.dumps(row) for row in states]):
            self.assertTrue(queue._preflight_terminal_without_report(path))
            self.assertTrue(queue._preflight_terminal_without_report(path))
            self.assertFalse(queue._preflight_terminal_without_report(path))
            with self.assertRaisesRegex(ValueError, "unexpected"):
                queue._preflight_terminal_without_report(path)


if __name__ == "__main__":
    unittest.main()
