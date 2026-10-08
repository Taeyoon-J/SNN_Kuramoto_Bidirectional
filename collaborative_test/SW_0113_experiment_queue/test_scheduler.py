import json
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import coordinator
from coordinator import (active_public, gpu_reserve, handoff_valid,
                         ready_tasks, read_append_manifest)


class SchedulerContracts(unittest.TestCase):
    def test_independent_branches_can_run_concurrently(self):
        tasks = [
            {"task_id": "108", "depends_on": []},
            {"task_id": "109", "depends_on": []},
            {"task_id": "110", "depends_on": ["110-pre"]},
        ]
        ready = ready_tasks(tasks, set())
        self.assertEqual({t["task_id"] for t in ready}, {"108", "109"})

    def test_failed_dependency_blocks_only_its_branch(self):
        tasks = [
            {"task_id": "110-pre", "depends_on": []},
            {"task_id": "110-train", "depends_on": ["110-pre"]},
            {"task_id": "109-pre", "depends_on": []},
        ]
        ready = ready_tasks(tasks, set(), {"110-pre"})
        self.assertEqual([t["task_id"] for t in ready], ["109-pre"])

    def test_one_local_reservation_per_gpu_and_rechecks_owners(self):
        reservations = set()
        owners = lambda gpu: []
        memory = lambda gpu: 0
        self.assertEqual(gpu_reserve(0, reservations, owners, memory), 0)
        reservations.add(0)
        self.assertIsNone(gpu_reserve(0, reservations, owners, memory))
        self.assertIsNone(gpu_reserve(1, reservations, lambda gpu: [777], memory))
        self.assertIsNone(gpu_reserve(1, reservations, owners, lambda gpu: 513))

    def test_handoff_requires_parent_audit_and_no_owned_workers(self):
        path = Path("handoff.json")
        valid = {"status": "verified_superseded", "active_worker_count": 0,
                 "old_coordinator_alive": False, "archived_state_path": "/archive/state.json",
                 "archived_state_sha256": "abc"}
        with patch.object(Path, "read_text", return_value=json.dumps(valid)), \
             patch.object(Path, "is_file", return_value=True), \
             patch("coordinator.sha", return_value="abc"):
            self.assertTrue(handoff_valid(path))
            valid["active_worker_count"] = 1
        with patch.object(Path, "read_text", return_value=json.dumps(valid)), \
             patch.object(Path, "is_file", return_value=True), \
             patch("coordinator.sha", return_value="abc"):
            self.assertFalse(handoff_valid(path))

    def test_append_registry_is_monotonic_and_prefix_preserving(self):
        path = Path("append_registry.json")
        first = {"version": 1, "experiments": ["SW0111"]}
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(first)), \
             patch("coordinator.sha", return_value="first"):
            self.assertEqual(read_append_manifest(path, 0, []), (1, ["SW0111"], "first"))
        invalid = {"version": 2, "experiments": ["SW0112"]}
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(invalid)):
            with self.assertRaises(ValueError):
                read_append_manifest(path, 1, ["SW0111"])

    def test_active_worker_json_omits_process_objects_but_keeps_lease_path(self):
        payload = active_public({"pid": 42, "gpu": 1, "proc": object(),
                                 "lease_stream": io.StringIO(), "lease_path": "/tmp/gpu1.lock"})
        self.assertEqual(payload, {"pid": 42, "gpu": 1, "lease_path": "/tmp/gpu1.lock"})

    def test_resolution_branch_can_append_without_replacing_existing_tasks(self):
        payload = {"version": 2, "experiments": ["SW0112", "SW0114"]}
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value=json.dumps(payload)), \
             patch("coordinator.sha", return_value="resolution-registry"):
            self.assertEqual(read_append_manifest(Path("registry.json"), 1, ["SW0112"]),
                             (2, ["SW0112", "SW0114"], "resolution-registry"))

    def test_adapter_loading_restores_import_state_even_when_import_fails(self):
        original_path = list(coordinator.sys.path)
        prior_run = object()
        coordinator.sys.modules["run"] = prior_run
        def fail_after_contamination(*args, **kwargs):
            coordinator.sys.path.append("temporary-contamination")
            raise ImportError("synthetic adapter failure")
        with patch.object(coordinator.importlib.util, "spec_from_file_location",
                          side_effect=fail_after_contamination):
            with self.assertRaises(ImportError):
                coordinator._load_adapter("test", "collaborative_test/SW_0110_xy_graph_route")
        self.assertEqual(coordinator.sys.path, original_path)
        self.assertIs(coordinator.sys.modules.get("run"), prior_run)
        coordinator.sys.modules.pop("run", None)


if __name__ == "__main__":
    unittest.main()
