"""CPU lifecycle tests for the SW0162 prediction and score coordinators."""
from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0162_connected_signed_partition import prediction_queue as pq
from collaborative_test.SW_0162_connected_signed_partition import score_queue as sq


class QueueDependencyTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[3]
        self.root = self.workspace / ("t162q" + uuid.uuid4().hex[:8])
        self.root.mkdir()

    def tearDown(self):
        target = self.root.resolve()
        if not target.is_relative_to(self.workspace):
            raise RuntimeError("refusing to remove queue fixture outside workspace")
        shutil.rmtree(target)

    def test_live_preflight_waits_then_adopts_validated_report(self):
            root = self.root
            state_path, report_path = root / "preflight-state.json", root / "report.json"
            report_path.write_text(json.dumps({"status": "source8_preflight_complete"}), encoding="utf-8")
            waiting = {"experiment": "SW0162_connected_signed_partition",
                "status": "waiting_for_exclusive_gpu", "supervisor_pid": 1234}
            state_path.write_text(json.dumps(waiting), encoding="utf-8")
            q = pq.PredictionQueue.__new__(pq.PredictionQueue)
            q.owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
                "python", "-m", "collaborative_test.SW_0162_connected_signed_partition.preflight_queue"]})
            q.prepared_fn = mock.Mock(return_value={"ready": True})
            q.stop_event = SimpleNamespace(is_set=lambda: False)
            writes = []
            q._write_state = lambda status, **kw: writes.append((status, kw))

            def complete_after_poll(timeout):
                state_path.write_text(json.dumps({"experiment": "SW0162_connected_signed_partition",
                    "status": "complete", "supervisor_pid": 1234}), encoding="utf-8")

            q.stop_event.wait = complete_after_poll
            with (mock.patch.object(pq.owner, "_proc_record", return_value={"state": "S", "argv": [
                      "python", "-m", "collaborative_test.SW_0162_connected_signed_partition.preflight_queue"]}),
                  mock.patch.object(pq.preflight_queue, "STATE", state_path),
                  mock.patch.object(pq.preflight, "OUTPUT", report_path),
                  mock.patch.object(pq.preflight_queue, "validate_report", return_value=True),
                  mock.patch.object(pq.preflight, "implementation_fingerprint", return_value={"fp": "a"}),
                  mock.patch.object(pq.preflight_queue, "source_identity", return_value={"source": "b"})):
                result = q._preflight()
            self.assertEqual(result, {"ready": True})
            self.assertEqual(q.prepared_fn.call_count, 1)
            self.assertEqual(writes[0][0], "waiting_for_source_preflight")

    def test_active_preflight_status_without_live_owner_is_terminal(self):
            state_path = self.root / "state.json"
            state_path.write_text(json.dumps({"experiment": "SW0162_connected_signed_partition",
                "status": "running", "supervisor_pid": 5678}), encoding="utf-8")
            q = pq.PredictionQueue.__new__(pq.PredictionQueue)
            q.owner = SimpleNamespace(_proc_record=lambda pid: None)
            q.stop_event = SimpleNamespace(is_set=lambda: False)
            with mock.patch.object(pq.preflight_queue, "STATE", state_path):
                with self.assertRaisesRegex(RuntimeError, "terminal/nonlive"):
                    q._preflight()

    def test_score_coordinator_tracks_every_prediction_wait_state(self):
        self.assertTrue({"waiting_for_source_preflight", "waiting_stable_gpu",
            "waiting_for_exclusive_gpu", "waiting_for_prediction_owner", "running"}
            <= sq.ACTIVE_PREDICTION_STATES)
        self.assertEqual(pq.ACTIVE_PREFLIGHT_STATES, {"waiting_for_idle_gpu", "waiting_stable_gpu",
            "waiting_for_exclusive_gpu", "waiting_gpu_probe_error", "running"})


if __name__ == "__main__":
    unittest.main()
