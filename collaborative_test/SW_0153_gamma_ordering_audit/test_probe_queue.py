from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0153_gamma_ordering_audit import probe_queue as queue


def _valid_report(expected_fingerprint, ids, rows, source_sha, manifest_sha):
    names = ("naive_fixed_channel_shuffle", "naive_train_mean_channel_order",
             "consistent_global_channel_reindex", "per_image_patch_permutation")
    variants = {
        "naive_fixed_channel_shuffle": {"drive_mse": 0.1, "graph_adjacency_mse": 0.2,
            "actual_q_offdiag_mse": 0.3, "qcc_changed_fraction": 0.4},
        "naive_train_mean_channel_order": {"drive_mse": 0.1, "graph_adjacency_mse": 0.2,
            "actual_q_offdiag_mse": 0.3, "qcc_changed_fraction": 0.4},
        "consistent_global_channel_reindex": {"drive_max_abs": 1e-6,
            "graph_adjacency_max_abs": 1e-6, "actual_q_offdiag_mse": 1e-8,
            "qcc_changed_fraction": 0.0},
        "per_image_patch_permutation": {"drive_mse_after_node_unpermutation": 1e-8,
            "graph_adjacency_mse_after_node_unpermutation": 0.2,
            "actual_q_offdiag_mse_after_node_unpermutation": 0.2,
            "qcc_changed_fraction_after_node_unpermutation": 0.3},
    }
    return {"status": "complete", "experiment": queue.EXPERIMENT, "seed": 1,
        "count": queue.COUNT, "ids": ids, "pool_rows": rows,
        "source_core_sha256": source_sha, "source_manifest_sha256": manifest_sha,
        "ground_truth_used": False, "optimizer_updates": 0,
        "implementation_fingerprint": expected_fingerprint,
        "model_state_sha256_before_after": {"foundation": "a" * 64, "encoder": "b" * 64},
        "per_image": [{"image_id": image_id,
                       "variants": {name: dict(variants[name]) for name in names}}
                      for image_id in ids],
        "mean_change_summary": {name: {key: 0.1 for key in variants[name]} for name in names}}


class GammaProbeQueueTests(unittest.TestCase):
    def test_plan_is_single_fixed_first_four_train_source_probe(self):
        task = queue.task_plan()
        self.assertEqual(task["stage"], "audit_source")
        self.assertEqual(task["seed"], 1)
        self.assertEqual(task["count"], 4)
        self.assertEqual(task["depends_on"], [])
        argv = queue.command(task)
        self.assertIn("--count", argv)
        self.assertEqual(argv[argv.index("--count") + 1], "4")
        self.assertIn("--device", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        altered = dict(task, count=8)
        with self.assertRaises(ValueError):
            queue.command(altered)

    def test_report_validator_accepts_actual_report_schema_and_rejects_gt_claim(self):
        ids = [101, 103, 107, 109]
        rows = [0, 2, 4, 6]
        source_sha = "c" * 64
        manifest_path = Path(queue.HERE / "README.md")
        manifest_sha = queue.sha256_file(manifest_path)
        report_path = queue.HERE / "queue_test_report.json"
        expected = {"bound": "cpu-fixture"}
        try:
            with mock.patch.object(queue, "source_report_fingerprint", return_value=expected), \
                 mock.patch.object(queue.sw130, "source_contract",
                    return_value=("checkpoint", manifest_path, {}, rows, ids, source_sha)):
                report_path.write_text(json.dumps(_valid_report(expected, ids, rows, source_sha, manifest_sha)),
                                       encoding="utf-8")
                self.assertTrue(queue.validate_report(report_path, expected_fingerprint=expected))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["ground_truth_used"] = True
                report_path.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.validate_report(report_path, expected_fingerprint=expected))
        finally:
            report_path.unlink(missing_ok=True)

    def test_child_command_has_no_gt_and_writes_only_registered_output(self):
        task = queue.task_plan()
        argv = queue.command(task)
        self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0153_gamma_ordering_audit.audit_source"])
        self.assertEqual(Path(argv[argv.index("--output") + 1]), Path(task["output"]))
        self.assertNotIn("--dataset", argv)
        self.assertNotIn("--mask", argv)

    def test_unstable_gpu_lease_is_released_then_child_gets_stable_physical_mapping(self):
        output = queue.HERE / "queue_test_child_output.json"
        log = queue.HERE / "queue_test_child.log"
        for path in (output, log):
            path.unlink(missing_ok=True)
        instance = object.__new__(queue.ProbeQueue)
        instance.existing_complete = False
        instance.stop_event = __import__("threading").Event()
        instance.task = dict(queue.task_plan(), output=str(output), log=str(log))
        instance.report_fingerprint = {"bound": "queue-launch-fixture"}
        instance._verify_fingerprints = lambda: None
        instance._update = lambda *args, **kwargs: None
        class FakeLease:
            def __init__(self, fd): self.fd, self.closed = fd, False
            def fileno(self): return self.fd
            def close(self): self.closed = True
        first, second = FakeLease(111), FakeLease(222)
        lock_calls = []
        fake_fcntl = SimpleNamespace(LOCK_UN=1,
            flock=lambda fd, op: lock_calls.append((fd, op)))
        manifest_path = queue.HERE / "README.md"
        ids, rows, source_sha = [11, 12, 13, 14], [0, 1, 2, 3], "c" * 64
        report = _valid_report(instance.report_fingerprint, ids, rows, source_sha,
                                queue.sha256_file(manifest_path))

        class FakeProc:
            pid = 99999
            def poll(self): return 0
            def wait(self): return 0

        def launch(argv, **kwargs):
            output.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(kwargs["pass_fds"], (second.fileno(),))
            self.assertIn("--count", argv)
            return FakeProc()

        try:
            with mock.patch.object(queue, "OUTPUT", output), mock.patch.object(queue, "LOG", log), \
                 mock.patch.object(queue, "STATE", queue.HERE / "queue_test_state.json"), \
                 mock.patch.object(queue, "fcntl", fake_fcntl), \
                 mock.patch.object(queue, "source_report_fingerprint", return_value=instance.report_fingerprint), \
                 mock.patch.object(queue.sw130, "source_contract",
                     return_value=("checkpoint", manifest_path, {}, rows, ids, source_sha)), \
                 mock.patch.object(queue.owner, "_acquire", side_effect=[(0, "uuid0", 10, first),
                                                                          (2, "uuid2", 12, second)]) as acquire, \
                 mock.patch.object(queue.owner, "_stable_idle", side_effect=[False, True]) as stable, \
                 mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
                self.assertEqual(instance.launch_once(), "passed")
                self.assertEqual(acquire.call_count, 2)
                self.assertEqual(stable.call_count, 2)
                self.assertGreaterEqual(len(lock_calls), 2)
        finally:
            output.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            (queue.HERE / "queue_test_state.json").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
