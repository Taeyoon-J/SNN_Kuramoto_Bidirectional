from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0154_native64_resolution_compare import resource_queue as queue


def _fixture_report(fp, manifest_path, ids, rows, source_sha, assets):
    trace_shapes = {
        "component_spikes": [1, 4, 4096, 1024],
        "component_membrane": [1, 4, 4096, 1024],
        "component_gates": [1, 4, 4096, 1024],
        "spikes": [1, 4096, 1024], "membrane": [1, 4096, 1024],
        "theta": [1, 1024, 4096, 4],
    }
    report = {
        "status": "complete", "experiment": queue.EXPERIMENT,
        "stage": "source_only_resource_probe", "seed": 1,
        "source_image_id": ids[0], "source_pool_row": rows[0],
        "source_training_ids_sha256": queue.hashlib.sha256(
            queue.np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "source_core_sha256": source_sha,
        "source_manifest_sha256": queue.sha256_file(manifest_path),
        "source_manifest_status": "complete", "rgb_asset_validation": assets,
        "native32_parity_path": str(queue.ROOT / "collaborative_test/SW_0135_native32_spike_binding"
                                     / "results_archive/native32_zero_adapter_parity_seed1_b4.json"),
        "native32_parity_sha256": queue.sha256_file(
            queue.ROOT / "collaborative_test/SW_0135_native32_spike_binding"
            / "results_archive/native32_zero_adapter_parity_seed1_b4.json"),
        "implementation_fingerprint": fp,
        "ground_truth_used": False, "masks_read": False,
        "training_admission": False,
        "native_grid": [64, 64], "node_count": 4096, "gamma_shape": [1, 8, 4096],
        "rollout": {"total_steps": 1024, "settle": 512, "batch_size": 1,
            "optimizer_updates": 0,
            "trace_fields": {key: {"shape": shape, "dtype": "torch.float32", "finite": True}
                             for key, shape in trace_shapes.items()}},
        "readout": {"q_shape": [1, 4096, 4096], "labels_shape": [1, 64, 64]},
        "model_state_sha256_before": {"wrapped": "a" * 64, "encoder": "b" * 64},
        "model_state_sha256_after": {"wrapped": "a" * 64, "encoder": "b" * 64},
        "resource": {"graph_seconds": 1.0, "rollout_seconds": 2.0,
            "q_qcc_seconds": 0.5, "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200},
    }
    return report


class Native64ResourceQueueTests(unittest.TestCase):
    def test_plan_is_one_fixed_seed1_train_image_resource_probe(self):
        task = queue.task_plan()
        argv = queue.command(task)
        self.assertEqual(task["seed"], 1)
        self.assertEqual(task["stage"], "source_only_resource_probe")
        self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0154_native64_resolution_compare.resource_probe"])
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        self.assertNotIn("--mask", argv)
        with self.assertRaises(ValueError):
            queue.command(dict(task, seed=0))

    def test_report_validator_binds_actual_shapes_source_assets_and_no_gt(self):
        ids = [68304, 49032, 70041, 7829]
        rows = [4, 9, 15, 20]
        source_sha = "c" * 64
        manifest = queue.HERE / "README.md"
        assets = {"train_cache_sha256": "d" * 64}
        expected = queue.report_fingerprint()
        payload = _fixture_report(expected, manifest, ids, rows, source_sha, assets)
        path = queue.HERE / "queue_test_report.json"
        try:
            with mock.patch.object(queue.sw130, "source_contract",
                    return_value=("checkpoint", manifest, {"status": "complete"}, rows, ids, source_sha)), \
                 mock.patch.object(queue.sw130, "validate_rgb_assets", return_value=assets):
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertTrue(queue.validate_report(path, expected_fingerprint=expected))
                for key, value in (("ground_truth_used", True), ("training_admission", True)):
                    bad = dict(payload, **{key: value})
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    self.assertFalse(queue.validate_report(path, expected_fingerprint=expected))
                bad = json.loads(json.dumps(payload))
                bad["rollout"]["trace_fields"]["theta"]["shape"] = [1, 1]
                path.write_text(json.dumps(bad), encoding="utf-8")
                self.assertFalse(queue.validate_report(path, expected_fingerprint=expected))
        finally:
            path.unlink(missing_ok=True)

    def test_unstable_reservation_releases_then_launches_only_after_stable_lease(self):
        output = queue.HERE / "queue_test_child_report.json"
        log = queue.HERE / "queue_test_child.log"
        state = queue.HERE / "queue_test_state.json"
        for path in (output, log, state):
            path.unlink(missing_ok=True)
        instance = object.__new__(queue.ResourceQueue)
        instance.existing_complete = False
        instance.stop_event = threading.Event()
        instance.task = dict(queue.task_plan(), output=str(output), log=str(log))
        instance.report_fp = queue.report_fingerprint()
        instance._verify_fingerprints = lambda: None
        instance._update = lambda *args, **kwargs: None
        class Lease:
            def __init__(self, fd): self.fd, self.closed = fd, False
            def fileno(self): return self.fd
            def close(self): self.closed = True
        first, second = Lease(301), Lease(302)
        lock_calls = []
        fake_fcntl = SimpleNamespace(LOCK_UN=1,
            flock=lambda fd, op: lock_calls.append((fd, op)))
        ids = [68304, 49032, 70041, 7829]
        rows = [4, 9, 15, 20]
        source_sha = "c" * 64
        manifest = queue.HERE / "README.md"
        assets = {"train_cache_sha256": "d" * 64}
        payload = _fixture_report(instance.report_fp, manifest, ids, rows, source_sha, assets)
        class Child:
            pid = 76543
            def poll(self): return 0
            def wait(self): return 0
        def popen(argv, **kwargs):
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(kwargs["pass_fds"], (second.fileno(),))
            output.write_text(json.dumps(payload), encoding="utf-8")
            return Child()
        try:
            with mock.patch.object(queue, "OUTPUT", output), mock.patch.object(queue, "LOG", log), \
                 mock.patch.object(queue, "STATE", state), mock.patch.object(queue, "fcntl", fake_fcntl), \
                 mock.patch.object(queue.owner, "_acquire", side_effect=[(0, "uuid0", 20, first),
                                                                           (2, "uuid2", 12, second)]) as acquire, \
                 mock.patch.object(queue.owner, "_stable_idle", side_effect=[False, True]) as stable, \
                 mock.patch.object(queue.sw130, "source_contract",
                    return_value=("checkpoint", manifest, {"status": "complete"}, rows, ids, source_sha)), \
                 mock.patch.object(queue.sw130, "validate_rgb_assets", return_value=assets), \
                 mock.patch.object(queue.subprocess, "Popen", side_effect=popen):
                self.assertEqual(instance.launch_once(), "passed")
                self.assertEqual(acquire.call_count, 2)
                self.assertEqual(stable.call_count, 2)
                self.assertTrue(first.closed)
                self.assertTrue(second.closed)
                self.assertGreaterEqual(len(lock_calls), 2)
        finally:
            for path in (output, log, state):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
