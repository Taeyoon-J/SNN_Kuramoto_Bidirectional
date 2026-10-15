from __future__ import annotations

import unittest
import os
import subprocess
import sys

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import (
    calibration_v2 as calibration, preflight_v2, preflight_queue_v2)


class HistoricalCacheQModeTests(unittest.TestCase):
    @staticmethod
    def _report():
        source = {"selected_ids": list(range(8)), "source_cache_state_sha256": "e" * 64}
        rows = []
        for index in range(8):
            rows.append({
                "index": index, "image_id": index,
                "cache_q_producer_mode": ("live_trace_strided_b4" if index < 4
                                           else "saved_contiguous_b4"),
                "cache_context_verified": True,
                "q_sha256": "1" * 64, "a_sha256": "2" * 64,
                "cache_q_sha256": "3" * 64, "live_q_sha256": "4" * 64,
                "cache_replay_q_sha256": "5" * 64,
                "live_vs_cache_replay_max_abs_diff": 1e-6,
                "live_vs_cache_replay_threshold_0_5_edge_differences": 0,
                "live_component_spike_strides": [12288, 512, 1],
                "cache_q_input_strides": [6144, 512, 1] if index >= 4
                else [12288, 512, 2],
                "trace_sha256": {name: "a" * 64 for name in calibration.TRACE_FIELDS},
            })
        report = {
            "experiment": "SW0169_normalized_hard_reset", "source": source,
            "implementation_fingerprint": preflight_v2.implementation_fingerprint(),
            "queue_fingerprint_sha256": preflight_queue_v2.fingerprint_sha256(),
            "calibration": {
                "source_six_trace_parity": True, "source_q_parity": True,
                "source_a_parity": True, "source_cache_replay_exact": True,
                "source_model_state_sha256_before": "e" * 64,
                "source_model_state_sha256_after": "e" * 64, "source_rows": rows,
            },
            "training_admission": False, "training_ids": list(range(8)),
            "validation_data_read": False, "ground_truth_scope": "TRAIN8 only", "arms": {},
        }
        return report

    def test_first_batch_replays_live_strided_producer_mode(self):
        torch.manual_seed(169)
        backing = torch.randn(4, 4, 7, 2 * (calibration.SETTLE + 12))
        live = backing[..., ::2]
        self.assertEqual(tuple(live.shape), (4, 4, 7, calibration.SETTLE + 12))
        self.assertNotEqual(live.stride(-1), 1)

        components, mode = calibration._cache_component_input(
            live, [None] * 8, 0, device="cpu")
        self.assertEqual(mode, "live_trace_strided_b4")
        self.assertEqual(components.stride(-1), live.stride(-1))
        self.assertEqual(components.shape[-1], 12)

        classifier = calibration._import_keep_cuda(
            "snn_kuramoto_bidirectional.spike_classifier")
        q = classifier.spike_synchrony_affinity(
            components.mean(dim=1), components, settle=0, affinity_mode="spike")
        direct = classifier.spike_synchrony_affinity(
            live.detach()[..., calibration.SETTLE:].mean(dim=1),
            live.detach()[..., calibration.SETTLE:], settle=0, affinity_mode="spike")
        self.assertTrue(torch.equal(q, direct))

    def test_later_batches_replay_saved_contiguous_producer_mode(self):
        torch.manual_seed(170)
        backing = torch.randn(8, 4, 7, 24)
        live = backing[..., ::2]
        saved = [live[i].cpu().numpy().copy() for i in range(8)]
        components, mode = calibration._cache_component_input(
            live[4:], saved, 4, device="cpu")
        self.assertEqual(mode, "saved_contiguous_b4")
        self.assertEqual(components.stride(-1), 1)
        expected = torch.from_numpy(np.stack(saved[4:8]))
        self.assertTrue(torch.equal(components, expected))

        classifier = calibration._import_keep_cuda(
            "snn_kuramoto_bidirectional.spike_classifier")
        q = calibration._q_from_components(components)
        direct = classifier.spike_synchrony_affinity(
            expected.mean(dim=1), expected, settle=0, affinity_mode="spike")
        self.assertTrue(torch.equal(q, direct))
        diag = calibration._q_difference(q[0], direct[0])
        self.assertEqual(diag["live_vs_cache_replay_max_abs_diff"], 0.0)
        self.assertEqual(diag["live_vs_cache_replay_threshold_0_5_edge_differences"], 0)

    def test_replay_mode_is_fixed_to_registered_batch_starts(self):
        trace = torch.zeros(4, 4, 2, calibration.STEPS)
        with self.assertRaisesRegex(ValueError, "registered B4 starts"):
            calibration._cache_component_input(trace, [None] * 8, 2, device="cpu")

    def test_v2_report_validator_binds_modes_order_and_live_cache_diagnostics(self):
        report = self._report()
        accepted = preflight_queue_v2.validate_report(
            report, source_proof=report["source"],
            fingerprint=report["queue_fingerprint_sha256"])
        self.assertIs(accepted, report)
        bad = self._report()
        bad["calibration"]["source_rows"][4]["cache_q_producer_mode"] = "live_trace_strided_b4"
        with self.assertRaisesRegex(ValueError, "producer mode/order"):
            preflight_queue_v2.validate_report(
                bad, source_proof=bad["source"],
                fingerprint=bad["queue_fingerprint_sha256"])

    def test_post_gpu_cpu_recheck_command_uses_v2_producer_modules(self):
        command = preflight_queue_v2._cpu_revalidation_command()
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "-c")
        script = command[2]
        self.assertIn("calibration_v2 as c", script)
        self.assertIn("preflight_v2 as p", script)
        self.assertIn("p._source_identity(c.prepare_calibration_cpu())", script)
        self.assertNotIn("import calibration as c", script)
        import_prefix = script.split("print(json.dumps")[0]
        probe = import_prefix + "print(c.__name__ + '|' + p.__name__)"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", probe], cwd=preflight_queue_v2.ROOT,
                                env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=60, check=True)
        self.assertEqual(result.stdout.strip(),
                         "collaborative_test.SW_0169_normalized_hard_reset.calibration_v2|"
                         "collaborative_test.SW_0169_normalized_hard_reset.preflight_v2")


if __name__ == "__main__":
    unittest.main()
