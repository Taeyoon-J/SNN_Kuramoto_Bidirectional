import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0156_signed_coupling import source_screen_queue as queue
from collaborative_test.SW_0156_signed_coupling import screen


class FakeRgbCache:
    dtype = np.dtype(np.uint8)
    shape = (70000, 128, 128, 3)

    def __getitem__(self, key):
        return np.zeros((128, 128, 3), dtype=np.uint8)


class SourceScreenQueueTests(unittest.TestCase):
    def test_real_screen_and_queue_fingerprints_resolve_actual_sw0145_api(self):
        report_fp = screen.implementation_fingerprint()
        sw145_path = Path(screen.sw145.__file__).resolve()
        key = sw145_path.relative_to(screen.ROOT).as_posix()
        self.assertIn(key, report_fp)
        self.assertEqual(report_fp[key], screen.sha256_file(sw145_path))
        queue_fp = queue.implementation_fingerprint()
        self.assertEqual(queue_fp["report_fingerprint"], report_fp)

    def test_exception_cleanup_stops_owned_child_before_unlock(self):
        events = []

        class Proc:
            def poll(self):
                return None if "stop" not in events else 0

        class Lease:
            def fileno(self):
                return 41
            def close(self):
                events.append("close")

        with mock.patch.object(queue.owner, "_stop_owned",
                               side_effect=lambda _proc: events.append("stop")), \
             mock.patch.object(queue, "fcntl", SimpleNamespace(
                 LOCK_UN=8, flock=lambda _fd, _mode: events.append("unlock"))):
            queue._cleanup_child_before_lease_release(Proc(), Lease())
        self.assertEqual(events, ["stop", "unlock", "close"])

    def test_registered_single_task_and_child_visibility(self):
        task = queue.task_plan()
        self.assertEqual(task["seed"], 1)
        self.assertEqual(task["steps"], 1024)
        self.assertEqual(task["settle"], 512)
        self.assertEqual(task["depends_on"], ["SW0143_T_seed1_complete"])
        argv = queue.command(task)
        self.assertIn("collaborative_test.SW_0156_signed_coupling.screen", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        with self.assertRaises(ValueError):
            queue.command({**task, "seed": 0})

    def _report(self):
        source = {
            "source_core_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "source_training_ids_sha256": hashlib.sha256(np.asarray([501], dtype="<i8").tobytes()).hexdigest(),
            "encoder_sha256": "d" * 64,
            "feature_preprocessing_sha256": "e" * 64,
            "rgb_asset_validation": {"train_cache_sha256": "f" * 64},
        }
        shapes = {"theta": [1, 1024, 1024, 4], "membrane": [1, 1024, 1024],
                  "spikes": [1, 1024, 1024], "component_membrane": [1, 4, 1024, 1024],
                  "component_spikes": [1, 4, 1024, 1024],
                  "component_gates": [1, 4, 1024, 1024]}
        traces = {k: {"shape": shapes[k], "dtype": "torch.float32"}
                  for k in screen.TRACE_FIELDS}
        dynamics = {"settled_steps": 512,
                    "per_component": [{"weighted_spike_mean": .1, "spike_variance": .09,
                                        "gate_mean": .5, "gate_active_fraction": 1.0,
                                        "event_on_fraction_when_gate_positive": .1,
                                        "gate_positive_support_count": 512}
                                       for _ in range(4)],
                    "actual_spike_q_offdiagonal": {"mean": .4, "min": .1, "max": .9}}
        arm = {"seconds": 1.25, "trace_schema": traces, "dynamics": dynamics}
        rich_arm = {**arm, "peak_cuda_allocated_bytes": 1234,
                    "finite_q": True, "qcc_label_counts": {"0": 1022, "1": 2}}
        return {
            "experiment": queue.EXPERIMENT,
            "status": "source_screen_complete", "seed": 1,
            "training_cache_row": 7, "training_image_id": 501,
            "training_rgb_sha256": hashlib.sha256(
                np.zeros((128, 128, 3), dtype=np.uint8).tobytes()).hexdigest(),
            "source_lineage": source,
            "sw0143_preflight_sha256": "2" * 64,
            "sw0143_T_prediction_sha256": "3" * 64,
            "sw0143_T_protocol_sha256": "4" * 64,
            "implementation": {"screen": "frozen"},
            "source_model_state_before_sha256": "5" * 64,
            "source_model_state_after_sha256": "5" * 64,
            "source_model_unchanged": True,
            "registered_parameters": {"steps": 1024, "settle": 512, "beta": .5,
                                       "source_top_k": 128, "gain": 8.,
                                       "coupling_backend": "factorized"},
            "zero_intervention_exact_parity": {
                "trace_fields": {k: True for k in screen.TRACE_FIELDS},
                "q_equal": True, "qcc_equal": True},
            "arms": {"source_T": arm, "beta0_exact_control": arm,
                     "signed_candidate": rich_arm, "matched_control": rich_arm},
            "signed_interaction_screen": {
                "candidate_negative_offdiagonal_abs_mass_fraction": .2,
                "ground_truth_used": False, "optimizer_updates": 0,
                "full320_authorized_by_this_record": False},
            "ground_truth_used": False, "optimizer_updates": 0,
        }

    def test_valid_report_and_strict_tamper_rejection(self):
        record = self._report()
        payload = [json.dumps(record)]
        report_path = Path("synthetic-screen.json")
        source = record["source_lineage"]
        contract = (Path("checkpoint.pt"), Path("manifest.json"), {}, [7], [501], "a" * 64)
        upstream = {"preflight_sha256": "2" * 64, "prediction_sha256": "3" * 64,
                    "protocol_sha256": "4" * 64, "source_lineage": source}
        with mock.patch.object(Path, "read_text", new=lambda _self, **_kw: payload[0]), \
             mock.patch.object(queue.screen, "implementation_fingerprint", return_value={"screen": "frozen"}), \
             mock.patch.object(queue.sw143.foundation.sw130, "source_contract", return_value=contract), \
             mock.patch.object(queue, "sha256_file", return_value="b" * 64), \
             mock.patch.object(queue.np, "load", return_value=FakeRgbCache()):
            self.assertTrue(queue.validate_report(report_path, upstream=upstream))
            record["zero_intervention_exact_parity"]["trace_fields"]["spikes"] = False
            payload[0] = json.dumps(record)
            self.assertFalse(queue.validate_report(report_path, upstream=upstream))

    def test_missing_upstream_is_pending_without_gpu_reservation(self):
        with mock.patch.object(queue.sw143, "_preflight_path", return_value=Path("Z:/absent/preflight.json")), \
             mock.patch.object(queue.sw143, "_arm_dir", return_value=Path("Z:/absent/T/seed1")):
            ready, state = queue.upstream_state()
        self.assertFalse(ready)
        self.assertEqual(state, "waiting_for_sw0143_T_seed1")

    def test_nonzero_readout_or_gt_use_cannot_validate(self):
        record = self._report()
        record["ground_truth_used"] = True
        payload = json.dumps(record)
        report_path = Path("synthetic-screen.json")
        contract = (Path("checkpoint.pt"), Path("manifest.json"), {}, [7], [501], "a" * 64)
        upstream = {"preflight_sha256": "2" * 64, "prediction_sha256": "3" * 64,
                    "protocol_sha256": "4" * 64, "source_lineage": record["source_lineage"]}
        with mock.patch.object(Path, "read_text", new=lambda _self, **_kw: payload), \
             mock.patch.object(queue.screen, "implementation_fingerprint", return_value={"screen": "frozen"}), \
             mock.patch.object(queue.sw143.foundation.sw130, "source_contract", return_value=contract), \
             mock.patch.object(queue, "sha256_file", return_value="b" * 64), \
             mock.patch.object(queue.np, "load", return_value=FakeRgbCache()):
            self.assertFalse(queue.validate_report(report_path, upstream=upstream))


if __name__ == "__main__":
    unittest.main()


