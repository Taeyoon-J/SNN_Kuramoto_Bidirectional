from __future__ import annotations

import shutil
import json
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np  # Keep NumPy before torch on local Windows.
import torch

from collaborative_test.SW_0161_supervised_spike_partition import cache
from collaborative_test.SW_0160_end_to_end_source_encoder import predict_evaluate as sw160_eval


class ExactTraceCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(cache.HERE) / ("cache_test_" + uuid.uuid4().hex)
        self.ids = [501, 502]
        self.rgb = ["a" * 64, "b" * 64]
        self.bindings = {"source": "source-sha", "ids": self.ids, "rgb": self.rgb,
                         "operator": "SW0143-T", "dtype": "float32"}
        self.rows = [np.full((4, 1024, 512), i + 0.25, dtype=np.float32) for i in range(2)]
        self.targets = [np.full((1024,), i, dtype=np.int64) for i in range(2)]

    def tearDown(self):
        if self.root.exists():
            resolved = self.root.resolve()
            module_dir = Path(cache.HERE).resolve()
            if resolved.parent != module_dir:
                raise AssertionError("refusing to clean outside the test-owned directory")
            shutil.rmtree(resolved)

    def test_exact_float32_row_commit_resume_and_final_manifest(self):
        store = cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        store.write_row(0, self.ids[0], self.rgb[0], self.rows[0], self.targets[0])

        # Simulate a crash after a complete atomic row publish but before ledger
        # update: only the fully validated, source-bound row is adopted.
        orphan = store._row_path(1)
        with orphan.open("xb") as stream:
            np.savez(stream, component_spikes=self.rows[1], targets32=self.targets[1],
                     index=np.asarray(1, dtype=np.int64), image_id=np.asarray(self.ids[1], dtype=np.int64),
                     rgb_sha256=np.asarray(self.rgb[1]),
                     bindings_sha256=np.asarray(store.bindings_sha256))
        resumed = cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        self.assertEqual(len(resumed.records), 2)
        trace, target = resumed.load_row(0)
        self.assertEqual(trace.dtype, np.float32)
        self.assertEqual(trace.shape, (4, 1024, 512))
        self.assertTrue(np.array_equal(trace, self.rows[0]))
        self.assertTrue(np.array_equal(target, self.targets[0]))
        manifest = resumed.finalize(bindings=self.bindings,
                                    source_model_state_before="c" * 64,
                                    source_model_state_after="c" * 64)
        self.assertEqual(manifest["count"], 2)
        self.assertIs(manifest["source_model_unchanged"], True)

    def test_partial_or_corrupt_published_row_never_admits(self):
        store = cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        store.write_row(0, self.ids[0], self.rgb[0], self.rows[0], self.targets[0])
        store._row_path(1).write_bytes(b"partial-npz")
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_orphan_row_with_wrong_identity_or_source_cannot_be_adopted(self):
        store = cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        orphan = store._row_path(0)
        with orphan.open("xb") as stream:
            np.savez(stream, component_spikes=self.rows[0], targets32=self.targets[0],
                     index=np.asarray(0, dtype=np.int64), image_id=np.asarray(999, dtype=np.int64),
                     rgb_sha256=np.asarray(self.rgb[0]),
                     bindings_sha256=np.asarray(store.bindings_sha256))
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)

    def test_score_guard_requires_matching_sw0160_complete_gt_last_report(self):
        queue_fp, score_fp = {"queue.py": "q" * 64}, {"evaluation.py": "e" * 64}
        score_sha = "s" * 64
        report = {"experiment": cache.SCORE_DEPENDENCY_EXPERIMENT, "status": "complete",
                  "seed_scope": [1], "ground_truth_opened_after_all_predictions_validated": True,
                  "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
                  "implementation_fingerprint": score_fp,
                  "means": {name: {metric: 0.5 for metric in sw160_eval.contract.METRICS}
                            for name in ("source97", "SW0143_T", *sw160_eval.ARMS)}}
        state = {"experiment": cache.SCORE_DEPENDENCY_EXPERIMENT, "status": "complete",
                 "score_sha256": score_sha, "queue_fingerprint": queue_fp,
                 "score_fingerprint": score_fp}
        dependency = cache._score_dependency_from_records(state, report, score_sha, queue_fp, score_fp)
        self.assertEqual(dependency["experiment"], cache.SCORE_DEPENDENCY_EXPERIMENT)
        self.assertIs(dependency["ground_truth_used_for_prediction"], False)

        # An SW0159 report or a newly published SW0160 score whose live queue
        # has not yet marked complete cannot satisfy the prerequisite.
        with self.assertRaisesRegex(ValueError, "SW0160"):
            cache._score_dependency_from_records({**state, "experiment": "SW0159_one_shot_event_regime"},
                                                  report, score_sha, queue_fp, score_fp)
        pending = {**state, "status": "running"}
        with self.assertRaisesRegex(RuntimeError, "live"):
            cache._score_dependency_from_records(pending, report, score_sha, queue_fp, score_fp,
                                                   score_owner_live=True)
        bad_means = {**report, "means": dict(report["means"])}
        bad_means["means"]["source97"] = {metric: float("nan") for metric in sw160_eval.contract.METRICS}
        with self.assertRaisesRegex(ValueError, "means schema"):
            cache._score_dependency_from_records(state, bad_means, score_sha, queue_fp, score_fp)

    def test_score_completion_published_during_dead_owner_probe_is_adopted(self):
        from collaborative_test.SW_0160_end_to_end_source_encoder import score_queue
        self.root.mkdir(parents=True, exist_ok=True)
        state_path, report_path = self.root / "state.json", self.root / "score.json"
        queue_fp, score_fp = {"queue.py": "q" * 64}, {"evaluation.py": "e" * 64}
        report = {"experiment": cache.SCORE_DEPENDENCY_EXPERIMENT, "status": "complete",
            "seed_scope": [1], "ground_truth_opened_after_all_predictions_validated": True,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
            "implementation_fingerprint": score_fp,
            "means": {name: {metric: 0.5 for metric in sw160_eval.contract.METRICS}
                      for name in ("source97", "SW0143_T", *sw160_eval.ARMS)}}
        report_path.write_text(json.dumps(report), encoding="utf-8")
        completed = {"experiment": cache.SCORE_DEPENDENCY_EXPERIMENT, "status": "complete",
            "score_sha256": cache.sha256_file(report_path), "queue_fingerprint": queue_fp,
            "score_fingerprint": score_fp}
        state_path.write_text(json.dumps({"experiment": cache.SCORE_DEPENDENCY_EXPERIMENT,
            "status": "running", "supervisor_pid": 123}), encoding="utf-8")

        def publish_then_report_dead(_pid):
            state_path.write_text(json.dumps(completed), encoding="utf-8")
            return False

        with (mock.patch.object(score_queue, "STATE", state_path),
              mock.patch.object(sw160_eval, "SCORE_PATH", report_path),
              mock.patch.object(score_queue, "implementation_fingerprint", return_value=queue_fp),
              mock.patch.object(sw160_eval, "implementation_fingerprint", return_value=score_fp),
              mock.patch.object(cache, "_score_queue_owner_live", side_effect=publish_then_report_dead)):
            dependency = cache._current_score_dependency()
        self.assertEqual(dependency["score_sha256"], cache.sha256_file(report_path))
        self.assertEqual(dependency["status"], "complete")

    def test_resume_rejects_changed_binding_or_tampered_committed_row(self):
        store = cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        store.write_row(0, self.ids[0], self.rgb[0], self.rows[0], self.targets[0])
        changed = dict(self.bindings, operator="different-operator")
        with self.assertRaisesRegex(ValueError, "different source/order/RGB"):
            cache._AtomicTraceCache(self.root, changed, self.ids, self.rgb)
        row = store._row_path(0)
        with row.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "file digest mismatch"):
            cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)

    def test_trace_and_target_schema_are_not_cast_or_truncated(self):
        with self.assertRaisesRegex(ValueError, "native float32"):
            cache._trace_row_array(self.rows[0].astype(np.float16))
        with self.assertRaisesRegex(ValueError, r"float32 \[4,1024,512\]"):
            cache._trace_row_array(np.zeros((4, 1024, 511), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"integer \[1024\]"):
            cache._target_row_array(np.zeros((1024,), dtype=np.float32))

    def test_production_final_validator_requires_all_registered_512_rows(self):
        store = cache._AtomicTraceCache(self.root, self.bindings, self.ids, self.rgb)
        for i in range(2):
            store.write_row(i, self.ids[i], self.rgb[i], self.rows[i], self.targets[i])
        store.finalize(bindings=self.bindings, source_model_state_before="d" * 64,
                       source_model_state_after="d" * 64)
        with self.assertRaisesRegex(ValueError, "manifest is invalid"):
            cache.validate_final_cache(self.root)


if __name__ == "__main__":
    unittest.main()
