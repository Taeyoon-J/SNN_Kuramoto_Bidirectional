"""CPU checks for source-bound SW0163 Q/A row storage."""
from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
import uuid
from unittest import mock

import numpy as np

from collaborative_test.SW_0163_relation_context_assignment.context_cache import (
    AtomicContextCache, _context_arrays, prepare_context_inputs_cpu,
)
from collaborative_test.SW_0163_relation_context_assignment import context_cache


class ContextCacheTests(unittest.TestCase):
    def setUp(self):
        module_dir = Path(__file__).resolve().parent
        self.root = module_dir / f"_c163_{uuid.uuid4().hex[:8]}"
        self.root.mkdir()
        self.cache_root = self.root / "cache"
        self.ids = list(range(1000, 1512))
        self.rgb_hashes = [f"{i:064x}" for i in range(1, 513)]
        self.bindings = {"ordered_image_ids": self.ids, "rgb_sha256": self.rgb_hashes,
                         "source_checkpoint_sha256": "a" * 64}
        self.q = np.zeros((1024, 1024), dtype=np.float32)
        self.a = np.eye(1024, dtype=np.float32) * np.float32(1e-6)

    def tearDown(self):
        module_dir = Path(__file__).resolve().parent
        resolved = self.root.resolve()
        if resolved.parent != module_dir or not resolved.name.startswith("_c163_"):
            raise RuntimeError("refusing to clean an unowned context-cache test directory")
        shutil.rmtree(resolved)

    def test_context_schema_rejects_wrong_dtype_asymmetry_and_nonfinite(self):
        q, a = _context_arrays(self.q, self.a)
        self.assertEqual(q.dtype, np.float32)
        self.assertEqual(a.dtype, np.float32)
        bad = self.q.copy(); bad[0, 1] = .1
        with self.assertRaisesRegex(ValueError, "symmetric"):
            _context_arrays(bad, self.a)
        bad = self.a.copy(); bad[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _context_arrays(self.q, bad)
        bad = self.q.copy(); bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _context_arrays(bad, self.a)
        with self.assertRaisesRegex(ValueError, "float32"):
            _context_arrays(self.q.astype(np.float64), self.a)

    def test_atomic_row_restart_adopts_only_source_bound_bytes(self):
        cache = AtomicContextCache(self.cache_root, self.bindings)
        cache.write(0, self.q, self.a, self.rgb_hashes[0])

        # Simulate process exit between atomic row publication and ledger update.
        (self.cache_root / "ledger.json").write_text(
            json.dumps({"rows": []}), encoding="utf-8")
        resumed = AtomicContextCache(self.cache_root, self.bindings)
        self.assertEqual(len(resumed.records), 1)
        q, a = resumed.load(0)
        np.testing.assert_array_equal(q, self.q)
        np.testing.assert_array_equal(a, self.a)
        resumed.write(0, self.q, self.a, self.rgb_hashes[0])
        changed_q = self.q.copy(); changed_q[2, 2] = 1
        with self.assertRaisesRegex(ValueError, "differs"):
            resumed.write(0, changed_q, self.a, self.rgb_hashes[0])
        with self.assertRaisesRegex(ValueError, "RGB"):
            resumed.write(0, self.q, self.a, "f" * 64)

    def test_wrong_existing_bindings_are_preserved_and_rejected(self):
        cache = AtomicContextCache(self.cache_root, self.bindings)
        cache.write(0, self.q, self.a, self.rgb_hashes[0])
        changed = dict(self.bindings)
        changed["source_checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "binding differs"):
            AtomicContextCache(self.cache_root, changed)
        self.assertTrue(cache.row_path(0).is_file())

    def test_public_cache_producer_manifest_roundtrips_through_actual_validator(self):
        # Use the real row producer/finalizer/validator with a one-row internal
        # fixture so the deployed 512-row pathway's binding-key contract is exercised.
        with mock.patch.object(context_cache, "COUNT", 1):
            cache = AtomicContextCache(self.cache_root, self.bindings)
            cache.write(0, self.q, self.a, self.rgb_hashes[0])
            cache.finalize(source_state_before="a" * 64,
                           source_state_after="a" * 64, b4_replay=True)
            checked = context_cache.validate_final_cache(
                self.cache_root, expected_bindings=self.bindings)
        self.assertEqual(checked["count"], 1)
        self.assertEqual(checked["context_rows"][0]["image_id"], self.ids[0])
        self.assertIs(checked["source_model_unchanged"], True)
        self.assertEqual(checked["source_model_state_before_sha256"], "a" * 64)

    def test_public_cpu_prepare_uses_actual_cache_contract_and_binds_dependencies(self):
        # Exercise the production preparation function with only its source
        # readers stubbed; keep the real validator entrypoints and fingerprints.
        source_manifest = self.root / "source_manifest.json"
        source_manifest.write_text("{}", encoding="utf-8")
        train_rgb = self.root / "train_rgb.npy"
        train_rgb.write_bytes(b"registered RGB cache fixture")
        hdf5 = self.root / "train.h5"
        hdf5.write_bytes(b"registered HDF5 fixture")
        old_cache = self.root / "source_cache"
        old_cache.mkdir()
        (old_cache / "manifest.json").write_text("{}", encoding="utf-8")
        ids = list(range(5000, 5512))
        rows = np.arange(512, dtype=np.int64)
        expected_old_bindings = {"source": "fixture"}
        prepared = {
            "source_contract": {"source_sha256": "c" * 64,
                                "manifest_path": source_manifest,
                                "rows": rows},
            "rgb_asset_validation": {"train_manifest_sha256": "d" * 64},
            "image_ids": ids,
            "rgb_sha256": [f"{i:064x}" for i in range(1, 513)],
            "rgb_rows": np.zeros((512, 128, 128, 3), dtype=np.uint8),
            "targets32": np.zeros((512, 1024), dtype=np.int64),
            "bindings": expected_old_bindings,
        }
        source_trace = {"status": "complete", "count": 512,
                        "bindings_sha256": "e" * 64,
                        "bindings": {"cache_implementation_fingerprint": {"cache.py": "f" * 64}}}

        with (mock.patch.object(context_cache.source_cache, "prepare_cache_cpu",
                                return_value=prepared) as prepare,
              mock.patch.object(context_cache.source_cache, "validate_final_cache",
                                return_value=source_trace) as validate,
              mock.patch.object(context_cache.source_cache, "CACHE_ROOT", old_cache),
              mock.patch.object(context_cache, "EXPECTED_SOURCE_CACHE_MANIFEST_SHA256",
                                context_cache.sha256_file(old_cache / "manifest.json")),
              mock.patch.object(context_cache.sw130, "TRAIN_RGB", train_rgb),
              mock.patch.object(context_cache.sw155, "HDF5_PATH", hdf5)):
            result = prepare_context_inputs_cpu()
        prepare.assert_called_once_with()
        validate.assert_called_once_with(old_cache, expected_bindings=expected_old_bindings)
        self.assertEqual(result["context_bindings"]["ordered_image_ids"], ids)
        self.assertEqual(result["context_bindings"]["source_rows"], rows.tolist())
        self.assertEqual(result["context_bindings"]["source_trace_cache_bindings_sha256"], "e" * 64)
        self.assertEqual(result["context_bindings"]["protocol_sha256"],
                         context_cache.sha256_file(context_cache.HERE / "protocol.json"))
        self.assertEqual(result["context_bindings"]["code_fingerprint"], context_cache._fingerprint())


if __name__ == "__main__":
    unittest.main()
