"""CPU producer/consumer contracts for SW0175 fixed-320 inference."""
from __future__ import annotations

import hashlib
import json
import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0175_critical_component_binding import predict_evaluate_v2 as evaluate


class PredictionContractTests(unittest.TestCase):
    def test_public_predictor_uses_separate_registered_supports_and_accepts_capacity_1024(self):
        source = np.arange(1, 1025, dtype=np.int32).reshape(32, 32)
        native = np.arange(1024, 0, -1, dtype=np.int32).reshape(32, 32)
        q = torch.eye(1024, dtype=torch.float32).unsqueeze(0)
        a = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        trace = {name: torch.ones((1,), dtype=torch.float32) for name in evaluate.capture.TRACE_FIELDS}
        observed = []

        def merge(q_value, a_value, gain, labels, *, grid_shape):
            observed.append(np.asarray(labels, dtype=np.int32).copy())
            return {"labels": np.asarray(labels, dtype=np.int32), "merge_count": 0}

        found = SimpleNamespace(encode=lambda image: torch.zeros((1, 1)),
            wrapped=SimpleNamespace(core=SimpleNamespace(
                kuramoto=object(), graph_generator=SimpleNamespace(
                    log_coupling_gain=torch.tensor(np.log(2.0))))))
        with (mock.patch.object(evaluate.capture, "capture_source_coupling", return_value=(trace, a)),
              mock.patch.object(evaluate.capture, "_trace_check"),
              mock.patch.object(evaluate.capture, "_validate_actual_A"),
              mock.patch.object(evaluate.capture, "_actual_q", return_value=q),
              mock.patch.object(evaluate.sw136, "_qcc_labels", return_value=native[None]),
              mock.patch.object(evaluate.solver, "foreground_induced_merge", side_effect=merge)):
            result = evaluate._predict_one(found, torch.zeros((1, 128, 128, 3), dtype=torch.uint8),
                {"source_T": {"labels": source[None]}}, 0)
        self.assertEqual(len(observed), 2)
        np.testing.assert_array_equal(observed[0], source)
        np.testing.assert_array_equal(observed[1], native)
        self.assertEqual(int(result["labels"]["conditional"].max()), 1024)
        self.assertEqual(int(result["labels"]["standalone"].max()), 1024)


    def test_public_cpu_factory_consumes_actual_nested_train_v2_cache_proof(self):
        central_temp = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        central_temp.mkdir(parents=True, exist_ok=True)
        root = central_temp / f"sw175-v2-source-proof-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            cache_root = root / "source_cache"
            cache_root.mkdir()
            cache_manifest = cache_root / "manifest.json"
            cache_manifest.write_text('{"status":"complete"}\n', encoding="utf-8")
            cache_sha = evaluate.sha256_file(cache_manifest)
            dataset = root / "dataset.h5"
            dataset.write_bytes(b"fixture HDF identity")
            rgb_path = root / "rgb.npy"
            rgb_path.write_bytes(b"mmap placeholder")
            rgb_meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
            expected_map = hashlib.sha256(np.arange(1320, 1640, dtype="<i8").tobytes()).hexdigest()
            rgb_meta_path.write_text(json.dumps({"status":"complete", "cache_shape":[320,128,128,3],
                "ids_mapping_sha256":expected_map}), encoding="utf-8")
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"source checkpoint")
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text("{}\n", encoding="utf-8")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            admitted = {"prepared": {"source_checkpoint_sha256": checkpoint_sha,
                "source_manifest_sha256": evaluate.sha256_file(source_manifest),
                "cache_manifest_sha256":"1"*64,"arrays_sha256":"2"*64,
                "cache_proof": {"image_ids": list(range(512))}},
                "proof": {"source_cache_manifest_sha256": cache_sha},
                "admission_report_sha256":"a"*64,"queue_fingerprint_sha256":"b"*64,
                "preflight_fingerprint": {"fixture":"source"}}
            manifests = {arm: {"bindings": evaluate.train._bindings(admitted, arm,
                "c"*64, "d"*64, "e"*64)} for arm in evaluate.ARMS}
            refs = {"source_T":{"ids":evaluate.IMAGE_IDS.copy(), "arm":"T",
                    "labels":np.ones((320,32,32),dtype=np.int32)},
                    "SW0164":{"ids":evaluate.IMAGE_IDS.copy(),
                    "labels":np.ones((320,32,32),dtype=np.int32)}}
            rgb = SimpleNamespace(dtype=np.dtype(np.uint8), shape=(320,128,128,3),
                                  __getitem__=None)
            # A tiny ndarray subclass gives the factory the production metadata contract
            # without hashing 15 GiB of repeated RGB fixture bytes.
            class RGBFixture:
                dtype = np.dtype(np.uint8)
                shape = (320,128,128,3)
                def __getitem__(self, _index): return np.zeros((1,),dtype=np.uint8)
            rgb = RGBFixture()
            with (mock.patch.object(evaluate.torch.cuda, "is_initialized", return_value=False),
                  mock.patch.object(evaluate.train, "validate_training_artifact",
                                    side_effect=lambda arm, _path: manifests[arm]),
                  mock.patch.object(evaluate, "_references", return_value=refs),
                  mock.patch.object(evaluate.sw130, "VAL_RGB", rgb_path),
                  mock.patch.object(evaluate.np, "load", return_value=rgb),
                  mock.patch.object(evaluate, "_hash_rgb_rows", return_value=["f"*64]*320),
                  mock.patch.object(evaluate.sw130.source97, "DATASET", dataset),
                  mock.patch.object(evaluate.sw130, "source_contract",
                    return_value=(checkpoint,source_manifest,{},None,None,checkpoint_sha)),
                  mock.patch.object(evaluate.source_cache, "CACHE_ROOT", cache_root)):
                prepared = evaluate.prepare_prediction_inputs_cpu()
            self.assertEqual(prepared["source_checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(prepared["rgb_hashes"], ["f"*64]*320)
            # The actual train_v2 producer places this identity only under source_proof.
            self.assertNotIn("source_cache_manifest_sha256", manifests[evaluate.ARMS[0]]["bindings"])
            self.assertEqual(evaluate._source_cache_manifest_sha256(
                manifests[evaluate.ARMS[0]]["bindings"]), cache_sha)
            cache_manifest.write_text('{"status":"tampered"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate._source_cache_manifest_sha256({"source_proof": {"source_cache_manifest_sha256": "short"}})
        finally:
            resolved_root = root.resolve(); resolved_temp = central_temp.resolve()
            self.assertEqual(resolved_root.parent, resolved_temp)
            for item in sorted(resolved_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if item.is_file(): item.unlink()
                elif item.is_dir(): item.rmdir()
            resolved_root.rmdir()

    def test_row_writer_reader_roundtrip_and_integrity(self):
        central_temp = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        central_temp.mkdir(parents=True, exist_ok=True)
        root = central_temp / f"sw175-row-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            labels = np.arange(1, 1025, dtype=np.int32).reshape(32, 32)
            support = np.ones((32, 32), dtype=np.int32)
            traces = {name: hashlib.sha256(name.encode()).hexdigest()
                      for name in evaluate.capture.TRACE_FIELDS}
            result = {"labels": {endpoint: labels.copy() for endpoint in evaluate.ENDPOINTS},
                "standalone_support": support,
                "q_sha256": "a" * 64, "actual_a_sha256": "b" * 64,
                "standalone_qcc_sha256": hashlib.sha256(
                    support.astype("<i4", copy=False).tobytes()).hexdigest(),
                "trace_sha256": traces, "gain": 1.25,
                "details": {endpoint: {"merge_count": 1, "support_foreground_patches": 1024}
                            for endpoint in evaluate.ENDPOINTS}}
            evaluate._commit_prediction_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d" * 64, "e" * 64,
                result, rollout_seconds=.5, cuda_peak_allocated_bytes=4096)
            loaded, _record = evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d" * 64, "e" * 64)
            for endpoint in evaluate.ENDPOINTS:
                np.testing.assert_array_equal(loaded[endpoint], labels)
            archive = root / "rows" / "0000" / "row.npz"
            archive.write_bytes(archive.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                evaluate._read_row(root, 0, int(evaluate.IMAGE_IDS[0]), "d" * 64, "e" * 64)
        finally:
            row = root / "rows" / "0000"
            if row.exists():
                for item in row.iterdir():
                    item.unlink()
                row.rmdir()
                (root / "rows").rmdir()
            if root.exists():
                root.rmdir()

    def test_fixed_bootstrap_uses_seed175_and_rejects_partial_vectors(self):
        values = np.linspace(0, 1, evaluate.COUNT)
        first = evaluate._bootstrap(values, np.zeros_like(values))
        self.assertEqual(first, evaluate._bootstrap(values, np.zeros_like(values)))
        self.assertEqual(first["seed"], 175)
        self.assertEqual(first["replicates"], 10_000)
        with self.assertRaises(ValueError):
            evaluate._bootstrap(values[:-1], np.zeros_like(values[:-1]))


if __name__ == "__main__":
    unittest.main()
