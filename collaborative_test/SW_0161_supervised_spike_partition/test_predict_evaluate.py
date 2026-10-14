"""Focused CPU contracts for SW161 prediction rows and GT-last scoring."""
import numpy as np
import unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0161_supervised_spike_partition import predict_evaluate as pe


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.root = pe.HERE / "_cpu_test_predict_rows"
        self.root.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_atomic_row_round_trip_and_identity_tamper_rejected(self):
        labels = np.zeros((32, 32), dtype=np.int64)
        labels[:, 16:] = 1
        rgb_sha, bind_sha = "a" * 64, "b" * 64
        pe._write_row(self.root, 0, 1320, rgb_sha, bind_sha, labels)
        loaded = pe._load_row(self.root, 0, 1320, rgb_sha, bind_sha)
        np.testing.assert_array_equal(labels, loaded)
        with self.assertRaises(ValueError):
            pe._load_row(self.root, 0, 1321, rgb_sha, bind_sha)
        row_json = pe._row_dir(self.root, 0) / "row.json"
        import json
        record = json.loads(row_json.read_text(encoding="utf-8"))
        record["image_id"] = 1321
        row_json.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ValueError):
            pe._load_row(self.root, 0, 1320, rgb_sha, bind_sha)

    def test_publish_is_create_once_and_binds_ground_truth_free_status(self):
        out = self.root / "prediction"
        labels = np.zeros((pe.COUNT, pe.GRID, pe.GRID), dtype=np.int64)
        protocol = {"experiment": pe.train.EXPERIMENT, "status": "complete",
                    "ground_truth_used_for_prediction": False}
        result = pe._publish_prediction(out, labels, protocol)
        self.assertEqual(result["prediction_sha256"], pe.sha256_file(out / "predictions.npz"))
        self.assertEqual(result["protocol_sha256"], pe.sha256_file(out / "protocol.json"))
        with self.assertRaises(FileExistsError):
            pe._publish_prediction(out, labels, protocol)

    def test_shared_prediction_producer_validator_binding_and_tamper_rejection(self):
        arm = pe.ARMS[0]
        artifact = {"manifest_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                    "model_sha256": "c" * 64}
        bindings = {"experiment": pe.train.EXPERIMENT, "arm": arm,
                    "training_manifest_sha256": artifact["manifest_sha256"],
                    "training_checkpoint_sha256": artifact["checkpoint_sha256"],
                    "source_model_state_sha256": "f" * 64}
        bind_sha = pe._sha_json(bindings)
        # Match the actual final producer protocol rather than a validator-only stub.
        protocol = {"experiment": pe.train.EXPERIMENT, "status": "complete", "arm": arm,
            "seed": 1, "image_ids": [1320, 1639], "count": 320, "grid_size": [32, 32],
            "prediction_shape": [320, 32, 32], "steps": 1024, "settle": 512,
            "live_tail_steps": 64, "optimizer_updates": 0,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": False,
            "prediction_bindings": bindings, "prediction_bindings_sha256": bind_sha,
            "prediction_row_commits_sha256": "d" * 64,
            "implementation_fingerprint": pe.implementation_fingerprint(),
            "training_manifest_sha256": artifact["manifest_sha256"],
            "training_checkpoint_sha256": artifact["checkpoint_sha256"],
            "training_model_state_sha256": artifact["model_sha256"],
            "source_model_state_sha256_before": bindings["source_model_state_sha256"],
            "source_model_state_sha256_after": bindings["source_model_state_sha256"],
            "model_state_sha256_before": artifact["model_sha256"],
            "model_state_sha256_after": artifact["model_sha256"]}
        self.assertTrue(pe._validate_prediction_protocol(protocol, arm=arm, bindings=bindings,
            bindings_sha=bind_sha, rows_sha="d" * 64, artifact=artifact))
        protocol["prediction_bindings"] = dict(bindings, training_manifest_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "prediction_bindings"):
            pe._validate_prediction_protocol(protocol, arm=arm, bindings=bindings,
                bindings_sha=bind_sha, rows_sha="d" * 64, artifact=artifact)

    def test_prediction_binding_uses_actual_cache_hdf5_identity_fields(self):
        dataset = self.root / "canonical.h5"
        dataset.write_bytes(b"registered source HDF identity")
        stat = dataset.stat()
        ids = list(range(4096))
        source = {"source_sha256": "1" * 64, "manifest_path": self.root / "source.json",
                  "image_ids": ids}
        source["manifest_path"].write_text("{}", encoding="utf-8")
        artifact = {"manifest_sha256": "2" * 64, "checkpoint_sha256": "3" * 64,
            "model_sha256": "4" * 64,
            "manifest": {"implementation_fingerprint": {}, "cache_manifest_sha256": "5" * 64,
                         "cache_bindings_sha256": "6" * 64}}
        cache_manifest = {"bindings_sha256": "6" * 64, "bindings": {
            "hdf5_path": str(dataset.resolve()), "hdf5_size": stat.st_size,
            "hdf5_mtime_ns": stat.st_mtime_ns}, "source_model_state_before_sha256": "7" * 64}
        rgb = np.zeros((pe.COUNT, 128, 128, 3), dtype=np.uint8)
        with mock.patch.object(pe.sw155, "source_contract", return_value=source), \
             mock.patch.object(pe.sw130, "validate_rgb_assets", return_value={"asset": "ok"}), \
             mock.patch.object(pe, "implementation_fingerprint", return_value={"predict_evaluate.py": "8" * 64}):
            bindings = pe._prediction_bindings(pe.ARMS[0], artifact,
                {"prediction_sha256": "9" * 64, "protocol_sha256": "a" * 64},
                {"prediction_sha256": "b" * 64, "protocol_sha256": "c" * 64},
                {"prediction_sha256": "d" * 64, "protocol_sha256": "e" * 64},
                rgb, dataset, cache_manifest)
        self.assertEqual(bindings["dataset_path"], str(dataset.resolve()))
        self.assertEqual(bindings["dataset_size_bytes"], stat.st_size)
        self.assertEqual(bindings["source_model_state_sha256"], "7" * 64)

    def test_shared_cpu_prepare_and_producer_protocol_use_real_artifact_schema(self):
        import hashlib
        import json

        arm = pe.ARMS[0]
        dataset = self.root / "canonical.h5"
        dataset.write_bytes(b"registered canonical dataset identity")
        dataset_stat = dataset.stat()
        source_manifest = self.root / "source_manifest.json"
        source_manifest.write_text("{}", encoding="utf-8")
        source = {"source_sha256": "1" * 64, "manifest_path": source_manifest,
                  "image_ids": list(range(4096)), "rows": np.arange(4096, dtype=np.int64)}
        cache_root = self.root / "cache"
        cache_root.mkdir()
        cache_manifest = {"status": "complete", "bindings_sha256": "6" * 64,
            "source_model_state_before_sha256": "7" * 64,
            "bindings": {"hdf5_path": str(dataset.resolve()),
                "hdf5_size": dataset_stat.st_size, "hdf5_mtime_ns": dataset_stat.st_mtime_ns}}
        cache_bytes = json.dumps(cache_manifest, sort_keys=True, separators=(",", ":")).encode()
        (cache_root / "manifest.json").write_bytes(cache_bytes)
        artifact = {"manifest_sha256": hashlib.sha256((cache_root / "manifest.json").read_bytes()).hexdigest(),
            "checkpoint_sha256": "3" * 64, "model_sha256": "4" * 64,
            "manifest": {"implementation_fingerprint": {"train.py": "5" * 64},
                "cache_manifest_sha256": hashlib.sha256((cache_root / "manifest.json").read_bytes()).hexdigest(),
                "cache_bindings_sha256": "6" * 64}}
        references = (
            {"prediction_sha256": "8" * 64, "protocol_sha256": "9" * 64},
            {"prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64},
            {"prediction_sha256": "c" * 64, "protocol_sha256": "d" * 64})
        rgb = np.zeros((pe.COUNT, 128, 128, 3), dtype=np.uint8)

        # This calls the production prepare function. Only source/assets and large
        # model/cache readers are substituted; artifact/cache field names are real.
        with mock.patch.object(pe.train, "CACHE_ROOT", cache_root), \
             mock.patch.object(pe, "_training_artifact", return_value=artifact), \
             mock.patch.object(pe, "_source_refs", return_value=references), \
             mock.patch.object(pe.sw155, "source_contract", return_value=source), \
             mock.patch.object(pe.sw130, "validate_rgb_assets", return_value={"rgb_assets": "validated"}), \
             mock.patch.object(pe.np, "load", return_value=rgb):
            cuda_initialized = pe.torch.cuda.is_initialized()
            prepared = pe.prepare_prediction_cpu(arm, training_root=self.root / "training", dataset=dataset)
            self.assertEqual(pe.torch.cuda.is_initialized(), cuda_initialized)

        bindings = prepared["bindings"]
        expected_provenance = {
            "training_manifest_sha256", "training_checkpoint_sha256", "training_model_state_sha256",
            "training_implementation_fingerprint", "cache_manifest_sha256", "cache_bindings_sha256",
            "source_model_state_sha256", "source_checkpoint_sha256", "source_manifest_sha256",
            "source_training_ids_sha256", "rgb_asset_validation", "source_qcc_prediction_sha256",
            "source_qcc_protocol_sha256", "source_T_prediction_sha256", "source_T_protocol_sha256",
            "slot_prediction_sha256", "slot_protocol_sha256", "dataset_path", "dataset_size_bytes",
            "dataset_mtime_ns", "image_ids_sha256", "steps", "settle", "live_tail_steps"}
        self.assertTrue(expected_provenance.issubset(bindings))
        self.assertEqual(bindings["dataset_path"], str(dataset.resolve()))
        self.assertEqual(bindings["dataset_size_bytes"], dataset_stat.st_size)
        self.assertEqual(bindings["dataset_mtime_ns"], dataset_stat.st_mtime_ns)

        # Mirror the final prediction producer's complete protocol and feed it
        # through the same strict validator used by both prediction and scoring.
        model_sha = artifact["model_sha256"]
        bind_sha = prepared["bindings_sha256"]
        protocol = {"experiment": pe.train.EXPERIMENT, "status": "complete", "arm": arm,
            "seed": pe.SEED, "image_ids": [int(pe.IMAGE_IDS[0]), int(pe.IMAGE_IDS[-1])],
            "count": pe.COUNT, "grid_size": [pe.GRID, pe.GRID],
            "prediction_shape": [pe.COUNT, pe.GRID, pe.GRID], "steps": 1024,
            "settle": 512, "live_tail_steps": 64, "optimizer_updates": 0,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": False,
            "prediction_bindings": bindings, "prediction_bindings_sha256": bind_sha,
            "prediction_row_commits_sha256": "e" * 64,
            "implementation_fingerprint": pe.implementation_fingerprint(),
            "training_manifest_sha256": artifact["manifest_sha256"],
            "training_checkpoint_sha256": artifact["checkpoint_sha256"],
            "training_model_state_sha256": model_sha,
            "source_model_state_sha256_before": bindings["source_model_state_sha256"],
            "source_model_state_sha256_after": bindings["source_model_state_sha256"],
            "model_state_sha256_before": model_sha, "model_state_sha256_after": model_sha}
        self.assertTrue(pe._validate_prediction_protocol(protocol, arm=arm, bindings=bindings,
            bindings_sha=bind_sha, rows_sha="e" * 64, artifact=artifact))
        for field in expected_provenance:
            altered = dict(protocol)
            altered["prediction_bindings"] = dict(bindings)
            altered["prediction_bindings"][field] = "tampered"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "prediction_bindings"):
                pe._validate_prediction_protocol(altered, arm=arm, bindings=bindings,
                    bindings_sha=bind_sha, rows_sha="e" * 64, artifact=artifact)

    def test_gate_requires_both_registered_paired_ci_lower_bounds(self):
        means = {name: {metric: .5 for metric in pe.contract.METRICS}
                 for name in (*pe.ARMS, "source97", "SW0143_T", "Slot70k")}
        means["mask_supervised_P"]["fg_ari"] = .8
        means["pair_supervised_P"]["fg_ari"] = .7
        means["SW0143_T"]["fg_ari"] = .6
        paired = {f"mask_supervised_P_minus_{base}_fg_ari": {"ci95": [.01, .2]}
                  for base in ("SW0143_T", "pair_supervised_P")}
        gate, _ = pe._registered_gate(means, paired)
        self.assertTrue(gate["registered_seed1_gate_pass"])
        paired["mask_supervised_P_minus_pair_supervised_P_fg_ari"]["ci95"] = [-.01, .2]
        gate, _ = pe._registered_gate(means, paired)
        self.assertFalse(gate["registered_seed1_gate_pass"])

    def test_score_freezes_every_prediction_before_hdf5_access(self):
        ready = set()
        temp_files = []
        records = {}
        try:
            for name in (*pe.ARMS, "source97", "SW0143_T", "Slot70k"):
                folder = self.root / name; folder.mkdir()
                archive, protocol = folder / "pred.npz", folder / "protocol.json"
                archive.write_bytes((name + " archive").encode())
                protocol.write_bytes((name + " protocol").encode())
                labels = np.zeros((pe.COUNT, pe.GRID, pe.GRID), dtype=np.int64)
                records[name] = {"ids": pe.IMAGE_IDS.copy(), "labels": labels,
                    "protocol": {"prediction_bindings": {"dataset_path": str((self.root/"data.h5").resolve()),
                        "dataset_size_bytes": 123, "dataset_mtime_ns": 456}},
                    "npz_path": archive, "protocol_path": protocol,
                    "prediction_sha256": pe.sha256_file(archive), "protocol_sha256": pe.sha256_file(protocol)}
            dataset = self.root / "data.h5"; dataset.write_bytes(b"fixture")
            stat = dataset.stat()
            # Match the fixture's stat values without editing production validators.
            records["source97"].update({})
            class FakeH5:
                def __enter__(self):
                    self.mask = self
                    self.shape = (1640, 128, 128)
                    return {"mask": self}
                def __exit__(self, *args): return False
                def __getitem__(self, item):
                    self.assert_gt()
                    return np.zeros((pe.COUNT, 128, 128), dtype=np.uint8)
                def assert_gt(self):
                    expected = {*pe.ARMS, "source97", "SW0143_T", "Slot70k"}
                    if ready != expected:
                        raise AssertionError("GT opened before all prediction validators completed")
            # Provide a fake dataset stat large enough for the fixed row range.
            class DatasetPath(type(dataset)):
                pass
            for record in records.values():
                record["protocol"]["prediction_bindings"].update({
                    "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns})
            def validate(arm, **kwargs):
                ready.add(arm); return records[arm]
            def refs():
                ready.add("source97"); ready.add("SW0143_T"); ready.add("Slot70k")
                return records["source97"], records["SW0143_T"], records["Slot70k"]
            def metrics(pred, target):
                return {metric: {"mean": .5, "valid_count": pe.COUNT,
                                 "per_image": np.full(pe.COUNT, .5).tolist()}
                        for metric in pe.contract.METRICS}
            with mock.patch.object(pe, "validate_prediction", side_effect=validate), \
                 mock.patch.object(pe, "_source_refs", side_effect=refs), \
                 mock.patch.object(pe.h5py, "File", return_value=FakeH5()), \
                 mock.patch.object(pe.contract, "modal_native32", side_effect=lambda x: x), \
                 mock.patch.object(pe.contract, "production_metrics32", side_effect=metrics), \
                 mock.patch.object(pe, "_registered_gate", return_value=({"registered_seed1_gate_pass": False}, {})):
                report = pe.score_all(output=self.root / "score.json", dataset=dataset)
            self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
            self.assertFalse(report["ground_truth_used_for_prediction"])
            self.assertTrue(report["ground_truth_used_for_scoring"])
            self.assertEqual(set(report["means"]), {*pe.ARMS, "source97", "SW0143_T", "Slot70k"})
        finally:
            import shutil
            shutil.rmtree(self.root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
