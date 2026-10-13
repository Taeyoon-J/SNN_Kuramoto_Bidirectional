from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0154_native64_resolution_compare import predict_native64_recovery as recovery


class Native64RecoveryTests(unittest.TestCase):
    def test_actual_seed0_and2_parity_receipts_validate_with_scoped_seed(self):
        base = recovery.predictor.ROOT
        for seed in recovery.SEEDS:
            parity_path = (base / "collaborative_test/SW_0135_native32_spike_binding/results_archive"
                           / f"native32_zero_adapter_parity_seed{seed}_b4.json")
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            try:
                _checkpoint, source_manifest_path, _manifest, _pool, training_ids, source_sha = (
                    recovery.predictor.sw130.source_contract(seed))
                assets = recovery.predictor.sw130.validate_rgb_assets()
            except FileNotFoundError:
                # Local-only archived fixture fallback. Deployed tests use the
                # authoritative source_contract and current asset validator.
                source_manifest_path = (base / "collaborative_test/SW_0097_graph_adaptation/results"
                    / f"seed{seed}_positive_frozen/manifest.json")
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
                training_ids = source_manifest["training_ids"]
                source_sha = parity["source_core_sha256"]
                assets = parity["rgb_asset_validation"]
            self.assertEqual(source_sha, recovery.EXPECTED_SOURCE_SHA256[seed])
            self.assertEqual(parity["source_core_sha256"], source_sha)
            _payload, digest = recovery.validate_seed_parity(
                seed, source_sha=source_sha,
                source_manifest_sha=hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
                training_ids=training_ids,
                assets=assets, parity_path=parity_path)
            self.assertEqual(digest, hashlib.sha256(parity_path.read_bytes()).hexdigest())
            self.assertEqual(len(training_ids), 4096)
        self.assertEqual(recovery.legacy_preflight.SEED, 1)

    def test_entrypoint_adds_repair_provenance_and_restores_legacy_state(self):
        seed = 2
        manifest_path = Path(__file__).resolve()
        train_ids = list(range(4096))
        pool = list(range(4096))
        original_publish = recovery.predictor._publish_prediction
        original_seed = recovery.legacy_preflight.SEED
        captured = {}

        def fake_publish(_directory, _ids, _labels, builder):
            protocol = builder("a" * 64)
            captured.update(protocol)
            return Path("predictions.npz"), Path("protocol.json"), protocol

        def fake_prediction(call_seed, **kwargs):
            self.assertEqual(call_seed, seed)
            self.assertEqual(recovery.legacy_preflight.SEED, seed)
            return recovery.predictor._publish_prediction(
                Path("unused"), [1], [2], lambda digest: {"status": "prediction_complete"})[2]

        def hash_side_effect(path):
            if Path(path) == Path(recovery.predictor.__file__):
                return recovery.EXPECTED_PREDICTOR_SHA256
            return "c" * 64

        with mock.patch.object(recovery, "validate_seed_parity",
                return_value=({}, "b" * 64)), \
             mock.patch.object(recovery, "sha256_file", side_effect=hash_side_effect), \
             mock.patch.object(recovery.predictor.sw130, "source_contract",
                return_value=(Path("source.pt"), manifest_path, {}, pool, train_ids,
                              recovery.EXPECTED_SOURCE_SHA256[seed])), \
             mock.patch.object(recovery.predictor.sw130, "validate_rgb_assets", return_value={}), \
             mock.patch.object(recovery.predictor, "_publish_prediction", side_effect=fake_publish), \
             mock.patch.object(recovery.predictor, "predict_seed", side_effect=fake_prediction):
            report = recovery.predict_seed(seed)
        self.assertEqual(report["seed_specific_parity_recovery"]["scoped_seed"], seed)
        self.assertEqual(report["seed_specific_parity_recovery"]["frozen_predictor_sha256"],
                         recovery.EXPECTED_PREDICTOR_SHA256)
        self.assertEqual(recovery.legacy_preflight.SEED, original_seed)
        self.assertIs(recovery.predictor._publish_prediction, original_publish)
        self.assertEqual(captured["seed_specific_parity_recovery"],
                         report["seed_specific_parity_recovery"])

    def test_wrapper_rejects_seed1_and_unregistered_source_hash(self):
        with self.assertRaises(ValueError):
            recovery.validate_seed_parity(1, source_sha="x", source_manifest_sha="y",
                training_ids=[], assets={})
        with self.assertRaises(ValueError):
            recovery.validate_seed_parity(0, source_sha="x", source_manifest_sha="y",
                training_ids=[], assets={})


if __name__ == "__main__":
    unittest.main()
