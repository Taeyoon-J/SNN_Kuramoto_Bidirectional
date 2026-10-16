"""CPU contracts for SW0177 immutable fixed320 prediction inputs."""
import json
import unittest
import uuid
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np  # NumPy/OpenMP before Torch and legacy imports.
import torch
from torch import nn

from collaborative_test.SW_0177_learned_coupling_polarity import predict_evaluate as evaluate


class PredictionEvaluationTests(unittest.TestCase):
    @staticmethod
    def _supports():
        source = np.zeros((evaluate.COUNT, evaluate.GRID, evaluate.GRID), dtype=np.int32)
        native = np.zeros_like(source)
        for i in range(evaluate.COUNT):
            source[i, 2:11, 3:16] = 1
            # Deliberately different standalone QCC support for every row.
            native[i, 18:30, (i % 4):((i % 4) + 10)] = 3
        return source, native

    def test_full320_conditional_source_and_standalone_native_support_are_separate(self):
        source, native = self._supports()
        labels = {"conditional": source.copy(), "standalone": native.copy()}
        checked, support = evaluate.validate_prediction_arrays(labels, native, source > 0)
        self.assertEqual(checked["conditional"].shape, (320, 32, 32))
        self.assertTrue(np.array_equal(checked["standalone"] > 0, support > 0))
        self.assertFalse(np.array_equal(support > 0, source > 0))

    def test_source_membership_mismatch_is_rejected_without_conflating_standalone(self):
        source, native = self._supports()
        conditional = source.copy()
        conditional[0, 0, 0] = 7
        with self.assertRaisesRegex(ValueError, "conditional endpoint"):
            evaluate.validate_prediction_arrays(
                {"conditional": conditional, "standalone": native.copy()}, native, source > 0)
        standalone = native.copy()
        standalone[0, 0, 0] = 9
        with self.assertRaisesRegex(ValueError, "standalone endpoint"):
            evaluate.validate_prediction_arrays(
                {"conditional": source, "standalone": standalone}, native, source > 0)

    def test_nested_training_proof_schema_is_used_for_prediction_binding(self):
        root = Path("fixture_root")
        artifacts, manifests, states, state_hashes = {}, {}, {}, {}
        for arm in evaluate.ARMS:
            artifacts[arm] = root / arm
            manifests[arm] = {"bindings": {"source_proof": {
                "source_cache_manifest_sha256": "a" * 64},
                "source_checkpoint_sha256": "b" * 64},
                "final_model_state_sha256": "c" * 64,
                "training_fingerprint": {"trainer": "d" * 64},
                "source_model_state_before_sha256": "7" * 64}
            states[arm] = {"queue_fingerprint_sha256": "e" * 64}
            state_hashes[arm] = "f" * 64
        prepared = {"artifacts": artifacts, "manifests": manifests, "states": states,
            "state_sha256": state_hashes, "references": {
                "source_T": {"prediction_sha256": "1" * 64, "protocol_sha256": "2" * 64},
                "SW0164": {"prediction_sha256": "3" * 64, "protocol_sha256": "4" * 64},
                "SW0164_score_sha256": "8" * 64},
            "rgb_meta_sha256": "5" * 64, "rgb_hashes": ["6" * 64] * evaluate.COUNT,
            "dataset_path": str(root / "dataset.h5"), "dataset_size_bytes": 10,
            "dataset_mtime_ns": 20}
        with mock.patch.object(evaluate, "implementation_fingerprint", return_value={"test": "fp"}), \
             mock.patch.object(evaluate, "sha256_file", side_effect=lambda path: "9" * 64):
            binding = evaluate._bindings(prepared, "signed")
        self.assertEqual(binding["source_cache_manifest_sha256"], "a" * 64)
        self.assertNotIn("source_cache_manifest_sha256", manifests["signed"]["bindings"])
        self.assertEqual(binding["source_checkpoint_sha256"], "b" * 64)

    def test_training_record_consumer_accepts_real_manifest_and_queue_field_levels(self):
        central = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        files = []
        manifests, state_paths, manifest_hashes = {}, {}, {}
        source_proof = {"source_cache_manifest_sha256": "a" * 64,
            "source_cache_state_sha256": "b" * 64, "source_checkpoint_sha256": "c" * 64}
        common = {"source_proof": source_proof, "source_state_sha256": "b" * 64,
            "source_checkpoint_sha256": "c" * 64, "cache_manifest_sha256": "d" * 64,
            "cache_arrays_sha256": "e" * 64, "ordered_train_ids": list(range(512)),
            "ordered_train_ids_sha256": "f" * 64, "initial_source_state_sha256": "1" * 64,
            "base_encoder_sha256": "2" * 64, "initial_head_state_sha256": "3" * 64,
            "head_seed": 177001, "optimizer_groups": [{"name": "core", "lr": 3e-5}],
            "optimizer_defaults": {"lr": 3e-5}, "training_fingerprint": {"files": {"x": "4" * 64}},
            "v2_preflight_report_sha256": "5" * 64, "v2_queue_fingerprint_sha256": "6" * 64}
        try:
            for arm in evaluate.ARMS:
                manifest_hash = ("7" if arm == "signed" else "8") * 64
                manifest_hashes[f"{arm}_seed1"] = manifest_hash
                manifest = {"experiment": "SW0177_learned_coupling_polarity_training",
                    "status": "training_complete", "arm": arm, "seed": 1,
                    "completed_updates": 64, "validation_data_read": False,
                    "bindings": {**common, "arm": arm, "seed": 1,
                        "experiment": "SW0177_learned_coupling_polarity_training",
                        "head_mode": arm}, "source_model_state_before_sha256": "9" * 64,
                    "final_model_state_sha256": "a" * 64}
                manifests[arm] = manifest
                state_path = central / f"sw177_{uuid.uuid4().hex}_{arm}_state.json"
                state = {"experiment": "SW0177_learned_coupling_polarity_training_queue",
                    "arm": arm, "status": "complete", "completed_updates": 64,
                    "result_manifest_sha256": manifest_hash, "validation_data_read": False}
                state_path.write_text(json.dumps(state), encoding="utf-8")
                files.append(state_path); state_paths[arm] = state_path

            real_sha = evaluate.sha256_file
            def fake_sha(path):
                path = Path(path)
                if path.name == "manifest.json":
                    return manifest_hashes[path.parent.name]
                return real_sha(path)
            with mock.patch.object(evaluate, "_training_state_path",
                    side_effect=lambda arm: state_paths[arm]), \
                 mock.patch.object(evaluate, "sha256_file", side_effect=fake_sha), \
                 mock.patch.object(evaluate.train, "OUTPUT_ROOT", central / "artifacts"), \
                 mock.patch.object(evaluate.train, "validate_training_artifact",
                    side_effect=lambda arm, _path: manifests[arm]):
                records = evaluate._training_records()
            self.assertEqual(set(records["manifests"]), set(evaluate.ARMS))
            self.assertEqual(records["manifests"]["signed"]["bindings"]["source_proof"], source_proof)
            self.assertNotIn("source_cache_manifest_sha256", records["manifests"]["signed"]["bindings"])
        finally:
            for path in files:
                path.unlink(missing_ok=True)

    def test_bootstrap_is_paired_fixed177_and_requires_full_count(self):
        candidate = np.linspace(0, 1, evaluate.COUNT)
        got = evaluate._bootstrap(candidate, np.zeros_like(candidate))
        rng = np.random.RandomState(177)
        idx = rng.randint(0, evaluate.COUNT,
            size=(evaluate.BOOTSTRAP_REPLICATES, evaluate.COUNT))
        expected = np.percentile(candidate[idx].mean(axis=1), [2.5, 97.5])
        self.assertEqual(got["seed"], 177)
        self.assertEqual(got["replicates"], 10_000)
        np.testing.assert_array_equal(got["ci95"], expected)
        with self.assertRaises(ValueError):
            evaluate._bootstrap(candidate[:-1], np.zeros(evaluate.COUNT - 1))

    def test_load_model_registers_then_strictly_restores_real_head_checkpoint(self):
        """Exercise the loader with real modules/checkpoint bytes and a source-factory stub."""
        central = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        artifact = central / f"sw177_load_model_{uuid.uuid4().hex}"
        artifact.mkdir()

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_encoder = nn.Linear(3, 3)

        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = nn.Linear(3, 2)

        def fresh_found():
            return SimpleNamespace(wrapped=Wrapped(), encoder=Encoder(),
                patcher=nn.Identity())

        try:
            source = fresh_found()
            source.encoder.add_module("polarity_head",
                evaluate.seeded_edge_multiplier(seed=evaluate.train.SEED, mode="signed"))
            # Make the saved head observably different from its deterministic initializer.
            with torch.no_grad():
                source.encoder.polarity_head.layers[-1].weight.fill_(0.125)
                source.encoder.polarity_head.layers[-1].bias.fill_(-0.25)
            legacy = evaluate.train._legacy()
            expected_state = legacy._state_sha(source.wrapped, source.encoder)
            payload = {"experiment": evaluate.train.EXPERIMENT, "arm": "signed",
                "bindings": {"loader_fixture": "SW0177"},
                "model_state_sha256": expected_state,
                "wrapped": source.wrapped.state_dict(), "encoder": source.encoder.state_dict()}
            checkpoint = artifact / "checkpoint.pt"
            torch.save(payload, checkpoint)
            optimizer_path = artifact / "optimizer.pt"
            torch.save({"fixture": "optimizer-bytes-pinned"}, optimizer_path)
            manifest = {"bindings": payload["bindings"],
                "final_model_state_sha256": expected_state,
                "checkpoint_sha256": evaluate.sha256_file(checkpoint),
                "optimizer_sha256": evaluate.sha256_file(optimizer_path),
                "frozen_source_components_after": {"fixture": "unchanged"}}
            prepared = {"training_inputs": {"prepared": {"fixture": True}},
                "artifacts": {"signed": artifact}, "manifests": {"signed": manifest}}

            fresh = fresh_found()
            prep = evaluate.train.preflight
            with mock.patch.object(prep, "build_source", return_value=(
                    fresh, fresh.encoder.base_encoder, "source", "init")) as cpu_factory, \
                 mock.patch.object(prep, "_build_source", side_effect=AssertionError(
                    "CPU loader must not call the leased-GPU factory")), \
                 mock.patch.object(evaluate.torch.cuda, "is_initialized", return_value=False), \
                 mock.patch.object(legacy, "_frozen_source_components",
                    return_value=manifest["frozen_source_components_after"]):
                loaded, actual = evaluate._load_model("signed", prepared, "cpu")

            cpu_factory.assert_called_once_with(prepared["training_inputs"]["prepared"], "cpu")
            self.assertEqual(actual, expected_state)
            self.assertTrue(torch.equal(loaded.encoder.polarity_head.layers[-1].weight,
                                        source.encoder.polarity_head.layers[-1].weight))
            self.assertTrue(torch.equal(loaded.encoder.polarity_head.layers[-1].bias,
                                        source.encoder.polarity_head.layers[-1].bias))
            self.assertFalse(torch.equal(loaded.encoder.polarity_head.layers[-1].weight,
                evaluate.seeded_edge_multiplier(seed=evaluate.train.SEED,
                    mode="signed").layers[-1].weight))
            self.assertFalse(loaded.wrapped.training)
            self.assertFalse(loaded.encoder.training)
            self.assertFalse(loaded.patcher.training)
        finally:
            # The target is a UUID child directly under the canonical central temp root.
            self.assertEqual(artifact.resolve().parent, central.resolve())
            for child in artifact.iterdir():
                child.unlink()
            artifact.rmdir()


if __name__ == "__main__":
    unittest.main()
