from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0155_supervised_spike_relations import artifactvalidator, preflight_migration, train


class SW0155ArtifactValidatorTests(unittest.TestCase):
    def test_actual_training_complete_schema_and_payload_hashes(self):
        ids = list(range(20000, 24096))
        source_manifest = Path("/fixture/source.json")
        preflight = Path("/fixture/preflight.json")
        hdf = Path("/fixture/train.h5")
        assets = {"train_cache_sha256": "c" * 64}
        source = {"source_sha256": "a" * 64, "manifest_path": source_manifest,
                  "image_ids": ids}
        output = Path("/fixture/joint_seed1")
        rows = [{"update": index + 1,
                 "image_ids": ids[index * 16:(index + 1) * 16],
                 "mean_loss": 1.0,
                 "family_gradient_norms": {"spatial_residual": 1.0, "graph": 1.0,
                                            "native_core": 1.0, "integration_tensors": 1.0}}
                for index in range(128)]
        fp = {"train.py": "f" * 64}
        changed = {"spatial_residual": "1" * 64, "graph": "2" * 64,
                   "native_core": "3" * 64, "integration_tensors": "4" * 64}
        initial = {key: "0" * 64 for key in changed}
        hashes = {"checkpoint": "6" * 64, "optimizer": "7" * 64, "history": "8" * 64}
        bindings = {"experiment": train.EXPERIMENT, "arm": "joint", "seed": 1,
            "source_checkpoint_sha256": source["source_sha256"],
            "source_manifest_sha256": "9" * 64,
            "source_training_ids_sha256": train.ids_sha(ids),
            "ordered_train_ids_sha256": train.ids_sha(ids[:2048]),
            "rgb_asset_validation": assets, "preflight_sha256": "a" * 64,
            "hdf5_path": str(hdf.resolve()), "hdf5_size": 100, "hdf5_mtime_ns": 200,
            "steps": train.STEPS, "settle": train.SETTLE, "live_tail": train.TAIL,
            "ground_truth_used": True, "ground_truth_scope": "TRAIN masks only",
            "implementation_fingerprint": fp}
        manifest = {"status": "training_complete", "experiment": train.EXPERIMENT,
            "arm": "joint", "seed": 1, "completed_updates": 128,
            "logical_batch": 16, "microbatch": 4, "exposures": 2048,
            "train_ids": ids[:2048], "bindings": bindings,
            "initial_parameter_sha256": initial, "final_parameter_sha256": changed,
            "frozen_base_encoder_sha256_before_after": "5" * 64,
            **{f"{name}_sha256": digest for name, digest in hashes.items()},
            "implementation_fingerprint": fp,
            "optimizer_step_summary": {key: 128 for key in changed},
            "ground_truth_used": True, "ground_truth_scope": "TRAIN masks only"}
        payloads = {"manifest.json": json.dumps(manifest), "history.json": json.dumps(rows)}
        digest_by_name = {"source.json": "9" * 64, "preflight.json": "a" * 64,
                          preflight_migration.MIGRATED_REPORT_NAME: "a" * 64,
                          "checkpoint.pt": hashes["checkpoint"], "optimizer.pt": hashes["optimizer"],
                          "history.json": hashlib.sha256(payloads["history.json"].encode()).hexdigest()}
        manifest["history_sha256"] = digest_by_name["history.json"]
        payloads["manifest.json"] = json.dumps(manifest)
        def read_text(path, *args, **kwargs):
            return payloads[path.name]

        def sha(path):
            return digest_by_name[Path(path).name]

        fake_stat = SimpleNamespace(st_size=100, st_mtime_ns=200)
        with mock.patch.object(train, "source_contract", return_value=source), \
             mock.patch.object(train, "implementation_fingerprint", return_value=fp), \
             mock.patch.object(train, "PREFLIGHT_PATH", preflight), \
             mock.patch.object(train, "HDF5_PATH", hdf), \
             mock.patch.object(train, "sha256_file", side_effect=sha), \
             mock.patch("collaborative_test.SW_0130_phase_state_integration.run.validate_rgb_assets",
                        return_value=assets), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "read_text", autospec=True, side_effect=read_text), \
             mock.patch.object(Path, "stat", return_value=fake_stat):
            task = {"arm": "joint", "output": str(output)}
            self.assertTrue(artifactvalidator.validate_training_artifact(task))
            manifest["status"] = "complete"  # The old queue's invented status is rejected.
            payloads["manifest.json"] = json.dumps(manifest)
            self.assertFalse(artifactvalidator.validate_training_artifact(task))


if __name__ == "__main__":
    unittest.main()
