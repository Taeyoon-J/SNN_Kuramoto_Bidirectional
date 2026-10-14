from __future__ import annotations

# Import NumPy before Torch for the supported Windows CPU-test environment.
import numpy as np
import torch
import unittest
import json
import uuid
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0159_one_shot_event_regime import train


class _Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = torch.nn.Linear(2, 2, bias=False)
        self.native = torch.nn.Linear(2, 2, bias=False)


class _Wrapped(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()
        self.a_d = torch.nn.Parameter(torch.ones(4))
        self.a_m = torch.nn.Parameter(torch.ones(4))
        self.b = torch.nn.Parameter(torch.zeros(4))


class _Foundation:
    def __init__(self):
        self.encoder = torch.nn.Linear(2, 2)
        self.patcher = torch.nn.Linear(2, 2)
        self.wrapped = _Wrapped()


class SW0159TrainTests(unittest.TestCase):
    def test_trainability_freezes_encoder_and_t_graph_only(self):
        foundation = _Foundation()
        families = train.configure_trainability(foundation)
        self.assertEqual(set(families), {"native_core", "integration_tensors"})
        self.assertFalse(any(p.requires_grad for p in foundation.encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in foundation.patcher.parameters()))
        self.assertFalse(any(p.requires_grad for p in foundation.wrapped.core.graph_generator.parameters()))
        self.assertTrue(all(p.requires_grad for p in families["native_core"]))
        self.assertTrue(all(p.requires_grad for p in families["integration_tensors"]))
        self.assertEqual(sum(p.numel() for p in families["integration_tensors"]), 12)
        optimizer = train.make_optimizer(families)
        self.assertEqual([g["name"] for g in optimizer.param_groups],
                         ["native_core", "integration_tensors"])
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [3e-5, 1e-3])

    def test_one_shot_arm_initialization_sets_actual_physical_threshold(self):
        rms = train.calibration.SOURCE_MARGIN_RMS
        init = train.calibration.one_shot_bias(rms)
        record = {"initialization": init}
        candidate, control = _Foundation(), _Foundation()
        actual_candidate = train.initialize_arm(candidate, "candidate", record)
        actual_control = train.initialize_arm(control, "control", record)
        self.assertTrue(np.allclose(actual_candidate, init["thresholds"], rtol=2e-6, atol=1e-6))
        self.assertTrue(np.allclose(actual_control, [0.06] * 4, rtol=2e-6, atol=1e-7))
        self.assertTrue(torch.equal(control.wrapped.b, torch.zeros(4)))
        with self.assertRaisesRegex(ValueError, "unknown"):
            train.initialize_arm(candidate, "other", record)

    def test_parameter_hash_supports_scalar_parameters(self):
        self.assertEqual(len(train._parameter_hash([torch.nn.Parameter(torch.tensor(1.0))])), 64)

    def test_artifact_validator_accepts_actual_complete_manifest_and_rejects_tamper(self):
        root = train.HERE / f".test-artifact-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            out = root / "candidate_seed1"
            out.mkdir()
            preflight_path = root / "preflight.json"
            preflight_path.write_text("{}", encoding="utf-8")
            hdf = root / "train.h5"
            hdf.write_bytes(b"fixture hdf identity")
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text("{}", encoding="utf-8")
            ids = list(range(1000)) + list(range(1640, 4736))
            source = {"image_ids": ids, "rows": np.arange(4096, dtype=np.int64),
                      "source_sha256": "a" * 64, "manifest_path": source_manifest}
            assets = {"train_cache_sha256": "b" * 64}
            fp = {"fixture": "c" * 64}
            init = train.calibration.one_shot_bias(train.calibration.SOURCE_MARGIN_RMS)
            with mock.patch.object(train.sw155, "source_contract", return_value=source), \
                 mock.patch.object(train.sw130, "validate_rgb_assets", return_value=assets), \
                 mock.patch.object(train, "validate_preflight_admission", return_value={
                     "implementation_fingerprint": {"preflight": "d" * 64},
                     "initialization": init}), \
                 mock.patch.object(train, "implementation_fingerprint", return_value=fp):
                hstat = hdf.stat()
                bindings = {
                    "experiment": train.EXPERIMENT, "arm": "candidate", "seed": 1,
                    "source_checkpoint_sha256": "a" * 64,
                    "source_manifest_sha256": train.sha256_file(source_manifest),
                    "source_training_ids_sha256": train.sw155.ids_sha(ids),
                    "ordered_train_ids_sha256": train.sw155.ids_sha(ids[:2048]),
                    "rgb_asset_validation": assets,
                    "preflight_sha256": train.sha256_file(preflight_path),
                    "preflight_implementation_fingerprint": {"preflight": "d" * 64},
                    "hdf5_path": str(hdf.resolve()), "hdf5_size_bytes": hstat.st_size,
                    "hdf5_mtime_ns": hstat.st_mtime_ns, "updates": train.UPDATES,
                    "logical_batch": train.BATCH, "microbatch": train.MICRO,
                    "steps": train.STEPS, "settle": train.SETTLE, "live_tail": train.TAIL,
                    "initial_thresholds": list(init["thresholds"]),
                    "ground_truth_used": True, "ground_truth_scope": "registered TRAIN masks only",
                    "implementation_fingerprint": fp,
                    "calibration_sha256": train.calibration.REPORT_SHA256,
                }
                checkpoint = out / "checkpoint.pt"
                torch.save({"wrapped": {"weight": torch.ones(1)},
                            "encoder": {"weight": torch.ones(1)}}, checkpoint)
                parameters = [torch.nn.Parameter(torch.tensor([1.0])),
                              torch.nn.Parameter(torch.tensor([2.0]))]
                optimizer = torch.optim.Adam([
                    {"name": "native_core", "params": [parameters[0]], "lr": 3e-5},
                    {"name": "integration_tensors", "params": [parameters[1]], "lr": 1e-3}])
                for _ in range(train.UPDATES):
                    optimizer.zero_grad()
                    loss = (parameters[0] - 0.0).square().sum() + (parameters[1] - 0.0).square().sum()
                    loss.backward(); optimizer.step()
                optimizer_path = out / "optimizer.pt"
                torch.save(optimizer.state_dict(), optimizer_path)
                history = [{"update": index + 1,
                    "image_ids": ids[index * train.BATCH:(index + 1) * train.BATCH],
                    "mean_loss": 1.0,
                    "family_gradient_norms": {"native_core": 0.1, "integration_tensors": 0.2}}
                    for index in range(train.UPDATES)]
                history_path = out / "history.json"
                history_path.write_text(json.dumps(history), encoding="utf-8")
                actual_steps = {group["name"]: max(
                    int(optimizer.state[p]["step"].item()) for p in group["params"])
                    for group in optimizer.param_groups}
                manifest = {
                    "status": "training_complete", "experiment": train.EXPERIMENT,
                    "arm": "candidate", "seed": 1, "completed_updates": train.UPDATES,
                    "logical_batch": train.BATCH, "microbatch": train.MICRO,
                    "exposures": train.BATCH * train.UPDATES, "train_ids": ids[:2048],
                    "bindings": bindings, "initial_thresholds": list(init["thresholds"]),
                    "initial_trainable_parameter_sha256": {"native_core": "e" * 64,
                                                               "integration_tensors": "f" * 64},
                    "final_trainable_parameter_sha256": {"native_core": "1" * 64,
                                                             "integration_tensors": "2" * 64},
                    "frozen_encoder_sha256": "3" * 64, "frozen_graph_sha256": "4" * 64,
                    "checkpoint_sha256": train.sha256_file(checkpoint),
                    "optimizer_sha256": train.sha256_file(optimizer_path),
                    "history_sha256": train.sha256_file(history_path),
                    "implementation_fingerprint": fp,
                    "optimizer_step_summary": actual_steps,
                    "ground_truth_used": True,
                    "ground_truth_scope": "registered TRAIN masks only"}
                (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                self.assertTrue(train.validate_training_artifact("candidate", directory=out,
                    preflight_path=preflight_path, hdf5_path=hdf, source=source, assets=assets))
                with history_path.open("a", encoding="utf-8") as stream:
                    stream.write(" ")
                self.assertFalse(train.validate_training_artifact("candidate", directory=out,
                    preflight_path=preflight_path, hdf5_path=hdf, source=source, assets=assets))
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
