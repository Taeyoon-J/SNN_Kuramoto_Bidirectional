import unittest
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0187_joint_encoder_unique_scaling import predict_evaluate as pe


class PredictionContracts(unittest.TestCase):
    def test_prediction_fingerprint_binds_both_model_arms_and_test_contracts(self):
        fingerprint = pe.implementation_fingerprint()
        self.assertEqual(fingerprint["arms"], ["control512", "candidate1024"])
        self.assertIn("collaborative_test/SW_0187_joint_encoder_unique_scaling/train.py",
            fingerprint["files"])
        self.assertIn("collaborative_test/SW_0187_joint_encoder_unique_scaling/prediction_queue.py",
            fingerprint["files"])

    def test_worker_cli_help(self):
        with self.assertRaises(SystemExit) as result: pe.main(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_strict_loader_uses_fresh_factory_components_with_real_producer_manifest(self):
        import torch
        from collaborative_test.SW_0187_joint_encoder_unique_scaling import train

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.base_encoder = torch.nn.Linear(2, 2); self.proj = torch.nn.Linear(2, 2)

        wrapped, encoder = torch.nn.Linear(2, 2), Encoder()
        params = list(wrapped.parameters()) + list(encoder.proj.parameters())
        optimizer = torch.optim.Adam([
            dict(params=params,name="fixture_group",parameter_names=[f"fixture_{i}" for i in range(len(params))]),
            dict(params=list(encoder.base_encoder.parameters()),name="source_base",parameter_names=["base.weight","base.bias"])])
        found = SimpleNamespace(wrapped=wrapped, encoder=encoder, patcher=torch.nn.Identity())
        factory_binding = {"initial_model_state_sha256": train.source._state_sha(wrapped, encoder),
            "source_binding": {"fixture": "seed-specific"}}
        import copy
        initial_base_sha=train.source._base_encoder_sha(found)
        fresh_found=copy.deepcopy(found)
        fresh_optimizer=torch.optim.Adam(list(fresh_found.wrapped.parameters())+list(fresh_found.encoder.parameters()))
        sum(parameter.square().sum() for parameter in list(wrapped.parameters())+list(encoder.parameters())).backward()
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        fresh_components = {"frozen_base": "producer-factory-snapshot"}
        utils = SimpleNamespace(_optimizer_summary=lambda _optimizer: {"state_entries": 0},
            _frozen_source_components=lambda _found, _base: fresh_components)

        parent = Path(train.ROOT) / "collaborative_testing" / "temporary" / f"sw183-loader-{uuid.uuid4().hex}"
        stage, work = parent / "artifact", parent / "work"
        stage.mkdir(parents=True); work.mkdir()
        (work / "boundary.ptx").write_bytes(b"producer-fixture-boundary")
        history = [{"proposal": i + 1, "image_indices": list(range(16))} for i in range(64)]
        inputs = {"ordered_train_ids": list(range(1024)), "dataset_manifest_sha256": "a" * 64,
            "dataset_arrays_sha256": "b" * 64}
        bindings = {"initial_model_state_sha256": factory_binding["initial_model_state_sha256"],
            "factory_binding": factory_binding,"base_encoder_sha256":initial_base_sha}
        train._write_state_bundle(stage, "control512", 0, history, bindings, work, inputs,
            accepted=1, proposals=64, prefix_state_sha="c" * 64, prefix_rng_sha="d" * 64,
            model_parts=(found, optimizer, {}))
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("frozen_source_components_after", manifest)
        prepared = {"seed": 0, "training_inputs": object(), "artifacts": {
            "control512": {"manifest": manifest, "path": stage}}}
        setup = (None, None, fresh_found.encoder.base_encoder, fresh_components)
        with mock.patch.object(train.source, "_make_fresh",
                return_value=(fresh_found, fresh_optimizer, None, factory_binding, setup)), \
             mock.patch.object(train.source, "_module", return_value=utils):
            loaded, actual_sha = pe._load_model(0, "control512", prepared, "cpu")
        self.assertIs(loaded, fresh_found)
        self.assertEqual(actual_sha, manifest["model_state_sha256"])

    def test_conditional_and_standalone_foregrounds_are_independent(self):
        source = np.zeros((pe.COUNT, pe.GRID, pe.GRID), dtype=bool)
        support = np.zeros((pe.COUNT, pe.GRID, pe.GRID), dtype=np.int32)
        conditional = np.zeros_like(support)
        standalone = np.zeros_like(support)
        source[:, :4, :5] = True
        conditional[source] = 17
        support[:, 9:12, 14:18] = 1
        standalone[:, 9:12, 14:18] = 23
        labels, checked_support = pe.native.validate_prediction_arrays(
            {"conditional": conditional, "standalone": standalone}, support, source)
        self.assertTrue(np.array_equal(labels["conditional"] > 0, source))
        self.assertTrue(np.array_equal(labels["standalone"] > 0, checked_support > 0))
        self.assertFalse(np.array_equal(labels["standalone"] > 0, source))

    def test_wrong_endpoint_foreground_is_rejected(self):
        source = np.zeros((pe.COUNT, pe.GRID, pe.GRID), dtype=bool)
        support = np.zeros((pe.COUNT, pe.GRID, pe.GRID), dtype=np.int32)
        labels = {key: np.zeros_like(support) for key in pe.ENDPOINTS}
        support[:, 1, 1] = 1; labels["standalone"][:, 1, 1] = 3
        labels["conditional"][:, 2, 2] = 2
        with self.assertRaises(ValueError):
            pe.native.validate_prediction_arrays(labels, support, source)


if __name__ == "__main__": unittest.main()
