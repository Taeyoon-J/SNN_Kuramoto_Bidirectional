import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0158_membrane_spike_credit import spike_credit, train


class TrainContractTests(unittest.TestCase):
    @staticmethod
    def foundation():
        from collaborative_test.SW_0158_membrane_spike_credit.test_spike_credit import (
            MembraneSpikeCreditTests)
        wrapped = MembraneSpikeCreditTests.tiny_wrapped()
        spike_credit.install_spike_credit(wrapped, spike_credit.SOURCE_T_MARGIN_RMS)
        return SimpleNamespace(wrapped=wrapped,
            encoder=torch.nn.Linear(2, 2), patcher=torch.nn.Linear(2, 2))

    def test_parameter_partition_and_optimizer_groups_are_source_t_only(self):
        foundation = self.foundation()
        families, optimizer = train._configure(foundation)
        self.assertEqual(set(families), {"native_core", "integration_tensors", "eta"})
        self.assertEqual([group["name"] for group in optimizer.param_groups],
                         ["native_core", "integration_tensors", "eta"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [3e-5, 1e-3, 1e-3])
        self.assertTrue(all(parameter.requires_grad
                            for group in families.values() for parameter in group))
        self.assertFalse(any(parameter.requires_grad for parameter in foundation.encoder.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in foundation.patcher.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in
                             foundation.wrapped.core.graph_generator.parameters()))

    def test_durable_checkpoint_restores_eta_and_fixed_component_scales(self):
        foundation = self.foundation()
        families, optimizer = train._configure(foundation)
        # Materialize genuine Adam moments for the new eta parameter.
        optimizer.zero_grad(set_to_none=True)
        (foundation.wrapped.sw0158_eta.square().sum()).backward()
        optimizer.step()
        bindings = {"experiment": train.EXPERIMENT, "preflight": "p" * 64,
                    "source_t_calibration": spike_credit.SW157_REPORT_SHA256}
        runtime = durable.runtime_signature(torch.device("cpu"))
        # Keep the scratch checkpoint under the writable repository tree.  On
        # some managed Windows installs the system temp directory denies
        # creation even though the checked-out workspace is writable.
        with tempfile.TemporaryDirectory(prefix="sw0158_boundary_",
                                         dir=train.HERE) as tmp:
            work = Path(tmp) / "checkpoints"
            work.mkdir()
            durable.save_boundary(work, completed_updates=1,
                wrapped=foundation.wrapped, encoder=foundation.encoder,
                optimizer=optimizer, history=[{"update": 1}], bindings=bindings,
                runtime=runtime, device=torch.device("cpu"))
            eta_saved = foundation.wrapped.sw0158_eta.detach().clone()
            scale_saved = foundation.wrapped.sw0158_margin_rms.detach().clone()
            with torch.no_grad():
                foundation.wrapped.sw0158_eta.add_(.25)
                foundation.wrapped.sw0158_margin_rms.mul_(2)
            loaded = durable.load_latest(work, wrapped=foundation.wrapped,
                encoder=foundation.encoder, optimizer=optimizer,
                expected_bindings=bindings, expected_runtime=runtime,
                device=torch.device("cpu"), max_updates=128)
            self.assertEqual(loaded["completed_updates"], 1)
            self.assertEqual(len(loaded["history"]), 1)
            self.assertTrue(torch.equal(foundation.wrapped.sw0158_eta, eta_saved))
            self.assertTrue(torch.equal(foundation.wrapped.sw0158_margin_rms, scale_saved))
            self.assertGreater(int(optimizer.state[foundation.wrapped.sw0158_eta]["step"]), 0)

    def test_training_fingerprint_resolves_source_and_durable_dependencies(self):
        fingerprint = train.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0158_membrane_spike_credit/preflight.py", fingerprint)
        self.assertIn("collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py", fingerprint)
        self.assertIn("collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
                      fingerprint)

    def test_registered_matched_training_arms_and_rejects_unregistered_arm(self):
        self.assertEqual(train.ARMS, ("live_membrane_credit", "detached_membrane_control"))
        with self.assertRaisesRegex(ValueError, "unregistered SW0158 arm"):
            train.train_arm("other", device="cpu")


if __name__ == "__main__":
    unittest.main()
