from __future__ import annotations

import copy
import hashlib
import unittest

import numpy as np
import torch

from collaborative_test.SW_0150_spike_spatial_mixture.preflight import (
    SEED, transplant_decoder_adam, validate_saved_parity_payload,
)


def ids_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def parity_fixture():
    ids = list(range(20000, 24096))
    fingerprint = {"source.py": "a" * 64}
    differences = {name: {"exact": True} for name in
                   ("theta", "membrane", "spikes", "component_spikes", "q", "old_objective")}
    horizon64 = {"steps": 64, "settle": 32, "exact_parity": True,
                 "differences": copy.deepcopy(differences)}
    horizon1024 = {"steps": 1024, "settle": 512, "exact_parity": True,
                   "differences": copy.deepcopy(differences)}
    payload = {
        "experiment": "SW0135_native32_spike_binding",
        "stage": "native32_zero_adapter_parity",
        "status": "native32_zero_adapter_parity_passed",
        "source_seed": SEED,
        "source_core_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "training_image_ids": ids[:4],
        "training_image_ids_sha256": ids_digest(ids[:4]),
        "registered_train_order_sha256": ids_digest(ids),
        "rgb_asset_validation": {"train_cache_sha256": "d" * 64},
        "implementation_fingerprint": fingerprint,
        "gamma_shape": [4, 8, 1024],
        "microbatch_size": 4,
        "ground_truth_used": False,
        "masks_read": False,
        "optimizer_updates": 0,
        "training_admission": False,
        "horizons": [{"steps": 64, "settle": 32}, {"steps": 1024, "settle": 512}],
        "arms": {"phase": [horizon64, horizon1024],
                 "constant": [copy.deepcopy(horizon64), copy.deepcopy(horizon1024)]},
    }
    return payload, ids, fingerprint


class PreflightContractTests(unittest.TestCase):
    def test_saved_source_parity_payload_is_bound_to_current_lineage(self):
        payload, ids, fingerprint = parity_fixture()
        self.assertTrue(validate_saved_parity_payload(
            payload, source_sha="b" * 64, source_manifest_sha="c" * 64,
            training_ids=ids, asset_validation=payload["rgb_asset_validation"],
            expected_fingerprint=fingerprint))

        bad = copy.deepcopy(payload)
        bad["registered_train_order_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_saved_parity_payload(
                bad, source_sha="b" * 64, source_manifest_sha="c" * 64,
                training_ids=ids, asset_validation=payload["rgb_asset_validation"],
                expected_fingerprint=fingerprint)

    def test_saved_source_parity_rejects_nonexact_t1024_field(self):
        payload, ids, fingerprint = parity_fixture()
        payload["arms"]["phase"][1]["differences"]["q"]["exact"] = False
        with self.assertRaisesRegex(ValueError, "non-exact parity field"):
            validate_saved_parity_payload(
                payload, source_sha="b" * 64, source_manifest_sha="c" * 64,
                training_ids=ids, asset_validation=payload["rgb_asset_validation"],
                expected_fingerprint=fingerprint)

    def test_warm_decoder_adam_moments_transplant_into_named_group(self):
        torch.manual_seed(1505)
        warm_decoder = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(),
                                           torch.nn.Linear(5, 2))
        warm_parameters = list(warm_decoder.parameters())
        warm = torch.optim.Adam(warm_parameters, lr=1e-3)
        for step in range(32):
            x = torch.full((4, 3), (step + 1) / 32.0)
            warm.zero_grad(set_to_none=True)
            warm_decoder(x).square().mean().backward()
            warm.step()

        native = torch.nn.Linear(2, 2)
        pilot_decoder = copy.deepcopy(warm_decoder)
        pilot_parameters = list(pilot_decoder.parameters())
        pilot = torch.optim.Adam([
            {"params": list(native.parameters()), "lr": 3e-5, "name": "graph"},
            {"params": pilot_parameters, "lr": 1e-3, "name": "decoder"},
        ])
        cpu_rng = torch.random.get_rng_state().clone()
        copied = transplant_decoder_adam(warm, pilot, warm_parameters, pilot_parameters)
        self.assertEqual(copied, len(warm_parameters))
        self.assertTrue(torch.equal(cpu_rng, torch.random.get_rng_state()))
        for source, destination in zip(warm_parameters, pilot_parameters):
            self.assertEqual(int(warm.state[source]["step"]), 32)
            self.assertEqual(int(pilot.state[destination]["step"]), 32)
            self.assertTrue(torch.equal(warm.state[source]["exp_avg"],
                                        pilot.state[destination]["exp_avg"]))

        for _ in range(2):
            pilot.zero_grad(set_to_none=True)
            decoded = pilot_decoder(torch.ones((4, 3)))
            native(decoded).sum().backward()
            pilot.step()
        self.assertEqual(int(pilot.state[pilot_parameters[0]]["step"]), 34)
        self.assertEqual(int(pilot.state[native.weight]["step"]), 2)
        self.assertEqual([group["name"] for group in pilot.param_groups], ["graph", "decoder"])

    def test_warm_adam_transplant_rejects_wrong_decoder_group(self):
        source = torch.nn.Linear(2, 2)
        target = torch.nn.Linear(2, 2)
        warm = torch.optim.Adam(source.parameters(), lr=1e-3)
        warm.zero_grad(set_to_none=True)
        source(torch.ones((2, 2))).sum().backward()
        warm.step()
        wrong = torch.optim.Adam([{"params": list(target.parameters()), "name": "not_decoder"}])
        with self.assertRaisesRegex(ValueError, "decoder group"):
            transplant_decoder_adam(warm, wrong, list(source.parameters()),
                                    list(target.parameters()))

    def test_warm_adam_transplant_rejects_same_length_reordered_parameters(self):
        source = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2))
        target = copy.deepcopy(source)
        warm_params = list(source.parameters())
        warm = torch.optim.Adam(warm_params, lr=1e-3)
        warm.zero_grad(set_to_none=True)
        source(torch.ones((2, 3))).sum().backward()
        warm.step()
        target_params = list(target.parameters())
        pilot = torch.optim.Adam([{"params": target_params, "name": "decoder"}], lr=1e-3)

        with self.assertRaisesRegex(ValueError, "registered decoder parameters"):
            transplant_decoder_adam(warm, pilot, list(reversed(warm_params)), target_params)
        with self.assertRaisesRegex(ValueError, "matching decoder group"):
            transplant_decoder_adam(warm, pilot, warm_params, list(reversed(target_params)))


if __name__ == "__main__":
    unittest.main()
