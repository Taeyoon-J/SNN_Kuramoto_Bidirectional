"""CPU regressions for the exact SW0130 diagnostic perturbations and evidence."""
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
sys.path[:] = [entry for entry in sys.path
               if not entry or Path(entry).resolve() != HERE]

import numpy as np
import torch

from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import (
    SharedRGBDecoder, reconstruct_one,
)
from collaborative_test.SW_0131_decoder_partition_diagnosis import run


class DecoderDiagnosisTests(unittest.TestCase):
    def test_row_shuffle_matches_registered_torch_generator(self):
        torch.manual_seed(12)
        q = torch.rand(256, 256)
        labels = torch.arange(256) % 4
        hard = torch.nn.functional.one_hot(labels, num_classes=4).float()
        features = torch.rand(256, 8)
        target = torch.rand(256, 3)
        decoder = SharedRGBDecoder()
        actual = run._reconstruct_condition(q, hard, features, target, decoder,
                                            "row_shuffled", [100, 101], 17)
        generator = torch.Generator(device="cpu").manual_seed(147)
        expected_permutation = torch.randperm(256, generator=generator)
        expected = reconstruct_one(q, hard[expected_permutation], features, target,
                                   decoder, True)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_image_mean_content_is_assignment_independent_in_forward(self):
        torch.manual_seed(15)
        q = torch.rand(256, 256)
        labels_a = torch.arange(256) % 3
        labels_b = (torch.arange(256) // 3) % 3
        hard_a = torch.nn.functional.one_hot(labels_a, num_classes=3).float()
        hard_b = torch.nn.functional.one_hot(labels_b, num_classes=3).float()
        features = torch.rand(256, 8)
        target = torch.rand(256, 3)
        decoder = SharedRGBDecoder()
        pred_a, loss_a, _ = run._reconstruct_condition(q, hard_a, features, target,
            decoder, "image_mean_content", list(range(256)), 0)
        pred_b, loss_b, _ = run._reconstruct_condition(q, hard_b, features, target,
            decoder, "image_mean_content", list(range(256)), 0)
        self.assertTrue(torch.allclose(pred_a, pred_b, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(loss_a, loss_b, rtol=1e-6, atol=1e-7))

    def test_failed_seed_can_bind_audited_warm_decoder_without_preflight_json(self):
        root = ROOT / f"sw131_fixture_{uuid.uuid4().hex[:8]}"
        archive = root / "archive"
        archive.mkdir(parents=True)
        warm_path = archive / "preflight_decoder_seed2.pt"
        payload = {
            "decoder_state_dict": {"weight": torch.ones(2)},
            "optimizer_state_dict": {"state": {
                index: {"step": torch.tensor(32.)} for index in range(6)},
                "param_groups": []},
        }
        torch.save(payload, warm_path)
        digest = run.sw130.sha(warm_path)
        audit_path = root / "warm_decoder_source_audit_20261009.json"
        audit_path.write_text(json.dumps([{
            "seed": 2, "sha256": digest,
            "schema": ["decoder_state_dict", "optimizer_state_dict"],
            "adam_steps": [32.0] * 6, "all_decoder_finite": True,
        }]), encoding="utf-8")
        expected = dict(run.EXPECTED_WARM_SHA)
        expected[2] = digest
        try:
            with mock.patch.object(run, "HERE", root), \
                    mock.patch.object(run.sw130, "ARCHIVE", archive), \
                    mock.patch.object(run, "EXPECTED_WARM_SHA", expected):
                preflight_path, report, actual_path, actual_sha, loaded, actual_audit, audit_sha = \
                    run._warmup_inputs(2)
            self.assertIsNone(preflight_path)
            self.assertIsNone(report)
            self.assertEqual(actual_path, warm_path)
            self.assertEqual(actual_sha, digest)
            self.assertEqual(actual_audit, audit_path)
            self.assertEqual(audit_sha, run.sw130.sha(audit_path))
            self.assertEqual(len(loaded["optimizer_state_dict"]["state"]), 6)
        finally:
            warm_path.unlink(missing_ok=True)
            audit_path.unlink(missing_ok=True)
            archive.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
