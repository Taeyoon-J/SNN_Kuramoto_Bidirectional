"""Focused fixed320 GT-last score path regression."""
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0152_source_preserving_partition_equivariance import evaluate


class _MaskDataset:
    shape = (1640, 128, 128)

    def __init__(self, pixels):
        self.pixels = pixels

    def __getitem__(self, key):
        assert key == slice(1320, 1640)
        return self.pixels


class _H5:
    def __init__(self, pixels):
        self.values = {"mask": _MaskDataset(pixels)}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __contains__(self, key):
        return key in self.values

    def __getitem__(self, key):
        return self.values[key]


class ScoreAllTests(unittest.TestCase):
    def _fixture(self):
        # Two foreground objects per image make all three production metrics valid.
        patch = np.zeros((32, 32), dtype=np.uint8)
        patch[5:20, 5:14] = 1
        patch[5:20, 18:27] = 2
        target = np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)
        target = np.repeat(target[None], 320, axis=0)
        labels = np.repeat(patch[None], 320, axis=0).astype(np.int64)
        ids = np.arange(1320, 1640, dtype=np.int64)
        folder = HERE / "results_archive" / f"score_fixture_{os.getpid()}_{uuid.uuid4().hex}"
        folder.mkdir(parents=True)
        pred = {}
        for name in (*evaluate.ARMS, "source_p"):
            n = folder / f"{name}.npz"
            p = folder / f"{name}.json"
            n.write_bytes(name.encode("ascii") + b"-npz")
            p.write_bytes(name.encode("ascii") + b"-protocol")
            pred[name] = {"ids": ids, "qcc": labels, "p": labels,
                "protocol": {"training_manifest_sha256": None, "checkpoint_sha256": None},
                "npz": n, "protocol_path": p, "prediction_sha256": evaluate.sha(n),
                "protocol_sha256": evaluate.sha(p)}
        source_npz = folder / "source-ref.npz"
        source_protocol = folder / "source-ref.json"
        source_npz.write_bytes(b"source-qcc")
        source_protocol.write_bytes(b"source-protocol")
        source = {"labels": labels, "archive_path": source_npz,
                  "protocol_path": source_protocol, "prediction_sha256": evaluate.sha(source_npz),
                  "protocol_sha256": evaluate.sha(source_protocol)}
        slot_npz = folder / "slot.npz"
        slot_protocol = folder / "slot.json"
        slot_npz.write_bytes(b"slot-prediction")
        slot_protocol.write_bytes(b"slot-protocol")
        slot = {"ids": ids, "labels": labels, "npz": slot_npz, "protocol": slot_protocol,
                "prediction_sha256": evaluate.sha(slot_npz), "protocol_sha256": evaluate.sha(slot_protocol)}
        return folder, target, pred, source, slot

    def test_complete_score_uses_production_metrics_and_freezes_all_predictions_before_gt(self):
        folder, target, predictions, source, slot = self._fixture()
        calls = []
        def validate(name, **_kwargs):
            calls.append(name)
            return predictions[name]
        output = folder / "score.json"
        try:
            with mock.patch.object(evaluate, "validate_prediction", side_effect=validate), \
                 mock.patch.object(evaluate.sw137, "validate_prediction", return_value=source), \
                 mock.patch.object(evaluate, "_slot_reference", return_value=slot), \
                 mock.patch.object(evaluate.h5py, "File", side_effect=lambda *_a, **_k: (calls.append("GT") or _H5(target))):
                result = evaluate.score_all(output=output, dataset=folder / "fake.h5")
            self.assertEqual(calls[:3], ["equivariance", "mass_control", "source_p"])
            self.assertEqual(calls[3], "GT")
            self.assertEqual(set(result["scores"]), {"equivariance", "mass_control", "source_p", "source97_qcc", "slot32_seed1"})
            self.assertEqual(set(result["scores"]["equivariance"]["qcc"]), set(evaluate.contract.METRICS))
            self.assertEqual(set(result["promotion_gate"]), {
                "equivariance_qcc_fg_at_least_source_T_plus_0_02",
                "equivariance_vs_source_qcc_fg_ci_lower_positive",
                "equivariance_vs_control_fg_ci_lower_positive"})
            self.assertTrue(result["ground_truth_opened_after_all_predictions_validated"])
            self.assertFalse(result["ground_truth_used_for_prediction"])
        finally:
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()

    def test_invalid_candidate_is_rejected_before_ground_truth_open(self):
        folder, _target, _predictions, _source, _slot = self._fixture()
        calls = []
        output = folder / "score.json"
        try:
            with mock.patch.object(evaluate, "validate_prediction", side_effect=lambda name, **_kw: (
                    calls.append(name), (_ for _ in ()).throw(ValueError("tampered")))[1]), \
                 mock.patch.object(evaluate.h5py, "File", side_effect=lambda *_a, **_k: calls.append("GT")):
                with self.assertRaisesRegex(ValueError, "tampered"):
                    evaluate.score_all(output=output, dataset=folder / "fake.h5")
            self.assertNotIn("GT", calls)
            self.assertFalse(output.exists())
        finally:
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()


if __name__ == "__main__":
    unittest.main()
