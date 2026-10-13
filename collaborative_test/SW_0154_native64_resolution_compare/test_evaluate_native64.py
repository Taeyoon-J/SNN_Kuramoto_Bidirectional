from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0154_native64_resolution_compare import evaluate_native64 as evaluation
from collaborative_test.SW_0154_native64_resolution_compare.resolution64 import modal_native64

COUNT = contract.COUNT
IMAGE_IDS = contract.IMAGE_IDS


class _Dataset:
    shape = (100000, 128, 128, 1)

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key == "shape":
            return self.shape
        if isinstance(key, slice):
            return self.values[key.start - contract.IMAGE_IDS[0]:key.stop - contract.IMAGE_IDS[0]]
        return self.values[key]


class _H5:
    def __init__(self, values):
        self.mask = _Dataset(values)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __contains__(self, key):
        return key == "mask"

    def __getitem__(self, key):
        if key != "mask":
            raise KeyError(key)
        return self.mask


class Native64EvaluationTests(unittest.TestCase):
    def test_gt_is_not_opened_until_all_native_predictions_validate(self):
        events = []

        def reject(seed, **kwargs):
            events.append(("prediction", seed))
            raise ValueError("tampered native64 prediction")

        with mock.patch.object(evaluation, "validate_native_prediction", side_effect=reject), \
             mock.patch.object(evaluation.score_slot64, "prepare_all_slot_predictions") as slots, \
             mock.patch.object(evaluation.h5py, "File") as open_gt:
            with self.assertRaisesRegex(ValueError, "tampered native64"):
                evaluation.score_all(dataset=Path(__file__), prediction_root=Path("unused"),
                                     slot_root=Path("unused"))
        slots.assert_not_called()
        open_gt.assert_not_called()
        self.assertEqual(events, [("prediction", 0)])

    def test_complete_fixed320_score_uses_actual_metrics_and_gt_last(self):
        events = []
        ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
        gt = np.zeros((320, 128, 128, 1), dtype=np.uint8)
        gt[:, 8:64, 8:64, 0] = 1
        gt[:, 64:120, 64:120, 0] = 2
        labels64 = modal_native64(gt)
        dataset = Path(__file__).resolve()
        stat = dataset.stat()
        prediction_fingerprint = {"predict.py": "a" * 64}
        dummy_hashes = {}
        own, slots = {}, {}
        for seed in (0, 1, 2):
            npz = Path(f"native_seed{seed}.npz")
            proto = Path(f"native_seed{seed}.json")
            snpz = Path(f"slot_seed{seed}.npz")
            sproto = Path(f"slot_seed{seed}.json")
            native_d, native_p, slot_d, slot_p = (
                f"{seed + 1:064x}", f"{seed + 11:064x}",
                f"{seed + 21:064x}", f"{seed + 31:064x}")
            dummy_hashes.update({str(npz): native_d, str(proto): native_p,
                                 str(snpz): slot_d, str(sproto): slot_p})
            work_dir = evaluation.predict_native64.WORK_ROOT / f"seed{seed}"
            ledger = [f"{seed + index + 101:064x}" for index in range(COUNT)]
            dummy_hashes.update({str(work_dir / f"image_{image_id}.npz"): ledger[index]
                                 for index, image_id in enumerate(IMAGE_IDS)})
            protocol = {"canonical_dataset": {"path": str(dataset), "size": stat.st_size,
                                               "mtime_ns": stat.st_mtime_ns},
                        "source_checkpoint_sha256": f"{seed + 41:064x}",
                        "model_initialization_seed": 154002,
                        "work_ledger": {"directory": str(work_dir.resolve()),
                                        "completed_image_records": COUNT, "sha256": ledger}}
            own[seed] = {"seed": seed, "ids": ids, "labels": labels64,
                         "npz_path": npz, "protocol_path": proto,
                         "prediction_sha256": native_d, "protocol_sha256": native_p,
                         "protocol": protocol}
            slots[seed] = {"seed": seed, "ids": ids, "labels64": labels64,
                           "npz_path": snpz, "protocol_path": sproto,
                           "prediction_sha256": slot_d, "protocol_sha256": slot_p,
                           "protocol": {"checkpoint_sha256": f"{seed + 51:064x}"}}

        def validate(seed, **kwargs):
            events.append(("prediction", seed))
            self.assertEqual(kwargs["expected_fingerprint"], prediction_fingerprint)
            return own[seed]

        def open_file(path, mode):
            events.append(("gt_open", str(path), mode))
            return _H5(gt)

        with mock.patch.object(evaluation, "implementation_fingerprint", return_value={"eval.py": "b" * 64}), \
             mock.patch.object(evaluation.predict_native64, "implementation_fingerprint",
                               return_value=prediction_fingerprint), \
             mock.patch.object(evaluation, "validate_native_prediction", side_effect=validate), \
             mock.patch.object(evaluation.score_slot64, "prepare_all_slot_predictions", return_value=slots), \
             mock.patch.object(evaluation, "sha256_file", side_effect=lambda path: dummy_hashes[str(path)]), \
             mock.patch.object(evaluation.h5py, "File", side_effect=open_file):
            report = evaluation.score_all(dataset=dataset, prediction_root=Path("unused"),
                                          slot_root=Path("unused"))

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["seed_scope"], [0, 1, 2])
        self.assertEqual(set(report["means"]["native64"]), set(contract.METRICS))
        self.assertEqual(report["means"]["native64"]["fg_ari"]["three_seed_mean"], 1.0)
        self.assertTrue(report["ground_truth_opened_after_all_six_predictions_validated"])
        first_gt = next(i for i, item in enumerate(events) if item[0] == "gt_open")
        self.assertEqual([item[0] for item in events[:first_gt]], ["prediction"] * 3)
        self.assertEqual(len(events[first_gt:]), 1)
        self.assertEqual(report["paired_native64_minus_slot64"]["1"]["fg_ari"]["mean_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
