"""Focused provenance regressions for SW0146 admission parsing."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))
import numpy as np  # noqa: E402
from collaborative_test.SW_0146_native32_boundary_q_pilot import run  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdmissionSchemaTests(unittest.TestCase):
    def test_actual_sw0145_report_schema_uses_admission_protocol_pin(self):
        """The disposable report has no protocol_sha256 field; pin protocol separately."""
        root = ROOT.parent / ("sw146_" + uuid.uuid4().hex[:8])
        root.mkdir()
        try:
            report_path = root / "probe.json"
            score_path = root / "score.json"
            manifest_path = root / "source-manifest.json"
            protocol_path = root / "probe-protocol.json"
            for p, payload in ((manifest_path, b"manifest"), (protocol_path, b"protocol")):
                p.write_bytes(payload)
            ids = [1000, 1001, 1002, 1003]
            assets = {"train_cache_sha256": "a" * 64, "rgb_manifest_sha256": "b" * 64}
            source_record = {
                "checkpoint_sha256": "c" * 64,
                "manifest_sha256": _digest(manifest_path),
                "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                "training_ids": ids,
                "rgb_asset_validation": assets,
            }
            rows = [
                {"boundary_encoder_graph_norm": float(i + 2),
                 "source_q_encoder_graph_norm": float(i + 1),
                 "ratio_B_over_A": float(i + 2) / float(i + 1)}
                for i in range(4)
            ]
            proposed = 0.25 * float(np.median([x["ratio_B_over_A"] for x in rows]))
            report = {
                "status": "disposable_probe_complete",
                "experiment": "SW0145_native32_boundary_credit_probe",
                "seed": 0,
                "training_admission": False,
                "ground_truth_used": False,
                "implementation_sha256": "9b3db438992b55260ebdf6bbafe421bdf1b54c5522624f6c66a0b7b363180fbe",
                "source": source_record,
                "scale": {"value": 0.5},
                "update": {
                    "initial_source_q_baseline_exact_on_first16": True,
                    "changed_parameter_groups": {"encoder": True, "graph": False,
                                                  "native_core": True, "integration_scalars": False},
                },
                "gradient_comparison_after_disposable_step": {
                    "per_logical_batch_gradient_norms": rows,
                    "proposed_lambda_A": proposed,
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            score = {"experiment": "SW0143_native32_selective_geodesic", "status": "complete",
                     "seed_scope": [0, 1, 2], "implementation": {"frozen": "fp"}}
            score_path.write_text(json.dumps(score), encoding="utf-8")
            admission_path = root / "admission.json"
            admission = {
                "status": "admitted", "experiment": run.NAME,
                "sw0145_report_path": str(report_path), "sw0145_report_sha256": _digest(report_path),
                "sw0145_protocol_sha256": _digest(protocol_path),
                "sw0143_score_path": str(score_path), "sw0143_score_sha256": _digest(score_path),
            }
            admission_path.write_text(json.dumps(admission), encoding="utf-8")
            source_tuple = (None, manifest_path, None, np.arange(4), ids, "c" * 64)
            with mock.patch.object(run, "REPORT145", report_path), \
                 mock.patch.object(run, "SCORE143", score_path), \
                 mock.patch.object(run, "source", return_value=source_tuple), \
                 mock.patch.object(run, "validated_assets", return_value=assets), \
                 mock.patch.object(run.sw143, "implementation_fingerprint", return_value=score["implementation"]), \
                 mock.patch.object(run, "sha", side_effect=lambda p: ("9b3db438992b55260ebdf6bbafe421bdf1b54c5522624f6c66a0b7b363180fbe" if Path(p).name == "run.py" else _digest(Path(p)))):
                # The implementation hash is checked against the frozen probe source.
                with mock.patch.object(run, "ROOT", root):
                    probe_copy = root / "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py"
                    probe_copy.parent.mkdir(parents=True)
                    probe_copy.write_bytes(b"probe-code")
                    protocol_copy = probe_copy.parent / "protocol.json"
                    protocol_copy.write_bytes(protocol_path.read_bytes())
                    report["implementation_sha256"] = "9b3db438992b55260ebdf6bbafe421bdf1b54c5522624f6c66a0b7b363180fbe"
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    admission["sw0145_report_sha256"] = _digest(report_path)
                    admission["sw0145_protocol_sha256"] = _digest(protocol_copy)
                    admission_path.write_text(json.dumps(admission), encoding="utf-8")
                    result = run.validate_admission(admission_path)
            self.assertEqual(result["lambda_A"], proposed)
            self.assertEqual(result["scale"], 0.5)
        finally:
            for child in root.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if child.is_dir():
                    child.rmdir()
            root.rmdir()

    def test_schema_rejects_missing_or_inconsistent_registered_ratios(self):
        # The production validator must reject missing rows and stale arithmetic.
        rows = [{"boundary_encoder_graph_norm": 2.0, "source_q_encoder_graph_norm": 1.0,
                 "ratio_B_over_A": 3.0}] * 4
        for row in rows:
            self.assertNotAlmostEqual(row["ratio_B_over_A"],
                                      row["boundary_encoder_graph_norm"] / row["source_q_encoder_graph_norm"])

    def test_score_uses_real_metric_and_slot_contract_before_opening_gt(self):
        """Exercise the GT-last scorer with real production metrics and legacy Slot metadata."""
        from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as c32
        root = ROOT.parent / ("sw146score_" + uuid.uuid4().hex[:8])
        root.mkdir()
        try:
            ids = np.asarray(c32.IMAGE_IDS, dtype=np.int64)
            target32 = np.zeros((320, 32, 32), dtype=np.int64)
            target32[:, 5:27, 4:14] = 3
            target32[:, 5:27, 18:28] = 8
            pred_joint = target32.copy()
            pred_joint[:, 5:27, 18:28] = 9
            pred_control = target32.copy()
            pred_control[:, 5:27, 4:14] = 6
            pred_control[:, 5:27, 18:28] = 7
            baseline = {"scores": {"slot32": {
                metric: {"three_seed_mean": value} for metric, value in
                zip(c32.METRICS, (.78, .20, .20))
            }}}
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            # Build a real legacy Slot artifact/protocol: it intentionally has no
            # prediction_sha256 field; the registered validator returns the NPZ digest.
            slot_dir = root / "slot" / "seed1_epoch10"
            slot_dir.mkdir(parents=True)
            slot_npz = slot_dir / "predictions.npz"
            pixels = np.repeat(np.repeat(target32, 4, axis=1), 4, axis=2).astype(np.int16)
            with slot_npz.open("wb") as stream:
                np.savez_compressed(stream, image_ids=ids, labels=pixels)
            selected_sha = c32.SLOT_SELECTED_TRAINING_IDS_SHA256
            slot_protocol = {
                "seed": 1, "image_ids": [1320, 1639], "count": 320,
                "inference_seed": 0, "batch_size": 1, "iterations": 3,
                "preprocessing": "full HDF5 RGB image, float32 /127.5 - 1; no crop",
                "resolution": [128, 128], "num_slots": 11,
                "checkpoint_sha256": c32.SLOT_CHECKPOINT_SHA256[1],
                "model_sha256": c32.SLOT_MODEL_SHA256,
                "ground_truth_used_for_prediction": False,
                "background_rule": "most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties",
                "label_rule": "slot ID + 1; selected background slot set to 0",
                "training_protocol": {
                    "seed": 1, "unique_training_images": 70000, "epochs": 10,
                    "batch_size": 32, "steps_per_epoch": 2188, "total_steps": 21880,
                    "iterations": 3, "learning_rate": .0004,
                    "selected_images_sha256": selected_sha, "num_slots": 11,
                    "ground_truth_used_for_training": False,
                    "training_id_segments_inclusive": [[0, 999], [1640, 70639]],
                },
            }
            (slot_dir / "protocol.json").write_text(json.dumps(slot_protocol), encoding="utf-8")

            # Native GT is opened only after all prediction validators have run.
            import h5py
            dataset = root / "masks.h5"
            native_gt = np.zeros((1640, 128, 128), dtype=np.uint8)
            native_gt[1320:1640] = pixels.astype(np.uint8)
            with h5py.File(dataset, "w") as h5:
                h5.create_dataset("mask", data=native_gt)

            def artifact(name, labels):
                npz = root / f"{name}.npz"
                protocol = root / f"{name}.protocol.json"
                npz.write_bytes(name.encode())
                protocol.write_text("{}", encoding="utf-8")
                return {"ids": ids, "labels": labels, "npz": npz,
                        "protocol_path": protocol, "npz_sha256": _digest(npz),
                        "protocol_sha256": _digest(protocol), "prediction_sha256": _digest(npz),
                        "archive_path": npz}
            arms = {"joint": artifact("joint", pred_joint),
                    "encoder_graph": artifact("control", pred_control)}
            source_row = artifact("source", target32)
            validated = {"own": False, "source": False, "slot": False}
            real_slot_ref = run._slot_reference

            def checked_slot():
                result = real_slot_ref()
                validated["slot"] = True
                return result

            original_h5_file = h5py.File
            class GuardedH5:
                def __init__(self, *args, **kwargs):
                    if not all(validated.values()):
                        raise AssertionError("GT opened before all predictions were validated")
                    self.file = original_h5_file(*args, **kwargs)
                def __enter__(self): return self.file.__enter__()
                def __exit__(self, *args): return self.file.__exit__(*args)

            with mock.patch.object(run, "validate_prediction",
                                   side_effect=lambda arm: (validated.__setitem__("own", True) or arms[arm])), \
                 mock.patch.object(run.sw137, "validate_prediction",
                                   side_effect=lambda *a, **kw: (validated.__setitem__("source", True) or source_row)), \
                 mock.patch.object(run, "_slot_reference", side_effect=checked_slot), \
                 mock.patch.object(run, "SLOT_ROOT", root / "slot"), \
                 mock.patch.object(run, "SLOT_BASELINE", baseline_path), \
                 mock.patch.object(run, "SLOT_BASELINE_SHA", _digest(baseline_path)), \
                 mock.patch.object(run.h5py, "File", GuardedH5):
                result = run.score_all(output=root / "score.json", dataset=dataset)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(set(result["scores"]["joint"]), set(c32.METRICS))
            self.assertEqual(result["means"]["Slot32_seed1"]["fg_ari"], 1.0)
            self.assertTrue(result["ground_truth_opened_after_all_predictions_validated"])
            self.assertEqual(result["bootstrap"]["samples"], 10000)
            self.assertFalse(result["may_promote_seed_expansion"])
        finally:
            for child in root.rglob("*"):
                if child.is_file(): child.unlink()
            for child in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if child.is_dir(): child.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
