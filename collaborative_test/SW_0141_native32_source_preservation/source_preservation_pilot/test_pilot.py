"""CPU-only contract tests for the private, unregistered source-preservation draft."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
from pathlib import Path
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot
import source_preservation_loss as anchor


def _sha(ch):
    return ch * 64


class FakeFoundation:
    def __init__(self):
        self.seed = 1
        self.seed = 1
        self.image_ids = [101, 102, 103, 104]
        self.pool_indices = np.asarray([201, 202, 203, 204], dtype=np.int64)
        self.provenance = {
            "source_core_sha256": _sha("a"), "source_manifest_sha256": _sha("b"),
            "encoder_sha256": _sha("c"), "feature_preprocessing_sha256": _sha("d"),
            "rgb_asset_validation": {"train_manifest_sha256": _sha("e")},
            "implementation_fingerprint": {"native/dynamics.py": _sha("f")},
        }


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.root = HERE / ("fixture_" + uuid.uuid4().hex[:10])
        self.root.mkdir()

    def tearDown(self):
        cache = self.root / "cache"
        if cache.exists():
            for name in ("manifest.json", "teacher_q.f32"):
                path = cache / name
                if path.exists(): path.unlink()
            cache.rmdir()
        incomplete = self.root / "incomplete"
        if incomplete.exists(): incomplete.rmdir()
        prediction = self.root / "predictions" / "c_only"
        if prediction.exists():
            for name in ("protocol.json", "predictions.npz"):
                path = prediction / name
                if path.exists(): path.unlink()
            prediction.rmdir()
            prediction.parent.rmdir()
        for name in ("pilot.json", "lambda.json"):
            path = self.root / name
            if path.exists(): path.unlink()
        # Remove only known files/dirs created by the successful score fixture.
        for rel in (
            "summary.json", "source.npz", "source_protocol.json", "old.npz", "old_protocol.json",
            "c_only_seed1/checkpoint.pt", "c_only_seed1/optimizer.pt",
            "c_only_seed1/history.json", "c_only_seed1/manifest.json",
            "c_only_seed1/TRAINING_COMPLETED.json",
            "scored_predictions/c_only/predictions.npz", "scored_predictions/c_only/protocol.json",
            "scored_predictions/c_plus_anchor/predictions.npz", "scored_predictions/c_plus_anchor/protocol.json",
            "slot/seed1_epoch10/predictions.npz", "slot/seed1_epoch10/protocol.json",
        ):
            path = self.root / rel
            if path.exists(): path.unlink()
        for rel in ("scored_predictions/c_only", "scored_predictions/c_plus_anchor",
                    "scored_predictions", "slot/seed1_epoch10", "slot", "training"):
            path = self.root / rel
            if path.exists(): path.rmdir()
        train_dir = self.root / "c_only_seed1"
        if train_dir.exists(): train_dir.rmdir()
        self.root.rmdir()

    def _make_cache(self):
        directory = self.root / "cache"
        directory.mkdir()
        # Exact symmetric float32 Q values, shaped like one production B4.
        q = np.zeros((4, 1024, 1024), dtype=np.float32)
        for row in q:
            np.fill_diagonal(row, 1.0)
        payload_path = directory / "teacher_q.f32"
        with payload_path.open("xb") as stream:
            stream.write(q.tobytes(order="C"))
        chunk_sha = hashlib.sha256(q.tobytes(order="C")).hexdigest()
        fp = pilot.implementation_fingerprint()
        source_hashes = {
            "source_core_sha256": _sha("a"), "source_manifest_sha256": _sha("b"),
            "encoder_sha256": _sha("c"), "feature_statistics_sha256": _sha("d"),
            "rgb_cache_manifest_sha256": _sha("e"),
            "implementation_fingerprint_sha256": pilot._canonical_sha(fp),
        }
        manifest = pilot._build_cache_manifest(
            seed=1, ids=[101, 102, 103, 104], rows=[201, 202, 203, 204],
            source_hashes=source_hashes, cache_sha=pilot.sha256_file(payload_path),
            chunk_hashes=[chunk_sha])
        manifest["rgb_asset_validation"] = FakeFoundation().provenance["rgb_asset_validation"]
        manifest["foundation_implementation_fingerprint"] = FakeFoundation().provenance["implementation_fingerprint"]
        manifest["cache_file"] = payload_path.name
        pilot._write_once(directory / "manifest.json", manifest)
        return directory, q

    def test_cache_manifest_is_helper_validated_and_b4_read_hash_checked(self):
        directory, q = self._make_cache()
        cache = pilot.TeacherCache(directory, FakeFoundation(), expected_count=4)
        loaded = cache.read_b4(0, "cpu")
        self.assertEqual(tuple(loaded.shape), (4, 1024, 1024))
        self.assertTrue(torch.equal(loaded, torch.from_numpy(q.copy())))
        cache.verify_complete_payload()
        cache.close()

    def test_corrupt_teacher_chunk_fails_before_returning_tensor(self):
        directory, _ = self._make_cache()
        path = directory / "teacher_q.f32"
        cache = pilot.TeacherCache(directory, FakeFoundation(), expected_count=4)
        cache.close()
        with path.open("r+b") as stream:
            stream.seek(64)
            byte = stream.read(1)
            stream.seek(64)
            stream.write(bytes([byte[0] ^ 0x01]))
        cache = pilot.TeacherCache(directory, FakeFoundation(), expected_count=4)
        with self.assertRaisesRegex(ValueError, "chunk 0.*SHA"):
            cache.read_b4(0, "cpu")
        cache.close()

    def test_incomplete_cache_is_not_resumed_or_treated_as_complete(self):
        directory = self.root / "incomplete"
        directory.mkdir()
        with self.assertRaises(FileNotFoundError):
            pilot.TeacherCache(directory, FakeFoundation(), expected_count=4)
        self.assertEqual(list(directory.iterdir()), [])
        with self.assertRaises(FileExistsError):
            pilot.write_teacher_cache(1, directory, device="cpu", count=4096)

    def test_cache_ordering_and_source_identity_reject_misbound_foundation(self):
        directory, _ = self._make_cache()
        foundation = FakeFoundation()
        foundation.image_ids = [102, 101, 103, 104]
        with self.assertRaisesRegex(ValueError, "IDs or pool-row ordering mismatch"):
            pilot.TeacherCache(directory, foundation, expected_count=4)
        foundation = FakeFoundation()
        foundation.provenance["source_core_sha256"] = _sha("9")
        with self.assertRaisesRegex(ValueError, "source hash mismatch for source_core"):
            pilot.TeacherCache(directory, foundation, expected_count=4)
        foundation = FakeFoundation()
        foundation.seed = 0
        with self.assertRaisesRegex(ValueError, "source seed"):
            pilot.TeacherCache(directory, foundation, expected_count=4)
        foundation = FakeFoundation()
        foundation.seed = 0
        with self.assertRaisesRegex(ValueError, "source seed"):
            pilot.TeacherCache(directory, foundation, expected_count=4)

    def test_anchor_is_offdiagonal_teacher_detached_student_gradient_live(self):
        teacher = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        student = teacher.clone().requires_grad_(True)
        with torch.no_grad():
            student[0, 3, 9] = student[0, 9, 3] = 0.5
        before = teacher.clone()
        loss = anchor.source_q_anchor(student, teacher)
        expected = 2.0 * .25 / (1024 * 1023)
        self.assertAlmostEqual(float(loss.detach()), expected, places=12)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(teacher, before))

    def test_module_import_and_cpu_contracts_do_not_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(pilot.ROOT.name, "patch_v2_sw")

    def test_registered_c_only_and_c_plus_anchor_objectives_have_exact_terms(self):
        c = torch.tensor(2.0, requires_grad=True)
        a = torch.tensor(3.0, requires_grad=True)
        c_only = pilot.compose_objective(c, lambda_c=4.0)
        self.assertEqual(float(c_only.detach()), 8.0)
        c_only.backward(retain_graph=True)
        self.assertEqual(float(c.grad), 4.0)
        self.assertIsNone(a.grad)
        combined = pilot.compose_objective(c, lambda_c=4.0,
                                           anchor_loss=a, lambda_a=5.0)
        self.assertEqual(float(combined.detach()), 23.0)
        combined.backward()
        self.assertEqual(float(a.grad), 5.0)
        with self.assertRaisesRegex(ValueError, "without the source-Q anchor"):
            pilot.compose_objective(c, lambda_c=4.0, lambda_a=1.0)

    def test_source_cache_parity_uses_registered_tolerance_and_finite_guard(self):
        source = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        close = source.clone(); close[0, 0, 1] = 1e-7
        parity = pilot._q_cache_parity(source, close)
        self.assertTrue(parity["allclose_atol_2e-6_rtol_1e-6"])
        self.assertFalse(parity["exact"])
        far = source.clone(); far[0, 0, 1] = .1
        self.assertFalse(pilot._q_cache_parity(source, far)["allclose_atol_2e-6_rtol_1e-6"])
        invalid = source.clone(); invalid[0, 0, 1] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            pilot._q_cache_parity(source, invalid)

    def test_unregistered_output_and_lambda_records_are_create_only(self):
        target = self.root / "pilot.json"
        pilot._write_once(target, {"status": "draft"})
        with self.assertRaises(FileExistsError):
            pilot._write_once(target, {"status": "replace"})
        lambda_path = self.root / "lambda.json"
        record = {"status": "incomplete", "lambda_A": 2.0,
                  "implementation_fingerprint": pilot.implementation_fingerprint()}
        pilot._write_once(lambda_path, record)
        with self.assertRaisesRegex(ValueError, "self hash"):
            pilot._load_lambda_a(lambda_path)

    def test_nonfinite_or_wrong_order_q_is_rejected(self):
        teacher = torch.zeros((4, 1024, 1024), dtype=torch.float32)
        student = teacher.clone().requires_grad_(True)
        teacher[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            anchor.source_q_anchor(student, teacher)
        with self.assertRaisesRegex(ValueError, "complete contiguous B4"):
            anchor.validate_teacher_cache_manifest({}, expected_image_ids=[1, 2, 3],
                expected_pool_rows=[4, 5, 6], expected_source_hashes={},
                expected_cache_sha256=_sha("a"), expected_chunk_sha256=[])

    def test_paired_bootstrap_is_deterministic_and_common_image_paired(self):
        values = np.linspace(-.2, .3, 320)
        first = pilot.paired_bootstrap_ci(values, samples=200)
        second = pilot.paired_bootstrap_ci(values, samples=200)
        self.assertEqual(first, second)
        self.assertLess(first[0], float(values.mean()))
        self.assertGreater(first[1], float(values.mean()))

    def test_score_rejects_prediction_before_any_ground_truth_open(self):
        root = self.root / "predictions"
        folder = root / "c_only"
        folder.mkdir(parents=True)
        labels = np.zeros((320, 32, 32), dtype=np.int64)
        ids = np.arange(1320, 1640, dtype=np.int64)
        npz = folder / "predictions.npz"
        with npz.open("xb") as stream:
            np.savez_compressed(stream, image_ids=ids, qcc_labels=labels)
        pilot._write_once(folder / "protocol.json", {
            "status": "predictions_complete_local_unregistered", "seed": 1,
            "arm": "c_only", "ground_truth_used_for_prediction": False,
            "prediction_sha256": "0" * 64,
            "implementation_fingerprint": pilot.implementation_fingerprint()})
        import h5py
        with mock.patch.object(h5py, "File", side_effect=AssertionError("GT was opened")) as opened:
            with self.assertRaises((ValueError, FileNotFoundError)):
                pilot.score_pilot(prediction_root=root, output=self.root / "summary.json")
            opened.assert_not_called()

    def _write_own_prediction_fixture(self, root, arm, manifest):
        folder = root / arm
        folder.mkdir(parents=True)
        labels = np.zeros((320, 32, 32), dtype=np.int64)
        ids = np.arange(1320, 1640, dtype=np.int64)
        npz = folder / "predictions.npz"
        with npz.open("xb") as stream:
            np.savez_compressed(stream, image_ids=ids, qcc_labels=labels)
        verified = {"manifest": manifest, "manifest_sha256": "d" * 64,
                    "checkpoint_sha256": "c" * 64}
        proto = pilot._build_own_prediction_protocol(arm, 1, verified, labels)
        proto["prediction_sha256"] = pilot.sha256_file(npz)
        pilot._write_once(folder / "protocol.json", proto)
        return proto

    def test_score_success_validates_every_arm_and_freezes_before_gt(self):
        import h5py
        prediction_root = self.root / "scored_predictions"
        prediction_root.mkdir()
        training_root = self.root / "training"
        training_root.mkdir()
        manifests = {}
        for arm in pilot.ARMS:
            manifests[arm] = {"source_core_sha256": "a" * 64,
                              "source_manifest_sha256": "b" * 64,
                              "source_training_ids_sha256": "e" * 64}
            self._write_own_prediction_fixture(prediction_root, arm, manifests[arm])
        trained = {arm: {"manifest": manifests[arm], "manifest_sha256": "d" * 64,
                         "checkpoint_sha256": "c" * 64} for arm in pilot.ARMS}
        events = []
        ids = np.arange(1320, 1640, dtype=np.int64)
        labels = np.zeros((320, 32, 32), dtype=np.int64)
        source_npz, source_proto = self.root / "source.npz", self.root / "source_protocol.json"
        old_npz, old_proto = self.root / "old.npz", self.root / "old_protocol.json"
        for path in (source_npz, old_npz):
            with path.open("xb") as stream: np.savez_compressed(stream, image_ids=ids, qcc_labels=labels)
        source_proto.write_text("{}", encoding="utf-8")
        old_proto.write_text("{}", encoding="utf-8")
        slot_dir = self.root / "slot" / "seed1_epoch10"
        slot_dir.mkdir(parents=True)
        slot_npz, slot_proto = slot_dir / "predictions.npz", slot_dir / "protocol.json"
        with slot_npz.open("xb") as stream: np.savez_compressed(stream, image_ids=ids, labels=np.zeros((320,128,128), dtype=np.int64))
        slot_proto.write_text(json.dumps({"prediction_sha256": pilot.sha256_file(slot_npz)}), encoding="utf-8")
        metric = {name: {"mean": .2, "valid_count": 320,
                         "per_image": np.full(320, .2).tolist()}
                  for name in ("fg_ari", "foreground_iou", "matched_object_iou")}

        def fake_source(*args, **kwargs):
            events.append("source")
            return {"labels": labels, "ids": ids, "archive_path": source_npz,
                    "prediction_sha256": pilot.sha256_file(source_npz),
                    "protocol_path": source_proto,
                    "protocol_sha256": pilot.sha256_file(source_proto)}

        def fake_old(*args, **kwargs):
            events.append("old")
            return {"labels": labels, "ids": ids, "prediction_path": old_npz,
                    "prediction_sha256": pilot.sha256_file(old_npz),
                    "protocol_path": old_proto,
                    "protocol_sha256": pilot.sha256_file(old_proto)}

        class FakeH5:
            def __enter__(self):
                if len(events) != 5:
                    raise AssertionError("GT opened before both own arms and references were validated")
                events.append("GT")
                class Dataset:
                    shape = (1640, 128, 128, 1)
                    def __getitem__(self, key): return np.zeros((320,128,128,1), dtype=np.uint8)
                return {"mask": Dataset()}
            def __exit__(self, *args): return False

        def fake_metrics(pred, target):
            return metric

        def fake_training(folder, arm, seed):
            events.append("own:" + arm)
            return trained[arm]

        with mock.patch.object(pilot, "validate_pilot_checkpoint", side_effect=fake_training), \
             mock.patch.object(pilot.sw137, "validate_prediction", side_effect=fake_source), \
             mock.patch.object(pilot.sw139_eval, "validate_arm_prediction", side_effect=fake_old), \
             mock.patch.object(pilot.contract32, "validate_slot_protocol", side_effect=lambda *a, **k: events.append("slot")), \
             mock.patch.object(pilot.contract32, "modal_native32", side_effect=lambda x: np.zeros((320,32,32), dtype=np.int64)), \
             mock.patch.object(pilot.contract32, "production_metrics32", side_effect=fake_metrics), \
             mock.patch.object(h5py, "File", side_effect=lambda *a, **k: FakeH5()):
            summary = pilot.score_pilot(prediction_root=prediction_root,
                training_root=training_root, source_root=self.root, old_c_root=self.root,
                slot_root=self.root / "slot", dataset=self.root / "unused.h5",
                output=self.root / "summary.json", bootstrap_samples=5)
        self.assertEqual(events[-1], "GT")
        self.assertEqual(summary["status"], "complete_local_unregistered")

    def test_own_prediction_metadata_is_bound_to_completed_training_before_gt(self):
        root = self.root / "predictions"; root.mkdir()
        arm = "c_only"
        manifest = {"source_core_sha256": "a"*64, "source_manifest_sha256": "b"*64,
                    "source_training_ids_sha256": "e"*64}
        proto = self._write_own_prediction_fixture(root, arm, manifest)
        trained = {"manifest": manifest, "manifest_sha256": "d"*64,
                   "checkpoint_sha256": "c"*64}
        protocol_path = root / arm / "protocol.json"
        for key, bad in (("settle", 64), ("steps", 512), ("readout", {"threshold": .7}),
                         ("checkpoint_sha256", "f"*64),
                         ("training_manifest_sha256", "0"*64),
                         ("source_manifest_sha256", "9"*64),
                         ("prediction_shape", [320, 16, 16])):
            altered = dict(proto); altered[key] = bad
            protocol_path.write_text(json.dumps(altered), encoding="utf-8")
            with mock.patch.object(pilot, "validate_pilot_checkpoint", return_value=trained):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    pilot.validate_own_prediction(arm, prediction_root=root, training_root=self.root)

    def test_real_checkpoint_validator_producer_protocol_roundtrip(self):
        """Exercise actual checkpoint validator return fields through protocol producer/consumer."""
        ids = list(range(1000, 1000 + pilot.TRAIN_N))
        rows = list(range(2000, 2000 + pilot.TRAIN_N))
        fp = pilot.implementation_fingerprint()
        asset_hashes = {"train_manifest_sha256": _sha("a"), "train_cache_sha256": _sha("b")}
        source_sha = pilot.sw130.source97.EXPECTED_SOURCE_SHAS[1]
        foundation = SimpleNamespace(
            image_ids=ids, pool_indices=np.asarray(rows, dtype=np.int64),
            provenance={"source_core_sha256": source_sha,
                        "source_manifest_sha256": _sha("c"),
                        "source_training_ids_sha256": anchor.image_ids_sha256(ids),
                        "rgb_asset_validation": asset_hashes,
                        "implementation_fingerprint": fp})
        train_dir = self.root / "c_only_seed1"
        train_dir.mkdir()
        checkpoint = train_dir / "checkpoint.pt"
        optimizer = train_dir / "optimizer.pt"
        history_path = train_dir / "history.json"
        torch.save({"seed": 1, "arm": "c_only", "implementation_fingerprint": fp,
                    "training_ids_sha256": anchor.image_ids_sha256(ids),
                    "training_pool_rows_sha256": anchor.image_ids_sha256(rows),
                    "asset_hashes": asset_hashes,
                    "foundation_implementation_fingerprint": fp,
                    "wrapped_state_dict": {"p": torch.tensor([1.0])},
                    "encoder_state_dict": {"q": torch.tensor([2.0])}}, checkpoint)
        torch.save({"state": {"0": {"step": torch.tensor(float(pilot.sw139.UPDATES)),
                                     "exp_avg": torch.tensor([0.0]),
                                     "exp_avg_sq": torch.tensor([0.0])}}}, optimizer)
        history = []
        for update in range(1, pilot.sw139.UPDATES + 1):
            batch_ids = ids[(update - 1) * pilot.LOGICAL_BATCH:update * pilot.LOGICAL_BATCH]
            history.append({"update": update, "image_ids": batch_ids,
                            "image_ids_sha256": anchor.image_ids_sha256(batch_ids),
                            "c_micro_mean": [0.1] * pilot.MICROS,
                            "gradient_norm_before_clip": 1.0,
                            "anchor_micro_mean": [],
                            "first_update_teacher_cache_b4_parity": []})
        history_path.write_text(json.dumps({"updates": pilot.sw139.UPDATES,
                                            "records": history}), encoding="utf-8")
        manifest_path = train_dir / "manifest.json"
        manifest = {"experiment": pilot.PILOT_NAME, "status": "training_complete_local_unregistered",
                    "seed": 1, "arm": "c_only", "updates": pilot.sw139.UPDATES,
                    "passes": 1, "unique_training_images": pilot.TRAIN_N,
                    "total_image_exposures": pilot.TRAIN_N * 2, "views_per_image": 2,
                    "batch_size": pilot.LOGICAL_BATCH, "microbatch_size": pilot.MICROBATCH,
                    "steps": pilot.STEPS, "settle": pilot.SETTLE, "live_tail": pilot.TAIL,
                    "ordered_training_ids": ids,
                    "ordered_training_ids_sha256": anchor.image_ids_sha256(ids),
                    "ordered_pool_rows": rows,
                    "ordered_pool_rows_sha256": anchor.image_ids_sha256(rows),
                    "source_core_sha256": source_sha,
                    "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
                    "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
                    "asset_hashes": asset_hashes,
                    "foundation_implementation_fingerprint": fp,
                    "implementation_fingerprint": fp,
                    "checkpoint_sha256": pilot.sha256_file(checkpoint),
                    "optimizer_sha256": pilot.sha256_file(optimizer),
                    "history_sha256": pilot.sha256_file(history_path),
                    "ground_truth_used_for_training": False}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        pilot._write_once(train_dir / "TRAINING_COMPLETED.json", {
            "status": "training_complete_local_unregistered",
            "manifest_sha256": pilot.sha256_file(manifest_path),
            "checkpoint_sha256": pilot.sha256_file(checkpoint),
            "updates": pilot.sw139.UPDATES, "ground_truth_used_for_training": False})
        with mock.patch.object(pilot, "source_bundle", return_value=foundation):
            verified = pilot.validate_pilot_checkpoint(train_dir, arm="c_only", seed=1)
            self.assertEqual(verified["manifest_sha256"], pilot.sha256_file(manifest_path))
            pred_dir = self.root / "predictions" / "c_only"
            pred_dir.mkdir(parents=True)
            labels = np.zeros((320, 32, 32), dtype=np.int64)
            with (pred_dir / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, image_ids=np.arange(1320, 1640), qcc_labels=labels)
            proto = pilot._build_own_prediction_protocol("c_only", 1, verified, labels)
            proto["prediction_sha256"] = pilot.sha256_file(pred_dir / "predictions.npz")
            pilot._write_once(pred_dir / "protocol.json", proto)
            consumed = pilot.validate_own_prediction(
                "c_only", prediction_root=pred_dir.parent, training_root=self.root)
        consumed_protocol = json.loads(consumed["protocol"].read_text(encoding="utf-8"))
        self.assertEqual(consumed_protocol["training_manifest_sha256"], pilot.sha256_file(manifest_path))
        self.assertEqual(consumed_protocol["source_manifest_sha256"],
                         foundation.provenance["source_manifest_sha256"])

    def test_teacher_rejects_rgb_payload_or_full_foundation_fingerprint_change(self):
        directory, _ = self._make_cache()
        foundation = FakeFoundation()
        foundation.provenance["rgb_asset_validation"] = {"train_manifest_sha256": _sha("8")}
        with self.assertRaisesRegex(ValueError, "RGB payload/assets"):
            pilot.TeacherCache(directory, foundation, expected_count=4)
        foundation = FakeFoundation()
        foundation.provenance["implementation_fingerprint"] = {"native/dynamics.py": _sha("9")}
        with self.assertRaisesRegex(ValueError, "foundation dependencies"):
            pilot.TeacherCache(directory, foundation, expected_count=4)


if __name__ == "__main__":
    unittest.main()
