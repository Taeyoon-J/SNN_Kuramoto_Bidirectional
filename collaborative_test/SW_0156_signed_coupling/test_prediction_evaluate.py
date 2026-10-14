import os
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from collaborative_test.SW_0156_signed_coupling import evaluate, predict, prediction_queue
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143


class FakeMaskDataset:
    shape = (1640, 128, 128)

    def __init__(self):
        patch = np.zeros((32, 32), dtype=np.uint8)
        patch[4:28, 3:14] = 1
        patch[4:28, 18:29] = 2
        self.pixels = np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)

    def __getitem__(self, item):
        self.assert_fixed_slice(item)
        return np.repeat(self.pixels[None], 320, axis=0)

    @staticmethod
    def assert_fixed_slice(item):
        if not isinstance(item, slice) or (item.start, item.stop) != (1320, 1640):
            raise AssertionError("unexpected GT mask access before fixed320 contract")


class FakeH5:
    def __init__(self, *args, **kwargs):
        self.dataset = FakeMaskDataset()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __contains__(self, key):
        return key == "mask"

    def __getitem__(self, key):
        if key != "mask":
            raise KeyError(key)
        return self.dataset


class Fixed320PipelineTests(unittest.TestCase):
    def test_partial_prediction_ledger_is_a_resume_target(self):
        target = Path("candidate")
        exists = {"candidate": False, "protocol.json": False}
        with mock.patch.object(Path, "exists", autospec=True,
                               side_effect=lambda p: exists.get(p.name, False)), \
             mock.patch.object(Path, "is_file", autospec=True, side_effect=lambda _p: True):
            self.assertEqual(prediction_queue._target_state(target), "missing")
            exists["candidate"] = True
            self.assertEqual(prediction_queue._target_state(target), "ledger")
            exists["protocol.json"] = True
            with self.assertRaises(FileExistsError):
                prediction_queue._target_state(target)

    def test_owned_child_is_stopped_before_prediction_lease_unlock(self):
        events = []

        class Proc:
            def poll(self):
                return None if not events or "stop" not in events else 0

        class Lease:
            def fileno(self):
                return 42
            def close(self):
                events.append("close")

        with mock.patch.object(prediction_queue.owner, "_stop_owned",
                               side_effect=lambda _proc: events.append("stop")), \
             mock.patch.object(prediction_queue, "fcntl", SimpleNamespace(
                 LOCK_UN=8, flock=lambda _fd, _mode: events.append("unlock"))):
            prediction_queue._cleanup_child_before_lease_release(Proc(), Lease())
        self.assertEqual(events, ["stop", "unlock", "close"])

    def test_partial_ledger_reaches_child_and_exception_stops_before_unlock(self):
        events = []

        class FakePath:
            def __init__(self, name): self.name = str(name)
            def exists(self): return self.name.endswith("attempt01.log") is False
            def open(self, *_args, **_kwargs): return self
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def mkdir(self, **_kwargs): pass
            def glob(self, *_args): return []
            def __truediv__(self, child): return FakePath(self.name + "/" + child)
            def __str__(self): return self.name

        class Lease:
            def fileno(self): return 43
            def close(self): events.append("close")

        class Proc:
            pid = 9876
            returncode = None
            def poll(self): return self.returncode
            def wait(self): return self.returncode

        proc = Proc()
        q = object.__new__(prediction_queue.PredictionQueue)
        q.stop_event = __import__("threading").Event()
        q.state_lock = __import__("threading").RLock()
        q.fingerprint = {}
        q._verify = lambda: None
        q._update = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected state write failure"))
        fake_target = FakePath("target")
        fake_logs = FakePath("logs")

        def acquire(_stop): return 2, "GPU-test", 0, Lease()

        def stop_owned(actual_proc):
            self.assertIs(actual_proc, proc)
            events.append("stop")
            proc.returncode = -15

        with mock.patch.object(prediction_queue.predict, "prediction_dir", return_value=fake_target), \
             mock.patch.object(prediction_queue, "LOGS", fake_logs), \
             mock.patch.object(prediction_queue, "_valid_completed", side_effect=ValueError("incomplete")), \
             mock.patch.object(prediction_queue, "_target_state", return_value="ledger"), \
             mock.patch.object(prediction_queue.owner, "_acquire", side_effect=acquire), \
             mock.patch.object(prediction_queue.owner, "_stable_idle", return_value=True), \
             mock.patch.object(prediction_queue.owner, "_stop_owned", side_effect=stop_owned), \
             mock.patch.object(prediction_queue, "_upstream", return_value={"screen_sha256": "x"}), \
             mock.patch.object(prediction_queue.subprocess, "Popen", return_value=proc) as popen, \
             mock.patch.object(prediction_queue, "fcntl", SimpleNamespace(
                 LOCK_UN=8, flock=lambda _fd, _mode: events.append("unlock"))):
            with self.assertRaisesRegex(RuntimeError, "injected state write failure"):
                q._run_gpu_arm("signed_candidate", {"screen_sha256": "x"})
        popen.assert_called_once()
        self.assertEqual(events, ["stop", "unlock", "close"])

    def test_paired_bootstrap_uses_common_fixed320_rows_and_is_deterministic(self):
        source = np.linspace(0, 1, 320)
        candidate = source + .125
        first = evaluate.paired_image_bootstrap(candidate, source)
        second = evaluate.paired_image_bootstrap(candidate, source)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], .125)
        self.assertEqual(first["ci95"], [.125, .125])
        self.assertTrue(first["common_image_resamples"])
        with self.assertRaises(ValueError):
            evaluate.paired_image_bootstrap(candidate[:-1], source[:-1])

    def test_prediction_arms_and_task_dependencies_are_fixed(self):
        self.assertEqual(predict.ARMS,
                         ("source_T", "signed_candidate", "matched_control"))
        tasks = prediction_queue.task_plan()
        self.assertEqual([task["task_id"] for task in tasks],
                         ["source_T", "signed_candidate", "matched_control", "score_seed1"])
        self.assertEqual(tasks[-1]["depends_on"],
                         ["source_T", "signed_candidate", "matched_control"])
        self.assertFalse(tasks[0]["gpu"])
        self.assertTrue(tasks[1]["gpu"] and tasks[2]["gpu"])
        with self.assertRaises(ValueError):
            predict.prediction_dir("unregistered")

    def test_fingerprints_resolve_current_local_dependencies(self):
        self.assertEqual(predict.fingerprint()[
            "collaborative_test/SW_0156_signed_coupling/predict.py"],
            predict.sha256_file(predict.HERE / "predict.py"))
        evaluation_fp = evaluate.implementation_fingerprint()
        self.assertEqual(evaluation_fp[
            "collaborative_test/SW_0156_signed_coupling/evaluate.py"],
            evaluate.sha256_file(evaluate.HERE / "evaluate.py"))

    def test_cpu_parent_hides_cuda_before_loading_torch_modules(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    def test_prediction_child_import_preserves_assigned_cuda_without_initializing(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "3"
        env["PYTHONPATH"] = str(predict.ROOT)
        code = (
            "import os, numpy, torch; "
            "import collaborative_test.SW_0156_signed_coupling.predict; "
            "print(os.environ.get('CUDA_VISIBLE_DEVICES')); "
            "print(torch.cuda.is_initialized())"
        )
        completed = subprocess.run([sys.executable, "-c", code], cwd=predict.ROOT,
            env=env, text=True, capture_output=True, timeout=60, check=True)
        self.assertEqual(completed.stdout.splitlines(), ["3", "False"])

    def test_score_all_freezes_all_predictions_before_opening_gt(self):
        ids = np.arange(1320, 1640, dtype=np.int64)
        labels32 = np.zeros((320, 32, 32), dtype=np.int64)
        labels32[:, 4:28, 3:14] = 1
        labels32[:, 4:28, 18:29] = 2
        fake_paths = {arm: {
            "ids": ids, "labels": labels32, "npz_path": Path(f"{arm}.npz"),
            "protocol_path": Path(f"{arm}.json"), "prediction_sha256": "a" * 64,
            "protocol_sha256": "b" * 64,
        } for arm in predict.ARMS}
        slot = {"ids": ids, "labels": labels32, "native_path": Path("slot.npz"),
                "protocol_path": Path("slot.json"), "npz_sha": "c" * 64,
                "protocol_sha": "d" * 64,
                "protocol": {"checkpoint_sha256": "e" * 64}}
        calls = []

        def validate(_directory, *, arm, **_kwargs):
            calls.append(("prediction_validated", arm))
            return fake_paths[arm]

        def slot_rows():
            calls.append(("slot_validated",))
            return slot

        class OpeningFakeH5(FakeH5):
            def __enter__(self):
                if len(calls) != 4:
                    raise AssertionError("GT opened before all three arms and Slot were validated")
                calls.append(("gt_opened",))
                return self

        baseline_sha = sw143.SLOT_BASELINE_SHA256
        real_sha = evaluate.sha256_file
        row_hashes = {
            "source_T.npz": "a" * 64, "source_T.json": "b" * 64,
            "signed_candidate.npz": "a" * 64, "signed_candidate.json": "b" * 64,
            "matched_control.npz": "a" * 64, "matched_control.json": "b" * 64,
            "slot.npz": "c" * 64, "slot.json": "d" * 64,
        }

        def fake_sha(path):
            if Path(path) == evaluate.SLOT_BASELINE:
                return baseline_sha
            return row_hashes[Path(path).name] if Path(path).name in row_hashes else real_sha(path)

        output = Path("synthetic-score.json")
        class CapturedOutput(io.StringIO):
            saved = ""

            def close(self):
                self.saved = self.getvalue()
                super().close()

        captured = CapturedOutput()
        real_open = Path.open

        def capture_score(self, *args, **kwargs):
            return captured if self == output else real_open(self, *args, **kwargs)

        with mock.patch.object(evaluate.predict, "validate_prediction", side_effect=validate), \
             mock.patch.object(evaluate, "_read_slot_seed1", side_effect=slot_rows), \
             mock.patch.object(evaluate, "sha256_file", side_effect=fake_sha), \
             mock.patch.object(evaluate, "_dataset_identity", return_value={
                 "resolved_path": "/registered/clevr.h5", "size_bytes": 100, "mtime_ns": 200}), \
             mock.patch.object(evaluate.h5py, "File", OpeningFakeH5), \
             mock.patch.object(Path, "open", new=capture_score):
            report = evaluate.score_all(output=output)
        self.assertEqual(calls[-1], ("gt_opened",))
        self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
        self.assertIs(report["ground_truth_used_for_prediction"], False)
        self.assertIs(report["ground_truth_used_for_scoring"], True)
        self.assertEqual(set(report["registered_gate"]), {
            "candidate_fg_ari_at_least_0.01_above_source_T",
            "candidate_fg_ari_at_least_0.01_above_control",
            "candidate_minus_source_T_fg_ci_lower_positive",
            "candidate_minus_control_fg_ci_lower_positive"})
        self.assertIn('"status": "complete"', captured.saved)
        # Exercise the actual producer -> validator contract on the same
        # serialized report, including the top-level GT-use flags.
        real_read_text = Path.read_text

        def read_score(self, *args, **kwargs):
            return captured.saved if self == output else real_read_text(self, *args, **kwargs)

        with mock.patch.object(Path, "read_text", new=read_score), \
             mock.patch.object(evaluate.predict, "validate_prediction", side_effect=validate), \
             mock.patch.object(evaluate, "_read_slot_seed1", side_effect=slot_rows), \
             mock.patch.object(evaluate, "sha256_file", side_effect=fake_sha), \
             mock.patch.object(evaluate, "_dataset_identity", return_value={
                 "resolved_path": "/registered/clevr.h5", "size_bytes": 100, "mtime_ns": 200}):
            self.assertTrue(evaluate.validate_score(output))


if __name__ == "__main__":
    unittest.main()
