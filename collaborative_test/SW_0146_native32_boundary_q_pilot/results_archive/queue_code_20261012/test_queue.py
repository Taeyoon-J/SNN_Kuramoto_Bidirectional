"""CPU tests for the private SW0146 queue contract."""
from __future__ import annotations
import importlib.util
import os
import sys
import tempfile
import unittest
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "patch_v2_sw"))
spec = importlib.util.spec_from_file_location("sw0146_probe_queue_test", HERE / "probe_queue.py")
queue = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = queue
spec.loader.exec_module(queue)


class QueueContractTests(unittest.TestCase):
    def test_seven_task_plan_and_two_preflight_barrier(self):
        tasks = queue.task_plan()
        self.assertEqual(len(tasks), 7)
        ids = [row["task_id"] for row in tasks]
        self.assertEqual(ids, ["preflight_encoder_graph", "preflight_joint",
                               "train_encoder_graph", "train_joint",
                               "predict_encoder_graph", "predict_joint", "score"])
        deps = queue._dependencies()
        self.assertEqual(deps["train_joint"], ("preflight_encoder_graph", "preflight_joint"))
        self.assertEqual(deps["score"], ("predict_encoder_graph", "predict_joint"))
        self.assertEqual(deps["predict_joint"], ("train_joint",))

    def test_score_child_hides_cuda_and_gpu_child_maps_device(self):
        source = {"CUDA_VISIBLE_DEVICES": "3", "PATH": "preserved"}
        score = queue.child_environment("score", base=source)
        train = queue.child_environment("train", 2, source)
        self.assertEqual(score["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(train["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(score["OMP_NUM_THREADS"], "1")
        self.assertEqual(train["MKL_NUM_THREADS"], "1")
        self.assertEqual(score["PATH"], "preserved")
        with self.assertRaises(ValueError):
            queue.child_environment("preflight", 4, source)

    def test_invalid_terminal_artifact_cancels_waiting_reservations(self):
        stop = threading.Event()
        self.assertFalse(queue._record_outcome(stop, False))
        self.assertTrue(stop.is_set())
        self.assertTrue(queue._record_outcome(stop, True))

    def test_stable_idle_window_resets_on_owner_appearance(self):
        now = [0.0]
        samples = iter([("GPU-A", 10, []), ("GPU-A", 10, [99])])
        class Stop:
            def is_set(self): return False
        result = queue.helper.stable_idle_window(
            lambda: next(samples), stop_event=Stop(), monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds), duration=10, interval=5)
        self.assertEqual(result, (False, "owner_or_memory"))

    @unittest.skipIf(os.name != "posix", "pass_fds is a Linux queue guarantee")
    def test_gpu_lease_fd_can_be_inherited_by_child(self):
        import subprocess
        with tempfile.TemporaryDirectory(prefix="sw146fd-") as tmp:
            fd = os.open(Path(tmp) / "gpu.lock", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                proc = subprocess.run([sys.executable, "-c", "import os,sys; os.fstat(int(sys.argv[1]))", str(fd)],
                                      pass_fds=(fd,), capture_output=True, check=False)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
