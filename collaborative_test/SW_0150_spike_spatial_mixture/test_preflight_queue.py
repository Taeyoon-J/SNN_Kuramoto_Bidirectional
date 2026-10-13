from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import unittest
import uuid
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # preload before torch on Windows environments with multiple OpenMP DLLs
_NUMPY_PRELOADED = np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0150_spike_spatial_mixture import preflight, preflight_queue as queue


GROUPS = {"encoder", "graph", "native_core", "integration_scalars", "decoder"}


class ScratchDirectory:
    """Create a test-owned workspace directory; remove only its known descendants."""
    def __init__(self, prefix):
        self.path = ROOT / f".{prefix}_{uuid.uuid4().hex}"

    def __enter__(self):
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, *_exc):
        root = self.path.resolve()
        if not root.is_relative_to(ROOT.resolve()):
            raise RuntimeError("test scratch path escaped repository")
        children = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
        for child in children:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        root.rmdir()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_report(warm_path: Path, parity_path: Path) -> dict:
    ids = list(range(20000, 24096))
    assets = {key: ("a" if index % 2 == 0 else "b") * 64 for index, key in enumerate(
        ("train_cache_sha256", "train_manifest_sha256", "validation_cache_sha256",
         "validation_manifest_sha256", "encoder_source_sha256",
         "feature_preprocessing_sha256"))}
    groups = sorted(GROUPS)
    arms = {}
    for arm in ("encoder_graph", "joint"):
        arms[arm] = {
            "status": "disposable_two_updates_complete", "optimizer_updates": 2,
            "history": [{"mean_nll": 1.0, "family_gradient_norms":
                         {group: 0.1 for group in groups}} for _ in range(2)],
            "initial_parameter_group_sha256": {group: "c" * 64 for group in groups},
            "final_parameter_group_sha256": {group: "d" * 64 for group in groups},
            "changed_parameter_groups": {group: True for group in groups},
            "image_ids": list(range(20000, 20032)), "final_mean_nll": 0.9,
        }
    return {
        "experiment": preflight.EXPERIMENT,
        "stage": "source1_shared_warm_and_disposable_preflight",
        "status": "source_preflight_complete", "seed": 1,
        "training_admission": False, "ground_truth_used": False, "masks_read": False,
        "optimizer_updates": {"shared_decoder_warmup": 32, "encoder_graph": 2, "joint": 2},
        "exposures": {"warmup_images": 512, "each_disposable_arm_images": 32,
                      "full_native32_70000_image_training": False},
        "implementation_fingerprint": preflight.implementation_fingerprint(),
        "source": {"training_ids": ids, "training_ids_sha256": preflight._ids_sha(ids),
                   "rgb_asset_validation": assets,
                   "checkpoint_sha256": "e" * 64, "manifest_sha256": "f" * 64},
        "saved_source_parity": {"path": str(parity_path),
                                "sha256": digest(parity_path.read_bytes()),
                                "status": "native32_zero_adapter_parity_passed",
                                "source1_actual_source_exact_T1024_settle512": True},
        "warm_checkpoint": {"path": str(warm_path), "sha256": digest(warm_path.read_bytes()),
                            "shared_exactly_across_disposable_arms": True,
                            "optimizer_state_steps": {"maximum": 32, "count": 10}},
        "likelihood": {"sigma2": 0.02}, "arms": arms,
    }


class QueueTests(unittest.TestCase):
    def test_plan_and_child_command_are_only_the_registered_preflight(self):
        tasks = queue.task_plan()
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual((task["stage"], task["seed"], task["depends_on"]), ("preflight", 1, []))
        argv = queue.command(task)
        self.assertIn("-m", argv)
        self.assertIn("collaborative_test.SW_0150_spike_spatial_mixture.preflight", argv)
        self.assertEqual(argv[argv.index("--output") + 1], task["output"])
        self.assertEqual(argv[argv.index("--warm-output") + 1], task["warm_output"])
        self.assertEqual(argv[argv.index("--parity") + 1], task["parity"])
        self.assertNotIn("train", " ".join(argv))
        self.assertFalse(Path(task["warm_output"]).resolve().is_relative_to(ROOT.resolve()))

    def test_report_validator_binds_production_fields_and_warm_bytes(self):
        with ScratchDirectory("sw150_queue") as tmp:
            root = Path(tmp)
            parity = root / "parity.json"
            parity.write_text("{}\n", encoding="utf-8")
            warm = root / "warm.pt"
            warm.write_bytes(b"synthetic CPU fixture only")
            report = valid_report(warm, parity)
            with mock.patch.object(queue, "PARITY", parity):
                self.assertTrue(queue.validate_report(report, warm_artifact=warm))
                wrong_ids = json.loads(json.dumps(report))
                wrong_ids["source"]["training_ids"][-1] += 1
                self.assertFalse(queue.validate_report(wrong_ids, warm_artifact=warm))
                self.assertFalse(queue.validate_report(report, warm_artifact=warm,
                                                      warm_digest="0" * 64))
                parity.write_text("changed\n", encoding="utf-8")
                self.assertFalse(queue.validate_report(report, warm_artifact=warm))

    def test_existing_outputs_and_warm_markers_refuse_restart(self):
        with ScratchDirectory("sw150_no_restart") as tmp:
            marker = Path(tmp) / "warm.pt"
            marker.write_bytes(b"preserve")
            with self.assertRaisesRegex(FileExistsError, "no restart"):
                queue.check_no_existing_artifacts([marker])
            self.assertEqual(marker.read_bytes(), b"preserve")

    def _queue_fixture(self, root: Path):
        runtime = root / "runtime"
        runtime.mkdir()
        output, log, warm = root / "result.json", runtime / "main.log", root / "private_warm.pt"
        parity = root / "parity.json"
        parity.write_text("{}\n", encoding="utf-8")
        task = {"task_id": "preflight_seed1", "stage": "preflight", "seed": 1,
                "depends_on": [], "output": str(output), "warm_output": str(warm),
                "parity": str(parity), "log": str(log)}
        q = queue.PreflightQueue.__new__(queue.PreflightQueue)
        q.tasks = [task]
        q.stop_event = threading.Event()
        q.state_lock = threading.RLock()
        q.queue_lock = None
        q.state = {"status": "running", "tasks": {task["task_id"]:
                    {"task": task, "status": "queued"}}}
        return q, task, runtime, output, log, warm, parity

    def test_actual_execute_argv_terminal_status_and_artifact_validation(self):
        with ScratchDirectory("sw150_execute") as tmp:
            q, task, runtime, output, log, warm, parity = self._queue_fixture(Path(tmp))
            lease_path = Path(tmp) / "gpu2.lock"
            observed = {}

            class CompletedChild:
                pid = os.getpid()

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            def popen(argv, **kwargs):
                observed["argv"] = argv
                observed["env"] = kwargs["env"]
                observed["cwd"] = kwargs["cwd"]
                self.assertEqual(kwargs["start_new_session"], True)
                self.assertEqual(len(kwargs["pass_fds"]), 1)
                warm.write_bytes(b"warm fixture")
                report = valid_report(warm, parity)
                output.write_text(json.dumps(report), encoding="utf-8")
                return CompletedChild()

            with (mock.patch.object(queue, "RUNTIME", runtime),
                  mock.patch.object(queue, "STATE", runtime / "state.json"),
                  mock.patch.object(queue, "OUTPUT", output),
                  mock.patch.object(queue, "LOG", log),
                  mock.patch.object(queue, "WARM_ARTIFACT", warm),
                  mock.patch.object(queue, "PARITY", parity),
                  mock.patch.object(queue.owner_queue, "_acquire",
                                    return_value=(2, "GPU-2", 8, lease_path.open("a+"))),
                  mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True),
                  mock.patch.object(queue.subprocess, "Popen", side_effect=popen)):
                self.assertEqual(q.run(), 0)

            self.assertEqual(observed["cwd"], queue.ROOT)
            self.assertEqual(observed["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertIn("collaborative_test.SW_0150_spike_spatial_mixture.preflight",
                          observed["argv"])
            self.assertEqual(q.state["status"], "preflight_complete")
            self.assertEqual(q.state["tasks"][task["task_id"]]["status"], "passed")
            self.assertTrue(q.state["tasks"][task["task_id"]]["artifact_valid"])

    def test_nonzero_child_is_terminal_and_is_not_restarted(self):
        with ScratchDirectory("sw150_failed_child") as tmp:
            q, task, runtime, output, log, warm, parity = self._queue_fixture(Path(tmp))

            class FailedChild:
                pid = os.getpid()

                def poll(self):
                    return 7

                def wait(self):
                    return 7

            with (mock.patch.object(queue, "RUNTIME", runtime),
                  mock.patch.object(queue, "STATE", runtime / "state.json"),
                  mock.patch.object(queue, "OUTPUT", output),
                  mock.patch.object(queue, "LOG", log),
                  mock.patch.object(queue, "WARM_ARTIFACT", warm),
                  mock.patch.object(queue, "PARITY", parity),
                  mock.patch.object(queue.owner_queue, "_acquire",
                                    return_value=(0, "GPU-0", 4,
                                                  (Path(tmp) / "gpu0.lock").open("a+"))),
                  mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True),
                  mock.patch.object(queue.subprocess, "Popen", return_value=FailedChild()) as spawn):
                self.assertEqual(q.run(), 1)
                self.assertEqual(spawn.call_count, 1)
            self.assertEqual(q.state["status"], "preflight_failed")
            self.assertEqual(q.state["tasks"][task["task_id"]]["status"],
                             "failed_preflight_or_artifact_validation")
            self.assertFalse(output.exists())

    def test_unstable_reservation_is_released_before_second_admission(self):
        with ScratchDirectory("sw150_reacquire") as tmp:
            q, task, runtime, output, log, warm, parity = self._queue_fixture(Path(tmp))
            lease_a_path, lease_b_path = Path(tmp) / "a.lock", Path(tmp) / "b.lock"
            lease_a, lease_b = lease_a_path.open("a+"), lease_b_path.open("a+")
            acquired = [(1, "GPU-1", 9, lease_a), (3, "GPU-3", 7, lease_b)]

            class CompletedChild:
                pid = os.getpid()

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            def popen(argv, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
                warm.write_bytes(b"warm after second admission")
                output.write_text(json.dumps(valid_report(warm, parity)), encoding="utf-8")
                return CompletedChild()

            with (mock.patch.object(queue, "RUNTIME", runtime),
                  mock.patch.object(queue, "STATE", runtime / "state.json"),
                  mock.patch.object(queue, "OUTPUT", output),
                  mock.patch.object(queue, "LOG", log),
                  mock.patch.object(queue, "WARM_ARTIFACT", warm),
                  mock.patch.object(queue, "PARITY", parity),
                  mock.patch.object(queue.owner_queue, "_acquire", side_effect=acquired) as acquire,
                  mock.patch.object(queue.owner_queue, "_stable_idle", side_effect=[False, True]),
                  mock.patch.object(queue.subprocess, "Popen", side_effect=popen)):
                self.assertEqual(q.run(), 0)
            self.assertEqual(acquire.call_count, 2)
            self.assertTrue(lease_a.closed)
            self.assertTrue(lease_b.closed)
            self.assertEqual(q.state["tasks"][task["task_id"]]["physical_gpu"], 3)


if __name__ == "__main__":
    unittest.main()
