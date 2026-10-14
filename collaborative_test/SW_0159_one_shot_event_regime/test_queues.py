from __future__ import annotations

import os
import json
import subprocess
import sys
import uuid
import unittest
from unittest import mock

from collaborative_test.SW_0159_one_shot_event_regime import prediction_queue, score_queue, train


class SW0159QueueTests(unittest.TestCase):
    def test_prediction_routes_exact_registered_arm_module_and_unique_runtime(self):
        control = prediction_queue.configure_arm("control")
        candidate = prediction_queue.configure_arm("candidate")
        self.assertNotEqual(control["runtime"], candidate["runtime"])
        self.assertNotEqual(control["output"], candidate["output"])
        argv = prediction_queue.command("candidate")
        self.assertIn("collaborative_test.SW_0159_one_shot_event_regime.predict_evaluate", argv)
        self.assertEqual(argv[argv.index("--arm") + 1], "candidate")
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")

    def test_importing_queue_libraries_preserves_assigned_child_visibility(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(prediction_queue.ROOT)
        env["CUDA_VISIBLE_DEVICES"] = "3"
        code = ("import os; "
            "import collaborative_test.SW_0159_one_shot_event_regime.prediction_queue; "
            "import collaborative_test.SW_0159_one_shot_event_regime.score_queue; "
            "print(os.environ.get('CUDA_VISIBLE_DEVICES'))")
        result = subprocess.run([sys.executable, "-c", code], cwd=prediction_queue.ROOT,
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3")

    def test_score_queue_waits_for_each_live_prediction_owner(self):
        root = score_queue.ROOT / f".test159queue{uuid.uuid4().hex[:8]}"
        arm_dir = root / "control"
        arm_dir.mkdir(parents=True)
        state = {"experiment": train.EXPERIMENT, "arm": "control", "status": "running",
                 "supervisor_pid": 123, "child_pid": 456}
        path = arm_dir / "state.json"
        path.write_text(json.dumps(state), encoding="utf-8")
        try:
            with mock.patch.object(score_queue, "PREDICTION_QUEUE_ROOT", root), \
                 mock.patch.object(score_queue, "_process_has_args",
                                   side_effect=lambda pid, module, arm: pid == 456 and arm == "control"):
                self.assertTrue(score_queue._prediction_queue_is_live("control"))
                self.assertFalse(score_queue._prediction_queue_is_live("candidate"))
                state["status"] = "failed"
                path.write_text(json.dumps(state), encoding="utf-8")
                self.assertFalse(score_queue._prediction_queue_is_live("control"))
        finally:
            path.unlink(); arm_dir.rmdir(); root.rmdir()


if __name__ == "__main__":
    unittest.main()
