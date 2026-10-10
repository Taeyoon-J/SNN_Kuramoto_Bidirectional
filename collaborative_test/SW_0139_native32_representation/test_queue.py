from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for item in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from collaborative_test.SW_0139_native32_representation import pilot_queue as queue
from collaborative_test.SW_0139_native32_representation import run


class SW0139QueueTests(unittest.TestCase):
    def test_task_graph_keeps_control_context_independent_and_cross_view_calibrated(self):
        tasks = {t["task_id"]: t for t in queue.task_plan()}
        self.assertEqual(len(tasks), 11)
        self.assertEqual(tasks["preflight_control_seed1"]["depends_on"], [])
        self.assertEqual(tasks["preflight_context_residual_seed1"]["depends_on"], [])
        self.assertEqual(tasks["preflight_cross_view_seed1"]["depends_on"], ["lambda_seed0"])
        self.assertEqual(tasks["train_control_seed1"]["depends_on"], ["preflight_control_seed1"])
        self.assertIn("lambda_seed0", tasks["train_cross_view_seed1"]["depends_on"])
        self.assertEqual(tasks["score_seed1"]["depends_on"], [
            "predict_control_seed1", "predict_context_residual_seed1", "predict_cross_view_seed1"])

    def test_commands_are_module_invocations_and_bind_arm_artifacts(self):
        task = next(t for t in queue.task_plan() if t["task_id"] == "train_cross_view_seed1")
        argv = queue.command(task, "cuda:0")
        self.assertIn("-m", argv)
        self.assertIn("--stage", argv)
        self.assertIn("train", argv)
        self.assertIn("cross_view", argv)
        self.assertIn("--preflight-path", argv)
        self.assertIn(str(run.LAMBDA_PATH), argv)

    def test_score_child_hides_cuda_before_import_while_model_stages_map_gpu(self):
        score = next(t for t in queue.task_plan() if t["stage"] == "score")
        predict = next(t for t in queue.task_plan() if t["stage"] == "predict")
        score_env = queue.child_environment(score, 2, "/tmp/sw139-score")
        predict_env = queue.child_environment(predict, 2, "/tmp/sw139-predict")
        self.assertEqual(score_env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(predict_env["CUDA_VISIBLE_DEVICES"], "2")

    def test_preflight_validator_accepts_producer_schema_and_rejects_bad_credit(self):
        fixture = ROOT / ("sw139_queue_fixture_" + uuid.uuid4().hex[:10])
        fixture.mkdir()
        path = fixture / "preflight.json"
        task = {"task_id": "preflight_cross_view_seed1", "stage": "preflight",
                "seed": 1, "arm": "cross_view"}
        row = {"status": "disposable_update_complete", "seed": 1, "arm": "cross_view",
               "ground_truth_used": False, "implementation_fingerprint": run.implementation_fingerprint(),
               "optimizer_updates": 1, "steps": [{"changed_parameters": 1, "gradient_norm": 1.0}],
               "source_unchanged_except_trainable_graph": True,
               "separate_old_contrastive_gradient_families": {
                   "old": {"encoder": 2.0, "graph": 1.0},
                   "contrastive": {"encoder": 4.0, "graph": 3.0}},
               "lambda_sha256": "lambda-sha"}
        artifact = queue.artifact_path
        try:
            path.write_text(json.dumps(row), encoding="utf-8")
            real_sha = queue.run.sha256_file
            def bound_sha(value):
                if Path(value) == Path(queue.run.LAMBDA_PATH):
                    return "lambda-sha"
                return real_sha(value)
            with mock.patch.object(queue, "artifact_path", return_value=path), \
                 mock.patch.object(queue.run, "sha256_file", side_effect=bound_sha):
                self.assertTrue(queue.valid_result(task))
            row["separate_old_contrastive_gradient_families"]["contrastive"]["graph"] = 0.0
            path.write_text(json.dumps(row), encoding="utf-8")
            with mock.patch.object(queue, "artifact_path", return_value=path), \
                 mock.patch.object(queue.run, "sha256_file", side_effect=bound_sha):
                self.assertFalse(queue.valid_result(task))
        finally:
            if path.exists():
                path.unlink()
            fixture.rmdir()

    def test_dry_run_states_no_retry_and_gt_late_contract(self):
        tasks = queue.task_plan()
        self.assertEqual(queue.MAX_PARALLEL, 3)
        self.assertEqual(queue.command(tasks[-1], "cuda:0")[1:3], ["-m", "collaborative_test.SW_0139_native32_representation.evaluate"])
        self.assertFalse(any(t["stage"] == "evaluate" for t in tasks))
        self.assertEqual(tasks[-1]["stage"], "score")


if __name__ == "__main__":
    unittest.main()
