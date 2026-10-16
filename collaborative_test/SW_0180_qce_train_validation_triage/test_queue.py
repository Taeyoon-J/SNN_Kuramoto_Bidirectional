import json
import shutil
import unittest
from pathlib import Path
from unittest import mock
import uuid
from contextlib import redirect_stdout
from io import StringIO

from collaborative_test.SW_0180_qce_train_validation_triage import queue


class SW0180QueueContractTests(unittest.TestCase):
    def test_worker_command_binds_seed_and_physical_device(self):
        command = queue.worker_command(2, 3)
        self.assertEqual(command[command.index("--seed") + 1], "2")
        self.assertEqual(command[command.index("--physical-gpu") + 1], "3")
        self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
        with self.assertRaises(ValueError):
            queue.worker_command(9, 0)

    def test_training_receipts_must_be_complete_and_no_validation(self):
        states = {}
        repo = Path(__file__).resolve().parents[2]
        central = repo.parent / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        tmp = central / f"sw180_queue_test_{uuid.uuid4().hex}"
        tmp.mkdir()
        fake_trainers, fake_queues = {}, {}
        for seed in queue.run_api.SEEDS:
            trainer = type("Trainer", (), {"UPDATES": 64})
            module_name = ("collaborative_test.SW_0178_temporal_class_competition.training_queue"
                           if seed == 1 else
                           "collaborative_test.SW_0179_qce_seed_replication.training_queue")
            state_path = tmp / f"{seed}.json"
            state = {"experiment": "SW0178_training_queue" if seed == 1 else "SW0179_training_queue",
                     "arm": "product_q_ce", "status": "complete", "completed_updates": 64,
                     "validation_data_read": False, "manifest_sha256": "a" * 64}
            if seed != 1:
                state["seed"] = seed
            state_path.write_text(json.dumps(state), encoding="utf-8")
            if seed == 1:
                q = type("Queue", (), {"_state_path": (lambda p: lambda arm: p)(state_path)})
            else:
                def state_path_for_seed(arm, requested_seed, base=tmp):
                    return base / f"{int(requested_seed)}.json"
                q = type("Queue", (), {"_state_path": staticmethod(state_path_for_seed)})
            fake_trainers[seed] = (trainer, module_name)
            fake_queues[module_name] = q
        with mock.patch.object(queue, "_training_sources", return_value=fake_trainers), \
             mock.patch.object(queue, "_load", side_effect=lambda name: fake_queues[name]):
            ready = queue._training_ready(owner=object())
        self.assertEqual(set(ready), {"0", "1", "2"})
        active = json.loads((tmp / "0.json").read_text(encoding="utf-8"))
        active.update(status="waiting_for_stable_gpu", supervisor_pid=4321)
        (tmp / "0.json").write_text(json.dumps(active), encoding="utf-8")

        class LiveOwner:
            @staticmethod
            def _proc_record(pid):
                return {"state": "S", "argv": ["python", "-m",
                    "collaborative_test.SW_0179_qce_seed_replication.training_queue"]}

        with mock.patch.object(queue, "_training_sources", return_value=fake_trainers), \
             mock.patch.object(queue, "_load", side_effect=lambda name: fake_queues[name]):
            self.assertIs(queue._training_ready(owner=LiveOwner()), False)
        bad_path = tmp / "0.json"
        bad = json.loads(bad_path.read_text(encoding="utf-8")); bad["status"] = "complete"; bad["validation_data_read"] = True
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        with mock.patch.object(queue, "_training_sources", return_value=fake_trainers), \
             mock.patch.object(queue, "_load", side_effect=lambda name: fake_queues[name]):
            with self.assertRaises(ValueError):
                queue._training_ready(owner=object())
        self.assertEqual(tmp.resolve().parent, central.resolve())
        shutil.rmtree(tmp)

    def test_pending_state_requires_live_exact_training_module(self):
        class Owner:
            @staticmethod
            def _proc_record(pid):
                return {"state": "S", "argv": ["python", "-m",
                    "collaborative_test.SW_0179_qce_seed_replication.training_queue", "--arm"]}
        self.assertTrue(queue._proc_module(Owner(), 123, "collaborative_test.SW_0179_qce_seed_replication.training_queue"))
        self.assertFalse(queue._proc_module(Owner(), 123, "collaborative_test.SW_0178_temporal_class_competition.training_queue"))

    def test_complete_terminal_uses_production_state_writer_without_duplicate_status(self):
        terminal = {"status": "complete", "report_path": "/tmp/report.json",
                    "report_sha256": "b" * 64, "seed": 2}
        with mock.patch.object(queue, "_write_state") as writer:
            queue._publish_complete_state(2, terminal)
        writer.assert_called_once_with(2, "complete", report_path="/tmp/report.json",
                                       report_sha256="b" * 64)

    def test_parent_cli_returns_zero_and_emits_json_on_success(self):
        output = StringIO()
        result = {"status": "complete", "seed": 1, "report_sha256": "c" * 64}
        with mock.patch.object(queue, "run", return_value=result), redirect_stdout(output):
            code = queue._worker_entry(["--seed", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
