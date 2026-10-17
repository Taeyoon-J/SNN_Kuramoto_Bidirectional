import unittest

from collaborative_test.SW_0187_joint_encoder_unique_scaling import prediction_queue as q


class _Owner:
    def _proc_record(self, pid):
        if pid != 501: return None
        return {"state": "S", "argv": ["python", "-m", q.TRAIN_MODULE, "--source-seed", "1"]}


class PredictionQueueContracts(unittest.TestCase):
    def test_registered_training_pending_states_are_not_terminal_errors(self):
        owner = _Owner()
        statuses = {"validating_train_inputs", "waiting_for_exclusive_gpu", "waiting_for_stable_gpu",
            "running_worker", "yielding_foreign_owner"}
        self.assertEqual(statuses, q.ACTIVE_TRAIN)
        for status in statuses:
            state = {"experiment": "SW0187_training_queue", "seed": 1,
                "supervisor_pid": 501, "status": status}
            self.assertIs(q.training_terminal(owner, 1, state), False)

    def test_wrong_module_or_worker_identity_is_not_live(self):
        owner = _Owner()
        self.assertTrue(q._module_live(owner, 501, q.TRAIN_MODULE, 1, worker=False))
        self.assertFalse(q._module_live(owner, 501, q.MODULE, 1, worker=False))
        self.assertFalse(q._module_live(owner, 501, q.TRAIN_MODULE, 2, worker=False))
        self.assertFalse(q._module_live(owner, 501, q.TRAIN_MODULE, 1, worker=True))

    def test_worker_command_binds_armset_seed_and_physical_gpu(self):
        command = q.worker_command(2, 3)
        self.assertEqual(command[command.index("-m") + 1], q.MODULE)
        self.assertEqual(command[command.index("--source-seed") + 1], "2")
        self.assertEqual(command[command.index("--physical-gpu") + 1], "3")
        with self.assertRaises(ValueError): q.worker_command(3, 3)

    def test_cli_help_loads_without_creating_runtime_state(self):
        with self.assertRaises(SystemExit) as result: q.main(["--help"])
        self.assertEqual(result.exception.code, 0)


if __name__ == "__main__": unittest.main()
