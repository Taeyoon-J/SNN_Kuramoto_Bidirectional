import os
import unittest

from collaborative_test.SW_0167_structured_relation_energy import training_queue


class TrainingQueueTests(unittest.TestCase):
    def test_exact_worker_module_and_arm_command(self):
        argv = training_queue._worker_command("structured_hinge", 3)
        self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0167_structured_relation_energy.training_queue"])
        self.assertEqual(argv[3:], ["--worker", "--arm", "structured_hinge", "--physical-gpu", "3"])
        with self.assertRaises(ValueError):
            training_queue._worker_command("unregistered", 0)

    def test_parent_queue_import_hides_cuda_and_arm_status_paths_are_disjoint(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertNotEqual(training_queue._state_path("structured_hinge"),
                            training_queue._state_path("local_margin"))
        self.assertTrue(training_queue._state_path("structured_hinge").name.endswith("_state.json"))


if __name__ == "__main__":
    unittest.main()
