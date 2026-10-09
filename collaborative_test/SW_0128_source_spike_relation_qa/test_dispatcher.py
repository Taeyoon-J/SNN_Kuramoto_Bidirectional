import unittest

from collaborative_test.SW_0128_source_spike_relation_qa import dispatcher


class DispatcherTests(unittest.TestCase):
    def test_generation_precedes_global_scoring(self):
        tasks = dispatcher.task_plan()
        self.assertEqual([row["stage"] for row in tasks], ["generate", "score"])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_id"]])

    def test_gpu_must_be_both_owner_free_and_below_memory_limit(self):
        self.assertTrue(dispatcher.gpu_is_exclusive(512, []))
        self.assertFalse(dispatcher.gpu_is_exclusive(513, []))
        self.assertFalse(dispatcher.gpu_is_exclusive(16, [12345]))

    def test_only_generation_gets_cuda_device_flag(self):
        generate, score = dispatcher.task_plan()
        gen_argv = dispatcher.SourceRelationDispatcher._argv(generate, "cuda:0")
        score_argv = dispatcher.SourceRelationDispatcher._argv(score, "cpu")
        self.assertIn("--device", gen_argv)
        self.assertIn("cuda:0", gen_argv)
        self.assertNotIn("--device", score_argv)


if __name__ == "__main__":
    unittest.main()
