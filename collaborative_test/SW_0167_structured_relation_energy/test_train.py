import numpy as np
import torch
import unittest

from collaborative_test.SW_0167_structured_relation_energy import train, structured_energy


class _Source:
    def __init__(self, rows):
        self.rows = rows
    def load_row(self, index):
        return self.rows[index]


class _Context:
    def __init__(self, rows):
        self.rows = rows
    def load(self, index):
        return self.rows[index]


class TrainTests(unittest.TestCase):
    def test_fixed_exposure_schedule_is_eight_full_permutations(self):
        one = train._epoch_orders()
        two = train._epoch_orders()
        self.assertEqual(one, two)
        self.assertEqual(len(one), 8)
        self.assertTrue(all(len(x) == 512 and sorted(x) == list(range(512)) for x in one))
        self.assertEqual((train.LOGICAL_BATCH, train.MICRO, train.UPDATES), (16, 4, 256))
        self.assertEqual(train.EPOCHS * 512, train.UPDATES * train.LOGICAL_BATCH)

    def test_one_logical_update_runs_both_registered_losses_on_shared_features(self):
        rng = np.random.default_rng(81)
        n, t = 4, 12
        source_rows, context_rows = [], []
        supports, targets = [], []
        for _ in range(16):
            s = rng.random((4, n, t), dtype=np.float32)
            q = rng.random((n, n), dtype=np.float32)
            q = (q + q.T) * 0.5
            np.fill_diagonal(q, 1.0)
            a = rng.random((n, n), dtype=np.float32)
            a = (a + a.T) * 0.5
            np.fill_diagonal(a, 0.0)
            source_rows.append((s, np.zeros((2, 2), dtype=np.int64)))
            context_rows.append((q, a))
            support = np.ones((2, 2), dtype=np.int32)
            target = np.array([[1, 1], [2, 2]], dtype=np.int32)
            supports.append(support)
            targets.append(target)
        caches = {"source": _Source(source_rows), "context": _Context(context_rows), "gain": 7.6}
        for arm in train.ARMS:
            torch.manual_seed(train.HEAD_SEED)
            head = structured_energy.PairRelationMLP()
            initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
            optimizer = torch.optim.Adam(head.parameters(), lr=train.LR)
            row = train._update(arm, head, optimizer, list(range(16)), caches,
                                supports, targets, torch.device("cpu"))
            self.assertTrue(np.isfinite(row["mean_loss"]))
            self.assertTrue(np.isfinite(row["gradient_norm"]))
            self.assertEqual(row["images_seen"], 16)
            self.assertTrue(any(not torch.equal(initial[k], v) for k, v in head.state_dict().items()))

    def test_artifact_validator_rejects_wrong_arm_and_partial_history(self):
        out = train.HERE / "results_archive" / "test_nonexistent_partial_artifact"
        with self.assertRaises((FileNotFoundError, ValueError)):
            train.validate_training_artifact("structured_hinge", out)
        with self.assertRaises(ValueError):
            train.validate_training_artifact("unknown_arm")


if __name__ == "__main__":
    unittest.main()
