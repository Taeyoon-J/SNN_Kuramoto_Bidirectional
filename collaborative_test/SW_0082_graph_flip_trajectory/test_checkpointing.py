import tempfile
import unittest
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory import assert_checkpoint_equal, save_epoch_checkpoint


class EpochCheckpointTests(unittest.TestCase):
    def test_epoch_save_refuses_overwrite_and_final_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = nn.Linear(3, 2)
            epoch = save_epoch_checkpoint(model, root / "epochs", 1)
            torch.save(model.state_dict(), root / "final.pt")
            self.assertTrue(assert_checkpoint_equal(root / "final.pt", epoch))
            with self.assertRaises(FileExistsError):
                save_epoch_checkpoint(model, root / "epochs", 1)

    def test_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = nn.Linear(2, 2)
            epoch = save_epoch_checkpoint(model, root / "epochs", 1)
            with torch.no_grad():
                model.weight.add_(1)
            torch.save(model.state_dict(), root / "final.pt")
            with self.assertRaises(AssertionError):
                assert_checkpoint_equal(root / "final.pt", epoch)


if __name__ == "__main__":
    unittest.main()
