import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, arm_config


class ArmMatrixTests(unittest.TestCase):
    def test_four_coarse_mechanistic_arms(self):
        self.assertEqual(set(ARMS), {"A", "B", "C", "D"})
        self.assertEqual(arm_config("A")["reconstruction_weight"], 1.0)
        self.assertEqual(arm_config("B")["reconstruction_weight"], 0.0)
        self.assertEqual(arm_config("C")["initialization"], "sw0072_seed1_core_trainable")
        self.assertEqual(arm_config("D")["reconstruction_weight"], 0.3)
        self.assertNotEqual(arm_config("A")["initialization"], arm_config("C")["initialization"])

    def test_reject_unknown_arm(self):
        with self.assertRaises(ValueError):
            arm_config("E")


if __name__ == "__main__":
    unittest.main()
