import unittest
from summarize import summarize

class SummaryTests(unittest.TestCase):
    def test_strict_all_metric_gate(self):
        base={"fg_ari":.7,"foreground_iou":.4,"matched_object_iou":.5}
        rows={"baseline":base,"E":dict(base),
              "F":{"fg_ari":.71,"foreground_iou":.41,"matched_object_iou":.51},
              "G":{"fg_ari":.8,"foreground_iou":.3,"matched_object_iou":.6}}
        x=summarize(rows)
        self.assertFalse(x["arms"]["E"]["all_three_strictly_improved"])
        self.assertTrue(x["arms"]["F"]["all_three_strictly_improved"])
        self.assertFalse(x["arms"]["G"]["all_three_strictly_improved"])
if __name__=="__main__": unittest.main()
