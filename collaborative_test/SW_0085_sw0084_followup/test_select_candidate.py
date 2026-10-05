import unittest
from collaborative_test.SW_0085_sw0084_followup.select_candidate import select


class SelectionTest(unittest.TestCase):
    def summary(self):
        base = {"fg_ari": .7, "foreground_iou": .4, "matched_object_iou": .5}
        arms = {}
        for arm in "ABCD":
            arms[arm] = {epoch: {"metrics": dict(base)} for epoch in ("05", "10")}
        return {"baseline_sw0072_seed1": base, "arms": arms}

    def test_prefers_all_three_advance(self):
        data = self.summary()
        data["arms"]["A"]["05"]["metrics"] = {"fg_ari": .71, "foreground_iou": .41,
                                                   "matched_object_iou": .51}
        data["arms"]["B"]["10"]["metrics"] = {"fg_ari": .9, "foreground_iou": .39,
                                                   "matched_object_iou": .8}
        result = select(data)
        self.assertEqual((result["chosen"]["arm"], result["chosen"]["epoch"]), ("A", "05"))
        self.assertTrue(result["full320_promotion"])

    def test_no_advance_selects_best_worst_relative_delta(self):
        data = self.summary()
        for arm in "ABCD":
            for epoch in ("05", "10"):
                data["arms"][arm][epoch]["metrics"] = {
                    "fg_ari": .60, "foreground_iou": .30, "matched_object_iou": .40}
        data["arms"]["A"]["05"]["metrics"] = {"fg_ari": .69, "foreground_iou": .39,
                                                   "matched_object_iou": .49}
        data["arms"]["B"]["10"]["metrics"] = {"fg_ari": .69, "foreground_iou": .4,
                                                   "matched_object_iou": .5}
        result = select(data)
        self.assertEqual((result["chosen"]["arm"], result["chosen"]["epoch"]), ("B", "10"))
        self.assertFalse(result["full320_promotion"])


if __name__ == "__main__":
    unittest.main()

