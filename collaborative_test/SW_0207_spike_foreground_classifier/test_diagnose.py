import unittest
import numpy as np
from .diagnose import foreground_scores


class ForegroundScoreTests(unittest.TestCase):
    def test_object_macro_weighting_and_label_renaming(self):
        g=np.array([[0,0,1],[0,2,2]],np.int64);p=np.array([[0,1,1],[0,1,0]],np.int64)
        score=foreground_scores(p,g)
        self.assertAlmostEqual(score['object_balanced_foreground_recall'],.75)
        self.assertAlmostEqual(score['background_accuracy'],2/3)
        self.assertAlmostEqual(score['class_balanced_foreground_accuracy'],(.75+2/3)/2)
        self.assertEqual(score,foreground_scores(p,np.where(g>0,9-g,0)))

    def test_degenerate_classifier_is_not_mistaken_for_high_accuracy(self):
        g=np.array([[0,0],[1,2]],np.int64)
        for p in (np.zeros_like(g),np.ones_like(g)):
            self.assertEqual(foreground_scores(p,g)['class_balanced_foreground_accuracy'],.5)
