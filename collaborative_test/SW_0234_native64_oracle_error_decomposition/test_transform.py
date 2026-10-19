import unittest
import numpy as np
from .transform import transform


class TransformTests(unittest.TestCase):
    def test_merge_retains_background_only_groups_and_foreground_support(self):
        p=np.array([[7,8,0,9],[7,8,0,9]]);t=np.array([[1,1,0,0],[1,1,0,0]])
        out=transform(p,t,merge=True)
        np.testing.assert_array_equal(out>0,p>0)
        self.assertEqual(out[0,0],out[0,1]);self.assertNotEqual(out[0,0],out[0,3])

    def test_coverage_preserves_existing_assignments_and_recovers_lost_object(self):
        p=np.array([[0,5,0],[0,0,0]]);t=np.array([[1,1,0],[2,2,0]])
        out=transform(p,t,cover=True)
        self.assertEqual(out[0,0],5);self.assertEqual(out[1,0],out[1,1]);self.assertGreater(out[1,0],0)
        self.assertNotEqual(out[1,0],5);self.assertEqual(out[0,2],0)

    def test_combined_operation_does_not_silently_split_straddling_group(self):
        p=np.array([[5,5],[0,0]]);t=np.array([[1,2],[1,2]])
        out=transform(p,t,merge=True,cover=True)
        self.assertEqual(out[0,0],out[0,1])
        np.testing.assert_array_equal(transform(p,t),p)


if __name__=='__main__':unittest.main()
