import unittest
import numpy as np
from collaborative_test.SW_0185_qce_spatial_connectivity import completion_audit_v2 as v2

class ExplicitSlotArchives(unittest.TestCase):
    def test_native_pixel_archive_pooling_without_any_candidate_reference_mapping(self):
        ids=np.arange(1320,1640,dtype=np.int64)
        pixels=np.zeros((320,128,128),dtype=np.int32)
        pixels[:,:4,:2]=7;pixels[:,:4,2:4]=3
        labels=v2.checked_slot_arrays(ids,pixels,expected_ids=ids)
        self.assertEqual(labels.shape,(320,32,32))
        self.assertTrue(np.all(labels[:,0,0]==3))

    def test_other_ID_order_or_native_resolution_cannot_silently_pass(self):
        ids=np.arange(1320,1640,dtype=np.int64);pixels=np.zeros((320,128,128),dtype=np.int32)
        with self.assertRaises(ValueError):v2.checked_slot_arrays(ids[::-1],pixels,expected_ids=ids)
        with self.assertRaises(ValueError):v2.checked_slot_arrays(ids,pixels[:,:32,:32],expected_ids=ids)

    def test_cli_help_does_not_read_ground_truth(self):
        with self.assertRaises(SystemExit) as result:v2.main(['--help'])
        self.assertEqual(result.exception.code,0)

if __name__=='__main__':unittest.main()
