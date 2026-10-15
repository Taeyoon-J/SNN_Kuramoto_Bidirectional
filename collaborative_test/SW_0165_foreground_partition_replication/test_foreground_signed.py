"""The replication imports the sealed SW0164 solver without edits."""
import hashlib, unittest
from pathlib import Path
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as frozen
from collaborative_test.SW_0165_foreground_partition_replication import foreground_signed as localcopy

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class SolverLineage(unittest.TestCase):
 def test_local_review_copy_is_byte_identical_to_frozen_solver(self):
  self.assertEqual(sha(localcopy.__file__),sha(frozen.__file__))
 def test_production_solver_preserves_singletons_background_and_connectivity(self):
  import numpy as np
  q=np.full((6,6),.5); a=np.zeros_like(q); labels=np.array([[1,1,0],[0,2,3]],np.int32)
  result=frozen.foreground_induced_merge(q,a,1.0,labels,grid_shape=(2,3))
  np.testing.assert_array_equal(result["labels"]>0,labels>0)
  self.assertEqual(result["discarded_mass"],0)
  self.assertEqual(result["merge_count"],0)
if __name__=="__main__": unittest.main()
