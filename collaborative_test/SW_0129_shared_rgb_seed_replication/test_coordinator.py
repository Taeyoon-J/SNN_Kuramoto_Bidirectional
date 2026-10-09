import unittest
from collaborative_test.SW_0129_shared_rgb_seed_replication import coordinator

class CoordinatorTests(unittest.TestCase):
 def test_score_contract_rejects_missing_or_mismatched_per_image_rows(self):
  score={'metrics':{m:.5 for m in coordinator.METRICS},'valid_count':{m:320 for m in coordinator.METRICS},'per_image':{m:[.5]*320 for m in coordinator.METRICS}}
  row={'sweep':[{'scored_targets':{'our_hdf5':score}}]}
  self.assertTrue(coordinator._valid_score(row))
  score['per_image']['fg_ari'][0]=float('nan')
  self.assertFalse(coordinator._valid_score(row))
 def test_commands_use_new_runner_and_fixed_seed_pair(self):
  for task in coordinator.task_plan():
   argv=coordinator.command(task)
   self.assertIn(str(coordinator.run.RUNNER),argv)
   self.assertNotIn('SW0106_spike_partition_rgb/run.py',' '.join(argv))

if __name__=='__main__':unittest.main()
