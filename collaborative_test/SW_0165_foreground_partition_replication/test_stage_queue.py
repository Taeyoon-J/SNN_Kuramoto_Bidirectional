"""Seed-isolated CPU contracts for the SW0165 supervisor."""
import unittest
from collaborative_test.SW_0165_foreground_partition_replication import stage_queue as q, predict_evaluate as evaluation

class StageQueueContracts(unittest.TestCase):
 def test_seed_is_required_and_embedded_in_each_child_command(self):
  for seed in evaluation.SEEDS:
   pre=q.command("source8_preflight",seed); pred=q.command("predict",seed)
   self.assertIn("collaborative_test.SW_0165_foreground_partition_replication.preflight",pre)
   self.assertIn("collaborative_test.SW_0165_foreground_partition_replication.predict_evaluate",pred)
   self.assertIn("--seed",pre); self.assertIn(str(seed),pre)
   self.assertIn("--seed",pred); self.assertIn(str(seed),pred)
   self.assertIn("--stage",pred); self.assertIn("predict",pred)
  with self.assertRaises(ValueError): q.command("source8_preflight",1)
  with self.assertRaises(ValueError): q.command("score",0)

 def test_fingerprint_resolves_new_namespace_and_seed_specific_artifacts(self):
  fp=q.implementation_fingerprint()
  for name in ("stage_queue.py","predict_evaluate.py","preflight.py","foreground_signed.py","protocol.json","DESIGN_KO.md"):
   key=f"collaborative_test/SW_0165_foreground_partition_replication/{name}"
   self.assertEqual(len(fp[key]),64)
  self.assertEqual(evaluation.score_path(0).name,"evaluation_seed0_20261014.json")
  self.assertEqual(evaluation.score_path(2).name,"evaluation_seed2_20261014.json")
  with self.assertRaises(ValueError): evaluation.score_path(1)

if __name__=="__main__": unittest.main()