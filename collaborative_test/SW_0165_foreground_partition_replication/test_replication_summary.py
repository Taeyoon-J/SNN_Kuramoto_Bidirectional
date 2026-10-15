"""Registered aggregate seed-gate semantics for SW0165."""
import shutil,unittest,uuid
from pathlib import Path
import numpy as np
from collaborative_test.SW_0165_foreground_partition_replication import replication_summary as rs

class ReplicationSummaryTests(unittest.TestCase):
 def test_three_seed_mean_gate_requires_two_positive_seeds_not_three(self):
  reports={}
  deltas={0:.005,1:.005,2:None}
  for seed in (0,1,2):
   scores={}
   for metric,base in (("fg_ari",.70),("foreground_iou",.60),("matched_object_iou",.55)):
    source=np.full(320,.5 if metric=="fg_ari" else base,dtype=np.float64)
    candidate=source.copy()
    if metric=="fg_ari":
     if seed==2:
      candidate += np.tile(np.asarray([.4,-.4]),160)
     else: candidate+=deltas[seed]
    elif metric=="matched_object_iou": candidate+=.01
    scores["foreground_induced"]={**scores.get("foreground_induced",{}),metric:{"per_image":candidate.tolist(),"mean":float(candidate.mean())}}
    scores["source_T"]={**scores.get("source_T",{}),metric:{"per_image":source.tolist(),"mean":float(source.mean())}}
    slot=source-.01
    scores["Slot70k"]={**scores.get("Slot70k",{}),metric:{"per_image":slot.tolist(),"mean":float(slot.mean())}}
   reports[seed]={"status":"complete","seed_scope":[seed],"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,
     "source_T_foreground_iou_exact":True,"score_sha256":str(seed)*64,"scores":scores,
     "means":{arm:{m:row["mean"] for m,row in vals.items()} for arm,vals in scores.items()}}
  folder=Path(__file__).resolve().parent/("summary-"+uuid.uuid4().hex[:8]); folder.mkdir()
  try:
   result=rs.summarize(reports,output=folder/"summary.json")
   gate=result["replication_gate"]
   self.assertTrue(gate["candidate_fg_ari_three_seed_mean_above_source_T"])
   self.assertTrue(gate["at_least_two_of_three_seed_fg_ari_gains_positive"])
   self.assertFalse(gate["three_seed_paired_fg_ari_ci_lower_above_zero"])
   self.assertFalse(gate["replication_gate_passed"])
  finally: shutil.rmtree(folder)

if __name__=="__main__": unittest.main()
