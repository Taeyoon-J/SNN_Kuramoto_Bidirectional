"""Seed-bound SW0165 source-preflight validator tests."""
import json,shutil,unittest,uuid
from pathlib import Path
from collaborative_test.SW_0165_foreground_partition_replication import preflight as pf
from collaborative_test.SW_0162_connected_signed_partition import preflight as source

class PreflightTests(unittest.TestCase):
 def setUp(self): self.tmp=Path(__file__).resolve().parent/("pref165-"+uuid.uuid4().hex[:10]); self.tmp.mkdir()
 def tearDown(self):
  root=Path(__file__).resolve().parent.resolve(); p=self.tmp.resolve()
  if p.is_relative_to(root): shutil.rmtree(p)
 def record(self,seed):
  trace={k:format(i+1,"064x") for i,k in enumerate(source.TRACE_FIELDS)}
  expected={"seed":seed,"checkpoint_sha256":"c"*64,"manifest_sha256":"d"*64,"ordered_train_ids_sha256":"e"*64,
   "rgb_cache_size_bytes":10,"rgb_cache_mtime_ns":12,"rgb_manifest_sha256":"f"*64,"source_positions":list(pf.POSITIONS),
   "selected_pool_rows":list(range(8)),"selected_image_ids":list(range(8)),"rgb_batch_sha256":"1"*64,"encoder_source_sha256":"2"*64,
   "feature_stats_sha256":"3"*64,"feature_stats_mode":"standardize","feature_clip":3.0,"rgb_image_sha256":["4"*64]*8}
  src={**expected,"source_state_sha256_before":"5"*64,"source_state_sha256_after":"5"*64}
  images=[{"source_position":pos,"rgb_sha256":"4"*64,"trace_parity":True,"q_parity":True,"qcc_parity":True,
   "foreground_support_exact":True,"deterministic_solver_replay":True,"trace_sha256":trace,"q_sha256":"a"*64,"actual_a_sha256":"b"*64} for pos in pf.POSITIONS]
  rec={"experiment":"SW0165_foreground_partition_replication","status":"source8_preflight_complete","seed":seed,"ground_truth_used":False,
   "optimizer_updates":0,"implementation_fingerprint":{"x":"a"},"source_operator":{"arm":"T","steps":1024,"settle":512,"live_tail":0,
   "optimizer_updates":0,"backend":"factorized"},"source":src,"images":images}
  return rec,expected
 def write(self,rec):
  p=self.tmp/"r.json"; p.write_text(json.dumps(rec),encoding="utf-8"); return p
 def test_seeds_are_explicit_and_namespaced(self):
  self.assertEqual(pf.SEEDS,(0,2)); self.assertNotEqual(pf.output_for_seed(0),pf.output_for_seed(2))
  for seed in (-1,1,3):
   with self.assertRaises(ValueError): pf.output_for_seed(seed)
 def test_validator_binds_requested_seed_not_any_registered_seed(self):
  for seed in (0,2):
   rec,expected=self.record(seed); p=self.write(rec)
   self.assertTrue(pf.validate_report(p,expected_seed=seed,expected_source=expected,expected_fingerprint={"x":"a"}))
   self.assertFalse(pf.validate_report(p,expected_seed=1,expected_source=expected))
   rec["seed"]=2 if seed==0 else 0; p.write_text(json.dumps(rec),encoding="utf-8")
   self.assertFalse(pf.validate_report(p,expected_seed=seed))
 def test_source_or_any_train_trace_tampering_rejects(self):
  rec,expected=self.record(0); p=self.write(rec)
  self.assertFalse(pf.validate_report(p,expected_seed=0,expected_source={**expected,"checkpoint_sha256":"9"*64}))
  rec["images"][7]["trace_parity"]=False; p.write_text(json.dumps(rec),encoding="utf-8")
  self.assertFalse(pf.validate_report(p,expected_seed=0,expected_source=expected))
if __name__=="__main__": unittest.main()