"""CPU contract tests for the SW0164 source-only preflight validator."""
import unittest
from pathlib import Path
import numpy as np
from collaborative_test.SW_0164_foreground_preserving_partition import preflight as pf
from collaborative_test.SW_0162_connected_signed_partition import preflight as source

class PreflightContracts(unittest.TestCase):
 def _record(self):
  hashes={k:format(i+1,"064x") for i,k in enumerate(source.TRACE_FIELDS)}
  image={"trace_parity":True,"q_parity":True,"qcc_parity":True,"foreground_support_exact":True,"deterministic_solver_replay":True,
   "trace_sha256":hashes,"q_sha256":"a"*64,"actual_a_sha256":"b"*64,"rgb_sha256":"4"*64}
  expected={"checkpoint_sha256":"c"*64,"manifest_sha256":"d"*64,"ordered_train_ids_sha256":"e"*64,
   "rgb_cache_size_bytes":10,"rgb_cache_mtime_ns":12,"rgb_manifest_sha256":"f"*64,"source_positions":list(pf.POSITIONS),
   "selected_pool_rows":list(range(8)),"selected_image_ids":list(range(8)),"rgb_batch_sha256":"1"*64,
   "encoder_source_sha256":"2"*64,"feature_stats_sha256":"3"*64,"feature_stats_mode":"standardize","feature_clip":3.0,
   "rgb_image_sha256":["4"*64]*8}
  source_record={**expected,"source_state_sha256_before":"5"*64,"source_state_sha256_after":"5"*64}
  return {"experiment":"SW0164_foreground_preserving_partition","status":"source8_preflight_complete","seed":1,
   "ground_truth_used":False,"optimizer_updates":0,"implementation_fingerprint":{"fp":"a"},
   "source_operator":{"arm":"T","steps":1024,"settle":512,"live_tail":0,"optimizer_updates":0,"backend":"factorized"},
   "source":source_record,"images":[dict(image,source_position=pos) for pos in pf.POSITIONS]}, expected

 def test_complete_report_binds_all_eight_train_source_rows(self):
  record,expected=self._record(); path=Path(__file__).with_name("_preflight_contract_fixture.json")
  try:
   path.write_text(__import__("json").dumps(record),encoding="utf-8")
   self.assertTrue(pf.validate_report(path,expected_fingerprint={"fp":"a"},expected_source=expected))
   record["images"][5]["foreground_support_exact"]=False
   path.write_text(__import__("json").dumps(record),encoding="utf-8")
   self.assertFalse(pf.validate_report(path,expected_fingerprint={"fp":"a"},expected_source=expected))
  finally:
   if path.exists(): path.unlink()

 def test_source_identity_mismatch_is_rejected(self):
  record,expected=self._record(); path=Path(__file__).with_name("_preflight_contract_fixture.json")
  try:
   path.write_text(__import__("json").dumps(record),encoding="utf-8")
   changed=dict(expected,checkpoint_sha256="9"*64)
   self.assertFalse(pf.validate_report(path,expected_fingerprint={"fp":"a"},expected_source=changed))
   self.assertFalse(pf.validate_report(path,expected_fingerprint={"fp":"wrong"},expected_source=expected))
  finally:
   if path.exists(): path.unlink()

if __name__=="__main__": unittest.main()
