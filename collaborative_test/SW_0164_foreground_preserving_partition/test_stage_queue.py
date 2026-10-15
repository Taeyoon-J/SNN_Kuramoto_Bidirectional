"""CPU safety contracts for the owner-aware SW0164 stage supervisor."""
import unittest
import json,shutil,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import h5py,numpy as np
from collaborative_test.SW_0164_foreground_preserving_partition import stage_queue as q
from collaborative_test.SW_0164_foreground_preserving_partition import preflight,predict_evaluate as pe
from collaborative_test.SW_0162_connected_signed_partition import preflight as source
import hashlib

def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<16),b""): h.update(b)
 return h.hexdigest()

class StageQueueContracts(unittest.TestCase):
 def _fixture(self,root,ids):
  refs={}
  for name in ("source_T","source_qcc","slot"):
   labels=np.zeros((len(ids),32,32),np.int64); labels[:,:,16:]=1
   npz=root/(name+".npz"); proto=root/(name+".json"); np.savez(npz,image_ids=ids,labels=labels); proto.write_text("{}")
   refs[name]={"ids":ids,"labels":labels,"prediction_sha256":_sha(npz),"protocol_sha256":_sha(proto),"protocol_path":proto,"prediction_path":npz,"archive_path":npz,"npz_path":npz,"baseline_sha256":"f"*64}
  rgbpath=root/"rgb.npy"; np.save(rgbpath,np.zeros((2,128,128,3),np.uint8)); st=rgbpath.stat(); dataset=root/"data.h5"
  expected={"checkpoint_sha256":"c"*64,"manifest_sha256":"d"*64,"ordered_train_ids_sha256":"e"*64,
   "rgb_cache_size_bytes":10,"rgb_cache_mtime_ns":12,"rgb_manifest_sha256":"f"*64,"source_positions":list(preflight.POSITIONS),
   "selected_pool_rows":list(range(8)),"selected_image_ids":list(range(8)),"rgb_batch_sha256":"1"*64,
   "encoder_source_sha256":"2"*64,"feature_stats_sha256":"3"*64,"feature_stats_mode":"standardize","feature_clip":3.0,"rgb_image_sha256":["4"*64]*8}
  source_record={**expected,"source_state_sha256_before":"5"*64,"source_state_sha256_after":"5"*64}
  images=[{"rgb_sha256":"4"*64,"trace_sha256":{k:format(i+1,"064x") for i,k in enumerate(source.TRACE_FIELDS)},"q_sha256":"a"*64,"actual_a_sha256":"b"*64,
   "trace_parity":True,"q_parity":True,"qcc_parity":True,"foreground_support_exact":True,"deterministic_solver_replay":True} for _ in range(8)]
  pref={"experiment":"SW0164_foreground_preserving_partition","status":"source8_preflight_complete","seed":1,"ground_truth_used":False,"optimizer_updates":0,
   "implementation_fingerprint":{"pref":"c"},"source":source_record,"source_operator":{"arm":"T","steps":1024,"settle":512,"live_tail":0,"optimizer_updates":0,"backend":"factorized"},"images":images}
  prepared={"references":refs,"rgb":np.zeros((2,128,128,3),np.uint8),"rgb_path":rgbpath,"rgb_manifest_sha256":"6"*64,"rgb_hashes":["7"*64,"8"*64],
   "dataset":dataset,"dataset_path":str(dataset.resolve()),"dataset_size_bytes":0,"dataset_mtime_ns":0,"preflight":pref,"preflight_sha256":"9"*64}
  return prepared,expected,pref

 def _write_rows(self,ids,prepared,work,pred):
  folder=pe._work(work); folder.mkdir(parents=True); bindings=pe._bindings(prepared); bsha=pe._json_sha(bindings)
  pe._atomic_json(folder/"bindings.json",{"bindings":bindings,"bindings_sha256":bsha})
  control=prepared["references"]["source_T"]["labels"]
  traces={k:format(i+1,"064x") for i,k in enumerate(source.TRACE_FIELDS)}
  for i,image_id in enumerate(ids):
   labels=np.zeros((32,32),np.int32); labels[:,16:24]=1; labels[:,24:]=2
   pe._write_row(folder,i,int(image_id),prepared["rgb_hashes"][i],bsha,labels,control[i].astype(np.int32),"a"*64,"b"*64,1.,0,traces)
  model=bindings["source_model_state_sha256"]
  pe._atomic_json(folder/"source_state.json",{"source_state_sha256_before":model,"source_state_sha256_after":model,"source_unchanged":True,"bindings_sha256":bsha,"row_commits_sha256":pe._ledger_sha(folder)})
 def test_commands_are_stage_specific_and_prediction_never_scores(self):
  pre=q.command("source8_preflight"); pred=q.command("predict")
  self.assertIn("collaborative_test.SW_0164_foreground_preserving_partition.preflight",pre)
  self.assertIn("collaborative_test.SW_0164_foreground_preserving_partition.predict_evaluate",pred)
  self.assertIn("--stage",pred); self.assertIn("predict",pred)
  with self.assertRaises(ValueError): q.command("score")
 def test_final_physical_idle_guard_rejects_uuid_owner_and_memory(self):
  obj=q.StageQueue.__new__(q.StageQueue)
  for value,expected in [(("uuid-x",10,[]),True),(("uuid-y",10,[]),False),(("uuid-x",10,[77]),False),(("uuid-x",513,[]),False)]:
   obj.owner=SimpleNamespace(gpu_guard=SimpleNamespace(_gpu_info=lambda gpu,v=value:v))
   ok,detail=obj._idle(1,"uuid-x")
   self.assertEqual(ok,expected); self.assertIn("reason",detail)
 def test_waiting_preflight_requires_registered_live_owner_process(self):
  obj=q.StageQueue.__new__(q.StageQueue)
  obj.owner=SimpleNamespace(_proc_record=lambda pid:{"state":"S","argv":["collaborative_test.SW_0162_connected_signed_partition.preflight_queue"]} if pid==12 else None)
  self.assertTrue(obj._preflight_live({"supervisor_pid":12}))
  self.assertFalse(obj._preflight_live({"supervisor_pid":13}))
  obj.owner=SimpleNamespace(_proc_record=lambda pid:{"state":"Z","argv":["collaborative_test.SW_0162_connected_signed_partition.preflight_queue"]})
  self.assertFalse(obj._preflight_live({"supervisor_pid":12}))
 def test_queue_fingerprint_resolves_real_dependencies(self):
  fp=q.implementation_fingerprint()
  for name in ("stage_queue.py","predict_evaluate.py","preflight.py","foreground_signed.py","protocol.json","DESIGN_KO.md"):
   key=f"collaborative_test/SW_0164_foreground_preserving_partition/{name}"
   self.assertEqual(len(fp[key]),64)

 def test_run_lifecycle_uses_real_serialized_preflight_prediction_and_score_validators(self):
  root=Path(__file__).resolve().parents[2]/("queue164"+uuid.uuid4().hex[:8]); root.mkdir()
  ids=np.asarray([1320,1321]); prepared,expected,record=self._fixture(root,ids); dataset=prepared["dataset"]
  with h5py.File(dataset,"w") as h:
   masks=h.create_dataset("mask",shape=(1322,128,128,1),dtype=np.uint8,chunks=(1,128,128,1)); a=np.zeros((2,128,128,1),np.uint8); a[:,:,:64]=1; masks[1320:1322]=a
  st=dataset.stat(); prepared.update(dataset_size_bytes=st.st_size,dataset_mtime_ns=st.st_mtime_ns)
  runtime=root/"runtime"; pred=root/"pred"; work=root/"work"; score=root/"score.json"; pref=root/"preflight.json"
  upstream=root/"upstream.json"; upstream.write_text(json.dumps({"experiment":"SW0162_connected_signed_partition","status":"complete"}))
  lock_api=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=4,flock=lambda *args:None)
  original={"runtime":q.RUNTIME,"state":q.STATE,"lock":q.LOCK,"log":q.LOG,"source_state":q.source_queue.STATE,
   "pref_output":preflight.OUTPUT,"predroot":pe.PREDICTION_ROOT,"workroot":pe.WORK_ROOT,"score":pe.SCORE_PATH,
   "count":pe.COUNT,"ids":pe.IMAGE_IDS,"contract_count":pe.contract.COUNT,"contract_ids":pe.contract.IMAGE_IDS}
  fixedfp={"fixture":"164"}
  test_case=self
  try:
   q.RUNTIME=runtime; q.STATE=runtime/"state.json"; q.LOCK=runtime/"queue.lock"; q.LOG=runtime/"prediction.log"
   q.source_queue.STATE=upstream; preflight.OUTPUT=pref; pe.PREDICTION_ROOT=pred; pe.WORK_ROOT=work; pe.SCORE_PATH=score
   pe.COUNT=2; pe.IMAGE_IDS=ids; pe.contract.COUNT=2; pe.contract.IMAGE_IDS=ids
   with (mock.patch.object(q,"fcntl",lock_api),mock.patch.object(q,"implementation_fingerprint",return_value={"queue":"a"}),
    mock.patch.object(q.evaluation,"implementation_fingerprint",return_value={"eval":"b"}),
    mock.patch.object(q.source_queue,"validate_report",return_value=True),mock.patch.object(q.source_queue,"source_identity",return_value={}),mock.patch.object(q.evaluation,"_expected_source",return_value=expected),
    mock.patch.object(q.evaluation,"prepare_prediction_inputs_cpu",return_value=prepared),mock.patch.object(q.evaluation,"prepare_prediction_cpu",return_value=prepared),
    mock.patch.object(q.evaluation,"implementation_fingerprint",return_value={"eval":"b"}),
    mock.patch.object(pe,"implementation_fingerprint",return_value=fixedfp),mock.patch.object(preflight,"implementation_fingerprint",return_value={"pref":"c"}),
    mock.patch.object(pe,"validate_preflight",return_value=True),mock.patch.object(preflight,"validate_report",wraps=preflight.validate_report),
    mock.patch.object(q.StageQueue,"_gpu_stage",autospec=True) as gpu_stage):
    def fake_stage(self,stage,*,prepared=None):
     if stage=="source8_preflight": pref.write_text(json.dumps(record),encoding="utf-8")
     else:
      test_case._write_rows(ids,prepared,work,pred)
      pe.publish_predictions(prepared=prepared,prediction_root=pred,work_root=work)
     return 0
    gpu_stage.side_effect=fake_stage
    status=q.StageQueue(prepared_fn=lambda:prepared).run()
   self.assertEqual(status,0)
   final=json.loads(q.STATE.read_text()); self.assertEqual(final["status"],"complete")
   self.assertEqual([call.args[1] for call in gpu_stage.call_args_list],["source8_preflight","predict"])
   self.assertTrue(final["ground_truth_used_for_scoring"]); self.assertFalse(final["ground_truth_used_for_prediction"])
  finally:
   q.RUNTIME=original["runtime"]; q.STATE=original["state"]; q.LOCK=original["lock"]; q.LOG=original["log"]
   q.source_queue.STATE=original["source_state"]; preflight.OUTPUT=original["pref_output"]
   pe.PREDICTION_ROOT=original["predroot"]; pe.WORK_ROOT=original["workroot"]; pe.SCORE_PATH=original["score"]
   pe.COUNT=original["count"]; pe.IMAGE_IDS=original["ids"]; pe.contract.COUNT=original["contract_count"]; pe.contract.IMAGE_IDS=original["contract_ids"]
   p=root.resolve()
   if p.is_relative_to(Path(__file__).resolve().parents[2]): shutil.rmtree(p)

if __name__=="__main__": unittest.main()
