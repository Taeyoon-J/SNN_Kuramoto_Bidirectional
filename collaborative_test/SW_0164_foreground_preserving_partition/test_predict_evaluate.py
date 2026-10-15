"""Public fixed320 prediction, source-control, and GT-last CPU contracts."""
import hashlib,json,shutil,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import h5py,numpy as np,torch
from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as pe
from collaborative_test.SW_0162_connected_signed_partition import preflight as source
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<16),b""): h.update(b)
 return h.hexdigest()

class PredictionEvaluationTests(unittest.TestCase):
 def setUp(self): self.tmp=Path(__file__).resolve().parents[2]/("tpe164"+uuid.uuid4().hex[:8]); self.tmp.mkdir()
 def tearDown(self):
  root=Path(__file__).resolve().parents[2].resolve(); p=self.tmp.resolve()
  if not p.is_relative_to(root): raise RuntimeError("unsafe test cleanup")
  shutil.rmtree(p)

 def _refs(self,ids):
  refs={}
  for name in ("source_T","source_qcc","slot"):
   labels=np.zeros((len(ids),32,32),np.int64); labels[:,:,16:]=1
   if name=="source_qcc": labels[:,0,0]=127
   npz=self.tmp/(name+".npz"); proto=self.tmp/(name+".json")
   np.savez(npz,image_ids=ids,labels=labels); proto.write_text('{"status":"complete"}',encoding="utf-8")
   row={"ids":ids,"labels":labels,"prediction_sha256":sha(npz),"protocol_sha256":sha(proto),"protocol_path":proto}
   if name=="source_T": row.update(arm="T",npz_path=npz)
   elif name=="source_qcc": row.update(archive_path=npz)
   else: row.update(prediction_path=npz,archive_path=npz,baseline_sha256="f"*64)
   refs[name]=row
  return refs

 def _prepared(self,ids):
  refs=self._refs(ids); rgbpath=self.tmp/"rgb.npy"; np.save(rgbpath,np.zeros((2,128,128,3),np.uint8))
  dataset=self.tmp/"canonical.h5"
  sb={"checkpoint_sha256":"1"*64,"manifest_sha256":"2"*64,"source_state_sha256_before":"3"*64,
   "ordered_train_ids_sha256":"4"*64,"checkpoint_path":"source.pt","manifest_path":"source.json"}
  images=[{"rgb_sha256":"5"*64} for _ in range(8)]
  training_source={**sb,"source_state_sha256_after":"3"*64}
  pre={"source":training_source,"source_operator":{"arm":"T","steps":1024,"settle":512,"live_tail":0,"optimizer_updates":0},
   "experiment":"SW0164_foreground_preserving_partition","status":"source8_preflight_complete","seed":1,
   "ground_truth_used":False,"optimizer_updates":0,"implementation_fingerprint":{},"images":images}
  return {"references":refs,"rgb":np.zeros((2,128,128,3),np.uint8),"rgb_path":rgbpath,"rgb_manifest_sha256":"6"*64,
   "rgb_hashes":["7"*64,"8"*64],"dataset":dataset,"dataset_path":str(dataset.resolve()),"dataset_size_bytes":0,"dataset_mtime_ns":0,
   "preflight":pre,"preflight_sha256":"9"*64}

 def _write_complete_rows(self,ids,prepared,workroot,predroot):
  work=pe._work(workroot); work.mkdir(parents=True)
  binds=pe._bindings(prepared); bsha=pe._json_sha(binds); pe._atomic_json(work/"bindings.json",{"bindings":binds,"bindings_sha256":bsha})
  src=prepared["references"]["source_T"]["labels"]
  trace={k:format(i+1,"064x") for i,k in enumerate(source.TRACE_FIELDS)}
  for i,image_id in enumerate(ids):
   control=src[i].astype(np.int32).copy(); candidate=np.zeros((32,32),np.int32); fg=control>0
   candidate[fg]=np.where(np.indices((32,32))[1][fg]<24,1,2)
   pe._write_row(work,i,int(image_id),prepared["rgb_hashes"][i],bsha,candidate,control,"a"*64,"b"*64,1.0,0,trace)
  expected=binds["source_model_state_sha256"]
  pe._atomic_json(work/"source_state.json",{"source_state_sha256_before":expected,"source_state_sha256_after":expected,
    "source_unchanged":True,"bindings_sha256":bsha,"row_commits_sha256":pe._ledger_sha(work)})
  return work

 def test_heterogeneous_actual_reference_path_fields_normalize(self):
  ids=np.asarray([1320,1321]); refs=self._refs(ids)
  slotdir=self.tmp/"seed1_epoch10"; slotdir.mkdir(); snp=slotdir/"predictions.npz"; sp=slotdir/"protocol.json"
  np.savez(snp,image_ids=ids,labels=refs["slot"]["labels"]); sp.write_text("{}")
  base=self.tmp/"baseline.json"; base.write_text("{}"); refs["slot"].update(prediction_path=snp,protocol_path=sp)
  with (mock.patch.object(pe,"IMAGE_IDS",ids),mock.patch.object(pe,"COUNT",2),
    mock.patch.object(pe.sw137,"validate_prediction",return_value=refs["source_qcc"]),
    mock.patch.object(pe.sw143,"validate_arm_prediction",return_value=refs["source_T"]),
    mock.patch.object(pe,"SLOT_ROOT",self.tmp),mock.patch.object(pe,"SLOT_BASELINE",base),mock.patch.object(pe,"SLOT_BASELINE_SHA256",sha(base)),
    mock.patch.object(contract,"validate_slot_protocol",return_value=sha(snp)),
    mock.patch.object(contract,"require_fixed_image_ids",side_effect=lambda x,**kw:np.asarray(x)),
    mock.patch.object(contract,"slot_native_pixels_to32",side_effect=lambda x:np.asarray(x))):
   got=pe._references()
  self.assertEqual(got["source_qcc"]["prediction_path"],refs["source_qcc"]["archive_path"])
  self.assertEqual(got["source_T"]["prediction_path"],refs["source_T"]["npz_path"])

 def test_shared_real_solver_call_uses_actual_coupling_and_replays_qcc(self):
  qcc=np.zeros((1,32,32),np.int32); qcc[:,4:7,5:8]=1
  found=SimpleNamespace(wrapped=SimpleNamespace(core=SimpleNamespace(kuramoto=object(),graph_generator=SimpleNamespace(log_coupling_gain=torch.tensor(0.)))) ,encode=lambda image:torch.zeros((1,8,1024)))
  trace={k:torch.ones(1) for k in source.TRACE_FIELDS}; a=torch.full((1,1024,1024),.01); q=torch.full((1,1024,1024),.5)
  calls=[]
  def capture(graph,fn): self.assertIs(graph,found.wrapped.core.kuramoto); calls.append("capture"); return fn(),a
  with (mock.patch.object(pe.source,"capture_source_coupling",side_effect=capture),mock.patch.object(pe,"late_rollout",return_value=trace),
    mock.patch.object(pe.source,"_trace_check"),mock.patch.object(pe.source,"_validate_actual_A"),mock.patch.object(pe.source,"_actual_q",return_value=q),
    mock.patch.object(pe.sw136,"_qcc_labels",return_value=qcc),mock.patch.object(pe.source,"_tensor_sha",return_value="a"*64)):
   labels,control,qsha,asha,gain,merges,th,diagnostics=pe._predict_one(found,torch.zeros((1,128,128,3)),{"source_T":{"labels":qcc[0][None]}},0)
  self.assertEqual(calls,["capture"]); self.assertEqual(gain,1.0); self.assertEqual(merges,0)
  np.testing.assert_array_equal(labels>0,qcc[0]>0); np.testing.assert_array_equal(control,qcc[0])
  self.assertEqual(diagnostics["discarded_foreground_mass"],0)
  self.assertEqual(diagnostics["candidate_foreground_fraction"],diagnostics["control_foreground_fraction"])

 def test_public_row_publisher_validator_and_source_receipt_are_bound(self):
  ids=np.asarray([1320,1321]); prepared=self._prepared(ids); wr=self.tmp/"work"; pred=self.tmp/"pred"
  with (mock.patch.object(pe,"COUNT",2),mock.patch.object(pe,"IMAGE_IDS",ids),
    mock.patch.object(contract,"COUNT",2),mock.patch.object(contract,"IMAGE_IDS",ids),
    mock.patch.object(contract,"require_fixed_image_ids",side_effect=lambda x,**kw:np.asarray(x)),
    mock.patch.object(pe,"implementation_fingerprint",return_value={"cpu_fixture":"a"*64})):
   self._write_complete_rows(ids,prepared,wr,pred)
   pe.publish_predictions(prepared=prepared,prediction_root=pred,work_root=wr)
   for arm in pe.ARMS: self.assertEqual(pe.validate_prediction(arm,prediction_root=pred,work_root=wr,prepared=prepared)["labels"].shape,(2,32,32))
   proto=pred/"foreground_induced_seed1"/"protocol.json"; x=json.loads(proto.read_text()); x["source_state_receipt_sha256"]="0"*64; proto.write_text(json.dumps(x))
   with self.assertRaises(ValueError): pe.validate_prediction("foreground_induced",prediction_root=pred,work_root=wr,prepared=prepared)

 def test_variable_k_candidate_accepts1024_but_rejects1025(self):
  ids=np.asarray([1320,1321]); prepared=self._prepared(ids); wr=self.tmp/"work"; wr.mkdir()
  binds=pe._bindings(prepared); bsha=pe._json_sha(binds)
  labels=(np.arange(1024,dtype=np.int32)+1).reshape(32,32); control=np.ones((32,32),np.int32)
  traces={k:format(i+1,"064x") for i,k in enumerate(source.TRACE_FIELDS)}
  pe._write_row(wr,0,int(ids[0]),prepared["rgb_hashes"][0],bsha,labels,control,"a"*64,"b"*64,1.,0,traces)
  got,_,_=pe._read_row(wr,0,int(ids[0]),prepared["rgb_hashes"][0],bsha)
  self.assertEqual(int(got.max()),1024)
  over=labels.copy(); over[-1,-1]=1025
  pe._rowdir(wr,0).rename(pe._rowdir(wr,0).with_name("0000.old"))
  pe._write_row(wr,0,int(ids[0]),prepared["rgb_hashes"][0],bsha,over,control,"a"*64,"b"*64,1.,0,traces)
  with self.assertRaises(ValueError): pe._read_row(wr,0,int(ids[0]),prepared["rgb_hashes"][0],bsha)

 def test_score_is_gt_last_independent_and_requires_exact_fg_iou(self):
  ids=np.asarray([1320,1321]); prepared=self._prepared(ids); gtpath=prepared["dataset"]
  with h5py.File(gtpath,"w") as h:
   m=h.create_dataset("mask",shape=(1322,128,128,1),dtype=np.uint8,chunks=(1,128,128,1)); arr=np.zeros((2,128,128,1),np.uint8); arr[:,:,:64]=1; m[1320:1322]=arr
  st=gtpath.stat(); prepared.update(dataset_size_bytes=st.st_size,dataset_mtime_ns=st.st_mtime_ns)
  wr=self.tmp/"work"; pred=self.tmp/"pred"; score=self.tmp/"score.json"
  with (mock.patch.object(pe,"COUNT",2),mock.patch.object(pe,"IMAGE_IDS",ids),mock.patch.object(contract,"COUNT",2),mock.patch.object(contract,"IMAGE_IDS",ids),
    mock.patch.object(contract,"require_fixed_image_ids",side_effect=lambda x,**kw:np.asarray(x)),mock.patch.object(pe,"implementation_fingerprint",return_value={"fixture":"b"*64})):
   self._write_complete_rows(ids,prepared,wr,pred); pe.publish_predictions(prepared=prepared,prediction_root=pred,work_root=wr)
   report=pe.score_all(output=score,prediction_root=pred,work_root=wr,dataset=gtpath,prepared=prepared)
   valid=pe.validate_score(score,prediction_root=pred,work_root=wr,dataset=gtpath,prepared=prepared)
   self.assertEqual(valid["status"],"valid"); self.assertTrue(report["source_T_foreground_iou_exact"])
   corrupt=json.loads(score.read_text()); corrupt["seed_scope"]=[0]; score.write_text(json.dumps(corrupt))
   with self.assertRaises(ValueError): pe.validate_score(score,prediction_root=pred,work_root=wr,dataset=gtpath,prepared=prepared)

if __name__=="__main__": unittest.main()
