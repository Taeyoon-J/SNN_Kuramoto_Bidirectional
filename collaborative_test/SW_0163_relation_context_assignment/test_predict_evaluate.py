"""CPU integration contracts for SW0163 shared prediction and GT-last score."""
from __future__ import annotations
import hashlib, json, shutil, unittest, uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import torch

from collaborative_test.SW_0163_relation_context_assignment import predict_evaluate as pe
from collaborative_test.SW_0163_relation_context_assignment import model
from collaborative_test.SW_0162_connected_signed_partition import preflight as sw162
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(65536),b""): h.update(b)
    return h.hexdigest()

class PredictionEvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=Path(__file__).resolve().parents[2]/("tpe"+uuid.uuid4().hex[:8]); self.tmp.mkdir()
    def tearDown(self):
        p=self.tmp.resolve()
        if not p.is_relative_to(Path(__file__).resolve().parents[2]): raise RuntimeError("unsafe test cleanup")
        shutil.rmtree(p)

    def test_source_image_uses_real_kuramoto_capture_and_one_qcc_control(self):
        ids=np.asarray([1320]); qcc=np.zeros((1,32,32),dtype=np.int64); qcc[:,:,16:]=1
        coupling=object(); graph=SimpleNamespace(kuramoto=coupling)
        found=SimpleNamespace(wrapped=SimpleNamespace(core=graph),encode=lambda image:torch.zeros((1,8,1024)))
        trace={key:torch.ones(1) for key in sw162.TRACE_FIELDS}
        trace["component_spikes"]=torch.ones((1,4,1024,1024))
        q=torch.full((1,1024,1024),.5); a=torch.full((1,1024,1024),.01)
        called=[]
        class Head:
            def __call__(self,s,qarg,aarg):
                called.append((s.shape,qarg,aarg)); p=torch.zeros((1,1024,11)); p[...,2]=1
                return p,None,None
        heads={name:Head() for name in pe.ARMS}
        def capture(target,fn):
            self.assertIs(target,coupling); called.append("capture"); return fn(),a
        with (mock.patch.object(pe.sw162,"capture_source_coupling",side_effect=capture),
              mock.patch.object(pe,"late_rollout",return_value=trace) as rollout,
              mock.patch.object(pe.sw162,"_actual_q",return_value=q),
              mock.patch.object(pe.sw136,"_qcc_labels",return_value=qcc),
              mock.patch.object(pe.sw162,"_validate_actual_A"),
              mock.patch.object(pe.sw162,"_tensor_sha",return_value="a"*64)):
            output,control,_=pe.predict_source_image(found,heads,torch.zeros((1,128,128,3)),
                {"source_T":{"labels":qcc}},0)
        self.assertEqual(called[0],"capture")
        rollout.assert_called_once()
        self.assertEqual(len(called),5)
        self.assertTrue(all(item[0]==(1,4,1024,512) and item[1] is q and item[2] is a for item in called[1:]))
        self.assertEqual(set(output),set(pe.ARMS)); self.assertEqual(np.unique(output["Q_A"]).tolist(),[0])
        np.testing.assert_array_equal(control,qcc[0])

    def _fixture(self):
        ids=np.asarray([1320,1321],dtype=np.int64)
        source=np.zeros((2,32,32),dtype=np.int64); source[:,:,16:]=1
        qcc=source.copy(); qcc[:,:,0]=2
        slot=source.copy(); refs={}
        for name,labels in (("source_T",source),("source_qcc",qcc),("slot",slot)):
            npz=self.tmp/f"{name}.npz"; proto=self.tmp/f"{name}.json"
            np.savez(npz,image_ids=ids,labels=labels); proto.write_text(json.dumps({"status":"complete"}))
            refs[name]={"ids":ids,"labels":labels,"prediction_sha256":sha(npz),"protocol_sha256":sha(proto),
                "protocol_path":proto,"arm":"T","baseline_sha256":"b"*64}
            if name=="source_T": refs[name]["npz_path"]=npz
            elif name=="source_qcc": refs[name]["archive_path"]=npz
            else: refs[name]["prediction_path"]=npz
        rgb=self.tmp/"rgb.npy"; rgb.write_bytes(b"fixture")
        dataset=self.tmp/"canonical.h5"
        source_binding={"source_checkpoint_sha256":"c"*64,"source_manifest_sha256":"d"*64,
            "source_model_state_sha256":"e"*64,"source_trace_manifest_sha256":"f"*64,
            "context_cache_manifest_sha256":"1"*64,"preflight_report_sha256":"2"*64}
        training={a:{"bindings":source_binding,"checkpoint_sha256":(str(i+3)*64)[:64],"final_model_state_sha256":(str(i+7)*64)[:64]} for i,a in enumerate(pe.ARMS)}
        return ids,source,refs,{"references":refs,"training":training,"training_manifest_sha256":{a:(str(i+1)*64)[:64] for i,a in enumerate(pe.ARMS)},
          "rgb_path":rgb,"rgb_manifest_sha256":"3"*64,"rgb_hashes":["4"*64,"5"*64],"dataset":dataset,
          "dataset_path":str(dataset.resolve()),"dataset_size_bytes":0,"dataset_mtime_ns":0}

    def _source_state_receipt(self,work,bsha,prepared):
        expected=prepared["training"][pe.ARMS[0]]["bindings"]["source_model_state_sha256"]
        pe._write_once(pe._source_state_path(work),{"experiment":"SW0163_relation_context_assignment",
          "bindings_sha256":bsha,"expected_source_state_sha256":expected,
          "source_state_sha256_before":expected,"source_state_sha256_after":expected,
          "source_unchanged":True,"row_count":2,"row_commits_sha256":pe._ledger_sha(work),
          "implementation_fingerprint":pe.implementation_fingerprint()})

    def test_reference_resolver_normalizes_real_heterogeneous_validator_paths(self):
        ids=np.asarray([1320,1321]); labs=np.zeros((2,32,32),dtype=np.int64)
        qcc={"ids":ids,"labels":labs,"prediction_sha256":"a"*64,"protocol_sha256":"b"*64,
             "archive_path":self.tmp/"qcc.npz"}
        t={"ids":ids,"labels":labs,"prediction_sha256":"c"*64,"protocol_sha256":"d"*64,
           "npz_path":self.tmp/"t.npz","arm":"T"}
        slotdir=self.tmp/"seed1_epoch10"; slotdir.mkdir(); snp=slotdir/"predictions.npz"; sp=slotdir/"protocol.json"
        baseline=self.tmp/"baseline.json"; baseline.write_text("{}"); np.savez(snp,image_ids=ids,labels=labs)
        sp.write_text("{}")
        with (mock.patch.object(pe,"IMAGE_IDS",ids),mock.patch.object(pe,"COUNT",2),
              mock.patch.object(pe.sw137,"validate_prediction",return_value=qcc),
              mock.patch.object(pe.sw143,"validate_arm_prediction",return_value=t),
              mock.patch.object(pe,"SLOT_ROOT",self.tmp),mock.patch.object(pe,"SLOT_BASELINE",baseline),
              mock.patch.object(pe,"SLOT_BASELINE_SHA256",sha(baseline)),
              mock.patch.object(contract,"validate_slot_protocol",return_value=sha(snp)),
              mock.patch.object(contract,"require_fixed_image_ids",side_effect=lambda x,**kw:np.asarray(x)),
              mock.patch.object(contract,"slot_native_pixels_to32",side_effect=lambda x:np.asarray(x))):
            refs=pe._references()
        self.assertEqual(refs["source_qcc"]["prediction_path"],qcc["archive_path"])
        self.assertEqual(refs["source_T"]["prediction_path"],t["npz_path"])
        self.assertEqual(refs["source_qcc"]["archive_path"],qcc["archive_path"])

    def test_source_control_keeps_native_label_capacity_but_heads_stay_eleven_way(self):
        ids=np.asarray([1320]); control=np.zeros((32,32),dtype=np.int32); control[0,0]=127
        heads={a:np.zeros((32,32),dtype=np.int32) for a in pe.ARMS}
        trace={k:"a"*64 for k in sw162.TRACE_FIELDS}; trace.update(q_sha256="b"*64,actual_a_sha256="c"*64)
        good=self.tmp/"good"; pe._write_row(good,0,1320,"d"*64,"e"*64,heads,control,trace)
        _,loaded,_=pe._read_row(good,0,1320,"d"*64,"e"*64)
        self.assertEqual(int(loaded.max()),127)
        bad_heads={a:v.copy() for a,v in heads.items()}; bad_heads["Q_A"][0,0]=11
        bad=self.tmp/"bad"; pe._write_row(bad,0,1320,"d"*64,"e"*64,bad_heads,np.zeros((32,32),dtype=np.int32),trace)
        with self.assertRaises(ValueError): pe._read_row(bad,0,1320,"d"*64,"e"*64)

    def test_public_rows_publish_validate_and_reject_tampering(self):
        ids,source,refs,prepared=self._fixture()
        with (mock.patch.object(pe,"COUNT",2),mock.patch.object(pe,"IMAGE_IDS",ids),
              mock.patch.object(contract,"require_fixed_image_ids",side_effect=lambda got,**kw: np.asarray(got))):
            workroot=self.tmp/"work"; work=pe._work(workroot); binds=pe._bindings(prepared); bsha=pe._json_sha(binds)
            pe._write_once(work/"bindings.json",{"bindings":binds,"bindings_sha256":bsha})
            predictions={a:np.zeros((32,32),dtype=np.int32) for a in pe.ARMS}
            for a in pe.ARMS: predictions[a][:,8*(pe.ARMS.index(a)+1):]=pe.ARMS.index(a)+1
            trace={k:"a"*64 for k in sw162.TRACE_FIELDS}; trace.update(q_sha256="b"*64,actual_a_sha256="c"*64)
            for i,image_id in enumerate(ids): pe._write_row(work,i,int(image_id),prepared["rgb_hashes"][i],bsha,predictions,source[i].astype(np.int32),trace)
            self._source_state_receipt(work,bsha,prepared)
            outroot=self.tmp/"predictions"
            pe.publish_predictions(prepared=prepared,prediction_root=outroot,work_root=workroot)
            for arm in (*pe.ARMS,"source_T_control"):
                result=pe.validate_prediction(arm,prediction_root=outroot,work_root=workroot,prepared=prepared)
                self.assertEqual(result["labels"].shape,(2,32,32))
            path=outroot/"Q_A_seed1"/"predictions.npz"
            with np.load(path,allow_pickle=False) as z: altered=z["labels"].copy(); saved_ids=z["image_ids"].copy()
            altered[0,0,0]=9
            with path.open("wb") as f: np.savez_compressed(f,image_ids=saved_ids,labels=altered)
            with self.assertRaises(ValueError): pe.validate_prediction("Q_A",prediction_root=outroot,work_root=workroot,prepared=prepared)

    def test_score_producer_recomputes_frozen_predictions_and_rejects_coherent_edit(self):
        import h5py
        ids,source,refs,prepared=self._fixture()
        # Small fixture dimensions replace only the registered row-count constants.
        gtfile=prepared["dataset"]
        with h5py.File(gtfile,"w") as h:
            masks=h.create_dataset("mask",shape=(1322,128,128,1),dtype=np.uint8,chunks=(1,128,128,1),compression="lzf")
            g=np.zeros((2,128,128,1),dtype=np.uint8); g[:,:,:64,0]=1; masks[1320:1322]=g
        stat=gtfile.stat(); prepared.update(dataset_size_bytes=stat.st_size,dataset_mtime_ns=stat.st_mtime_ns)
        with (mock.patch.object(pe,"COUNT",2),mock.patch.object(pe,"IMAGE_IDS",ids),
              mock.patch.object(contract,"COUNT",2),mock.patch.object(contract,"IMAGE_IDS",ids),
              mock.patch.object(contract,"require_fixed_image_ids",side_effect=lambda got,**kw: np.asarray(got)),
              mock.patch.object(pe,"implementation_fingerprint",return_value={"test":"z"*64})):
            workroot=self.tmp/"work"; work=pe._work(workroot); binding=pe._bindings(prepared); bsha=pe._json_sha(binding)
            pe._write_once(work/"bindings.json",{"bindings":binding,"bindings_sha256":bsha})
            row_labels={a:source[0].astype(np.int32).copy() for a in pe.ARMS}
            for j,a in enumerate(pe.ARMS): row_labels[a][:,j*4:(j+1)*4]=2
            trace={k:"a"*64 for k in sw162.TRACE_FIELDS}; trace.update(q_sha256="b"*64,actual_a_sha256="c"*64)
            for i,image_id in enumerate(ids): pe._write_row(work,i,int(image_id),prepared["rgb_hashes"][i],bsha,row_labels,source[i].astype(np.int32),trace)
            self._source_state_receipt(work,bsha,prepared)
            predroot=self.tmp/"scorepreds"; pe.publish_predictions(prepared=prepared,prediction_root=predroot,work_root=workroot)
            scorepath=self.tmp/"score.json"
            report=pe.score_all(output=scorepath,prediction_root=predroot,work_root=workroot,dataset=gtfile,prepared=prepared)
            checked=pe.validate_score(scorepath,prediction_root=predroot,work_root=workroot,dataset=gtfile,prepared=prepared)
            self.assertEqual(set(checked["scores"]),set((*pe.ARMS,"source_T_control","source_T","source_qcc","Slot70k")))
            self.assertTrue(checked["ground_truth_opened_after_all_predictions_validated"])
            wrong_seed=json.loads(scorepath.read_text()); wrong_seed["seed_scope"]=[0]
            scorepath.write_text(json.dumps(wrong_seed),encoding="utf-8")
            with self.assertRaises(ValueError): pe.validate_score(scorepath,prediction_root=predroot,work_root=workroot,dataset=gtfile,prepared=prepared)
            scorepath.write_text(json.dumps(report),encoding="utf-8")
            edited=json.loads(scorepath.read_text()); edited["means"]["Q_A"]["fg_ari"]+=.01
            scorepath.write_text(json.dumps(edited),encoding="utf-8")
            with self.assertRaises(ValueError): pe.validate_score(scorepath,prediction_root=predroot,work_root=workroot,dataset=gtfile,prepared=prepared)

if __name__=="__main__": unittest.main()
