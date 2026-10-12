from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import contextlib, io, json, sys, unittest, uuid, tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test'),str(ROOT/'snn_kuramoto_bidirectional')]
from collaborative_test.SW_0142_native32_masked_completion import run
from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder, RelativeSlotRGBDecoder
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract32

class FakeFoundation:
    def encode(self, rgb):
        self.last_masked=rgb.clone()
        return torch.zeros((rgb.shape[0],8,1024),device=rgb.device)
    def rollout(self,gamma,*,total_steps,live_tail_steps):
        self.last_tail=live_tail_steps
        b=gamma.shape[0];t=torch.arange(total_steps,device=gamma.device,dtype=torch.float32)
        base=((t[None,None,None,:] + torch.arange(1024,device=gamma.device)[None,None,:,None])%7<3).float()
        return {'component_spikes':base.expand(b,4,1024,total_steps).clone()}

class MaskedCompletionTests(unittest.TestCase):
    def test_mask_rule_reproducible_quarter_and_exposure_bound(self):
        rgb=torch.full((2,128,128,3),77,dtype=torch.uint8)
        masked,tiles=run.mask_rgb(rgb,[1320,1321],[0,1])
        self.assertEqual(tiles.shape,(2,16));self.assertEqual(len(np.unique(tiles[0])),16)
        self.assertFalse(np.array_equal(tiles[0],tiles[1]))
        self.assertTrue(torch.equal(rgb,torch.full_like(rgb,77)))
        self.assertTrue(torch.equal(masked[0],run.mask_rgb(rgb[:1],[1320],0)[0][0]))
        for row in range(2):
            changed=(masked[row]!=77).any(-1)
            self.assertEqual(int(changed.sum()),16*16*16)
            self.assertTrue(torch.all(masked[row][changed]==128))

    def test_completion_loss_matches_formula_and_has_prediction_gradient(self):
        target=torch.arange(128*128*3,dtype=torch.float32).reshape(1,128,128,3).remainder(251)/250
        pred=(target*.7).clone().requires_grad_(True)
        _,tiles=run.mask_rgb(torch.zeros((1,128,128,3),dtype=torch.uint8),[1320],0)
        loss=run.completion_loss(pred,target,tiles);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.abs().sum()),0)
        mask=run.pixel_mask(tiles,'cpu');sq=(pred.detach()-target).square()
        lm=(sq*mask[...,None]).sum()/(mask.sum()*3);lv=(sq*(~mask)[...,None]).sum()/((~mask).sum()*3)
        expected=(lm+.25*lv)/target.var((1,2),unbiased=False).mean(-1).clamp_min(1e-6)[0]
        self.assertAlmostEqual(float(loss.detach()),float(expected.detach()),places=7)

    def test_preflight_event_occupancy_uses_membrane_threshold_and_gate_emission(self):
        # Event is measured from membrane > .06*exp(b), even when the gate is zero.
        b=torch.tensor([0.,0.,0.,0.])
        membrane=torch.full((1,4,2,1024),.06)
        membrane[0,0,0,:512]=.061;membrane[0,0,0,512:]=.059
        membrane[0,1,0,:512]=.061;membrane[0,1,0,512:]=.059
        gates=torch.ones_like(membrane);gates[0,0,0,0]=0.;gates[0,1,0,1]=.5
        trace={"component_membrane":membrane,"component_gates":gates}
        row=run._event_occupancy(trace,b)
        self.assertEqual(row["event_threshold"],"component_membrane > 0.06 * exp(b[component])")
        self.assertEqual(row["all_steps"]["events"]["per_component"],[.25,.25,0.,0.])
        self.assertAlmostEqual(row["all_steps"]["emitted_spikes"]["per_component"][0],511/2048)
        self.assertAlmostEqual(row["all_steps"]["emitted_spikes"]["per_component"][1],(511.5)/2048)
        self.assertEqual(row["settled_512"]["events"]["per_component"],[0.,0.,0.,0.])

    def test_masked_forward_uses_full_actual_history_and_sw135_head(self):
        torch.manual_seed(142);f=FakeFoundation();binder=NativeSpikeSlotBinder(seed=142);decoder=RelativeSlotRGBDecoder(seed=106)
        rgb=torch.randint(0,256,(1,128,128,3),dtype=torch.uint8)
        out=run.masked_forward(f,binder,decoder,rgb,[1320],17,live_tail=64)
        self.assertEqual(f.last_tail,64);self.assertEqual(tuple(out['P'].shape),(1,1024,11));self.assertEqual(tuple(out['prediction'].shape),(1,128,128,3));self.assertTrue(torch.isfinite(out['loss']))
        out['loss'].backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in binder.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()))

    def test_registered_training_order_is_consumed_as_provided(self):
        original=run.sw130.source_contract;ids=np.arange(5000,9096,dtype=np.int64).tolist();rows=np.arange(4096,dtype=np.int64)
        try:
            run.sw130.source_contract=lambda seed:(None,None,None,rows,ids,'abc')
            got_ids,got_rows=run.ordered_train();self.assertEqual(got_ids,ids);np.testing.assert_array_equal(got_rows,rows)
        finally:run.sw130.source_contract=original

    def test_real_legacy_slot_protocol_without_prediction_sha_uses_contract_digest(self):
        temp=Path(__import__('tempfile').gettempdir())/('sw142_slot_'+uuid.uuid4().hex[:10]);temp.mkdir()
        npz=temp/'predictions.npz';protocol_path=temp/'protocol.json'
        try:
            labels=np.zeros((320,128,128),dtype=np.int16);ids=np.arange(1320,1640,dtype=np.int64)
            with npz.open('xb') as f:np.savez_compressed(f,image_ids=ids,labels=labels)
            training={'seed':1,'unique_training_images':70000,'epochs':10,'batch_size':32,'steps_per_epoch':2188,'total_steps':21880,'iterations':3,'learning_rate':0.0004,'selected_images_sha256':contract32.SLOT_SELECTED_TRAINING_IDS_SHA256,'num_slots':11,'ground_truth_used_for_training':False,'training_id_segments_inclusive':[[0,999],[1640,70639]]}
            protocol={'seed':1,'image_ids':[1320,1639],'count':320,'inference_seed':0,'batch_size':1,'iterations':3,'preprocessing':'full HDF5 RGB image, float32 /127.5 - 1; no crop','resolution':[128,128],'num_slots':11,'checkpoint_sha256':contract32.SLOT_CHECKPOINT_SHA256[1],'model_sha256':contract32.SLOT_MODEL_SHA256,'ground_truth_used_for_prediction':False,'background_rule':'most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties','label_rule':'slot ID + 1; selected background slot set to 0','training_protocol':training}
            self.assertNotIn('prediction_sha256',protocol)
            protocol_path.write_text(json.dumps(protocol),encoding='utf-8')
            row=run._slot32_reference(temp)
            self.assertEqual(row['prediction_sha256'],run.sha256(npz));self.assertEqual(row['labels'].shape,(320,32,32));self.assertTrue(np.all(row['labels']==0))
        finally:
            for item in (npz,protocol_path):
                if item.exists():item.unlink()
            temp.rmdir()

    def test_cli_score_terminal_json_serialization_path(self):
        output=io.StringIO()
        with patch_score({'status':'synthetic_complete'}),contextlib.redirect_stdout(output):
            self.assertEqual(run.main(['--stage','score','--output','unused.json']),0)
        self.assertEqual(json.loads(output.getvalue()),{'status':'synthetic_complete'})

    def test_full_score_validates_predictions_and_legacy_slot_before_opening_gt(self):
        import h5py
        temp=Path(tempfile.gettempdir())/('sw142_score_'+uuid.uuid4().hex[:10]);temp.mkdir()
        results=temp/'results';results.mkdir();slot_root=temp/'slot';slot_dir=slot_root/'seed1_epoch10';slot_dir.mkdir(parents=True)
        source_dir=temp/'source';source_dir.mkdir();dataset=temp/'masks.h5';baseline=run.HERE/'results_archive'/('slot_baseline_test_'+uuid.uuid4().hex[:10]+'.json');events=[]
        baseline.parent.mkdir(parents=True,exist_ok=True)
        ids=np.arange(1320,1640,dtype=np.int64);labels=np.zeros((320,32,32),dtype=np.int64);labels[:,8:24,8:16]=1;labels[:,8:24,16:24]=2
        try:
            training_rows={}
            for arm in run.ARMS:
                arm_dir=results/arm;arm_dir.mkdir();npz=arm_dir/'predictions.npz';protocol=arm_dir/'protocol.json';checkpoint=arm_dir/'checkpoint.fixture'
                with npz.open('xb') as stream:np.savez_compressed(stream,image_ids=ids,primary_labels=labels,qcc_labels=labels)
                checkpoint.write_bytes((arm+' checkpoint').encode());manifest_sha='manifest-'+arm
                training_rows[arm]={"manifest":{"source_initial_state_sha256":"same-source-state","warm_checkpoint_sha256":"same-warm-checkpoint","warm_optimizer_sha256":"same-warm-optimizer"},"manifest_sha256":manifest_sha,"checkpoint":checkpoint}
                row={"experiment":run.NAME,"status":"complete","seed":1,"arm":arm,"count":320,"prediction_shape":[320,32,32],"steps":1024,"settle":512,"ground_truth_used_for_prediction":False,"prediction_sha256":run.sha256(npz),"training_manifest_sha256":manifest_sha,"checkpoint_sha256":run.sha256(checkpoint),"implementation_fingerprint":run.fingerprint()}
                protocol.write_text(json.dumps(row),encoding='utf-8')

            source_npz=source_dir/'predictions.npz';source_protocol=source_dir/'protocol.json'
            with source_npz.open('xb') as stream:np.savez_compressed(stream,image_ids=ids,labels=labels)
            source_protocol.write_text('{}',encoding='utf-8')
            slot_npz=slot_dir/'predictions.npz';slot_protocol=slot_dir/'protocol.json';pixels=np.repeat(np.repeat(labels,4,axis=1),4,axis=2).astype(np.int16)
            with slot_npz.open('xb') as stream:np.savez_compressed(stream,image_ids=ids,labels=pixels)
            training={"seed":1,"unique_training_images":70000,"epochs":10,"batch_size":32,"steps_per_epoch":2188,"total_steps":21880,"iterations":3,"learning_rate":0.0004,"selected_images_sha256":contract32.SLOT_SELECTED_TRAINING_IDS_SHA256,"num_slots":11,"ground_truth_used_for_training":False,"training_id_segments_inclusive":[[0,999],[1640,70639]]}
            slot_row={"seed":1,"image_ids":[1320,1639],"count":320,"inference_seed":0,"batch_size":1,"iterations":3,"preprocessing":"full HDF5 RGB image, float32 /127.5 - 1; no crop","resolution":[128,128],"num_slots":11,"checkpoint_sha256":contract32.SLOT_CHECKPOINT_SHA256[1],"model_sha256":contract32.SLOT_MODEL_SHA256,"ground_truth_used_for_prediction":False,"background_rule":"most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties","label_rule":"slot ID + 1; selected background slot set to 0","training_protocol":training}
            slot_protocol.write_text(json.dumps(slot_row),encoding='utf-8')
            baseline.write_text(json.dumps({"scores":{"slot32":{m:{"three_seed_mean":.2} for m in contract32.METRICS}}}),encoding='utf-8')
            with h5py.File(dataset,'w') as h:
                mask=h.create_dataset('mask',shape=(1640,128,128),dtype='u1',fillvalue=0)
                mask[1320:1640]=np.repeat(np.repeat(labels,4,axis=1),4,axis=2).astype(np.uint8)

            original_file=h5py.File
            def tracked_file(*args,**kwargs):events.append('gt_open');return original_file(*args,**kwargs)
            def validate_source(*args,**kwargs):
                events.append('source_validate')
                return {"ids":ids,"labels":labels,"archive_path":source_npz,"protocol_path":source_protocol,"prediction_sha256":run.sha256(source_npz),"protocol_sha256":run.sha256(source_protocol)}
            output=temp/'score.json'
            with patch.object(run,'RESULTS',results),patch.object(run,'validate_training',side_effect=lambda arm:training_rows[arm]), \
                 patch.object(run.sw137,'validate_prediction',side_effect=validate_source),patch.object(run,'SLOT_BASELINE_PATH',baseline), \
                 patch.object(run,'SLOT_BASELINE_SHA256',run.sha256(baseline)),patch('h5py.File',side_effect=tracked_file):
                first_npz=results/run.ARMS[0]/'predictions.npz';original=first_npz.read_bytes();first_npz.write_bytes(original+b'changed')
                with self.assertRaises(ValueError):run.score(output=output,dataset=dataset,source_root=source_dir,slot_root=slot_root)
                self.assertNotIn('gt_open',events);first_npz.write_bytes(original)
                row=run.score(output=output,dataset=dataset,source_root=source_dir,slot_root=slot_root)
            self.assertEqual(row['status'],'scored_complete');self.assertIn('may_justify_seed_expansion',row['promotion_gate'])
            self.assertFalse(row['promotion_gate']['may_justify_seed_expansion'])
            self.assertEqual(row['scores']['core_trainable']['primary']['fg_ari']['valid_count'],320)
            self.assertEqual(events,['source_validate','gt_open'])
        finally:
            for item in temp.rglob('*'):
                if item.is_file():item.unlink()
            for directory in sorted((p for p in temp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):directory.rmdir()
            temp.rmdir()
            if baseline.exists():baseline.unlink()

class patch_score:
    def __init__(self,value):self.value=value
    def __enter__(self):self.old=run.score;run.score=lambda **kwargs:self.value
    def __exit__(self,*args):run.score=self.old

if __name__=='__main__':unittest.main()
