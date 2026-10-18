"""Immutable, GT-last full320 confirmation of two actual native64 checkpoints."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from collaborative_test.SW_0193_native64_probe_screen import run as screen, score as metrics
from collaborative_test.SW_0204_native64_component_credit import train as candidate

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
IDS=np.arange(1320,1640,dtype=np.int64)
ARMS=('warm192','component204')
MODELS={'warm192':'2dd2b0d64602b85d4082bcaddcd21dfe8c60d34af1ea78a2ed29bf03c8ba4b6e',
        'component204':'f8d4ac3d9aa1ad0b6177abdd468ae0f5a713f06c9cac1be86635ce2237f36e88'}
sha=screen.sha
write_json=screen.write_json

def offsets(shard):
    if type(shard) is not int or shard not in range(4):raise ValueError('shard must be integer0..3')
    return tuple(range(shard,len(IDS),4))

def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('run.py','queue.py','test_run.py','DESIGN_KO.md')},
                parents=dict(screen=screen.fingerprint(),candidate=candidate.fingerprint()),
                image_ids=IDS.tolist(),models=MODELS,seed_scope=[0],inference_seed_base=193000,
                grid=[64,64],optimizer_updates=0,mask_supervised=True)

def unpack_checkpoint(raw):
    envelope=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    data=envelope['payload_bytes']
    if sha(data)!=envelope['payload_sha256']:raise ValueError('checkpoint payload hash mismatch')
    return torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)

def prepare_cpu():
    import h5py
    from collaborative_test.SW_0130_phase_state_integration import run as source
    from collaborative_test.SW_0179_qce_seed_replication import train as state_api
    if torch.cuda.is_initialized():raise RuntimeError('CPU preparation must precede CUDA')
    original=screen.prepare_inputs_cpu()
    states={'warm192':original['candidate_state']}
    report=json.loads((candidate.ARCHIVE/'training/report.json').read_bytes())
    raw=(candidate.ARCHIVE/'training_work/boundary.ptx').read_bytes()
    payload=unpack_checkpoint(raw)
    metadata=candidate.probe.checkpoint.inspect_committed_metadata(candidate.ARCHIVE/'training_work')
    if (report['status']!='complete_native64_finite_Adam' or payload['history']!=report['history']
        or payload['bindings']!=report['bindings'] or metadata['bindings']!=report['bindings']
        or report['bindings']['fingerprint']!=candidate.fingerprint()
        or payload['completed_updates']!=4):raise ValueError('completed204 checkpoint/report/source mismatch')
    states['component204']={'wrapped':payload['wrapped_state'],'encoder':payload['encoder_state']}
    for arm,state in states.items():
        if state_api._state_sha_from_payload(state)!=MODELS[arm]:raise ValueError('wrong model state')
        original['found'].wrapped.load_state_dict(state['wrapped'],strict=True)
        original['found'].encoder.load_state_dict(state['encoder'],strict=True)
    dataset=Path(original['bindings']['dataset']['path']);stat=dataset.stat()
    rgb=np.load(source.VAL_RGB,mmap_mode='r',allow_pickle=False).copy()
    with h5py.File(dataset,'r') as stream:pixels=np.asarray(stream['image'][IDS[0]:IDS[-1]+1])
    if pixels.dtype!=np.uint8:pixels=np.rint(np.clip(pixels.astype(np.float32),0,1)*255).astype(np.uint8)
    if rgb.shape!=(320,128,128,3) or rgb.dtype!=np.uint8 or not np.array_equal(rgb,pixels):raise ValueError('full320 canonical RGB/cache mismatch')
    binding=dict(fingerprint=fingerprint(),models=MODELS,RGB_sha256=sha(rgb.tobytes()),
        RGB_row_sha256=[sha(row.tobytes()) for row in rgb],
        dataset=dict(path=str(dataset.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns),
        warm_source_binding=original['bindings'],candidate_boundary_sha256=sha(raw),
        candidate_report_sha256=sha((candidate.ARCHIVE/'training/report.json').read_bytes()))
    return original['found'],states,rgb,binding

def row_binding(binding,arm,offset):
    if arm not in ARMS or type(offset) is not int or offset not in range(320):raise ValueError('invalid prediction identity')
    return dict(arm=arm,image_id=int(IDS[offset]),inference_seed=193000+offset,
        model_sha256=MODELS[arm],RGB_sha256=binding['RGB_row_sha256'][offset],
        source_bindings_sha256=sha(json.dumps(binding,sort_keys=True).encode()))

def validate_overlap(arm,offset,labels):
    old=None
    if arm=='warm192' and offset<32:old=screen.ARCHIVE/'candidate/rows'/f'{offset:04d}'
    elif arm=='component204' and offset in range(0,32,2):old=candidate.ARCHIVE/'candidate/rows'/f'{offset:04d}'
    if old is not None:
        record=json.loads((old/'row.json').read_bytes())
        previous=screen.validate_saved_row(old,record['binding'])
        if record['binding']['model_sha256']!=MODELS[arm] or record['binding']['inference_seed']!=193000+offset or not np.array_equal(previous,labels):raise ValueError('new prediction does not exactly reproduce frozen mini')

def predict(shard):
    found,states,rgb,binding=prepare_cpu()
    admission=json.loads((ARCHIVE/'registration_v1/admission.json').read_bytes())
    if binding!=admission['binding'] or admission['cuda_initialized']:raise ValueError('admission changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    found=screen.probe.model64.move_to_device(found,device)
    for arm in ARMS:
        found.wrapped.load_state_dict(states[arm]['wrapped'],strict=True);found.encoder.load_state_dict(states[arm]['encoder'],strict=True)
        if screen.native._state_sha(found.wrapped,found.encoder)!=MODELS[arm]:raise ValueError('GPU model restore mismatch')
        found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
        rows=[]
        for offset in offsets(shard):
            expected=row_binding(binding,arm,offset);folder=ARCHIVE/arm/'rows'/f'{offset:04d}'
            if folder.exists():labels=screen.validate_saved_row(folder,expected)
            else:
                torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset);started=time.perf_counter()
                with torch.no_grad():
                    image=torch.from_numpy(rgb[offset:offset+1]).to(device)
                    gamma,trace,adj,z,q,rho=screen.native._native_forward(found,image,live_tail_steps=0)
                    labels=screen.prediction_from_trace(found,trace,adj,q)
                validate_overlap(arm,offset,labels)
                stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
                with (stage/'labels.npz').open('xb') as stream:
                    np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
                write_json(stage/'row.json',dict(binding=expected,prediction_sha256=sha((stage/'labels.npz').read_bytes()),seconds=time.perf_counter()-started))
                stage.rename(folder);del image,gamma,trace,adj,z,q,rho
            validate_overlap(arm,offset,labels);rows.append(labels)
        if screen.native._state_sha(found.wrapped,found.encoder)!=MODELS[arm]:raise RuntimeError('inference mutated model')
        protocol=dict(binding=binding,arm=arm,shard=shard,image_ids=IDS[list(offsets(shard))].tolist(),
            labels_sha256=sha(np.stack(rows).tobytes()),validation_GT_read=False,optimizer_updates=0)
        path=ARCHIVE/arm/f'shard{shard}.json'
        if path.exists():
            if json.loads(path.read_bytes())!=protocol:raise ValueError('completed shard changed')
        else:write_json(path,protocol)

def validated_all():
    binding=json.loads((ARCHIVE/'registration_v1/admission.json').read_bytes())['binding']
    if binding['fingerprint']!=fingerprint():raise ValueError('source changed')
    result={}
    for arm in ARMS:
        arrays={}
        for shard in range(4):
            protocol=json.loads((ARCHIVE/arm/f'shard{shard}.json').read_bytes())
            if (protocol['binding']!=binding or protocol['arm']!=arm or protocol['shard']!=shard
                or protocol['image_ids']!=IDS[list(offsets(shard))].tolist()
                or protocol['validation_GT_read'] is not False or protocol['optimizer_updates']!=0):raise ValueError('invalid frozen shard')
            rows=[]
            for offset in offsets(shard):
                row=screen.validate_saved_row(ARCHIVE/arm/'rows'/f'{offset:04d}',row_binding(binding,arm,offset))
                validate_overlap(arm,offset,row);arrays[offset]=row;rows.append(row)
            if sha(np.stack(rows).tobytes())!=protocol['labels_sha256']:raise ValueError('shard array hash mismatch')
        result[arm]=np.stack([arrays[i] for i in range(320)])
    return result,binding

def read_gt(binding):
    import h5py
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    dataset=Path(binding['dataset']['path']);stat=dataset.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(binding['dataset']['size'],binding['dataset']['mtime_ns']):raise ValueError('dataset changed')
    with h5py.File(dataset,'r') as stream:pixels=np.asarray(stream['mask'][IDS[0]:IDS[-1]+1,...,0])
    gt=clevr_mask_patch(torch.from_numpy(pixels.copy()),2)['patch_labels'].cpu().numpy().astype(np.int64)
    if gt.shape!=(320,64,64) or not np.array_equal(gt,metrics.independent.modal_labels(pixels,patch=2)):raise ValueError('independent full320 modal GT mismatch')
    return gt

def summarize(vectors):
    keys=metrics.independent.METRICS
    delta=vectors['component204']-vectors['warm192']
    if delta.shape!=(320,3) or not all(v.shape==(320,3) and np.isfinite(v).all() for v in vectors.values()):raise ValueError('finite fixed320 metrics required')
    draws=np.random.RandomState(208).randint(0,320,(10000,320));ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    return dict(scores={arm:dict(zip(keys,v.mean(0).tolist())) for arm,v in vectors.items()},
        paired_component204_minus_warm192={key:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,key in enumerate(keys)},
        preferred_for_next_training=max(ARMS,key=lambda arm:float(vectors[arm][:,0].mean())),
        full320_three_seed_goal_verified=False)

def score():
    predictions,binding=validated_all()
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    slots=metrics.slot.prepare_all_slot_predictions(SLOT_ROOT)
    if any(not np.array_equal(row['ids'],IDS) for row in slots.values()):raise ValueError('Slot full320 IDs mismatch')
    gt=read_gt(binding);vectors={};errors=[]
    for arm,pred in dict(predictions,**{'slot_seed'+str(s):row['labels64'] for s,row in slots.items()}).items():
        vectors[arm],error=metrics.measured_vectors(pred,gt);errors.append(error)
    vectors['slot_three_seed_mean']=np.mean([vectors['slot_seed'+str(s)] for s in range(3)],axis=0)
    summary=dict(summarize(vectors),status='complete_native64_full320_seed0_confirmation',grid=[64,64],image_count=320,
        seed_scope=[0],mask_supervised=True,new_training_updates=0,new_70000_training=False,
        validation_selected_not_untouched_test=True,ground_truth_opened_after_all_predictions_validated=True,
        independent_maximum_metric_abs_error=max(errors),binding=binding,
        slot_prediction_sha256={str(s):row['prediction_sha256'] for s,row in slots.items()})
    write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    write_json(ARCHIVE/'score_v1/summary.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=range(4));parser.add_argument('--cpu-output',type=Path);parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if args.cpu_output:
        _,_,_,binding=prepare_cpu();write_json(args.cpu_output,dict(binding=binding,cuda_initialized=torch.cuda.is_initialized()))
    elif args.score:print(json.dumps(score()))
    elif args.shard is not None:predict(args.shard)
    else:parser.error('choose CPU admission, shard inference or scoring')

if __name__=='__main__':main()
