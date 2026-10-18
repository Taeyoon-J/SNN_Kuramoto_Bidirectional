"""Full320 reference for independent native64 TRAIN8 adapted seeds1/2."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from collaborative_test.SW_0208_native64_full320_confirmation import run as full
from collaborative_test.SW_0205_native64_weak_seed_adaptation import run as source

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
IDS=full.IDS
SEEDS=(1,2)
MODELS={'1':'e1e16281612b2171b6822e0ca70dee051f095ad7ebc9239a9675b8e8d9b391e0',
        '2':'dff381ebcd5280a2a7bf7b99ef8ee3b5af30e91f3384962ef47c6d89b7da53d0'}
sha=full.sha
write_json=full.write_json

def require_seed(seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('independent seed1/2 required')
    return seed

def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('run.py','queue.py','test_run.py','DESIGN_KO.md')},
        source_files={n:sha((source.HERE/n).read_bytes()) for n in ('run.py','evaluate.py','queue.py','test_train.py','DESIGN_KO.md')},
        seed0_full_reference=full.fingerprint(),models=MODELS,seeds=[1,2],image_ids=IDS.tolist(),
        inference_seed_base=193000,grid=[64,64],new_training_updates=0)

def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'

def prepare_cpu(seed):
    require_seed(seed);source.configure(seed)
    prepared=source.prepare_inputs_cpu();found=prepared['found']
    path=source.ARCHIVE/'training_work/boundary.ptx';raw=path.read_bytes();payload=full.unpack_checkpoint(raw)
    report_path=source.ARCHIVE/'training/report.json';report=source.validate_report(json.loads(report_path.read_bytes()),prepared['bindings'])
    if report['status']!='complete_native64_training_admission' or payload['bindings']!=prepared['bindings'] or payload['history']!=report['history'] or payload['completed_updates']!=2:raise ValueError('completed TRAIN8 source mismatch')
    if report['final_model_state_sha256']!=MODELS[str(seed)]:raise ValueError('wrong independent source')
    found.wrapped.load_state_dict(payload['wrapped_state'],strict=True);found.encoder.load_state_dict(payload['encoder_state'],strict=True)
    if source.native._state_sha(found.wrapped,found.encoder)!=MODELS[str(seed)]:raise ValueError('actual source weights differ')
    mini,mini_protocol=source.validated_pair()
    if mini_protocol['model_sha256']!=MODELS[str(seed)] or mini.shape!=(16,64,64):raise ValueError('wrong mini reference')
    from collaborative_test.SW_0130_phase_state_integration import run as data
    import h5py
    rgb=np.load(data.VAL_RGB,mmap_mode='r',allow_pickle=False).copy()
    dataset=Path(prepared['bindings']['reference_binding']['dataset']['path']);stat=dataset.stat()
    with h5py.File(dataset,'r') as stream:pixels=np.asarray(stream['image'][IDS[0]:IDS[-1]+1])
    if pixels.dtype!=np.uint8:pixels=np.rint(np.clip(pixels.astype(np.float32),0,1)*255).astype(np.uint8)
    if rgb.shape!=(320,128,128,3) or rgb.dtype!=np.uint8 or not np.array_equal(rgb,pixels):raise ValueError('full320 RGB/canonical mismatch')
    _,original=full.validated_all()
    if original['RGB_sha256']!=sha(rgb.tobytes()):raise ValueError('seed0 uses different full images')
    binding=dict(fingerprint=fingerprint(),seed=seed,model_sha256=MODELS[str(seed)],RGB_sha256=sha(rgb.tobytes()),
        RGB_row_sha256=[sha(row.tobytes()) for row in rgb],dataset=dict(path=str(dataset.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns),
        training_bindings=prepared['bindings'],source_boundary_sha256=sha(raw),source_report_sha256=sha(report_path.read_bytes()),
        mini_protocol_sha256=sha((source.ARCHIVE/'candidate/protocol.json').read_bytes()),mini_labels_sha256=sha(mini.tobytes()),
        seed0_summary_sha256=sha((full.ARCHIVE/'score_v1/summary.json').read_bytes()))
    return found,rgb,binding,mini

def row_binding(binding,offset):
    if type(offset) is not int or offset not in range(320):raise ValueError('invalid fixed320 offset')
    return dict(seed=binding['seed'],image_id=int(IDS[offset]),inference_seed=193000+offset,
        model_sha256=MODELS[str(binding['seed'])],RGB_sha256=binding['RGB_row_sha256'][offset],
        source_bindings_sha256=sha(json.dumps(binding,sort_keys=True).encode()))

def check_mini(offset,labels,mini):
    if offset in range(0,32,2) and not np.array_equal(labels,mini[offset//2]):raise ValueError('frozen205 mini mask was not exactly reproduced')

def predict(seed,shard):
    found,rgb,binding,mini=prepare_cpu(seed);archive=archive_for(seed)
    admission=json.loads((archive/'registration_v1/admission.json').read_bytes())
    if admission['binding']!=binding or admission['cuda_initialized']:raise ValueError('CPU admission changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    source.model64.move_to_device(found,device);found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    if source.native._state_sha(found.wrapped,found.encoder)!=MODELS[str(seed)]:raise ValueError('GPU restore changed model')
    rows=[]
    for offset in full.offsets(shard):
        expected=row_binding(binding,offset);folder=archive/'rows'/f'{offset:04d}'
        if folder.exists():labels=full.screen.validate_saved_row(folder,expected)
        else:
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset);started=time.perf_counter()
            with torch.no_grad():
                image=torch.from_numpy(rgb[offset:offset+1]).to(device)
                gamma,trace,adj,z,q,rho=source.native._native_forward(found,image,live_tail_steps=0)
                labels=full.screen.prediction_from_trace(found,trace,adj,q)
            check_mini(offset,labels,mini)
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
            with (stage/'labels.npz').open('xb') as stream:
                np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
            write_json(stage/'row.json',dict(binding=expected,prediction_sha256=sha((stage/'labels.npz').read_bytes()),seconds=time.perf_counter()-started))
            stage.rename(folder);del image,gamma,trace,adj,z,q,rho
        check_mini(offset,labels,mini);rows.append(labels)
    if source.native._state_sha(found.wrapped,found.encoder)!=MODELS[str(seed)]:raise ValueError('inference mutated source')
    protocol=dict(binding=binding,shard=shard,image_ids=IDS[list(full.offsets(shard))].tolist(),
        labels_sha256=sha(np.stack(rows).tobytes()),validation_GT_read=False,optimizer_updates=0)
    path=archive/f'shard{shard}.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen shard changed')
    else:write_json(path,protocol)

def validated_seed(seed):
    source.configure(seed);mini,mini_protocol=source.validated_pair()
    archive=archive_for(seed);binding=json.loads((archive/'registration_v1/admission.json').read_bytes())['binding']
    if binding['fingerprint']!=fingerprint() or binding['seed']!=seed or binding['model_sha256']!=MODELS[str(seed)] or binding['mini_labels_sha256']!=sha(mini.tobytes()) or binding['mini_protocol_sha256']!=sha((source.ARCHIVE/'candidate/protocol.json').read_bytes()):raise ValueError('source/admission changed')
    arrays={}
    for shard in range(4):
        protocol=json.loads((archive/f'shard{shard}.json').read_bytes())
        if protocol['binding']!=binding or protocol['shard']!=shard or protocol['image_ids']!=IDS[list(full.offsets(shard))].tolist() or protocol['validation_GT_read'] is not False or protocol['optimizer_updates']!=0:raise ValueError('invalid frozen prediction shard')
        rows=[]
        for offset in full.offsets(shard):
            labels=full.screen.validate_saved_row(archive/'rows'/f'{offset:04d}',row_binding(binding,offset));check_mini(offset,labels,mini);arrays[offset]=labels;rows.append(labels)
        if sha(np.stack(rows).tobytes())!=protocol['labels_sha256']:raise ValueError('shard arrays changed')
    return np.stack([arrays[i] for i in range(320)]),binding

def summarize(vectors):
    ours=np.stack([vectors['seed'+str(s)] for s in range(3)])
    slots=np.stack([vectors['slot_seed'+str(s)] for s in range(3)])
    if ours.shape!=(3,320,3) or slots.shape!=ours.shape or not np.isfinite(ours).all() or not np.isfinite(slots).all():raise ValueError('three independent full320 finite vectors required')
    means=dict(our_three_seed_mean=ours.mean(0),slot_three_seed_mean=slots.mean(0));delta=means['our_three_seed_mean']-means['slot_three_seed_mean']
    draws=np.random.RandomState(210).randint(0,320,(10000,320));ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    scores={key:dict(zip(full.metrics.independent.METRICS,value.mean(0).tolist())) for key,value in dict(vectors,**means).items()}
    strict=bool(np.all(ours.mean((0,1))>slots.mean((0,1))))
    return dict(scores=scores,paired_our_mean_minus_slot_mean={key:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,key in enumerate(full.metrics.independent.METRICS)},
        three_seed_strictly_exceeds_Slot_means=strict,native64_FG090_three_seed_goal_verified=bool(strict and ours[:,:,0].mean()>=.90))

def score():
    original,b0=full.validated_all();predictions={'seed0':original['warm192']};bindings={}
    for seed in SEEDS:
        predictions['seed'+str(seed)],bindings[str(seed)]=validated_seed(seed)
        if bindings[str(seed)]['RGB_sha256']!=b0['RGB_sha256'] or bindings[str(seed)]['dataset']!=b0['dataset'] or bindings[str(seed)]['seed0_summary_sha256']!=sha((full.ARCHIVE/'score_v1/summary.json').read_bytes()):raise ValueError('different fixed320 evaluation data')
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    slots=full.metrics.slot.prepare_all_slot_predictions(SLOT_ROOT)
    if any(not np.array_equal(row['ids'],IDS) for row in slots.values()):raise ValueError('different Slot IDs')
    gt=full.read_gt(b0);vectors={};errors=[]
    for key,pred in dict(predictions,**{'slot_seed'+str(s):row['labels64'] for s,row in slots.items()}).items():
        vectors[key],error=full.metrics.measured_vectors(pred,gt);errors.append(error)
    summary=dict(summarize(vectors),status='complete_native64_full320_three_seed_TRAIN8_reference',image_count=320,grid=[64,64],seed_scope=[0,1,2],
        mask_supervised=True,unsupervised_leaderboard_eligible=False,new_training_updates=0,new70000_training=False,
        native64_additional_unique_TRAIN_per_seed=8,GT_read_after_all_masks_validated=True,
        validation_selected_not_untouched_test=True,independent_maximum_metric_abs_error=max(errors),bindings=bindings,
        source_seed0_summary_sha256=sha((full.ARCHIVE/'score_v1/summary.json').read_bytes()),fingerprint=fingerprint())
    write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}));write_json(ARCHIVE/'score_v1/summary.json',summary);return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--shard',type=int,choices=range(4));parser.add_argument('--cpu-output',type=Path);parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.cpu_output:
        _,_,binding,_=prepare_cpu(args.seed);write_json(args.cpu_output,dict(binding=binding,cuda_initialized=torch.cuda.is_initialized()))
    elif args.seed is not None and args.shard is not None:predict(args.seed,args.shard)
    else:parser.error('choose CPU admission, prediction or score')

if __name__=='__main__':main()
