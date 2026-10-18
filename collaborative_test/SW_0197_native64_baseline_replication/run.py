"""Independent seed1/2 native64 baseline jobs; reuse frozen seed0, no updates."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from collaborative_test.SW_0192_native64_incumbent_mini import run as train
from collaborative_test.SW_0193_native64_probe_screen import run as screen,score as metrics

HERE=Path(__file__).resolve().parent
ROOT_ARCHIVE=HERE/'results_archive'
RUN_SEED=None
ARCHIVE=ROOT_ARCHIVE
ARMS=('prediction',)
REFERENCE_SHA='6e8e036718c520eec04226fc75e6356644de73bcea74e338bb7c174be9cbd4eb'


def archive_for(seed):
    if type(seed) is not int or seed not in (1,2):raise ValueError('independent seed1 or2 required')
    return ROOT_ARCHIVE/f'seed{seed}'


def configure(seed):
    global RUN_SEED,ARCHIVE
    ARCHIVE=archive_for(seed);RUN_SEED=seed


def fingerprint(seed=None):
    seed=RUN_SEED if seed is None else seed;archive_for(seed)
    return dict(files={name:train.sha((HERE/name).read_bytes()) for name in
        ('run.py','queue.py','test_run.py','DESIGN_KO.md')},seed=seed,parent=train.fingerprint(),
        readout=screen.fingerprint(),image_ids=screen.IDS.tolist(),optimizer_updates=0,reference_summary_sha256=REFERENCE_SHA)


def prepare_cpu():
    if torch.cuda.is_initialized():raise RuntimeError('CPU admission before CUDA required')
    inputs=train.incumbent.prepare_training_inputs_cpu()
    found,_optimizer,_families,proof=train.model64.build_from_incumbent(RUN_SEED,inputs)
    if train.native._state_sha(found.wrapped,found.encoder)!=proof['native64_initial_model_state_sha256']:
        raise ValueError('actual baseline model differs from native64 factory proof')
    _controls,ref=metrics.validated_pair()
    if train.sha((screen.ARCHIVE/'score_v1/summary.json').read_bytes())!=REFERENCE_SHA:
        raise ValueError('verified source development reference changed')
    from collaborative_test.SW_0130_phase_state_integration import run as data
    rgb=np.load(data.VAL_RGB,mmap_mode='r',allow_pickle=False)[:32].copy()
    if rgb.dtype!=np.uint8 or rgb.shape!=(32,128,128,3) or train.sha(rgb.tobytes())!=ref['development_RGB_sha256']:
        raise ValueError('fixed canonical development RGB mismatch')
    binding=dict(fingerprint=fingerprint(),factory_proof=proof,reference_binding=ref,
        model_sha256=proof['native64_initial_model_state_sha256'],seed=RUN_SEED,
        known_TRAIN_GT_available_in_factory=True,validation_GT_used_for_prediction=False,optimizer_updates=0)
    return found,None,rgb,binding


def row_binding(binding,offset):
    return dict(seed=binding['seed'],image_id=int(screen.IDS[offset]),inference_seed=193000+offset,
        model_sha256=binding['model_sha256'],RGB_sha256=binding['reference_binding']['development_RGB_row_sha256'][offset],
        bindings_sha256=train.sha(json.dumps(binding,sort_keys=True).encode()))


def predict(device='cuda:0',expected_binding=None):
    found,_unused,rgb,binding=prepare_cpu()
    if expected_binding is not None and binding!=expected_binding:raise ValueError('baseline CPU admission changed')
    device=torch.device(device);torch.cuda.set_device(device);train.model64.move_to_device(found,device)
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    rows=[]
    for offset in range(32):
        expected=row_binding(binding,offset);folder=ARCHIVE/'prediction/rows'/f'{offset:04d}'
        if folder.exists():labels=screen.validate_saved_row(folder,expected)
        else:
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset);started=time.perf_counter()
            with torch.no_grad():
                image=torch.from_numpy(rgb[offset:offset+1]).to(device)
                gamma,trace,adj,z,q,rho=train.native._native_forward(found,image,live_tail_steps=0)
                labels=screen.prediction_from_trace(found,trace,adj,q)
            stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
            with (stage/'labels.npz').open('xb') as stream:
                np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
            train.write_json(stage/'row.json',dict(binding=expected,prediction_sha256=train.sha((stage/'labels.npz').read_bytes()),seconds=time.perf_counter()-started))
            stage.rename(folder);del image,gamma,trace,adj,z,q,rho
        rows.append(labels)
    if train.native._state_sha(found.wrapped,found.encoder)!=binding['model_sha256']:raise RuntimeError('inference mutated baseline')
    protocol=dict(binding=binding,labels_sha256=train.sha(np.stack(rows).tobytes()))
    path=ARCHIVE/'prediction/protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen baseline protocol changed')
    else:train.write_json(path,protocol)


def validated_pair(seed=None):
    seed=RUN_SEED if seed is None else seed;archive=archive_for(seed)
    protocol=json.loads((archive/'prediction/protocol.json').read_bytes());binding=protocol['binding']
    if (binding['fingerprint']!=fingerprint(seed) or binding['seed']!=seed
            or binding['validation_GT_used_for_prediction'] is not False or binding['optimizer_updates']!=0):
        raise ValueError('invalid frozen seed baseline')
    labels=np.stack([screen.validate_saved_row(archive/'prediction/rows'/f'{i:04d}',row_binding(binding,i)) for i in range(32)])
    if train.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('baseline array changed')
    return labels,binding


def maybe_aggregate():
    if not all((archive_for(s)/'prediction/protocol.json').exists() for s in (1,2)):return
    import fcntl
    with (ROOT_ARCHIVE/'aggregate.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        pairs={s:validated_pair(s) for s in (1,2)}
        controls,ref=metrics.validated_pair()
        if any(binding['reference_binding']!=ref for _labels,binding in pairs.values()):raise ValueError('baseline seed cohorts differ')
        prior_path=screen.ARCHIVE/'score_v1/summary.json'
        if train.sha(prior_path.read_bytes())!=REFERENCE_SHA:raise ValueError('Slot development reference changed')
        prior=json.loads(prior_path.read_bytes())
        if prior['bindings']!=ref:raise ValueError('Slot reference uses different development cohort')
        gt=metrics.read_ground_truth(ref)
        vectors={'seed0':metrics.measured_vectors(controls['baseline'],gt)[0],
            **{'seed'+str(s):metrics.measured_vectors(v[0],gt)[0] for s,v in pairs.items()}}
        mean=np.mean(list(vectors.values()),axis=0)
        summary=dict(status='native64_baseline_three_seed_development_complete',grid=[64,64],images=32,
            seed_scope=[0,1,2],scores={name:dict(zip(metrics.independent.METRICS,values.mean(0).tolist())) for name,values in vectors.items()},
            three_seed_mean=dict(zip(metrics.independent.METRICS,mean.mean(0).tolist())),optimizer_updates=0,
            comparable_Slot_three_seed_mean=prior['scores']['slot_three_seed_mean'],
            reference_summary_sha256=REFERENCE_SHA,
            full320_three_seed_goal_verified=False,scope='fixed32 development only; native32 incumbents mapped64 without extra adaptation',
            seed0_protocol_sha256=train.sha((screen.ARCHIVE/'baseline/protocol.json').read_bytes()),
            seed12_protocol_sha256={str(s):train.sha((archive_for(s)/'prediction/protocol.json').read_bytes()) for s in (1,2)})
        for name,value in [('evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()})),('summary.json',summary)]:
            path=ROOT_ARCHIVE/'aggregate_v1'/name
            if path.exists():
                if json.loads(path.read_bytes())!=value:raise ValueError('frozen aggregate changed')
            else:train.write_json(path,value)


def score():
    labels,binding=validated_pair()
    _controls,ref=metrics.validated_pair()
    if ref!=binding['reference_binding']:raise ValueError('baseline cohort changed')
    gt=metrics.read_ground_truth(ref);vectors,error=metrics.measured_vectors(labels,gt)
    summary=dict(status='completed_native64_read_only_baseline',seed=RUN_SEED,images=32,grid=[64,64],
        scores=dict(zip(metrics.independent.METRICS,vectors.mean(0).tolist())),binding=binding,
        independent_metric_max_abs_error=error,optimizer_updates=0,new_training=False,
        full320_three_seed_goal_verified=False,raw_imagewise_score_public=False)
    train.write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors=vectors.tolist()))
    train.write_json(ARCHIVE/'score_v1/summary.json',summary)
    maybe_aggregate();return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(1,2),required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--cpu-output',type=Path)
    args=parser.parse_args();configure(args.seed)
    if args.worker:predict()
    elif args.cpu_output:
        _found,_state,_rgb,binding=prepare_cpu();train.write_json(args.cpu_output,dict(binding=binding,cuda_initialized=torch.cuda.is_initialized()))
    else:parser.error('choose worker or CPU output')


if __name__=='__main__':main()
