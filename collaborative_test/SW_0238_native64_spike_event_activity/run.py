"""Two actual TRAIN traces per fixed accepted232 checkpoint; no diagnostic labels or updates."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0237_native64_converged_full320 import run as reference
from .spike_gate_activity import summarize

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive';SEEDS=(0,1,2)
sha=reference.sha;write_json=reference.write_json;require_seed=reference.require_seed

def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'

def validate_ids(ids):
    if (len(ids)!=2 or len(set(ids))!=2 or any(type(i) is not int or not 0<=i<70640 or 1000<=i<1640 for i in ids)):
        raise ValueError('exactly two unique registered TRAIN IDs outside evaluation required')

def fingerprint():
    names=('DESIGN_KO.md','run.py','queue.py','spike_gate_activity.py','test_spike_gate_activity.py','test_pipeline.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},accepted64_selection=reference.scope.selection_fingerprint(),
                model=reference.scope.model.fingerprint(),shared_reference_run_sha256=sha(Path(reference.__file__).read_bytes()),shared_reference_scope_sha256=sha(Path(reference.scope.__file__).read_bytes()),seeds=list(SEEDS),images_per_seed=2,total_steps=1024,settle_steps=512,
                additional_optimizer_updates=0,diagnostic_ground_truth_input=False,diagnostic_not_performance_or_causal_ablation=True)

def prepare_cpu(seed):
    require_seed(seed)
    if torch.cuda.is_initialized():raise ValueError('CPU admission before CUDA required')
    found,_,_,parent=reference.source.prepare_cpu(seed)
    report,ref,_,_=reference.scope.restore(found,seed)
    ids=list(map(int,report['binding']['TRAIN_ids'][:2]));validate_ids(ids)
    dataset=Path(parent['dataset']['path']);stat=dataset.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(parent['dataset']['size'],parent['dataset']['mtime_ns']):raise ValueError('dataset changed')
    import h5py
    with h5py.File(dataset,'r') as stream:rgb=reference.scope.canonical_rgb(np.stack([stream['image'][i] for i in ids]))
    if rgb.shape!=(2,128,128,3) or rgb.dtype!=np.uint8:raise ValueError('native canonical TRAIN input required')
    binding=dict(fingerprint=fingerprint(),seed=seed,candidate=ref,dataset=parent['dataset'],TRAIN_ids=ids,RGB_sha256=sha(rgb.tobytes()),diagnostic_GT_used=False)
    return dict(found=found,rgb=rgb,binding=binding)

def admit(seed):
    p=prepare_cpu(seed)
    result=dict(binding=p['binding'],cuda_initialized=False,optimizer_updates=0,actual_CPU_restore_passed=True)
    write_json(archive_for(seed)/'admission.json',result);return result

def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed)
    if json.loads((a/'admission.json').read_bytes())['binding']!=p['binding']:raise ValueError('source changed after admission')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    found=p['found'];reference.source.source.source.model64.move_to_device(found,device)
    expected=reference.scope.model.resume_policy.compatible_runtime(reference.scope.model.checkpoint.runtime_signature(device))
    if expected!=p['binding']['candidate']['expected_runtime']:raise ValueError('runtime differs from accepted checkpoint')
    rows=[]
    for i,image in enumerate(p['rgb']):
        path=a/'rows'/f'{i:04d}.json'
        if not path.exists():
            torch.manual_seed(238000+seed*10+i);torch.cuda.manual_seed(238000+seed*10+i)
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=reference.source.source.full.screen.native._native_forward(found,torch.from_numpy(image[None]).to(device),live_tail_steps=0)
                s=trace['component_spikes'][...,512:].detach().cpu().numpy()
                g=trace['component_gates'][...,512:].detach().cpu().numpy()
                result=summarize(s,g)
            write_json(path,dict(binding=p['binding'],image_id=p['binding']['TRAIN_ids'][i],index=i,
                inference_seed=238000+seed*10+i,activity=result,diagnostic_GT_used=False))
            del gamma,trace,adj,z,q,rho,s,g
        row=json.loads(path.read_bytes())
        if row['binding']!=p['binding'] or row['index']!=i or row['image_id']!=p['binding']['TRAIN_ids'][i] or row['diagnostic_GT_used'] is not False:
            raise ValueError('existing diagnostic row changed')
        rows.append(sha(path.read_bytes()))
    if reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['binding']['candidate']['model_sha256']:
        raise ValueError('diagnostic inference changed actual model')
    protocol=dict(binding=p['binding'],row_sha256=rows,additional_optimizer_updates=0,diagnostic_GT_used=False)
    if (a/'protocol.json').exists():
        if json.loads((a/'protocol.json').read_bytes())!=protocol:raise ValueError('completed diagnostic protocol changed')
    else:write_json(a/'protocol.json',protocol)

def validated_seed(seed):
    a=archive_for(seed);ad=json.loads((a/'admission.json').read_bytes());protocol=json.loads((a/'protocol.json').read_bytes())
    if protocol['binding']!=ad['binding'] or ad['binding']['fingerprint']!=fingerprint() or protocol['additional_optimizer_updates']!=0 or protocol['diagnostic_GT_used'] is not False:
        raise ValueError('diagnostic protocol changed')
    if sha((reference.scope.model.archive_for(seed,64)/'encoder_work/boundary.ptx').read_bytes())!=ad['binding']['candidate']['checkpoint_sha256']:
        raise ValueError('fixed checkpoint changed')
    rows=[]
    for i,digest in enumerate(protocol['row_sha256']):
        raw=(a/'rows'/f'{i:04d}.json').read_bytes();row=json.loads(raw)
        if sha(raw)!=digest or row['binding']!=ad['binding'] or row['diagnostic_GT_used'] is not False:raise ValueError('diagnostic row changed')
        json.dumps(row,allow_nan=False);rows.append(row)
    if len(rows)!=2:raise ValueError('two actual TRAIN traces required')
    return rows

def score():
    rows=[r for s in SEEDS for r in validated_seed(s)]
    fractions=[r['activity']['fraction_of_positive_gate_samples_suppressed_by_event'] for r in rows]
    cases=[dict(seed=r['binding']['seed'],image_id=r['image_id'],index=r['index'],
                full_model_sha256=r['binding']['candidate']['model_sha256'],activity=r['activity']) for r in rows]
    result=dict(status='complete_actual_native64_TRAIN_event_activity_diagnostic',fingerprint=fingerprint(),seed_scope=list(SEEDS),
        actual_TRAIN_forward_count=6,rows=cases,additional_optimizer_updates=0,diagnostic_GT_used=False,
        all_observed_spikes_bitwise_equal_to_gates=all(r['activity']['gate_copy_bitwise_equal'] for r in rows),
        max_observed_event_suppression_fraction=max([v for v in fractions if v is not None],default=None),
        performance_or_causal_improvement_claim=False,full320_incumbent_unchanged=True)
    write_json(ARCHIVE/'score_v1/summary.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--admit',action='store_true');p.add_argument('--score',action='store_true');args=p.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    else:p.error('choose CPU admission or frozen diagnostic collection')
