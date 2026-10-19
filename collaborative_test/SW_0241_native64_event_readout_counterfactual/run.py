"""Frozen six-case event readout intervention; no labels, training or model mutation."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0238_native64_spike_event_activity import run as base
from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import native_component_rho_q
from .readout import predict_pair,compare_labels

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive';SEEDS=(0,1,2)
sha=base.sha;write_json=base.write_json;require_seed=base.require_seed
reference=base.reference


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('DESIGN_KO.md','run.py','queue.py','readout.py','test_readout.py')},
        source238_fingerprint=base.fingerprint(),seeds=list(SEEDS),cases_per_seed=2,
        intervention='downstream_observed_gate_only_spikes_with_recomputed_Q',
        changes_upstream_dynamics=False,GT_used=False,optimizer_updates=0)


def prepare_cpu(seed):
    p=base.prepare_cpu(seed)
    p['binding']=dict(p['binding'],fingerprint=fingerprint(),source238_fingerprint=base.fingerprint())
    return p


def admit(seed):
    p=prepare_cpu(seed)
    proof=dict(binding=p['binding'],cuda_initialized=False,optimizer_updates=0,actual_CPU_restore_passed=True)
    write_json(archive_for(seed)/'admission.json',proof);return proof


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed)
    if json.loads((a/'admission.json').read_bytes())['binding']!=p['binding']:raise ValueError('CPU admission changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    found=p['found'];reference.source.source.source.model64.move_to_device(found,device)
    runtime=reference.scope.model.resume_policy.compatible_runtime(reference.scope.model.checkpoint.runtime_signature(device))
    if runtime!=p['binding']['candidate']['expected_runtime']:raise ValueError('accepted runtime differs')
    hashes=[]
    for i,image in enumerate(p['rgb']):
        folder=a/'rows'/f'{i:04d}'
        if not folder.exists():
            rng=238000+seed*10+i
            torch.manual_seed(rng);torch.cuda.manual_seed(rng)
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=reference.source.source.full.screen.native._native_forward(found,torch.from_numpy(image[None]).to(device),live_tail_steps=0)
                actual,bypass,comparison=predict_pair(found,trace,adj,q,reference.source.source.full.screen.prediction_from_trace,native_component_rho_q)
                stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
                np.savez_compressed(stage/'raw_trace.npz',component_spikes=trace['component_spikes'].cpu().numpy(),
                    component_gates=trace['component_gates'].cpu().numpy(),adjacency=adj.cpu().numpy())
                np.savez_compressed(stage/'masks.npz',actual=actual,gate_only=bypass)
            write_json(stage/'row.json',dict(binding=p['binding'],index=i,image_id=p['binding']['TRAIN_ids'][i],
                inference_seed=rng,comparison=comparison,raw_trace_sha256=sha((stage/'raw_trace.npz').read_bytes()),
                masks_sha256=sha((stage/'masks.npz').read_bytes()),GT_used=False,optimizer_updates=0))
            stage.rename(folder);del gamma,trace,adj,z,q,rho,actual,bypass
        validate_row(folder,p['binding'],i)
        hashes.append(sha((folder/'row.json').read_bytes()))
    if reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['binding']['candidate']['model_sha256']:
        raise ValueError('frozen forward changed accepted model')
    protocol=dict(binding=p['binding'],row_sha256=hashes,optimizer_updates=0,GT_used=False,changes_upstream_dynamics=False)
    if (a/'protocol.json').exists():
        if json.loads((a/'protocol.json').read_bytes())!=protocol:raise ValueError('existing protocol differs')
    else:write_json(a/'protocol.json',protocol)


def validate_row(folder,binding,index):
    row=json.loads((folder/'row.json').read_bytes())
    if row['binding']!=binding or row['index']!=index or row['image_id']!=binding['TRAIN_ids'][index] or row['inference_seed']!=238000+binding['seed']*10+index or row['GT_used'] or row['optimizer_updates']:
        raise ValueError('frozen row identity changed')
    for name,key in (('raw_trace.npz','raw_trace_sha256'),('masks.npz','masks_sha256')):
        if sha((folder/name).read_bytes())!=row[key]:raise ValueError('actual archived tensors changed')
    with np.load(folder/'masks.npz',allow_pickle=False) as data:
        if data['actual'].shape!=(64,64) or compare_labels(data['actual'],data['gate_only'])!=row['comparison']:
            raise ValueError('paired masks do not reproduce comparison')
    return row


def validated_seed(seed):
    a=archive_for(seed);ad=json.loads((a/'admission.json').read_bytes());protocol=json.loads((a/'protocol.json').read_bytes())
    binding=ad['binding']
    if protocol['binding']!=binding or binding['fingerprint']!=fingerprint() or protocol['GT_used'] or protocol['optimizer_updates'] or protocol['changes_upstream_dynamics']:
        raise ValueError('registered protocol changed')
    if sha((reference.scope.model.archive_for(seed,64)/'encoder_work/boundary.ptx').read_bytes())!=binding['candidate']['checkpoint_sha256']:
        raise ValueError('accepted checkpoint changed')
    if len(protocol['row_sha256'])!=2:raise ValueError('two frozen cases required')
    rows=[]
    for i,digest in enumerate(protocol['row_sha256']):
        folder=a/'rows'/f'{i:04d}'
        if sha((folder/'row.json').read_bytes())!=digest:raise ValueError('row hash changed')
        rows.append(validate_row(folder,binding,i))
    return rows


def score():
    rows=[r for seed in SEEDS for r in validated_seed(seed)]
    result=dict(status='complete_six_case_event_readout_counterfactual',fingerprint=fingerprint(),
        cases=[dict(seed=r['binding']['seed'],case_index=r['index'],comparison=r['comparison']) for r in rows],
        identical_partition_case_count=sum(r['comparison']['partition_equal_up_to_foreground_ID_permutation'] for r in rows),
        total_cases=6,GT_used=False,optimizer_updates=0,changes_upstream_dynamics=False,
        full_membrane_causal_necessity_proven=False,performance_improvement_claim=False,full320_incumbent_unchanged=True)
    write_json(ARCHIVE/'score_v1/summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--admit',action='store_true');p.add_argument('--score',action='store_true');args=p.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    else:p.error('choose CPU admission or completed diagnostic score')
