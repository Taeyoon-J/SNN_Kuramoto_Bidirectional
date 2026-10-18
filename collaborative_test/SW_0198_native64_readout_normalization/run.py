"""Factorial frozen TRAIN8/64 x reference1024/actual4096 mask comparison."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .readout import foreground_induced_merge
from collaborative_test.SW_0194_native64_train64 import train,evaluate
from collaborative_test.SW_0195_native64_generalization_gap import run as pair

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
OFFSETS=tuple(range(0,32,4))
ARMS=('TRAIN8','TRAIN64')


def fingerprint():
    return dict(files={name:train.sha((HERE/name).read_bytes()) for name in
        ('DESIGN_KO.md','run.py','readout.py','queue.py','test_readout.py','test_run.py')},
        parent=pair.fingerprint(),offsets=list(OFFSETS),optimizer_updates=0,
        normalizers=[1024,4096],primary='TRAIN8 new-minus-reference')


def prepare_cpu():
    found,states,_train_rgb,pair_binding=pair.prepare_cpu()
    from . import readout
    if readout.signed.NATIVE_NODES!=1024:raise ValueError('actual factory changed the frozen readout normalizer')
    controls,ref=evaluate.screen_score.validated_pair();candidate,protocol=evaluate.validated_candidate()
    if protocol['reference_binding']!=ref:raise ValueError('different frozen comparison cohorts')
    from collaborative_test.SW_0130_phase_state_integration import run as data
    rgb=np.load(data.VAL_RGB,mmap_mode='r',allow_pickle=False)[:32].copy()
    if train.sha(rgb.tobytes())!=ref['development_RGB_sha256']:raise ValueError('canonical development RGB changed')
    old=dict(TRAIN8=controls['candidate'][list(OFFSETS)],TRAIN64=candidate[list(OFFSETS)])
    binding=dict(fingerprint=fingerprint(),pair_binding=pair_binding,models=pair_binding['models'],
        image_ids=evaluate.screen.IDS[list(OFFSETS)].tolist(),reference_binding=ref,
        RGB_rows_sha256=[train.sha(rgb[i].tobytes()) for i in OFFSETS],
        frozen_old_mask_sha256={arm:[train.sha(row.tobytes()) for row in old[arm]] for arm in ARMS},
        reference_protocol_sha256={
            'TRAIN8':train.sha((evaluate.screen.ARCHIVE/'candidate/protocol.json').read_bytes()),
            'TRAIN64':train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes())},
        validation_GT_used_for_prediction=False,known_TRAIN_GT_available_in_factory=True,
        actual_reference_node_normalizer=readout.signed.NATIVE_NODES,
        source_pair_energy_sha256=train.sha(Path(readout.signed.__file__).read_bytes()))
    return found,states,np.ascontiguousarray(rgb[list(OFFSETS)]),binding


def row_binding(binding,arm,index):
    return dict(arm=arm,image_id=binding['image_ids'][index],model_sha256=binding['models'][arm],
        inference_seed=193000+OFFSETS[index],RGB_sha256=binding['RGB_rows_sha256'][index],
        binding_sha256=train.sha(json.dumps(binding,sort_keys=True).encode()))


def read_row(folder,expected,frozen_old_sha):
    meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'masks.npz').read_bytes()
    if meta['binding']!=expected or meta['prediction_sha256']!=train.sha(raw):raise ValueError('frozen mask row changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as stream:values={key:stream[key].copy() for key in ('old','new')}
    if any(v.dtype!=np.int64 or v.shape!=(64,64) or (v<0).any() for v in values.values()):raise ValueError('invalid native64 masks')
    if train.sha(values['old'].tobytes())!=frozen_old_sha:raise ValueError('reference mask differs from frozen193/194')
    if not np.array_equal(values['old']>0,values['new']>0):raise ValueError('foreground support changed')
    return values


def predict(device='cuda:0',expected_binding=None):
    found,states,rgb,binding=prepare_cpu()
    if expected_binding is not None and binding!=expected_binding:raise ValueError('CPU admission changed')
    device=torch.device(device);torch.cuda.set_device(device);train.probe.model64.move_to_device(found,device)
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    for arm in ARMS:
        if arm=='TRAIN64':
            found.wrapped.load_state_dict(states['wrapped'],strict=True);found.encoder.load_state_dict(states['encoder'],strict=True)
        if train.probe.native._state_sha(found.wrapped,found.encoder)!=binding['models'][arm]:raise ValueError('wrong frozen model')
        for index,offset in enumerate(OFFSETS):
            folder=ARCHIVE/arm/'rows'/f'{index:04d}';expected=row_binding(binding,arm,index)
            if folder.exists():read_row(folder,expected,binding['frozen_old_mask_sha256'][arm][index]);continue
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset);started=time.perf_counter()
            with torch.no_grad():
                image=torch.from_numpy(rgb[index:index+1]).to(device)
                gamma,trace,adj,z,q,rho=train.probe.native._native_forward(found,image,live_tail_steps=0)
                old=evaluate.screen.prediction_from_trace(found,trace,adj,q)
                if train.sha(old.tobytes())!=binding['frozen_old_mask_sha256'][arm][index]:raise ValueError('same-RNG reference replay did not match frozen mask')
                gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
                new=np.asarray(foreground_induced_merge(q[0].detach().cpu().numpy().astype(np.float64),
                    adj[0].detach().cpu().numpy().astype(np.float64),gain,old,grid_shape=(64,64))['labels'],dtype=np.int64)
            if not np.array_equal(old>0,new>0):raise ValueError('new readout changed foreground')
            stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
            with (stage/'masks.npz').open('xb') as stream:
                np.savez_compressed(stream,old=old,new=new);stream.flush();os.fsync(stream.fileno())
            train.write_json(stage/'row.json',dict(binding=expected,prediction_sha256=train.sha((stage/'masks.npz').read_bytes()),seconds=time.perf_counter()-started))
            stage.rename(folder);del image,gamma,trace,adj,z,q,rho
        if train.probe.native._state_sha(found.wrapped,found.encoder)!=binding['models'][arm]:raise RuntimeError('readout mutated model')
        protocol=dict(binding=binding,arm=arm,rows_sha256=[train.sha((ARCHIVE/arm/'rows'/f'{i:04d}'/'row.json').read_bytes()) for i in range(8)])
        path=ARCHIVE/arm/'protocol.json'
        if path.exists():
            if json.loads(path.read_bytes())!=protocol:raise ValueError('arm protocol changed')
        else:train.write_json(path,protocol)


def validated_pair():
    result={};common=None
    for arm in ARMS:
        protocol=json.loads((ARCHIVE/arm/'protocol.json').read_bytes());binding=protocol['binding']
        if protocol['arm']!=arm or binding['fingerprint']!=fingerprint() or binding['validation_GT_used_for_prediction'] is not False:
            raise ValueError('invalid readout protocol')
        if common is not None and binding!=common:raise ValueError('readout cohorts differ')
        common=binding;rows=[]
        if len(binding['image_ids'])!=8 or len(set(binding['image_ids']))!=8:raise ValueError('fixed eight-image cohort required')
        for i in range(8):
            folder=ARCHIVE/arm/'rows'/f'{i:04d}'
            if train.sha((folder/'row.json').read_bytes())!=protocol['rows_sha256'][i]:raise ValueError('row protocol changed')
            rows.append(read_row(folder,row_binding(binding,arm,i),binding['frozen_old_mask_sha256'][arm][i]))
        result[arm]={key:np.stack([r[key] for r in rows]) for key in ('old','new')}
    return result,common


def score():
    labels,binding=validated_pair()
    controls,ref=evaluate.screen_score.validated_pair();candidate,protocol=evaluate.validated_candidate()
    if ref!=binding['reference_binding'] or protocol['reference_binding']!=ref:raise ValueError('frozen references changed')
    for arm,actual in [('TRAIN8',controls['candidate']),('TRAIN64',candidate)]:
        if not np.array_equal(labels[arm]['old'],actual[list(OFFSETS)]):raise ValueError('reference predictions changed')
    gt=evaluate.screen_score.read_ground_truth(ref)[list(OFFSETS)]
    vectors={f'{arm}_{mode}':evaluate.screen_score.measured_vectors(values,gt)[0]
        for arm,variants in labels.items() for mode,values in variants.items()}
    metrics=evaluate.screen_score.independent.METRICS
    scores={name:dict(zip(metrics,values.mean(0).tolist())) for name,values in vectors.items()}
    rng=np.random.RandomState(198);draws=rng.randint(0,8,size=(10000,8));paired={}
    for name,a,b in [('TRAIN8_new_minus_old','TRAIN8_old','TRAIN8_new'),('TRAIN64_new_minus_old','TRAIN64_old','TRAIN64_new'),
                     ('new_TRAIN64_minus_TRAIN8','TRAIN8_new','TRAIN64_new')]:
        delta=vectors[b]-vectors[a];ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
        paired[name]={m:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,m in enumerate(metrics)}
    for arm in ARMS:
        if not np.array_equal(vectors[arm+'_old'][:,1],vectors[arm+'_new'][:,1]):raise ValueError('foreground IoU must be identical')
    primary=paired['TRAIN8_new_minus_old'];fg=primary['fg_ari']
    summary=dict(status='completed_native64_readout_node_normalization_screen',images=8,grid=[64,64],seed_scope=[0],
        scores=scores,paired=paired,primary_gate_passed=fg['mean']>=.01 and fg['ci95'][0]>0 and primary['matched_object_iou']['mean']>=0,
        binding=binding,optimizer_updates=0,foreground_support_identical=True,new_training=False,
        full320_three_seed_goal_verified=False,causal_data_scaling_proved=False,causal_gate_contribution_proved=False,raw_imagewise_scores_public=False)
    train.write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(ARCHIVE/'score_v1/summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--cpu-output',type=Path)
    args=parser.parse_args()
    if args.worker:predict()
    elif args.cpu_output:
        _found,_state,_rgb,binding=prepare_cpu();train.write_json(args.cpu_output,dict(binding=binding,cuda_initialized=torch.cuda.is_initialized()))
    else:parser.error('choose worker or CPU admission')


if __name__=='__main__':main()
