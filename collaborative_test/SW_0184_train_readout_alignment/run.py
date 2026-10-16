"""Read-only TRAIN16 replay of both frozen SW0183 models."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0183_shared_prefix_unique_scaling import train, predict_evaluate as pe
from collaborative_test.SW_0180_qce_train_validation_triage import run as helpers

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXPERIMENT='SW0184_train_readout_alignment'
SEEDS=(0,1,2)
POSITIONS=tuple(range(0,512,64))+tuple(range(512,1024,64))
ARMS=('control512','candidate1024')
BATCH=4
sha256_file=pe.sha256_file
write_report=helpers.write_report

def implementation_fingerprint():
    files={str(Path(__file__).relative_to(ROOT)):sha256_file(__file__),
        str((HERE/'DESIGN_KO.md').relative_to(ROOT)):sha256_file(HERE/'DESIGN_KO.md'),
        str(Path(helpers.__file__).relative_to(ROOT)):sha256_file(helpers.__file__)}
    return dict(experiment=EXPERIMENT,files=files,parent_prediction=pe.implementation_fingerprint(),
        positions=list(POSITIONS),seeds=list(SEEDS),arms=list(ARMS),batch=BATCH,
        steps=1024,settle=512,optimizer_updates=0,validation_data_read=False)

def _assert_cache(inputs):
    ids=np.asarray(inputs['ordered_train_ids'],dtype=np.int64)
    arrays=inputs['arrays']; rgb=np.asarray(arrays['rgb']); modal=np.asarray(arrays['modal_gt'])
    if (ids.shape!=(1024,) or len(set(ids.tolist()))!=1024 or
        rgb.shape!=(1024,128,128,3) or rgb.dtype!=np.uint8 or
        modal.shape!=(1024,32,32) or modal.dtype!=np.int32 or modal.min()<0):
        raise ValueError('SW0184 requires admitted canonical TRAIN1024 cache')
    if not np.all(((ids>=0)&(ids<=999))|((ids>=1640)&(ids<=70639))):
        raise ValueError('SW0184 cohort contains non-TRAIN IDs')
    return ids,rgb,modal

def prepare_inputs_cpu(seed_subset=SEEDS):
    if torch.cuda.is_initialized(): raise RuntimeError('CPU admission precedes CUDA')
    seeds=tuple(seed_subset)
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in SEEDS for s in seeds):
        raise ValueError('invalid seed subset')
    inputs=train.prepare_training_inputs_cpu(); ids,_,_=_assert_cache(inputs)
    tq=train._module(train.QUEUE_MODULE); records={}; states={}
    for seed in seeds:
        state_path=tq.state_path(seed); state=json.loads(state_path.read_text())
        if (state.get('status')!='complete' or state.get('seed')!=seed or
            state.get('experiment')!='SW0183_training_queue' or state.get('validation_data_read') is not False):
            raise ValueError('frozen SW0183 training not complete')
        artifacts={}
        for arm in ARMS:
            path=train._paths(train.OUTPUT_ROOT,arm,seed)[1]
            item=train.validate_training_artifact(arm,path,seed=seed)
            if state['arm_manifests'][arm]!=item['manifest_sha256']: raise ValueError('lineage mismatch')
            artifacts[arm]={**item,'path':path}
        prepared=dict(training_inputs=inputs,artifacts=artifacts)
        for arm in ARMS:
            found,digest=pe._load_model(seed,arm,prepared,'cpu')
            assert digest==artifacts[arm]['manifest']['model_state_sha256']
            del found
        records[seed]=prepared; states[str(seed)]=dict(path=str(state_path),sha256=sha256_file(state_path))
    assert not torch.cuda.is_initialized()
    return dict(inputs=inputs,records=records,training_states=states,selected_seeds=list(seeds),
        ordered_cohort_ids=ids[list(POSITIONS)].tolist(),cuda_initialized=False)

def pair_diagnostics(q,rho,labels):
    """Class-balanced pair statistics; diagonal excluded; BG scope explicit."""
    q=np.asarray(q,dtype=np.float64); rho=np.asarray(rho,dtype=np.float64); y=np.asarray(labels).reshape(-1)
    n=len(y)
    if q.shape!=(n,n) or rho.shape!=(4,n,n) or not np.isfinite(q).all() or not np.isfinite(rho).all():
        raise ValueError('invalid Q/rho geometry or values')
    same=y[:,None]==y[None,:]; off=~np.eye(n,dtype=bool)
    scopes={'all':off,'foreground':off&(y[:,None]>0)&(y[None,:]>0)}
    result={}
    for scope,valid in scopes.items():
        rows=[]
        for cls in np.unique(y):
            if scope=='foreground' and cls==0: continue
            queries=y==cls
            pos=valid&queries[:,None]&same; neg=valid&queries[:,None]&(~same)
            if not pos.any() or not neg.any(): continue
            rows.append(dict(same_q=float(q[pos].mean()),different_q=float(q[neg].mean()),
                same_threshold_recall=float((q[pos]>=.5).mean()),cross_threshold_rate=float((q[neg]>=.5).mean()),
                rho_same=[float(x[pos].mean()) for x in rho],rho_different=[float(x[neg].mean()) for x in rho]))
        if rows:
            result[scope]={key:np.mean([r[key] for r in rows],axis=0).tolist() for key in rows[0]}
            result[scope]['valid_classes']=len(rows)
        else: result[scope]={'valid_classes':0}
    return result

@torch.no_grad()
def _forward(found,rgb,target,support,seed,batch_index):
    helpers._paired_seed_rng(seed,batch_index)
    gamma,trace,adj,z,q,rho=helpers.native._native_forward(found,rgb,live_tail_steps=0)
    if q.shape!=(BATCH,1024,1024) or rho.shape!=(BATCH,4,1024,1024): raise ValueError('native geometry changed')
    qcc=np.asarray(helpers.sw136._qcc_labels(trace),dtype=np.int32)
    assert qcc.shape==(BATCH,32,32)
    loss,margin=helpers._class_loss_and_margin(q,target.reshape(BATCH,-1).to(q.device))
    gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().cpu())
    assert np.isfinite(gain) and gain>0
    support=qcc if support is None else support
    outputs={key:[] for key in ('conditional','standalone')}; pairs=[]
    for i in range(BATCH):
        qn=q[i].cpu().numpy().astype(np.float64); an=adj[i].cpu().numpy().astype(np.float64)
        pairs.append(pair_diagnostics(qn,rho[i].cpu().numpy(),target[i].cpu().numpy()))
        for ep,s in (('conditional',support[i]),('standalone',qcc[i])):
            mask=np.asarray(helpers.solver.foreground_induced_merge(qn,an,gain,s,grid_shape=(32,32))['labels'],dtype=np.int32)
            assert np.array_equal(mask>0,s>0)
            outputs[ep].append(mask)
    return dict(qce=loss.cpu().tolist(),margin=margin.cpu().tolist(),pairs=pairs,qcc=qcc,
        outputs={k:np.stack(v) for k,v in outputs.items()},gain=gain,
        digests={k:helpers._sha_array(v.detach().cpu().numpy()) for k,v in
            (('gamma',gamma),('Q',q),('rho',rho),('A',adj),('S',trace['component_spikes']))})

def run_diagnostic(prepared,*,seed_subset=SEEDS,device='cuda:0'):
    if prepared.get('cuda_initialized') is not False: raise ValueError('unadmitted inputs')
    dev=torch.device(device)
    if dev.type!='cuda' or not torch.cuda.is_initialized(): raise RuntimeError('leased GPU required')
    seeds=tuple(seed_subset)
    if not seeds or not set(seeds).issubset(prepared['selected_seeds']): raise ValueError('unadmitted seeds')
    ids,rgb,modal=_assert_cache(prepared['inputs'])
    report=dict(experiment=EXPERIMENT,status='complete',implementation_fingerprint=implementation_fingerprint(),
        optimizer_updates=0,validation_data_read=False,ground_truth_scope='TRAIN modal cache only',
        cohort_ids=ids[list(POSITIONS)].tolist(),positions=list(POSITIONS),seeds={})
    for seed in seeds:
        prepared_seed=prepared['records'][seed]; all_rows={}; models={}
        for arm in ARMS:
            found,before=pe._load_model(seed,arm,prepared_seed,dev); rows=[]
            for batch_index,start in enumerate(range(0,len(POSITIONS),BATCH)):
                indices=list(POSITIONS[start:start+BATCH])
                images=torch.from_numpy(np.ascontiguousarray(rgb[indices])).to(dev)
                targets=torch.from_numpy(np.ascontiguousarray(modal[indices])).to(dev)
                support=None if arm=='control512' else all_rows['control512'][batch_index]['qcc']
                rows.append(_forward(found,images,targets,support,seed,batch_index))
                del images,targets
            after=train.source._state_sha(found.wrapped,found.encoder)
            if before!=after: raise ValueError('no-update replay mutated model')
            models[arm]=dict(before_sha256=before,after_sha256=after,state_unchanged=True,
                manifest_sha256=prepared_seed['artifacts'][arm]['manifest_sha256'])
            all_rows[arm]=rows
            del found
        summary={}
        for arm,rows in all_rows.items():
            masks={ep:np.concatenate([r['outputs'][ep] for r in rows]) for ep in ('conditional','standalone')}
            summary[arm]=dict(qce=[x for r in rows for x in r['qce']],margin=[x for r in rows for x in r['margin']],
                pairs=[x for r in rows for x in r['pairs']],
                metrics={ep:helpers._metric_arrays(values,modal[list(POSITIONS)]) for ep,values in masks.items()},
                qcc_foreground_fraction=[float((m>0).mean()) for r in rows for m in r['qcc']],
                batch_digests=[r['digests'] for r in rows],gain=rows[0]['gain'])
        report['seeds'][str(seed)]=dict(models=models,arms=summary)
    return report
