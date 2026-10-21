"""Terminal fixed320 masks only; all twelve protocols precede score GT access."""
import argparse,io,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import scope,factory,run
from collaborative_test.SW_0240_native64_70000_cached_io import train as old,data
from collaborative_test.SW_0237_native64_converged_full320 import evaluate as ref

def predict(p,report,device):
    import h5py
    a=factory.archive(p['model'],p['seed']);found=p['found'];found.encoder.eval();found.wrapped.eval()
    binding=dict(binding=p['binding'],training_report_sha256=run.sha((a/'training_report.json').read_bytes()),model_sha256=report['final_model_sha256'])
    with h5py.File(p['dataset']['path'],'r') as stream:rgb=data.canonical_rgb(np.asarray(stream['image'][1320:1640]))
    rows=[]
    for i,image in enumerate(rgb):
        folder=a/'rows'/f'{i:04d}';row_binding=dict(binding=binding,index=i,image_id=1320+i,RGB_sha256=run.sha(image.tobytes()),inference_seed=193000+i)
        if folder.exists():
            meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'prediction.npz').read_bytes()
            if any(meta[k]!=v for k,v in row_binding.items()) or meta['data_sha256']!=run.sha(raw):raise ValueError('frozen prediction row changed')
            with np.load(io.BytesIO(raw),allow_pickle=False) as values:labels=values['labels'].copy()
        else:
            torch.manual_seed(193000+i);torch.cuda.manual_seed(193000+i)
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=old.source.source.full.screen.native._native_forward(found,torch.from_numpy(image[None]).to(device),live_tail_steps=0)
                labels=np.asarray(old.source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64)
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            old.helpers.save_arrays(stage/'prediction.npz',labels=labels)
            run.write_json(stage/'row.json',dict(row_binding,data_sha256=run.sha((stage/'prediction.npz').read_bytes())))
            stage.rename(folder);del gamma,trace,adj,z,q,rho
        if labels.shape!=(64,64) or labels.dtype!=np.int64 or np.any(labels<0):raise ValueError('invalid native64 labels')
        rows.append(labels)
    if factory.model_sha(found)!=report['final_model_sha256']:raise ValueError('inference mutated final model')
    protocol=dict(binding=binding,image_ids=list(scope.EVAL_IDS),labels_sha256=run.sha(np.stack(rows).tobytes()),RGB_sha256=run.sha(rgb.tobytes()),total_steps=1024,settle_steps=512,DEV_GT_read=False,optimizer_updates=0)
    path=a/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('completed protocol changed')
    else:run.write_json(path,protocol)

def validated(model,seed):
    a=factory.archive(model,seed);report=json.loads((a/'training_report.json').read_bytes());protocol=json.loads((a/'protocol.json').read_bytes())
    if report['binding']['fingerprint']!=run.fingerprint() or report['actual_Adam_steps']!=[scope.UPDATES] or report['exposures']!=350000:raise ValueError('different registered budget')
    if report['binding']['model']!=model or report['binding']['seed']!=seed or protocol['binding']['model_sha256']!=report['final_model_sha256'] or protocol['binding']['training_report_sha256']!=run.sha((a/'training_report.json').read_bytes()):raise ValueError('source/model/report identity mismatch')
    if protocol['binding']['binding']!=report['binding'] or protocol['image_ids']!=list(scope.EVAL_IDS) or protocol['DEV_GT_read'] is not False or (protocol['total_steps'],protocol['settle_steps'])!=(1024,512):raise ValueError('changed evaluation contract')
    if run.sha((a/'encoder_work/boundary.ptx').read_bytes())!=report['checkpoint_sha256']:raise ValueError('terminal checkpoint changed')
    rows=[]
    for i in range(320):
        folder=a/'rows'/f'{i:04d}';meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'prediction.npz').read_bytes()
        if meta['binding']!=protocol['binding'] or meta['index']!=i or meta['image_id']!=1320+i or meta['inference_seed']!=193000+i or run.sha(raw)!=meta['data_sha256']:raise ValueError('row metadata or digest differs')
        with np.load(io.BytesIO(raw),allow_pickle=False) as values:rows.append(values['labels'].copy())
    labels=np.stack(rows)
    if labels.shape!=(320,64,64) or labels.dtype!=np.int64 or np.any(labels<0) or run.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('mask array contract differs')
    return dict(labels=labels,report=report,protocol=protocol)

def score_and_audit():
    bundle={(m,s):validated(m,s) for m in scope.MODELS for s in scope.SEEDS};slots=ref.load_slot_masks()
    datasets=[v['report']['binding']['dataset'] for v in bundle.values()]
    if any(v!=datasets[0] for v in datasets) or len({v['protocol']['RGB_sha256'] for v in bundle.values()})!=1:raise ValueError('different evaluation dataset across jobs')
    # Validate every final checkpoint and actual Adam/history before scoring.
    proofs={}
    for (m,s),value in bundle.items():
        p=run.prepare(m,s);a=factory.archive(m,s);report=value['report']
        saved=factory.payload(a/'encoder_work/boundary.ptx',report['checkpoint_sha256'])
        if saved['bindings']!=report['checkpoint_binding'] or saved['completed_updates']!=scope.UPDATES or saved['runtime']!=report['runtime']:raise ValueError('saved checkpoint binding/runtime differs')
        if {int(v['step'].item()) for v in saved['optimizer_state']['state'].values()}!={scope.UPDATES}:raise ValueError('actual Adam budget differs')
        if saved['optimizer_schema']!=old.checkpoint.optimizer_schema(p['optimizer']):raise ValueError('optimizer groups differ')
        run.history_check(saved['history'],s)
        p['found'].encoder.load_state_dict(saved['encoder_state'],strict=True);old.projection.module(p['found']).load_state_dict(saved['wrapped_state'],strict=True)
        if factory.model_sha(p['found'])!=report['final_model_sha256'] or not run.invariant(p):raise ValueError('restored actual model/teacher/core differs')
        proofs[f'{m}_seed{s}']=dict(actual_final_CPU_restore=True,actual_Adam_steps=scope.UPDATES,exact_35000_per_epoch_verified=True)
        del p,saved
    gt=old.source.source.full.read_gt(dict(dataset=datasets[0]));metric=old.source.source.full.metrics;keys=metric.independent.METRICS
    vectors={};error=0.
    for (m,s),value in bundle.items():
        v,d=metric.measured_vectors(value['labels'],gt);error=max(error,d);vectors[f'{m}_seed{s}']=v
    for s in scope.SEEDS:
        v,d=metric.measured_vectors(slots[s]['labels64'],gt);error=max(error,d);vectors[f'slot_seed{s}']=v
    if error>=1e-12 or any(v.shape!=(320,3) or not np.isfinite(v).all() for v in vectors.values()):raise ValueError('metric agreement or shape failed')
    groups={m:np.stack([vectors[f'{m}_seed{s}'] for s in scope.SEEDS]) for m in (*scope.MODELS,'slot')}
    scores={m:dict(zip(keys,v.mean((0,1)).tolist())) for m,v in groups.items()}
    differences={}
    draw=np.random.RandomState(249).randint(0,320,(10000,320))
    for m in scope.MODELS:
        delta=(groups[m]-groups['240']).mean(0);ci=np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
        differences[m]={k:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,k in enumerate(keys)}
    result=dict(status='complete_twelve_jobs_independently_audited',specification=scope.specification(),scores=scores,
        per_seed={k:dict(zip(keys,v.mean(0).tolist())) for k,v in vectors.items()},paired_model_minus240=differences,
        ranking_by_mean_FG_ARI=sorted(scope.MODELS,key=lambda m:(-scores[m]['fg_ari'],m)),mask_supervised=True,
        inherited_training_budgets_not_equalized=True,evaluation_scope='reused fixed validation, not untouched test',
        raw_masks_independently_rescored=4800,independent_metric_max_abs_error=error,checkpoint_audits=proofs,
        goal_work_complete=True,followup_model_search=False)
    run.write_json(factory.ARCHIVE/'score_v1/private_vectors.json',{k:v.tolist() for k,v in vectors.items()})
    run.write_json(factory.ARCHIVE/'score_v1/summary.json',result)
    return result

if __name__=='__main__':print(json.dumps(score_and_audit()),flush=True)
