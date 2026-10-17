"""Independent NumPy/SciPy recomputation after SW0185 predictions freeze."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

METRICS=('fg_ari','foreground_iou','matched_object_iou')

def modal_labels(pixels,patch=4):
    values=np.asarray(pixels)
    if values.ndim==4 and values.shape[-1]==1:values=values[...,0]
    if values.ndim!=3 or not np.issubdtype(values.dtype,np.integer) or np.any(values<0):
        raise ValueError('nonnegative integer pixel masks required')
    b,h,w=values.shape
    if type(patch) is not int or patch<1 or h%patch or w%patch:raise ValueError('unaligned patch geometry')
    blocks=values.reshape(b,h//patch,patch,w//patch,patch).transpose(0,1,3,2,4).reshape(b,h//patch,w//patch,patch*patch)
    labels=np.zeros((b,h//patch,w//patch),dtype=np.int64);best=np.zeros_like(labels)
    for label in np.unique(values):
        votes=(blocks==label).sum(-1);take=votes>best
        labels[take]=label;best[take]=votes[take]
    return labels

def image_metrics(pred,target):
    pred,target=np.asarray(pred),np.asarray(target)
    if pred.shape!=target.shape or pred.ndim!=2:raise ValueError('aligned 2D label maps required')
    fg=target>0;pf=pred>0;n=int(fg.sum())
    if n<2:raise ValueError('at least two true foreground patches required')
    _,ti=np.unique(target[fg],return_inverse=True);_,pi=np.unique(pred[fg],return_inverse=True)
    table=np.zeros((ti.max()+1,pi.max()+1),dtype=np.int64);np.add.at(table,(ti,pi),1)
    choose=lambda x:np.sum(x*(x-1)/2,dtype=np.float64)
    common=choose(table);tpairs=choose(table.sum(1));ppairs=choose(table.sum(0))
    expected=tpairs*ppairs/(n*(n-1)/2);denom=(tpairs+ppairs)/2-expected
    ari=float((common-expected)/denom) if denom else 1.
    union=(pf|fg).sum();fiou=float((pf&fg).sum()/union) if union else 1.
    tids=np.unique(target[fg]);pids=np.unique(pred[pf]);ious=np.zeros((len(tids),len(pids)))
    for i,a in enumerate(tids):
        for j,b in enumerate(pids):
            t,q=target==a,pred==b;ious[i,j]=(t&q).sum()/(t|q).sum()
    rows,cols=linear_sum_assignment(-ious)
    return np.asarray([ari,fiou,float(ious[rows,cols].sum()/len(tids))],dtype=np.float64)

def compare_prediction(labels,gt,published,seed):
    measured=np.array([image_metrics(p,t) for p,t in zip(labels,gt)])
    if measured.shape!=(320,3):raise ValueError('exact fixed320 geometry required')
    errors=[]
    for j,metric in enumerate(METRICS):
        expected=np.asarray(published[metric]['per_seed_per_image'][str(seed)],dtype=np.float64)
        error=float(np.max(np.abs(measured[:,j]-expected)))
        if not np.isfinite(error) or error>1e-12:raise ValueError(f'independent {metric} mismatch: {error}')
        errors.append(error)
    return max(errors)

def verify_arithmetic(report):
    for values in report['scores'].values():
        for item in values.values():
            matrix=np.array([item['per_seed_per_image'][str(s)] for s in (0,1,2)],dtype=np.float64)
            if matrix.shape!=(3,320) or not np.isfinite(matrix).all():raise ValueError('incomplete seed/image vectors')
            if np.max(np.abs(matrix.mean(0)-item['per_image']))>1e-12:raise ValueError('image pairing mismatch')
            if abs(float(matrix.mean())-item['mean'])>1e-12:raise ValueError('overall mean mismatch')
            for i,seed in enumerate((0,1,2)):
                if abs(float(matrix[i].mean())-item['per_seed_mean'][str(seed)])>1e-12:raise ValueError('seed mean mismatch')
    sample=np.random.RandomState(178).randint(0,320,size=(10000,320));paired={}
    for endpoint in ('conditional','standalone'):
        left=report['scores'][f'candidate_{endpoint}']
        for group in ('SW0165_foreground_induced','Slot70k','SW0181'):
            right=report['scores'][f'{group}_{endpoint}'];name=f'{endpoint}_candidate_minus_{group}'
            paired[name]={}
            for metric in METRICS:
                delta=np.asarray(left[metric]['per_image'])-np.asarray(right[metric]['per_image'])
                mean=float(delta.mean());ci=np.percentile(delta[sample].mean(1),[2.5,97.5])
                stored=report['paired_bootstrap'][name][metric]
                if abs(mean-stored['mean'])>1e-12 or np.max(np.abs(ci-stored['ci95']))>1e-12:raise ValueError('paired CI mismatch')
                paired[name][metric]=dict(mean=mean,ci95=ci.tolist())
    scores=report['scores'];c=scores['candidate_standalone'];slot=scores['Slot70k_standalone']
    fg=c['fg_ari']['mean'];sw=scores['SW0165_foreground_induced_standalone'];old=scores['SW0181_standalone']
    gate=dict(primary_endpoint='standalone',
        fg_strictly_above_sw0165_three_seed_mean=fg>sw['fg_ari']['mean'],
        fg_paired_ci95_lower_above_zero_vs_sw0165=paired['standalone_candidate_minus_SW0165_foreground_induced']['fg_ari']['ci95'][0]>0,
        fg_strictly_above_slot_three_seed_mean=fg>slot['fg_ari']['mean'],
        fg_paired_ci95_lower_above_zero_vs_slot=paired['standalone_candidate_minus_Slot70k']['fg_ari']['ci95'][0]>0,
        fg_strictly_above_sw0181_three_seed_mean=fg>old['fg_ari']['mean'],
        fg_paired_ci95_lower_above_zero_vs_sw0181=paired['standalone_candidate_minus_SW0181']['fg_ari']['ci95'][0]>0,
        foreground_iou_strictly_above_slot_three_seed_mean=c['foreground_iou']['mean']>slot['foreground_iou']['mean'],
        matched_object_iou_strictly_above_slot_three_seed_mean=c['matched_object_iou']['mean']>slot['matched_object_iou']['mean'],
        fg_ari_at_least_0_87_goal=fg>=.87,mask_supervised=True,unsupervised_leaderboard_eligible=False)
    gate['primary_gate_passed']=bool(fg>old['fg_ari']['mean'] and
        paired['standalone_candidate_minus_SW0181']['fg_ari']['ci95'][0]>0 and
        all(c[m]['mean']>slot[m]['mean'] for m in METRICS))
    if report['gate']!=gate:raise ValueError('registered gate recomputation mismatch')
    return gate,paired

def audit_completed():
    # Imports and GT access are deliberately delayed until this owned CPU worker.
    import h5py
    from collaborative_test.SW_0185_qce_spatial_connectivity import score
    from collaborative_test.SW_0185_qce_spatial_connectivity import train
    raw=score.SCORE_PATH.read_bytes();report=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if report.get('status')!='complete' or report.get('experiment')!='SW0185_qce_spatial_connectivity_three_seed_score':
        raise ValueError('unexpected completed score producer')
    prepared=score.prepare_score_inputs_cpu()  # Revalidates all archives before GT.
    if report['implementation_fingerprint']!=score.implementation_fingerprint():raise ValueError('score source changed')
    for seed in (0,1,2):
        frozen=prepared['frozen'][seed];published=report['candidate_prediction_hashes'][str(seed)]
        if published['prediction_sha256']!=frozen['result']['prediction_sha256'] or published['protocol_sha256']!=frozen['result']['protocol_sha256']:
            raise ValueError('score does not bind current actual prediction')
    with h5py.File(prepared['dataset'],'r') as hdf:
        gt=modal_labels(np.asarray(hdf['mask'][1320:1640]))
    errors=[];training={}
    for seed in (0,1,2):
        frozen=prepared['frozen'][seed]
        for endpoint in ('conditional','standalone'):
            errors.append(compare_prediction(frozen['result']['labels'][endpoint],gt,report['scores'][f'candidate_{endpoint}'],seed))
        slot=frozen['prepared']['references']['slot']['labels']
        errors.append(compare_prediction(slot,gt,report['scores']['Slot70k_standalone'],seed))
        artifact=train.OUTPUT_ROOT/f'{score.pe.ARM}_seed{seed}'
        manifest=train.validate_training_artifact(score.pe.ARM,artifact,seed=seed)
        training[str(seed)]=dict(accepted_updates=manifest['accepted_updates'],
            initial_model_state_sha256=manifest['initial_model_state_sha256'],final_model_state_sha256=manifest['final_model_state_sha256'],
            checkpoint_sha256=manifest['checkpoint_sha256'],manifest_sha256=score.sha256_file(artifact/'manifest.json'))
    gate,paired=verify_arithmetic(report)
    if hashlib.sha256(score.SCORE_PATH.read_bytes()).hexdigest()!=digest:raise ValueError('score changed during audit')
    return dict(status='SW0185_independent_actual_prediction_metrics_GT_modal_means_CI_and_gates_verified',
        score_sha256=digest,maximum_metric_abs_error=max(errors),candidate_endpoint_image_rows=1920,
        slot_image_rows=960,training=training,gate=gate,paired_bootstrap=paired,
        means={g:{m:v['mean'] for m,v in rows.items()} for g,rows in report['scores'].items()},
        validation_data_read_for_audit=True,optimizer_updates=0,incumbent_changed=False,
        observed_unix=time.time(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
