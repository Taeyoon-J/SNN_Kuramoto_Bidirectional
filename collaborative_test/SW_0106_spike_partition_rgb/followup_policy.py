"""Pure, CPU-testable promotion gates for the separately queued SW0106 follow-up."""
import json
import math
from pathlib import Path

import numpy as np

METRICS=('fg_ari','foreground_iou','matched_object_iou')
SLOT_MEAN={'fg_ari':0.7749334666743514,'foreground_iou':0.20358913115224261,'matched_object_iou':0.20693697915214362}
SLOT_IOU_FLOORS={'foreground_iou':0.25358913,'matched_object_iou':0.25693698}
BOOTSTRAP_REPLICATES=10000
BOOTSTRAP_SEED=1060106

def load_eval(path,ids=(1320,1639)):
    d=json.loads(Path(path).read_text())
    if d.get('images')!=320 or d.get('ids')!=list(ids):
        raise ValueError(f'evaluation must contain exactly320 images {ids}: {path}')
    if d.get('ground_truth_used_for_prediction') is not False:
        raise ValueError(f'ground truth was used for prediction: {path}')
    rows=d.get('sweep')
    if not isinstance(rows,list) or not rows:
        raise ValueError(f'missing evaluator sweep: {path}')
    scored=rows[0].get('scored_targets',{}).get('our_hdf5')
    if not isinstance(scored,dict):raise ValueError(f'missing our_hdf5 target: {path}')
    per_image={};metrics={}
    for name in METRICS:
        values=np.asarray(scored.get('per_image',{}).get(name,[]),dtype=np.float64)
        if int(scored.get('valid_count',{}).get(name,-1))!=320 or values.shape!=(320,):
            raise ValueError(f'{name} must have 320 valid per-image values: {path}')
        if not np.isfinite(values).all():raise ValueError(f'nonfinite {name}: {path}')
        mean=float(scored.get('metrics',{}).get(name,float('nan')))
        if not math.isfinite(mean) or abs(float(values.mean())-mean)>1e-12:
            raise ValueError(f'{name} aggregate does not match per-image mean: {path}')
        per_image[name]=values;metrics[name]=mean
    return {'metrics':metrics,'per_image':per_image,'path':str(path)}

def paired_image_bootstrap(candidate,reference,replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED):
    """Resample the common320 image IDs; average paired differences across seeds first."""
    c=np.asarray(candidate,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    if c.shape!=(3,320) or r.shape!=(3,320) or replicates<1000:
        raise ValueError('bootstrap requires paired three-seed x 320 arrays and >=1000 replicates')
    per_image_delta=(c-r).mean(axis=0)
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,320,size=(replicates,320))
    boot=per_image_delta[sampled].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975],method='linear')
    return {'replicates':replicates,'seed':seed,'unit':'image ID; seed-paired deltas averaged per image',
      'ci95':[float(lo),float(hi)],'mean_delta':float(per_image_delta.mean()),'lower_bound_positive':bool(lo>0)}

def pilot_promotion_gate(candidate,control,sw97):
    """Registered 3-seed pilot gate for launching one full70k continuation."""
    if any(len(rows)!=3 for rows in (candidate,control,sw97)):
        raise ValueError('pilot gate requires all three seeds for candidate, joint control, and SW0097')
    means={}
    for label,rows in (('candidate',candidate),('control',control),('sw0097',sw97)):
        means[label]={m:float(np.mean([x['metrics'][m] for x in rows])) for m in METRICS}
    positive_control=sum(candidate[s]['metrics']['fg_ari']>control[s]['metrics']['fg_ari'] for s in range(3))
    positive_sw97=sum(candidate[s]['metrics']['fg_ari']>sw97[s]['metrics']['fg_ari'] for s in range(3))
    bootstrap={}
    for reference,rows in (('control',control),('sw0097',sw97)):
        bootstrap[reference]=paired_image_bootstrap(
          np.stack([x['per_image']['fg_ari'] for x in candidate]),
          np.stack([x['per_image']['fg_ari'] for x in rows]))
    checks={
      'candidate_mean_fg_plus_0.01_vs_control':means['candidate']['fg_ari']>=means['control']['fg_ari']+.01,
      'candidate_mean_fg_plus_0.01_vs_SW0097':means['candidate']['fg_ari']>=means['sw0097']['fg_ari']+.01,
      'at_least_two_seed_fg_gains_vs_control':positive_control>=2,
      'at_least_two_seed_fg_gains_vs_SW0097':positive_sw97>=2,
      'mean_foreground_iou_slot_plus_0.05':means['candidate']['foreground_iou']>=SLOT_IOU_FLOORS['foreground_iou'],
      'mean_object_iou_slot_plus_0.05':means['candidate']['matched_object_iou']>=SLOT_IOU_FLOORS['matched_object_iou'],
      'paired_image_bootstrap_positive_vs_control':bootstrap['control']['lower_bound_positive'],
      'paired_image_bootstrap_positive_vs_SW0097':bootstrap['sw0097']['lower_bound_positive']}
    return {'status':'passed' if all(checks.values()) else 'failed','means':means,
      'seed_fg_positive_counts':{'control':positive_control,'sw0097':positive_sw97},
      'bootstrap_fg':bootstrap,'checks':checks,'registered_bootstrap_replicates':BOOTSTRAP_REPLICATES,
      'registered_bootstrap_seed':BOOTSTRAP_SEED}

def full_validation_reserve_gate(candidate,control,slot):
    """Final validation gate to freeze models before any reserve slice is read."""
    if any(len(rows)!=3 for rows in (candidate,control,slot)):
        raise ValueError('reserve unlock requires all3 candidate/control/Slot full320 endpoints')
    means={label:{m:float(np.mean([x['metrics'][m] for x in rows])) for m in METRICS}
      for label,rows in (('candidate',candidate),('control',control),('slot',slot))}
    checks={f'candidate_mean_{m}_strictly_above_slot':means['candidate'][m]>means['slot'][m] for m in METRICS}
    return {'status':'passed' if all(checks.values()) else 'failed','means':means,'checks':checks}

