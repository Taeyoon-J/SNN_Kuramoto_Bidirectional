"""Single fixed larger-budget comparison; preserve the failed245 expansion gate."""
import json
from pathlib import Path
import numpy as np
from collaborative_test.SW_0245_native64_precision_three_seed64 import run as prior


def prior_evidence():
    raw=(prior.ROOT_ARCHIVE/'score_v1/summary.json').read_bytes();audit_raw=(prior.ROOT_ARCHIVE/'completion_audit_recovery_v2/summary.json').read_bytes()
    score=json.loads(raw);audit=json.loads(audit_raw)
    if score['fingerprint']!=prior.family_fingerprint() or audit['fingerprint']!=score['fingerprint'] or audit['score_sha256']!=prior.sha(raw) or audit['audit_code_sha256']!=prior.sha((prior.HERE/'completion_audit_recovery_v2.py').read_bytes()) or audit['raw_masks_independently_rescored']!=768 or audit['independent_metrics_per_seed_means_and_deltas_maximum_abs_error']>=1e-12:
        raise ValueError('completed independently audited245 controls required')
    return dict(score_sha256=prior.sha(raw),audit_sha256=prior.sha(audit_raw),negative_original_full320_expansion_gate_preserved=not score['within16_promising_for_separate_full320_confirmation'])


def aggregate_vectors(vectors,metrics):
    names=('control512','within16_512','control128','within16_128','parent237','slot')
    if set(vectors)!=set(names) or len(metrics)!=3:raise ValueError('all fixed-budget models/metrics required')
    arrays={k:np.asarray(vectors[k],dtype=float) for k in names}
    if any(v.shape!=(3,32,3) or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('complete three-seed32 vectors required')
    per={k:v.mean(1) for k,v in arrays.items()};means={k:v.mean(0) for k,v in per.items()};qualified=[];delta={}
    for arm,new,old in (('control','control512','control128'),('within16','within16_512','within16_128')):
        change=means[new]-means[old];delta[arm]=dict(zip(metrics,change.tolist()))
        if change[0]>=.01 and means[new][0]-means['parent237'][0]>=.01 and np.all(means[new]>means['slot']):qualified.append(arm)
    winner=sorted(qualified,key=lambda arm:(-means['control512' if arm=='control' else 'within16_512'][0],arm))[0] if qualified else None
    return dict(scores={k:dict(zip(metrics,v.tolist())) for k,v in means.items()},per_seed={k:{str(i):dict(zip(metrics,row.tolist())) for i,row in enumerate(v)} for k,v in per.items()},
        budget512_minus128=delta,within16_512_minus_control512=dict(zip(metrics,(means['within16_512']-means['control512']).tolist())),qualified_arms_for_separate_full320_confirmation=qualified,selected_arm_for_separate_full320_confirmation=winner)
