"""Strict registered three-seed64 aggregation; do not promote to full320."""
import numpy as np


def aggregate_vectors(vectors,metrics):
    names=('control','within16','parent237','slot')
    if set(vectors)!=set(names) or len(metrics)!=3:raise ValueError('all registered models and metrics required')
    arrays={k:np.asarray(vectors[k],dtype=np.float64) for k in names}
    if any(v.shape!=(3,64,3) or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('complete three-seed64 finite vectors required')
    per={k:v.mean(1) for k,v in arrays.items()};means={k:v.mean(0) for k,v in per.items()}
    delta=means['within16']-means['control'];parent=means['within16']-means['parent237']
    positive=bool(delta[0]>=.01 and parent[0]>=.01 and np.all(means['within16'][1:]>means['slot'][1:]))
    return dict(scores={k:dict(zip(metrics,v.tolist())) for k,v in means.items()},
        per_seed={k:{str(i):dict(zip(metrics,row.tolist())) for i,row in enumerate(v)} for k,v in per.items()},
        within16_minus_control=dict(zip(metrics,delta.tolist())),within16_minus_parent237=dict(zip(metrics,parent.tolist())),
        within16_promising_for_separate_full320_confirmation=positive)
