"""Reject apparently complete histories that duplicate, omit or reshuffle images."""
import copy
import numpy as np
from collaborative_test.SW_0239_native64_best_70000 import completion_audit as audit


def history(seed,ids,epochs=2,batch=2):
    out=[]
    for epoch in range(epochs):
        order=ids[np.random.RandomState(239000+seed+10000*epoch).permutation(len(ids))]
        for start in range(0,len(ids),batch):
            out.append(dict(update=len(out)+1,TRAIN_ids=order[start:start+batch].tolist(),
                loss=.2,frequency_denoising=.1,gamma_anchor=.1,gradient_norm=.5))
    return out


def test_independent_actual_epoch_coverage_and_seed_order():
    ids=np.array([0,1,1640,70639])
    rows=history(2,ids)
    proof=audit.audit_history(rows,2,count=4,batch=2,epochs=2,registered_ids=ids)
    assert proof['unique_TRAIN_images']==4 and proof['logical_exposures']==8
    for bad in ('repeated','wrong_order','missing_step','nonfinite'):
        changed=copy.deepcopy(rows)
        if bad=='repeated':changed[1]['TRAIN_ids']=changed[0]['TRAIN_ids']
        elif bad=='wrong_order':changed[0]['TRAIN_ids']=changed[0]['TRAIN_ids'][::-1]
        elif bad=='missing_step':changed=changed[:-1]
        else:changed[0]['loss']=float('nan')
        try:audit.audit_history(changed,2,count=4,batch=2,epochs=2,registered_ids=ids)
        except ValueError:pass
        else:raise AssertionError('corrupted complete-data history accepted: '+bad)


def test_nonfinite_or_mismatched_comparison_rejected():
    assert audit.difference([1,2],[1,2])==0
    for a,b in (([1],[1,2]),([float('nan')],[0])):
        try:audit.difference(a,b)
        except ValueError:pass
        else:raise AssertionError('invalid numerical audit comparison accepted')
