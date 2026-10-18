"""Audit all frozen paired masks before GT; compare three seeds on fixed8 IDs."""
import json
from pathlib import Path
import numpy as np
from . import run as api


def paired(delta):
    if delta.shape != (8,3) or not np.isfinite(delta).all():
        raise ValueError('eight paired images, three finite metrics required')
    draws=np.random.RandomState(203).randint(0,8,(10000,8))
    ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    return {m:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist())
            for i,m in enumerate(api.metrics.independent.METRICS)}


def run():
    # Require all six arrays to be immutable/validated before reading GT.
    all_labels={};bindings={};protocols={}
    for seed in (0,1,2):
        api.configure(seed)
        labels,binding=api.validated_pair()
        original,ref=api.reference()
        if binding['reference_binding']!=ref or not np.array_equal(labels['old'],original[list(api.OFFSETS)]):
            raise ValueError('registered original-mask replay changed')
        admitted=json.loads((api.ARCHIVE/'registration_v1/admission.json').read_bytes())
        if admitted['binding']!=binding or admitted['cuda_initialized']:
            raise ValueError('actual CPU model admission changed')
        all_labels[seed]=labels;bindings[seed]=binding
        protocols[str(seed)]=api.train.sha((api.ARCHIVE/'prediction/protocol.json').read_bytes())
    ref=bindings[0]['reference_binding']
    if any(b['reference_binding']!=ref for b in bindings.values()):
        raise ValueError('seed cohorts differ')
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    slots={seed:api.metrics.slot._load_slot(SLOT_ROOT,seed) for seed in (0,1,2)}
    for slot in slots.values():
        if not np.array_equal(slot['ids'][:32],api.screen.IDS):
            raise ValueError('comparable Slot IDs differ')
    gt=api.metrics.read_ground_truth(ref)[list(api.OFFSETS)]
    vectors={};maximum=0.
    for seed,labels in all_labels.items():
        for mode,pred in labels.items():
            name=f'seed{seed}_{mode}'
            values,error=api.metrics.measured_vectors(pred,gt)
            vectors[name]=values;maximum=max(maximum,error)
        old=vectors[f'seed{seed}_old'];new=vectors[f'seed{seed}_new']
        if not np.array_equal(old[:,1],new[:,1]):
            raise ValueError('foreground IoU changed')
        source=json.loads((api.ROOT_ARCHIVE/f'seed{seed}/score_v1/summary.json').read_bytes())
        scores={mode:dict(zip(api.metrics.independent.METRICS,vectors[f'seed{seed}_{mode}'].mean(0).tolist())) for mode in ('old','new')}
        if source['scores']!=scores or source['paired']!=paired(new-old):
            raise ValueError('individual completed scores did not independently replay')
        vectors[f'slot_seed{seed}'],error=api.metrics.measured_vectors(slots[seed]['labels64'][list(api.OFFSETS)],gt)
        maximum=max(maximum,error)
    for mode in ('old','new'):
        vectors[mode+'_three_seed_mean']=np.mean([vectors[f'seed{s}_{mode}'] for s in (0,1,2)],axis=0)
    vectors['slot_three_seed_mean']=np.mean([vectors[f'slot_seed{s}'] for s in (0,1,2)],axis=0)
    delta=paired(vectors['new_three_seed_mean']-vectors['old_three_seed_mean'])
    metrics=api.metrics.independent.METRICS
    gate=delta['fg_ari']['mean']>=.01 and delta['fg_ari']['ci95'][0]>0 and delta['matched_object_iou']['mean']>=0
    scores={name:dict(zip(metrics,v.mean(0).tolist())) for name,v in vectors.items()}
    summary=dict(status='three_seed_fixed8_geometric_consensus_independently_audited',
        seed_scope=[0,1,2],images_per_seed=8,image_ids=bindings[0]['image_ids'],grid=[64,64],
        scores=scores,paired_new_minus_old=delta,
        paired_new_minus_comparable_Slot=paired(vectors['new_three_seed_mean']-vectors['slot_three_seed_mean']),
        extension_gate_passed=bool(gate),optimizer_updates=0,new_training=False,
        foreground_support_identical=True,source_models_mask_supervised=True,
        unsupervised_leaderboard_eligible=False,causal_gate_contribution_proved=False,
        full320_three_seed_goal_verified=False,raw_imagewise_scores_public=False,
        bootstrap_scope='paired images, conditional on these three fixed seed lineages',
        independent_maximum_metric_abs_error=maximum,protocol_sha256=protocols,
        source_sha256=api.train.sha(Path(__file__).read_bytes()))
    for name,value in [('summary.json',summary),('evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))]:
        target=api.ROOT_ARCHIVE/'aggregate_v1'/name
        if target.exists():
            if json.loads(target.read_bytes())!=value:raise ValueError('preserve frozen aggregate')
        else:api.train.write_json(target,value)
    print(json.dumps(summary))
    return summary


if __name__=='__main__':run()
