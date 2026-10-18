"""One predetermined secondary64 validation; never change original16 results."""
import json
import numpy as np
from . import run


def validated_all():return {s:run.validated_seed(s) for s in run.SEEDS}


def score():
    bundle=validated_all();datasets=[bundle[s]['protocol']['binding']['dataset'] for s in run.SEEDS]
    if any(v!=datasets[0] for v in datasets):raise ValueError('different fixed evaluation data')
    gt=run.source.source.full.read_gt(dict(dataset=datasets[0]))[:64];vectors={};maximum=0.;metrics=run.source.source.full.metrics.independent.METRICS
    for seed in run.SEEDS:
        for arm,key in (('baseline','baseline'),('joint_projection','labels')):
            vectors[f'{arm}_seed{seed}'],error=run.source.source.full.metrics.measured_vectors(bundle[seed][key],gt);maximum=max(maximum,error)
    groups={arm:np.stack([vectors[f'{arm}_seed{s}'] for s in run.SEEDS]) for arm in ('baseline','joint_projection')}
    delta=(groups['joint_projection']-groups['baseline']).mean(0)
    if delta.shape!=(64,3):raise ValueError('paired64 images from three fixed models required')
    draw=np.random.RandomState(225).randint(0,64,(10000,64));ci=np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
    changes={k:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,k in enumerate(metrics)}
    gate=bool(delta[:,0].mean()>=.01 and ci[0,0]>0);balanced=bool(gate and all(delta[:,i].mean()>=0 for i in (1,2)))
    summary=dict(status='complete_fixed223_secondary64_validation',scores={k:dict(zip(metrics,v.mean((0,1)).tolist())) for k,v in groups.items()},
        per_seed_scores={k:dict(zip(metrics,v.mean(0).tolist())) for k,v in vectors.items()},paired_joint_minus_baseline=changes,
        full320_candidate_gate=gate,FG_track_candidate_gate=gate,balanced_candidate_gate=balanced,optimizer_updates=0,
        image_count=64,seed_scope=list(run.SEEDS),fixed_checkpoint_experiment='SW0223',new_prediction_images_per_seed=48,reused_mini_images_per_seed=16,
        replay_images_per_seed=1,source16_gate_failure_preserved=True,independent_metric_maximum_abs_error=maximum,
        all_three_frozen64_protocols_validated_before_DEV_GT=True,mask_supervised_lineage=True,new70000_training=False,
        conditional_image_bootstrap_not_seed_population_CI=True,full320_incumbent_unchanged=True,fingerprint=run.fingerprint())
    run.write_json(run.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}));run.write_json(run.ARCHIVE/'score_v1/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(score()))
