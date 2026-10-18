"""Same original2px modal evaluation, after all three adapters freeze masks."""
import json
import numpy as np
from . import run


def validated_all():return {s:run.validated_seed(s) for s in run.SEEDS}


def score():
    bundle=validated_all();datasets=[bundle[s]['protocol']['binding']['parent']['dataset'] for s in run.SEEDS]
    if any(v!=datasets[0] for v in datasets):raise ValueError('different evaluation dataset')
    gt=run.source.source.full.read_gt(dict(dataset=datasets[0]))[list(run.OFFSETS)];vectors={};maximum=0.;metrics=run.source.source.full.metrics.independent.METRICS
    for seed in run.SEEDS:
        for arm,key in (('baseline','baseline'),('aligned','labels')):
            vectors[f'{arm}_seed{seed}'],error=run.source.source.full.metrics.measured_vectors(bundle[seed][key],gt);maximum=max(maximum,error)
    groups={arm:np.stack([vectors[f'{arm}_seed{s}'] for s in run.SEEDS]) for arm in ('baseline','aligned')}
    changes=run.reader.paired(groups['aligned'],groups['baseline']);fg=changes[metrics[0]]
    gate=bool(fg['mean']>=.01 and fg['ci95'][0]>0);balanced=bool(gate and all(changes[k]['mean']>=0 for k in metrics[1:]))
    summary=dict(status='complete_native64_feature_cell_alignment_mini',scores={k:dict(zip(metrics,v.mean((0,1)).tolist())) for k,v in groups.items()},
        per_seed_scores={k:dict(zip(metrics,v.mean(0).tolist())) for k,v in vectors.items()},paired_aligned_minus_baseline=changes,
        FG_track_candidate_gate=gate,balanced_candidate_gate=balanced,full320_candidate_gate=gate,optimizer_updates=0,geometry=run.alignment.geometry(),
        image_count=16,seed_scope=list(run.SEEDS),mask_supervised_source_lineage=True,new70000_training=False,RGB_GT_resize_or_crop=False,
        independent_metric_maximum_abs_error=maximum,all_three_frozen_protocols_validated_before_DEV_GT=True,full320_incumbent_unchanged=True,fingerprint=run.fingerprint())
    run.write_json(run.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}));run.write_json(run.ARCHIVE/'score_v1/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(score()))
