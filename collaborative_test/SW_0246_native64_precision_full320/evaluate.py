"""Full320 candidate/237/Slot comparison after every protocol is frozen."""
import json
import numpy as np
from . import run
METRICS=run.training.reference_score.METRICS


def validated_all():return {s:run.validated_seed(s) for s in run.SEEDS}


def references():
    ref=run.reference;raw=(ref.ARCHIVE/'score_v1/summary.json').read_bytes();audit=json.loads((ref.ARCHIVE/'completion_audit_v1/summary.json').read_bytes());score=json.loads(raw)
    if score['fingerprint']!=ref.fingerprint() or audit['score_sha256']!=run.sha(raw) or audit['raw_masks_independently_rescored']!=2880 or audit['independent_metrics_means_and_bootstrap_maximum_abs_error']>=1e-12:raise ValueError('audited full320237 benchmark required')
    return score


def verify_reference_means(groups,accepted):
    error=0.
    for arm,name in (('parent237','joint_projection'),('slot','slot')):
        error=max(error,float(np.abs(groups[arm].mean((0,1))-np.array([accepted['scores'][name][k] for k in METRICS])).max()))
    if error>=1e-12:raise ValueError('fixed full320237 or Slot metric contract changed')
    return error


def score():
    bundles=validated_all();slots=run.training.reference_score.load_slot_masks();accepted=references()
    datasets=[bundles[s]['protocol']['binding']['dataset'] for s in run.SEEDS];hashes=[bundles[s]['protocol']['binding']['RGB_sha256'] for s in run.SEEDS]
    if any(d!=datasets[0] for d in datasets) or len(set(hashes))!=1:raise ValueError('different full320 seed datasets')
    gt=run.reference.source.source.full.read_gt(dict(dataset=datasets[0]));vectors={k:[] for k in ('within16','parent237','slot')};maximum=0.
    for seed in run.SEEDS:
        for name,labels in (('within16',bundles[seed]['labels']),('parent237',bundles[seed]['baseline']),('slot',slots[seed]['labels64'])):
            value,error=run.reference.source.source.full.metrics.measured_vectors(labels,gt);vectors[name].append(value);maximum=max(maximum,float(error))
    vectors={k:np.stack(v) for k,v in vectors.items()}
    if any(v.shape!=(3,320,3) or not np.isfinite(v).all() for v in vectors.values()) or maximum>=1e-12:raise ValueError('complete finite independently scored full320 vectors required')
    reference_error=verify_reference_means(vectors,accepted);means={k:v.mean((0,1)) for k,v in vectors.items()};strict=bool(np.all(means['within16']>means['slot']))
    result=dict(status='complete_fixed245_native64_full320_confirmation',scores={k:dict(zip(METRICS,v.tolist())) for k,v in means.items()},
        per_seed={k:{str(s):dict(zip(METRICS,values[s].mean(0).tolist())) for s in run.SEEDS} for k,values in vectors.items()},
        candidate_minus_parent237=dict(zip(METRICS,(means['within16']-means['parent237']).tolist())),three_seed_strictly_exceeds_Slot_means=strict,
        native64_FG090_three_seed_goal_verified=bool(strict and means['within16'][0]>=.90),image_count=320,seed_scope=list(run.SEEDS),grid=[64,64],
        reused_reference_images_per_seed=64,new_prediction_images_per_seed=256,replay_images_per_seed=1,optimizer_updates=0,new70000_training=False,
        fixed_checkpoint_experiment='SW0245',selection_experiment='SW0245',matched_extra_training_control_only_evaluated_on64=True,
        mask_supervised_lineage=True,unsupervised_leaderboard_eligible=False,validation_selected_not_untouched_test=True,
        all_three_full_protocols_and_Slot_predictions_validated_before_DEV_GT=True,independent_metric_maximum_abs_error=maximum,
        unchanged_reference_metrics_maximum_abs_error=reference_error,fingerprint=run.fingerprint())
    run.write_json(run.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=result,vectors={k:v.tolist() for k,v in vectors.items()}));run.write_json(run.ARCHIVE/'score_v1/summary.json',result);return result
