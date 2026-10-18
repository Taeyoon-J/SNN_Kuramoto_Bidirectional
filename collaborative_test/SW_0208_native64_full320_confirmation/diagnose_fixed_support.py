"""Posthoc GT-assisted attribution on already frozen/evaluated masks, no model."""
import json
from pathlib import Path
import numpy as np
from . import run as api
from collaborative_test.SW_0166_fixed_support_ari_diagnosis import diagnose as legacy

def analyze(prediction,gt):
    if prediction.shape!=gt.shape or prediction.shape[1:]!=(64,64):raise ValueError('same native64 population required')
    # A concrete GT-assisted positive partition, not a learned prediction.
    pure=np.where(prediction>0,gt+1,0).astype(np.int64)
    if not np.array_equal(pure>0,prediction>0):raise AssertionError('oracle changed fixed foreground')
    attached=np.stack([legacy._idealized_attachment(g,p) for p,g in zip(prediction,gt)])
    real,e1=api.metrics.measured_vectors(prediction,gt)
    pure_scores,e2=api.metrics.measured_vectors(pure,gt)
    repaired_scores,e3=api.metrics.measured_vectors(attached,gt)
    if not np.array_equal(real[:,1],pure_scores[:,1]):raise AssertionError('fixed-support foreground IoU changed')
    pairs=[legacy._pair_summary(g,p,p>0) for p,g in zip(prediction,gt)]
    miss=[r['missed_patch_count']/r['foreground_patch_count'] for r in pairs]
    def mean_defined(key):
        values=[r[key] for r in pairs if r[key] is not None]
        return float(np.mean(values)) if values else None
    return dict(images=len(gt),actual_FG_ARI=float(real[:,0].mean()),
        GT_pure_positive_partition_fixed_support_reference_FG_ARI=float(pure_scores[:,0].mean()),
        GT_missed_foreground_attachment_reference_FG_ARI=float(repaired_scores[:,0].mean()),
        actual_foreground_IoU=float(real[:,1].mean()),foreground_miss_fraction=float(np.mean(miss)),
        covered_same_object_split_rate=mean_defined('covered_split_rate'),
        covered_different_object_merge_rate=mean_defined('covered_merge_rate'),
        independent_maximum_metric_abs_error=max(e1,e2,e3),
        references_use_GT=True,references_are_not_predictors=True,
        references_are_not_a_proof_a_specific_classifier_can_reach_them=True)

def run():
    masks,binding=api.validated_all();gt=api.read_gt(binding)
    original_path=api.ARCHIVE/'score_v1/summary.json';original=json.loads(original_path.read_bytes())
    full=analyze(masks['warm192'],gt)
    if abs(full['actual_FG_ARI']-original['scores']['warm192']['fg_ari'])>1e-12:raise ValueError('different frozen full reference')
    offsets=list(range(0,32,2));mini={'0':analyze(masks['warm192'][offsets],gt[offsets])}
    from collaborative_test.SW_0205_native64_weak_seed_adaptation import run as other
    protocols={}
    for seed in (1,2):
        other.configure(seed);labels,protocol=other.validated_pair()
        if protocol['binding']['reference_binding']['dataset']!=binding['dataset']:raise ValueError('mini reference uses different canonical data')
        mini[str(seed)]=analyze(labels,gt[offsets])
        protocols[str(seed)]=api.sha((other.ARCHIVE/'candidate/protocol.json').read_bytes())
    result=dict(status='posthoc_GT_assisted_native64_fixed_support_attribution',
        warm192_seed0_full320=full,shared16_independent_TRAIN8_seeds=mini,
        full_source_summary_sha256=api.sha(original_path.read_bytes()),mini_source_protocol_sha256=protocols,
        source_sha256=api.sha(Path(__file__).read_bytes()),new_model_forward_passes=0,new_training_updates=0,
        does_not_modify_registered209_or210=True,not_a_new_leaderboard_result=True,
        reference_definition='GT-pure positive groups with original BG membership; separate GT-assisted missed-FG attachment, nonadditive diagnostics')
    api.write_json(api.ARCHIVE/'support_diagnosis_v1/summary.json',result);return result

if __name__=='__main__':print(json.dumps(run()))
