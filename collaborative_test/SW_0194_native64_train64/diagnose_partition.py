"""Posthoc aggregates from frozen masks; no GPU, tuning or model changes."""
import json
import numpy as np
from . import train,evaluate
from collaborative_test.SW_0189_frozen_error_attribution.run import errors

REFERENCE='5b3fbfadd86dd4431f34bdb09b9b275bbdf6caa9a02ebb1e8d2fa135129cda13'
KEYS=('foreground_miss_rate','background_false_positive_rate','pair_split_rate',
      'pair_merge_rate','pair_merge_via_predicted_background_rate',
      'object_balanced_split_rate','object_pair_balanced_merge_rate')
PAIR_KEYS=KEYS[2:]


def paired_summary(before,after,keys):
    output={};rng=np.random.RandomState(194189)
    for key in keys:
        pairs=np.asarray([(a[key],b[key]) for a,b in zip(before,after)
                          if a[key] is not None and b[key] is not None],dtype=np.float64)
        if pairs.size==0:
            output[key]=dict(defined_images=0,before=None,after=None,delta=None,ci95=None);continue
        if not np.isfinite(pairs).all():raise ValueError('nonfinite diagnostic')
        delta=pairs[:,1]-pairs[:,0];draws=rng.randint(0,len(delta),size=(10000,len(delta)))
        output[key]=dict(defined_images=len(delta),before=float(pairs[:,0].mean()),after=float(pairs[:,1].mean()),
            delta=float(delta.mean()),ci95=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist())
    return output


def analyze_masks(before,after,gt):
    if before.shape!=after.shape or before.shape!=gt.shape or gt.ndim!=3 or len(gt)==0:
        raise ValueError('matching nonempty mask batches required')
    full_before=[];full_after=[];common_before=[];common_after=[];coverage=[]
    for a,b,g in zip(before,after,gt):
        full_before.append(errors(a,g));full_after.append(errors(b,g))
        common=(g>0)&(a>0)&(b>0)
        masked_gt=np.where(common,g,0)
        common_before.append(errors(np.where(common,a,0),masked_gt))
        common_after.append(errors(np.where(common,b,0),masked_gt))
        if (g>0).any():coverage.append(float(common.sum()/np.count_nonzero(g>0)))
    return dict(images=len(gt),full_GT_foreground=paired_summary(full_before,full_after,KEYS),
        common_retained_GT_foreground=paired_summary(common_before,common_after,PAIR_KEYS),
        common_retained_foreground_fraction_mean=float(np.mean(coverage)) if coverage else None,
        quantities_are_not_additive_error_shares=True,causal_claim=False)


def run():
    summary_path=train.ARCHIVE/'score_v1/summary.json'
    if train.sha(summary_path.read_bytes())!=REFERENCE:raise ValueError('fixed completed194 result changed')
    after,protocol=evaluate.validated_candidate()
    controls,ref=evaluate.screen_score.validated_pair()
    if protocol['reference_binding']!=ref:raise ValueError('diagnostic cohorts differ')
    gt=evaluate.screen_score.read_ground_truth(ref)
    if after.shape!=(32,64,64):raise ValueError('registered native64 development cohort required')
    report=analyze_masks(controls['candidate'],after,gt)
    from pathlib import Path
    import collaborative_test.SW_0189_frozen_error_attribution.run as attribution
    report.update(status='completed_frozen_native64_partition_diagnosis',scope='seed0 development32, TRAIN64 minus TRAIN8',
        candidate_summary_sha256=REFERENCE,source_error_operator_sha256=train.sha(Path(attribution.__file__).read_bytes()),
        analysis_source_sha256=train.sha(Path(__file__).read_bytes()),
        TRAIN8_protocol_sha256=train.sha((evaluate.screen.ARCHIVE/'candidate/protocol.json').read_bytes()),
        TRAIN64_protocol_sha256=train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes()),
        optimizer_updates=0,new_model_forward_passes=0,incumbent_changed=False,raw_imagewise_scores_public=False)
    path=train.ARCHIVE/'partition_diagnosis_v1/summary.json'
    if path.exists():
        if json.loads(path.read_bytes())!=report:raise ValueError('preserve different completed diagnostic')
    else:train.write_json(path,report)
    return report


if __name__=='__main__':print(json.dumps(run()),flush=True)
