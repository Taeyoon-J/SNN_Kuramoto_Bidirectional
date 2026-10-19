"""CPU-only oracle error decomposition of accepted227 full320 masks."""
import ctypes,sys
if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP admission failed')
import json
from pathlib import Path
import numpy as np
from .transform import transform
from collaborative_test.SW_0227_native64_joint_projection_full320 import run as source,evaluate


def diagnose():
    a=source.ARCHIVE;score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw)
    audit=json.loads((a/'completion_audit_v1/summary.json').read_bytes())
    if audit['score_sha256']!=source.sha(score_raw) or score['fingerprint']!=source.fingerprint():raise ValueError('accepted227 score changed')
    bundle=evaluate.validated_all();datasets=[bundle[s]['protocol']['binding']['dataset'] for s in source.SEEDS]
    if any(d!=datasets[0] for d in datasets):raise ValueError('different datasets')
    for seed in source.SEEDS:
        if source.sha((source.archive_for(seed)/'protocol.json').read_bytes())!=audit['candidate_references'][str(seed)]['protocol_sha256']:raise ValueError('accepted227 protocol changed')
    # All actual checkpoints/protocols and all960 frozen masks validated before GT.
    gt=source.source.source.full.read_gt(dict(dataset=datasets[0]));metric=source.source.source.full.metrics.independent
    result={};error=0.
    for name,merge,cover in (('actual',False,False),('oracle_merge',True,False),('oracle_coverage',False,True),('oracle_both',True,True)):
        vectors=[];per_seed={}
        for seed in source.SEEDS:
            values=np.stack([metric.image_metrics(transform(p,t,merge=merge,cover=cover),t) for p,t in zip(bundle[seed]['labels'],gt)])
            if not np.isfinite(values).all():raise ValueError('nonfinite metric')
            if name=='actual':
                expected=[score['per_seed_scores'][f'joint_projection_seed{seed}'][k] for k in metric.METRICS]
                error=max(error,float(np.max(np.abs(values.mean(0)-expected))))
            vectors.append(values);per_seed[str(seed)]=dict(zip(metric.METRICS,values.mean(0).tolist()))
        result[name]=dict(mean=dict(zip(metric.METRICS,np.stack(vectors).mean((0,1)).tolist())),per_seed=per_seed)
    if error>1e-12:raise ValueError('actual227 score not reproduced')
    if abs(result['actual']['mean']['foreground_iou']-result['oracle_merge']['mean']['foreground_iou'])>1e-12:raise ValueError('merge changed foreground IoU')
    summary=dict(status='complete_GT_assisted_full320_diagnostic_not_model_result',arms=result,actual_metric_maximum_abs_error=error,
        accepted_score_sha256=source.sha(score_raw),seeds=[0,1,2],image_count=320,grid=[64,64],GT_used_to_transform_predictions=True,
        operations_are_not_proven_optimal_or_mathematical_upper_bounds=True,actual_best_model_unchanged=True,
        additional_training_updates=0,additional_model_forwards=0,peer_commit='087cd82',
        source_files={n:source.sha((Path(__file__).parent/n).read_bytes()) for n in ('run.py','transform.py','test_transform.py','README.md')})
    source.write_json(Path(__file__).parent/'results_archive/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(diagnose()))
