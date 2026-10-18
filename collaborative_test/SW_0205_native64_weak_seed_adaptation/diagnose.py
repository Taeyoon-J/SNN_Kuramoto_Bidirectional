"""Posthoc attribution on frozen before/after masks; no model forward or update."""
import argparse,json
from pathlib import Path
import numpy as np
from . import run as api,evaluate
from collaborative_test.SW_0189_frozen_error_attribution.run import errors
from collaborative_test.SW_0194_native64_train64.diagnose_partition import analyze_masks,paired_summary


def run(seed):
    api.configure(seed)
    candidate,protocol=evaluate.validated_candidate();binding=protocol['binding']
    state=json.loads((api.ARCHIVE/'queue/state.json').read_bytes())
    score=api.ARCHIVE/'score_v1/summary.json'
    if state['status']!='complete' or state['summary_sha256']!=api.sha(score.read_bytes()):raise ValueError('completed fixed result required')
    baseline,source=api.baseline.validated_pair(seed)
    if (source['reference_binding']!=binding['reference_binding'] or source['model_sha256']!=binding['baseline_model_sha256']
        or api.sha(baseline.tobytes())!=binding['baseline_masks_sha256']):raise ValueError('source baseline changed')
    baseline=baseline[list(evaluate.OFFSETS)]
    gt=api.metrics.read_ground_truth(binding['reference_binding'])[list(evaluate.OFFSETS)]
    report=analyze_masks(baseline,candidate,gt)
    before=[errors(p,g) for p,g in zip(baseline,gt)];after=[errors(p,g) for p,g in zip(candidate,gt)]
    from collaborative_test.SW_0189_frozen_error_attribution import run as operator
    from collaborative_test.SW_0194_native64_train64 import diagnose_partition as paired
    report.update(status='completed_frozen_weak_seed_adaptation_attribution',seed=seed,grid=[64,64],
        scope='same16 development images; short TRAIN8 candidate minus independent native64 baseline',
        count_and_precision=paired_summary(before,after,('true_object_count','predicted_positive_label_count','raw_count_exact','foreground_precision')),
        source_error_operator_sha256=api.sha(Path(operator.__file__).read_bytes()),
        source_paired_operator_sha256=api.sha(Path(paired.__file__).read_bytes()),
        analysis_source_sha256=api.sha(Path(__file__).read_bytes()),candidate_summary_sha256=state['summary_sha256'],
        candidate_protocol_sha256=api.sha((api.ARCHIVE/'candidate/protocol.json').read_bytes()),
        baseline_protocol_sha256=binding['baseline_protocol_sha256'],optimizer_updates=0,new_model_forward_passes=0,
        raw_label_count_is_not_trained_count_classifier=True,raw_imagewise_scores_public=False,full320_three_seed_goal_verified=False)
    if api.torch.cuda.is_initialized():raise RuntimeError('CPU-only diagnostic unexpectedly initialized CUDA')
    path=api.ARCHIVE/'partition_diagnosis_v1/summary.json'
    if path.exists():
        if json.loads(path.read_bytes())!=report:raise ValueError('preserve frozen attribution')
    else:api.write_json(path,report)
    print(json.dumps(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(1,2),required=True)
    run(p.parse_args().seed)
