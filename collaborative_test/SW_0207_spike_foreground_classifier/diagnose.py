"""Compare frozen mask topology and TRAIN/validation foreground separability."""
import json
from pathlib import Path
import numpy as np
from . import run as api
from collaborative_test.SW_0194_native64_train64.diagnose_partition import analyze_masks,paired_summary


def foreground_scores(prediction,target):
    p=np.asarray(prediction)>0;g=np.asarray(target)
    if p.shape!=g.shape or g.ndim!=2 or g.dtype.kind not in 'iu' or (g<0).any():raise ValueError('matching integer GT grid required')
    objects=np.unique(g[g>0])
    if not len(objects) or not (g==0).any():raise ValueError('both GT objects and BG required')
    recall=float(np.mean([p[g==c].mean() for c in objects]));background=float((~p[g==0]).mean())
    return dict(object_balanced_foreground_recall=recall,background_accuracy=background,class_balanced_foreground_accuracy=.5*(recall+background))


def run():
    api.configure(2);labels,protocol=api.validated_pair();binding=protocol['binding']
    summary=json.loads((api.ARCHIVE/'score_v1/summary.json').read_bytes())
    state=json.loads((api.ARCHIVE/'queue/state.json').read_bytes())
    if state['status']!='complete' or state['summary_sha256']!=api.sha((api.ARCHIVE/'score_v1/summary.json').read_bytes()):raise ValueError('completed fixed result required')
    gt=api.metrics.read_ground_truth(binding['reference_binding'])[list(api.OFFSETS)]
    old=[foreground_scores(p,g) for p,g in zip(labels['old'],gt)]
    new=[foreground_scores(p,g) for p,g in zip(labels['new'],gt)]
    report=analyze_masks(labels['old'],labels['new'],gt)
    train=json.loads((api.ARCHIVE/'classifier_v1/report.json').read_bytes())
    if train['final_parameter_sha256']!=protocol['classifier_sha256']:raise ValueError('frozen classifier changed')
    validation=float(np.mean([r['class_balanced_foreground_accuracy'] for r in new]))
    from collaborative_test.SW_0189_frozen_error_attribution import run as errors
    from collaborative_test.SW_0194_native64_train64 import diagnose_partition as paired
    report.update(status='frozen_node_foreground_classifier_generalization_and_partition_attribution',seed=2,grid=[64,64],
        validation_images=16,TRAIN_images=8,TRAIN_sampled_balanced_node_accuracy=train['balanced_node_accuracy'],
        validation_exact_class_balanced_foreground_accuracy=validation,
        TRAIN_minus_validation_balanced_accuracy=train['balanced_node_accuracy']-validation,
        validation_paired_classifier_scores=paired_summary(old,new,tuple(old[0])),
        accuracy_scope='TRAIN sampled object-uniform positives/BG, validation exact object-macro recall and BG accuracy; finite sampling also differs',
        candidate_summary_sha256=api.sha((api.ARCHIVE/'score_v1/summary.json').read_bytes()),
        analysis_source_sha256=api.sha(Path(__file__).read_bytes()),optimizer_updates=0,new_model_forward_passes=0,
        error_operator_sha256=api.sha(Path(errors.__file__).read_bytes()),paired_operator_sha256=api.sha(Path(paired.__file__).read_bytes()),
        raw_imagewise_scores_public=False,full320_three_seed_goal_verified=False)
    path=api.ARCHIVE/'diagnosis_v1/summary.json'
    if path.exists():
        if json.loads(path.read_bytes())!=report:raise ValueError('preserve frozen diagnosis')
    else:api.write_json(path,report)
    print(json.dumps(report));return report


if __name__=='__main__':run()
