"""Recover only CPU scoring; preserve frozen training and prediction source.

The original evaluator accidentally renamed the SW0204 control package when
renaming the candidate arm. All training and candidate masks already completed.
This separate module imports the real control explicitly and revalidates every
saved prediction before GT. It never runs an optimizer or a native forward.
"""
import argparse,json
from pathlib import Path
import numpy as np
from . import train,evaluate
from collaborative_test.SW_0204_native64_component_credit import train as control,evaluate as comparison


def file_sha(path):return train.sha(Path(path).read_bytes())


def validated_bundle():
    labels,protocol=evaluate.validated_candidate();binding=protocol['bindings']
    control_labels,control_protocol=comparison.validated_candidate()
    if (file_sha(control.ARCHIVE/'candidate/protocol.json')!=binding['control_protocol_sha256']
        or file_sha(control.ARCHIVE/'training/report.json')!=binding['control_report_sha256']
        or control_protocol['model_sha256']!=binding['control_model_sha256']
        or train.sha(control_labels.tobytes())!=binding['control_masks_sha256']):raise ValueError('frozen SW0204 control changed')
    controls,ref=evaluate.screen_score.validated_pair()
    if protocol['reference_binding']!=ref or control_protocol['reference_binding']!=ref:raise ValueError('different control cohort')
    for key in ('parent_boundary_sha256','parent_model_sha256','parent_optimizer_content_sha256','parent_RNG_content_sha256'):
        if binding[key]!=control_protocol['bindings'][key]:raise ValueError('different control model/Adam/RNG start')
    report=json.loads((train.ARCHIVE/'training/report.json').read_bytes())
    return dict(energy_ranking=labels,control=control_labels,warm=controls['candidate'][list(evaluate.OFFSETS)]),ref,report,binding,protocol


def score():
    predictions,ref,report,binding,protocol=validated_bundle()
    gt=evaluate.screen_score.read_ground_truth(ref)[list(evaluate.OFFSETS)]
    vectors={};maximum=0.
    for arm,pred in predictions.items():
        vectors[arm],error=evaluate.screen_score.measured_vectors(pred,gt);maximum=max(maximum,error)
    paired={k:evaluate.paired_change(vectors['energy_ranking'],vectors[k]) for k in ('control','warm')}
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors
    counts={name:[errors(p,g) for p,g in zip(pred,gt)] for name,pred in predictions.items()}
    count_summary={name:{key:float(np.mean([r[key] for r in rows])) for key in ('true_object_count','predicted_positive_label_count','raw_count_exact')} for name,rows in counts.items()}
    recovery=dict(reason='original evaluator control import was incorrectly renamed',
        original_evaluator_sha256=file_sha(Path(evaluate.__file__)),
        files={n:file_sha(Path(__file__).parent/n) for n in ('recover_score.py','test_recover_score.py')},
        original_queue_terminal_state_preserved=True,training_repeated=False,additional_core_forwards=0,additional_optimizer_updates=0)
    summary=dict(status='native64_finite_Adam_mini_complete',grid=[64,64],seed_scope=[0],images=16,
        scores={k:dict(zip(evaluate.screen_score.independent.METRICS,v.mean(0).tolist())) for k,v in vectors.items()},
        paired_energy_ranking_minus=paired,screen_gate_passed=evaluate.extension_gate(paired,report),
        candidate_ranking_parameter_credit_verified=report['ranking_parameter_credit_verified'],
        candidate_new_accepted_updates=report['new_accepted_updates'],candidate_CNN_changed=report['CNN_changed_from_warm'],
        candidate_new_applied_steps=report['new_accepted_updates'],candidate_new_descent_steps=report['new_descent_steps'],
        candidate_model_sha256=protocol['model_sha256'],candidate_protocol_sha256=file_sha(train.ARCHIVE/'candidate/protocol.json'),
        control_protocol_sha256=binding['control_protocol_sha256'],training_report_sha256=protocol['training_report_sha256'],
        independent_maximum_metric_abs_error=maximum,mask_supervised=True,unsupervised_leaderboard_eligible=False,
        live_tail_steps=64,control_live_tail_steps=64,inference_total_steps=1024,inference_readout_unchanged=True,
        training_update_policy='finite Adam without minibatch descent rejection',raw_mask_count_diagnostics=count_summary,
        raw_mask_count_is_not_a_trained_count_classifier=True,extra_read_only_TRAIN_gradient_admission_images=0,
        logical_training_proposal_exposures=16,full320_three_seed_goal_verified=False,raw_imagewise_values_included=False,
        score_recovery=recovery)
    train.write_json(train.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(train.ARCHIVE/'score_v1/summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    if args.admit:
        predictions,_,_,_,_=validated_bundle();print(json.dumps(dict(status='actual48_frozen_masks_and_correct_SW0204_control_admitted_before_GT',arms=list(predictions),additional_core_forwards=0,additional_optimizer_updates=0)))
    else:print(json.dumps(score()),flush=True)
