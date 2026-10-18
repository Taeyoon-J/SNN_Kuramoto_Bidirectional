"""Only infer the new TRAIN64 candidate; reuse the verified frozen controls."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import train
from collaborative_test.SW_0193_native64_probe_screen import run as screen,score as screen_score

OFFSETS=tuple(range(0,32,2))

def row_binding(bindings,model_sha,ref_binding,offset):
    return dict(image_id=int(screen.IDS[offset]),inference_seed=193000+offset,
        RGB_sha256=ref_binding['development_RGB_row_sha256'][offset],model_sha256=model_sha,
        training_bindings_sha256=train.sha(json.dumps(bindings,sort_keys=True).encode()))


def predict_candidate(prepared,report,device='cuda:0'):
    from collaborative_test.SW_0130_phase_state_integration import run as source
    found=prepared['found'];device=torch.device(device)
    if train.probe.native._state_sha(found.wrapped,found.encoder)!=report['final_model_state_sha256']:raise ValueError('candidate model differs from completed training')
    ref=prepared['reference_binding']
    rgb=np.load(source.VAL_RGB,mmap_mode='r',allow_pickle=False)[:len(screen.IDS)].copy()
    if train.sha(rgb.tobytes())!=ref['development_RGB_sha256']:raise ValueError('fixed development RGB changed')
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False)
    found.wrapped.eval();found.encoder.eval()
    model_sha=report['final_model_state_sha256'];rows=[]
    for offset in OFFSETS:
        expected=row_binding(prepared['bindings'],model_sha,ref,offset)
        folder=train.ARCHIVE/'candidate/rows'/f'{offset:04d}'
        if folder.exists():labels=screen.validate_saved_row(folder,expected)
        else:
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
            started=time.perf_counter()
            with torch.no_grad():
                image=torch.from_numpy(rgb[offset:offset+1]).to(device)
                _gamma,trace,adj,_z,q,_rho=train.probe.native._native_forward(found,image,live_tail_steps=0)
                labels=screen.prediction_from_trace(found,trace,adj,q)
            stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
            with (stage/'labels.npz').open('xb') as stream:
                np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
            train.write_json(stage/'row.json',dict(binding=expected,prediction_sha256=train.sha((stage/'labels.npz').read_bytes()),seconds=time.perf_counter()-started))
            stage.rename(folder)
            del image,trace,adj,q,_gamma,_z,_rho
        rows.append(labels)
    if train.probe.native._state_sha(found.wrapped,found.encoder)!=model_sha:raise RuntimeError('candidate inference mutated trained model')
    protocol=dict(bindings=prepared['bindings'],reference_binding=ref,image_ids=screen.IDS[list(OFFSETS)].tolist(),
        model_sha256=model_sha,labels_sha256=train.sha(np.stack(rows).tobytes()),
        training_report_sha256=train.sha((train.ARCHIVE/'training/report.json').read_bytes()),validation_GT_read=False)
    path=train.ARCHIVE/'candidate/protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen candidate protocol changed')
    else:train.write_json(path,protocol)


def validated_candidate():
    protocol=json.loads((train.ARCHIVE/'candidate/protocol.json').read_bytes())
    binding=protocol['bindings']
    if binding['fingerprint']!=train.fingerprint() or protocol['validation_GT_read'] is not False or protocol['image_ids']!=screen.IDS[list(OFFSETS)].tolist():raise ValueError('invalid native64 TRAIN64 protocol')
    report_path=train.ARCHIVE/'training/report.json'
    if train.sha(report_path.read_bytes())!=protocol['training_report_sha256']:raise ValueError('training report changed')
    report=json.loads(report_path.read_bytes())
    if report['bindings']!=binding or report['status']!='complete_native64_finite_Adam' or report['final_model_state_sha256']!=protocol['model_sha256']:raise ValueError('candidate training/model mismatch')
    parent=json.loads((train.probe.ARCHIVE/'probe_seed0/report.json').read_bytes())
    train.validate_history(report['history'],parent['history'])
    if (len(report['history'])!=train.TOTAL or report['inherited_proposals']!=2
        or report['new_proposals']!=train.TOTAL-2 or report['total_unique_TRAIN']!=16 or report['total_exposures']!=16):raise ValueError('candidate is not complete TRAIN64 training')
    accepted=sum(row['accepted'] for row in report['history'])
    if report['accepted_updates']!=accepted or report['actual_optimizer_steps']!=[accepted]:raise ValueError('candidate committed updates/Adam steps mismatch')
    rows=[]
    for offset in OFFSETS:
        expected=row_binding(binding,protocol['model_sha256'],protocol['reference_binding'],offset)
        rows.append(screen.validate_saved_row(train.ARCHIVE/'candidate/rows'/f'{offset:04d}',expected))
    labels=np.stack(rows)
    if train.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('candidate array digest mismatch')
    return labels,protocol


def paired_change(candidate,control):
    delta=np.asarray(candidate,dtype=np.float64)-np.asarray(control,dtype=np.float64)
    if delta.shape!=(16,3) or not np.isfinite(delta).all():raise ValueError('fixed16 finite metric vectors required')
    draws=np.random.RandomState(199).randint(0,16,size=(10000,16))
    ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    return {key:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist())
        for i,key in enumerate(screen_score.independent.METRICS)}


def extension_gate(paired,candidate):
    versus_control=paired['control'];versus_warm=paired['warm']
    return bool(versus_control['fg_ari']['mean']>=.01 and versus_control['fg_ari']['ci95'][0]>0
        and versus_warm['fg_ari']['mean']>=0
        and all(v['matched_object_iou']['mean']>=0 for v in paired.values())
        and candidate['new_accepted_updates']>=1 and candidate['committed_model_changed']
        and candidate['CNN_changed_from_warm'])


def score_candidate():
    labels,protocol=validated_candidate();binding=protocol['bindings']
    from collaborative_test.SW_0202_native64_finite_Adam import train as control,evaluate as comparison
    control_labels,control_protocol=comparison.validated_candidate()
    if (train.sha((control.ARCHIVE/'candidate/protocol.json').read_bytes())!=binding['control_protocol_sha256']
        or train.sha((control.ARCHIVE/'training/report.json').read_bytes())!=binding['control_report_sha256']
        or control_protocol['model_sha256']!=binding['control_model_sha256']
        or train.sha(control_labels.tobytes())!=binding['control_masks_sha256']):raise ValueError('frozen control changed')
    controls,ref=screen_score.validated_pair()
    if protocol['reference_binding']!=ref or control_protocol['reference_binding']!=ref:raise ValueError('different matched cohorts')
    for key in ('parent_boundary_sha256','parent_model_sha256','parent_optimizer_content_sha256','parent_RNG_content_sha256'):
        if binding[key]!=control_protocol['bindings'][key]:raise ValueError('different model/Adam/RNG starts')
    gt=screen_score.read_ground_truth(ref)[list(OFFSETS)]
    predictions=dict(component_credit=labels,control=control_labels,warm=controls['candidate'][list(OFFSETS)])
    vectors={};maximum=0.
    for arm,pred in predictions.items():
        vectors[arm],error=screen_score.measured_vectors(pred,gt);maximum=max(maximum,error)
    paired={k:paired_change(vectors['component_credit'],vectors[k]) for k in ('control','warm')}
    report=json.loads((train.ARCHIVE/'training/report.json').read_bytes())
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors
    counts={name:[errors(p,g) for p,g in zip(pred,gt)] for name,pred in predictions.items()}
    count_summary={name:{key:float(np.mean([r[key] for r in rows])) for key in ('true_object_count','predicted_positive_label_count','raw_count_exact')} for name,rows in counts.items()}
    summary=dict(status='native64_finite_Adam_mini_complete',grid=[64,64],seed_scope=[0],images=16,
        scores={k:dict(zip(screen_score.independent.METRICS,v.mean(0).tolist())) for k,v in vectors.items()},
        paired_component_credit_minus=paired,screen_gate_passed=extension_gate(paired,report),
        candidate_new_accepted_updates=report['new_accepted_updates'],candidate_CNN_changed=report['CNN_changed_from_warm'],
        candidate_new_applied_steps=report['new_accepted_updates'],candidate_new_descent_steps=report['new_descent_steps'],
        candidate_model_sha256=protocol['model_sha256'],candidate_protocol_sha256=train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes()),
        control_protocol_sha256=binding['control_protocol_sha256'],training_report_sha256=protocol['training_report_sha256'],
        independent_maximum_metric_abs_error=maximum,mask_supervised=True,unsupervised_leaderboard_eligible=False,
        live_tail_steps=64,control_live_tail_steps=64,inference_total_steps=1024,inference_readout_unchanged=True,training_update_policy='finite Adam without minibatch descent rejection',
        raw_mask_count_diagnostics=count_summary,raw_mask_count_is_not_a_trained_count_classifier=True,
        extra_read_only_TRAIN_gradient_admission_images=0,logical_training_proposal_exposures=16,
        full320_three_seed_goal_verified=False,raw_imagewise_values_included=False)
    train.write_json(train.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(train.ARCHIVE/'score_v1/summary.json',summary);return summary
