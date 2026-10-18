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
    if report['bindings']!=binding or report['status']!='complete_native64_long_credit' or report['final_model_state_sha256']!=protocol['model_sha256']:raise ValueError('candidate training/model mismatch')
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


def gradient_admission(prepared,device='cuda:0'):
    """One TRAIN image, no optimizer step; cap allocator, restore exact boundary."""
    import gc
    device=torch.device(device);torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.8,device)
    found=train.probe.model64.move_to_device(prepared['found'],device)
    runtime=train.probe.resume_policy.compatible_runtime(train.probe.checkpoint.runtime_signature(device))
    work=train.ARCHIVE/'training_work'
    train.restore_for_training(prepared,work,device,runtime)
    folder=train.ARCHIVE/'gradient_admission';path=folder/'passed.json'
    if path.exists():
        prior=json.loads(path.read_bytes())
        if prior['bindings']!=prepared['bindings']:raise ValueError('gradient admission lineage changed')
        return True
    if (folder/'rejected.json').exists():return False
    modules=dict(wrapped=found.wrapped,encoder=found.encoder);optimizer=prepared['optimizer']
    snapshot=train.probe.tx.capture_transaction(modules,optimizer,offload_to_cpu=True)
    before_model=train.probe.native._state_sha(found.wrapped,found.encoder)
    torch.cuda.reset_peak_memory_stats(device);failure=None;facts={}
    try:
        torch.manual_seed(192002);torch.cuda.manual_seed(192002)
        pair_rng=train.probe.tx.capture_rng_state()
        before,_=train.batch_loss(prepared,[8],device,live=False)
        train.probe.tx.restore_rng_state(pair_rng)
        measured,emitted=train.batch_loss(prepared,[8],device,live=True,credit=True)
        gradients={name:float(sum(float(p.grad.detach().double().square().sum().cpu())
            for _,p in entries if p.grad is not None)**.5) for name,entries in prepared['families'].items()}
        tolerance=max(1e-7,1e-6*abs(before))
        facts=dict(before_loss=before,gradient_forward_loss=measured,parity_error=abs(measured-before),
            parity_tolerance=tolerance,emitted_S_gradient_l2=emitted,family_gradient_norms=gradients)
        if not np.isfinite(list(gradients.values())+[before,measured,emitted]).all():failure='nonfinite credit'
        elif abs(measured-before)>=tolerance:failure='no-grad/live forward parity failed'
        elif emitted<=0 or any(v<=0 for v in gradients.values()):failure='missing actual emitted-S/family credit'
    except torch.cuda.OutOfMemoryError:
        failure='bounded allocator admission rejected full512 time credit'
    finally:
        peak=torch.cuda.max_memory_allocated(device)
        train.probe.tx.restore_transaction(snapshot,modules,optimizer,restore_rng=True,clear_grad=True)
        gc.collect();torch.cuda.empty_cache()
    if train.probe.native._state_sha(found.wrapped,found.encoder)!=before_model:raise RuntimeError('read-only admission changed weights')
    if train.content_sha(optimizer.state_dict())!=train.content_sha(snapshot['optimizer']):raise RuntimeError('admission changed Adam')
    result=dict(status='passed_full512_credit' if failure is None else 'rejected_full512_credit',
        failure=failure,bindings=prepared['bindings'],live_tail_steps=train.LIVE_TAIL,
        no_grad_prefix_steps=1024-train.LIVE_TAIL,settled_score_steps=512,images=1,TRAIN_indices=[8],
        optimizer_updates=0,allocator_fraction=.8,peak_allocated_bytes=peak,
        model_Adam_RNG_restored=True,extra_read_only_gradient_admission=True,**facts)
    def finite_json(value):
        if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,list):return [finite_json(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    result=finite_json(result)
    train.write_json(path if failure is None else folder/'rejected.json',result)
    return failure is None


def score_candidate():
    labels,protocol=validated_candidate();binding=protocol['bindings']
    from collaborative_test.SW_0199_native64_absolute_calibration import train as control,evaluate as comparison
    control.configure('control');control_labels,control_protocol=comparison.validated_candidate()
    if (train.sha((control.ARCHIVE/'candidate/protocol.json').read_bytes())!=binding['control_protocol_sha256']
        or train.sha((control.ARCHIVE/'training/report.json').read_bytes())!=binding['control_report_sha256']
        or control_protocol['model_sha256']!=binding['control_model_sha256']
        or train.sha(control_labels.tobytes())!=binding['control_masks_sha256']):raise ValueError('frozen control changed')
    controls,ref=screen_score.validated_pair()
    if protocol['reference_binding']!=ref or control_protocol['reference_binding']!=ref:raise ValueError('different matched cohorts')
    for key in ('parent_boundary_sha256','parent_model_sha256','parent_optimizer_content_sha256','parent_RNG_content_sha256'):
        if binding[key]!=control_protocol['bindings'][key]:raise ValueError('different model/Adam/RNG starts')
    admission=json.loads((train.ARCHIVE/'gradient_admission/passed.json').read_bytes())
    if admission['bindings']!=binding or admission['live_tail_steps']!=512:raise ValueError('full-window admission missing')
    gt=screen_score.read_ground_truth(ref)[list(OFFSETS)]
    predictions=dict(long_credit=labels,control=control_labels,warm=controls['candidate'][list(OFFSETS)])
    vectors={};maximum=0.
    for arm,pred in predictions.items():
        vectors[arm],error=screen_score.measured_vectors(pred,gt);maximum=max(maximum,error)
    paired={k:paired_change(vectors['long_credit'],vectors[k]) for k in ('control','warm')}
    report=json.loads((train.ARCHIVE/'training/report.json').read_bytes())
    summary=dict(status='native64_full_settled_window_credit_mini_complete',grid=[64,64],seed_scope=[0],images=16,
        scores={k:dict(zip(screen_score.independent.METRICS,v.mean(0).tolist())) for k,v in vectors.items()},
        paired_long_credit_minus=paired,screen_gate_passed=extension_gate(paired,report),
        candidate_new_accepted_updates=report['new_accepted_updates'],candidate_CNN_changed=report['CNN_changed_from_warm'],
        candidate_model_sha256=protocol['model_sha256'],candidate_protocol_sha256=train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes()),
        control_protocol_sha256=binding['control_protocol_sha256'],training_report_sha256=protocol['training_report_sha256'],
        independent_maximum_metric_abs_error=maximum,mask_supervised=True,unsupervised_leaderboard_eligible=False,
        live_tail_steps=512,control_live_tail_steps=64,inference_total_steps=1024,inference_readout_unchanged=True,
        extra_read_only_TRAIN_gradient_admission_images=1,logical_training_proposal_exposures=16,
        full320_three_seed_goal_verified=False,raw_imagewise_values_included=False)
    train.write_json(train.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(train.ARCHIVE/'score_v1/summary.json',summary);return summary
