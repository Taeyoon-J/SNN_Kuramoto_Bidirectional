"""Only infer the new TRAIN64 candidate; reuse the verified frozen controls."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import train
from collaborative_test.SW_0193_native64_probe_screen import run as screen,score as screen_score


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
    for offset,image_id in enumerate(screen.IDS):
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
    protocol=dict(bindings=prepared['bindings'],reference_binding=ref,image_ids=screen.IDS.tolist(),
        model_sha256=model_sha,labels_sha256=train.sha(np.stack(rows).tobytes()),
        training_report_sha256=train.sha((train.ARCHIVE/'training/report.json').read_bytes()),validation_GT_read=False)
    path=train.ARCHIVE/'candidate/protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen candidate protocol changed')
    else:train.write_json(path,protocol)


def validated_candidate():
    protocol=json.loads((train.ARCHIVE/'candidate/protocol.json').read_bytes())
    binding=protocol['bindings']
    if binding['fingerprint']!=train.fingerprint() or protocol['validation_GT_read'] is not False or protocol['image_ids']!=screen.IDS.tolist():raise ValueError('invalid native64 TRAIN64 protocol')
    report_path=train.ARCHIVE/'training/report.json'
    if train.sha(report_path.read_bytes())!=protocol['training_report_sha256']:raise ValueError('training report changed')
    report=json.loads(report_path.read_bytes())
    if report['bindings']!=binding or report['status']!='complete_native64_TRAIN64' or report['final_model_state_sha256']!=protocol['model_sha256']:raise ValueError('candidate training/model mismatch')
    parent=json.loads((train.probe.ARCHIVE/'probe_seed0/report.json').read_bytes())
    train.validate_history(report['history'],parent['history'])
    if (len(report['history'])!=train.TOTAL or report['inherited_proposals']!=2
        or report['new_proposals']!=train.TOTAL-2 or report['total_unique_TRAIN']!=64 or report['total_exposures']!=64):raise ValueError('candidate is not complete TRAIN64 training')
    accepted=sum(row['accepted'] for row in report['history'])
    if report['accepted_updates']!=accepted or report['actual_optimizer_steps']!=[accepted]:raise ValueError('candidate committed updates/Adam steps mismatch')
    rows=[]
    for offset in range(len(screen.IDS)):
        expected=row_binding(binding,protocol['model_sha256'],protocol['reference_binding'],offset)
        rows.append(screen.validate_saved_row(train.ARCHIVE/'candidate/rows'/f'{offset:04d}',expected))
    labels=np.stack(rows)
    if train.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('candidate array digest mismatch')
    return labels,protocol


def score_candidate():
    labels,protocol=validated_candidate()
    controls,ref=screen_score.validated_pair()
    if ref!=protocol['reference_binding']:raise ValueError('control cohort/model binding changed')
    for arm in screen.ARMS:
        if train.sha((screen.ARCHIVE/arm/'protocol.json').read_bytes())!=protocol['bindings']['reference_protocol_sha256'][arm]:raise ValueError('frozen control protocol changed')
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    slots={seed:screen_score.slot._load_slot(SLOT_ROOT,seed) for seed in (0,1,2)}
    for row in slots.values():
        if not np.array_equal(row['ids'][:len(screen.IDS)],screen.IDS):raise ValueError('Slot development IDs mismatch')
    gt=screen_score.read_ground_truth(ref)  # Candidate and all controls frozen first.
    predictions={'baseline':controls['baseline'],'TRAIN8':controls['candidate'],'TRAIN64':labels,
        **{'slot_seed'+str(seed):row['labels64'][:len(screen.IDS)] for seed,row in slots.items()}}
    vectors={};maximum=0.
    for name,pred in predictions.items():
        vectors[name],error=screen_score.measured_vectors(pred,gt);maximum=max(maximum,error)
    vectors['slot_three_seed_mean']=np.mean([vectors['slot_seed'+str(s)] for s in (0,1,2)],axis=0)
    metrics=screen_score.independent.METRICS
    scores={name:{metric:float(values[:,i].mean()) for i,metric in enumerate(metrics)} for name,values in vectors.items()}
    draws=np.random.RandomState(194).randint(0,len(screen.IDS),size=(10000,len(screen.IDS)))
    paired={}
    for control in ('baseline','TRAIN8'):
        delta=vectors['TRAIN64']-vectors[control];ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
        paired[control]={metric:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,metric in enumerate(metrics)}
    fg=paired['TRAIN8']['fg_ari']
    summary=dict(status='SW0194_native64_TRAIN64_mini_complete_independently_checked',
        scope='seed0 predetermined32 development images; full320/three-seed goal remains unverified',
        grid=[64,64],seed_scope=[0],development_images=len(screen.IDS),total_unique_TRAIN=64,
        inherited_proposals=2,new_proposals=14,total_proposals=16,mask_supervised=True,
        unsupervised_leaderboard_eligible=False,isolated_causal_data_scaling_claim=False,
        scores=scores,paired_TRAIN64_minus_control=paired,
        screen_gate_passed=fg['mean']>=.01 and fg['ci95'][0]>0,full320_three_seed_FG090_goal_achieved=False,
        independent_maximum_metric_abs_error=maximum,candidate_model_sha256=protocol['model_sha256'],
        training_report_sha256=protocol['training_report_sha256'],
        candidate_protocol_sha256=train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes()),
        bindings=protocol['bindings'],validation_GT_opened_after_candidate_and_controls_frozen=True,
        raw_imagewise_score_public=False)
    out=train.ARCHIVE/'score_v1'
    train.write_json(out/'evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(out/'summary.json',summary)
    return summary

if __name__=='__main__':print(json.dumps(score_candidate()),flush=True)
