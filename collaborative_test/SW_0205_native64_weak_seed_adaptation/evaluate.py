"""Freeze native64 candidate masks before GT; pair each weak seed with itself."""
import io,json,os,time
import numpy as np
import torch
from . import run as api
from collaborative_test.SW_0193_native64_probe_screen import run as screen
OFFSETS=tuple(range(0,32,2))


def row_binding(binding,model,index):
    offset=OFFSETS[index]
    return dict(seed=api.SEED,image_id=int(screen.IDS[offset]),inference_seed=193000+offset,
        model_sha256=model,binding_sha256=api.sha(json.dumps(binding,sort_keys=True).encode()),
        RGB_sha256=binding['reference_binding']['development_RGB_row_sha256'][offset])


def predict_candidate(prepared,report,device):
    binding=prepared['bindings'];found=prepared['found']
    from collaborative_test.SW_0130_phase_state_integration import run as data
    rgb=np.load(data.VAL_RGB,mmap_mode='r',allow_pickle=False)[:32].copy()
    if api.sha(rgb.tobytes())!=binding['reference_binding']['development_RGB_sha256']:raise ValueError('development RGB changed')
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    masks=[]
    for index,offset in enumerate(OFFSETS):
        expected=row_binding(binding,report['final_model_state_sha256'],index)
        folder=api.ARCHIVE/'candidate/rows'/f'{index:04d}'
        if folder.exists():labels=screen.validate_saved_row(folder,expected)
        else:
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
            with torch.no_grad():
                image=torch.from_numpy(rgb[offset:offset+1]).to(device)
                gamma,trace,adj,z,q,rho=api.native._native_forward(found,image,live_tail_steps=0)
                labels=screen.prediction_from_trace(found,trace,adj,q)
            stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            with (stage/'labels.npz').open('xb') as stream:
                np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
            api.write_json(stage/'row.json',dict(binding=expected,prediction_sha256=api.sha((stage/'labels.npz').read_bytes())))
            stage.rename(folder);del image,gamma,trace,adj,z,q,rho
        masks.append(labels)
    if api.native._state_sha(found.wrapped,found.encoder)!=report['final_model_state_sha256']:raise ValueError('inference mutated trained model')
    protocol=dict(binding=binding,labels_sha256=api.sha(np.stack(masks).tobytes()),
        model_sha256=report['final_model_state_sha256'],training_report_sha256=api.sha((api.ARCHIVE/'training/report.json').read_bytes()),
        validation_GT_used_for_prediction=False,image_ids=screen.IDS[list(OFFSETS)].tolist())
    api.write_json(api.ARCHIVE/'candidate/protocol.json',protocol)


def validated_candidate():
    protocol=json.loads((api.ARCHIVE/'candidate/protocol.json').read_bytes());binding=protocol['binding']
    if binding['fingerprint']!=api.fingerprint() or protocol['validation_GT_used_for_prediction'] is not False:raise ValueError('candidate protocol changed')
    if protocol['image_ids']!=screen.IDS[list(OFFSETS)].tolist():raise ValueError('fixed sixteen IDs changed')
    raw=(api.ARCHIVE/'training/report.json').read_bytes()
    if api.sha(raw)!=protocol['training_report_sha256']:raise ValueError('trained report changed')
    report=api.validate_report(json.loads(raw),binding)
    if report['final_model_state_sha256']!=protocol['model_sha256']:raise ValueError('trained model changed')
    labels=np.stack([screen.validate_saved_row(api.ARCHIVE/'candidate/rows'/f'{i:04d}',row_binding(binding,protocol['model_sha256'],i)) for i in range(16)])
    if api.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('candidate masks changed')
    return labels,protocol


def score_candidate():
    labels,protocol=validated_candidate();binding=protocol['binding']
    original,source=api.baseline.validated_pair(api.SEED)
    if (api.sha(original.tobytes())!=binding['baseline_masks_sha256']
        or source['model_sha256']!=binding['baseline_model_sha256']
        or source['reference_binding']!=binding['reference_binding']
        or api.sha((api.baseline.archive_for(api.SEED)/'prediction/protocol.json').read_bytes())!=binding['baseline_protocol_sha256']):raise ValueError('independent baseline changed')
    gt=api.metrics.read_ground_truth(binding['reference_binding'])[list(OFFSETS)]
    vectors={};maximum=0.
    for name,pred in [('baseline',original[list(OFFSETS)]),('candidate',labels)]:
        vectors[name],error=api.metrics.measured_vectors(pred,gt);maximum=max(maximum,error)
    delta=vectors['candidate']-vectors['baseline'];draws=np.random.RandomState(205).randint(0,16,(10000,16))
    ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    names=api.metrics.independent.METRICS
    paired={m:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,m in enumerate(names)}
    report=json.loads((api.ARCHIVE/'training/report.json').read_bytes())
    complete=len(report['history'])==2 and not any(r['credit_failures'] for r in report['history'])
    gate=bool(complete and report['accepted_updates']>0 and report['source_CNN_changed']
        and paired['fg_ari']['mean']>=.01 and paired['fg_ari']['ci95'][0]>0
        and paired['foreground_iou']['mean']>=0 and paired['matched_object_iou']['mean']>=0)
    summary=dict(status='completed_independent_weak_seed_TRAIN8_screen',seed=api.SEED,images=16,grid=[64,64],
        scores={name:dict(zip(names,v.mean(0).tolist())) for name,v in vectors.items()},paired_candidate_minus_baseline=paired,
        extension_gate_passed=gate,training_proposals_completed=len(report['history']),actual_Adam_steps=report['actual_optimizer_steps'],
        accepted_updates=report['accepted_updates'],source_CNN_changed=report['source_CNN_changed'],
        extra_unique_TRAIN_proposal_images=4*len(report['history']),mask_supervised=True,
        control='same independent native32 incumbent mapped64; no additional64 adaptation',
        candidate_model_sha256=protocol['model_sha256'],training_report_sha256=protocol['training_report_sha256'],
        independent_maximum_metric_abs_error=maximum,full320_three_seed_goal_verified=False,unsupervised_leaderboard_eligible=False)
    api.write_json(api.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    api.write_json(api.ARCHIVE/'score_v1/summary.json',summary)
    return summary
