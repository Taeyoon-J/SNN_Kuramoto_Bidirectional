"""Post-score audit and source-frequency diagnostics without another model run.

The frozen scorer already reconstructs all three actual full models. This
separate audit checks encoder payloads, teachers, all saved masks/frequencies,
independent metrics and paired CIs. It does not repeat full-model factories.
"""
import io,json,ctypes,sys
if sys.platform == "linux":
    if ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:raise RuntimeError("per-process THP disable failed")
from pathlib import Path
import numpy as np
import torch
from . import train,evaluate,projection
from collaborative_test.SW_0221_native64_frequency_pretraining import diagnose_frequency as diagnostic


def validate_finite_json(value):
    if isinstance(value,dict):
        for item in value.values():validate_finite_json(item)
    elif isinstance(value,list):
        for item in value:validate_finite_json(item)
    elif isinstance(value,float) and not np.isfinite(value):
        raise ValueError('nonfinite audit input')


def check_statistics(vectors,score):
    validate_finite_json(score)
    groups={arm:np.stack([vectors[f'{arm}_seed{s}'] for s in train.SEEDS]) for arm in ('baseline','center1','relative_separation')}
    if any(v.shape!=(3,16,3) or not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('finite three-seed fixed16 vectors required')
    maximum=0.
    for name,value in vectors.items():maximum=max(maximum,float(np.max(np.abs(value.mean(0)-[score['per_seed_scores'][name][k] for k in evaluate.METRICS]))))
    for name,value in groups.items():maximum=max(maximum,float(np.max(np.abs(value.mean((0,1))-[score['scores'][name][k] for k in evaluate.METRICS]))))
    draw=np.random.RandomState(221).randint(0,16,(10000,16));changes={}
    for control_name in ('baseline','center1'):
        delta=(groups['relative_separation']-groups[control_name]).mean(0);ci=np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
        changes[control_name]=(delta,ci)
        for i,key in enumerate(evaluate.METRICS):
            expected=score['paired_joint_minus'][control_name][key];maximum=max(maximum,abs(expected['mean']-float(delta[:,i].mean())),float(np.max(np.abs(expected['ci95']-ci[:,i]))))
    delta,ci=changes['baseline'];gate=bool(delta[:,0].mean()>=.01 and ci[0,0]>0 and groups['relative_separation'][:,:,0].mean()>groups['center1'][:,:,0].mean())
    balanced=bool(gate and all(delta[:,i].mean()>=0 for i in (1,2)))
    if maximum>1e-12 or gate!=score['FG_track_candidate_gate'] or gate!=score['full320_candidate_gate'] or balanced!=score['balanced_candidate_gate']:
        raise ValueError('metric/CI/candidate gates mismatch')
    return maximum,gate,balanced


def run():
    a=train.ARCHIVE;score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw);private=json.loads((a/'score_v1/evaluation_private.json').read_bytes())
    if score['fingerprint']!=train.fingerprint() or private['summary']!=score or score['status']!='complete_native64_source_relative_separation_mini':raise ValueError('complete frozen three-model score required')
    validate_finite_json(score);validate_finite_json(private)
    if score['images']!=16 or score['grid']!=[64,64] or score['seed_scope']!=[0,1,2] or train.CENTER_WEIGHT!=1. or train.SEPARATION_WEIGHT!=1.:
        raise ValueError('wrong relative_separation mini scope')
    models={};dataset=None
    for seed in train.SEEDS:
        p=train.archive_for(seed,64);report=json.loads((p/'training_report.json').read_bytes());protocol=json.loads((p/'protocol.json').read_bytes());admission=json.loads((p/'admission.json').read_bytes());initial=json.loads((p/'initial.json').read_bytes())
        binding=report['binding'];state=json.loads((p/'queue/state.json').read_bytes())
        if binding!=admission['binding'] or binding['fingerprint']!=train.fingerprint() or binding['parent_model_sha256']!=train.source.MODELS[str(seed)]:raise ValueError('wrong admitted source')
        if report['other_core_updates']!=0 or report['core_projection_updates']!=128 or not report['projection_changed'] or report['actual_Adam_steps']!=[128] or not report['encoder_changed'] or report['DEV_GT_read'] is not False:raise ValueError('wrong encoder-only update protocol')
        if protocol['binding']!=dict(training_report_sha256=train.sha((p/'training_report.json').read_bytes()),binding=binding,model_sha256=report['final_model_sha256']) or protocol['DEV_GT_read'] is not False or protocol['image_ids']!=[1320+i for i in evaluate.OFFSETS] or protocol['total_steps']!=1024 or protocol['settle_steps']!=512:raise ValueError('wrong frozen prediction protocol')
        if state['status']!='complete' or state['protocol_sha256']!=train.sha((p/'protocol.json').read_bytes()):raise ValueError('queue not complete')
        train.checkpoint.inspect_committed_metadata(p/'encoder_work');raw=(p/'encoder_work/boundary.ptx').read_bytes()
        outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);payload=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
        expected_checkpoint_binding=dict(binding=binding,initial_metadata_sha256=train.sha((p/'initial.json').read_bytes()))
        if train.sha(raw)!=report['checkpoint_file_sha256'] or payload['completed_updates']!=128 or set(payload['wrapped_state'])!={'weight'} or payload['wrapped_state']['weight'].shape!=(4,8) or payload['history']!=report['history'] or payload['bindings']!=expected_checkpoint_binding or report['checkpoint_binding']!=expected_checkpoint_binding:raise ValueError('wrong actual committed encoder state')
        if train.state_sha(payload['encoder_state'])!=report['final_encoder_sha256'] or sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})!=[128]:raise ValueError('actual encoder/Adam mismatch')
        if train.state_sha(payload['wrapped_state'])!=report['final_projection_sha256'] or report['final_projection_sha256']==binding['initial_projection_sha256'] or report['history'][0]['projection_gradient_norm']<=0:raise ValueError('actual learned projection/gradient proof failed')
        if any(row['indices']!=train.batch_indices(seed,64,i) or row['update']!=i+1 for i,row in enumerate(payload['history'])):raise ValueError('TRAIN order changed')
        teacher_raw=(p/'teacher.npz').read_bytes()
        if train.sha(teacher_raw)!=report['teacher_file_sha256'] or report['teacher_file_sha256']!=initial['teacher_file_sha256'] or report['teacher_file_sha256']!=admission['teacher_file_sha256']:raise ValueError('teacher file changed')
        with np.load(io.BytesIO(teacher_raw),allow_pickle=False) as values:teacher={k:torch.from_numpy(values[k].copy()) for k in values.files}
        if train.state_sha(teacher)!=binding['teacher_arrays_sha256']:raise ValueError('teacher content changed')
        if set(teacher)!=set(train.BASE_KEYS)|{'class_labels'} or train.state_sha({k:teacher[k] for k in train.BASE_KEYS})!=binding['base_teacher_arrays_sha256']:
            raise ValueError('source center/weight/gamma teacher changed')
        class_labels=teacher['class_labels']
        if class_labels.dtype!=torch.int64 or tuple(class_labels.shape)!=(64,4096) or bool((class_labels<0).any()):
            raise ValueError('actual TRAIN class teacher invalid')
        for losses in (report['initial_TRAIN_phase_loss'],report['final_TRAIN_phase_loss']):
            expected=losses['within_object_variance']+losses['source_center_MSE']
            total=losses['frequency_denoising']+losses['relative_separation']+losses['gamma_anchor']
            if abs(expected-losses['frequency_denoising'])>2e-6 or abs(total-losses['total'])>2e-6:
                raise ValueError('relative_separation actual loss decomposition mismatch')
        labels=[];frequency=[]
        for i in range(16):
            folder=p/'rows'/f'{i:04d}';mask,freq=evaluate.read_row(folder,protocol['binding'],i);meta=json.loads((folder/'row.json').read_bytes())
            if meta['image_id']!=1320+evaluate.OFFSETS[i] or meta['inference_seed']!=193000+evaluate.OFFSETS[i]:raise ValueError('wrong image/RNG')
            labels.append(mask);frequency.append(freq)
        labels=np.stack(labels)
        if train.sha(labels.tobytes())!=protocol['labels_sha256'] or train.sha((p/'baseline.npz').read_bytes())!=admission['baseline_file_sha256']:raise ValueError('candidate/baseline changed')
        with np.load(p/'baseline.npz',allow_pickle=False) as values:baseline=values['labels'].copy()
        if train.sha(baseline.tobytes())!=binding['DEV_baseline_sha256']:raise ValueError('baseline identity changed')
        if train.sha((p/'reference_frequency.npz').read_bytes())!=report['reference_frequency_file_sha256'] or report['reference_frequency_file_sha256']!=initial['reference_frequency_file_sha256']:raise ValueError('source frequency changed')
        with np.load(p/'reference_frequency.npz',allow_pickle=False) as values:reference=values['frequency'].copy()
        current=binding['parent']['dataset']
        if dataset is None:dataset=current
        elif dataset!=current:raise ValueError('different evaluation datasets')
        control,control_ref=projection.control_reference(seed,binding)
        if control_ref!=binding['matched_center1_control']:raise ValueError('matched encoder-only control changed')
        with np.load(p/'control.npz',allow_pickle=False) as values:
            if not np.array_equal(values['labels'],control):raise ValueError('admitted control arrays changed')
        control_folder=projection.control.archive_for(seed,64);control_protocol=json.loads((control_folder/'protocol.json').read_bytes())
        control_frequency=np.stack([projection.control_eval.read_row(control_folder/'rows'/f'{i:04d}',control_protocol['binding'],i)[1] for i in range(16)])
        models[seed]=dict(labels=labels,baseline=baseline,control=control,control_frequency=control_frequency,frequency=np.stack(frequency),reference=reference,report=report,projection_sha256=train.state_sha(payload['wrapped_state']),
            checkpoint_sha256=train.sha(raw),encoder_sha256=report['final_encoder_sha256'],teacher_sha256=report['teacher_file_sha256'],protocol_sha256=train.sha((p/'protocol.json').read_bytes()))
    # All three checkpoint/teacher/mask protocols checked before any DEV GT read.
    gt=train.source.source.full.read_gt(dict(dataset=dataset))[list(evaluate.OFFSETS)];independent=train.source.source.full.metrics.independent
    vectors={};maximum=0.;frequency_summary={}
    for seed,model in models.items():
        for arm,key in (('baseline','baseline'),('center1','control'),('relative_separation','labels')):
            vector=np.stack([independent.image_metrics(mask,target) for mask,target in zip(model[key],gt)]);name=f'{arm}_seed{seed}';vectors[name]=vector
            maximum=max(maximum,float(np.max(np.abs(vector-np.asarray(private['vectors'][name])))))
        before=diagnostic.frequency_statistics(model['reference'],gt);after=diagnostic.frequency_statistics(model['frequency'],gt);control_stats=diagnostic.frequency_statistics(model['control_frequency'],gt)
        keys=('within_object_variance','adjacent_class_mean_squared_distance','global_frequency_variance');select=[0,2,3]
        centers=[]
        for old,new,grid in zip(model['reference'],model['frequency'],gt):
            flat=grid.reshape(-1);centers.append(float(np.mean([np.mean((new[flat==k].mean(0)-old[flat==k].mean(0))**2) for k in np.unique(flat)])))
        frequency_summary[str(seed)]=dict(before=dict(zip(keys,before[:,select].mean(0).tolist())),after=dict(zip(keys,after[:,select].mean(0).tolist())),center1=dict(zip(keys,control_stats[:,select].mean(0).tolist())),
            source_object_center_MSE=float(np.mean(centers)),DEV_frequency_denoising_MSE=float(after[:,0].mean()+np.mean(centers)),
            TRAIN_losses=dict(before=model['report']['initial_TRAIN_phase_loss'],after=model['report']['final_TRAIN_phase_loss']))
    error,gate,balanced=check_statistics(vectors,score);maximum=max(maximum,error)
    if maximum>1e-12 or train.sha((a/'score_v1/summary.json').read_bytes())!=train.sha(score_raw):raise ValueError('metric/CI/gates/score mismatch')
    result=dict(status='complete_actual_relative_separation_teacher_control_and_mask_audit',score_sha256=train.sha(score_raw),
        diagnostic_source_sha256=train.sha(Path(__file__).read_bytes()),frequency_helper_sha256=train.sha(Path(diagnostic.__file__).read_bytes()),
        checkpoints={str(s):{k:model[k] for k in ('checkpoint_sha256','encoder_sha256','teacher_sha256','protocol_sha256','projection_sha256')} for s,model in models.items()},
        source_full_models_already_reconstructed_by_frozen_scorer=True,independent_full_model_factory_repetitions=0,
        actual_encoder_checkpoints=3,actual_Adam_steps=128,core_projection_updates=128,other_core_updates=0,actual_projection_parameters=32,raw_masks_independently_rescored=144,
        independent_metrics_means_and_bootstrap_maximum_abs_error=maximum,full320_candidate_gate=gate,balanced_candidate_gate=balanced,
        per_seed_frequency=frequency_summary,all_three_frozen_protocols_checked_before_DEV_GT=True,
        additional_model_forwards=0,additional_training_updates=0,raw_imagewise_vectors_published=False,
        frequency_changes_are_not_gating_causal_evidence=True)
    train.write_json(a/'completion_audit_v1/summary.json',result);return result


if __name__=='__main__':print(json.dumps(run()))
