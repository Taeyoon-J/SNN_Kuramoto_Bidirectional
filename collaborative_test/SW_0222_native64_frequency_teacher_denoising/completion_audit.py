"""Post-score audit and source-frequency diagnostics without another model run.

The frozen scorer already reconstructs all three actual full models. This
separate audit checks encoder payloads, teachers, all saved masks/frequencies,
independent metrics and paired CIs. It does not repeat full-model factories.
"""
import io,json
from pathlib import Path
import numpy as np
import torch
from . import train,evaluate
from collaborative_test.SW_0221_native64_frequency_pretraining import diagnose_frequency as diagnostic


def run():
    a=train.ARCHIVE;score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw);private=json.loads((a/'score_v1/evaluation_private.json').read_bytes())
    if score['fingerprint']!=train.fingerprint() or private['summary']!=score or score['status']!='complete_native64_source_teacher_denoising_mini':raise ValueError('complete frozen three-model score required')
    models={};dataset=None
    for seed in train.SEEDS:
        p=train.archive_for(seed,64);report=json.loads((p/'training_report.json').read_bytes());protocol=json.loads((p/'protocol.json').read_bytes());admission=json.loads((p/'admission.json').read_bytes());initial=json.loads((p/'initial.json').read_bytes())
        binding=report['binding'];state=json.loads((p/'queue/state.json').read_bytes())
        if binding!=admission['binding'] or binding['fingerprint']!=train.fingerprint() or binding['parent_model_sha256']!=train.source.MODELS[str(seed)]:raise ValueError('wrong admitted source')
        if report['core_updates']!=0 or report['actual_Adam_steps']!=[128] or not report['encoder_changed'] or report['DEV_GT_read'] is not False:raise ValueError('wrong encoder-only update protocol')
        if protocol['binding']!=dict(training_report_sha256=train.sha((p/'training_report.json').read_bytes()),binding=binding,model_sha256=report['final_model_sha256']) or protocol['DEV_GT_read'] is not False or protocol['image_ids']!=[1320+i for i in evaluate.OFFSETS] or protocol['total_steps']!=1024 or protocol['settle_steps']!=512:raise ValueError('wrong frozen prediction protocol')
        if state['status']!='complete' or state['protocol_sha256']!=train.sha((p/'protocol.json').read_bytes()):raise ValueError('queue not complete')
        train.checkpoint.inspect_committed_metadata(p/'encoder_work');raw=(p/'encoder_work/boundary.ptx').read_bytes()
        outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);payload=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
        expected_checkpoint_binding=dict(binding=binding,initial_metadata_sha256=train.sha((p/'initial.json').read_bytes()))
        if train.sha(raw)!=report['checkpoint_file_sha256'] or payload['completed_updates']!=128 or payload['wrapped_state'] or payload['history']!=report['history'] or payload['bindings']!=expected_checkpoint_binding or report['checkpoint_binding']!=expected_checkpoint_binding:raise ValueError('wrong actual committed encoder state')
        if train.state_sha(payload['encoder_state'])!=report['final_encoder_sha256'] or sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})!=[128]:raise ValueError('actual encoder/Adam mismatch')
        if any(row['indices']!=train.batch_indices(seed,64,i) or row['update']!=i+1 for i,row in enumerate(payload['history'])):raise ValueError('TRAIN order changed')
        teacher_raw=(p/'teacher.npz').read_bytes()
        if train.sha(teacher_raw)!=report['teacher_file_sha256'] or report['teacher_file_sha256']!=initial['teacher_file_sha256'] or report['teacher_file_sha256']!=admission['teacher_file_sha256']:raise ValueError('teacher file changed')
        with np.load(io.BytesIO(teacher_raw),allow_pickle=False) as values:teacher={k:torch.from_numpy(values[k].copy()) for k in values.files}
        if train.state_sha(teacher)!=binding['teacher_arrays_sha256']:raise ValueError('teacher content changed')
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
        models[seed]=dict(labels=labels,baseline=baseline,frequency=np.stack(frequency),reference=reference,report=report,
            checkpoint_sha256=train.sha(raw),encoder_sha256=report['final_encoder_sha256'],teacher_sha256=report['teacher_file_sha256'],protocol_sha256=train.sha((p/'protocol.json').read_bytes()))
    # All three checkpoint/teacher/mask protocols checked before any DEV GT read.
    gt=train.source.source.full.read_gt(dict(dataset=dataset))[list(evaluate.OFFSETS)];independent=train.source.source.full.metrics.independent
    vectors={};maximum=0.;frequency_summary={}
    for seed,model in models.items():
        for arm,key in (('baseline','baseline'),('denoised','labels')):
            vector=np.stack([independent.image_metrics(mask,target) for mask,target in zip(model[key],gt)]);name=f'{arm}_seed{seed}';vectors[name]=vector
            maximum=max(maximum,float(np.max(np.abs(vector-np.asarray(private['vectors'][name])))))
        before=diagnostic.frequency_statistics(model['reference'],gt);after=diagnostic.frequency_statistics(model['frequency'],gt)
        keys=('within_object_variance','adjacent_class_mean_squared_distance','global_frequency_variance');select=[0,2,3]
        centers=[]
        for old,new,grid in zip(model['reference'],model['frequency'],gt):
            flat=grid.reshape(-1);centers.append(float(np.mean([np.mean((new[flat==k].mean(0)-old[flat==k].mean(0))**2) for k in np.unique(flat)])))
        frequency_summary[str(seed)]=dict(before=dict(zip(keys,before[:,select].mean(0).tolist())),after=dict(zip(keys,after[:,select].mean(0).tolist())),
            source_object_center_MSE=float(np.mean(centers)),DEV_frequency_denoising_MSE=float(after[:,0].mean()+np.mean(centers)),
            TRAIN_losses=dict(before=model['report']['initial_TRAIN_phase_loss'],after=model['report']['final_TRAIN_phase_loss']))
    groups={arm:np.stack([vectors[f'{arm}_seed{s}'] for s in train.SEEDS]) for arm in ('baseline','denoised')}
    for name,value in vectors.items():maximum=max(maximum,float(np.max(np.abs(value.mean(0)-[score['per_seed_scores'][name][k] for k in evaluate.METRICS]))))
    for name,value in groups.items():maximum=max(maximum,float(np.max(np.abs(value.mean((0,1))-[score['scores'][name][k] for k in evaluate.METRICS]))))
    delta=(groups['denoised']-groups['baseline']).mean(0);draw=np.random.RandomState(221).randint(0,16,(10000,16));ci=np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
    for i,key in enumerate(evaluate.METRICS):
        expected=score['paired_denoised_minus_baseline'][key];maximum=max(maximum,abs(expected['mean']-float(delta[:,i].mean())),float(np.max(np.abs(expected['ci95']-ci[:,i]))))
    gate=bool(delta[:,0].mean()>=.01 and ci[0,0]>0 and delta[:,2].mean()>=0)
    if maximum>1e-12 or gate!=score['full320_candidate_gate'] or train.sha((a/'score_v1/summary.json').read_bytes())!=train.sha(score_raw):raise ValueError('metric/CI/gate/score mismatch')
    result=dict(status='complete_actual_teacher_checkpoint_mask_and_frequency_audit',score_sha256=train.sha(score_raw),
        diagnostic_source_sha256=train.sha(Path(__file__).read_bytes()),frequency_helper_sha256=train.sha(Path(diagnostic.__file__).read_bytes()),
        checkpoints={str(s):{k:model[k] for k in ('checkpoint_sha256','encoder_sha256','teacher_sha256','protocol_sha256')} for s,model in models.items()},
        source_full_models_already_reconstructed_by_frozen_scorer=True,independent_full_model_factory_repetitions=0,
        actual_encoder_checkpoints=3,actual_Adam_steps=128,core_updates=0,raw_masks_independently_rescored=96,
        independent_metrics_means_and_bootstrap_maximum_abs_error=maximum,full320_candidate_gate=gate,
        per_seed_frequency=frequency_summary,all_three_frozen_protocols_checked_before_DEV_GT=True,
        additional_model_forwards=0,additional_training_updates=0,raw_imagewise_vectors_published=False,
        frequency_changes_are_not_gating_causal_evidence=True)
    train.write_json(a/'completion_audit_v1/summary.json',result);return result


if __name__=='__main__':print(json.dumps(run()))
