"""Freeze every final spike mask before reading DEV ground truth."""
import argparse,io,json,os,time
import numpy as np
import torch
from . import train,projection
from collaborative_test.SW_0221_native64_frequency_pretraining.frequency_loss import effective_frequency

OFFSETS=tuple(range(0,32,2))
METRICS=train.source.source.full.metrics.independent.METRICS


def read_row(folder,binding,index):
    record=json.loads((folder/'row.json').read_bytes());raw=(folder/'prediction.npz').read_bytes()
    if record['binding']!=binding or record['index']!=index or record['data_sha256']!=train.sha(raw):raise ValueError('prediction provenance changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:labels=arrays['labels'].copy();frequency=arrays['frequency'].copy()
    if labels.shape!=(64,64) or labels.dtype!=np.int64 or (labels<0).any():raise ValueError('invalid native64 mask')
    if frequency.shape!=(4096,4) or not np.isfinite(frequency).all():raise ValueError('invalid actual frequency')
    return labels,frequency


def predict(prepared,report,device):
    found=prepared['found'];a=train.archive_for(prepared['seed'],prepared['count'])
    binding=dict(training_report_sha256=train.sha((a/'training_report.json').read_bytes()),binding=prepared['binding'],model_sha256=report['final_model_sha256'])
    if train.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=binding['model_sha256']:raise ValueError('prediction model mismatch')
    found.wrapped.eval();found.encoder.eval();found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False)
    rows=[]
    for index,offset in enumerate(OFFSETS):
        folder=a/'rows'/f'{index:04d}'
        if folder.exists():labels,_=read_row(folder,binding,index)
        else:
            torch.manual_seed(193000+offset)
            if torch.device(device).type=='cuda':torch.cuda.manual_seed(193000+offset)
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=train.source.source.full.screen.native._native_forward(found,torch.from_numpy(prepared['dev'][index:index+1]).to(device),live_tail_steps=0)
                labels=np.asarray(train.source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64)
                frequency=effective_frequency(found.wrapped.core,gamma)[0].cpu().numpy()
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            train.save_arrays(stage/'prediction.npz',labels=labels,frequency=frequency)
            train.write_json(stage/'row.json',dict(binding=binding,index=index,image_id=1320+offset,inference_seed=193000+offset,data_sha256=train.sha((stage/'prediction.npz').read_bytes())))
            stage.rename(folder);labels,_=read_row(folder,binding,index)
            del gamma,trace,adj,z,q,rho,frequency
        rows.append(labels)
    if train.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=binding['model_sha256']:raise ValueError('inference changed model')
    protocol=dict(binding=binding,labels_sha256=train.sha(np.stack(rows).tobytes()),image_ids=[1320+i for i in OFFSETS],DEV_GT_read=False,total_steps=1024,settle_steps=512)
    path=a/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen prediction protocol changed')
    else:train.write_json(path,protocol)
    return protocol


def validated_job(seed,count):
    a=train.archive_for(seed,count);p=train.prepare_cpu(seed,count)
    report=json.loads((a/'training_report.json').read_bytes());initial=json.loads((a/'initial.json').read_bytes())
    metadata=train.checkpoint.inspect_committed_metadata(a/'encoder_work')
    expected=dict(binding=p['binding'],initial_metadata_sha256=train.sha((a/'initial.json').read_bytes()))
    if (report['binding']!=p['binding'] or initial['binding']!=p['binding'] or metadata['bindings']!=expected
        or metadata['completed_updates']!=train.STEPS or report['checkpoint_binding']!=expected
        or report['actual_Adam_steps']!=[train.STEPS] or not report['encoder_changed']
        or report['checkpoint_file_sha256']!=train.sha((a/'encoder_work/boundary.ptx').read_bytes())
        or report['frozen_nonprojection_core_sha256']!=p['binding']['frozen_nonprojection_core_sha256']
        or report['input_statistics_sha256']!=p['binding']['input_statistics_sha256']
        or report['DEV_GT_read'] is not False):raise ValueError('encoder checkpoint/source invariant failed')
    outer=torch.load(a/'encoder_work/boundary.ptx',map_location='cpu',weights_only=False)
    payload=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
    if set(payload['wrapped_state'])!={'weight'} or payload['history']!=report['history'] or len(report['history'])!=train.STEPS:raise ValueError('unexpected core state/history')
    for step,row in enumerate(report['history']):
        if row['indices']!=train.batch_indices(seed,count,step) or row['update']!=step+1:raise ValueError('wrong TRAIN order')
    if sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})!=[train.STEPS]:raise ValueError('actual optimizer counter mismatch')
    projection.module(p['found']).load_state_dict(payload['wrapped_state'],strict=True)
    p['found'].encoder.load_state_dict(payload['encoder_state'],strict=True)
    if (projection.frozen_core_sha(p['found'])!=p['binding']['frozen_nonprojection_core_sha256'] or not report['projection_changed']
        or train.state_sha(projection.module(p['found']).state_dict())!=report['final_projection_sha256']
        or train.state_sha(p['found'].wrapped.state_dict())!=report['final_core_sha256']):raise ValueError('projection-only core update invariant failed')
    if (train.state_sha(p['found'].encoder.state_dict())!=report['final_encoder_sha256']
        or train.source.source.full.screen.native._state_sha(p['found'].wrapped,p['found'].encoder)!=report['final_model_sha256']):raise ValueError('actual final model mismatch')
    protocol=json.loads((a/'protocol.json').read_bytes())
    binding=dict(training_report_sha256=train.sha((a/'training_report.json').read_bytes()),binding=p['binding'],model_sha256=report['final_model_sha256'])
    if protocol['binding']!=binding or protocol['DEV_GT_read'] is not False or protocol['total_steps']!=1024 or protocol['settle_steps']!=512 or protocol['image_ids']!=[1320+i for i in OFFSETS]:raise ValueError('invalid frozen protocol')
    labels=np.stack([read_row(a/'rows'/f'{index:04d}',binding,index)[0] for index in range(16)])
    if train.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('final masks changed')
    if train.sha((a/'reference_frequency.npz').read_bytes())!=initial['reference_frequency_file_sha256']:raise ValueError('reference frequencies changed')
    with np.load(a/'baseline.npz',allow_pickle=False) as arrays:
        if not np.array_equal(arrays['labels'],p['baseline']):raise ValueError('original baseline changed')
    teacher=a/'teacher.npz';admission=json.loads((a/'admission.json').read_bytes())
    if (report['teacher_file_sha256']!=train.sha(teacher.read_bytes()) or initial['teacher_file_sha256']!=report['teacher_file_sha256']
        or admission['binding']!=p['binding'] or admission['CUDA_initialized'] is not False
        or admission['teacher_file_sha256']!=report['teacher_file_sha256']):raise ValueError('frozen teacher file/admission changed')
    with np.load(teacher,allow_pickle=False) as arrays:saved_teacher={k:torch.from_numpy(arrays[k].copy()) for k in arrays.files}
    if train.state_sha(saved_teacher)!=p['binding']['teacher_arrays_sha256']:raise ValueError('source teacher arrays changed')
    with np.load(a/'control.npz',allow_pickle=False) as arrays:
        if not np.array_equal(arrays['labels'],p['control_labels']):raise ValueError('frozen encoder-only control changed')
    return dict(labels=labels,baseline=p['baseline'],control=p['control_labels'],report=report,protocol=protocol)


def validated_all():
    return {(s,64):validated_job(s,64) for s in train.SEEDS}


def score():
    bundle=validated_all()
    datasets=[bundle[s,64]['report']['binding']['parent']['dataset'] for s in train.SEEDS]
    if any(v!=datasets[0] for v in datasets):raise ValueError('different evaluation datasets')
    gt=train.source.source.full.read_gt(dict(dataset=datasets[0]))[list(OFFSETS)]
    vectors={};maximum=0.
    for seed in train.SEEDS:
        for arm,key in (('baseline','baseline'),('encoder_only','control'),('joint_projection','labels')):
            vectors[f'{arm}_seed{seed}'],error=train.source.source.full.metrics.measured_vectors(bundle[seed,64][key],gt);maximum=max(maximum,error)
    groups={arm:np.stack([vectors[f'{arm}_seed{s}'] for s in train.SEEDS]) for arm in ('baseline','encoder_only','joint_projection')}
    from collaborative_test.SW_0221_native64_frequency_pretraining.evaluate import paired
    changes={k:paired(groups['joint_projection'],groups[k]) for k in ('baseline','encoder_only')};fg=changes['baseline'][METRICS[0]]
    fg_gate=bool(fg['mean']>=.01 and fg['ci95'][0]>0 and groups['joint_projection'][:,:,0].mean()>groups['encoder_only'][:,:,0].mean())
    balanced=bool(fg_gate and all(changes['baseline'][k]['mean']>=0 for k in METRICS[1:]))
    summary=dict(status='complete_native64_joint_sensory_projection_mini',scores={k:dict(zip(METRICS,v.mean((0,1)).tolist())) for k,v in groups.items()},
        per_seed_scores={k:dict(zip(METRICS,v.mean(0).tolist())) for k,v in vectors.items()},paired_joint_minus=changes,
        full320_candidate_gate=fg_gate,FG_track_candidate_gate=fg_gate,balanced_candidate_gate=balanced,
        training_losses={str(s):dict(before=bundle[s,64]['report']['initial_TRAIN_phase_loss'],after=bundle[s,64]['report']['final_TRAIN_phase_loss']) for s in train.SEEDS},
        projection_learning={str(s):dict(changed=bundle[s,64]['report']['projection_changed'],first_gradient_norm=bundle[s,64]['report']['history'][0]['projection_gradient_norm']) for s in train.SEEDS},
        independent_maximum_metric_abs_error=maximum,grid=[64,64],seed_scope=list(train.SEEDS),images=16,mask_supervised=True,
        core_projection_updates=train.STEPS,other_core_updates=0,encoder_updates=train.STEPS,all_three_protocols_validated_before_DEV_GT=True,
        new70000_training=False,full320_three_seed_goal_verified=False,previous_full320_best_unchanged=True,fingerprint=train.fingerprint())
    train.write_json(train.ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(train.ARCHIVE/'score_v1/summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    if args.admit:print(json.dumps(dict(status='all_three_actual_projection_models_and_matched_controls_CPU_validated',arms=len(validated_all()))))
    else:print(json.dumps(score()))
