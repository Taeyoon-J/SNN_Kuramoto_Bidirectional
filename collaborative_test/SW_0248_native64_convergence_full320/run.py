"""Conditional full320 fixed-checkpoint confirmation; no additional optimization."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import scope

training=scope.training;reference=training.reference;reader=training.reader;helpers=training.helpers
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive'
SEEDS=scope.SEEDS;OFFSETS=scope.OFFSETS;sha=training.sha;write_json=training.write_json;require_seed=training.require_seed


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    names=('DESIGN_KO.md','scope.py','run.py','evaluate.py','queue.py','test_scope.py','test_pipeline.py','conftest.py','completion_audit.py','test_audit_bootstrap.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},training=training.family_fingerprint(),selection=scope.qualification(),
        seeds=list(SEEDS),grid=[64,64],validation_images=320,reused_offsets=list(scope.MINI),new_native_forwards_per_seed=289,optimizer_updates=0)


def prepare_cpu(seed):
    selection=scope.qualification()
    if not selection['eligible']:raise ValueError('negative247 screen; full320 expansion must remain unstarted')
    p=scope.restore_seed(seed);report=p['report'];dataset=p['source_binding']['dataset'];stat=Path(dataset['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(dataset['size'],dataset['mtime_ns']):raise ValueError('source dataset changed')
    p['binding']=dict(fingerprint=fingerprint(),seed=seed,dataset=dataset,RGB_sha256=sha(p['rgb'].tobytes()),RGB_row_sha256=[sha(v.tobytes()) for v in p['rgb']],
        baseline_sha256=sha(p['baseline'].tobytes()),mini_labels_sha256=sha(p['mini'].tobytes()),mini_frequency_sha256=sha(p['mini_frequency'].tobytes()),
        candidate_checkpoint_sha256=report['checkpoint_sha256'],candidate_model_sha256=report['final_model_sha256'],expected_runtime=report['runtime'],selected_arm=p['selected_arm'],
        training_report_sha256=sha((training.archive_for(seed,p['selected_arm'])/'training_report.json').read_bytes()),source32_protocol_sha256=p['protocol_sha256'],DEV_GT_read=False)
    return p


def admit(seed):
    p=prepare_cpu(seed);a=archive_for(seed)
    helpers.save_arrays(a/'baseline.npz',labels=p['baseline']);helpers.save_arrays(a/'mini.npz',labels=p['mini'],frequency=p['mini_frequency'])
    result=dict(status='actual_trained247_model_and_canonical320_CPU_admitted',binding=p['binding'],cuda_initialized=False,optimizer_updates=0,
        baseline_file_sha256=sha((a/'baseline.npz').read_bytes()),mini_file_sha256=sha((a/'mini.npz').read_bytes()))
    write_json(a/'admission.json',result);return result


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed);ad=json.loads((a/'admission.json').read_bytes())
    if ad['binding']!=p['binding']:raise ValueError('admitted trained checkpoint/source changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0);found=p['found']
    reference.source.source.source.model64.move_to_device(found,device);found.encoder.eval();found.wrapped.eval()
    actual=training.resume_policy.compatible_runtime(training.checkpoint.runtime_signature(device))
    if actual!=p['binding']['expected_runtime']:raise ValueError('trained source numerical runtime differs')
    replay=a/'replay.json'
    if not replay.exists():
        mask,frequency=reference.forward(found,p['rgb'][0],device,0)
        if not np.array_equal(mask,p['mini'][0]) or not np.array_equal(frequency,p['mini_frequency'][0]):raise ValueError('actual247 model fails exact original mask/frequency replay')
        write_json(replay,dict(binding=p['binding'],mask_bitwise_equal=True,frequency_bitwise_equal=True,additional_native_forwards=1))
    values=[]
    for offset in OFFSETS:
        folder=a/'rows'/f'{offset:04d}'
        if not folder.exists():
            if offset in scope.MINI:
                i=scope.MINI.index(offset);mask=p['mini'][i];frequency=p['mini_frequency'][i];origin='frozen247_validation32'
            else:mask,frequency=reference.forward(found,p['rgb'][offset],device,offset);origin='new_native_forward'
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            helpers.save_arrays(stage/'prediction.npz',labels=mask,frequency=frequency)
            write_json(stage/'row.json',dict(binding=p['binding'],index=offset,image_id=1320+offset,inference_seed=193000+offset,RGB_sha256=p['binding']['RGB_row_sha256'][offset],origin=origin,data_sha256=sha((stage/'prediction.npz').read_bytes())))
            stage.rename(folder)
        labels,_=reader.read_row(folder,p['binding'],offset);values.append(labels)
    if reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['binding']['candidate_model_sha256']:raise ValueError('fixed trained model changed during inference')
    result=dict(binding=p['binding'],labels_sha256=sha(np.stack(values).tobytes()),image_ids=[1320+i for i in OFFSETS],DEV_GT_read=False,total_steps=1024,settle_steps=512,
        optimizer_updates=0,reused_reference_images=32,new_prediction_images=288,replay_file_sha256=sha(replay.read_bytes()))
    path=a/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('completed full320 prediction protocol changed')
    else:write_json(path,result)


def validated_seed(seed):
    a=archive_for(seed);ad=json.loads((a/'admission.json').read_bytes());b=ad['binding'];protocol=json.loads((a/'protocol.json').read_bytes())
    if b['fingerprint']!=fingerprint() or b['selected_arm']!=scope.qualification()['selected_arm'] or b['seed']!=seed or protocol['binding']!=b:raise ValueError('fixed full320 source/protocol changed')
    raw=(training.archive_for(seed,b['selected_arm'])/'encoder_work/boundary.ptx').read_bytes()
    if sha(raw)!=b['candidate_checkpoint_sha256']:raise ValueError('actual selected checkpoint changed')
    source=training.archive_for(seed,b['selected_arm'])
    if sha((source/'training_report.json').read_bytes())!=b['training_report_sha256'] or sha((source/'protocol.json').read_bytes())!=b['source32_protocol_sha256']:raise ValueError('accepted training report or source32 protocol changed')
    if protocol['optimizer_updates']!=0 or protocol['DEV_GT_read'] or protocol['image_ids']!=[1320+i for i in OFFSETS] or protocol['total_steps']!=1024 or protocol['settle_steps']!=512 or protocol['reused_reference_images']!=32 or protocol['new_prediction_images']!=288:raise ValueError('different full320 scope')
    replay=json.loads((a/'replay.json').read_bytes())
    if replay['binding']!=b or not replay['mask_bitwise_equal'] or not replay['frequency_bitwise_equal'] or sha((a/'replay.json').read_bytes())!=protocol['replay_file_sha256']:raise ValueError('exact source replay required')
    if sha((a/'baseline.npz').read_bytes())!=ad['baseline_file_sha256'] or sha((a/'mini.npz').read_bytes())!=ad['mini_file_sha256']:raise ValueError('frozen controls changed')
    with np.load(a/'mini.npz',allow_pickle=False) as data:mini=data['labels'].copy();freq=data['frequency'].copy()
    if mini.shape!=(32,64,64) or freq.shape!=(32,4096,4) or sha(mini.tobytes())!=b['mini_labels_sha256'] or sha(freq.tobytes())!=b['mini_frequency_sha256']:raise ValueError('different reused32 arrays')
    values=[]
    for offset in OFFSETS:
        folder=a/'rows'/f'{offset:04d}';mask,frequency=reader.read_row(folder,b,offset);row=json.loads((folder/'row.json').read_bytes())
        if row['image_id']!=1320+offset or row['inference_seed']!=193000+offset or row['RGB_sha256']!=b['RGB_row_sha256'][offset] or row['origin']!=('frozen247_validation32' if offset in scope.MINI else 'new_native_forward'):raise ValueError('wrong full320 row identity')
        if offset in scope.MINI:
            i=scope.MINI.index(offset)
            if not np.array_equal(mask,mini[i]) or not np.array_equal(frequency,freq[i]):raise ValueError('reused32 prediction changed')
        values.append(mask)
    labels=np.stack(values)
    if sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('full320 mask stack changed')
    with np.load(a/'baseline.npz',allow_pickle=False) as data:baseline=data['labels'].copy()
    if baseline.shape!=(320,64,64) or sha(baseline.tobytes())!=b['baseline_sha256']:raise ValueError('current237 reference changed')
    return dict(labels=labels,baseline=baseline,protocol=protocol)


def score():
    from .evaluate import score as measure
    return measure()


if __name__=='__main__':
    import ctypes,sys
    if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP disable failed')
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--admit',action='store_true');parser.add_argument('--score',action='store_true');parser.add_argument('--source-check',action='store_true');args=parser.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    elif args.source_check and args.seed is not None:
        scope.restore_seed(args.seed);print(json.dumps(dict(status='actual247_model_restored_on_CPU_without_GT_or_optimizer_updates',seed=args.seed,cuda_initialized=torch.cuda.is_initialized())))
    else:parser.error('choose source check, admission or scoring')
