"""Fixed223 models, first64 validation images, no additional training."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import scope
from collaborative_test.SW_0217_native64_energy_refinement import run as source
from collaborative_test.SW_0221_native64_frequency_pretraining import train as helpers,evaluate as reader

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive';SEEDS=(0,1,2);OFFSETS=scope.OFFSETS
sha=source.sha;write_json=source.write_json;require_seed=source.require_seed


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    names=('DESIGN_KO.md','scope.py','run.py','evaluate.py','queue.py','test_scope.py','test_pipeline.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},candidate=scope.model.fingerprint(),original_source=source.fingerprint(),
        shared_reader_sha256=sha(Path(reader.__file__).read_bytes()),seeds=list(SEEDS),IDs=[1320+i for i in OFFSETS],reused_mini_offsets=list(scope.MINI),
        replay_offsets=[0],expected_new_native_forwards_per_seed=49,optimizer_updates=0,selection='predetermined first64 validation images; fixed232 checkpoints')


def prepare_cpu(seed):
    if torch.cuda.is_initialized():raise ValueError('CPU admission before CUDA')
    found,mini_rgb,mini_baseline,parent=source.prepare_cpu(seed);report,ref,mini,frequency=scope.restore(found,seed)
    all0,b0=source.source.full.validated_all()
    if seed==0:baseline=all0['warm192'][:64].copy()
    else:
        all_labels,b=source.source.validated_seed(seed)
        if b['model_sha256']!=report['binding']['parent_model_sha256'] or b['RGB_sha256']!=b0['RGB_sha256']:raise ValueError('different fixed320 baseline')
        baseline=all_labels[:64].copy()
    import h5py
    dataset=Path(b0['dataset']['path']);stat=dataset.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(b0['dataset']['size'],b0['dataset']['mtime_ns']):raise ValueError('dataset changed')
    with h5py.File(dataset,'r') as stream:rgb=np.asarray(stream['image'][1320:1384])
    rgb=scope.canonical_rgb(rgb)
    if rgb.shape!=(64,128,128,3) or [sha(v.tobytes()) for v in rgb]!=b0['RGB_row_sha256'][:64]:raise ValueError('canonical64 RGB mismatch')
    if not np.array_equal(rgb[list(scope.MINI)],mini_rgb) or not np.array_equal(baseline[list(scope.MINI)],mini_baseline) or sha(mini_rgb.tobytes())!=report['binding']['DEV_RGB_sha256']:raise ValueError('secondary cohort does not contain exact original16')
    binding=dict(fingerprint=fingerprint(),seed=seed,candidate=ref,dataset=b0['dataset'],RGB_sha256=sha(rgb.tobytes()),RGB_row_sha256=[sha(v.tobytes()) for v in rgb],
        baseline_sha256=sha(baseline.tobytes()),source_binding_sha256=sha(json.dumps(parent,sort_keys=True).encode()),DEV_GT_read=False)
    return dict(found=found,rgb=rgb,baseline=baseline,mini=mini,mini_frequency=frequency,binding=binding,seed=seed)


def admit(seed):
    p=prepare_cpu(seed);a=archive_for(seed);helpers.save_arrays(a/'baseline.npz',labels=p['baseline']);helpers.save_arrays(a/'mini.npz',labels=p['mini'],frequency=p['mini_frequency'])
    result=dict(status='actual_fixed232_model_and_canonical64_CPU_admitted',binding=p['binding'],cuda_initialized=False,optimizer_updates=0,
        baseline_file_sha256=sha((a/'baseline.npz').read_bytes()),mini_file_sha256=sha((a/'mini.npz').read_bytes()))
    write_json(a/'admission.json',result);return result


def forward(found,image,device,offset):
    torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
    with torch.no_grad():
        gamma,trace,adj,z,q,rho=source.source.full.screen.native._native_forward(found,torch.from_numpy(image[None]).to(device),live_tail_steps=0)
        labels=np.asarray(source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64);frequency=helpers.effective_frequency(found.wrapped.core,gamma)[0].cpu().numpy()
    return labels,frequency


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed);admission=json.loads((a/'admission.json').read_bytes())
    if admission['binding']!=p['binding']:raise ValueError('fixed232 model/source changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0);found=p['found'];source.source.source.model64.move_to_device(found,device)
    actual=scope.model.resume_policy.compatible_runtime(scope.model.checkpoint.runtime_signature(device))
    if actual!=p['binding']['candidate']['expected_runtime']:raise ValueError('inference runtime differs from fixed232 model')
    replay=a/'replay.json'
    if not replay.exists():
        labels,frequency=forward(found,p['rgb'][0],device,0)
        if not np.array_equal(labels,p['mini'][0]) or not np.array_equal(frequency,p['mini_frequency'][0]):raise ValueError('restored model does not exactly replay original232 mask')
        write_json(replay,dict(binding=p['binding'],mask_bitwise_equal=True,frequency_bitwise_equal=True,image_id=1320,inference_seed=193000,
            frequency_max_abs_error=float(np.abs(frequency-p['mini_frequency'][0]).max()),additional_native_forwards=1))
    rows=[]
    for offset in OFFSETS:
        folder=a/'rows'/f'{offset:04d}'
        if folder.exists():labels,_=reader.read_row(folder,p['binding'],offset)
        else:
            if offset in scope.MINI:
                i=scope.MINI.index(offset);labels=p['mini'][i];frequency=p['mini_frequency'][i];origin='frozen232_mini'
            else:labels,frequency=forward(found,p['rgb'][offset],device,offset);origin='new_native_forward'
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            helpers.save_arrays(stage/'prediction.npz',labels=labels,frequency=frequency)
            write_json(stage/'row.json',dict(binding=p['binding'],index=offset,image_id=1320+offset,inference_seed=193000+offset,
                RGB_sha256=p['binding']['RGB_row_sha256'][offset],origin=origin,data_sha256=sha((stage/'prediction.npz').read_bytes())))
            stage.rename(folder);labels,_=reader.read_row(folder,p['binding'],offset)
        rows.append(labels)
    if source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['binding']['candidate']['model_sha256']:raise ValueError('fixed model mutated during inference')
    protocol=dict(binding=p['binding'],labels_sha256=sha(np.stack(rows).tobytes()),image_ids=[1320+i for i in OFFSETS],DEV_GT_read=False,total_steps=1024,settle_steps=512,
        optimizer_updates=0,reused_mini_images=16,new_prediction_images=48,replay_file_sha256=sha(replay.read_bytes()))
    path=a/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen secondary protocol changed')
    else:write_json(path,protocol)


def validated_seed(seed):
    a=archive_for(seed);admission=json.loads((a/'admission.json').read_bytes());binding=admission['binding'];protocol=json.loads((a/'protocol.json').read_bytes())
    current=scope.model.archive_for(seed,64)/'encoder_work/boundary.ptx'
    if binding['fingerprint']!=fingerprint() or binding['seed']!=seed or sha(current.read_bytes())!=binding['candidate']['checkpoint_sha256']:raise ValueError('fixed candidate changed')
    if protocol['binding']!=binding or protocol['optimizer_updates']!=0 or protocol['DEV_GT_read'] is not False or protocol['image_ids']!=[1320+i for i in OFFSETS] or protocol['total_steps']!=1024 or protocol['settle_steps']!=512 or protocol['reused_mini_images']!=16 or protocol['new_prediction_images']!=48:raise ValueError('wrong secondary protocol')
    replay=json.loads((a/'replay.json').read_bytes())
    if replay['binding']!=binding or not replay['mask_bitwise_equal'] or not replay['frequency_bitwise_equal'] or sha((a/'replay.json').read_bytes())!=protocol['replay_file_sha256']:raise ValueError('original-mini replay missing')
    with np.load(a/'mini.npz',allow_pickle=False) as values:mini=values['labels'].copy()
    labels=[]
    for offset in OFFSETS:
        folder=a/'rows'/f'{offset:04d}';mask,_=reader.read_row(folder,binding,offset);meta=json.loads((folder/'row.json').read_bytes())
        if meta['image_id']!=1320+offset or meta['inference_seed']!=193000+offset or meta['RGB_sha256']!=binding['RGB_row_sha256'][offset] or meta['origin']!=('frozen232_mini' if offset in scope.MINI else 'new_native_forward'):raise ValueError('wrong image/RNG/reuse identity')
        if offset in scope.MINI and not np.array_equal(mask,mini[scope.MINI.index(offset)]):raise ValueError('original mini changed')
        labels.append(mask)
    if sha(np.stack(labels).tobytes())!=protocol['labels_sha256'] or sha((a/'baseline.npz').read_bytes())!=admission['baseline_file_sha256'] or sha((a/'mini.npz').read_bytes())!=admission['mini_file_sha256']:raise ValueError('fixed mask arrays changed')
    with np.load(a/'baseline.npz',allow_pickle=False) as values:baseline=values['labels'].copy()
    if baseline.shape!=(64,64,64) or baseline.dtype!=np.int64 or sha(baseline.tobytes())!=binding['baseline_sha256']:raise ValueError('wrong original-source64 masks')
    return dict(labels=np.stack(labels),baseline=baseline,protocol=protocol)


def score():
    from .evaluate import score as measure
    return measure()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--admit',action='store_true');parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    else:parser.error('choose CPU admission or frozen scoring')
