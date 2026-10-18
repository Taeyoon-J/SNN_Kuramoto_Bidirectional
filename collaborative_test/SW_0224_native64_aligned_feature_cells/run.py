"""Zero-update native64 feature-cell alignment comparison."""
import argparse,hashlib,json,os,time,types
from pathlib import Path
import numpy as np
import torch
from . import alignment
from collaborative_test.SW_0217_native64_energy_refinement import run as source
from collaborative_test.SW_0221_native64_frequency_pretraining import train as helpers,evaluate as reader

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive';SEEDS=(0,1,2);OFFSETS=tuple(range(0,32,2))
sha=source.sha;write_json=source.write_json;require_seed=source.require_seed
CHECKPOINTS={0:('SW_0192_native64_incumbent_mini/results_archive/probe_work_seed0/boundary.ptx','14d0386db64d3890163574ac2421b1e80d3100536ec25c9940434e0689327e82'),
 1:('SW_0205_native64_weak_seed_adaptation/results_archive/seed1/training_work/boundary.ptx','9ef50f95559ca47f0f8f87e1cf7d78e3f8a44c5fb51445b3acd03bf7a9b24e85'),
 2:('SW_0205_native64_weak_seed_adaptation/results_archive/seed2/training_work/boundary.ptx','b88668e8cac21ca1b40d65660051443f6ec1881e623a7ee368fb745726f3525c')}


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    names=('DESIGN_KO.md','alignment.py','run.py','evaluate.py','queue.py','test_alignment.py','test_pipeline.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},source=source.fingerprint(),
        shared_helpers_sha256=sha(Path(helpers.__file__).read_bytes()),reader_sha256=sha(Path(reader.__file__).read_bytes()),
        checkpoints={str(k):list(v) for k,v in CHECKPOINTS.items()},seeds=list(SEEDS),offsets=list(OFFSETS),optimizer_updates=0,
        feature_border_padding=1,feature_border_mode='replicate',RGB_GT_resize_or_crop=False,native_grid=[64,64])


def checkpoint_sha(seed):
    relative,expected=CHECKPOINTS[require_seed(seed)];value=sha((ROOT/'collaborative_test'/relative).read_bytes())
    if value!=expected:raise ValueError('immutable source checkpoint bytes changed')
    return value


def prepare_cpu(seed):
    if torch.cuda.is_initialized():raise ValueError('CPU admission before CUDA required')
    found,rgb,baseline,parent=source.prepare_cpu(seed);found.encoder.eval();found.wrapped.eval();found.encoder.requires_grad_(False);found.wrapped.requires_grad_(False)
    with torch.no_grad():
        image=torch.from_numpy(rgb[:1]);old=found.encode(image);new=alignment.encode(found,image)
    model=source.source.full.screen.native._state_sha(found.wrapped,found.encoder)
    if model!=source.MODELS[str(seed)]:raise ValueError('alignment preparation changed source weights')
    binding=dict(fingerprint=fingerprint(),seed=seed,parent=parent,source_parameter_sha256=model,source_checkpoint_sha256=checkpoint_sha(seed),
        source_statistics_sha256=helpers.statistics_sha(found),RGB_sha256=sha(rgb.tobytes()),baseline_sha256=sha(baseline.tobytes()),
        CPU_gamma_change_RMS=float((new-old).square().mean().sqrt()),geometry=alignment.geometry(),DEV_GT_read=False)
    return dict(found=found,rgb=rgb,baseline=baseline,binding=binding,seed=seed)


def admit(seed):
    p=prepare_cpu(seed);a=archive_for(seed);helpers.save_arrays(a/'baseline.npz',labels=p['baseline'])
    result=dict(status='actual_source_feature_alignment_CPU_admitted',binding=p['binding'],cuda_initialized=False,optimizer_updates=0,baseline_file_sha256=sha((a/'baseline.npz').read_bytes()))
    write_json(a/'admission.json',result);return result


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed);admission=json.loads((a/'admission.json').read_bytes())
    if p['binding']!=admission['binding']:raise ValueError('admitted source/adapter changed')
    found=p['found'];device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    source.source.source.model64.move_to_device(found,device);found.encode=types.MethodType(alignment.encode,found)
    labels=[]
    for index,offset in enumerate(OFFSETS):
        folder=a/'rows'/f'{index:04d}'
        if folder.exists():mask,_=reader.read_row(folder,p['binding'],index)
        else:
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=source.source.full.screen.native._native_forward(found,torch.from_numpy(p['rgb'][index:index+1]).to(device),live_tail_steps=0)
                mask=np.asarray(source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64)
                frequency=helpers.effective_frequency(found.wrapped.core,gamma)[0].cpu().numpy()
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            helpers.save_arrays(stage/'prediction.npz',labels=mask,frequency=frequency)
            write_json(stage/'row.json',dict(binding=p['binding'],index=index,image_id=1320+offset,inference_seed=193000+offset,data_sha256=sha((stage/'prediction.npz').read_bytes())))
            stage.rename(folder);mask,_=reader.read_row(folder,p['binding'],index);del gamma,trace,adj,z,q,rho,frequency
        labels.append(mask)
    if source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['binding']['source_parameter_sha256'] or helpers.statistics_sha(found)!=p['binding']['source_statistics_sha256']:raise ValueError('zero-update inference mutated source')
    protocol=dict(binding=p['binding'],labels_sha256=sha(np.stack(labels).tobytes()),image_ids=[1320+i for i in OFFSETS],DEV_GT_read=False,total_steps=1024,settle_steps=512,
        parameter_sha256_before_after=p['binding']['source_parameter_sha256'],optimizer_updates=0,adapter_code_sha256=sha((HERE/'alignment.py').read_bytes()))
    path=a/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen alignment protocol changed')
    else:write_json(path,protocol)


def validated_seed(seed):
    a=archive_for(seed);admission=json.loads((a/'admission.json').read_bytes());binding=admission['binding'];protocol=json.loads((a/'protocol.json').read_bytes())
    if binding['fingerprint']!=fingerprint() or binding['seed']!=seed or binding['source_checkpoint_sha256']!=checkpoint_sha(seed) or binding['source_parameter_sha256']!=source.MODELS[str(seed)]:raise ValueError('source checkpoint/adapter changed')
    if protocol['binding']!=binding or protocol['DEV_GT_read'] is not False or protocol['optimizer_updates']!=0 or protocol['total_steps']!=1024 or protocol['settle_steps']!=512 or protocol['parameter_sha256_before_after']!=binding['source_parameter_sha256'] or protocol['adapter_code_sha256']!=sha((HERE/'alignment.py').read_bytes()) or protocol['image_ids']!=[1320+i for i in OFFSETS]:raise ValueError('invalid alignment protocol')
    rows=[]
    for i,offset in enumerate(OFFSETS):
        folder=a/'rows'/f'{i:04d}';mask,_=reader.read_row(folder,binding,i);meta=json.loads((folder/'row.json').read_bytes())
        if meta['image_id']!=1320+offset or meta['inference_seed']!=193000+offset:raise ValueError('wrong image/RNG')
        rows.append(mask)
    if sha(np.stack(rows).tobytes())!=protocol['labels_sha256'] or sha((a/'baseline.npz').read_bytes())!=admission['baseline_file_sha256']:raise ValueError('candidate/baseline file changed')
    with np.load(a/'baseline.npz',allow_pickle=False) as values:baseline=values['labels'].copy()
    if baseline.shape!=(16,64,64) or baseline.dtype!=np.int64 or sha(baseline.tobytes())!=binding['baseline_sha256']:raise ValueError('wrong same-model source masks')
    return dict(labels=np.stack(rows),baseline=baseline,protocol=protocol)


def score():
    from .evaluate import score as measure
    return measure()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--admit',action='store_true');parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    else:parser.error('choose CPU admission or frozen scoring; predict through exclusive queue')
