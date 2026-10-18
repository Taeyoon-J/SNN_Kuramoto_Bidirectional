"""Encoder-only effective-frequency pretraining; actual spike readout stays fixed."""
import argparse,hashlib,io,json,math,time,os
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0217_native64_energy_refinement import run as source
from collaborative_test.SW_0186_full_encoder_qce import train as training_data
from collaborative_test.SW_0191_partition_error_feedback import resume_policy
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint
from .frequency_loss import make_targets,effective_frequency,frequency_objective

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive'
SEEDS=(0,1,2);COUNTS=(64,512);STEPS=128;BATCH=16;BASE_LR=3e-5;CONTEXT_LR=3e-4
sha=source.sha;write_json=source.write_json


def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def statistics_sha(found):
    def raw(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    return sha(raw(found.feature_mean).tobytes()+raw(found.feature_std).tobytes()+repr(float(found.feature_clip)).encode())


def require_job(seed,count):
    if type(seed) is not int or seed not in SEEDS or type(count) is not int or count not in COUNTS:raise ValueError('registered seed0/1/2 and TRAIN64/512 required')
    return seed,count


def archive_for(seed,count):require_job(seed,count);return ARCHIVE/f'seed{seed}'/f'TRAIN{count}'


def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('DESIGN_KO.md','frequency_loss.py','train.py','evaluate.py','queue.py','test_frequency_loss.py','test_pipeline.py')},
        source=source.fingerprint(),drive_implementation_sha256=sha((ROOT/'snn_kuramoto_bidirectional/gamma_initializer.py').read_bytes()),
        checkpoint_implementation_sha256=sha(Path(checkpoint.__file__).read_bytes()),seeds=list(SEEDS),TRAIN_counts=list(COUNTS),
        encoder_updates=STEPS,batch=BATCH,learning_rates=dict(source_base=BASE_LR,context_encoder=CONTEXT_LR),
        core_updates=0,native_grid=[64,64],production_readout_unchanged=True)


def optimizer_for(found):
    found.wrapped.requires_grad_(False);found.wrapped.eval();found.encoder.requires_grad_(True)
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for m in found.encoder.modules()):
        raise ValueError('registered source encoder must have no running statistics/dropout')
    groups=[dict(name='source_base',params=list(found.encoder.base_encoder.parameters()),lr=BASE_LR),
            dict(name='context_encoder',params=list(found.encoder.context.parameters()),lr=CONTEXT_LR)]
    parameters=[p for g in groups for p in g['params']]
    if len(parameters)!=len({id(p) for p in parameters}) or {id(p) for p in parameters}!={id(p) for p in found.encoder.parameters()}:
        raise ValueError('exact disjoint whole-encoder optimizer required')
    return torch.optim.Adam(groups)


def prepare_cpu(seed,count):
    require_job(seed,count)
    if torch.cuda.is_initialized():raise ValueError('CPU admission before CUDA required')
    found,dev,baseline,parent=source.prepare_cpu(seed)
    inputs=training_data.prepare_training_inputs_cpu();ids=list(map(int,inputs['ordered_train_ids'][:512]));rgb=np.asarray(inputs['arrays']['rgb'][:512]).copy()
    if len(ids)!=512 or len(set(ids))!=512 or any(1000<=i<1640 for i in ids):raise ValueError('fixed TRAIN pool overlaps evaluation')
    import h5py
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    with h5py.File(parent['dataset']['path'],'r') as stream:pixels=np.stack([stream['mask'][i,...,0] for i in ids])
    labels=source.source.full.metrics.independent.modal_labels(pixels,patch=2).astype(np.int64)
    exact=clevr_mask_patch(torch.from_numpy(pixels.copy()),2)['patch_labels'].cpu().numpy().astype(np.int64)
    if not np.array_equal(labels,exact) or rgb.shape!=(512,128,128,3) or rgb.dtype!=np.uint8 or labels.shape!=(512,64,64):raise ValueError('actual canonical128/native64 TRAIN geometry mismatch')
    if found.wrapped.core.kuramoto.omega.shape!=(4096,4) or found.wrapped.core.kuramoto.freq_gain is None:raise ValueError('actual4-component sensory-frequency source required')
    optimizer=optimizer_for(found)
    binding=dict(fingerprint=fingerprint(),seed=seed,TRAIN_count=count,TRAIN_ids=ids[:count],
        TRAIN_RGB_sha256=sha(rgb[:count].tobytes()),TRAIN_labels_sha256=sha(labels[:count].tobytes()),
        parent=parent,parent_model_sha256=source.MODELS[str(seed)],frozen_core_sha256=state_sha(found.wrapped.state_dict()),
        input_statistics_sha256=statistics_sha(found),DEV_RGB_sha256=sha(dev.tobytes()),DEV_baseline_sha256=sha(baseline.tobytes()),
        actual_frequency_gain=float(found.wrapped.core.kuramoto.freq_gain.detach()),DEV_GT_read=False)
    return dict(found=found,optimizer=optimizer,rgb=rgb[:count].copy(),labels=labels[:count].copy(),
        dev=dev,baseline=baseline,binding=binding,seed=seed,count=count)


def save_arrays(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(values) or any(not np.array_equal(old[k],v) for k,v in values.items()):raise ValueError('existing arrays changed')
        return
    stage=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with stage.open('xb') as stream:
        np.savez_compressed(stream,**values);stream.flush();os.fsync(stream.fileno())
    stage.rename(path)


def admit(seed,count):
    p=prepare_cpu(seed,count);a=archive_for(seed,count)
    save_arrays(a/'baseline.npz',labels=p['baseline'])
    value=dict(status='actual_native64_encoder_frequency_TRAIN_CPU_admitted',binding=p['binding'],CUDA_initialized=False,
        baseline_file_sha256=sha((a/'baseline.npz').read_bytes()))
    write_json(a/'admission.json',value);return value


def batch_indices(seed,count,step):
    if count<BATCH or count%BATCH or type(step) is not int or step<0:raise ValueError('complete fixed-size TRAIN batches required')
    epoch=(step*BATCH)//count;offset=(step*BATCH)%count
    return np.random.RandomState(221000+seed+10000*epoch).permutation(count)[offset:offset+BATCH].tolist()


def measure(found,rgb,targets,device):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(rgb),BATCH):
            frequency=effective_frequency(found.wrapped.core,found.encode(torch.from_numpy(rgb[start:start+BATCH]).to(device)))
            loss,parts=frequency_objective(frequency,{k:v[start:start+BATCH] for k,v in targets.items()})
            rows.append([float(loss.cpu()),float(parts['within'].cpu()),float(parts['adjacent_separation'].cpu())])
    values=np.asarray(rows).mean(0);return dict(total=float(values[0]),within=float(values[1]),adjacent_separation=float(values[2]))


def fit(prepared,device,runtime):
    found=prepared['found'];optimizer=prepared['optimizer'];seed=prepared['seed'];count=prepared['count'];a=archive_for(seed,count);device=torch.device(device)
    if state_sha(found.wrapped.state_dict())!=prepared['binding']['frozen_core_sha256']:raise ValueError('source core changed before training')
    targets=make_targets(prepared['labels']);dummy=torch.nn.Identity().to(device);initial=a/'initial.json';reference=a/'reference_frequency.npz'
    if not initial.exists():
        before=measure(found,prepared['rgb'],targets,device);frequency=[]
        with torch.no_grad():
            for image in prepared['dev']:
                frequency.append(effective_frequency(found.wrapped.core,found.encode(torch.from_numpy(image[None]).to(device)))[0].cpu().numpy())
        save_arrays(reference,frequency=np.stack(frequency))
        write_json(initial,dict(binding=prepared['binding'],TRAIN_phase_loss=before,reference_frequency_file_sha256=sha(reference.read_bytes()),
            initial_encoder_sha256=state_sha(found.encoder.state_dict())))
    init=json.loads(initial.read_bytes())
    if init['binding']!=prepared['binding'] or sha(reference.read_bytes())!=init['reference_frequency_file_sha256']:raise ValueError('initial frequency reference changed')
    checkpoint_binding=dict(binding=prepared['binding'],initial_metadata_sha256=sha(initial.read_bytes()));work=a/'encoder_work'
    if (work/'boundary.ptx').exists():
        loaded=checkpoint.load_latest(work,wrapped=dummy,encoder=found.encoder,optimizer=optimizer,expected_bindings=checkpoint_binding,expected_runtime=runtime,device=device,max_updates=STEPS)
        history=loaded['history']
    else:
        torch.manual_seed(221000+seed)
        if device.type=='cuda':torch.cuda.manual_seed(221000+seed)
        history=[]
        checkpoint.save_boundary(work,completed_updates=0,wrapped=dummy,encoder=found.encoder,optimizer=optimizer,history=history,bindings=checkpoint_binding,runtime=runtime,device=device)
    for step in range(len(history),STEPS):
        indices=batch_indices(seed,count,step);optimizer.zero_grad(set_to_none=True)
        frequency=effective_frequency(found.wrapped.core,found.encode(torch.from_numpy(prepared['rgb'][indices]).to(device)))
        loss,parts=frequency_objective(frequency,{k:v[indices] for k,v in targets.items()})
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite encoder-frequency loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(found.encoder.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite encoder gradient')
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in found.encoder.parameters()):raise FloatingPointError('nonfinite encoder parameter; preserve last committed boundary')
        history.append(dict(update=step+1,indices=indices,loss=float(loss.detach().cpu()),within=float(parts['within'].detach().cpu()),
            adjacent_separation=float(parts['adjacent_separation'].detach().cpu()),preclip_gradient_norm=float(norm.cpu())))
        checkpoint.save_boundary(work,completed_updates=len(history),wrapped=dummy,encoder=found.encoder,optimizer=optimizer,history=history,bindings=checkpoint_binding,runtime=runtime,device=device)
        del frequency,loss,parts
    if any(row['indices']!=batch_indices(seed,count,index) or row['update']!=index+1 for index,row in enumerate(history)):raise ValueError('encoder TRAIN order changed')
    steps=sorted({int(v['step'].item()) for v in optimizer.state.values()})
    if steps!=[STEPS] or state_sha(found.wrapped.state_dict())!=prepared['binding']['frozen_core_sha256'] or statistics_sha(found)!=prepared['binding']['input_statistics_sha256']:raise ValueError('encoder counter/frozen source invariant failed')
    path=a/'training_report.json'
    if path.exists():
        report=json.loads(path.read_bytes())
        if (report['binding']!=prepared['binding'] or report['checkpoint_binding']!=checkpoint_binding
            or report['history']!=history or report['actual_Adam_steps']!=steps
            or report['checkpoint_file_sha256']!=sha((work/'boundary.ptx').read_bytes())
            or report['final_encoder_sha256']!=state_sha(found.encoder.state_dict())
            or report['final_model_sha256']!=source.source.full.screen.native._state_sha(found.wrapped,found.encoder)):
            raise ValueError('completed encoder report changed')
        return report
    report=dict(status='complete_native64_encoder_frequency_pretraining',binding=prepared['binding'],checkpoint_binding=checkpoint_binding,
        history=history,actual_Adam_steps=steps,encoder_updates=STEPS,core_updates=0,unique_TRAIN_stage_images=count,logical_TRAIN_exposures=STEPS*BATCH,
        initial_TRAIN_phase_loss=init['TRAIN_phase_loss'],final_TRAIN_phase_loss=measure(found,prepared['rgb'],targets,device),
        final_encoder_sha256=state_sha(found.encoder.state_dict()),final_model_sha256=source.source.full.screen.native._state_sha(found.wrapped,found.encoder),
        frozen_core_sha256=state_sha(found.wrapped.state_dict()),input_statistics_sha256=statistics_sha(found),
        encoder_changed=state_sha(found.encoder.state_dict())!=init['initial_encoder_sha256'],reference_frequency_file_sha256=init['reference_frequency_file_sha256'],
        checkpoint_file_sha256=sha((work/'boundary.ptx').read_bytes()),DEV_GT_read=False,new70000_training=False,runtime=runtime)
    write_json(path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS,required=True);parser.add_argument('--count',type=int,choices=COUNTS,required=True);parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    if args.admit:print(json.dumps(admit(args.seed,args.count)))
    else:parser.error('use the admitted exclusive queue for GPU work')
