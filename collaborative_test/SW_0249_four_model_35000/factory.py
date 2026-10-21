"""Restore actual named checkpoints; recreate only the two authorized missing seeds."""
import io,json
from pathlib import Path
import numpy as np
import torch
from . import scope
from collaborative_test.SW_0240_native64_70000_cached_io import train as old,data
from collaborative_test.SW_0244_native64_within_precision import run as precision

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'collaborative_testing/generated/SW0249_four_model_35000'
sha=old.sha;state_sha=old.state_sha

def archive(model,seed):scope.require(model,seed);return ARCHIVE/f'model{model}'/f'seed{seed}'

def reproduction(seed):return ARCHIVE/'reproduce244'/f'seed{seed}'

def payload(path,expected=None):
    raw=path.read_bytes()
    if expected and sha(raw)!=expected:raise ValueError('source checkpoint hash differs')
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('source checkpoint envelope differs')
    return torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)

def model_sha(found):return old.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)

def load_weights(found,path,expected=None,with_projection=True):
    saved=payload(path,expected)
    found.encoder.load_state_dict(saved['encoder_state'],strict=True)
    if with_projection:old.projection.module(found).load_state_dict(saved['wrapped_state'],strict=True)
    return saved

def original237(found,seed):
    a=ROOT/'collaborative_test/SW_0232_native64_exact_teacher_resume/results_archive'/f'seed{seed}'/'TRAIN64'
    report=json.loads((a/'training_report.json').read_bytes())
    expected=report.get('checkpoint_file_sha256',report.get('checkpoint_sha256'))
    if not expected:raise ValueError('source237 checkpoint digest missing')
    saved=load_weights(found,a/'encoder_work/boundary.ptx',expected)
    if saved['completed_updates']!=512 or model_sha(found)!=report['final_model_sha256']:raise ValueError('wrong actual237 source')
    return dict(experiment='237',checkpoint_sha256=expected,model_sha256=model_sha(found),runtime=report['runtime'])

def source(found,model,seed):
    ref=original237(found,seed)
    if model=='237':return ref
    if model=='240':a=old.archive_for(seed);with_projection=True
    elif model=='244' and seed in (0,2):
        a=reproduction(seed);with_projection=True
    else:
        name='SW_0244_native64_within_precision' if model=='244' else 'SW_0245_native64_precision_three_seed64'
        a=ROOT/'collaborative_test'/name/'results_archive/within16'/f'seed{seed}';with_projection=False
    report=json.loads((a/'training_report.json').read_bytes())
    saved=load_weights(found,a/'encoder_work/boundary.ptx',report['checkpoint_sha256'],with_projection)
    expected_steps=43750 if model=='240' else 128
    if saved['completed_updates']!=expected_steps or model_sha(found)!=report['final_model_sha256']:raise ValueError('wrong named final model')
    return dict(experiment=model,checkpoint_sha256=report['checkpoint_sha256'],model_sha256=model_sha(found),runtime=report['runtime'],
                reproduced_missing_original_seed=model=='244' and seed in (0,2))

def prepare(model,seed,for_reproduction=False):
    scope.require(model,seed)
    if torch.cuda.is_initialized():raise ValueError('source/teacher CPU preparation before CUDA')
    teacher,_,_,binding=old.source.prepare_cpu(seed)
    found,_,_,binding2=old.source.prepare_cpu(seed)
    if binding!=binding2:raise ValueError('foundation factory mismatch')
    teacher.wrapped.requires_grad_(False);teacher.encoder.requires_grad_(False)
    teacher.wrapped.eval();teacher.encoder.eval()
    ref=original237(found,seed) if for_reproduction else source(found,model,seed)
    if old.helpers.statistics_sha(found)!=old.helpers.statistics_sha(teacher):raise ValueError('input statistics differ')
    opt=old.helpers.optimizer_for(found)
    if not for_reproduction:opt=old.projection.enable(found,opt)
    return dict(found=found,teacher=teacher,optimizer=opt,source=ref,dataset=binding['dataset'],model=model,seed=seed,
                frozen_core_sha256=old.projection.frozen_core_sha(found),full_core_sha256=state_sha(found.wrapped.state_dict()),statistics_sha256=old.helpers.statistics_sha(found))

def need_reproduction(model,seed):return model=='244' and seed in (0,2) and not (reproduction(seed)/'training_report.json').exists()

def reproduction_preflight(seed):
    if seed not in (0,2):raise ValueError('only original missing244 seeds')
    p=prepare('244',seed,for_reproduction=True)
    module=ROOT/'collaborative_test/SW_0245_native64_precision_three_seed64/results_archive/within16'/f'seed{seed}'
    ad=json.loads((module/'admission.json').read_bytes());ids=ad['binding']['TRAIN_ids']
    rgb,labels=data.read_batch(None,np.asarray(ids,dtype=np.int64))
    if sha(rgb.tobytes())!=ad['binding']['TRAIN_RGB_sha256'] or sha(labels.tobytes())!=ad['binding']['TRAIN_labels_sha256']:raise ValueError('original244 training source differs')
    with np.load(module/'teacher.npz',allow_pickle=False) as values:targets={k:torch.from_numpy(values[k].copy()) for k in values.files}
    if state_sha(targets)!=ad['binding']['teacher_arrays_sha256']:raise ValueError('original teacher arrays differ')
    gamma=p['found'].encode(torch.from_numpy(rgb[:16]));freq=old.helpers.effective_frequency(p['found'].wrapped.core,gamma)
    groups=precision.groups_from_labels(labels[:16]);loss,_=precision.objective(gamma,freq,{k:v[:16] for k,v in targets.items()},groups,16.)
    gradients=torch.autograd.grad(loss,tuple(p['found'].encoder.parameters()))
    if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(g).all()) for g in gradients):raise ValueError('reproduction CPU gradient failed')
    if state_sha(p['found'].wrapped.state_dict())!=p['full_core_sha256'] or p['optimizer'].state:raise ValueError('reproduction preflight mutated core/optimizer')
    result=dict(status='original244_missing_seed_CPU_preflight_passed',seed=seed,source=p['source'],TRAIN_ids=ids,
                teacher_sha256=ad['binding']['teacher_arrays_sha256'],loss=float(loss.detach()),optimizer_updates=0,original_core_projection_frozen=True)
    old.write_json(reproduction(seed)/'CPU_preflight.json',result)
    return result

def reproduce(seed,device):
    # Original244 128-update encoder-only recipe; no structure or loss changes.
    p=prepare('244',seed,for_reproduction=True);found=p['found'];opt=p['optimizer'];a=reproduction(seed)
    proof=json.loads((a/'CPU_preflight.json').read_bytes())
    if proof['source']!=p['source'] or proof['seed']!=seed:raise ValueError('reproduction admission source differs')
    module=ROOT/'collaborative_test/SW_0245_native64_precision_three_seed64/results_archive/within16'/f'seed{seed}'
    ad=json.loads((module/'admission.json').read_bytes());ids=ad['binding']['TRAIN_ids']
    rgb,labels=data.read_batch(None,np.asarray(ids,dtype=np.int64))
    if sha(rgb.tobytes())!=ad['binding']['TRAIN_RGB_sha256'] or sha(labels.tobytes())!=ad['binding']['TRAIN_labels_sha256']:raise ValueError('original244 TRAIN tensors differ')
    with np.load(module/'teacher.npz',allow_pickle=False) as values:targets={k:torch.from_numpy(values[k].copy()) for k in values.files}
    if len(ids)!=64 or old.helpers.state_sha(targets)!=ad['binding']['teacher_arrays_sha256']:raise ValueError('original244 teacher/input mismatch')
    groups=precision.groups_from_labels(labels)
    torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    old.source.source.source.model64.move_to_device(found,device)
    targets={k:v.to(device) for k,v in targets.items()};groups=precision.groups_to_device(groups,device)
    runtime=old.resume_policy.compatible_runtime(old.checkpoint.runtime_signature(device))
    if runtime!=p['source']['runtime']:raise ValueError('source runtime differs')
    binding=dict(kind='authorized_missing244_reproduction',seed=seed,source=p['source'],TRAIN_ids=ids,teacher_sha256=ad['binding']['teacher_arrays_sha256'])
    work=a/'encoder_work'
    if (work/'boundary.ptx').exists():
        saved=old.checkpoint.load_latest(work,wrapped=old.projection.module(found),encoder=found.encoder,optimizer=opt,
            expected_bindings=binding,expected_runtime=runtime,device=device,max_updates=128);history=saved['history']
    else:torch.manual_seed(244000+seed);torch.cuda.manual_seed(244000+seed);history=[]
    for step in range(len(history),128):
        idx=precision.batch_indices(seed,64,step);opt.zero_grad(set_to_none=True)
        gamma=found.encode(torch.from_numpy(rgb[idx]).to(device));freq=old.helpers.effective_frequency(found.wrapped.core,gamma)
        loss,_=precision.objective(gamma,freq,{k:v[idx] for k,v in targets.items()},tuple(groups[i] for i in idx),16.)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(found.encoder.parameters(),1.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):raise ValueError('nonfinite reproduction gradient')
        opt.step();history.append(dict(update=step+1,indices=idx,loss=float(loss.detach()),gradient_norm=float(norm)))
        if len(history)%16==0:old.checkpoint.save_boundary(work,completed_updates=len(history),wrapped=old.projection.module(found),encoder=found.encoder,optimizer=opt,history=history,bindings=binding,runtime=runtime,device=device)
    if state_sha(found.wrapped.state_dict())!=p['full_core_sha256']:raise ValueError('reproduction changed core/projection')
    if {int(v['step'].item()) for v in opt.state.values()}!={128}:raise ValueError('wrong reproduction Adam steps')
    old.write_json(a/'training_report.json',dict(status='completed_authorized244_seed_reproduction',binding=binding,runtime=runtime,
        original_checkpoint_existed=False,original_recipe_preserved=True,new_core_updates=0,
        final_model_sha256=model_sha(found),checkpoint_sha256=sha((work/'boundary.ptx').read_bytes())))
