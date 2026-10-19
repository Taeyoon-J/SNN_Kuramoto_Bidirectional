"""Restore accepted model/Adam exactly instead of recomputing the first128 steps."""
import copy
import numpy as np
import torch


def inspect_source(prepared):
    from . import train,prefix_guard
    report,raw=prefix_guard.control_files(prepared);payload=prefix_guard.unpack(raw)
    steps=sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})
    if (payload['completed_updates']!=128 or len(payload['history'])!=128 or steps!=[128]
        or payload['bindings']!=report['checkpoint_binding']
        or payload['optimizer_schema']!=train.checkpoint.optimizer_schema(prepared['optimizer'])):
        raise ValueError('accepted model/Adam128/schema/binding mismatch')
    return report,payload


def restore_states(prepared,payload):
    from . import projection
    projection.module(prepared['found']).load_state_dict(payload['wrapped_state'],strict=True)
    prepared['found'].encoder.load_state_dict(payload['encoder_state'],strict=True)
    prepared['optimizer'].load_state_dict(payload['optimizer_state'])


def model_sha(prepared):
    from . import train
    return train.source.source.full.screen.native._state_sha(prepared['found'].wrapped,prepared['found'].encoder)


def admit_cpu(prepared):
    from . import train,projection
    if torch.cuda.is_initialized():raise ValueError('accepted-checkpoint admission must precede CUDA')
    found=prepared['found'];report,payload=inspect_source(prepared)
    old_encoder=copy.deepcopy(found.encoder.state_dict());old_projection=copy.deepcopy(projection.module(found).state_dict())
    old_optimizer=copy.deepcopy(prepared['optimizer'].state_dict());before=model_sha(prepared)
    frozen=projection.frozen_core_sha(found);rng=torch.get_rng_state().clone()
    try:
        restore_states(prepared,payload)
        if model_sha(prepared)!=report['final_model_sha256'] or projection.frozen_core_sha(found)!=frozen:
            raise ValueError('actual accepted full model not restored')
        n=min(16,len(prepared['rgb']));gamma=found.encode(torch.from_numpy(prepared['rgb'][:n]))
        frequency=train.effective_frequency(found.wrapped.core,gamma)
        loss,_=train.teacher_objective(gamma,frequency,{k:v[:n] for k,v in prepared['targets'].items()})
        grad=torch.autograd.grad(loss,projection.module(found).weight)[0]
        if not bool(torch.isfinite(grad).all()) or float(grad.norm())<=0:raise ValueError('accepted start has no finite gradient')
        result=dict(model_sha256=report['final_model_sha256'],checkpoint_sha256=report['checkpoint_file_sha256'],
            actual_Adam_steps=[128],loss=float(loss.detach()),projection_gradient_norm=float(grad.norm()),optimizer_updates=0)
    finally:
        found.encoder.load_state_dict(old_encoder,strict=True);projection.module(found).load_state_dict(old_projection,strict=True)
        prepared['optimizer'].load_state_dict(old_optimizer);torch.set_rng_state(rng)
    if model_sha(prepared)!=before or any(p.grad is not None for p in list(found.encoder.parameters())+projection.parameters(found)):
        raise ValueError('read-only accepted-checkpoint admission mutated source')
    return result


def start(prepared,bindings,work,runtime,device):
    from . import train,projection,prefix_guard
    report,_=inspect_source(prepared)
    if (work/'boundary.ptx').exists():raise ValueError('preserve existing child checkpoint; use normal resume')
    source=train.accepted.archive_for(prepared['seed'],512)/'encoder_work'
    loaded=train.checkpoint.load_latest(source,wrapped=projection.module(prepared['found']),encoder=prepared['found'].encoder,
        optimizer=prepared['optimizer'],expected_bindings=report['checkpoint_binding'],expected_runtime=runtime,device=device,max_updates=128)
    history=loaded['history']
    if len(history)!=128 or model_sha(prepared)!=report['final_model_sha256']:
        raise ValueError('accepted128 actual model/history restoration failed')
    train.checkpoint.save_boundary(work,completed_updates=128,wrapped=projection.module(prepared['found']),encoder=prepared['found'].encoder,
        optimizer=prepared['optimizer'],history=history,bindings=bindings,runtime=runtime,device=device)
    prefix_guard.verify_and_save(prepared,history,work/'boundary.ptx')
    return history
