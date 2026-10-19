"""Rebind a frozen239 boundary without changing model, Adam, RNG or TRAIN history."""
import copy
import io
import json
from pathlib import Path
import torch
from collaborative_test.SW_0239_native64_best_70000 import train as reference


def entry(seed):
    from . import train
    raw=(train.ARCHIVE/'migration.json').read_bytes();manifest=json.loads(raw)
    if not manifest['old_single_GPU_owner_confirmed_stopped']:
        raise ValueError('239 owner must stop before successor registration')
    if manifest['source239_fingerprint']!=reference.fingerprint():
        raise ValueError('original registered239 source changed')
    return manifest['seeds'][str(seed)]


def confirm_old_owner_stopped():
    from . import train
    manifest=json.loads((train.ARCHIVE/'migration.json').read_bytes())
    if not manifest['old_single_GPU_owner_confirmed_stopped']:
        raise ValueError('original239 GPU owner has not stopped')
    for identity in manifest['stopped_processes']:
        path=Path('/proc')/str(identity['pid'])/'stat'
        if path.exists():
            raw=path.read_text();tail=raw[raw.rfind(')')+2:].split()
            if int(tail[19])==identity['start_ticks'] and tail[0] not in ('Z','X'):
                raise ValueError('original239 worker/supervisor still live; never lease a second family GPU')


def inspect_parent(p):
    from . import train
    item=entry(p['seed'])
    if item['completed_updates']==0 and item['checkpoint_sha256'] is None:return None
    path=train.archive_for(p['seed'])/'inherited239/boundary.ptx';raw=path.read_bytes()
    if train.sha(raw)!=item['checkpoint_sha256']:raise ValueError('preserved239 boundary changed')
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    encoded=outer['payload_bytes']
    if train.sha(encoded)!=outer['payload_sha256']:raise ValueError('preserved239 envelope corrupted')
    payload=torch.load(io.BytesIO(encoded),map_location='cpu',weights_only=False)
    n=payload['completed_updates']
    if (n!=item['completed_updates'] or len(payload['history'])!=n or not 0<n<=train.data.TOTAL_STEPS
            or payload['bindings']['binding']!=p['original239_binding']
            or payload['bindings']['admission_sha256']!=item['original239_admission_sha256']
            or payload['optimizer_schema']!=train.checkpoint.optimizer_schema(p['optimizer'])
            or sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})!=[n]):
        raise ValueError('preserved239 model/Adam/history/binding differs')
    train.check_history(payload['history'],p['seed'])
    return payload


def admit_cpu(p):
    from . import train
    if torch.cuda.is_initialized():raise ValueError('CPU-only checkpoint resume admission required')
    payload=inspect_parent(p)
    if payload is None:return dict(inherited239_updates=0,fresh_registered_seed=True,optimizer_updates=0)
    found=p['found'];encoder=copy.deepcopy(found.encoder.state_dict())
    projection=copy.deepcopy(train.projection.module(found).state_dict())
    optimizer=copy.deepcopy(p['optimizer'].state_dict());rng=torch.get_rng_state().clone()
    before=train.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)
    try:
        found.encoder.load_state_dict(payload['encoder_state'],strict=True)
        train.projection.module(found).load_state_dict(payload['wrapped_state'],strict=True)
        p['optimizer'].load_state_dict(payload['optimizer_state'])
        import h5py
        step=min(payload['completed_updates'],train.data.TOTAL_STEPS-1)
        with h5py.File(p['binding']['dataset']['path'],'r') as stream:
            rgb,labels=train.data.read_batch(stream,train.data.batch_ids(p['seed'],step))
        probe=train.gradient_probe(p,rgb,labels)
        result=dict(inherited239_updates=payload['completed_updates'],actual_CPU_model_Adam_gradient_restore_passed=True,
            restored_model_sha256=train.source.source.full.screen.native._state_sha(found.wrapped,found.encoder),
            probe=probe,optimizer_updates=0)
    finally:
        found.encoder.load_state_dict(encoder,strict=True)
        train.projection.module(found).load_state_dict(projection,strict=True)
        p['optimizer'].load_state_dict(optimizer);torch.set_rng_state(rng)
    if train.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=before or p['optimizer'].state:
        raise ValueError('CPU checkpoint probe mutated registered initial state')
    return result


def start(p,bindings,work,runtime,device):
    from . import train
    if (work/'boundary.ptx').exists():raise ValueError('use normal child resume for an existing boundary')
    payload=inspect_parent(p)
    if payload is None:
        torch.manual_seed(239000+p['seed']);torch.cuda.manual_seed(239000+p['seed']);history=[]
    else:
        loaded=train.checkpoint.load_latest(train.archive_for(p['seed'])/'inherited239',
            wrapped=train.projection.module(p['found']),encoder=p['found'].encoder,optimizer=p['optimizer'],
            expected_bindings=payload['bindings'],expected_runtime=runtime,device=device,max_updates=train.data.TOTAL_STEPS)
        history=loaded['history']
        if train.state_sha(p['found'].encoder.state_dict())!=train.state_sha(payload['encoder_state']) or train.state_sha(train.projection.module(p['found']).state_dict())!=train.state_sha(payload['wrapped_state']):
            raise ValueError('actual resumed parameter bits differ from preserved239')
    train.checkpoint.save_boundary(work,completed_updates=len(history),wrapped=train.projection.module(p['found']),
        encoder=p['found'].encoder,optimizer=p['optimizer'],history=history,bindings=bindings,runtime=runtime,device=device)
    train.write_json(train.archive_for(p['seed'])/'inherited_boundary_proof.json',dict(
        inherited239_updates=len(history),parent=entry(p['seed']),Adam_model_RNG_history_restored=True,
        resumed_full_model_sha256=train.source.source.full.screen.native._state_sha(p['found'].wrapped,p['found'].encoder)))
    return history
