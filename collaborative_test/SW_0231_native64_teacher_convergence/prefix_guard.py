"""Prove the first128 updates reproduce the accepted SW0223 model and Adam."""
import io,json
from pathlib import Path
import numpy as np
import torch


def equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,np.ndarray):
        return isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(equal(x,y) for x,y in zip(left,right))
    return type(left)==type(right) and left==right


def compare_payloads(current,control):
    if current['completed_updates']!=128 or control['completed_updates']!=128:
        raise ValueError('actual128-update prefixes required')
    for key in ('wrapped_state','encoder_state','optimizer_state','optimizer_schema','history'):
        if not equal(current[key],control[key]):raise ValueError('accepted128 prefix differs: '+key)
    for key in ('torch_cpu','torch_cuda'):
        if not equal(current['rng'].get(key),control['rng'].get(key)):raise ValueError('accepted128 Torch RNG differs: '+key)


def unpack(raw):
    from . import train
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if train.sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('prefix payload checksum changed')
    return torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)


def control_files(prepared):
    from . import train,projection
    folder=projection.control.archive_for(prepared['seed'],64)
    raw=(folder/'training_report.json').read_bytes();report=json.loads(raw)
    ref=prepared['binding']['matched_128_control']
    checkpoint=(folder/'encoder_work/boundary.ptx').read_bytes()
    if train.sha(raw)!=ref['report_sha256'] or report['final_model_sha256']!=ref['model_sha256'] or train.sha(checkpoint)!=report['checkpoint_file_sha256']:
        raise ValueError('accepted128 checkpoint/report changed')
    return report,checkpoint


def verify_and_save(prepared,history,path):
    from . import train,projection
    report,control=control_files(prepared);raw=Path(path).read_bytes();current=unpack(raw)
    compare_payloads(current,unpack(control))
    if current['history']!=history or current['bindings']['binding']!=prepared['binding']:
        raise ValueError('prefix is not this actual training trajectory')
    if (train.state_sha(current['encoder_state'])!=train.state_sha(prepared['found'].encoder.state_dict())
        or train.state_sha(current['wrapped_state'])!=train.state_sha(projection.module(prepared['found']).state_dict())):
        raise ValueError('prefix snapshot is not the current actual encoder/projection')
    model=train.source.source.full.screen.native._state_sha(prepared['found'].wrapped,prepared['found'].encoder)
    if model!=report['final_model_sha256']:raise ValueError('whole actual128 model differs')
    folder=train.archive_for(prepared['seed'],64);snapshot=folder/'prefix128.ptx'
    if snapshot.exists():
        if snapshot.read_bytes()!=raw:raise ValueError('immutable prefix snapshot changed')
    else:snapshot.write_bytes(raw)
    proof=dict(binding=prepared['binding'],completed_updates=128,prefix_checkpoint_sha256=train.sha(raw),
        matched_checkpoint_sha256=train.sha(control),matched_model_sha256=model,
        encoder_projection_Adam_history_and_Torch_RNG_bitwise_equal=True)
    target=folder/'prefix128_proof.json'
    if target.exists():
        if json.loads(target.read_bytes())!=proof:raise ValueError('prefix proof changed')
    else:train.write_json(target,proof)
    return proof


def validate_saved(prepared):
    from . import train
    folder=train.archive_for(prepared['seed'],64);proof=json.loads((folder/'prefix128_proof.json').read_bytes())
    _,control=control_files(prepared);raw=(folder/'prefix128.ptx').read_bytes();current=unpack(raw)
    if (proof['binding']!=prepared['binding'] or current['bindings']['binding']!=prepared['binding']
        or proof['prefix_checkpoint_sha256']!=train.sha(raw) or proof['matched_checkpoint_sha256']!=train.sha(control)
        or proof['matched_model_sha256']!=prepared['binding']['matched_128_control']['model_sha256']
        or proof['completed_updates']!=128 or proof['encoder_projection_Adam_history_and_Torch_RNG_bitwise_equal'] is not True):
        raise ValueError('invalid saved actual128 prefix')
    compare_payloads(current,unpack(control));return proof
