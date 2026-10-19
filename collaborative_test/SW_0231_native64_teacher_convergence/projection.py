"""Learn the real sensory projection; freeze every other core state tensor."""
import json
import numpy as np
import torch
from collaborative_test.SW_0223_native64_joint_sensory_projection import train as control,evaluate as control_eval

KEY='core.gamma_channel_proj.weight'


def module(found):
    value=found.wrapped.core.gamma_channel_proj
    if value is None or set(value.state_dict())!={'weight'} or value.weight.ndim!=2:raise ValueError('bias-free actual channel projection required')
    return value


def parameters(found):return list(module(found).parameters())


def frozen_core_sha(found):
    values=found.wrapped.state_dict()
    if KEY not in values:raise ValueError('registered actual projection state key missing')
    return control.state_sha({k:v for k,v in values.items() if k!=KEY})


def enable(found,optimizer):
    value=module(found)
    if any(id(p)==id(value.weight) for group in optimizer.param_groups for p in group['params']):raise ValueError('projection already optimized')
    value.requires_grad_(True)
    if {k for k,v in found.wrapped.named_parameters() if v.requires_grad}!={KEY}:raise ValueError('only the declared core projection may learn')
    optimizer.add_param_group(dict(name='sensory_projection',params=parameters(found),lr=control.parent.CONTEXT_LR))
    return optimizer


def admit_gradient_cpu(prepared):
    if torch.cuda.is_initialized():raise ValueError('gradient admission must precede CUDA')
    found=prepared['found'];count=min(16,len(prepared['rgb']))
    before=control.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)
    gamma=found.encode(torch.from_numpy(prepared['rgb'][:count]));frequency=control.effective_frequency(found.wrapped.core,gamma)
    loss,_=control.teacher_objective(gamma,frequency,{k:v[:count] for k,v in prepared['targets'].items()})
    gradient=torch.autograd.grad(loss,module(found).weight)[0]
    if not bool(torch.isfinite(gradient).all()) or float(gradient.norm())<=0:raise ValueError('actual sensory projection has no finite gradient credit')
    if before!=control.source.source.full.screen.native._state_sha(found.wrapped,found.encoder) or any(p.grad is not None for p in list(found.encoder.parameters())+parameters(found)):raise ValueError('read-only CPU gradient admission mutated source')
    return dict(images=count,loss=float(loss.detach()),projection_gradient_norm=float(gradient.norm()),optimizer_updates=0,source_unchanged=True)


def control_reference(seed,binding):
    folder=control.archive_for(seed,64);score_raw=(control.ARCHIVE/'score_v1/summary.json').read_bytes();score=json.loads(score_raw)
    audit=json.loads((control.ARCHIVE/'completion_audit_v1/summary.json').read_bytes());report_raw=(folder/'training_report.json').read_bytes();report=json.loads(report_raw);protocol_raw=(folder/'protocol.json').read_bytes();protocol=json.loads(protocol_raw)
    if score['fingerprint']!=control.fingerprint() or audit['score_sha256']!=control.sha(score_raw):raise ValueError('accepted encoder-only control changed')
    previous=report['binding']
    for key in ('TRAIN_ids','TRAIN_RGB_sha256','TRAIN_labels_sha256','DEV_RGB_sha256','DEV_baseline_sha256','teacher_arrays_sha256','parent_model_sha256'):
        if previous[key]!=binding[key]:raise ValueError('different matched source/data/teacher: '+key)
    if protocol['binding']!=dict(training_report_sha256=control.sha(report_raw),binding=previous,model_sha256=report['final_model_sha256']) or protocol['DEV_GT_read'] is not False:raise ValueError('wrong encoder-only prediction protocol')
    proof=audit['checkpoints'][str(seed)]
    if proof['protocol_sha256']!=control.sha(protocol_raw) or proof['checkpoint_sha256']!=control.sha((folder/'encoder_work/boundary.ptx').read_bytes()) or report['checkpoint_file_sha256']!=proof['checkpoint_sha256']:raise ValueError('encoder-only actual checkpoint changed')
    labels=np.stack([control_eval.read_row(folder/'rows'/f'{i:04d}',protocol['binding'],i)[0] for i in range(16)])
    if control.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('encoder-only mask bytes changed')
    ref=dict(score_sha256=control.sha(score_raw),audit_sha256=control.sha((control.ARCHIVE/'completion_audit_v1/summary.json').read_bytes()),
        report_sha256=control.sha(report_raw),protocol_sha256=control.sha(protocol_raw),model_sha256=report['final_model_sha256'],labels_sha256=control.sha(labels.tobytes()))
    return labels,ref
