"""Real full-gradient admission with exact TRAIN35k initial-encoder statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from full_benchmark.model import FreshModel
from full_benchmark.loss import InitialTeacher,objective
from full_benchmark.geodesic import distance_power2
from full_benchmark.statistics import fit_clevr
from snn_kuramoto_bidirectional.sw240.geodesic import geodesic_distance_power2,inverse_softplus
from snn_kuramoto_bidirectional.sw240.data import train_ids
from mod_benchmark import common as c
from readout_benchmark import runtime as r


def digest(module):
    h=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def operator_test(device):
    rows=[]
    for dtype in (torch.float64,torch.float32):
        generator=torch.Generator().manual_seed(255)
        raw=torch.randn(1,19,5,generator=generator,dtype=dtype,device='cpu').to(device).requires_grad_()
        z=torch.nn.functional.normalize(raw,dim=-1)
        positions=torch.arange(19,device=device,dtype=dtype)*.25
        euclidean=(positions[:,None]-positions[None]).abs()
        contrast=torch.tensor(inverse_softplus(8.),device=device,dtype=dtype)
        reference=geodesic_distance_power2(z,euclidean,contrast,chunk=3)
        actual=distance_power2(z,euclidean,contrast)
        weight=torch.linspace(.1,1.,19**2,device=device,dtype=dtype).reshape_as(actual)
        grad0=torch.autograd.grad((reference*weight).sum(),raw,retain_graph=True)[0]
        grad1=torch.autograd.grad((actual*weight).sum(),raw)[0]
        tolerance=1e-9 if dtype==torch.float64 else 2e-4
        value_error=float((actual-reference).detach().abs().max());gradient_error=float((grad1-grad0).detach().abs().max())
        assert value_error<tolerance and gradient_error<tolerance,(dtype,value_error,gradient_error)
        rows.append(dict(dtype=str(dtype),maximum_value_error=value_error,maximum_gradient_error=gradient_error))
    return rows


def run(root,out,variant,seed):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    assert torch.cuda.is_available();device='cuda:0';c.immutable()
    algebra=operator_test(device);c.write(out/'operator_admission.json',dict(status='GEMM_values_and_gradients_verified',checks=algebra))
    c.write(out/'progress.json',dict(stage='constructing_fresh_model',variant=variant,seed=seed))
    model=FreshModel(variant,seed).to(device)
    normalization=fit_clevr(model,progress=lambda count:c.write(out/'progress.json',dict(stage='TRAIN_only_initial_statistics',completed=count,total=35000)))
    r.save(out/'normalization.pt',normalization)
    groups=model.parameter_groups();body_groups=[g for g in groups if g['name']!='mask_head']
    optimizer=torch.optim.Adam(body_groups)
    head_optimizer=torch.optim.Adam([g for g in groups if g['name']=='mask_head']) if model.head is not None else None
    teacher=InitialTeacher(model) if variant!=253 else None
    ids=train_ids(35000)[:2]
    images=r.read_rgb(root,'clevr','train',ids)
    labels=r.truth('clevr','train',ids) if variant!=253 else None
    if model.grid==32 and labels is not None:raise AssertionError('unexpected supervised native32 branch')
    checks=[];started=time.time();coverage={name:0. for name,p in model.named_parameters() if p.requires_grad}
    initial={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    for index in range(2):
        image=torch.from_numpy(images[index:index+1]).to(device)
        optimizer.zero_grad(set_to_none=True)
        if head_optimizer:head_optimizer.zero_grad(set_to_none=True)
        c.write(out/'progress.json',dict(stage='full_native_forward',step=index+1,variant=variant,seed=seed))
        output=model(image,inference_seed=255000+int(ids[index]),live_tail_steps=64)
        total,base,mask,parts=objective(model,output,rgb=image,
            labels=labels[index:index+1] if labels is not None else None,teacher=teacher)
        c.write(out/'progress.json',dict(stage='full_graph_core_backward',step=index+1,loss=float(total.detach())))
        total.backward()
        norms={g['name']:sum(float(p.grad.detach().abs().sum()) for p in g['params'] if p.grad is not None) for g in groups}
        missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
        if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('nonfinite full-gradient admission')
        for name,p in model.named_parameters():
            if p.requires_grad and p.grad is not None:coverage[name]+=float(p.grad.detach().abs().sum())
        torch.nn.utils.clip_grad_norm_([p for g in body_groups for p in g['params']],1.)
        if head_optimizer:torch.nn.utils.clip_grad_norm_(list(model.head.parameters()),1.)
        optimizer.step()
        if head_optimizer:head_optimizer.step()
        checks.append(dict(step=index+1,loss=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
            group_gradient_L1=norms,parameters_without_gradient=missing))
        del total,base,mask,output
        c.write(out/'progress.json',dict(stage='update_committed',step=index+1,gradient_L1=norms))
    changed={name:float((p.detach()-initial[name]).abs().max()) for name,p in model.named_parameters() if p.requires_grad}
    assert all(v>0 for v in coverage.values()),('inactive active parameter tensors',[name for name,v in coverage.items() if v==0])
    for g in groups:
        ids_in_group={id(p) for p in g['params']}
        assert any(changed[name]>0 for name,p in model.named_parameters() if id(p) in ids_in_group),g['name']
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),head_optimizer=head_optimizer.state_dict() if head_optimizer else None,
               variant=variant,seed=seed,steps=2,diagnostic_only=True,not_a_35000_training_checkpoint=True)
    r.save(out/'resume.pt',state)
    reloaded=torch.load(out/'resume.pt',map_location=device,weights_only=True)
    before=digest(model);model.load_state_dict(reloaded['model']);optimizer.load_state_dict(reloaded['optimizer'])
    if head_optimizer:head_optimizer.load_state_dict(reloaded['head_optimizer'])
    assert digest(model)==before and {int(s['step'].item()) for s in optimizer.state.values()}=={2}
    result=dict(status='actual_random_initialized_full_group_CUDA_admission_passed',variant=variant,seed=seed,
        grid=model.grid,TRAIN_ids=ids.tolist(),steps=2,total_rollout_steps=1024,gradient_tail_steps=64,
        all_graph_derivatives_live=True,pretrained_weights_loaded=False,target_mask_used=variant!=253,
        fixed_architectural_parameters=['wrapped.core.graph_generator.geodesic_contrast (T=8 constant)'],
        checks=checks,parameter_changes=changed,optimizer_resume_verified=True,benchmark_unchanged=c.immutable(),
        operator_checks=algebra,peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-started,
        active_tensor_gradient_coverage=coverage,
        normalization={k:v for k,v in normalization.items() if k not in ('mean','std')})
    c.write(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--variant',type=int,default=252,choices=[240,252,253,254]);parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();run(args.root,args.output,args.variant,args.seed)
