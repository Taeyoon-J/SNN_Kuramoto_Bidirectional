"""Balanced soft same-object relation loss, durable full-batch Adam."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collaborative_test.SW_0211_spike_anchor_center_assignment import training as old


class PairHead(nn.Module):
    def __init__(self,channels=548):
        super().__init__();self.register_buffer('mean',torch.zeros(1,channels));self.register_buffer('scale',torch.ones(1,channels))
        self.network=nn.Sequential(nn.Linear(channels,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,x):return self.network(((x-self.mean)/self.scale).clamp(-5,5)).flatten()


def objective(logits,target,weights):
    positive=target*weights;negative=(1-target)*weights
    if positive.sum()<=0 or negative.sum()<=0:raise ValueError('TRAIN needs positive and negative relation mass')
    return .5*((F.softplus(-logits)*positive).sum()/positive.sum()+(F.softplus(logits)*negative).sum()/negative.sum())


def fit(features,targets,weights,science,*,work,output,device='cuda:0',steps=128,stop_after=None):
    if features.ndim!=2 or features.dtype!=np.float32 or targets.shape!=(len(features),) or weights.shape!=targets.shape or not np.isfinite(features).all() or not np.isfinite(targets).all() or not np.isfinite(weights).all() or (weights<=0).any() or (targets<0).any() or (targets>1).any():raise ValueError('finite TRAIN pair table required')
    device=torch.device(device);work=Path(work);output=Path(output)
    mean=features.mean(0,dtype=np.float64).astype(np.float32);scale=np.sqrt(features.var(0,dtype=np.float64)).clip(1e-4).astype(np.float32)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        torch.manual_seed(214002);head=PairHead(features.shape[1]).to(device);head.mean.copy_(torch.from_numpy(mean[None]).to(device));head.scale.copy_(torch.from_numpy(scale[None]).to(device))
        optimizer=torch.optim.Adam([dict(name='component_relations',params=list(head.parameters()),lr=1e-3)]);empty=nn.Identity()
        runtime=old.compatible_runtime(old.checkpoint.runtime_signature(device))
        binding=dict(science=science,features_sha256=old.cache.sha(features.tobytes()),targets_sha256=old.cache.sha(targets.tobytes()),weights_sha256=old.cache.sha(weights.tobytes()),steps=steps,initial_model_sha256=old.state_sha(head.state_dict()))
        x=torch.from_numpy(features).to(device);y=torch.from_numpy(targets).to(device);w=torch.from_numpy(weights).to(device)
        initial=float(objective(head(x),y,w).detach())
        if (work/'boundary.ptx').exists():
            saved=old.checkpoint.load_latest(work,wrapped=head,encoder=empty,optimizer=optimizer,expected_bindings=binding,expected_runtime=runtime,device=device,max_updates=steps);cursor=saved['completed_updates'];history=saved['history']
        else:
            if work.exists() and any(work.iterdir()) and not old.checkpoint.is_only_uncommitted_temp(work):raise FileExistsError('preserve unexpected pair work')
            cursor=0;history=[];old.checkpoint.save_boundary(work,completed_updates=0,wrapped=head,encoder=empty,optimizer=optimizer,history=history,bindings=binding,runtime=runtime,device=device)
        for step in range(cursor,steps):
            optimizer.zero_grad(set_to_none=True);value=objective(head(x),y,w);value.backward()
            if not torch.isfinite(value) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('invalid relation loss/gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.));optimizer.step()
            if not np.isfinite(norm) or norm<=0 or any(not torch.isfinite(p).all() for p in head.parameters()):raise FloatingPointError('invalid relation update')
            history.append(dict(update=step+1,loss=float(value.detach()),gradient_norm=norm))
            old.checkpoint.save_boundary(work,completed_updates=step+1,wrapped=head,encoder=empty,optimizer=optimizer,history=history,bindings=binding,runtime=runtime,device=device)
            if stop_after is not None and step+1>=stop_after:return None
        actual=sorted({int(v['step'].item()) for v in optimizer.state.values()})
        if actual!=[steps]:raise ValueError('wrong actual Adam count')
        report=dict(binding=binding,history=history,actual_Adam_steps=actual,initial_TRAIN_loss=initial,final_TRAIN_loss=float(objective(head(x),y,w).detach()),model_sha256=old.state_sha(head.state_dict()),parameters=sum(p.numel() for p in head.parameters()),DEV_GT_read=False,core_optimizer_steps=0)
        path=output/'report.json'
        if path.exists():
            previous=json.loads(path.read_bytes())
            if any(previous[k]!=report[k] for k in report if k!='final_TRAIN_loss') or not np.isclose(previous['final_TRAIN_loss'],report['final_TRAIN_loss'],rtol=1e-6,atol=1e-6):raise ValueError('immutable relation report changed')
            report=previous
        else:old.cache.write_json(path,report)
        return report
