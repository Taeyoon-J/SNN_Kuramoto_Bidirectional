"""Durable center-head Adam updates; physical source is not in this optimizer."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import cache
from .center_head import CenterHead,normalization,targets,loss
from collaborative_test.SW_0207_spike_foreground_classifier.classifier import state_sha
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint
from collaborative_test.SW_0191_partition_error_feedback.resume_policy import compatible_runtime

TARGET_KEYS=('foreground','foreground_weights','heat','offset','offset_weights','object_count')

def batch_order(count,steps,*,batch=4,seed=211002):
    if count<batch or count%batch or steps<1:raise ValueError('whole positive TRAIN batches required')
    rng=np.random.RandomState(seed);result=[]
    while len(result)<steps:
        order=rng.permutation(count).tolist();result.extend([order[i:i+batch] for i in range(0,count,batch)])
    return result[:steps]

def prepare_targets(labels):
    rows=[targets(y) for y in labels]
    return {key:torch.from_numpy(np.stack([row[key] for row in rows])) for key in TARGET_KEYS}

def select_targets(values,indices,device):return {key:value[indices].to(device) for key,value in values.items()}

@torch.no_grad()
def full_train_loss(head,x,y,device):
    values=[]
    for index in range(0,len(x),4):
        indices=list(range(index,min(index+4,len(x))))
        value,_=loss(head(x[indices].to(device)),select_targets(y,indices,device));values.append(float(value))
    return float(np.mean(values))

def fit(features,labels,cache_protocol,*,device='cuda:0',steps=128,snapshots=cache.SNAPSHOTS,work=None,output=None,stop_after=None):
    if features.dtype!=np.float32 or features.ndim!=4 or labels.shape!=(len(features),*features.shape[2:]):raise ValueError('TRAIN feature/target geometry mismatch')
    if any(type(s) is not int or not 1<=s<=steps for s in snapshots):raise ValueError('invalid preregistered snapshots')
    device=torch.device(device);work=cache.ARCHIVE/'head_work' if work is None else Path(work)
    output=cache.ARCHIVE/'head_training' if output is None else Path(output)
    x=torch.from_numpy(features);y=prepare_targets(labels);mean,scale=normalization(features)
    order=batch_order(len(x),steps)
    fork_devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=fork_devices),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        torch.manual_seed(211002)
        head=CenterHead(channels=features.shape[1]).to(device);head.mean.copy_(torch.from_numpy(mean).to(device));head.scale.copy_(torch.from_numpy(scale).to(device))
        optimizer=torch.optim.Adam([dict(name='center_readout',params=list(head.parameters()),lr=3e-4)])
        empty=torch.nn.Identity();runtime=compatible_runtime(checkpoint.runtime_signature(device))
        binding=dict(science=cache.fingerprint(),cache_protocol_sha256=cache.sha(json.dumps(cache_protocol,sort_keys=True).encode()),
            features_sha256=cache.sha(features.tobytes()),TRAIN_labels_sha256=cache.sha(labels.tobytes()),
            mean_sha256=cache.sha(mean.tobytes()),scale_sha256=cache.sha(scale.tobytes()),steps=steps,snapshots=list(snapshots),batch_order=order,
            initial_model_sha256=state_sha(head.state_dict()),core_optimizer_steps=0)
        initial_path=output/'initial.json'
        if initial_path.exists():
            initial=json.loads(initial_path.read_bytes())
            if initial['binding']!=binding:raise ValueError('initial head or data changed')
        else:
            initial=dict(binding=binding,TRAIN_loss=full_train_loss(head,x,y,device));cache.write_json(initial_path,initial)
        if (work/'boundary.ptx').exists():
            restored=checkpoint.load_latest(work,wrapped=head,encoder=empty,optimizer=optimizer,expected_bindings=binding,expected_runtime=runtime,device=device,max_updates=steps)
            cursor=restored['completed_updates'];history=restored['history']
        else:
            if work.exists() and any(work.iterdir()) and not checkpoint.is_only_uncommitted_temp(work):raise FileExistsError('preserve unexpected head work')
            cursor=0;history=[]
            checkpoint.save_boundary(work,completed_updates=0,wrapped=head,encoder=empty,optimizer=optimizer,history=history,bindings=binding,runtime=runtime,device=device)
        if [r['indices'] for r in history]!=order[:cursor]:raise ValueError('restored head order changed')
        def snapshot(step):
            folder=output/f'step{step}';path=folder/'boundary.ptx';raw=(work/'boundary.ptx').read_bytes()
            if path.exists():
                if path.read_bytes()!=raw:raise ValueError('immutable head snapshot differs')
            else:folder.mkdir(parents=True,exist_ok=True);checkpoint._atomic_bytes(path,raw)
            record=dict(binding=binding,step=step,model_sha256=state_sha(head.state_dict()),boundary_sha256=cache.sha(raw),
                actual_Adam_steps=sorted({int(v['step'].item()) for v in optimizer.state.values()}),
                TRAIN_loss=full_train_loss(head,x,y,device),parameters=sum(p.numel() for p in head.parameters()))
            saved=folder/'report.json'
            if saved.exists():
                old=json.loads(saved.read_bytes())
                # Preserve exact weights/history; tolerate only reduction roundoff
                # in this diagnostic loss when a lease moves between equal GPUs.
                if any(old[k]!=record[k] for k in record if k!='TRAIN_loss') or not np.isclose(old['TRAIN_loss'],record['TRAIN_loss'],rtol=1e-6,atol=1e-6):raise ValueError('head snapshot report differs')
            else:cache.write_json(saved,record)
        if cursor in snapshots:snapshot(cursor)
        for index in range(cursor,steps):
            started=time.perf_counter();indices=order[index];head.train();optimizer.zero_grad(set_to_none=True)
            value,parts=loss(head(x[indices].to(device)),select_targets(y,indices,device));value.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('head gradient missing/nonfinite')
            gradient=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.))
            if not np.isfinite(gradient) or gradient<=0:raise FloatingPointError('head has no finite training credit')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in head.parameters()):raise FloatingPointError('nonfinite head parameter')
            history.append(dict(update=index+1,indices=indices,loss=float(value.detach()),parts=parts,gradient_norm=gradient,seconds=time.perf_counter()-started))
            checkpoint.save_boundary(work,completed_updates=index+1,wrapped=head,encoder=empty,optimizer=optimizer,history=history,bindings=binding,runtime=runtime,device=device)
            if index+1 in snapshots:snapshot(index+1)
            if stop_after is not None and index+1>=stop_after:return None
        actual_steps=sorted({int(v['step'].item()) for v in optimizer.state.values()})
        if actual_steps!=[steps]:raise ValueError('actual head Adam count mismatch')
        report=dict(status='completed_center_head_TRAIN_only',binding=binding,history=history,actual_Adam_steps=actual_steps,
            final_model_sha256=state_sha(head.state_dict()),initial_TRAIN_loss=initial['TRAIN_loss'],
            final_TRAIN_loss=full_train_loss(head,x,y,device),core_optimizer_steps=0,DEV_GT_read=False)
        final=output/'report.json'
        if final.exists():
            saved=json.loads(final.read_bytes())
            if any(saved[k]!=report[k] for k in report if k!='final_TRAIN_loss') or not np.isclose(saved['final_TRAIN_loss'],report['final_TRAIN_loss'],rtol=1e-6,atol=1e-6):raise ValueError('completed head report changed')
            report=saved
        else:cache.write_json(final,report)
        return report
