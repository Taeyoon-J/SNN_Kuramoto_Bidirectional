"""Mine once per proposal; freeze targets across gradient and post-loss checks."""
import hashlib
import numpy as np
import torch
from collaborative_test.SW_0179_qce_seed_replication import train as base
from collaborative_test.SW_0175_critical_component_binding import train_v2 as native
from collaborative_test.SW_0136_native32_transfer import evaluate as masks
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver
from . import objective


def _digest(indices,rows):
    h=hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes())
    for index in indices:h.update(np.asarray(rows[int(index)],dtype='<i8').tobytes())
    return h.hexdigest()


def install(found,indices,rows):
    indices=tuple(map(int,indices))
    if len(set(indices))!=len(indices) or set(indices)!=set(rows):raise ValueError('feedback context indices differ')
    copied={i:np.asarray(rows[i],dtype=np.int64).copy() for i in indices}
    if any(r.ndim!=1 or np.any(r<0) for r in copied.values()):raise ValueError('invalid frozen production labels')
    found._sw0191_feedback_context=dict(indices=indices,rows=copied,sha256=_digest(indices,copied))


def context_sha(found):
    context=found._sw0191_feedback_context
    if context['sha256']!=_digest(context['indices'],context['rows']):raise ValueError('feedback targets mutated')
    return context['sha256']


def require_batch(found,indices):
    if found._sw0191_feedback_context['indices']!=tuple(map(int,indices)):
        raise ValueError('stale feedback batch')
    context_sha(found)


def get_prediction(found,index,device):
    return torch.tensor(found._sw0191_feedback_context['rows'][int(index)],dtype=torch.int64,device=device)


@torch.no_grad()
def prepare_and_measure(found,image_indices,inputs,device):
    rows={};losses=[]
    for offset in range(0,base.BATCH,base.MICRO):
        indices=image_indices[offset:offset+base.MICRO]
        rgb=torch.from_numpy(np.ascontiguousarray(inputs['arrays']['rgb'][indices])).to(device)
        target=base.flatten_modal_targets(inputs['arrays']['modal_gt'][indices]).to(device)
        _,trace,adj,z,q,rho=native._native_forward(found,rgb,live_tail_steps=0)
        qcc=np.asarray(masks._qcc_labels(trace),dtype=np.int32)
        gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().cpu())
        for local,index in enumerate(indices):
            labels=np.asarray(solver.foreground_induced_merge(q[local].cpu().numpy().astype(np.float64),
                adj[local].cpu().numpy().astype(np.float64),gain,qcc[local],grid_shape=(32,32))['labels'],dtype=np.int64).reshape(-1)
            rows[int(index)]=labels
            pred=torch.from_numpy(labels).to(device)
            losses.append(float(objective.loss_for_image(q[local],rho[local],target[local],pred).cpu()))
        del rgb,target,trace,adj,z,q,rho
    if len(losses)!=base.BATCH or not np.isfinite(losses).all():raise FloatingPointError('nonfinite/incomplete proposal mining loss')
    install(found,image_indices,rows)
    return float(np.mean(losses))
