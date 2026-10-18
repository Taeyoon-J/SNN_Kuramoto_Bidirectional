"""Supervised foreground from actual four-gating temporal node statistics."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ForegroundClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(12));self.register_buffer('scale',torch.ones(12))
        self.layers=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.layers(((x-self.mean)/self.scale).clamp(-5,5)).squeeze(-1)

def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()



def sample_nodes(labels,*,seed,count=8192):
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.dtype.kind not in 'iu' or (labels<0).any():raise ValueError('nonnegative GT grid required')
    if type(count) is not int or count<2 or count%2:raise ValueError('even node budget required')
    flat=labels.reshape(-1);background=np.flatnonzero(flat==0)
    groups=[np.flatnonzero(flat==c) for c in np.unique(flat) if c>0]
    if not len(background) or not groups:raise ValueError('both TRAIN foreground and BG required')
    rng=np.random.RandomState(seed);half=count//2;choice=rng.randint(0,len(groups),half)
    indices=np.empty(count,dtype=np.int64)
    for k,g in enumerate(groups):
        where=np.flatnonzero(choice==k);indices[where]=g[rng.randint(0,len(g),len(where))]
    indices[half:]=background[rng.randint(0,len(background),half)]
    target=np.r_[np.ones(half,np.float32),np.zeros(half,np.float32)]
    if not np.array_equal(flat[indices]>0,target.astype(bool)):raise ValueError('invalid foreground samples')
    return indices,target


def node_features(trace,*,settle=512):
    spikes=trace['component_spikes'];membrane=trace['component_membrane']
    if spikes.ndim!=4 or spikes.shape[1]!=4 or membrane.shape!=spikes.shape:raise ValueError('[B,4,N,T] component traces required')
    if type(settle) is not int or not 0<=settle<spikes.shape[-1]:raise ValueError('invalid settled prefix')
    s=spikes[...,settle:].float();m=membrane[...,settle:].float()
    features=torch.cat((s.mean(-1),m.mean(-1),m.std(-1,unbiased=False)),dim=1).transpose(1,2).contiguous()
    if not torch.isfinite(features).all():raise ValueError('nonfinite node features')
    return features


def fit(features,target,*,seed=207002):
    x=np.asarray(features);y=np.asarray(target)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=12 or y.dtype!=np.float32 or y.shape!=(len(x),) or not len(x):raise ValueError('float32 node features[N,12] and target[N] required')
    if not np.isfinite(x).all() or not np.isin(y,[0.,1.]).all() or np.unique(y).size!=2:raise ValueError('finite features and both binary classes required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);head=ForegroundClassifier()
        inputs=torch.from_numpy(np.ascontiguousarray(x));labels=torch.from_numpy(np.ascontiguousarray(y))
        head.mean.copy_(inputs.mean(0));head.scale.copy_(inputs.std(0,unbiased=False).clamp_min(1e-4))
        if not torch.isfinite(head.mean).all() or not torch.isfinite(head.scale).all():raise ValueError('nonfinite TRAIN normalization')
        initial=state_sha(head.state_dict());optimizer=torch.optim.Adam(head.parameters(),lr=.001)
        before=float(F.binary_cross_entropy_with_logits(head(inputs),labels).detach())
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(head(inputs),labels)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite classifier objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('nonfinite classifier gradient')
            optimizer.step()
        with torch.no_grad():
            logits=head(inputs);after=float(F.binary_cross_entropy_with_logits(logits,labels));accuracy=float(((logits>=0)==labels.bool()).float().mean())
        steps=sorted({int(s['step'].item()) for s in optimizer.state.values()})
        if steps!=[200] or not np.isfinite(after) or any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('invalid fitted foreground classifier')
        report=dict(seed=seed,steps=200,actual_Adam_steps=steps,parameters=sum(p.numel() for p in head.parameters()),
            initial_loss=before,final_loss=after,balanced_node_accuracy=accuracy,initial_parameter_sha256=initial,
            final_parameter_sha256=state_sha(head.state_dict()),node_samples=len(x),positive_nodes=int(y.sum()),
            normalization='TRAIN sample mean/std, floor1e-4, clip5',core_optimizer_steps=0)
        head.eval();head.requires_grad_(False)
        return head,optimizer.state_dict(),report


@torch.no_grad()
def predict_foreground(head,features):
    parameter=next(head.parameters())
    if features.ndim!=2 or features.shape[1]!=12 or features.dtype!=parameter.dtype or features.device!=parameter.device or not torch.isfinite(features).all():raise ValueError('finite node features matching classifier device/dtype required')
    return torch.sigmoid(head(features))
