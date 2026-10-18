"""Small supervised classifier of four signed emitted-spike correlations."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RelationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(4,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,x):return self.layers(x).squeeze(-1)


def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sample_pairs(labels,*,seed,count=8192):
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.size<2 or labels.dtype.kind not in 'iu' or (labels<0).any():raise ValueError('nonnegative integer GT grid required')
    if type(count) is not int or count<2 or count%2:raise ValueError('even positive pair budget required')
    groups=[np.flatnonzero(labels.reshape(-1)==c) for c in np.unique(labels)]
    large=[g for g in groups if len(g)>=2]
    if len(groups)<2 or not large:raise ValueError('TRAIN image must have positive and negative relations')
    rng=np.random.RandomState(seed);half=count//2
    pos_class=rng.randint(0,len(large),half);i=np.empty(count,dtype=np.int64);j=i.copy()
    for k,g in enumerate(large):
        where=np.flatnonzero(pos_class==k);a=rng.randint(0,len(g),len(where));b=rng.randint(0,len(g)-1,len(where));b+=b>=a
        i[where]=g[a];j[where]=g[b]
    a=rng.randint(0,len(groups),half);b=rng.randint(0,len(groups)-1,half);b+=b>=a
    for k,g in enumerate(groups):
        where=np.flatnonzero(a==k);i[half+where]=g[rng.randint(0,len(g),len(where))]
        where=np.flatnonzero(b==k);j[half+where]=g[rng.randint(0,len(g),len(where))]
    target=np.r_[np.ones(half,dtype=np.float32),np.zeros(half,dtype=np.float32)]
    if np.any(i==j) or not np.array_equal(labels.reshape(-1)[i]==labels.reshape(-1)[j],target.astype(bool)):
        raise ValueError('sampled relation labels invalid')
    return i,j,target


def fit(features,target,*,seed=206002):
    x=np.asarray(features);y=np.asarray(target)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=4 or y.dtype!=np.float32 or y.shape!=(len(x),) or not len(x):raise ValueError('float32 features[N,4] and target[N] required')
    if not np.isfinite(x).all() or (np.abs(x)>1).any() or not np.isin(y,[0.,1.]).all() or np.unique(y).size!=2:
        raise ValueError('finite correlations and both binary classes required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);head=RelationClassifier();initial=state_sha(head.state_dict())
        optimizer=torch.optim.Adam(head.parameters(),lr=1e-3)
        inputs=torch.from_numpy(np.ascontiguousarray(x));labels=torch.from_numpy(np.ascontiguousarray(y))
        before=float(F.binary_cross_entropy_with_logits(head(inputs),labels).detach())
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(head(inputs),labels)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite classifier objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('invalid classifier gradient')
            optimizer.step()
        with torch.no_grad():
            logits=head(inputs);after=float(F.binary_cross_entropy_with_logits(logits,labels));accuracy=float(((logits>=0)==labels.bool()).float().mean())
        if not np.isfinite(after) or any(not torch.isfinite(p).all() for p in head.parameters()):raise FloatingPointError('invalid fitted classifier')
        steps=sorted({int(s['step'].item()) for s in optimizer.state.values()})
        if steps!=[200]:raise ValueError('classifier Adam step count differs')
        report=dict(seed=seed,steps=200,actual_Adam_steps=steps,parameters=sum(p.numel() for p in head.parameters()),
            initial_loss=before,final_loss=after,balanced_pair_accuracy=accuracy,initial_parameter_sha256=initial,
            final_parameter_sha256=state_sha(head.state_dict()),pair_samples=len(x),positive_pairs=int(y.sum()),
            sym_component_product_pair_accuracy=float(np.mean((np.maximum(x,0).prod(1)>=.5)==y.astype(bool))),core_optimizer_steps=0)
        head.eval();head.requires_grad_(False)
        return head,optimizer.state_dict(),report


@torch.no_grad()
def predict_affinity(head,rho,*,chunk=65536):
    if rho.ndim!=3 or rho.shape[0]!=4 or rho.shape[1]!=rho.shape[2] or not rho.shape[1]:raise ValueError('four signed square component matrices required')
    if type(chunk) is not int or chunk<=0:raise ValueError('positive chunk required')
    parameter=next(head.parameters())
    if rho.device!=parameter.device or rho.dtype!=parameter.dtype:raise ValueError('classifier and correlations must share device/dtype')
    if not torch.isfinite(rho).all() or (rho.abs()>1).any():raise ValueError('finite correlations in [-1,1] required')
    n=rho.shape[1];out=torch.empty((n,n),device=rho.device,dtype=rho.dtype)
    for start in range(0,n*n,chunk):
        indices=torch.arange(start,min(start+chunk,n*n),device=rho.device);i=indices//n;j=indices%n
        x=(.5*(rho[:,i,j]+rho[:,j,i])).T
        out.view(-1)[start:start+len(indices)]=torch.sigmoid(head(x))
    return .5*(out+out.T)
