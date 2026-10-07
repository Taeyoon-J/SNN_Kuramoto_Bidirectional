"""Differentiable RGB reconstruction through the exact hard spike partition."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels

GRID=16
N_PATCHES=GRID*GRID
FEATURE_DIM=8
TEMPERATURE=.10

def groups_to_onehot(groups,device=None,dtype=torch.float32,n_patches=N_PATCHES):
 if n_patches!=N_PATCHES: raise ValueError('production classifier is registered for a 16x16 grid')
 labels=spatial_components_to_patch_labels([groups],GRID,device=device).reshape(-1)
 onehot=F.one_hot(labels,num_classes=int(labels.max().item())+1).to(dtype=dtype)
 if not torch.equal(onehot.sum(-1),torch.ones(n_patches,device=device,dtype=dtype)):
  raise AssertionError('hard partition is not one-hot')
 return onehot

def assignment_weights(q,hard,credit=True,temperature=TEMPERATURE):
 if q.ndim!=2 or q.shape[0]!=q.shape[1] or hard.ndim!=2 or hard.shape[0]!=q.shape[0]:
  raise ValueError('expected q[N,N] and hard[N,K]')
 if temperature!=TEMPERATURE: raise ValueError('temperature is frozen at .10')
 hard=hard.detach()
 if credit:
  eye=torch.eye(q.shape[0],device=q.device,dtype=q.dtype)
  q0=q*(1.-eye)
  count=hard.sum(dim=0)
  other_count=(count.unsqueeze(0)-hard).clamp_min(1.)
  affinity=(q0@hard)/other_count
  p=torch.softmax(affinity/TEMPERATURE,dim=-1)
  w=hard+(p-p.detach())
 else:
  p=None; w=hard
 return w,p

def slot_features(w,features):
 if features.ndim!=2 or features.shape[0]!=w.shape[0]: raise ValueError('expected F[N,C]')
 f=features.detach()
 denom=w.sum(dim=0).unsqueeze(-1).clamp_min(1e-8)
 return (w.transpose(0,1)@f)/denom

def patch_centers(device,dtype):
 one=(torch.arange(GRID,device=device,dtype=dtype)+.5)*(2./GRID)-1.
 yy,xx=torch.meshgrid(one,one,indexing='ij')
 return torch.stack((xx.reshape(-1),yy.reshape(-1)),dim=-1)

class SharedRGBDecoder(nn.Module):
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3),nn.Sigmoid())
 def forward(self,slot_content,xy):
  if slot_content.ndim!=2 or slot_content.shape[-1]!=FEATURE_DIM or xy.shape!=(N_PATCHES,2):
   raise ValueError('expected slot content [K,8] and patch centers [256,2]')
  k=slot_content.shape[0]
  content=slot_content[:,None,:].expand(k,N_PATCHES,FEATURE_DIM)
  coord=xy[None,:,:].expand(k,N_PATCHES,2)
  return self.net(torch.cat((content,coord),dim=-1))

def reconstruct_one(q,hard,features,target_rgb,decoder,assignment_credit=True):
 """Return per-image patch RGB prediction, MSE, and assignment diagnostics."""
 if target_rgb.shape!=(N_PATCHES,3): raise ValueError('target must be [256,3] patch RGB means')
 w,p=assignment_weights(q,hard,credit=assignment_credit)
 content=slot_features(w,features)
 decoded=decoder(content,patch_centers(content.device,content.dtype))
 prediction=torch.einsum('nk,knc->nc',w,decoded)
 loss=F.mse_loss(prediction,target_rgb)
 if not torch.equal(w.detach(),hard): raise AssertionError('straight-through forward partition differs from hard labels')
 return prediction,loss,{'W':w,'P':p,'K':int(hard.shape[1]),'group_sizes':hard.sum(0).detach()}

def rgb_patch_means(images):
 """RGB float images [B,3,128,128] in [0,1] -> [B,256,3] means."""
 if images.ndim!=4 or images.shape[1:]!=(3,128,128): raise ValueError('expected [B,3,128,128]')
 return F.avg_pool2d(images,kernel_size=8,stride=8).permute(0,2,3,1).reshape(images.shape[0],N_PATCHES,3)
