"""SW0259R1: validation-diagnosed routing release, unchanged head architecture.

Lives outside the immutable first-round deployment. Reuses mask-free RGB-trained
weights and cached activities only. No evaluation truth or supervision is loaded.
"""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)


def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,p)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
 root=Path(a.root);out=root/'refinement_prior_release'/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 code=Path(json.loads((root/'coordinator_launch.json').read_bytes())['code'])
 sys.path.insert(0,str(code/'unsupervised_readout_benchmark'))
 from model import ReconstructionReadout
 source=root/f'seed{a.seed}'
 while not (source/'arms/rgb/complete.json').exists():time.sleep(15)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.08,0)
 checkpoint=source/'arms/rgb/best.pt'
 fingerprint=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 saved=torch.load(checkpoint,map_location='cuda:0',weights_only=True)
 train=[torch.load(q,weights_only=True,map_location='cpu') for q in sorted((source/'cache/train').glob('*.pt'))]
 val=[torch.load(q,weights_only=True,map_location='cpu') for q in sorted((source/'cache/validation').glob('*.pt'))]
 assert len(train)==256 and len(val)==32
 ids=[r['image_id'] for r in train];assert len(set(ids))==256 and not set(ids)&set(range(1000,1640))
 for arm,activity_weight in [('release_rgb',0.),('release_activity',.03)]:
  d=out/arm;d.mkdir(parents=True,exist_ok=True)
  if (d/'complete.json').exists():continue
  model=ReconstructionReadout(a.seed).cuda();model.load_state_dict(saved['model'])
  with torch.no_grad():model.head.prior_scale.fill_(.5);model.head.residual_scale.clamp_(min=2.)
  opt=torch.optim.Adam(model.parameters(),lr=.003);best=float('inf');start=0
  if (d/'resume.pt').exists():
   resume=torch.load(d/'resume.pt',weights_only=True,map_location='cuda:0');model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);start=int(resume['epoch']);best=float(resume['best'])
  initial=ReconstructionReadout(a.seed).cuda().eval()
  for epoch in range(start,a.epochs):
   order=np.random.RandomState(259000+a.seed+10000*epoch).permutation(256);losses=[]
   model.train()
   for index in order:
    row=train[index];opt.zero_grad(set_to_none=True)
    rgb,detail=model(row['features'].cuda(),row['regions'].numpy())
    loss=F.mse_loss(rgb,row['rgb'].cuda())+activity_weight*detail['coherence']+.003*model.head.prior_scale.square()
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm)
    opt.step()
    with torch.no_grad():model.head.prior_scale.clamp_(0.,.5)
    losses.append(float(loss.detach()))
   model.eval();mse=[];changed=[];groups=[]
   with torch.no_grad():
    for row in val:
     f=row['features'].cuda();r=row['regions'].numpy();rgb,detail=model(f,r)
     mse.append(float(F.mse_loss(rgb,row['rgb'].cuda())))
     pred=model.predict(f,r);changed.append(float(np.mean(pred!=initial.predict(f,r))));groups.append(len(np.unique(pred)))
   validation=dict(rgb_mse=float(np.mean(mse)),mean_hard_groups=float(np.mean(groups)),changed_from_initial_head=float(np.mean(changed)))
   if validation['rgb_mse']<best:
    best=validation['rgb_mse'];save(d/'best.pt',dict(model=model.state_dict(),epoch=epoch+1,validation=validation,arm=arm))
   save(d/'resume.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best))
   row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=validation,prior_scale=float(model.head.prior_scale.detach()),residual_scale=float(model.head.residual_scale.detach()))
   with (d/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
   write(out/'progress.json',dict(stage='refine_head_decoder',arm=arm,**row,updated=time.time()))
  write(d/'complete.json',dict(arm=arm,best_validation_mse=best,epochs=a.epochs,updates=256*a.epochs,unique_TRAIN=256,mask_supervision=False,
   warm_start=str(checkpoint),warm_start_sha256=fingerprint,warm_start_selected_epoch=int(saved['epoch']),warm_start_training_masks_used=False,
   architecture='unchanged252 NeutralRegionHead and unchanged first-round decoder',prior_constraint=[0.,.5],prior_L2=.003,activity_weight=activity_weight))
 write(out/'complete.json',dict(seed=a.seed,status='two_prior_release_arms_trained',mask_supervision=False))

if __name__=='__main__':main()
