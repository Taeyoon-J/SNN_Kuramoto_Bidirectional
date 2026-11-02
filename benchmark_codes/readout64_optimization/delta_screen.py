"""Fresh controlled R2 screen using old warm weights and immutable input caches."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--arm',required=True);a=p.parse_args()
 root=Path(a.root);sys.path.insert(0,a.code)
 import run as opt
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from delta_model import DeltaReadout,DELTA_ARMS
 assert a.arm in DELTA_ARMS
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.15,0)
 old,_,_,_=opt.original(a.seed);model=DeltaReadout(old,a.arm).cuda();optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
 data_root=Path('/work/USERS/kevinswk/experiments/SW0260_readout64_r1')
 train=opt.load_rows(data_root,a.seed,'train',128);val=opt.load_rows(data_root,a.seed,'validation',32);truth=opt.validation_targets()
 out=root/'screen'/f'seed{a.seed}'/a.arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 with torch.no_grad():
  for row in [train[0],*val]:
   r=opt.device_row(row);pred=model.predict(r);expected=(opt.expand32(row['prior'].T).T.argmax(0)+1).reshape(64,64).cuda()
   assert torch.equal(pred,expected),'R2 initial routing must exactly preserve259'
 initial=opt.validate(model,val,truth,out)
 best=-float('inf');start=0
 if (out/'resume.pt').exists():
  saved=torch.load(out/'resume.pt',weights_only=True,map_location='cuda:0');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];best=saved['best']
 for epoch in range(start,10):
  model.train();losses=[]
  for index in np.random.RandomState(261000+a.seed+10000*epoch).permutation(128):
   optimizer.zero_grad(set_to_none=True);loss,_,_=model(opt.device_row(train[index]));loss.backward()
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(loss) and torch.isfinite(norm)
   assert torch.isfinite(model.fine.weight.grad).all();optimizer.step();losses.append(float(loss.detach()))
  model.eval();validation=opt.validate(model,val,truth,out)
  if validation['fg_ari']>best:
   best=validation['fg_ari'];opt.save(out/'best.pt',dict(model=opt.state(model),epoch=epoch+1,arm=a.arm,seed=a.seed,validation=validation,initial_validation=initial,routing_variant='baseline_preserving_delta'))
  opt.save(out/'resume.pt',dict(model=opt.state(model),optimizer=optimizer.state_dict(),epoch=epoch+1,best=best))
  row=dict(epoch=epoch+1,updates=128*(epoch+1),loss=float(np.mean(losses)),validation=validation,updated=time.time())
  with (out/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
  opt.write(out/'progress.json',row)
 assert {int(v['step']) for v in optimizer.state.values()}=={1280}
 opt.write(out/'complete.json',dict(seed=a.seed,arm=a.arm,best_validation_FGARI=best,initial_validation=initial,initial_all32_validation_masks_equal_old259=True,unique_TRAIN=128,epochs=10,updates=1280,TRAIN_masks_used=False,labelled_validation_selection=True,core_grid=32,head_grid=64))

if __name__=='__main__':main()
