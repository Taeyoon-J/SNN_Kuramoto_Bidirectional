"""Extend the two validation-selected screen arms using exact Adam continuation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--arm',required=True);a=p.parse_args()
 root=Path(a.root);sys.path.insert(0,a.code)
 import run as opt
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.15,0)
 source=root/'screen'/f'seed{a.seed}'/a.arm
 assert (source/'complete.json').exists()
 old,_,_,_=opt.original(a.seed);model=opt.FineReadout(old,a.arm).cuda()
 optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
 prefix=torch.load(source/'resume.pt',map_location='cuda:0',weights_only=True)
 model.load_state_dict(prefix['model']);optimizer.load_state_dict(prefix['optimizer'])
 assert {int(x['step']) for x in optimizer.state.values()}=={1024}
 for group in optimizer.param_groups:group['lr']=.0003
 out=root/'extended'/f'seed{a.seed}'/a.arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 best=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
 if not (out/'best.pt').exists():opt.save(out/'best.pt',dict(**best,selection_stage='screen'))
 best_score=float(best['validation']['fg_ari']);start=0
 if (out/'resume.pt').exists():
  saved=torch.load(out/'resume.pt',weights_only=True,map_location='cuda:0');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=int(saved['epoch']);best_score=float(saved['best_score'])
 train=opt.load_rows(root,a.seed,'train',256);val=opt.load_rows(root,a.seed,'validation',32);truth=opt.validation_targets()
 assert len({r['image_id'] for r in train})==256 and not any(1000<=r['image_id']<1640 for r in train)
 for epoch in range(start,20):
  order=np.random.RandomState(260100+a.seed+10000*epoch).permutation(256);values=[];model.train()
  for index in order:
   optimizer.zero_grad(set_to_none=True);loss,_,_=model(opt.device_row(train[index]));loss.backward()
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   assert torch.isfinite(loss) and torch.isfinite(norm);optimizer.step();values.append(float(loss.detach()))
  model.eval();v=opt.validate(model,val,truth,out)
  if v['fg_ari']>best_score:
   best_score=v['fg_ari'];opt.save(out/'best.pt',dict(model=opt.state(model),epoch=epoch+1,validation=v,arm=a.arm,seed=a.seed,selection_stage='extended',total_updates=1024+256*(epoch+1)))
  opt.save(out/'resume.pt',dict(model=opt.state(model),optimizer=optimizer.state_dict(),epoch=epoch+1,best_score=best_score,total_updates=1024+256*(epoch+1),torch_rng=torch.get_rng_state()))
  value=dict(epoch=epoch+1,total_updates=1024+256*(epoch+1),loss=float(np.mean(values)),validation=v,updated=time.time())
  with (out/'training.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
  opt.write(out/'progress.json',value)
 assert {int(x['step']) for x in optimizer.state.values()}=={6144}
 opt.write(out/'complete.json',dict(seed=a.seed,arm=a.arm,best_validation_FGARI=best_score,source_screen_sha256=opt.sha(source/'resume.pt'),additional_unique_TRAIN=256,additional_epochs=20,additional_updates=5120,total_updates=6144,TRAIN_mask_supervision=False,labelled_validation_only=True,core_grid=32,head_grid=64))

if __name__=='__main__':main()
