"""Resumable SW0260 cache preparation, actual GPU admission and mask-free fit."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from model import FineReadout,ARMS,expand32
OLD=Path('/work/USERS/kevinswk/experiments/SW0259_unsupervised_readout')
LEGACY=Path('/Data0/kevinswk/patch_v2_sw')
DATA=LEGACY/'datasets/object_centric_data/clevr_10-full.hdf5'
DATA=Path('/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def sources():return {p.name:sha(p) for p in HERE.glob('*.py')}

def original(seed):
 code=Path(json.loads((OLD/'coordinator_launch.json').read_bytes())['code'])
 spec=importlib.util.spec_from_file_location('sw0260_original_arch',code/'unsupervised_readout_benchmark/model.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 path=OLD/'publication_20261102/bundle/unsupervised_readout_benchmark/checkpoints'/f'sw0259_seed{seed}.pt'
 blob=torch.load(path,weights_only=True,map_location='cpu')
 model=module.ReconstructionReadout(seed);model.load_state_dict(blob['readout_state'],strict=True)
 return model,module,blob,path

def image(split,index,row,train_rgb,fixed_rgb):
 if split=='train':
  i=row['image_id'];return np.asarray(train_rgb[i if i<1000 else i-640])
 if split=='fixed320':return np.asarray(fixed_rgb[index])
 import h5py
 with h5py.File(DATA,'r') as f:return np.asarray(f['image'][row['image_id']])

def prepare(root,seed):
 sys.path.insert(0,str(LEGACY))
 from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
 from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
 model,module,blob,checkpoint=original(seed)
 encoder=CNNFeatureEncoder(8,3,3,bias=True).cuda().eval().requires_grad_(False);encoder.load_state_dict(blob['encoder_state'],strict=True)
 patcher=FeaturePatchGammaInitializer(grid_size=64).cuda()
 mean=blob['feature_mean'].cuda();std=blob['feature_std'].cuda()
 folder=LEGACY/'data/SW_0106_spike_partition_rgb'
 train_rgb=np.load(folder/'train_rgb_uint8.npy',mmap_mode='r');fixed_rgb=np.load(folder/'validation_rgb_uint8.npy',mmap_mode='r')
 out=root/'inputs'/f'seed{seed}';identity=dict(seed=seed,core_grid=32,head_grid=64,code=sources(),checkpoint_sha256=sha(checkpoint),source=blob['source_provenance'])
 if (out/'binding.json').exists():assert json.loads((out/'binding.json').read_bytes())==identity
 else:write(out/'binding.json',identity)
 for split,count in [('train',256),('validation',32),('fixed320',320)]:
  source=OLD/f'seed{seed}/cache'/split
  assert json.loads((source/'complete.json').read_bytes())['source_unchanged']
  for i in range(count):
   target=out/split/f'{i:04d}.pt'
   if target.exists():continue
   row=torch.load(source/f'{i:04d}.pt',map_location='cpu',weights_only=True)
   rgb=image(split,i,row,train_rgb,fixed_rgb)
   assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rgb_sha256']
   with torch.no_grad():
    x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda().float()/255.
    gamma=patcher(((encoder(x)-mean)/std).clamp(-blob['feature_clip'],blob['feature_clip']))[0].T.cpu()
    masks,prior=module.module.candidates(row['features'],row['regions'].numpy())
    target_rgb=F.avg_pool2d(x,2)[0].permute(1,2,0).reshape(4096,3).cpu()
   assert gamma.shape==(4096,8)
   save(target,dict(features=row['features'],masks=masks.cpu(),prior=prior.cpu(),gamma=gamma,rgb=target_rgb,
    image_id=row['image_id'],rgb_sha256=row['rgb_sha256'],baseline32=row['regions']))
   if i%16==0:write(out/'progress.json',dict(split=split,completed=i+1,total=count,updated=time.time()))
 write(out/'complete.json',dict(core_recurrence_executions=0,core_unchanged=True,encoder_sha256=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in encoder.state_dict().values())).hexdigest(),head_grid=64))

def load_rows(root,seed,split,count):
 return [torch.load(root/'inputs'/f'seed{seed}'/split/f'{i:04d}.pt',map_location='cpu',weights_only=True) for i in range(count)]
def device_row(row):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in row.items()}

def validation_targets():
 # Labels belong only to held-out validation, never TRAIN or fixed320 tuning.
 import h5py
 with h5py.File(DATA,'r') as f:pixel=torch.from_numpy(np.asarray(f['mask'][1000:1032]))
 sys.path.insert(0,str(LEGACY))
 from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
 return clevr_mask_patch(pixel,2)['patch_labels']

@torch.no_grad()
def validate(model,val,truth,out):
 predictions=[];mse=[];change=[]
 for row in val:
  r=device_row(row);_,parts,logits=model(r);pred=(logits.argmax(0)+1).reshape(64,64)
  baseline=expand32(row['prior'].T).T.argmax(0).reshape(64,64)+1
  predictions.append(pred.cpu());mse.append(float(parts['mse']));change.append(float((pred.cpu()!=baseline).float().mean()))
 predictions=torch.stack(predictions)
 # Freeze validation predictions before reading the evaluation labels in metric code.
 save(out/'validation_predictions.pt',dict(labels=predictions,image_ids=list(range(1000,1032))))
 from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks
 metrics=evaluate_patch_masks(predictions,truth)
 return dict(fg_ari=float(metrics['mean']['fg_ari']),rgb_mse=float(np.mean(mse)),changed_fraction=float(np.mean(change)))

def admission(root,seed):
 original_model,_,_,_=original(seed)
 row=device_row(load_rows(root,seed,'train',1)[0]);reports=[]
 for arm in ARMS:
  model=FineReadout(original_model,arm).cuda();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
  loss,_,_=model(row);loss.backward()
  assert torch.isfinite(loss)
  gradients=[p.grad for p in model.parameters() if p.requires_grad]
  assert all(g is not None and torch.isfinite(g).all() for g in gradients)
  assert model.fine.weight.grad.abs().sum()>0,'fine64 input has no gradient'
  assert sum(float(p.grad.abs().sum()) for p in model.head.embedding.parameters())>0,'routing head has no gradient'
  before=model.fine.weight.detach().clone();opt.step();assert not torch.equal(before,model.fine.weight)
  # Prove four children can differ: fine adapter participates in true64 logits.
  logits=model.routing(row)[0].reshape(-1,32,2,32,2)
  variation=float((logits[:,:,0,:,0]-logits[:,:,1,:,1]).abs().max().detach())
  assert variation>0,'output is only a repeated32 mask'
  save(root/f'admission_restore_seed{seed}.pt',dict(model=state(model),optimizer=opt.state_dict()))
  restored=torch.load(root/f'admission_restore_seed{seed}.pt',weights_only=True,map_location='cuda:0')
  clone=FineReadout(original_model,arm).cuda();clone.load_state_dict(restored['model']);opt2=torch.optim.Adam([p for p in clone.parameters() if p.requires_grad],lr=.001);opt2.load_state_dict(restored['optimizer'])
  assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
  reports.append(dict(arm=arm,finite_gradients=True,head_and_fine_adapter_update=True,optimizer_restore=True,subpatch_logit_variation=variation))
 write(root/f'admission_seed{seed}.json',dict(rows=reports,core_grid=32,output_grid=64,code=sources()))

def train(root,seed,arm,count,epochs):
 assert json.loads((root/f'admission_seed{seed}.json').read_bytes())['code']==sources()
 old,_,_,_=original(seed);model=FineReadout(old,arm).cuda()
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
 out=root/'screen'/f'seed{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 tr=load_rows(root,seed,'train',count);val=load_rows(root,seed,'validation',32);truth=validation_targets()
 assert len({r['image_id'] for r in tr})==count and not any(1000<=r['image_id']<1640 for r in tr)
 best=-float('inf');start=0
 if (out/'resume.pt').exists():
  saved=torch.load(out/'resume.pt',map_location='cuda:0',weights_only=True);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['epoch'];best=saved['best']
 for epoch in range(start,epochs):
  model.train();values=[]
  order=np.random.RandomState(260000+seed+10000*epoch).permutation(count)
  for index in order:
   opt.zero_grad(set_to_none=True);loss,_,_=model(device_row(tr[index]));loss.backward()
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();values.append(float(loss.detach()))
  model.eval();v=validate(model,val,truth,out)
  if v['fg_ari']>best:
   best=v['fg_ari'];save(out/'best.pt',dict(model=state(model),epoch=epoch+1,validation=v,arm=arm,seed=seed))
  save(out/'resume.pt',dict(model=state(model),optimizer=opt.state_dict(),epoch=epoch+1,best=best))
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,updates=(epoch+1)*count,loss=float(np.mean(values)),validation=v))+'\n')
  write(out/'progress.json',dict(epoch=epoch+1,epochs=epochs,validation=v,updated=time.time()))
 write(out/'complete.json',dict(seed=seed,arm=arm,best_validation_FGARI=best,unique_TRAIN=count,epochs=epochs,updates=count*epochs,training_masks_used=False,labelled_validation_used=True,core_grid=32,head_grid=64))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','admission','train']);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True,choices=range(3));p.add_argument('--arm',choices=list(ARMS));p.add_argument('--train',type=int,default=128);p.add_argument('--epochs',type=int,default=8);a=p.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.15,0)
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
 if a.command=='prepare':prepare(root,a.seed)
 elif a.command=='admission':admission(root,a.seed)
 else:train(root,a.seed,a.arm,a.train,a.epochs)
