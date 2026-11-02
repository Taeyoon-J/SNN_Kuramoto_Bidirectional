"""Offline64-output inference using published SW0259 parent and trained context head."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def load(checkpoint,device='cpu'):
 manifest=json.loads((HERE/'checkpoints/manifest.json').read_bytes())
 record=next(r for r in manifest['checkpoints'] if Path(r['file']).name==Path(checkpoint).name)
 assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==record['sha256']
 parent=HERE.parent/'unsupervised_readout_benchmark'
 parent_checkpoint=parent/'checkpoints'/f"sw0259_seed{record['seed']}.pt"
 assert hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()==record['parent_sha256']
 backbone=module('sw0260_parent_portable',parent/'portable.py')
 found,old,parent_blob=backbone.load(parent_checkpoint,device)
 sys.path.insert(0,str(HERE))
 existing=sys.modules.get('model')
 if existing is not None and Path(existing.__file__).resolve()!=HERE/'model.py':raise RuntimeError('Use a fresh Python process for SW0260 inference')
 from context_model import ContextReadout
 old._seed_hint=record['seed'];head=ContextReadout(old,record['arm']).to(device)
 saved=torch.load(checkpoint,map_location=device,weights_only=True);head.load_state_dict(saved['model'],strict=True);head.eval().requires_grad_(False)
 return found,head,record

@torch.no_grad()
def predict(found,head,rgb):
 from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
 from collaborative_test.SW_0162_connected_signed_partition import preflight as source
 from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver
 from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
 rgb=np.asarray(rgb)
 if rgb.dtype!=np.uint8 or rgb.shape!=(128,128,3):raise ValueError('exact uint8 RGB128x128 required')
 device=next(found.encoder.parameters()).device
 image=torch.from_numpy(rgb[None].copy()).to(device)
 trace,a=source.capture_source_coupling(found.wrapped.core.kuramoto,lambda:late_rollout(found.wrapped,found.encode(image),total_steps=1024,live_tail_steps=0))
 q=source._actual_q(trace)[0].cpu().numpy().astype(np.float64);gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
 regions=solver.foreground_induced_merge(q,a[0].cpu().numpy().astype(np.float64),gain,np.ones((32,32),dtype=np.int64),grid_shape=(32,32))['labels']
 vendor=HERE.parent/'vendor'
 activity=module('sw0260_activity',vendor/'readout_benchmark/head.py')
 neutral=module('sw0260_candidates',vendor/'neutral_readout/head.py')
 features=activity.activity_features(trace)[0].cpu();masks,prior=neutral.candidates(features,regions)
 maps=found.encoder(image.permute(0,3,1,2).float()/255.)
 gamma=FeaturePatchGammaInitializer(grid_size=64).to(device)(((maps-found.feature_mean)/found.feature_std).clamp(-found.feature_clip,found.feature_clip))[0].T
 row=dict(features=features.to(device),masks=masks.to(device),prior=prior.to(device),gamma=gamma)
 return head.predict(row).cpu().numpy()

if __name__=='__main__':
 import argparse
 from PIL import Image
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 found,head,_=load(a.checkpoint,a.device);labels=predict(found,head,np.asarray(Image.open(a.image).convert('RGB')))
 np.save(a.output,labels);print(json.dumps(dict(output=a.output,shape=list(labels.shape))))
