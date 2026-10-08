"""Export unfiltered TFDS CLEVR train0:70000 and encode it with frozen SW0097 assets."""
import argparse,hashlib,json,os,sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
FROZEN=ROOT/'data/SW_0107_official_full70k_transfer/frozen_sources.json'
ASSETS=Path('/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002')

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def export(tfds_dir,out,manifest):
 os.environ['CUDA_VISIBLE_DEVICES']='-1'
 import h5py,tensorflow as tf,tensorflow_datasets as tfds
 out=Path(out);manifest=Path(manifest)
 if out.exists() or manifest.exists():raise FileExistsError('preserve existing SW0107 export')
 freeze_sources()
 out.parent.mkdir(parents=True,exist_ok=True)
 ds=tfds.load('clevr:3.1.0',split='train',shuffle_files=False,data_dir=str(tfds_dir)).take(70000)
 def prep(i,row):
  x=tf.cast(row['image'][29:221,64:256,:],tf.float32)
  x=tf.image.resize(x,(128,128),method=tf.image.ResizeMethod.BILINEAR)
  return i,tf.clip_by_value(x/255.,0.,1.)
 ds=ds.enumerate().map(prep,num_parallel_calls=tf.data.AUTOTUNE).batch(128)
 with h5py.File(out,'x') as h:
  images=h.create_dataset('image',shape=(70000,128,128,3),dtype='float16',chunks=(16,128,128,3))
  indices=h.create_dataset('tfds_train_index',shape=(70000,),dtype='int64')
  n=0
  for ix,x in tfds.as_numpy(ds):
   m=len(ix);images[n:n+m]=x.astype(np.float16);indices[n:n+m]=ix;n+=m
   if n%8192<m:print(json.dumps({'exported':n}),flush=True)
 if n!=70000:raise AssertionError(f'expected exact first70000 TFDS rows, got {n}')
 ids=np.arange(70000,dtype='<i8')
 with h5py.File(out,'r') as h:
  if not np.array_equal(h['tfds_train_index'][:],ids):raise AssertionError('TFDS source IDs are not exactly0..69999')
  block_hashes=[]
  for start in range(0,70000,1024):
   block_hashes.append(hashlib.sha256(np.ascontiguousarray(h['image'][start:start+1024]).tobytes()).hexdigest())
 record={'status':'complete','source':'TFDS clevr:3.1.0 train','tfds_data_dir':str(tfds_dir),
  'source_indices':[0,69999],'images':70000,'filter':'none','skip':0,'shuffle_files':False,
  'preprocessing':'RGB image[29:221,64:256], bilinear resize128, float32/255 clipped[0,1], stored float16',
  'output':str(out),'output_sha256':digest(out),'index_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
  'rgb_block_sha256':block_hashes,'frozen_sources_path':str(FROZEN),'frozen_sources_sha256':digest(FROZEN),
  'ground_truth_masks_read':False}
 manifest.write_text(json.dumps(record,indent=2)+'\n')
 return record

def freeze_sources():
 if FROZEN.exists():raise FileExistsError('preserve prior SW0107 frozen-source manifest')
 records=[]
 for seed in range(3):
  p=CONTROL/f'seed{seed}_positive_frozen/core.pt'
  if not p.is_file():raise FileNotFoundError(f'SW0097 seed{seed} checkpoint missing: {p}')
  records.append({'seed':seed,'path':str(p),'sha256':digest(p)})
 encoder=ASSETS/'input_encoder/input_layer_encoder.pt';stats=ASSETS/'feature_preprocessing.pt'
 body={'status':'frozen_before_official_export','source_recipe':'SW0097 positive_frozen seeds0/1/2',
  'cores':records,'encoder':{'path':str(encoder),'sha256':digest(encoder)},
  'feature_stats':{'path':str(stats),'sha256':digest(stats)},'graph':'checkpoint graph frozen for training'}
 FROZEN.parent.mkdir(parents=True,exist_ok=True);FROZEN.write_text(json.dumps(body,indent=2)+'\n')
 return body

def encode_cache(source,manifest,gamma,gamma_manifest,device='cpu'):
 import h5py,torch
 sys.path.insert(0,str(ROOT))
 from snn_kuramoto_bidirectional.gamma_initializer import feature_maps_to_patch_gamma
 from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
 source=Path(source);manifest=Path(manifest);gamma=Path(gamma);gamma_manifest=Path(gamma_manifest)
 if not manifest.is_file() or json.loads(manifest.read_text()).get('status')!='complete':raise RuntimeError('verified official RGB export required')
 if gamma.exists() or gamma_manifest.exists():raise FileExistsError('preserve existing SW0107 gamma cache')
 stats=ASSETS/'feature_preprocessing.pt';encoder_path=ASSETS/'input_encoder/input_layer_encoder.pt'
 frozen=json.loads(FROZEN.read_text())
 if frozen.get('status')!='frozen_before_official_export' or frozen.get('encoder',{}).get('sha256')!=digest(encoder_path) or frozen.get('feature_stats',{}).get('sha256')!=digest(stats):
  raise AssertionError('registered encoder/statistics differ from pre-export freeze')
 if any(digest(CONTROL/f"seed{x['seed']}_positive_frozen/core.pt")!=x['sha256'] for x in frozen.get('cores',[])):
  raise AssertionError('SW0097 source checkpoint changed after pre-export freeze')
 st=torch.load(stats,map_location='cpu',weights_only=True);mean,std=st['mean'].float(),st['std'].float();clip=float(st.get('clip',3.))
 if st.get('mode')!='standardize':raise AssertionError('registered preprocessing mode mismatch')
 enc=load_input_encoder(str(encoder_path),num_kernels=8,kernel_size=3,channels=3,device=device).eval()
 rows=[]
 with h5py.File(source,'r') as h,torch.no_grad():
  for start in range(0,70000,128):
   x=torch.from_numpy(np.asarray(h['image'][start:start+128],dtype=np.float32).copy()).permute(0,3,1,2).to(device)
   f=enc(x);g=feature_maps_to_patch_gamma(((f-mean.to(device))/std.to(device)).clamp(-clip,clip),grid_size=16,device=device)
   rows.append(g.float().cpu())
   if (start//128+1)%100==0:print(json.dumps({'encoded':min(start+128,70000)}),flush=True)
 values=torch.cat(rows)
 if tuple(values.shape)!=(70000,8,256) or not bool(torch.isfinite(values).all()):raise AssertionError('invalid frozen official gamma')
 gamma.parent.mkdir(parents=True,exist_ok=True);torch.save(values,gamma)
 record={'status':'complete','shape':list(values.shape),'rgb_export_sha256':digest(source),'encoder_path':str(encoder_path),
  'encoder_sha256':digest(encoder_path),'feature_stats_path':str(stats),'feature_stats_sha256':digest(stats),
  'gamma_sha256':digest(gamma),'gamma_mode':'registered native frozen encoder; RGB float[0,1]; standardize+clip; 8x16x16',
  'ground_truth_used':False}
 gamma_manifest.write_text(json.dumps(record,indent=2)+'\n');return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['export','encode'],required=True)
 p.add_argument('--tfds-dir',default='/Data0/kevinswk/datasets/tfds');p.add_argument('--source',type=Path,required=True)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--gamma',type=Path);p.add_argument('--gamma-manifest',type=Path);p.add_argument('--device',default='cpu')
 a=p.parse_args()
 if a.stage=='export':r=export(a.tfds_dir,a.source,a.manifest)
 else:
  if a.gamma is None or a.gamma_manifest is None:raise ValueError('--gamma and --gamma-manifest required for encode')
  r=encode_cache(a.source,a.manifest,a.gamma,a.gamma_manifest,a.device)
 print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
