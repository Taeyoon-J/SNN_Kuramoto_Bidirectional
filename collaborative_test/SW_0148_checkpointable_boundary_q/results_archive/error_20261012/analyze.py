import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json,hashlib,time
from pathlib import Path
import numpy as np
import h5py
import sys
root=Path('/Data0/kevinswk/patch_v2_sw');sys.path.insert(0,str(root))
from collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32 import modal_native32
b=root/'collaborative_test/SW_0148_checkpointable_boundary_q'
e=json.loads((b/'results_archive/evaluation_seed1.json').read_text())
paths={a:b/f'predictions/{a}_seed1/predictions.npz' for a in ('encoder_graph','joint')}
paths['source97_seed1']=root/'collaborative_test/SW_0137_native32_source_qcc/results_archive/source_predictions/seed1/predictions.npz'
expected=dict(e['provenance']['candidate_predictions']);expected['source97_seed1']=e['provenance']['source97_prediction']
y={}
for a,p in paths.items():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[a]
 with np.load(p,allow_pickle=False) as z:
  assert np.array_equal(z['image_ids'],np.arange(1320,1640));y[a]=z['qcc_labels'].copy()
with h5py.File('/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5','r') as f:g=modal_native32(np.asarray(f['mask'][1320:1640]))
def pairs(v):return np.sum(v.astype(np.float64)*(v-1)/2)
def diag(p,t):
 fg=t>0; a=t[fg];c=p[fg];_,ai=np.unique(a,return_inverse=True);_,ci=np.unique(c,return_inverse=True)
 table=np.zeros((ai.max()+1,ci.max()+1),dtype=np.int64);np.add.at(table,(ai,ci),1)
 both=pairs(table);sameg=pairs(table.sum(1));samep=pairs(table.sum(0));total=len(a)*(len(a)-1)/2
 return {'fg_same_object_split_rate':float(1-both/sameg) if sameg else None,'fg_different_object_merge_rate':float((samep-both)/(total-sameg)) if total>sameg else None,'fg_as_background_rate':float(np.mean(p[fg]==0)),'background_as_fg_rate':float(np.mean(p[~fg]>0)),'predicted_fg_label_count':int(len(np.unique(p[p>0])))}
r={'scope':'Posthoc fixed320 explanatory error analysis only. No threshold selection or training targets. Pair rates condition on GT foreground; lower is better. Source and candidate archive SHA verified before GT.','observed_unix':time.time(),'arms':{}}
for a,p in y.items():
 ds=[diag(pi,ti) for pi,ti in zip(p,g)]
 r['arms'][a]={'means':{k:float(np.mean([d[k] for d in ds if d[k] is not None])) for k in ds[0]},'per_image':ds,'prediction_sha256':expected[a]}
print(json.dumps(r))
