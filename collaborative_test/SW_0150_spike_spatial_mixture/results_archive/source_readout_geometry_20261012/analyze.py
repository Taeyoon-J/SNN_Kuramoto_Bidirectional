import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json
from pathlib import Path
root=Path('/Data0/kevinswk/patch_v2_sw')
sys.path[:0]=[str(root),str(root/'collaborative_test'),str(root/'snn_kuramoto_bidirectional')]
import numpy as np
from collaborative_test.SW_0150_spike_spatial_mixture import evaluate
row=evaluate.validate_prediction('source_p')
def describe(labels):
 counts=[];sizes=[];tiny=[]
 for frame in labels:
  ids,mass=np.unique(frame,return_counts=True)
  foreground_mass=mass[ids!=0]
  counts.append(len(foreground_mass))
  tiny.append(int((foreground_mass<8).sum()))
  sizes.extend(foreground_mass.tolist())
 return {'mean_foreground_patch_fraction':float((labels!=0).mean()),'mean_nonbackground_label_count':float(np.mean(counts)),'median_nonbackground_label_count':float(np.median(counts)),'mean_nonbackground_groups_smaller_than_8_patches':float(np.mean(tiny)),'median_nonbackground_group_size':float(np.median(sizes)) if sizes else None,'image_count':len(labels)}
p=row['primary_labels'];q=row['qcc_labels']
fgp=p!=0;fgq=q!=0
intersection=(fgp&fgq).sum(axis=(1,2));union=(fgp|fgq).sum(axis=(1,2))
print(json.dumps({'experiment':'SW0150_spike_spatial_mixture','evidence_kind':'gt_free_readout_geometry_comparison','prediction_sha256':row['prediction_sha256'],'protocol_sha256':row['protocol_sha256'],'image_ids':[int(row['ids'][0]),int(row['ids'][-1])],'source_primary_P':describe(p),'source_secondary_QCC':describe(q),'mean_between_readout_foreground_jaccard':float(np.mean(np.divide(intersection,union,out=np.ones_like(intersection,dtype=float),where=union>0))),'ground_truth_read':False,'optimizer_updates':0,'limits':['These are two predictions of the same source model, not comparison with ground truth.','Group count is a descriptive readout property, not object count accuracy.','No Slot Attention score, learned improvement or promotion is established.']},allow_nan=False))
