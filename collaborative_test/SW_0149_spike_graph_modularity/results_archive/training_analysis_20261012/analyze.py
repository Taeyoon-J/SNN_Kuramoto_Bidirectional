import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,math,statistics
from pathlib import Path
root=Path('/Data0/kevinswk/patch_v2_sw')
sys.path[:0]=[str(root),str(root/'collaborative_test'),str(root/'snn_kuramoto_bidirectional')]
from collaborative_test.SW_0149_spike_graph_modularity import train
arms={}
for arm in train.ARMS:
 validated=train.validate_training(arm)
 manifest=validated['manifest']
 history=json.loads(validated['history'].read_text())
 losses=[statistics.mean(float(m['modularity_loss']) for m in row['microbatches']) for row in history]
 gradients={}
 for family in history[0]['gradient_norms']:
  norms=[float(row['gradient_norms'][family]) for row in history]
  if any(not math.isfinite(x) or x<0 for x in norms):raise ValueError('Invalid recorded gradient norm')
  gradients[family]={'minimum':min(norms),'maximum':max(norms),'median':statistics.median(norms),'positive_update_fraction':sum(x>0 for x in norms)/len(norms),'first16_mean':statistics.mean(norms[:16]),'last16_mean':statistics.mean(norms[-16:])}
 if any(not math.isfinite(x) for x in losses):raise ValueError('Invalid recorded training loss')
 arms[arm]={'manifest_sha256':validated['manifest_sha256'],'checkpoint_sha256':validated['checkpoint_sha256'],'history_sha256':train.sha(validated['history']),'updates':len(history),'exposures':sum(len(row['image_ids']) for row in history),'loss_first16_mean':statistics.mean(losses[:16]),'loss_last16_mean':statistics.mean(losses[-16:]),'loss_minimum':min(losses),'loss_maximum':max(losses),'gradient_norms':gradients,'initial_parameter_sha256':manifest.get('initial_parameter_sha256'),'final_parameter_sha256':manifest.get('final_parameter_sha256'),'optimizer_step_range':manifest['optimizer_step_range'],'ground_truth_used':False}
print(json.dumps({'experiment':train.NAME,'evidence_kind':'actual_completed_training_history_summary','arms':arms,'limits':['Training-loss windows contain different images and do not measure matched-image improvement.','Positive family gradients do not prove individual gate necessity or segmentation improvement.','Fixed320 evaluation and three-seed/native70k requirements remain pending.']},allow_nan=False))
