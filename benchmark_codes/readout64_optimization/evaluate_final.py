"""Final64 contract: freeze all predictions before truth, with learned-routing ablations."""
import argparse,copy,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True);p.add_argument('--factory');p.add_argument('--factory-class',default='ObjectiveReadout');a=p.parse_args();root=Path(a.root)
 while not (root/'extension_complete.json').exists():time.sleep(15)
 sys.path.insert(0,a.code)
 import run as opt
 factory=opt.FineReadout
 if a.factory:
  sys.path.insert(0,str(Path(a.factory).parent))
  spec=importlib.util.spec_from_file_location('sw0260_evaluation_factory',a.factory);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);factory=getattr(module,a.factory_class)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.15,0)
 selection=json.loads((root/'final_selection.json').read_bytes());arm=selection['arm']
 variants=['old259_lifted64','initial_structure','trained','zero_fine_adapter','coarsened_gamma']
 frozen=[]
 for seed in range(3):
  old,_,blob,source=opt.original(seed);old._seed_hint=seed;initial=factory(old,arm).cuda().eval()
  checkpoint=root/'extended'/f'seed{seed}'/arm/'best.pt'
  saved=torch.load(checkpoint,weights_only=True,map_location='cuda:0')
  trained=factory(old,arm).cuda().eval();trained.load_state_dict(saved['model'])
  zero=factory(old,arm).cuda().eval();zero.load_state_dict(saved['model'])
  with torch.no_grad():zero.fine.weight.zero_()
  rows=opt.load_rows(root,seed,'fixed320',320)
  assert [r['image_id'] for r in rows]==list(range(1320,1640))
  predictions={name:[] for name in variants}
  fine_variations=[]
  with torch.no_grad():
   for row in rows:
    r=opt.device_row(row)
    predictions['old259_lifted64'].append((opt.expand32(row['prior'].T).T.argmax(0)+1).reshape(64,64).numpy())
    predictions['initial_structure'].append(initial.predict(r).cpu().numpy())
    predictions['trained'].append(trained.predict(r).cpu().numpy())
    predictions['zero_fine_adapter'].append(zero.predict(r).cpu().numpy())
    averaged=r['gamma'].reshape(32,2,32,2,8).mean((1,3))
    coarse_gamma=averaged.repeat_interleave(2,0).repeat_interleave(2,1).reshape(4096,8)
    coarse=dict(r,gamma=coarse_gamma)
    predictions['coarsened_gamma'].append(trained.predict(coarse).cpu().numpy())
  out=root/'final'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
  arrays={name:np.stack(value) for name,value in predictions.items()}
  with np.load(opt.OLD/f'seed{seed}/predictions.npz') as original_predictions:
   expected=original_predictions['labels'].repeat(2,1).repeat(2,2)
   assert np.array_equal(arrays['old259_lifted64'],expected),'baseline differs from published259'
  path=out/'predictions.npz';np.savez_compressed(path,image_ids=np.arange(1320,1640),**arrays)
  record=dict(seed=seed,arm=arm,checkpoint_sha256=opt.sha(checkpoint),checkpoint_file=str(checkpoint),source_checkpoint_sha256=opt.sha(source),prediction_sha256=opt.sha(path),
    core_grid=32,head_grid=64,GT_used_for_prediction=False,selection=selection,factory_file=a.factory,factory_class=a.factory_class if a.factory else 'FineReadout',factory_sha256=opt.sha(a.factory) if a.factory else None,
    old_prediction_sha256=opt.sha(opt.OLD/f'seed{seed}/predictions.npz'),
    changed_from_old=float(np.mean(arrays['trained']!=arrays['old259_lifted64'])),
    changed_from_initial=float(np.mean(arrays['trained']!=arrays['initial_structure'])),
    fine_adapter_effect=float(np.mean(arrays['trained']!=arrays['zero_fine_adapter'])),
    fine_resolution_effect=float(np.mean(arrays['trained']!=arrays['coarsened_gamma'])))
  opt.write(out/'prediction_protocol.json',record);frozen.append(record)
  del trained,zero,initial;torch.cuda.empty_cache()
 opt.write(root/'all_predictions_frozen.json',dict(records=frozen,GT_used=False,variants=variants,created=time.time()))
 # Only now open the final fixed320 mask labels. Same native64 target for all arms.
 import h5py
 sys.path.insert(0,str(opt.LEGACY))
 from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch,evaluate_patch_masks
 with h5py.File(opt.DATA,'r') as f:truth=clevr_mask_patch(torch.from_numpy(np.asarray(f['mask'][1320:1640])),2)['patch_labels']
 results=[]
 for seed in range(3):
  out=root/'final'/f'seed{seed}'
  with np.load(out/'predictions.npz') as arrays:
   metrics={}
   for name in variants:
    scored=evaluate_patch_masks(torch.from_numpy(arrays[name]),truth)
    metrics[name]={k:float(v) for k,v in scored['mean'].items()}
  result=dict(seed=seed,arm=arm,metrics=metrics,protocol=frozen[seed]);opt.write(out/'result.json',result);results.append(result)
 mean={name:{key:float(np.mean([r['metrics'][name][key] for r in results])) for key in ['fg_ari','foreground_iou','matched_object_iou']} for name in variants}
 report=dict(selected_arm=arm,per_seed=results,mean=mean,core_grid=32,head_grid=64,
  FGARI_gain_over_same64_baseline=mean['trained']['fg_ari']-mean['old259_lifted64']['fg_ari'],
  learned_routing_gain=mean['trained']['fg_ari']-mean['initial_structure']['fg_ari'],
  fine_adapter_gain=mean['trained']['fg_ari']-mean['zero_fine_adapter']['fg_ari'],
  fine_resolution_gain=mean['trained']['fg_ari']-mean['coarsened_gamma']['fg_ari'],
  historical_native32_FGARI=.8216282697216665,historical32_is_not_same_contract=True,
  TRAIN_masks_used=False,labelled_validation_used_for_selection=True,foreground_iou_warning='all output IDs positive; rawFGIoU equals GT foreground area fraction')
 opt.write(root/'final_results.json',report);print(json.dumps(dict(mean=mean,gain=report['FGARI_gain_over_same64_baseline'],learned_gain=report['learned_routing_gain'])),flush=True)

if __name__=='__main__':main()
