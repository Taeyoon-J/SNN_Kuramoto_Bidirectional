"""Reuse immutable SW0251 activities; train and evaluate SW0252 independently."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from neutral_readout.head import NeutralRegionHead,objective,candidates
from readout_benchmark import runtime as r
from mod_benchmark import common as c


def selftest():
    torch.set_num_threads(1);torch.manual_seed(37)
    x=torch.randn(25,144)
    regions=np.array([[0]*5]*3+[[7]*5]*2)
    head=NeutralRegionHead()
    masks,prior=candidates(x,regions)
    assert bool((masks[:2].sum(0)==1).all()) and int(masks[0].sum())==15
    p=head.predict(x,regions)
    assert np.all(p>0) and np.array_equal(p[:3],np.ones((3,5),dtype=np.int64))
    renamed=np.where(regions==0,91,0)
    q=head.predict(x,renamed)
    assert np.array_equal(p.flatten()[:,None]==p.flatten()[None],q.flatten()[:,None]==q.flatten()[None])
    y=np.array([[0]*5,[0]*5,[2]*5,[2]*5,[8]*5])
    logits=head(x,regions);loss=objective(logits,y)
    yp=np.where(y==0,8,np.where(y==8,2,0))
    assert torch.allclose(loss,objective(logits,yp),atol=1e-6)
    order=torch.randperm(len(logits))
    assert torch.allclose(loss,objective(logits[order],y),atol=1e-6)
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert sum(float(p.grad.abs().sum()) for p in head.parameters())>0
    optimizer=torch.optim.Adam(head.parameters(),lr=.003)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    objective(head(x,regions),y).backward()
    assert sum(float(p.grad.abs().sum()) for p in head.embedding.parameters())>0
    print(json.dumps(dict(status='neutral_invariants_passed',checks=['largest_region_retained','zero_ID_ordinary_region',
        'all_pixels_assigned','proposal_ID_equivariance','GT_ID_invariance_including_background',
        'output_permutation_invariance','finite_connected_gradients'])))


def run(root,out):
    torch.set_num_threads(1);c.immutable()
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.12,0)
    device='cuda:0';out=Path(out);root=Path(root);out.mkdir(parents=True,exist_ok=True)
    own={p.name:c.sha(p) for p in Path(__file__).parent.glob('*.py')}
    binding=dict(code=own,source_code=r.fingerprint(),count=256,validation_count=64,epochs=20,lr=.003,
        train_ids=r.ids_for('clevr','train',256).tolist(),validation_ids=r.ids_for('clevr','validation',64).tolist(),
        no_background_slot=True,no_region_deletion=True,all_GT_regions_Hungarian_matched=True,
        baseline_masks_not_used_for_prediction=True,core_encoder_frozen=True)
    if (out/'binding.json').exists():assert json.loads((out/'binding.json').read_bytes())==binding
    else:c.write(out/'binding.json',binding)
    train_ids=np.asarray(binding['train_ids']);val_ids=np.asarray(binding['validation_ids'])
    truth=r.truth('clevr','train',train_ids);val_truth=r.truth('clevr','validation',val_ids)
    train_reports=[];fixed=[]
    metric=c.benchmark_module('evaluation')
    for seed in range(3):
        folder=r.row_folder(root,'clevr','zero_shot',seed)
        for split,count in [('train',256),('validation',64),('fixed320',320)]:
            record=json.loads((folder/f'complete_{split}_{count}.json').read_bytes())
            assert record['binding']['code']==binding['source_code']
        d=out/f'seed{seed}';d.mkdir(exist_ok=True)
        head=NeutralRegionHead(seed=251000+seed).to(device)
        optimizer=torch.optim.Adam(head.parameters(),lr=.003)
        first=0;history=[];best=-float('inf')
        if (d/'resume.pt').exists():
            checkpoint=torch.load(d/'resume.pt',map_location=device,weights_only=True)
            assert checkpoint['binding']==binding
            head.load_state_dict(checkpoint['head']);optimizer.load_state_dict(checkpoint['optimizer'])
            first=checkpoint['epoch'];history=checkpoint['history'];best=checkpoint['best']
        for epoch in range(first,20):
            head.train();losses=[]
            for index in np.random.RandomState(251000+seed+10000*epoch).permutation(len(train_ids)):
                row=r.read_row(folder,train_ids[index],device)
                optimizer.zero_grad(set_to_none=True)
                loss=objective(head(row['features'],row['all_support']),truth[index]);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                if not bool(torch.isfinite(norm)) or float(norm)<=0:raise FloatingPointError('disconnected gradient')
                optimizer.step();losses.append(float(loss.detach()))
            head.eval();pred=[]
            for image_id in val_ids:
                row=r.read_row(folder,image_id,device);pred.append(head.predict(row['features'],row['all_support']))
            score=r.values(pred,val_truth)
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=score,updates=(epoch+1)*256))
            if score['fg_ari']>best:
                best=score['fg_ari'];r.save(d/'best.pt',dict(head=head.state_dict(),binding=binding,seed=seed,epoch=epoch+1))
            r.save(d/'resume.pt',dict(head=head.state_dict(),optimizer=optimizer.state_dict(),binding=binding,
                                   epoch=epoch+1,history=history,best=best))
            c.write(out/'progress.json',dict(seed=seed,epoch=epoch+1,total_epochs=20,validation=score))
        train_reports.append(dict(seed=seed,history=history,best_validation_fg_ari=best,head_sha256=c.sha(d/'best.pt')))
        state=torch.load(d/'best.pt',map_location=device,weights_only=True);head.load_state_dict(state['head']);head.eval()
        predictions=[];ids=r.ids_for('clevr','fixed320',320)
        for image_id in ids:
            row=r.read_row(folder,image_id,device);predictions.append(head.predict(row['features'],row['all_support']))
        predictions=np.stack(predictions);assert np.all(predictions>0)
        path=d/'predictions_fixed320.npz'
        with path.open('wb') as stream:np.savez_compressed(stream,predictions=predictions,ids=ids)
        fixed.append(dict(seed=seed,path=str(path),sha256=c.sha(path),head_sha256=c.sha(d/'best.pt')))
    # Freeze all three predictions BEFORE opening fixed320 truth.
    c.write(out/'prediction_protocol.json',dict(binding=binding,records=fixed,GT_used_for_prediction=False))
    gt=r.truth('clevr','fixed320',r.ids_for('clevr','fixed320',320));rows=[]
    for record in fixed:
        with np.load(record['path'],allow_pickle=False) as v:pred=v['predictions'].copy()
        scored=metric.evaluate_patch_masks(torch.from_numpy(pred),torch.from_numpy(gt))
        rows.append(dict(seed=record['seed'],metrics={k:float(scored['mean'][k]) for k in r.KEYS},
            all_pixels_assigned=True,background_region_removed=False,
            mean_groups=float(np.mean([len(np.unique(p)) for p in pred]))))
        with (out/f"seed{record['seed']}"/'scores_fixed320.npz').open('wb') as stream:
            np.savez_compressed(stream,**{k:v.numpy() for k,v in scored['per_image'].items()})
    baseline=json.loads((root/'original_confirmation/result.json').read_bytes())
    result=dict(status='completed_fixed320_three_seed',experiment='SW0252',per_seed=rows,
        mean={k:float(np.mean([v['metrics'][k] for v in rows])) for k in r.KEYS},
        baseline240=baseline['mean_baseline'],baseline251=baseline['mean_modified'],
        training=train_reports,binding=binding,benchmark_unchanged=c.immutable(),
        scope='reused fixed320 validation, not untouched test',
        IoU_interpretation='Raw all-region output; no predicted background is identified or removed. Foreground IoU therefore treats every pixel as foreground.',
        created_unix=time.time())
    c.write(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('training','binding','benchmark_unchanged')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--root');p.add_argument('--output');a=p.parse_args()
    if a.selftest:selftest()
    else:run(a.root,a.output)
