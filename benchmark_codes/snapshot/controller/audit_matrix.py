"""Independent saved-history/checkpoint/frozen-mask audit and three-seed reports."""
import json,math
from pathlib import Path
import numpy as np
import torch
from mod_benchmark import common as c
from full_benchmark.data import Dataset
from full_benchmark.evaluation import metrics

KEYS=('fg_ari','foreground_iou','matched_object_iou')


def audit_seed(root,row):
    out=root/'jobs'/f"model{row['model']}"/row['dataset']/f"seed{row['seed']}"
    binding=json.loads((out/'binding.json').read_bytes());report=json.loads((out/'training_report.json').read_bytes())
    saved=json.loads((out/'evaluation/result.json').read_bytes());committed=json.loads((out/'committed.json').read_bytes())
    assert c.sha(out/'resume.pt')==committed['checkpoint_sha256']
    state=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
    assert state['updates']==2190 and state['images_seen']==35000 and state['next_epoch']==10 and state['next_offset']==0
    assert state['binding_sha256']==c.sha(out/'binding.json')
    assert state['initial_sha256']==c.sha(out/'initial.pt')
    for name in ('optimizer','head_optimizer'):
        if state[name] is not None:assert {int(x['step'].item()) for x in state[name]['state'].values()}=={2190}
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(state['model'].items()):digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    assert digest.hexdigest()==report['model_sha256'];del state
    ids=np.asarray(binding['dataset']['train_ids'],dtype=np.int64);assert len(np.unique(ids))==3500
    logs=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines() if line.strip()]
    assert len(logs)==2190
    seen=0
    for epoch in range(10):
        order=ids[np.random.RandomState(249000+row['seed']+10000*epoch).permutation(3500)]
        for index,start in enumerate(range(0,3500,16)):
            log=logs[epoch*219+index];selected=order[start:start+16].tolist();seen+=len(selected)
            assert log['update']==epoch*219+index+1 and log['epoch']==epoch+1
            assert log['ids']==selected and log['images_seen']==seen and math.isfinite(log['loss'])
    assert seen==35000
    protocol=json.loads((out/'evaluation/frozen_predictions.json').read_bytes())
    eval_ids=np.array(protocol['ids'],dtype=np.int64)
    assert protocol['GT_used_for_prediction'] is False and np.array_equal(eval_ids,binding['dataset']['evaluation_ids'])
    masks=[]
    for image_id in eval_ids:
        filename=f'{image_id:06d}.npy';path=out/'evaluation'/filename
        assert c.sha(path)==protocol['hashes'][filename]
        masks.append(np.load(path,allow_pickle=False))
    dataset=Dataset(row['dataset'],root);recomputed,_=metrics(np.stack(masks),dataset,eval_ids)
    for group in ('pixel','historical_fixed64_patch'):
        if group not in saved:continue
        for key in KEYS:assert math.isclose(saved[group][key],recomputed[group][key],rel_tol=0,abs_tol=1e-12)
    assert saved['valid_count']==recomputed['valid_count']
    return dict(seed=row['seed'],metrics=saved['pixel'],historical_fixed64_patch=saved.get('historical_fixed64_patch'),
        valid_count=saved['valid_count'],count=saved['count'],native_model_grid=saved['native_model_grid'],
        evaluation_contract=dataset.binding.get('evaluation_contract',dataset.binding['scoring']),
        initial_sha256=c.sha(out/'initial.pt'),checkpoint_sha256=committed['checkpoint_sha256'],
        all35000_training_exposures_and2190_updates_verified=True,all10_epoch_orders_exact=True,
        all_saved_masks_SHA_verified=True,immutable_metrics_independently_recomputed=True,
        effective_code_sha256=c.sha(out/'effective_code.json') if (out/'effective_code.json').exists() else None)


def write_reports(root,rows):
    root=Path(root);directory=root/'reports';directory.mkdir(exist_ok=True);finished=[]
    for model in (252,253,254,240):
        for dataset in ('clevr','multi_dsprites','tetrominoes','objects_room','clevrtex_original'):
            selected=[r for r in rows if r['model']==model and r['dataset']==dataset]
            if len(selected)!=3 or any(r['training']!='complete' for r in selected):continue
            path=directory/f'model{model}_{dataset}.json'
            if not path.exists():
                seeds=[audit_seed(root,row) for row in sorted(selected,key=lambda r:r['seed'])]
                result=dict(status='three_seeds_FULL3500x10_independently_audited',model=model,dataset=dataset,per_seed=seeds,
                    mean={k:float(np.mean([s['metrics'][k] for s in seeds])) for k in KEYS},
                    std_population={k:float(np.std([s['metrics'][k] for s in seeds])) for k in KEYS},
                    unique_TRAIN=3500,passes=10,seeds=[0,1,2],
                    raw_FG_IoU_semantics='all-positive exchangeable regions; no designated predicted background' if model in (252,254) else 'unchanged original largest-component predicted-background rule',
                    benchmark_unchanged=c.immutable())
                if dataset=='clevr':result['historical_fixed64_patch_mean']={k:float(np.mean([s['historical_fixed64_patch'][k] for s in seeds])) for k in KEYS}
                c.write(path,result)
            finished.append(dict(model=model,dataset=dataset,report=str(path),sha256=c.sha(path)))
    c.write(directory/'index.json',dict(audited_conditions=len(finished),required_conditions=20,reports=finished))
    return len(finished)
