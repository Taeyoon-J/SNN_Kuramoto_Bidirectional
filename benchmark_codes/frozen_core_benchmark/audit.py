"""Independent saved-history/checkpoint/frozen-mask audit and three-seed reports."""
import json,math
from pathlib import Path
import numpy as np
import torch
from mod_benchmark import common as c
from frozen_core_benchmark.data import Dataset
from full_benchmark.evaluation import metrics

KEYS=('fg_ari','foreground_iou','matched_object_iou')


def audit_seed(root,row):
    out=root/'jobs'/f"model{row['model']}"/row['dataset']/f"seed{row['seed']}"
    binding=json.loads((out/'binding.json').read_bytes());report=json.loads((out/'training_report.json').read_bytes())
    saved=json.loads((out/'evaluation/result.json').read_bytes());committed=json.loads((out/'committed.json').read_bytes())
    assert c.sha(out/'resume.pt')==committed['checkpoint_sha256']
    state=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
    assert state['updates']==21880 and state['images_seen']==350000 and state['next_epoch']==10 and state['next_offset']==0
    assert state['binding_sha256']==c.sha(out/'binding.json')
    assert state['initial_sha256']==c.sha(out/'initial.pt')
    for name in ('optimizer','head_optimizer'):
        if state[name] is not None:assert {int(x['step'].item()) for x in state[name]['state'].values()}=={21880}
    import hashlib
    digest=hashlib.sha256()
    for name,value in sorted(state['model'].items()):digest.update(name.encode());digest.update(value.contiguous().numpy().tobytes())
    assert digest.hexdigest()==report['model_sha256']
    frozen=hashlib.sha256()
    for name,value in sorted(state['model'].items()):
        if name.startswith('wrapped.'):
            frozen.update(name[len('wrapped.'):].encode());frozen.update(value.contiguous().numpy().tobytes())
    assert report['frozen_core_verified'] and frozen.hexdigest()==report['frozen_core_initial_and_final_sha256']
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
    for name,value in state['model'].items():
        if name.startswith('wrapped.'):
            assert torch.equal(value,initial['model'][name]),('fixed core changed',name)
    assert any(not torch.equal(value,initial['model'][name]) for name,value in state['model'].items() if name.startswith('encoder.'))
    del initial
    active_names=set(state['gradient_coverage'])
    assert all(name.startswith('encoder.') or name.startswith('head.') for name in active_names)
    assert all(v>0 for v in state['gradient_coverage'].values())
    del state
    ids=np.asarray(binding['dataset']['train_ids'],dtype=np.int64);assert len(np.unique(ids))==35000
    logs=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines() if line.strip()]
    assert len(logs)==21880
    seen=0
    for epoch in range(10):
        order=ids[np.random.RandomState(249000+row['seed']+10000*epoch).permutation(35000)]
        for index,start in enumerate(range(0,35000,16)):
            log=logs[epoch*2188+index];selected=order[start:start+16].tolist();seen+=len(selected)
            assert log['update']==epoch*2188+index+1 and log['epoch']==epoch+1
            assert log['ids']==selected and log['images_seen']==seen and math.isfinite(log['loss'])
    assert seen==350000
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
        all350000_training_exposures_and21880_updates_verified=True,all10_epoch_orders_exact=True,
        frozen_core_bitwise_unchanged=True,encoder_gradient_coverage_verified=True,
        all_saved_masks_SHA_verified=True,immutable_metrics_independently_recomputed=True,
        effective_code_sha256=c.sha(out/'effective_code.json') if (out/'effective_code.json').exists() else None)

