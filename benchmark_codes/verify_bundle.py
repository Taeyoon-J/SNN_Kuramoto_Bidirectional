#!/usr/bin/env python3
"""Pure-stdlib check: vendored source is byte-identical to the actual server export."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((HERE/'contracts/source_snapshot.json').read_text())
    count=0
    for name,record in manifest['origins'].items():
        path=HERE/name
        if not name.startswith('vendor/'):continue
        assert path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
        count+=1
    assert count>50
    for dataset in ('clevr','multi_dsprites','tetrominoes','objects_room','clevrtex_original'):
        split=json.loads((HERE/'contracts'/f'{dataset}_split.json').read_text())
        train=set(split['train_ids']);validation=set(split['validation_ids']);test=set(split['evaluation_ids'])
        assert len(train)==3500 and not (train&validation or train&test or validation&test)
    request=json.loads((HERE/'configs/active_request.json').read_text(encoding='utf-8'))
    conditions=request['conditions']
    assert len(conditions)==20 and len({(x['model'],x['dataset']) for x in conditions})==20
    assert all(x['seeds']==[0] and x['epochs']==10 and x['unique_train_images']==3500 for x in conditions)
    assert sum(x['phase']==1 for x in conditions)==12
    assert all(x['native_grid']==64 for x in conditions)
    revision=json.loads((HERE/'validation/native64_253_revision.json').read_text())
    assert revision['model_grid']==64 and revision['patches']==4096
    assert revision['encoder_parameters']==224 and revision['graph_top_k']==512
    assert not revision['training_masks_used'] and revision['optimizer_restore_verified']
    assert revision['all_active_tensor_gradients_nonzero_and_finite']
    assert revision['all_active_groups_changed']
    for filename,record in revision['source_files'].items():
        assert hashlib.sha256((HERE/'vendor/full_benchmark'/filename).read_bytes()).hexdigest()==record['sha256']
    files=list(HERE.rglob('*.py'))
    for path in files:compile(path.read_bytes(),str(path),'exec')
    assets=json.loads((HERE/'contracts/assets.json').read_text(encoding='utf-8-sig'))
    assert len({x['destination'] for x in assets['assets']})==len(assets['assets'])
    assert not any(x['destination'].endswith(('resume.pt','initial.pt')) for x in assets['assets'])
    return dict(status='vendored_source_and_split_integrity_passed',identical_server_source_files=count,python_files_compiled=len(files),datasets=5,unique_TRAIN_each=3500,learned_checkpoint_required=False)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
