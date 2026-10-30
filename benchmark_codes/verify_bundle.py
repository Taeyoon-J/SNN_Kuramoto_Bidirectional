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
    files=list(HERE.rglob('*.py'))
    for path in files:compile(path.read_bytes(),str(path),'exec')
    assets=json.loads((HERE/'contracts/assets.json').read_text(encoding='utf-8-sig'))
    assert len({x['destination'] for x in assets['assets']})==len(assets['assets'])
    assert not any(x['destination'].endswith(('resume.pt','initial.pt')) for x in assets['assets'])
    return dict(status='vendored_source_and_split_integrity_passed',identical_server_source_files=count,python_files_compiled=len(files),datasets=5,unique_TRAIN_each=3500,learned_checkpoint_required=False)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
