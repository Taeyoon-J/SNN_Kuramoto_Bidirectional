"""Independent single-seed reports; never label one seed as a three-seed mean."""
import json
from pathlib import Path
from mod_benchmark import common as c
from snapshot.controller.audit_matrix import audit_seed
def write_reports(root,rows):
    root=Path(root);directory=root/'reports/single_seed';directory.mkdir(parents=True,exist_ok=True);finished=[]
    assert all(row['seed']==0 for row in rows)
    for row in rows:
        if row['training']!='complete':continue
        out=root/'jobs'/f"model{row['model']}"/row['dataset']/'seed0'
        cached=root/'reports'/f"individual_model{row['model']}_{row['dataset']}_seed0.json"
        path=directory/f"model{row['model']}_{row['dataset']}.json"
        complete_sha=c.sha(out/'complete.json')
        if path.exists():
            result=json.loads(path.read_bytes());assert result['completion_sha256']==complete_sha
        else:
            if cached.exists():
                cache=json.loads(cached.read_bytes());assert cache['completion_sha256']==complete_sha
                verified=cache['independent_audit'];assert verified['checkpoint_sha256']==c.sha(out/'resume.pt')
            else:verified=audit_seed(root,row)
            assert verified['all35000_training_exposures_and2190_updates_verified']
            assert verified['all10_epoch_orders_exact'] and verified['immutable_metrics_independently_recomputed']
            result=dict(status='single_seed0_FULL3500x10_independently_audited',model=row['model'],dataset=row['dataset'],
                seed=0,seeds=[0],single_seed_only=True,unique_TRAIN=3500,epochs=10,exposures=35000,updates=2190,
                metrics=verified['metrics'],historical_fixed64_patch=verified.get('historical_fixed64_patch'),
                valid_count=verified['valid_count'],count=verified['count'],native_model_grid=verified['native_model_grid'],
                independent_audit=verified,completion_sha256=complete_sha,
                raw_FG_IoU_semantics='all-positive exchangeable regions; no designated predicted background' if row['model'] in (252,254) else 'original largest-component predicted-background rule',
                benchmark_unchanged=c.immutable())
            c.write(path,result)
        assert result['single_seed_only'] is True and result['seeds']==[0]
        assert 'mean' not in result and 'std_population' not in result
        finished.append(dict(model=row['model'],dataset=row['dataset'],seed=0,report=str(path),sha256=c.sha(path)))
    index=dict(mode='single_seed0_priority',selected_seeds=[0],audited_conditions=len(finished),required_conditions=20,reports=finished)
    c.write(directory/'index.json',index);c.write(root/'reports/index.json',index)
    return len(finished)
