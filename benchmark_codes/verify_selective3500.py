"""Stdlib verification of SW0258 code, component eligibility and actual GPU admissions."""
import hashlib,itertools,json
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
def verify():
    request=json.loads((HERE/'CURRENT_EXPERIMENT.json').read_bytes())
    assert request['experiment']=='SW0258_selective3500_fixed_core' and request['total_seed_jobs']==18
    assert request['fresh_unique_TRAIN_per_component']==3500 and request['epochs']==10 and request['fresh_encoder_updates']==2190
    assert request['core_weights_frozen'] and not request['encoder_core_backward']
    selection=json.loads((HERE/request['component_selection']).read_bytes());cases=selection['cases']
    expected={f'{v}/{d}/{s}' for v,d,s in itertools.product((252,253),('clevr','tetrominoes','clevrtex_original'),range(3))}
    assert set(cases)==expected and sum(x['encoder'] is not None for x in cases.values())==8
    assert sum(x['head_reused'] for x in cases.values())==1
    for key,case in cases.items():
        for component in ('encoder','head'):
            if component=='head' and not case['head_reused']:continue
            source=case[component]
            if source is not None:assert source['unique_train']>=3500 and source['exposures']>=35000
        if key.startswith('253/'):assert case['head'] is None and not case['head_reused']
    proof=json.loads((HERE/request['validation_snapshot']).read_bytes())
    assert proof['experiment']=='SW0258' and proof['full_goal_complete'] is False
    assert proof['transition']['old_checkpoints_and_feature_caches_preserved']
    for path,sha in proof['preflight']['code'].items():
        parts=PurePosixPath(path).parts;local=HERE.joinpath(*parts[parts.index('benchmark_codes')+1:])
        assert hashlib.sha256(local.read_bytes()).hexdigest()==sha,str(local)
    assert len(proof['admissions'])==18
    assert {(r['complete']['variant'],r['complete']['dataset'],r['complete']['seed']) for r in proof['admissions']}==set(
        itertools.product((252,253),('clevr','tetrominoes','clevrtex_original'),range(3)))
    for row in proof['admissions']:
        done=row['complete'];binding=row['binding'];enc=row['encoder']
        assert done['diagnostic_only'] and done['evaluation_count']==1 and binding['code']==proof['preflight']['code']
        assert not binding['core_updates'] and not binding['core_backward']
        if not enc['reused']:
            assert enc['images_seen']==32 and enc['updates']==2 and enc['core_fixed'] and not enc['core_backward']
            assert enc['optimizer_restore_verified'] and all(v>0 for v in enc['gradient_coverage'].values())
            assert enc['mask_training']==(done['variant']==252)
        if done['variant']==252 and not done['head_reused']:
            assert row['head']['updates']==2 and row['head']['optimizer_restore_verified']
            assert all(v>0 for v in row['head']['gradient_coverage'].values())
    return dict(status='SW0258_selection_source_and_actual18_GPU_admissions_verified',encoder_reuses=8,head_reuses=1,
        fresh_encoder3500x10_updates=2190,full_goal_complete=False,completed_scientific_results=proof['scientific_results_complete'])
if __name__=='__main__':print(json.dumps(verify(),indent=2))
