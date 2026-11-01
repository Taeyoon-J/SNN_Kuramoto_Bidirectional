"""Stdlib-only validation of SW0257 published source and actual admission evidence."""
import hashlib,itertools,json
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent


def verify():
    request=json.loads((HERE/'CURRENT_EXPERIMENT.json').read_bytes())
    assert request['experiment']=='SW0257_separate_encoder35000'
    assert request['total_seed_jobs']==18 and request['unique_TRAIN_per_job']==35000 and request['epochs']==10
    assert request['encoder_updates']==21880 and request['supervised_head_additional_updates']==35000
    assert not request['encoder_core_backward'] and request['encoder_core_simulations']==0
    proof=json.loads((HERE/request['validation_snapshot']).read_bytes())
    assert proof['total_conditions']==18 and proof['goal_complete'] is False
    pre=proof['preflight'];assert pre['experiment']=='SW0257' and not pre['encoder_core_backprop']
    assert pre['encoder_physical_batch']==16 and pre['head_budget']=='3500x10'
    count=0
    for origin,sha in pre['code'].items():
        parts=PurePosixPath(origin).parts;path=HERE.joinpath(*parts[parts.index('benchmark_codes')+1:])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,str(path);count+=1
    expected=pre['protected_benchmark']
    for name,sha in expected.items():assert hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()==sha
    assert len(proof['admissions'])==18
    assert {(r['binding']['variant'],r['binding']['dataset']['dataset'],r['binding']['seed']) for r in proof['admissions']}==set(
        itertools.product((252,253),('clevr','tetrominoes','clevrtex_original'),(0,1,2)))
    for row in proof['admissions']:
        done=row['complete'];binding=row['binding'];enc=row['encoder']
        assert binding['code']==pre['code']
        assert done['diagnostic_only'] and done['count']==1 and done['status']=='admission_passed'
        assert binding['diagnostic_only'] and not binding['core_backward'] and not binding['core_weight_updates']
        assert enc['updates']==2 and enc['images_seen']==32 and enc['core_simulation_calls']==0
        assert enc['core_bitwise_fixed'] and enc['optimizer_restore_verified']
        assert enc['mask_used']==(binding['variant']==252)
        assert all(x>0 for x in enc['gradient_coverage'].values())
        if binding['variant']==252:
            head=row['head'];assert head['updates']==2 and head['unique_train']==2
            assert head['encoder_core_gradients_absent'] and head['optimizer_restore_verified']
            assert all(v>0 for v in head['gradient_coverage'].values())
    return dict(status='SW0257_source_and_actual_partial_admissions_verified',source_files=count,
        actual_admissions=len(proof['admissions']),scientific_results_complete=proof['full_training_complete'],
        full_goal_complete=False,warning='admissions are diagnostics, not benchmark results')


if __name__=='__main__':print(json.dumps(verify(),indent=2))
