"""Stdlib-only verification of published SW0256 source and actual partial evidence."""
import hashlib,itertools,json,math
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent


def verify():
    base=HERE/'frozen_core_benchmark'
    request=json.loads((base/'REQUEST.json').read_bytes())
    cases={(r['model'],r['dataset'],s) for r in request['conditions'] for s in r['seeds']}
    expected=set(itertools.product((252,253),('clevr','tetrominoes','clevrtex_original'),(0,1,2)))
    assert cases==expected and request['total_seed_jobs']==18
    for row in request['conditions']:
        assert row['unique_train']==35000 and row['epochs']==10 and row['grid']==64
        assert row['exposures']==35000*10 and row['updates']==math.ceil(35000/16)*10==21880
        assert row['core_frozen'] and row['encoder_finetuned']
    proof=json.loads((base/'validation/runtime_snapshot.json').read_bytes())
    assert proof['status']=='partial_validation_snapshot_not_final_benchmark'
    for path,sha in proof['preparation']['code'].items():
        parts=PurePosixPath(path).parts;name=Path(*parts[parts.index('benchmark_codes')+1:])
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,str(name)
    inventory_sha=hashlib.sha256((base/'sources.frontier.json').read_bytes()).hexdigest()
    assert proof['CPU']['source_inventory_sha256']==proof['preflight']['sources_sha256']==inventory_sha
    initial={(r['model'],r['seed']):r for r in proof['CPU']['rows']}
    assert set(initial)==set(itertools.product((252,253),(0,1,2)))
    for row in initial.values():
        assert row['strict_spatial_mapping_verified'] and row['all_core_and_integration_parameters_frozen']
        assert row['encoder_pretrained'] and row['normalization_preserved']
        assert set(row['active_groups'])==({'encoder','mask_head'} if row['model']==252 else {'encoder'})
    for name,row in proof['preflight']['datasets'].items():
        b=row['binding'];train=set(b['train_ids']);val=set(b['validation_ids']);test=set(b['evaluation_ids'])
        assert len(train)==35000 and not (train&val or train&test or val&test)
        assert len(test)==(320 if name=='clevr' else 1000)
    for name,sha in proof['preflight']['protected_benchmark'].items():
        assert hashlib.sha256((HERE/'vendor'/name).read_bytes()).hexdigest()==sha
    for item in proof['admissions']:
        done=item['complete'];report=item['report']
        assert done['diagnostic_only'] and done['updates']==2 and done['images_seen']==4 and done['evaluation_count']==1
        assert report['frozen_core_verified'] and report['optimizer_resume_verified']
        assert report['frozen_core_initial_and_final_sha256']==initial[done['variant'],done['seed']]['mapped_core_sha256']
        assert all(v>0 for v in report['active_tensor_gradient_coverage'].values())
        assert report['mask_opened_during_training']==(done['variant']==252)
    return dict(status='SW0256_source_protocol_and_partial_real_evidence_verified',required_seed_jobs=18,
        source_files_verified=len(proof['preparation']['code']),pretrained_initializations_verified=len(initial),
        actual_GPU_admissions_verified=len(proof['admissions']),
        final_benchmark_completed_at_snapshot=proof['queue_snapshot']['scientific_complete'],
        full_goal_completion_claimed=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
