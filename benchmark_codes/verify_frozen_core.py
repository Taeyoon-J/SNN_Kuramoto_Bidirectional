"""Stdlib-only verification of published SW0256 source and actual partial evidence."""
import hashlib,itertools,json,math
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent


def canonical(code):
    result={}
    for path,sha in code.items():
        parts=PurePosixPath(path).parts
        name='/'.join(parts[parts.index('benchmark_codes')+1:])
        assert name not in result;result[name]=sha
    return result


def verify_execution_revision(base,baseline):
    revision=json.loads((base/'validation/execution_revision.json').read_bytes())
    contract=revision['contract'];assert contract['status']=='verified_fused_no_grad_prefix_only'
    old=canonical(contract['from_code']);new=canonical(contract['to_code'])
    assert old==canonical(baseline['preparation']['code'])
    changed={n for n in old.keys()|new.keys() if old.get(n)!=new.get(n)}
    assert changed=={'frozen_core_benchmark/model.py','frozen_core_benchmark/queue.py',
                     'frozen_core_benchmark/execution.py','vendor/snn_kuramoto_bidirectional/kuramoto_layer.py'}
    for name,sha in new.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
    proofs={}
    for name,record in revision['local_proofs'].items():
        path=base/'validation'/record['file'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
        assert record['sha256']==contract['proofs'][name]['sha256']
        proofs[name]=json.loads(raw)
    gpu=proofs['GPU'];cpu=proofs['CPU']
    assert gpu['status']=='all18_native64_full_forward_and_control_matched_gradients_passed'
    assert gpu['candidate_sha256']==new['vendor/snn_kuramoto_bidirectional/kuramoto_layer.py']
    assert gpu['physical_microbatch']==1 and gpu['logical_batch']==16
    assert gpu['rollout_steps']==1024 and gpu['gradient_tail_steps']==64
    assert gpu['no_grad_prefix_bmm_count']==2 and gpu['live_tail_bmm_count']==4
    expected=set(itertools.product((252,253),('clevr','tetrominoes','clevrtex_original'),(0,1,2)))
    assert len(gpu['rows'])==18 and {(r['model'],r['dataset'],r['seed']) for r in gpu['rows']}==expected
    for row in gpu['rows']:
        assert row['all_full_forward_tensor_hashes_equal'] and row['loss_equal'] and row['core_unchanged']
        assert row['gradient_limit']==max(1e-7,5*row['control_gradient']['relative_l2'])
        assert row['fused_prefix_gradient']['relative_l2']<=row['gradient_limit']
    before={(r['model'],r['seed']):r for r in baseline['CPU']['rows']}
    after={(r['model'],r['seed']):r for r in cpu['rows']}
    assert before.keys()==after.keys() and len(after)==6 and canonical(cpu['code'])==new
    for key in before:
        for name in ('mapped_core_sha256','encoder_sha256','active_groups'):assert before[key][name]==after[key][name]
    assert revision['activation']['running_training_packages_unchanged']
    assert revision['activation']['no_checkpoint_optimizer_binding_or_training_history_edits']
    assert revision['batch2']['status']=='queued_until14500MiB_is_free'
    checks={}
    for name,record in revision['additional_checks'].items():
        raw=(base/'validation'/record['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record['sha256'];checks[name]=json.loads(raw)
    inference=checks['inference']
    assert inference['status']=='all18_actual_optimized_package_inference_forwards_bitwise_equal'
    assert len(inference['rows'])==18 and {(r['model'],r['dataset'],r['seed']) for r in inference['rows']}==expected
    assert all(r['live_tail_steps']==0 and r['full1024_forward_bitwise_equal'] and r['core_unchanged'] for r in inference['rows'])
    assert inference['protected_benchmark']==baseline['preflight']['protected_benchmark']
    guard=checks['guards'];assert guard['status']=='execution_compatibility_fail_closed_checks_passed'
    assert {'unregistered_source_hash_rejected','incomplete_GPU_coverage_rejected',
            'changed_encoder_gradient_rejected','changed_initial_encoder_rejected'}<=set(guard['checks'])
    return new


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
    code=verify_execution_revision(base,proof)
    for name,sha in code.items():
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
        source_files_verified=len(code),pretrained_initializations_verified=len(initial),
        real_full_forward_gradient_execution_controls_verified=18,
        actual_GPU_admissions_verified=len(proof['admissions']),
        final_benchmark_completed_at_snapshot=proof['queue_snapshot']['scientific_complete'],
        full_goal_completion_claimed=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
