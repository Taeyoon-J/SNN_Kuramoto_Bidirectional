"""Explicit source compatibility for a measured, output-preserving execution change.

Old running jobs keep their immutable package, binding and checkpoint provenance.
Only the scheduler accepts their exact recorded source inventory alongside the
new implementation, after checking both the actual GPU and initialization proof.
"""
import json
from pathlib import Path,PurePosixPath
from mod_benchmark import common as c


def canonical(inventory):
    result={}
    for path,sha in inventory.items():
        parts=PurePosixPath(path).parts
        name='/'.join(parts[parts.index('benchmark_codes')+1:])
        assert name not in result
        result[name]=sha
    return result


def verify_contract(contract,current):
    assert contract['status']=='verified_fused_no_grad_prefix_only'
    assert contract['to_code']==current
    old=canonical(contract['from_code']);new=canonical(current)
    changed={n for n in old.keys()|new.keys() if old.get(n)!=new.get(n)}
    assert changed=={'frozen_core_benchmark/execution.py','frozen_core_benchmark/model.py',
                     'frozen_core_benchmark/queue.py','vendor/snn_kuramoto_bidirectional/kuramoto_layer.py'}
    proofs={}
    for name,record in contract['proofs'].items():
        assert c.sha(Path(record['path']))==record['sha256']
        proofs[name]=json.loads(Path(record['path']).read_bytes())
    gpu=proofs['GPU'];assert gpu['status']=='all18_native64_full_forward_and_control_matched_gradients_passed'
    assert gpu['candidate_sha256']==new['vendor/snn_kuramoto_bidirectional/kuramoto_layer.py']
    assert gpu['physical_microbatch']==1 and gpu['logical_batch']==16
    assert gpu['rollout_steps']==1024 and gpu['gradient_tail_steps']==64
    assert gpu['no_grad_prefix_bmm_count']==2 and gpu['live_tail_bmm_count']==4
    assert gpu['scientific_training_workers_modified'] is False
    rows=gpu['rows'];expected={(v,d,s) for v in (252,253)
        for d in ('clevr','tetrominoes','clevrtex_original') for s in range(3)}
    assert len(rows)==18 and {(r['model'],r['dataset'],r['seed']) for r in rows}==expected
    for r in rows:
        assert r['all_full_forward_tensor_hashes_equal'] and r['loss_equal'] and r['core_unchanged']
        assert r['gradient_limit']==max(1e-7,r['control_gradient']['relative_l2']*5)
        assert r['fused_prefix_gradient']['relative_l2']<=r['gradient_limit']
    cpu=proofs['CPU'];assert cpu['code']==current and cpu['source_inventory_sha256']==contract['sources_sha256']
    expected={(v,s) for v in (252,253) for s in range(3)}
    before={(r['model'],r['seed']):r for r in contract['previous_CPU_rows']}
    after={(r['model'],r['seed']):r for r in cpu['rows']}
    assert before.keys()==after.keys()==expected
    for key in expected:
        for name in ('mapped_core_sha256','encoder_sha256','active_groups'):
            assert before[key][name]==after[key][name]
        assert after[key]['all_core_and_integration_parameters_frozen']
    return contract


def verify_code(root,inventory,current):
    if inventory==current:return
    contract=verify_contract(json.loads((Path(root)/'execution_compatibility.json').read_bytes()),current)
    assert inventory==contract['from_code'],'unregistered scientific source inventory'


def allowed_entrypoints(root,current):
    result={str(Path(__file__).parents[1]/'run_frozen_core.py')}
    path=Path(root)/'execution_compatibility.json'
    if path.exists():
        contract=verify_contract(json.loads(path.read_bytes()),current)
        result.update(p for p in contract['from_code'] if p.endswith('/run_frozen_core.py'))
    return result
