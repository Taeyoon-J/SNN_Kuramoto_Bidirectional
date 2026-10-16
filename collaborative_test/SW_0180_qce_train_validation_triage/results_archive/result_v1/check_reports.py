import hashlib,json,math
from pathlib import Path
root=Path('/Data0/kevinswk/patch_v2_sw')
private=root/'collaborative_testing/collaborative_private_runtime_20261008/SW0180_delivery_20261016'
base=root/'collaborative_test/SW_0180_qce_train_validation_triage/results_archive'
probe=json.loads((private/'cpu_source_probe.json').read_text())
paths={'history.json':private/'training_history_summary.json'}
summary=[]
for seed in (0,1,2):
    state_path=base/f'train8_queue_20261016/seed{seed}_state.json'
    state=json.loads(state_path.read_text());assert state['status']=='complete' and state['seed']==seed
    path=Path(state['report_path']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==state['report_sha256']
    report=json.loads(raw);record=report['seeds'][str(seed)]
    expected=next(row for row in probe['records'] if row['seed']==seed)
    assert report['image_ids']==probe['image_ids'] and report['image_positions']==probe['positions']
    assert set(report['seeds'])=={str(seed)} and report['optimizer_updates']==0 and report['validation_data_read'] is False
    assert record['source_model_state_sha256']==expected['fresh_model_state_sha256']
    assert record['final_model_state_sha256']==expected['final_model_state_sha256']
    assert record['training_manifest_sha256']==expected['final_manifest_sha256']
    assert record['state_unchanged']=={'fresh':True,'final':True}
    for values in record['qce_per_image'].values():assert len(values)==8 and all(math.isfinite(v) for v in values)
    assert abs(record['qce_delta_mean']-(sum(record['qce_per_image']['final'])-sum(record['qce_per_image']['fresh']))/8)<1e-14
    mean=lambda values:sum(values)/len(values)
    summary.append(dict(seed=seed,report_sha256=state['report_sha256'],qce={stage:mean(values) for stage,values in record['qce_per_image'].items()},margin={stage:mean(values) for stage,values in record['true_minus_best_other_margin_per_image'].items()},graph_gain={stage:mean(values) for stage,values in record['graph_gain'].items()},metrics={stage:{endpoint:{metric:mean(values) for metric,values in metrics.items()} for endpoint,metrics in endpoints.items()} for stage,endpoints in record['readout_metrics'].items()},interpretation=record['triage_interpretation']))
    paths[f'seed{seed}.json']=path;paths[f'queue{seed}.json']=state_path
packet={}
for name,path in paths.items():
    raw=path.read_bytes();packet[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())
verification=dict(status='all_three_TRAIN8_reports_and_source_bindings_checked',seeds=summary,scope='Completed queue/report hashes, fixed TRAIN8 IDs, no-update scope, CPU-proven fresh/final model hashes, model immutability, and report arithmetic; masks were not independently rerun.')
text=json.dumps(verification,indent=2);packet['summary.json']=dict(sha256=hashlib.sha256(text.encode()).hexdigest(),text=text)
print(json.dumps(packet),flush=True)
