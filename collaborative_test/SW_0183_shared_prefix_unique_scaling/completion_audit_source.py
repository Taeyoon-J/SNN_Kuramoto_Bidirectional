import hashlib
import json
import time
from pathlib import Path
import numpy as np
import h5py
from scipy.optimize import linear_sum_assignment
from collaborative_test.SW_0183_shared_prefix_unique_scaling import score, train

root = Path('/Data0/kevinswk/patch_v2_sw')
private = root / 'collaborative_testing/collaborative_private_runtime_20261008/SW0183_delivery_20261016'
report = json.loads(score.SCORE_PATH.read_text())
assert report['status'] == 'complete'
assert score.sha256_file(score.SCORE_PATH) == 'e49ebeeadfaf88e5a2b33002adb0d787f9c795aa91767b1f5728ad03381d9a11'
p = score.prepare_score_inputs_cpu()
with h5py.File(p['dataset'], 'r') as hdf:
    pixels = np.asarray(hdf['mask'][1320:1640]).reshape(320,128,128)
blocks = pixels.reshape(320,32,4,32,4).transpose(0,1,3,2,4).reshape(320,32,32,16)
gt = np.zeros((320,32,32), dtype=np.int64)
best = np.zeros_like(gt)
for label in np.unique(pixels):
    count = (blocks == label).sum(-1)
    take = count > best
    gt[take] = label
    best[take] = count[take]
assert np.array_equal(gt, score.contract.modal_native32(pixels))

def metrics(pred, target):
    fg = target > 0
    _,ti = np.unique(target[fg],return_inverse=True)
    _,pi = np.unique(pred[fg],return_inverse=True)
    table = np.zeros((ti.max()+1,pi.max()+1),dtype=np.int64)
    np.add.at(table,(ti,pi),1)
    choose2 = lambda x: np.sum(x*(x-1)/2)
    common = choose2(table); pairs_t = choose2(table.sum(1)); pairs_p = choose2(table.sum(0))
    expected = pairs_t*pairs_p/(fg.sum()*(fg.sum()-1)/2)
    denominator = (pairs_t+pairs_p)/2-expected
    ari = float((common-expected)/denominator) if denominator else 1.
    pf = pred > 0
    union = (pf | fg).sum()
    fiou = float((pf & fg).sum()/union) if union else 1.
    tids = np.unique(target[fg]); pids = np.unique(pred[pf])
    ious = np.zeros((len(tids),len(pids)))
    for i, a in enumerate(tids):
        for j,b in enumerate(pids):
            t, q = target == a, pred == b
            ious[i,j] = (t & q).sum()/(t | q).sum()
    rows, cols = linear_sum_assignment(-ious)
    oiou = float(ious[rows,cols].sum()/len(tids))
    return ari, fiou, oiou

errors = []; lineage = []; diagnoses = {}
for seed in score.SEEDS:
    items = p['frozen'][seed]
    histories = {}
    for arm in score.ARMS:
        artifact = train.validate_training_artifact(arm, train._paths(train.OUTPUT_ROOT,arm,seed)[1],seed=seed)
        histories[arm] = artifact['history']
    a,b = [items['prepared']['artifacts'][arm]['manifest'] for arm in score.ARMS]
    for key in ('prefix_state_sha256','prefix_rng_sha256'):
        assert a[key] == b[key]
    assert histories['control512'][:32] == histories['candidate1024'][:32]
    assert histories['control512'][32]['start_transaction_sha256'] == histories['candidate1024'][32]['start_transaction_sha256']
    lineage.append(dict(seed=seed,prefix_state_sha256=a['prefix_state_sha256'],prefix_rng_sha256=a['prefix_rng_sha256'],
        shared_history_equal=True,first_branch_transaction_equal=True,
        accepted={arm:sum(row['accepted'] for row in histories[arm]) for arm in score.ARMS}))
    for arm in score.ARMS:
        for endpoint in score.ENDPOINTS:
            labels = items['results'][arm]['labels'][endpoint]
            independent = np.array([metrics(x,y) for x,y in zip(labels,gt)])
            group = f'{arm}_{endpoint}'
            for j,metric in enumerate(score.METRICS):
                published = np.asarray(report['scores'][group][metric]['per_seed_per_image'][str(seed)])
                error = float(np.max(np.abs(independent[:,j]-published)))
                assert error < 1e-12,(seed,group,metric,error)
                errors.append(error)
            fg = gt > 0; pf = labels > 0
            diagnoses[f'{seed}_{group}'] = dict(
                missed_true_fg_fraction=float(((~pf)&fg).sum()/fg.sum()),
                false_foreground_patch_fraction=float((pf&(~fg)).mean()),
                predicted_foreground_fraction=float(pf.mean()),
                accepted_total=sum(row['accepted'] for row in histories[arm]),
                accepted_branch=sum(row['accepted'] for row in histories[arm][32:]),
                branch_mean_loss_before=float(np.mean([row['before_loss'] for row in histories[arm][32:]])),
            )
sample = np.random.RandomState(178).randint(0,320,size=(10000,320))
for name,item in report['paired_bootstrap'].items():
    left,right = name.split('_minus_',1)
    suffix = next((f'_{ep}_{metric}',ep,metric) for ep in score.ENDPOINTS for metric in score.METRICS if right.endswith(f'_{ep}_{metric}'))
    tail,ep,metric = suffix
    group = right[:-len(tail)]
    delta = np.asarray(report['scores'][f'{left}_{ep}'][metric]['per_image'])-np.asarray(report['scores'][f'{group}_{ep}'][metric]['per_image'])
    ci = np.percentile(delta[sample].mean(1),[2.5,97.5])
    assert abs(float(delta.mean())-item['mean']) < 1e-12
    assert np.max(np.abs(ci-item['ci95'])) < 1e-12
for group,content in report['scores'].items():
    for metric,item in content.items():
        matrix = np.array([item['per_seed_per_image'][str(seed)] for seed in score.SEEDS])
        assert np.max(np.abs(matrix.mean(0)-item['per_image'])) < 1e-12
        assert abs(float(matrix.mean())-item['mean']) < 1e-12
audit = dict(status='SW0183_independent_metrics_modal_GT_fork_lineage_and_bootstrap_verified',
    score_sha256=score.sha256_file(score.SCORE_PATH), observed_unix=time.time(),
    independent_metric_rows=3*2*2*320,maximum_metric_abs_error=max(errors),
    lineage=lineage,diagnoses=diagnoses,paired_bootstrap_checks=len(report['paired_bootstrap']),
    means={g:{m:v['mean'] for m,v in values.items()} for g,values in report['scores'].items()},
    per_seed={g:{m:v['per_seed_mean'] for m,v in values.items()} for g,values in report['scores'].items()},
    gates=report['gates'],paired_bootstrap=report['paired_bootstrap'],incumbent_changed=False)
target=private/'completion_audit.json'
with target.open('x') as stream:json.dump(audit,stream,indent=2,allow_nan=False)
print(json.dumps(dict(status=audit['status'],maximum_metric_abs_error=max(errors),gates=audit['gates'])))
