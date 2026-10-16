"""Post-score binary-support diagnosis; no new model forward or tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as evaluate

private=evaluate.ROOT/'collaborative_testing/collaborative_private_runtime_20261008/SW0178_evaluation_delivery_20261016'
audit=json.loads((private/'final_score_verification.json').read_text())
assert audit['status']=='verified'
score_bytes=evaluate.SCORE_PATH.read_bytes()
assert hashlib.sha256(score_bytes).hexdigest()==audit['score_report_sha256']
report=json.loads(score_bytes)
dataset=Path(report['dataset_path'])
assert dataset.resolve()==Path(evaluate.sw130.source97.DATASET).resolve()
before=dataset.stat()
assert before.st_size==report['dataset_size_bytes'] and before.st_mtime_ns==report['dataset_mtime_ns']
with evaluate.h5py.File(dataset,'r') as handle:
    target=evaluate.contract.modal_native32(np.asarray(handle['mask'][1320:1640]))
truth=target>0
refs=evaluate._references()
labels={'SW0164_foreground_induced':refs['SW0164']['labels']}
snapshots={}
for arm in evaluate.ARMS:
    path=evaluate._out(arm)/'predictions.npz'
    snapshots[arm]=evaluate.sha256_file(path)
    assert snapshots[arm]==report['prediction_sha256'][arm]
    with np.load(path,allow_pickle=False) as arrays:
        assert np.array_equal(arrays['image_ids'],evaluate.IMAGE_IDS)
        for endpoint in evaluate.ENDPOINTS:labels[arm+'_'+endpoint]=arrays[endpoint].copy()
        labels[arm+'_native_QCC_support']=arrays['standalone_support'].copy()
groups={}
for name,prediction in labels.items():
    predicted=prediction>0
    tp=(predicted & truth).sum((1,2));fp=(predicted & ~truth).sum((1,2))
    fn=(~predicted & truth).sum((1,2));tn=(~predicted & ~truth).sum((1,2))
    precision=tp/np.maximum(tp+fp,1);recall=tp/np.maximum(tp+fn,1)
    iou=tp/np.maximum(tp+fp+fn,1)
    if name in report['scores']:assert abs(float(iou.mean())-report['scores'][name]['foreground_iou']['mean'])<1e-12
    counts=[];sizes=[]
    for row in prediction:
        ids,number=np.unique(row[row>0],return_counts=True)
        counts.append(len(ids));sizes.extend(number.tolist())
    groups[name]=dict(foreground_iou=float(iou.mean()),precision_mean=float(precision.mean()),recall_mean=float(recall.mean()),TP_total=int(tp.sum()),FP_total=int(fp.sum()),FN_total=int(fn.sum()),TN_total=int(tn.sum()),predicted_foreground_fraction_mean=float(predicted.mean()),true_foreground_fraction_mean=float(truth.mean()),BG_as_FG_rate=float(fp.sum()/max(int((fp+tn).sum()),1)),FG_as_BG_rate=float(fn.sum()/max(int((tp+fn).sum()),1)),label0_GT_object_contamination=float(fn.sum()/max(int((fn+tn).sum()),1)),nonzero_group_count_mean=float(np.mean(counts)),nonzero_group_count_quantiles=np.percentile(counts,[0,25,50,75,100]).tolist(),nonzero_group_size_quantiles=np.percentile(sizes,[0,25,50,75,100]).tolist(),per_image=dict(tp=tp.tolist(),fp=fp.tolist(),fn=fn.tolist(),precision=precision.tolist(),recall=recall.tolist()))
assert evaluate.SCORE_PATH.read_bytes()==score_bytes
assert all(evaluate.sha256_file(evaluate._out(arm)/'predictions.npz')==sha for arm,sha in snapshots.items())
after=dataset.stat();assert (after.st_size,after.st_mtime_ns)==(before.st_size,before.st_mtime_ns)
assert not evaluate.torch.cuda.is_initialized()
result=dict(status='post_score_foreground_support_diagnosis_complete',score_report_sha256=audit['score_report_sha256'],groups=groups,scope='fixed320 IDs1320..1639 after completed independent score audit; reserve unread',model_forward_performed=False,optimizer_updates=0,thresholds_or_BG_policy_changed=False,causal_limit='native_QCC foreground membership changed; cannot separate largest-background-component and discarded-small-group causes from frozen final support alone')
(private/'foreground_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],groups={name:{key:value for key,value in values.items() if key not in ('per_image','nonzero_group_count_quantiles','nonzero_group_size_quantiles')} for name,values in groups.items()})))
