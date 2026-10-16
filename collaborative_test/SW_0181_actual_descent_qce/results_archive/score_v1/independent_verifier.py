"""Independent post-completion metric/hash audit for SW0181; never trains."""
import hashlib,json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['USE_PYTORCH_KERNEL_CACHE']='0'
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
import h5py
from collaborative_test.SW_0181_actual_descent_qce import score,score_queue
from collaborative_test.SW_0165_foreground_partition_replication import predict_evaluate as sw165,replication_summary as sw165_summary
from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as sw164
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
SEEDS=(0,1,2)
ENDPOINTS=('conditional','standalone')
METRICS=tuple(contract.METRICS)
COUNT=320
BOOTSTRAP_SEED=178
BOOTSTRAP_REPLICATES=10000
SW164_SEED1_SCORE_SHA256='978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6'
SW165_SUMMARY_SHA256='72b17ca1a07d312400ab74f6ee7533c89f0e9d9a6c304378dd8e61c16ba379c5'
SW165_FG_MEAN=.8042989149474198
SLOT_FG_MEAN=.7816519719609135
SLOT_FG_IOU_MEAN=.20436012776655546
SLOT_OBJECT_IOU_MEAN=.2031649861884501

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _finite320(value, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (COUNT,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be 320 finite values")
    return array

def _assert_json_equal(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{label} differs from independent recomputation")

def _bootstrap(candidate, baseline, indices):
    delta = _finite320(candidate, "candidate vector") - _finite320(baseline, "baseline vector")
    samples = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()),
        "ci95": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "paired_image_after_three_seed_average"}

def _compare_metric_row(row, values, group, metric):
    if not isinstance(row, dict):
        raise ValueError(f"{group}/{metric} score row is invalid")
    reported = _finite320(row.get("per_image"), f"{group}/{metric}.per_image")
    if not np.array_equal(reported, values):
        raise ValueError(f"{group}/{metric} per-image vector mismatch")
    mean = float(values.mean())
    if not np.isfinite(row.get("mean", np.nan)) or float(row["mean"]) != mean:
        raise ValueError(f"{group}/{metric} mean mismatch")
    return mean

def _load_baseline_vectors():
    summary_path = Path(sw165_summary.OUTPUT)
    if _sha(summary_path) != SW165_SUMMARY_SHA256:
        raise ValueError("pinned SW0165 three-seed summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("experiment") != "SW0165_foreground_partition_replication":
        raise ValueError("pinned SW0165 summary schema is invalid")
    paths = {0: Path(sw165.score_path(0)), 1: Path(sw164.SCORE_PATH), 2: Path(sw165.score_path(2))}
    summary_seed_hashes = summary.get("seed_score_sha256", {})
    if summary_seed_hashes.get("1") != SW164_SEED1_SCORE_SHA256:
        raise ValueError("SW0165 summary does not bind the pinned SW0164 seed-1 report")
    vectors = {"SW0165_foreground_induced": {}, "Slot70k": {}}
    report_hashes = {}
    for seed in SEEDS:
        path = paths[seed]
        expected_sha = SW164_SEED1_SCORE_SHA256 if seed == 1 else summary_seed_hashes.get(str(seed))
        if not expected_sha or _sha(path) != expected_sha:
            raise ValueError(f"pinned SW0165/SW0164 score seed{seed} changed")
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if (report.get("status") != "complete" or report.get("seed_scope") != [seed]
                or report.get("ground_truth_used_for_prediction") is not False
                or report.get("ground_truth_used_for_scoring") is not True):
            raise ValueError(f"baseline seed{seed} score scope is invalid")
        report_hashes[str(seed)] = expected_sha
        for group, raw_group in (("SW0165_foreground_induced", "foreground_induced"), ("Slot70k", "Slot70k")):
            vectors[group][seed] = {}
            for metric in METRICS:
                row = report.get("scores", {}).get(raw_group, {}).get(metric, {})
                vector = _finite320(row.get("per_image"), f"baseline seed{seed}/{raw_group}/{metric}")
                if row.get("valid_count") != COUNT or abs(float(row.get("mean", np.nan)) - float(vector.mean())) > 1e-15:
                    raise ValueError(f"baseline seed{seed}/{raw_group}/{metric} summary mismatch")
                vectors[group][seed][metric] = vector
    aggregate = summary.get("three_seed_aggregate", {})
    for group, summary_prefix in (("SW0165_foreground_induced", "candidate"), ("Slot70k", "slot")):
        for metric in METRICS:
            vector = np.stack([vectors[group][seed][metric] for seed in SEEDS]).mean(axis=0)
            expected = aggregate.get(metric, {}).get(f"{summary_prefix}_mean_across_three_seeds")
            if expected is None or abs(float(vector.mean()) - float(expected)) > 1e-15:
                raise ValueError(f"baseline summary aggregation mismatch for {group}/{metric}")
            if group == "SW0165_foreground_induced" and metric == "fg_ari" and abs(float(vector.mean()) - SW165_FG_MEAN) > 1e-15:
                raise ValueError("SW0165 three-seed FG-ARI baseline mismatch")
            if group == "Slot70k":
                expected_slot = {"fg_ari": SLOT_FG_MEAN,
                    "foreground_iou": SLOT_FG_IOU_MEAN,
                    "matched_object_iou": SLOT_OBJECT_IOU_MEAN}.get(metric)
                if expected_slot is not None and abs(float(vector.mean()) - expected_slot) > 1e-15:
                    raise ValueError(f"Slot three-seed {metric} baseline mismatch")
    return vectors, report_hashes, _sha(summary_path)

def _derive_gate(scores, paired):
    candidate = scores["candidate_standalone"]
    sw = scores["SW0165_foreground_induced_standalone"]
    slot = scores["Slot70k_standalone"]
    checks = {
        "fg_strictly_above_sw0165_three_seed_mean": candidate["fg_ari"]["mean"] > sw["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_sw0165": paired["standalone_candidate_minus_SW0165_foreground_induced"]["fg_ari"]["ci95"][0] > 0,
        "fg_strictly_above_slot_three_seed_mean": candidate["fg_ari"]["mean"] > slot["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_slot": paired["standalone_candidate_minus_Slot70k"]["fg_ari"]["ci95"][0] > 0,
        "foreground_iou_strictly_above_slot_three_seed_mean": candidate["foreground_iou"]["mean"] > slot["foreground_iou"]["mean"],
        "matched_object_iou_strictly_above_slot_three_seed_mean": candidate["matched_object_iou"]["mean"] > slot["matched_object_iou"]["mean"],
    }
    return {"primary_endpoint": "standalone", **checks,
        "fg_ari_at_least_0_87_goal": candidate["fg_ari"]["mean"] >= 0.87,
        "primary_gate_passed": all(checks.values()),
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False}

def verify():
    assert not torch.cuda.is_initialized()
    queue_path=score_queue._state_path()
    queue=json.loads(queue_path.read_text())
    assert queue['experiment']=='SW0181_score_queue' and queue['status']=='complete'
    path=score.SCORE_PATH
    report=json.loads(path.read_text())
    assert queue['output']==str(path) and queue['score_sha256']==_sha(path)
    prepared=score.prepare_score_inputs_cpu()
    assert prepared['ground_truth_opened'] is False
    fingerprint=score.implementation_fingerprint()
    assert report['implementation_fingerprint']==fingerprint
    assert report['experiment']=='SW0181_actual_descent_qce_three_seed_score'
    assert report['status']=='complete' and report['seed_scope']==list(SEEDS)
    assert report['endpoint_scope']==list(ENDPOINTS) and report['metric_scope']==list(METRICS)
    assert report['ground_truth_used_for_prediction'] is False
    assert report['ground_truth_used_for_scoring'] is True
    assert report['mask_supervised'] is True and report['unsupervised_leaderboard_eligible'] is False
    assert report['ground_truth_scope']=='fixed320 canonical validation modal-native32 masks only'
    assert report['bootstrap']==dict(rng='numpy RandomState',seed=178,replicates=10000,
        resampling_unit='320 paired images after candidate three-seed average')
    baseline,baseline_hashes,summary_sha=_load_baseline_vectors()
    assert report['baseline_summary_sha256']==summary_sha
    assert report['baseline_score_sha256']==baseline_hashes
    tracked={str(path):_sha(path),str(queue_path):_sha(queue_path)}
    for seed in SEEDS:
        item=prepared['frozen'][seed]
        result=item['result']
        expected=dict(prediction_sha256=result['prediction_sha256'],
            protocol_sha256=result['protocol_sha256'],queue_state_sha256=item['state_sha256'])
        assert report['candidate_prediction_hashes'][str(seed)]==expected
        assert report['training_manifest_sha256'][str(seed)]==item['prepared']['manifest_sha256']
        assert queue['prediction_state_sha256'][str(seed)]==item['state_sha256']
        artifact=item['prepared']['artifact']
        for file in (item['state_path'],result['directory']/'predictions.npz',
                     result['directory']/'protocol.json',artifact/'manifest.json',
                     artifact/'checkpoint.pt',artifact/'optimizer.pt'):
            tracked[str(file)]=_sha(file)
        assert tracked[str(result['directory']/'predictions.npz')]==result['prediction_sha256']
        assert tracked[str(result['directory']/'protocol.json')]==result['protocol_sha256']
    dataset=Path(prepared['dataset'])
    metadata=dataset.stat()
    assert report['dataset']==dict(path=str(dataset.resolve()),size_bytes=metadata.st_size,mtime_ns=metadata.st_mtime_ns)
    assert np.array_equal(score.IMAGE_IDS,np.arange(1320,1640))
    with h5py.File(dataset,'r') as hdf:
        gt=contract.modal_native32(np.asarray(hdf['mask'][1320:1640]))
    recomputed={}
    for endpoint in ENDPOINTS:
        group='candidate_'+endpoint
        rows={seed:contract.production_metrics32(prepared['frozen'][seed]['result']['labels'][endpoint],gt)
              for seed in SEEDS}
        recomputed[group]={}
        for metric in METRICS:
            matrix=np.stack([_finite320(rows[seed][metric]['per_image'],group+'/'+metric) for seed in SEEDS])
            published=report['scores'][group][metric]
            assert set(published['per_seed_per_image'])==set(map(str,SEEDS))
            for seed in SEEDS:
                assert np.array_equal(np.asarray(published['per_seed_per_image'][str(seed)]),matrix[seed])
                assert abs(published['per_seed_mean'][str(seed)]-float(rows[seed][metric]['mean']))<1e-15
            average=matrix.mean(axis=0)
            mean=_compare_metric_row(published,average,group,metric)
            recomputed[group][metric]=dict(per_image=average,mean=mean)
    for control,vectors in baseline.items():
        for endpoint in ENDPOINTS:
            group=control+'_'+endpoint
            recomputed[group]={}
            for metric in METRICS:
                matrix=np.stack([vectors[seed][metric] for seed in SEEDS])
                published=report['scores'][group][metric]
                for seed in SEEDS:
                    assert np.array_equal(np.asarray(published['per_seed_per_image'][str(seed)]),matrix[seed])
                    assert abs(published['per_seed_mean'][str(seed)]-float(matrix[seed].mean()))<1e-15
                average=matrix.mean(axis=0)
                mean=_compare_metric_row(published,average,group,metric)
                recomputed[group][metric]=dict(per_image=average,mean=mean)
    indices=np.random.RandomState(178).randint(0,COUNT,size=(10000,COUNT))
    paired={}
    for endpoint in ENDPOINTS:
        for control in ('SW0165_foreground_induced','Slot70k'):
            key=endpoint+'_candidate_minus_'+control
            paired[key]={metric:_bootstrap(recomputed['candidate_'+endpoint][metric]['per_image'],
                recomputed[control+'_'+endpoint][metric]['per_image'],indices) for metric in METRICS}
    _assert_json_equal(report['paired_bootstrap'],paired,'paired bootstrap')
    _assert_json_equal(report['gate'],_derive_gate(recomputed,paired),'registered gate')
    assert score.implementation_fingerprint()==fingerprint
    _load_baseline_vectors()
    for file,sha in tracked.items():assert _sha(file)==sha,'artifact changed during audit: '+file
    after=dataset.stat()
    assert (after.st_size,after.st_mtime_ns)==(metadata.st_size,metadata.st_mtime_ns)
    assert not torch.cuda.is_initialized()
    return dict(status='SW0181_independent_fixed320_score_audit_passed',score_sha256=_sha(path),
        metric_vectors_recomputed=True,all_three_seeds=True,both_endpoints=True,
        paired_bootstrap_recomputed=True,gate_recomputed=True,artifact_hashes_unchanged=True,
        ground_truth_scope='validation IDs1320..1639 only; no reserve data',training_updates=0,
        metrics={group:{metric:row['mean'] for metric,row in rows.items()} for group,rows in recomputed.items()},
        gate=report['gate'],observed_unix=time.time())

if __name__=='__main__':
    result=verify()
    output=Path('/Data0/kevinswk/patch_v2_sw/collaborative_testing/collaborative_private_runtime_20261008/SW0181_delivery_20261016/independent_score_audit.json')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
