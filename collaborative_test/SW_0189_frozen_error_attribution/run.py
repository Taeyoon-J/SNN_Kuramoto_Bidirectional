"""Descriptive errors from already frozen masks; no training or new inference."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (0, 1, 2)
IDS = np.arange(1320, 1640, dtype=np.int64)
PINNED = {
    'SW0186': ('SW_0186_full_encoder_qce', 'evaluation_seedmean_20261016.json',
               'bf4e0964d4683687f4e15302eb0ef1e9283ca380aa38b1d2f4061ea1f03daded'),
    'SW0187': ('SW_0187_joint_encoder_unique_scaling', 'evaluation_seedmean_20261017.json',
               '31bcb373a41fe533d0f2218c5861e6c7022996396536c79f9e8279d0a8716b11')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(prediction, target):
    """Pair errors on GT foreground; predicted background remains a cluster.

    Pair rates weight patch pairs. Object-balanced versions weight each true
    object/object-pair equally. Count is raw positive-label count, not a matched
    count classifier. These quantities do not replace the registered metrics.
    """
    p, g = np.asarray(prediction), np.asarray(target)
    if p.shape != g.shape or p.ndim != 2 or p.size == 0:
        raise ValueError('matching nonempty two-dimensional label maps required')
    if any(x.dtype.kind not in 'iu' or np.any(x < 0) for x in (p, g)):
        raise ValueError('nonnegative integer instance labels required')
    p, g = p.ravel(), g.ravel()
    fg, bg = g > 0, g == 0
    retained, support = p > 0, p[fg]
    objects, gi = np.unique(g[fg], return_inverse=True)
    _, pi = np.unique(support, return_inverse=True)
    table = np.zeros((len(objects), len(np.unique(support))), dtype=np.int64)
    np.add.at(table, (gi, pi), 1)
    choose2 = lambda x: np.asarray(x, dtype=np.int64) * (np.asarray(x, dtype=np.int64) - 1) // 2
    sizes = table.sum(1)
    true_pairs = int(choose2(sizes).sum())
    true_positive_pairs = int(choose2(table).sum())
    predicted_pairs = int(choose2(table.sum(0)).sum())
    different_pairs = int(choose2(int(fg.sum()))) - true_pairs
    false_merge_pairs = predicted_pairs - true_positive_pairs
    zero_sizes = np.array([int(np.sum((g == obj) & (p == 0))) for obj in objects], dtype=np.int64)
    zero_merge_pairs = int(choose2(zero_sizes.sum()) - choose2(zero_sizes).sum())
    object_splits = [1. - float(choose2(row).sum()) / int(choose2(size))
                     for row, size in zip(table, sizes) if size >= 2]
    object_merges = [float(table[i] @ table[j]) / int(sizes[i] * sizes[j])
                     for i in range(len(sizes)) for j in range(i + 1, len(sizes))]
    ratio = lambda a, b: float(a / b) if b else None
    positive_count = int(np.unique(p[p > 0]).size)
    return dict(true_object_count=int(len(objects)), predicted_positive_label_count=positive_count,
        raw_count_exact=int(positive_count == len(objects)),
        foreground_miss_rate=ratio(np.sum(fg & ~retained), fg.sum()),
        background_false_positive_rate=ratio(np.sum(bg & retained), bg.sum()),
        foreground_precision=ratio(np.sum(fg & retained), retained.sum()),
        pair_split_rate=ratio(true_pairs - true_positive_pairs, true_pairs),
        pair_merge_rate=ratio(false_merge_pairs, different_pairs),
        pair_merge_via_predicted_background_rate=ratio(zero_merge_pairs, different_pairs),
        object_balanced_split_rate=float(np.mean(object_splits)) if object_splits else None,
        object_pair_balanced_merge_rate=float(np.mean(object_merges)) if object_merges else None)


def load_frozen():
    reports, inputs, groups, references = {}, {}, {}, {}
    for name, (folder, score_name, digest) in PINNED.items():
        path = ROOT/'collaborative_test'/folder/'results_archive'/score_name
        if sha(path) != digest:
            raise ValueError('completed score changed: ' + name)
        report = json.loads(path.read_bytes())
        if report['status'] != 'complete' or report['seed_scope'] != list(SEEDS):
            raise ValueError('completed three-seed score required')
        reports[name] = report
        inputs[str(path)] = digest
        arms = ('candidate',) if name == 'SW0186' else ('control512', 'candidate1024')
        date = '20261016' if name == 'SW0186' else '20261017'
        for arm in arms:
            for endpoint in ('conditional', 'standalone'):
                group = f'{name}_{arm}_{endpoint}'
                groups[group] = {}
                references[group] = report['scores'][f'{arm}_{endpoint}']
            for seed in SEEDS:
                suffix = f'product_q_ce_seed{seed}' if name == 'SW0186' else f'{arm}_seed{seed}'
                folder = ROOT/'collaborative_test'/PINNED[name][0]/f'results_archive/predictions_{date}'/suffix
                archive, protocol = folder/'predictions.npz', folder/'protocol.json'
                hashes = report['candidate_prediction_hashes'][str(seed)]
                if name == 'SW0187':
                    hashes = hashes[arm]
                if sha(archive) != hashes['prediction_sha256'] or sha(protocol) != hashes['protocol_sha256']:
                    raise ValueError('frozen prediction SHA mismatch')
                proto = json.loads(protocol.read_bytes())
                if proto['validation_data_read'] is not False or proto['optimizer_updates'] != 0:
                    raise ValueError('frozen inference must be GT-free with no updates')
                inputs[str(archive)], inputs[str(protocol)] = sha(archive), sha(protocol)
                with np.load(archive, allow_pickle=False) as stored:
                    if not np.array_equal(stored['image_ids'], IDS):
                        raise ValueError('fixed320 order changed')
                    for endpoint in ('conditional', 'standalone'):
                        labels = np.array(stored[endpoint], copy=True)
                        if labels.shape != (320, 32, 32) or labels.dtype.kind not in 'iu' or np.any(labels < 0):
                            raise ValueError('native32 integer labels required')
                        groups[f'{name}_{arm}_{endpoint}'][seed] = labels
    if reports['SW0186']['dataset'] != reports['SW0187']['dataset']:
        raise ValueError('source evaluations used different dataset snapshots')
    return groups, references, inputs, reports['SW0187']['dataset']


def run(output):
    # Only completed, hash-pinned predictions are used; no model is loaded.
    groups, references, inputs, dataset = load_frozen()
    from collaborative_test.SW_0185_qce_spatial_connectivity import completion_audit as independent
    from collaborative_test.SW_0186_full_encoder_qce.completion_audit import load_slot_native32
    for seed in SEEDS:
        slot = load_slot_native32(seed)
        groups.setdefault('Slot70k_standalone', {})[seed] = slot['labels']
        for key in ('archive', 'protocol'):
            inputs[str(slot[key])] = sha(slot[key])
    references['Slot70k_standalone'] = json.loads((ROOT/'collaborative_test'/PINNED['SW0187'][0]/
        'results_archive'/PINNED['SW0187'][1]).read_bytes())['scores']['Slot70k_standalone']
    import h5py
    path = Path(dataset['path'])
    before = path.stat()
    if before.st_size != dataset['size_bytes'] or before.st_mtime_ns != dataset['mtime_ns']:
        raise ValueError('dataset snapshot changed')
    with h5py.File(path, 'r') as hdf:
        gt = independent.modal_labels(np.asarray(hdf['mask'][1320:1640]))
    results, max_error = {}, 0.
    for group, seeds in groups.items():
        rows = {}
        for seed, predictions in seeds.items():
            rows[str(seed)] = []
            for i, (pred, target) in enumerate(zip(predictions, gt)):
                row = dict(image_id=int(IDS[i]), **errors(pred, target))
                values = independent.image_metrics(pred, target)
                for metric, value in zip(independent.METRICS, values):
                    row[metric] = float(value)
                    max_error = max(max_error, abs(value - references[group][metric]['per_seed_per_image'][str(seed)][i]))
                rows[str(seed)].append(row)
        summary = {}
        for key in rows['0'][0]:
            if key == 'image_id': continue
            values = [r[key] for seed_rows in rows.values() for r in seed_rows if r[key] is not None]
            summary[key] = dict(mean=float(np.mean(values)) if values else None, defined_rows=len(values))
        strata = {}
        for count in sorted({r['true_object_count'] for r in rows['0']}):
            selected = [r for seed_rows in rows.values() for r in seed_rows if r['true_object_count'] == count]
            strata[str(count)] = dict(image_count=len(selected)//3,
                **{key:float(np.mean([r[key] for r in selected])) for key in
                   ('fg_ari', 'foreground_iou', 'matched_object_iou', 'raw_count_exact', 'foreground_miss_rate')})
        results[group] = dict(summary=summary, by_true_object_count=strata, per_seed_per_image=rows)
    if max_error > 1e-12:
        raise ValueError('registered scores fail independent recomputation')
    for name, digest in inputs.items():
        if sha(name) != digest: raise ValueError('frozen input changed during analysis')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('dataset changed during analysis')
    report = dict(status='SW0189_completed_frozen_error_attribution', groups=results,
        fixed_image_ids=IDS.tolist(), seeds=list(SEEDS), input_sha256=inputs,
        maximum_registered_metric_abs_error=float(max_error),
        optimizer_updates=0, new_model_forward_passes=0, incumbent_changed=False,
        validation_masks_read_for_posthoc_diagnosis=True, causal_claim=False,
        scope='descriptive fixed320 native32 errors; no replacement metrics or new performance result',
        source_sha256=sha(__file__), observed_unix=time.time())
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'results_archive/analysis_v1/report.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    result = run(args.output)
    print(json.dumps(dict(status=result['status'], maximum_metric_error=result['maximum_registered_metric_abs_error'])))
