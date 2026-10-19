"""CPU-only post-score diagnosis of accepted225 masks; never change predictions."""
import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
if sys.platform == 'linux' and ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
    raise RuntimeError('per-process THP disable failed')
import numpy as np
from .statistics import pair_errors, visible_geometry
from collaborative_test.SW_0225_native64_joint_projection_validation64 import run as reference, evaluate as reader
from collaborative_test.SW_0227_native64_joint_projection_full320 import scope as admission

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'results_archive'
COUNTS = ('true_positive_pairs', 'GT_same_pairs', 'GT_different_pairs', 'false_merge_pairs',
          'false_merge_background_pairs', 'false_merge_foreground_pairs', 'false_split_pairs',
          'false_split_support_pairs', 'false_split_inside_foreground_pairs', 'GT_objects',
          'GT_objects_all_predicted_background')
MEANS = ('false_merge_rate', 'false_split_rate', 'foreground_precision', 'foreground_recall',
         'GT_all_background_object_fraction')


def summarize(rows):
    totals = {key: sum(int(r[key]) for r in rows) for key in COUNTS}
    return dict(rows=len(rows), totals=totals,
        per_image_mean={key: float(np.mean([r[key] for r in rows])) for key in MEANS},
        pair_mass_rates=dict(false_merge=totals['false_merge_pairs'] / max(1, totals['GT_different_pairs']),
                             false_split=totals['false_split_pairs'] / max(1, totals['GT_same_pairs']),
                             background_share_of_merges=totals['false_merge_background_pairs'] / max(1, totals['false_merge_pairs']),
                             support_share_of_splits=totals['false_split_support_pairs'] / max(1, totals['false_split_pairs'])))


def fingerprint():
    return dict(files={n: reference.sha((HERE / n).read_bytes()) for n in ('DESIGN_KO.md', 'statistics.py', 'test_statistics.py', 'run.py')},
                accepted225=admission.selection_fingerprint(), image_ids=list(range(1320, 1384)),
                seeds=[0, 1, 2], grid=[64, 64], GPU_forwards=0, training_updates=0)


def diagnose():
    if reference.torch.cuda.is_initialized():
        raise ValueError('CPU diagnosis requires no initialized CUDA')
    binding = fingerprint()
    bundle = reader.validated_all()
    datasets = [bundle[s]['protocol']['binding']['dataset'] for s in (0, 1, 2)]
    if any(v != datasets[0] for v in datasets):
        raise ValueError('different accepted data')
    protocols = {str(s): reference.sha((reference.archive_for(s) / 'protocol.json').read_bytes()) for s in (0, 1, 2)}
    accepted = json.loads(admission.reference_file('score').read_bytes())
    # All accepted masks/checkpoints/protocols were verified before this GT read.
    gt = reference.source.source.full.read_gt(dict(dataset=datasets[0]))[:64]
    metrics = reference.source.source.full.metrics.independent
    rows = {}; summaries = {}; maximum = 0.; predicted_multipart = {}
    for arm, key in (('baseline', 'baseline'), ('joint_projection', 'labels')):
        vectors = []; per_seed = {}; all_rows = []; nonlocal_clusters = 0; total_clusters = 0
        for seed in (0, 1, 2):
            masks = bundle[seed][key]
            vector = np.stack([metrics.image_metrics(p, t) for p, t in zip(masks, gt)])
            maximum = max(maximum, float(np.max(np.abs(vector.mean(0) - [accepted['per_seed_scores'][f'{arm}_seed{seed}'][k] for k in metrics.METRICS]))))
            vectors.append(vector); current = []
            for offset, (p, t) in enumerate(zip(masks, gt)):
                values = pair_errors(t, p); geometry = visible_geometry(p)
                nonlocal_clusters += geometry['multipart_objects']; total_clusters += geometry['objects']
                current.append(values); rows[f'{arm}_seed{seed}_image{1320+offset}'] = values
            per_seed[str(seed)] = summarize(current); all_rows.extend(current)
        mean = np.stack(vectors).mean((0, 1))
        maximum = max(maximum, float(np.max(np.abs(mean - [accepted['scores'][arm][k] for k in metrics.METRICS]))))
        summaries[arm] = dict(summarize(all_rows), per_seed=per_seed, accepted225_metrics_reproduced=dict(zip(metrics.METRICS, mean.tolist())))
        predicted_multipart[arm] = dict(positive_clusters=total_clusters, positive_clusters_with_multiple_four_connected_parts=nonlocal_clusters)
    if maximum > 1e-12:
        raise ValueError('diagnosis does not reproduce accepted225 evaluation')
    geometry = [visible_geometry(t) for t in gt]
    totals = {key: sum(int(r[key]) for r in geometry) for key in geometry[0]}
    summary = dict(status='complete_CPU_accepted225_partition_error_diagnosis', fingerprint=binding,
        accepted_protocol_sha256=protocols, unique_images=64, seeds=[0, 1, 2], prediction_rows_per_arm=192,
        arms=summaries, GT_visible_geometry=dict(totals=totals,
            multipart_object_fraction=totals['multipart_objects'] / max(1, totals['objects']),
            minor_component_foreground_mass_fraction=totals['minor_component_patches'] / max(1, totals['foreground_patches'])),
        predicted_positive_connectivity=predicted_multipart, accepted_metric_maximum_abs_error=maximum,
        all_three_accepted_protocols_checked_before_GT=True, analysis_only_not_model_result=True,
        GPU_forwards=0, training_updates=0, no_model_loss_mask_or_dataset_changed=True,
        not_an_ARI_upper_bound=True, causal_gating_claim=False, full320_incumbent_unchanged=True)
    reference.write_json(ARCHIVE / 'diagnosis_v1/private_rows.json', dict(summary=summary, rows=rows))
    reference.write_json(ARCHIVE / 'diagnosis_v1/summary.json', summary)
    return summary


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = diagnose()
    print(json.dumps(dict(status=summary['status'], arms={k: v['per_image_mean'] for k, v in summary['arms'].items()},
        pair_mass_rates={k: v['pair_mass_rates'] for k, v in summary['arms'].items()},
        GT_visible_geometry=summary['GT_visible_geometry'], predicted_connectivity=summary['predicted_positive_connectivity'],
        accepted_metric_maximum_abs_error=summary['accepted_metric_maximum_abs_error'])))
