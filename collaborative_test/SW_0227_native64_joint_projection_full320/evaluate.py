"""Exact full320 comparison with the incumbent and comparable Slot reference."""
import argparse
import json
import numpy as np
from . import run

METRICS = run.source.source.full.metrics.independent.METRICS


def validated_all():
    return {s: run.validated_seed(s) for s in run.SEEDS}


def load_slot_masks():
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    rows = run.source.source.full.metrics.slot.prepare_all_slot_predictions(SLOT_ROOT)
    if set(rows) != set(run.SEEDS):
        raise ValueError('three comparable Slot seeds required')
    for row in rows.values():
        if not np.array_equal(row['ids'], np.arange(1320, 1640)) or row['labels64'].shape != (320, 64, 64):
            raise ValueError('different Slot image order or native64 scope')
    return rows


def reference_scores():
    source = run.source.source
    path = source.ARCHIVE / 'score_v1/summary.json'
    reference = json.loads(path.read_bytes())
    if reference['fingerprint'] != source.fingerprint() or reference['image_count'] != 320:
        raise ValueError('full320 incumbent reference changed')
    return reference['scores']


def verify_reference_metrics(groups, accepted):
    maximum = 0.
    for arm, reference_name in (('baseline', 'our_three_seed_mean'), ('slot', 'slot_three_seed_mean')):
        error = float(np.max(np.abs(groups[arm].mean((0, 1)) - [accepted[reference_name][k] for k in METRICS])))
        maximum = max(maximum, error)
    if maximum > 1e-12:
        raise ValueError('incumbent or Slot metric contract changed')
    return maximum


def score():
    bundle = validated_all()
    datasets = [bundle[s]['protocol']['binding']['dataset'] for s in run.SEEDS]
    hashes = [bundle[s]['protocol']['binding']['RGB_sha256'] for s in run.SEEDS]
    if any(v != datasets[0] for v in datasets) or len(set(hashes)) != 1:
        raise ValueError('different full320 data')
    slots = load_slot_masks()
    accepted = reference_scores()
    # Every model's full prediction protocol and all Slot artifacts precede GT.
    gt = run.source.source.full.read_gt(dict(dataset=datasets[0]))
    vectors = {}; maximum = 0.
    for seed in run.SEEDS:
        for arm, labels in (('baseline', bundle[seed]['baseline']),
                            ('joint_projection', bundle[seed]['labels']),
                            ('slot', slots[seed]['labels64'])):
            vectors[f'{arm}_seed{seed}'], error = run.source.source.full.metrics.measured_vectors(labels, gt)
            maximum = max(maximum, error)
    groups = {arm: np.stack([vectors[f'{arm}_seed{s}'] for s in run.SEEDS])
              for arm in ('baseline', 'joint_projection', 'slot')}
    if any(v.shape != (3, 320, 3) or not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('finite three-seed full320 vectors required')
    reference_error = verify_reference_metrics(groups, accepted)
    draw = np.random.RandomState(227).randint(0, 320, (10000, 320))
    paired = {}
    for arm in ('baseline', 'slot'):
        delta = (groups['joint_projection'] - groups[arm]).mean(0)
        ci = np.percentile(delta[draw].mean(1), [2.5, 97.5], axis=0)
        paired[arm] = {k: dict(mean=float(delta[:, i].mean()), ci95=ci[:, i].tolist()) for i, k in enumerate(METRICS)}
    strict = bool(np.all(groups['joint_projection'].mean((0, 1)) > groups['slot'].mean((0, 1))))
    summary = dict(status='complete_fixed223_native64_full320_confirmation',
        scores={arm: dict(zip(METRICS, v.mean((0, 1)).tolist())) for arm, v in groups.items()},
        per_seed_scores={k: dict(zip(METRICS, v.mean(0).tolist())) for k, v in vectors.items()},
        paired_joint_minus=paired, three_seed_strictly_exceeds_Slot_means=strict,
        native64_FG090_three_seed_goal_verified=bool(strict and groups['joint_projection'][:, :, 0].mean() >= .90),
        image_count=320, grid=[64, 64], seed_scope=list(run.SEEDS), fixed_checkpoint_experiment='SW0223',
        selection_experiment='SW0225', original16_gate_failure_preserved=True,
        reused_reference_images_per_seed=64, new_prediction_images_per_seed=256,
        replay_images_per_seed=1, optimizer_updates=0, new70000_training=False,
        mask_supervised_lineage=True, unsupervised_leaderboard_eligible=False,
        conditional_image_bootstrap_not_seed_population_CI=True, validation_selected_not_untouched_test=True,
        all_three_full_protocols_and_Slot_predictions_validated_before_DEV_GT=True,
        independent_metric_maximum_abs_error=maximum,
        unchanged_reference_metrics_maximum_abs_error=reference_error,
        Slot_reference={str(s): dict(prediction_sha256=slots[s]['prediction_sha256'],
            protocol_sha256=slots[s]['protocol_sha256'], labels64_sha256=run.sha(slots[s]['labels64'].tobytes())) for s in run.SEEDS},
        fingerprint=run.fingerprint())
    run.write_json(run.ARCHIVE / 'score_v1/evaluation_private.json', dict(summary=summary, vectors={k: v.tolist() for k, v in vectors.items()}))
    run.write_json(run.ARCHIVE / 'score_v1/summary.json', summary)
    return summary


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(score()))
