"""Independently rescore frozen full320 masks; no model or training reruns."""
import argparse
import ctypes
import json
import sys
from pathlib import Path
if sys.platform == 'linux':
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise RuntimeError('per-process THP disable failed')
import numpy as np
from . import run, evaluate


def finite_difference(left, right):
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('finite equally shaped audit values required')
    return float(np.max(np.abs(left - right)))


def check_statistics(vectors, score, accepted):
    keys = evaluate.METRICS
    groups = {arm: np.stack([vectors[f'{arm}_seed{s}'] for s in run.SEEDS])
              for arm in ('baseline', 'joint_projection', 'slot')}
    if any(v.shape != (3, 320, 3) or not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('full320 three-model finite vectors required')
    maximum = 0.
    for name, value in vectors.items():
        maximum = max(maximum, finite_difference(value.mean(0), [score['per_seed_scores'][name][k] for k in keys]))
    for name, value in groups.items():
        maximum = max(maximum, finite_difference(value.mean((0, 1)), [score['scores'][name][k] for k in keys]))
    for arm, name in (('baseline', 'our_three_seed_mean'), ('slot', 'slot_three_seed_mean')):
        maximum = max(maximum, finite_difference(groups[arm].mean((0, 1)), [accepted[name][k] for k in keys]))
    draw = np.random.RandomState(237).randint(0, 320, (10000, 320))
    for arm in ('baseline', 'slot'):
        delta = (groups['joint_projection'] - groups[arm]).mean(0)
        ci = np.percentile(delta[draw].mean(1), [2.5, 97.5], axis=0)
        for i, key in enumerate(keys):
            expected = score['paired_joint_minus'][arm][key]
            maximum = max(maximum, finite_difference(expected['mean'], float(delta[:, i].mean())),
                          finite_difference(ci[:, i], expected['ci95']))
    strict = bool(np.all(groups['joint_projection'].mean((0, 1)) > groups['slot'].mean((0, 1))))
    goal = bool(strict and groups['joint_projection'][:, :, 0].mean() >= .90)
    if (maximum > 1e-12 or strict != score['three_seed_strictly_exceeds_Slot_means']
            or goal != score['native64_FG090_three_seed_goal_verified']):
        raise ValueError('independent full320 metrics/CI/goal mismatch')
    return maximum, strict, goal


def audit():
    a = run.ARCHIVE
    raw = (a / 'score_v1/summary.json').read_bytes(); score = json.loads(raw)
    private = json.loads((a / 'score_v1/evaluation_private.json').read_bytes())
    if score['fingerprint'] != run.fingerprint() or private['summary'] != score:
        raise ValueError('frozen full320 score changed')
    bundle = evaluate.validated_all(); slots = evaluate.load_slot_masks(); accepted = evaluate.reference_scores()
    if evaluate.current_incumbent_reference()!=score['current_full320_incumbent']:
        raise ValueError('current full320 incumbent evidence changed')
    datasets = [bundle[s]['protocol']['binding']['dataset'] for s in run.SEEDS]
    hashes = [bundle[s]['protocol']['binding']['RGB_sha256'] for s in run.SEEDS]
    if any(v != datasets[0] for v in datasets) or len(set(hashes)) != 1:
        raise ValueError('different full320 data')
    references = {}
    for seed in run.SEEDS:
        folder = run.archive_for(seed); protocol = bundle[seed]['protocol']
        state = json.loads((folder / 'queue/state.json').read_bytes())
        replay = json.loads((folder / 'replay.json').read_bytes())
        if (state['status'] != 'complete' or replay['binding'] != protocol['binding']
                or not replay['mask_bitwise_equal'] or not replay['frequency_bitwise_equal']):
            raise ValueError('full320 queue or exact replay incomplete')
        current = dict(prediction_sha256=slots[seed]['prediction_sha256'], protocol_sha256=slots[seed]['protocol_sha256'],
                       labels64_sha256=run.sha(slots[seed]['labels64'].tobytes()))
        if current != score['Slot_reference'][str(seed)]:
            raise ValueError('comparable Slot artifacts changed')
        references[str(seed)] = dict(candidate=protocol['binding']['candidate'],
            protocol_sha256=run.sha((folder / 'protocol.json').read_bytes()), Slot=current)
    # Every full prediction/cache/checkpoint/Slot protocol is verified before GT.
    gt = run.source.source.full.read_gt(dict(dataset=datasets[0])); vectors = {}; maximum = 0.
    for seed in run.SEEDS:
        for arm, labels in (('baseline', bundle[seed]['baseline']), ('joint_projection', bundle[seed]['labels']), ('slot', slots[seed]['labels64'])):
            name = f'{arm}_seed{seed}'
            value = np.stack([run.source.source.full.metrics.independent.image_metrics(p, t) for p, t in zip(labels, gt)])
            vectors[name] = value
            maximum = max(maximum, finite_difference(value, private['vectors'][name]))
    error, strict, goal = check_statistics(vectors, score, accepted); maximum = max(maximum, error)
    for i,key in enumerate(evaluate.METRICS):
        observed=np.stack([vectors[f'joint_projection_seed{s}'] for s in run.SEEDS])[:,:,i].mean()
        expected=score['current_full320_incumbent']['scores'][key]
        maximum=max(maximum,finite_difference(score['mean_candidate_minus_current_incumbent'][key],observed-expected))
    if maximum > 1e-12:
        raise ValueError('raw mask metrics disagree with saved vectors')
    result = dict(status='complete_independently_audited_fixed232_native64_full320',
        score_sha256=run.sha(raw), diagnostic_source_sha256=run.sha(Path(__file__).read_bytes()),
        candidate_references=references, raw_masks_independently_rescored=2880,
        independent_metrics_means_and_bootstrap_maximum_abs_error=maximum,
        three_seed_strictly_exceeds_Slot_means=strict, native64_FG090_three_seed_goal_verified=goal,
        original16_gate_failure_preserved=True, accepted64_selection=run.scope.selection_fingerprint(),
        all_three_full_protocols_and_Slot_artifacts_checked_before_DEV_GT=True,
        additional_model_forwards=0, additional_training_updates=0, raw_imagewise_vectors_published=False)
    run.write_json(a / 'completion_audit_v1/summary.json', result)
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(audit()))
