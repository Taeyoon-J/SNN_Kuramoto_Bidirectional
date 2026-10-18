"""Rescore all frozen masks independently; no physical model or training rerun."""
import json
from pathlib import Path
import numpy as np
from . import run, evaluate


def audit():
    a = run.ARCHIVE
    raw = (a / 'score_v1/summary.json').read_bytes()
    score = json.loads(raw)
    private = json.loads((a / 'score_v1/evaluation_private.json').read_bytes())
    if score['fingerprint'] != run.fingerprint() or private['summary'] != score:
        raise ValueError('frozen score changed')
    bundle = evaluate.validated_all()
    datasets = [bundle[s]['protocol']['binding']['dataset'] for s in run.SEEDS]
    if any(d != datasets[0] for d in datasets):
        raise ValueError('different evaluation data')
    # Every protocol and all checkpoint byte hashes are checked before reading GT.
    gt = run.source.source.full.read_gt(dict(dataset=datasets[0]))[:64]
    keys = run.source.source.full.metrics.independent.METRICS
    vectors = {}
    maximum = 0.
    references = {}
    for seed in run.SEEDS:
        folder = run.archive_for(seed)
        state = json.loads((folder / 'queue/state.json').read_bytes())
        if state['status'] != 'complete':
            raise ValueError('queue not complete')
        protocol = bundle[seed]['protocol']
        references[str(seed)] = dict(
            candidate=protocol['binding']['candidate'],
            protocol_sha256=run.sha((folder / 'protocol.json').read_bytes()),
            replay=json.loads((folder / 'replay.json').read_bytes()))
        for arm, key in (('baseline', 'baseline'), ('joint_projection', 'labels')):
            name = f'{arm}_seed{seed}'
            value = np.stack([run.source.source.full.metrics.independent.image_metrics(p, t)
                              for p, t in zip(bundle[seed][key], gt)])
            if value.shape != (64, 3):
                raise ValueError('wrong metric scope')
            vectors[name] = value
            maximum = max(maximum, float(np.max(np.abs(value - np.asarray(private['vectors'][name])))),
                          float(np.max(np.abs(value.mean(0) - [score['per_seed_scores'][name][k] for k in keys]))))
    groups = {arm: np.stack([vectors[f'{arm}_seed{s}'] for s in run.SEEDS])
              for arm in ('baseline', 'joint_projection')}
    for arm, value in groups.items():
        maximum = max(maximum, float(np.max(np.abs(value.mean((0, 1)) - [score['scores'][arm][k] for k in keys]))))
    delta = (groups['joint_projection'] - groups['baseline']).mean(0)
    draw = np.random.RandomState(225).randint(0, 64, (10000, 64))
    ci = np.percentile(delta[draw].mean(1), [2.5, 97.5], axis=0)
    for i, key in enumerate(keys):
        expected = score['paired_joint_minus_baseline'][key]
        maximum = max(maximum, abs(expected['mean'] - float(delta[:, i].mean())),
                      float(np.max(np.abs(ci[:, i] - expected['ci95']))))
    gate = bool(delta[:, 0].mean() >= .01 and ci[0, 0] > 0)
    balanced = bool(gate and all(delta[:, i].mean() >= 0 for i in (1, 2)))
    original = json.loads((run.scope.model.ARCHIVE / 'score_v1/summary.json').read_bytes())
    if original['full320_candidate_gate'] or not score['source16_gate_failure_preserved']:
        raise ValueError('original failed16 gate changed')
    if maximum > 1e-12 or gate != score['full320_candidate_gate'] or balanced != score['balanced_candidate_gate']:
        raise ValueError('independent metrics/CI/gates mismatch')
    result = dict(status='complete_fixed223_secondary64_independent_audit',
                  score_sha256=run.sha(raw), diagnostic_source_sha256=run.sha(Path(__file__).read_bytes()),
                  candidate_references=references, raw_masks_independently_rescored=384,
                  independent_metrics_means_and_bootstrap_maximum_abs_error=maximum,
                  original16_gate_failure_preserved=True, FG_track_candidate_gate=gate,
                  balanced_candidate_gate=balanced, additional_model_forwards=0,
                  additional_training_updates=0, full320_incumbent_unchanged=True)
    run.write_json(a / 'completion_audit_v1/summary.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit()))
