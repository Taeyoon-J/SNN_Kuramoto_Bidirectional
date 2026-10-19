"""Independently inspect complete TRAIN coverage, actual checkpoints and raw masks."""
import io
import json
from pathlib import Path
import numpy as np
import torch
from . import train, evaluate, data


def difference(left, right):
    a = np.asarray(left, dtype=np.float64); b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite matching audit geometry required')
    return float(np.max(np.abs(a-b)))


def audit_history(history, seed, *, count=70000, batch=16, epochs=10, registered_ids=None):
    if count % batch or len(history) != epochs * count // batch:
        raise ValueError('incomplete registered full-data update history')
    ids = data.train_ids() if registered_ids is None else np.asarray(registered_ids)
    if ids.shape != (count,) or len(np.unique(ids)) != count or not np.array_equal(np.sort(ids), ids):
        raise ValueError('unique sorted registered TRAIN IDs required')
    per_epoch = count // batch
    for epoch in range(epochs):
        rows = history[epoch*per_epoch:(epoch+1)*per_epoch]
        values = np.asarray([r['TRAIN_ids'] for r in rows])
        if values.shape != (per_epoch, batch) or values.dtype.kind not in 'iu':
            raise ValueError('actual complete integer batch IDs required')
        flat = values.reshape(-1)
        if not np.array_equal(np.sort(flat), ids):
            raise ValueError('epoch repeats or omits registered TRAIN images')
        expected = ids[np.random.RandomState(239000 + seed + 10000*epoch).permutation(count)]
        if not np.array_equal(flat, expected):
            raise ValueError('actual seed/epoch shuffle differs')
        for offset, row in enumerate(rows):
            if row['update'] != epoch*per_epoch+offset+1 or not all(np.isfinite(row[k]) for k in ('loss','frequency_denoising','gamma_anchor','gradient_norm')):
                raise ValueError('nonfinite loss or invalid actual Adam update boundary')
    return dict(unique_TRAIN_images=count, complete_passes=epochs,
        logical_exposures=count*epochs, independently_verified_actual_updates=len(history),
        every_epoch_exactly_covers_registered_split_once=True)


def check_statistics(vectors, score, selection):
    keys = train.source.source.full.metrics.independent.METRICS
    groups = {name: np.stack([vectors[f'{name}_seed{s}'] for s in train.SEEDS]) for name in ('ours','slot')}
    if any(v.shape != (3,320,3) or not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('complete finite seed/image/metric vectors required')
    maximum = 0.
    for name, value in vectors.items():
        maximum = max(maximum, difference(value.mean(0), [score['per_seed_scores'][name][k] for k in keys]))
    for name, value in groups.items():
        maximum = max(maximum, difference(value.mean((0,1)), [score['scores'][name][k] for k in keys]))
    maximum = max(maximum, difference(groups['slot'].mean((0,1)), [selection['selected']['slot'][k] for k in keys]))
    delta = (groups['ours']-groups['slot']).mean(0)
    draw = np.random.RandomState(239).randint(0,320,(10000,320))
    ci = np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
    for i,key in enumerate(keys):
        item = score['paired_ours_minus_slot'][key]
        maximum = max(maximum,difference(delta[:,i].mean(),item['mean']),difference(ci[:,i],item['ci95']))
    strict = bool(np.all(groups['ours'].mean((0,1)) > groups['slot'].mean((0,1))))
    goal = bool(strict and groups['ours'][:,:,0].mean() >= .9)
    if maximum >= 1e-12 or strict != score['three_seed_strictly_exceeds_Slot_means'] or goal != score['native64_FG090_three_seed_goal_verified']:
        raise ValueError('independent means, Slot identity, CI or target verification differs')
    return maximum, strict, goal


def audit_checkpoint(seed, report):
    folder = train.archive_for(seed); raw = (folder/'encoder_work/boundary.ptx').read_bytes()
    if train.sha(raw) != report['checkpoint_sha256']:
        raise ValueError('actual final checkpoint changed')
    outer = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    encoded = outer['payload_bytes']
    if train.sha(encoded) != outer['payload_sha256']:
        raise ValueError('actual checkpoint envelope is corrupt')
    payload = torch.load(io.BytesIO(encoded),map_location='cpu',weights_only=False)
    if payload['completed_updates'] != data.TOTAL_STEPS or payload['bindings'] != report['checkpoint_binding'] or payload['runtime'] != report['runtime']:
        raise ValueError('actual final checkpoint update/binding/runtime differs')
    proof = audit_history(payload['history'],seed)
    if train.sha(json.dumps(payload['history'],sort_keys=True,allow_nan=False).encode()) != report['history_sha256']:
        raise ValueError('actual history differs from training report')
    steps = sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})
    if steps != [data.TOTAL_STEPS] or steps != report['actual_Adam_steps']:
        raise ValueError('actual serialized Adam did not complete full budget')
    p = train.prepare_cpu(seed)
    if p['binding'] != report['binding'] or train.checkpoint.optimizer_schema(p['optimizer']) != payload['optimizer_schema']:
        raise ValueError('actual source restoration or optimizer schema differs')
    train.projection.module(p['found']).load_state_dict(payload['wrapped_state'],strict=True)
    p['found'].encoder.load_state_dict(payload['encoder_state'],strict=True)
    found = p['found']
    if (train.source.source.full.screen.native._state_sha(found.wrapped,found.encoder) != report['final_model_sha256']
            or train.projection.frozen_core_sha(found) != report['binding']['frozen_nonprojection_core_sha256']
            or train.helpers.statistics_sha(found) != report['binding']['input_statistics_sha256']):
        raise ValueError('actual final full model/frozen core/input statistics differs')
    return dict(proof,checkpoint_sha256=train.sha(raw),actual_Adam_steps=steps,
        restored_final_full_model_sha256=report['final_model_sha256'],
        frozen_nonprojection_core_and_input_statistics_unchanged=True)


def audit():
    if torch.cuda.is_initialized():
        raise ValueError('completion audit is CPU only')
    a = train.ARCHIVE
    raw = (a/'score_v1/summary.json').read_bytes(); score = json.loads(raw)
    private = json.loads((a/'score_v1/evaluation_private.json').read_bytes())
    if score['fingerprint'] != train.fingerprint() or private['summary'] != score:
        raise ValueError('frozen full-data score changed')
    selection = train.selection()
    if score['selected_source'] != selection['selected']:
        raise ValueError('initial best model differs from registered selection')
    bundle = {s:evaluate.validated_seed(s) for s in train.SEEDS}
    slots = evaluate.reference.load_slot_masks()
    checkpoints = {str(s):audit_checkpoint(s,bundle[s]['report']) for s in train.SEEDS}
    datasets = [bundle[s]['report']['binding']['dataset'] for s in train.SEEDS]
    hashes = [bundle[s]['protocol']['RGB_sha256'] for s in train.SEEDS]
    if any(v != datasets[0] for v in datasets) or len(set(hashes)) != 1:
        raise ValueError('different canonical fixed evaluation data')
    gt = train.source.source.full.read_gt(dict(dataset=datasets[0]))
    independent = train.source.source.full.metrics.independent
    vectors = {}; maximum = 0.
    for seed in train.SEEDS:
        for name,labels in (('ours',bundle[seed]['labels']),('slot',slots[seed]['labels64'])):
            key = f'{name}_seed{seed}'
            values = np.stack([independent.image_metrics(p,t) for p,t in zip(labels,gt)])
            vectors[key] = values; maximum = max(maximum,difference(values,private['vectors'][key]))
    error,strict,goal = check_statistics(vectors,score,selection); maximum = max(maximum,error)
    if maximum >= 1e-12:
        raise ValueError('raw masks do not independently reproduce frozen full-data score')
    figures = {name:train.sha((a/'visualizations'/name).read_bytes())
        for name in (f'image{i}_{model}.png' for i in (1320,1321,1322) for model in ('ours_seed0','slot_seed0'))}
    result = dict(status='complete_independently_audited_native64_TRAIN70000',
        score_sha256=train.sha(raw), diagnostic_source_sha256=train.sha(Path(__file__).read_bytes()),
        checkpoints=checkpoints,raw_masks_independently_rescored=1920,
        independent_metrics_means_and_bootstrap_maximum_abs_error=maximum,
        three_seed_strictly_exceeds_Slot_means=strict,native64_FG090_three_seed_goal_verified=goal,
        Slot_references={str(s):dict(prediction_sha256=slots[s]['prediction_sha256'],protocol_sha256=slots[s]['protocol_sha256'],
            labels64_sha256=train.sha(slots[s]['labels64'].tobytes())) for s in train.SEEDS},
        predetermined_seed0_figures_sha256=figures,additional_optimizer_updates=0,additional_model_forwards=0,
        full_training_reexecuted=False,actual_final_checkpoints_restored_on_CPU=True,
        all_three_prediction_protocols_and_Slot_artifacts_checked_before_evaluation_GT=True)
    train.write_json(a/'completion_audit_v1/summary.json',result)
    return result


if __name__ == '__main__':
    import ctypes,sys
    if sys.platform == 'linux' and ctypes.CDLL(None).prctl(41,1,0,0,0) != 0:
        raise RuntimeError('per-process THP disable failed')
    print(json.dumps(audit()))
