"""Read-only native64 TRAIN16 pair. Run only with an assigned exclusive GPU."""
import argparse
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .analyze import TRAIN_INDICES, paired_changes
from collaborative_test.SW_0194_native64_train64 import train, evaluate

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'results_archive'
ARMS = ('TRAIN8', 'TRAIN64')


def fingerprint():
    return dict(files={name: train.sha((HERE/name).read_bytes()) for name in
                      ('run.py', 'analyze.py', 'queue.py', 'DESIGN_KO.md', 'test_analyze.py', 'test_run.py')},
                parent=train.fingerprint(), train_indices=list(TRAIN_INDICES),
                inference_seed_base=195000, optimizer_updates=0)


def row_binding(binding, arm, offset):
    return dict(arm=arm, image_id=binding['image_ids'][offset],
                inference_seed=195000+offset, model_sha256=binding['models'][arm],
                RGB_sha256=binding['RGB_rows_sha256'][offset],
                experiment_binding_sha256=train.sha(json.dumps(binding, sort_keys=True).encode()))


def prepare_cpu():
    if torch.cuda.is_initialized(): raise RuntimeError('prepare on CPU before CUDA')
    _labels, protocol = evaluate.validated_candidate()  # Requires completed194, no GT.
    prepared = train.prepare_inputs_cpu()  # Known TRAIN GT available, never used by inference.
    raw = (train.ARCHIVE/'training_work/boundary.ptx').read_bytes()
    envelope = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if train.sha(envelope['payload_bytes']) != envelope['payload_sha256']:
        raise ValueError('candidate checkpoint corrupt')
    payload = torch.load(io.BytesIO(envelope['payload_bytes']), map_location='cpu', weights_only=False)
    report = json.loads((train.ARCHIVE/'training/report.json').read_bytes())
    if (payload['bindings'] != protocol['bindings'] or payload['completed_updates'] != 16
            or payload['history'] != report['history']):
        raise ValueError('candidate checkpoint/report lineage changed')
    from collaborative_test.SW_0179_qce_seed_replication import train as state_api
    states = dict(wrapped=payload['wrapped_state'], encoder=payload['encoder_state'])
    if state_api._state_sha_from_payload(states) != protocol['model_sha256']:
        raise ValueError('candidate checkpoint model differs from validation model')
    ids = [prepared['bindings']['ordered_TRAIN64_ids'][i] for i in TRAIN_INDICES]
    rgb = np.ascontiguousarray(prepared['rgb'][list(TRAIN_INDICES)])
    if set(ids) & set(evaluate.screen.IDS.tolist()): raise ValueError('TRAIN/development overlap')
    binding = dict(fingerprint=fingerprint(), parent_bindings=prepared['bindings'],
                   image_ids=ids, RGB_rows_sha256=[train.sha(v.tobytes()) for v in rgb],
                   models=dict(TRAIN8=prepared['parent_model_sha256'], TRAIN64=protocol['model_sha256']),
                   candidate_boundary_sha256=train.sha(raw),
                   validation_protocol_sha256=train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes()),
                   known_TRAIN_GT_available_in_factory=True, GT_used_for_prediction=False)
    return prepared['found'], states, rgb, binding


def predict(device='cuda:0', expected_binding=None):
    found, states, rgb, binding = prepare_cpu()
    if expected_binding is not None and binding != expected_binding:
        raise ValueError('worker inputs differ from CPU admission')
    device = torch.device(device); torch.cuda.set_device(device)
    train.probe.model64.move_to_device(found, device)
    found.wrapped.requires_grad_(False); found.encoder.requires_grad_(False)
    found.wrapped.eval(); found.encoder.eval()
    for arm in ARMS:
        if arm == 'TRAIN64':
            found.wrapped.load_state_dict(states['wrapped'], strict=True)
            found.encoder.load_state_dict(states['encoder'], strict=True)
        if train.probe.native._state_sha(found.wrapped, found.encoder) != binding['models'][arm]:
            raise ValueError('wrong frozen model')
        rows = []
        for offset in range(len(TRAIN_INDICES)):
            expected = row_binding(binding, arm, offset)
            folder = ARCHIVE/arm/'rows'/f'{offset:04d}'
            if folder.exists(): labels = evaluate.screen.validate_saved_row(folder, expected)
            else:
                torch.manual_seed(195000+offset); torch.cuda.manual_seed(195000+offset)
                started = time.perf_counter()
                with torch.no_grad():
                    image = torch.from_numpy(rgb[offset:offset+1]).to(device)
                    gamma, trace, adj, z, q, rho = train.probe.native._native_forward(found, image, live_tail_steps=0)
                    labels = evaluate.screen.prediction_from_trace(found, trace, adj, q)
                stage = folder.with_name(folder.name+f'.{time.time_ns()}.tmp')
                stage.mkdir(parents=True, exist_ok=False)
                with (stage/'labels.npz').open('xb') as stream:
                    np.savez_compressed(stream, labels=labels); stream.flush(); os.fsync(stream.fileno())
                train.write_json(stage/'row.json', dict(binding=expected,
                    prediction_sha256=train.sha((stage/'labels.npz').read_bytes()), seconds=time.perf_counter()-started))
                stage.rename(folder)
                del image, gamma, trace, adj, z, q, rho
            rows.append(labels)
        if train.probe.native._state_sha(found.wrapped, found.encoder) != binding['models'][arm]:
            raise RuntimeError('read-only inference mutated model')
        protocol = dict(binding=binding, arm=arm, labels_sha256=train.sha(np.stack(rows).tobytes()), optimizer_updates=0)
        path = ARCHIVE/arm/'protocol.json'
        if path.exists():
            if json.loads(path.read_bytes()) != protocol: raise ValueError('frozen protocol changed')
        else: train.write_json(path, protocol)


def validated_pair():
    result = {}; common = None
    for arm in ARMS:
        protocol = json.loads((ARCHIVE/arm/'protocol.json').read_bytes()); binding = protocol['binding']
        if (binding['fingerprint'] != fingerprint() or protocol['arm'] != arm
                or protocol['optimizer_updates'] != 0 or binding['GT_used_for_prediction'] is not False):
            raise ValueError('invalid diagnostic prediction protocol')
        if common is not None and binding != common: raise ValueError('cohort/model lineage differs')
        common = binding
        if len(binding['image_ids']) != 16 or len(set(binding['image_ids'])) != 16:
            raise ValueError('invalid TRAIN16 cohort')
        result[arm] = np.stack([evaluate.screen.validate_saved_row(
            ARCHIVE/arm/'rows'/f'{i:04d}', row_binding(binding, arm, i)) for i in range(16)])
        if train.sha(result[arm].tobytes()) != protocol['labels_sha256']: raise ValueError('array digest changed')
    return result, common


def score():
    predictions, binding = validated_pair()
    _val, protocol = evaluate.validated_candidate()
    if (protocol['bindings'] != binding['parent_bindings']
            or protocol['model_sha256'] != binding['models']['TRAIN64']
            or train.sha((train.ARCHIVE/'candidate/protocol.json').read_bytes()) != binding['validation_protocol_sha256']):
        raise ValueError('development model/lineage changed')
    private = json.loads((train.ARCHIVE/'score_v1/evaluation_private.json').read_bytes())
    public = json.loads((train.ARCHIVE/'score_v1/summary.json').read_bytes())
    if private['summary'] != public or public['candidate_model_sha256'] != binding['models']['TRAIN64']:
        raise ValueError('development vectors are not from completed194')
    # Score validation labels independently again; do not trust unbound private vectors.
    controls, ref = evaluate.screen_score.validated_pair()
    if ref['candidate_model_sha256'] != binding['models']['TRAIN8']:
        raise ValueError('TRAIN8 diagnostic and development models differ')
    gt_val = evaluate.screen_score.read_ground_truth(ref)
    dev = {}
    for arm, labels in (('TRAIN8', controls['candidate']), ('TRAIN64', _val)):
        dev[arm], _error = evaluate.screen_score.measured_vectors(labels, gt_val)
        if not np.allclose(dev[arm], np.asarray(private['vectors'][arm]), atol=1e-12, rtol=0):
            raise ValueError('development vectors differ from actual masks')
    import h5py
    from collaborative_test.SW_0130_phase_state_integration import run as data
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    if Path(data.source97.DATASET).resolve() != Path(ref['dataset']['path']).resolve():
        raise ValueError('TRAIN/development canonical datasets differ')
    with h5py.File(data.source97.DATASET, 'r') as stream:
        pixels = np.stack([stream['mask'][i,...,0] for i in binding['image_ids']])
    gt = clevr_mask_patch(torch.from_numpy(pixels.copy()), 2)['patch_labels'].numpy().astype(np.int64)
    if not np.array_equal(gt, evaluate.screen_score.independent.modal_labels(pixels, patch=2)):
        raise ValueError('independent TRAIN modal GT mismatch')
    vectors = {arm: evaluate.screen_score.measured_vectors(labels, gt)[0] for arm, labels in predictions.items()}
    summary = paired_changes(vectors['TRAIN8'], vectors['TRAIN64'], dev['TRAIN8'], dev['TRAIN64'])
    summary.update(binding=binding, optimizer_updates=0, raw_imagewise_score_public=False,
                   status='completed_read_only_generalization_diagnostic')
    train.write_json(ARCHIVE/'score_v1/evaluation_private.json', dict(summary=summary,
                     TRAIN_vectors={k: v.tolist() for k,v in vectors.items()}))
    train.write_json(ARCHIVE/'score_v1/summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--score', action='store_true'); parser.add_argument('--cpu-output', type=Path)
    args = parser.parse_args()
    if sum((args.worker, args.score, args.cpu_output is not None)) != 1: parser.error('choose exactly one action')
    if args.worker: predict()
    elif args.score: print(json.dumps(score()), flush=True)
    else:
        _found, _states, _rgb, binding = prepare_cpu()
        train.write_json(args.cpu_output, dict(binding=binding, cuda_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__': main()
