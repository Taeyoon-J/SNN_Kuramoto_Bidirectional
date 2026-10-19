"""Freeze native64 predictions for all three seeds before fixed320 GT scoring."""
import json
import os
import time
import numpy as np
import torch
from . import train, data
from collaborative_test.SW_0221_native64_frequency_pretraining import evaluate as reader
from collaborative_test.SW_0237_native64_converged_full320 import evaluate as reference


def predict(p, report, device):
    import h5py
    found = p['found']; a = train.archive_for(p['seed'])
    found.wrapped.requires_grad_(False); found.encoder.requires_grad_(False)
    found.wrapped.eval(); found.encoder.eval()
    binding = dict(training_report_sha256=train.sha((a / 'training_report.json').read_bytes()),
        binding=p['binding'], model_sha256=report['final_model_sha256'])
    with h5py.File(p['binding']['dataset']['path'], 'r') as stream:
        rgb = data.canonical_rgb(np.asarray(stream['image'][1320:1640]))
    masks = []
    for offset, image in enumerate(rgb):
        folder = a / 'rows' / f'{offset:04d}'
        if not folder.exists():
            torch.manual_seed(193000 + offset); torch.cuda.manual_seed(193000 + offset)
            with torch.no_grad():
                gamma, trace, adj, z, q, rho = train.source.source.full.screen.native._native_forward(
                    found, torch.from_numpy(image[None]).to(device), live_tail_steps=0)
                labels = np.asarray(train.source.source.full.screen.prediction_from_trace(found, trace, adj, q), dtype=np.int64)
                frequency = train.helpers.effective_frequency(found.wrapped.core, gamma)[0].cpu().numpy()
            stage = folder.with_name(folder.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
            stage.mkdir(parents=True)
            train.helpers.save_arrays(stage / 'prediction.npz', labels=labels, frequency=frequency)
            train.write_json(stage / 'row.json', dict(binding=binding, index=offset, image_id=1320 + offset,
                RGB_sha256=train.sha(image.tobytes()), inference_seed=193000 + offset,
                data_sha256=train.sha((stage / 'prediction.npz').read_bytes())))
            stage.rename(folder)
            del gamma, trace, adj, z, q, rho
        labels, _ = reader.read_row(folder, binding, offset)
        row = json.loads((folder / 'row.json').read_bytes())
        if row['image_id'] != 1320 + offset or row['RGB_sha256'] != train.sha(image.tobytes()) or row['inference_seed'] != 193000 + offset:
            raise ValueError('fixed evaluation image/order/RNG changed')
        masks.append(labels)
    if train.source.source.full.screen.native._state_sha(found.wrapped, found.encoder) != report['final_model_sha256']:
        raise ValueError('trained model changed during native inference')
    protocol = dict(binding=binding, image_ids=list(range(1320, 1640)), labels_sha256=train.sha(np.stack(masks).tobytes()),
        RGB_sha256=train.sha(rgb.tobytes()), total_steps=1024, settle_steps=512, DEV_GT_read=False,
        new_TRAIN_images=70000, complete_passes=10, new_optimizer_updates=data.TOTAL_STEPS)
    path = a / 'protocol.json'
    if path.exists() and json.loads(path.read_bytes()) != protocol:
        raise ValueError('completed evaluation protocol differs')
    if not path.exists():
        train.write_json(path, protocol)


def validated_seed(seed):
    a = train.archive_for(seed)
    report_raw = (a / 'training_report.json').read_bytes(); report = json.loads(report_raw)
    protocol = json.loads((a / 'protocol.json').read_bytes())
    if report['binding']['fingerprint'] != train.fingerprint() or report['new_optimizer_updates'] != data.TOTAL_STEPS:
        raise ValueError('source/full-training counter changed')
    if report['binding']['selection_sha256'] != train.sha((train.ARCHIVE / 'selection.json').read_bytes()):
        raise ValueError('selected best model changed')
    if report['complete_passes'] != 10 or report['unique_TRAIN_images'] != 70000 or report['new_TRAIN_exposures'] != 700000 or report['actual_Adam_steps'] != [data.TOTAL_STEPS]:
        raise ValueError('incomplete actual TRAIN70000 coverage')
    if report['checkpoint_sha256'] != train.sha((a / 'encoder_work/boundary.ptx').read_bytes()):
        raise ValueError('trained checkpoint changed')
    expected = dict(training_report_sha256=train.sha(report_raw), binding=report['binding'], model_sha256=report['final_model_sha256'])
    if protocol['binding'] != expected or protocol['image_ids'] != list(range(1320, 1640)) or protocol['DEV_GT_read'] is not False or protocol['total_steps'] != 1024 or protocol['settle_steps'] != 512:
        raise ValueError('native evaluation contract changed')
    masks = np.stack([reader.read_row(a / 'rows' / f'{i:04d}', expected, i)[0] for i in range(320)])
    if masks.shape != (320, 64, 64) or train.sha(masks.tobytes()) != protocol['labels_sha256']:
        raise ValueError('actual native masks changed')
    return dict(labels=masks, report=report, protocol=protocol)


def score():
    bundle = {s: validated_seed(s) for s in train.SEEDS}
    slots = reference.load_slot_masks()
    datasets = [bundle[s]['report']['binding']['dataset'] for s in train.SEEDS]
    rgb_hashes = [bundle[s]['protocol']['RGB_sha256'] for s in train.SEEDS]
    if any(v != datasets[0] for v in datasets) or len(set(rgb_hashes)) != 1:
        raise ValueError('different canonical evaluation data across seeds')
    # All three complete prediction protocols and comparable Slot artifacts precede GT.
    gt = train.source.source.full.read_gt(dict(dataset=datasets[0]))
    metrics = train.source.source.full.metrics
    vectors = {}; error = 0.
    for seed in train.SEEDS:
        for arm, labels in (('ours', bundle[seed]['labels']), ('slot', slots[seed]['labels64'])):
            value, diff = metrics.measured_vectors(labels, gt)
            vectors[f'{arm}_seed{seed}'] = value; error = max(error, diff)
    groups = {arm: np.stack([vectors[f'{arm}_seed{s}'] for s in train.SEEDS]) for arm in ('ours', 'slot')}
    if any(v.shape != (3, 320, 3) or not np.isfinite(v).all() for v in groups.values()) or error >= 1e-12:
        raise ValueError('independent full320 metric agreement failed')
    keys = metrics.independent.METRICS
    delta = (groups['ours'] - groups['slot']).mean(0)
    draw = np.random.RandomState(239).randint(0, 320, (10000, 320))
    ci = np.percentile(delta[draw].mean(1), [2.5, 97.5], axis=0)
    strict = bool(np.all(groups['ours'].mean((0, 1)) > groups['slot'].mean((0, 1))))
    summary = dict(status='complete_native64_TRAIN70000_fixed320_three_seeds', image_count=320,
        grid=[64, 64], seed_scope=list(train.SEEDS), fingerprint=train.fingerprint(),
        scores={k: dict(zip(keys, v.mean((0, 1)).tolist())) for k, v in groups.items()},
        per_seed_scores={k: dict(zip(keys, v.mean(0).tolist())) for k, v in vectors.items()},
        paired_ours_minus_slot={k: dict(mean=float(delta[:, i].mean()), ci95=ci[:, i].tolist()) for i, k in enumerate(keys)},
        selected_source=train.selection()['selected'], independent_metric_max_abs_error=error,
        three_seed_strictly_exceeds_Slot_means=strict,
        native64_FG090_three_seed_goal_verified=bool(strict and groups['ours'][:, :, 0].mean() >= .9),
        mask_supervised_training=True, Slot_reconstruction_unsupervised=True,
        evaluation_split='reused fixed validation; not untouched test',
        new_TRAIN_unique_images_per_seed=70000, new_TRAIN_complete_passes=10,
        new_optimizer_updates_per_seed=data.TOTAL_STEPS, maximum_concurrent_GPU_workers=1)
    train.write_json(train.ARCHIVE / 'score_v1/evaluation_private.json',
        dict(summary=summary, vectors={k: v.tolist() for k, v in vectors.items()}))
    train.write_json(train.ARCHIVE / 'score_v1/summary.json', summary)
    visualize(bundle[0]['labels'], slots[0]['labels64'], gt, datasets[0])
    return summary


def visualize(ours, slot, truth, dataset):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = train.ARCHIVE / 'visualizations'; out.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset['path'], 'r') as stream:
        rgb = data.canonical_rgb(np.asarray(stream['image'][1320:1323]))
    for i in range(3):
        for name, labels in (('ours_seed0', ours[i]), ('slot_seed0', slot[i])):
            objects = np.unique(labels[labels > 0])
            count = 3 + len(objects); rows = (count + 4) // 5
            fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows), squeeze=False)
            panels = [(rgb[i], 'RGB'), (labels, name), (truth[i], 'GT')]
            panels += [(labels == k, f'Instance {k}') for k in objects]
            for ax in axes.flat:
                ax.axis('off')
            for ax, (value, title) in zip(axes.flat, panels):
                ax.imshow(value, cmap='tab20' if value.ndim == 2 and value.dtype != bool else 'gray', interpolation='nearest')
                ax.set_title(title)
            fig.suptitle(f'Fixed image {1320+i}; native64; {name}')
            fig.tight_layout(); fig.savefig(out / f'image{1320+i}_{name}.png', dpi=160); plt.close(fig)
