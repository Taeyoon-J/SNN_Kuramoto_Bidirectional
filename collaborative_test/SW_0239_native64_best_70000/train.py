"""Stream all 70000 TRAIN images with the accepted fixed-source teacher recipe."""
import argparse
import importlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0221_native64_frequency_pretraining import train as helpers
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher, teacher_objective
from collaborative_test.SW_0232_native64_exact_teacher_resume import projection
from . import data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = HERE / 'results_archive'
SEEDS = (0, 1, 2)
source = helpers.source
checkpoint = helpers.checkpoint
resume_policy = helpers.resume_policy
sha = helpers.sha
write_json = helpers.write_json
state_sha = helpers.state_sha
CHECKPOINT_EVERY = 32


def require_seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('registered seed0/1/2 required')
    return seed


def archive_for(seed):
    return ARCHIVE / f'seed{require_seed(seed)}'


def fingerprint():
    names = ('DESIGN_KO.md', 'data.py', 'train.py', 'evaluate.py', 'queue.py', 'test_data.py', 'test_training.py', 'test_queue.py', 'conftest.py')
    return dict(files={n: sha((HERE / n).read_bytes()) for n in names},
        original_teacher_source=source.fingerprint(), split=data.split_binding(),
        teacher_loss_sha256=sha(Path(importlib.import_module(teacher_objective.__module__).__file__).read_bytes()),
        checkpoint_sha256=sha(Path(checkpoint.__file__).read_bytes()),
        learning_rates=[helpers.BASE_LR, helpers.CONTEXT_LR, helpers.CONTEXT_LR],
        teacher_device='cpu', reset_Adam_for_new_data_stage=True,
        core_updates='actual sensory projection only', other_core_updates=0,
        native_grid=[64, 64], total_steps=1024, settle_steps=512,
        sequential_seeds=list(SEEDS), maximum_concurrent_GPU_workers=1,
        checkpoint_every_updates=CHECKPOINT_EVERY)


def evidence(folder, server, local):
    paths = [p for p in (folder / server, folder / local) if p.exists()]
    if not paths or any(p.read_bytes() != paths[0].read_bytes() for p in paths[1:]):
        raise ValueError('missing or inconsistent accepted evidence')
    return paths[0].read_bytes()


def select_source():
    path = ARCHIVE / 'selection.json'
    if path.exists():
        return selection()
    choices = []
    for number, name in ((227, 'SW_0227_native64_joint_projection_full320'),
                         (237, 'SW_0237_native64_converged_full320')):
        module = importlib.import_module('collaborative_test.' + name + '.run')
        if not any((module.ARCHIVE / n).exists() for n in ('score_v1/summary.json', 'completed_v1/summary.json')):
            continue
        # A scored but not yet independently audited experiment is not eligible.
        if not any((module.ARCHIVE / n).exists() for n in ('completion_audit_v1/summary.json', 'completed_v1/completion_audit.json')):
            continue
        raw = evidence(module.ARCHIVE, 'score_v1/summary.json', 'completed_v1/summary.json')
        audit_raw = evidence(module.ARCHIVE, 'completion_audit_v1/summary.json', 'completed_v1/completion_audit.json')
        score = json.loads(raw); audit = json.loads(audit_raw)
        if (score['fingerprint'] != module.fingerprint() or audit['score_sha256'] != sha(raw)
                or audit['diagnostic_source_sha256'] != sha((module.HERE / 'completion_audit.py').read_bytes())
                or audit['raw_masks_independently_rescored'] != 2880
                or not np.isfinite(audit['independent_metrics_means_and_bootstrap_maximum_abs_error'])
                or audit['independent_metrics_means_and_bootstrap_maximum_abs_error'] >= 1e-12
                or score['image_count'] != 320 or score['seed_scope'] != list(SEEDS)):
            raise ValueError('completed source failed independent full320 audit')
        metrics = ('fg_ari', 'foreground_iou', 'matched_object_iou')
        own = score['scores']['joint_projection']; slot = score['scores']['slot']
        if not all(np.isfinite(own[k]) and own[k] > slot[k] for k in metrics):
            continue
        choices.append(dict(experiment=number, module=module.__name__, score_sha256=sha(raw),
            audit_sha256=sha(audit_raw), scores=own, slot=slot))
    if not choices:
        raise ValueError('no independently accepted full320 best source')
    chosen = max(choices, key=lambda v: (v['scores']['fg_ari'], -v['experiment']))
    record = dict(fingerprint=fingerprint(), selection_rule='highest full320 three-seed FG among independently audited 227/237 exceeding Slot on all metrics',
        selected=chosen, eligible=choices)
    write_json(path, record)
    return record


def selection():
    value = json.loads((ARCHIVE / 'selection.json').read_bytes())
    if value['fingerprint'] != fingerprint():
        raise ValueError('registered source changed after selection')
    module = importlib.import_module(value['selected']['module'])
    if (sha(evidence(module.ARCHIVE, 'score_v1/summary.json', 'completed_v1/summary.json')) != value['selected']['score_sha256']
            or sha(evidence(module.ARCHIVE, 'completion_audit_v1/summary.json', 'completed_v1/completion_audit.json')) != value['selected']['audit_sha256']):
        raise ValueError('selected full320 source evidence changed')
    return value


def optimizer_for(found):
    return projection.enable(found, helpers.optimizer_for(found))


def teacher_targets(teacher, rgb, labels):
    with torch.no_grad():
        gamma = teacher.encode(torch.from_numpy(rgb))
        frequency = helpers.effective_frequency(teacher.wrapped.core, gamma)
    return make_teacher(gamma.cpu().numpy(), frequency.cpu().numpy(), labels)


def prepare_cpu(seed):
    require_seed(seed)
    if torch.cuda.is_initialized():
        raise ValueError('CPU admission and original teacher must precede CUDA')
    chosen = selection()
    teacher, _, _, parent = source.prepare_cpu(seed)
    found, _, _, parent2 = source.prepare_cpu(seed)
    if parent != parent2:
        raise ValueError('different original foundation bindings')
    teacher.wrapped.requires_grad_(False); teacher.encoder.requires_grad_(False)
    teacher.wrapped.eval(); teacher.encoder.eval()
    module = importlib.import_module(chosen['selected']['module'])
    report, ref, _, _ = module.scope.restore(found, seed)
    dataset = dict(parent['dataset']); stat = Path(dataset['path']).stat()
    if (stat.st_size, stat.st_mtime_ns) != (dataset['size'], dataset['mtime_ns']):
        raise ValueError('canonical dataset changed')
    if helpers.statistics_sha(found) != helpers.statistics_sha(teacher):
        raise ValueError('student and source teacher input statistics differ')
    optimizer = optimizer_for(found)
    binding = dict(fingerprint=fingerprint(), selection_sha256=sha((ARCHIVE / 'selection.json').read_bytes()),
        seed=seed, source=ref, dataset=dataset, teacher_model_sha256=source.MODELS[str(seed)],
        frozen_nonprojection_core_sha256=projection.frozen_core_sha(found),
        input_statistics_sha256=helpers.statistics_sha(found), source_report_sha256=ref['report_sha256'],
        mask_supervised_training=True, DEV_GT_read=False, optimizer_reset=True)
    return dict(found=found, teacher=teacher, optimizer=optimizer, binding=binding, seed=seed)


def gradient_probe(p, rgb, labels):
    targets = teacher_targets(p['teacher'], rgb, labels)
    gamma = p['found'].encode(torch.from_numpy(rgb))
    frequency = helpers.effective_frequency(p['found'].wrapped.core, gamma)
    loss, parts = teacher_objective(gamma, frequency, targets)
    params = list(p['found'].encoder.parameters()) + projection.parameters(p['found'])
    grads = torch.autograd.grad(loss, params)
    if not all(bool(torch.isfinite(v).all()) for v in grads) or float(grads[-1].norm()) <= 0:
        raise ValueError('actual encoder/projection gradient admission failed')
    return dict(loss=float(loss.detach()), projection_gradient_norm=float(grads[-1].norm()),
        encoder_gradient_norm=float(torch.sqrt(sum(v.square().sum() for v in grads[:-1]))),
        frequency_denoising=float(parts['frequency_denoising'].detach()), gamma_anchor=float(parts['gamma_anchor'].detach()))


def admit(seed):
    import h5py
    p = prepare_cpu(seed); found = p['found']; start = time.monotonic()
    original = source.source.full.screen.native._state_sha(found.wrapped, found.encoder)
    probes = []
    with h5py.File(p['binding']['dataset']['path'], 'r') as stream:
        for step in (0, data.TOTAL_STEPS - 1):
            rgb, labels = data.read_batch(stream, data.batch_ids(seed, step))
            probes.append(gradient_probe(p, rgb, labels))
        # Confirm unchanged original CPU teacher against the actual accepted223 target cache.
        accepted = projection.control.archive_for(seed, 64)
        old_report = json.loads((accepted / 'training_report.json').read_bytes())
        ids = old_report['binding']['TRAIN_ids'][:data.BATCH]
        rgb, labels = data.read_batch(stream, ids)
        targets = teacher_targets(p['teacher'], rgb, labels)
        with np.load(accepted / 'teacher.npz', allow_pickle=False) as old:
            errors = {k: float(np.max(np.abs(v.numpy() - old[k][:data.BATCH]))) for k, v in targets.items()}
        if any(v != 0 for v in errors.values()):
            raise ValueError('streamed CPU teacher does not exactly reproduce accepted223 teacher arrays')
    if (source.source.full.screen.native._state_sha(found.wrapped, found.encoder) != original
            or original != p['binding']['source']['model_sha256']
            or source.source.full.screen.native._state_sha(p['teacher'].wrapped, p['teacher'].encoder) != p['binding']['teacher_model_sha256']
            or any(v.grad is not None for v in list(found.encoder.parameters()) + projection.parameters(found))
            or p['optimizer'].state):
        raise ValueError('read-only CPU admission mutated model/teacher/optimizer')
    result = dict(binding=p['binding'], actual_CPU_model_and_gradients_passed=True,
        original_teacher_bitwise_target_errors=errors, probes=probes,
        elapsed_seconds=time.monotonic() - start, CUDA_initialized=False, optimizer_updates=0)
    write_json(archive_for(seed) / 'admission.json', result)
    return result


def check_history(history, seed):
    for epoch in range((len(history) + data.STEPS_PER_EPOCH - 1) // data.STEPS_PER_EPOCH):
        sequence = data.epoch_ids(seed, epoch)
        start = epoch * data.STEPS_PER_EPOCH
        for step in range(start, min(len(history), start + data.STEPS_PER_EPOCH)):
            row = history[step]; offset = (step - start) * data.BATCH
            if row['update'] != step + 1 or row['TRAIN_ids'] != sequence[offset:offset + data.BATCH].tolist():
                raise ValueError('checkpoint TRAIN update order changed')
            if not all(np.isfinite(row[k]) for k in ('loss', 'frequency_denoising', 'gamma_anchor', 'gradient_norm')):
                raise ValueError('nonfinite committed training history')


def gpu_preflight(p, device):
    import h5py
    with h5py.File(p['binding']['dataset']['path'], 'r') as stream:
        rgb, labels = data.read_batch(stream, data.batch_ids(p['seed'], 0))
    found = p['found']; before = source.source.full.screen.native._state_sha(found.wrapped, found.encoder)
    targets = teacher_targets(p['teacher'], rgb, labels)
    gamma = found.encode(torch.from_numpy(rgb).to(device))
    frequency = helpers.effective_frequency(found.wrapped.core, gamma)
    loss, _ = teacher_objective(gamma, frequency, targets)
    params = list(found.encoder.parameters()) + projection.parameters(found)
    grads = torch.autograd.grad(loss, params)
    if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(v).all()) for v in grads) or float(grads[-1].norm()) <= 0:
        raise ValueError('actual assigned-GPU gradient preflight failed')
    loss_value = float(loss.detach()); norm = float(grads[-1].norm())
    del gamma, frequency, loss, grads, targets
    torch.manual_seed(239000 + p['seed']); torch.cuda.manual_seed(239000 + p['seed'])
    with torch.no_grad():
        gamma, trace, adj, z, q, rho = source.source.full.screen.native._native_forward(
            found, torch.from_numpy(rgb[:1]).to(device), live_tail_steps=0)
        mask = np.asarray(source.source.full.screen.prediction_from_trace(found, trace, adj, q))
    if mask.shape != (64, 64) or mask.dtype.kind not in 'iu' or (mask < 0).any():
        raise ValueError('actual native64 spike readout preflight failed')
    if source.source.full.screen.native._state_sha(found.wrapped, found.encoder) != before or p['optimizer'].state:
        raise ValueError('GPU preflight mutated model or optimizer')
    write_json(archive_for(p['seed']) / 'GPU_preflight_latest.json', dict(binding=p['binding'],
        actual_GPU_batch_gradient_passed=True, actual_native64_TRAIN_forward_passed=True,
        loss=loss_value, projection_gradient_norm=norm, optimizer_updates=0), replace=True)


def fit(p, device, runtime):
    import h5py
    seed = p['seed']; found = p['found']; optimizer = p['optimizer']; a = archive_for(seed)
    if json.loads((a / 'admission.json').read_bytes())['binding'] != p['binding']:
        raise ValueError('worker differs from actual CPU admission')
    binding = dict(binding=p['binding'], admission_sha256=sha((a / 'admission.json').read_bytes()))
    work = a / 'encoder_work'; params = list(found.encoder.parameters()) + projection.parameters(found)
    if (work / 'boundary.ptx').exists():
        loaded = checkpoint.load_latest(work, wrapped=projection.module(found), encoder=found.encoder, optimizer=optimizer,
            expected_bindings=binding, expected_runtime=runtime, device=device, max_updates=data.TOTAL_STEPS)
        history = loaded['history']; check_history(history, seed)
    else:
        torch.manual_seed(239000 + seed); torch.cuda.manual_seed(239000 + seed)
        history = []
        checkpoint.save_boundary(work, completed_updates=0, wrapped=projection.module(found), encoder=found.encoder,
            optimizer=optimizer, history=history, bindings=binding, runtime=runtime, device=device)
    with h5py.File(p['binding']['dataset']['path'], 'r') as stream:
        sequence = None; epoch_previous = None
        for step in range(len(history), data.TOTAL_STEPS):
            epoch, offset = divmod(step, data.STEPS_PER_EPOCH)
            if epoch != epoch_previous:
                sequence = data.epoch_ids(seed, epoch); epoch_previous = epoch
            ids = sequence[offset * data.BATCH:(offset + 1) * data.BATCH]
            rgb, labels = data.read_batch(stream, ids)
            targets = teacher_targets(p['teacher'], rgb, labels)
            optimizer.zero_grad(set_to_none=True)
            gamma = found.encode(torch.from_numpy(rgb).to(device))
            frequency = helpers.effective_frequency(found.wrapped.core, gamma)
            loss, parts = teacher_objective(gamma, frequency, targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite streamed teacher loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('nonfinite streamed gradient')
            optimizer.step()
            if not all(bool(torch.isfinite(v).all()) for v in params):
                raise FloatingPointError('nonfinite model; last committed checkpoint preserved')
            history.append(dict(update=step + 1, TRAIN_ids=ids.tolist(), loss=float(loss.detach().cpu()),
                frequency_denoising=float(parts['frequency_denoising'].detach().cpu()),
                gamma_anchor=float(parts['gamma_anchor'].detach().cpu()), gradient_norm=float(norm.cpu())))
            if (step + 1) % CHECKPOINT_EVERY == 0 or (step + 1) % data.STEPS_PER_EPOCH == 0:
                checkpoint.save_boundary(work, completed_updates=len(history), wrapped=projection.module(found), encoder=found.encoder,
                    optimizer=optimizer, history=history, bindings=binding, runtime=runtime, device=device)
                write_json(a / 'progress.json', dict(completed_updates=len(history), complete_passes=len(history) // data.STEPS_PER_EPOCH,
                    logical_exposures=len(history) * data.BATCH, seed=seed, updated_unix=time.time()), replace=True)
            del gamma, frequency, loss, parts, targets
    # TOTAL_STEPS is an exact epoch boundary, hence final checkpoint is committed above.
    check_history(history, seed)
    adam = sorted({int(v['step'].item()) for v in optimizer.state.values()})
    if (len(history) != data.TOTAL_STEPS or adam != [data.TOTAL_STEPS]
            or projection.frozen_core_sha(found) != p['binding']['frozen_nonprojection_core_sha256']
            or helpers.statistics_sha(found) != p['binding']['input_statistics_sha256']
            or source.source.full.screen.native._state_sha(p['teacher'].wrapped, p['teacher'].encoder) != p['binding']['teacher_model_sha256']):
        raise ValueError('full TRAIN70k counters/frozen core/statistics/teacher invariant failed')
    stat = Path(p['binding']['dataset']['path']).stat()
    if (stat.st_size, stat.st_mtime_ns) != (p['binding']['dataset']['size'], p['binding']['dataset']['mtime_ns']):
        raise ValueError('dataset changed during training')
    report = dict(status='complete_native64_best_TRAIN70000', binding=p['binding'], checkpoint_binding=binding,
        actual_Adam_steps=adam, new_optimizer_updates=len(history), unique_TRAIN_images=data.TRAIN_COUNT,
        complete_passes=data.EPOCHS, new_TRAIN_exposures=len(history) * data.BATCH,
        final_model_sha256=source.source.full.screen.native._state_sha(found.wrapped, found.encoder),
        final_encoder_sha256=state_sha(found.encoder.state_dict()), other_core_updates=0,
        core_projection_updates=len(history), source_teacher_unchanged=True, DEV_GT_read=False,
        history_sha256=sha(json.dumps(history, sort_keys=True, allow_nan=False).encode()),
        checkpoint_sha256=sha((work / 'boundary.ptx').read_bytes()), runtime=runtime)
    path = a / 'training_report.json'
    if path.exists() and json.loads(path.read_bytes()) != report:
        raise ValueError('completed training report differs')
    if not path.exists():
        write_json(path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--admit', action='store_true')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.select:
        print(json.dumps(select_source()))
    elif args.admit and args.seed is not None:
        print(json.dumps(admit(args.seed)))
    else:
        parser.error('choose frozen source selection or CPU admission')
