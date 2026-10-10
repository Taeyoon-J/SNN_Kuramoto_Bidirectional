"""SW0130 bounded first-stage implementation and TRAIN-only preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'snn_kuramoto_bidirectional'), str(ROOT / 'collaborative_test')]

from SW_0094_aligned_joint_pilot.run import ASSETS, GAMMA as GAMMA_TRAIN
from SW_0094_aligned_joint_pilot.run import hparams
from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import (
    SharedRGBDecoder, groups_to_onehot, reconstruct_one, rgb_patch_means,
)
from collaborative_test.SW_0110_xy_graph_route import run as source97
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss, phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity, spike_synchrony_components
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv

GAMMA_TRAIN_MANIFEST = source97.GAMMA_TRAIN_MANIFEST

SEEDS = (0, 1, 2)
ARMS = ('phase', 'constant')
BATCH = 16
UPDATES = 256
TRAIN_STEPS = 64
TRAIN_SETTLE = 32
CORE_GRAPH_LR = 3e-5
ENCODER_LR = 3e-6
DECODER_LR = 3e-4
INTEGRATION_LR = 1e-2
CLIP_NORM = 1.0
OUT = ROOT / 'trained_models/SW0130_phase_state_integration'
ARCHIVE = HERE / 'results_archive'
SOURCE_ROOT = ROOT / 'trained_models/SW0097_graph_adaptation'
CACHE = ROOT / 'data/SW_0106_spike_partition_rgb'
TRAIN_RGB = CACHE / 'train_rgb_uint8.npy'
VAL_RGB = CACHE / 'validation_rgb_uint8.npy'
GAMMA_VAL = ROOT / 'data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt'
FEATURE_STATS = ASSETS / 'feature_preprocessing.pt'
ENCODER_SOURCE = ASSETS / 'input_encoder/input_layer_encoder.pt'
METRICS = ('fg_ari', 'foreground_iou', 'matched_object_iou')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)


def implementation_fingerprint():
    files = (
        HERE / 'run.py', HERE / 'model.py', HERE / 'protocol.json',
        ROOT / 'collaborative_test/SW_0106_spike_partition_rgb/partition_rgb.py',
        ROOT / 'collaborative_test/SW_0110_xy_graph_route/run.py',
        ROOT / 'collaborative_test/SW_0094_aligned_joint_pilot/run.py',
        ROOT / 'snn_kuramoto_bidirectional/s2net_cls.py',
        ROOT / 'snn_kuramoto_bidirectional/graph_generator.py',
        ROOT / 'snn_kuramoto_bidirectional/dendric_layer.py',
        ROOT / 'snn_kuramoto_bidirectional/membrane_layer.py',
        ROOT / 'snn_kuramoto_bidirectional/kuramoto_layer.py',
        ROOT / 'snn_kuramoto_bidirectional/sinusoidal_gating.py',
        ROOT / 'snn_kuramoto_bidirectional/spike_classifier.py',
        ROOT / 'snn_kuramoto_bidirectional/loss_function.py',
        ROOT / 'snn_kuramoto_bidirectional/gamma_initializer.py',
        ROOT / 'snn_kuramoto_bidirectional/hyperparameter.py',
        ROOT / 'snn_kuramoto_bidirectional/training/train_s2net_core.py',
        ROOT / 'snn_kuramoto_bidirectional/training/train_gamma_initializer.py',
    )
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def source_contract(seed):
    checkpoint, manifest_path, manifest = source97.source_paths(seed)
    expected = source97.EXPECTED_SOURCE_SHAS[seed]
    actual = sha(checkpoint)
    if actual != expected:
        raise AssertionError(f'SW0097 seed{seed} checkpoint SHA mismatch: {actual}')
    if (manifest.get('status') != 'complete' or manifest.get('source_model_seed') != seed
            or manifest.get('steps') != 256 or manifest.get('batch') != 16
            or manifest.get('seed') != 117 + seed
            or manifest.get('ground_truth_used_for_training') is not False):
        raise AssertionError('registered SW0097 source manifest is not a completed 256-update run')
    ids_array, pool_indices = source97.train_indices(seed)
    ids = [int(value) for value in ids_array.tolist()]
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    if len(ids) != 4096 or len(set(ids)) != 4096:
        raise AssertionError('registered SW0097 training order must contain 4096 unique IDs')
    if manifest.get('training_ids') != ids:
        raise AssertionError('SW0097 manifest training IDs do not match the registered 117+seed order')
    return checkpoint, manifest_path, manifest, pool_indices, ids, actual


def validate_rgb_assets():
    expected_train_ids = np.concatenate((np.arange(1000, dtype='<i8'),
                                         np.arange(1640, 70640, dtype='<i8')))
    expected_val_ids = np.arange(1320, 1640, dtype='<i8')
    for cache_path, ids, shape in (
        (TRAIN_RGB, expected_train_ids, [70000, 128, 128, 3]),
        (VAL_RGB, expected_val_ids, [320, 128, 128, 3]),
    ):
        meta_path = cache_path.with_suffix(cache_path.suffix + '.complete.json')
        if not cache_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f'registered RGB cache or completion metadata missing: {cache_path}')
        meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        expected_sha = hashlib.sha256(ids.tobytes()).hexdigest()
        if (meta.get('status') != 'complete' or meta.get('cache_shape') != shape
                or meta.get('ids_mapping_sha256') != expected_sha):
            raise AssertionError(f'RGB cache metadata/ID mapping mismatch: {meta_path}')
        mmap = np.load(cache_path, mmap_mode='r')
        try:
            if list(mmap.shape) != shape or mmap.dtype != np.uint8:
                raise AssertionError(f'RGB cache shape/dtype mismatch: {cache_path}')
        finally:
            del mmap
    stats = torch.load(FEATURE_STATS, map_location='cpu', weights_only=True)
    if stats.get('mode') != 'standardize':
        raise AssertionError('registered encoder preprocessing mode mismatch')
    mean, std = stats['mean'], stats['std']
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise AssertionError('registered feature statistics must be finite with positive standard deviations')
    return {'train_cache_sha256': sha(TRAIN_RGB), 'train_manifest_sha256': sha(TRAIN_RGB.with_suffix(TRAIN_RGB.suffix + '.complete.json')),
            'validation_cache_sha256': sha(VAL_RGB), 'validation_manifest_sha256': sha(VAL_RGB.with_suffix(VAL_RGB.suffix + '.complete.json')),
            'encoder_source_sha256': sha(ENCODER_SOURCE), 'feature_preprocessing_sha256': sha(FEATURE_STATS)}


def make_criterion():
    return UnsupervisedS2NetLoss(spike_rate_weight=0., spike_smooth_weight=0.,
        spike_diversity_weight=0., structural_weight=0., plv_bimodality_weight=6.,
        plv_balance_weight=10., plv_coherence_weight=.5, plv_collapse_weight=1.,
        plv_target_density=.867, patch_grid_size=(16, 16))


def load_models(seed, device, arm='phase'):
    checkpoint, manifest_path, manifest, pool, ids, source_sha = source_contract(seed)
    hp = hparams('raw')
    hp.num_time_steps = TRAIN_STEPS
    core = source97.S2NetCore(hp.validate(), device=device).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    core.load_state_dict(state, strict=True)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:
        raise AssertionError('SW0130 requires the registered no-feedback/no-pulse source')
    wrapped = PhaseStateIntegration(core, arm)
    encoder = load_input_encoder(str(ENCODER_SOURCE), num_kernels=8, kernel_size=3,
                                 channels=3, device=device)
    encoder.requires_grad_(True)
    encoder.train()
    stats = torch.load(FEATURE_STATS, map_location=device, weights_only=True)
    mean, std = stats['mean'].to(device), stats['std'].to(device)
    clip = float(stats.get('clip', 3.))
    patcher = FeaturePatchGammaInitializer(grid_size=16).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(106)
        decoder = SharedRGBDecoder().to(device)
    return wrapped, encoder, patcher, mean, std, clip, decoder, pool, ids, source_sha


def encode(encoder, patcher, mean, std, clip, images):
    features = encoder(images.float() / 255.)
    return patcher(((features - mean) / std).clamp(-clip, clip))


def read_batch(cache, indices, device):
    arr = np.asarray(cache[np.asarray(indices, dtype=np.int64)]).copy()
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)


def affinity(core, settle=TRAIN_SETTLE):
    components = core.core.last_component_spikes
    if components is None:
        raise RuntimeError('actual component spike history missing')
    return spike_synchrony_affinity(components.mean(dim=1), components,
                                   settle=settle, affinity_mode='spike')


def hard_labels(spikes, components, settle=TRAIN_SETTLE):
    groups = spike_synchrony_components(
        spikes.detach().cpu(), synchrony_threshold=.50, min_group_size=2,
        settle=settle, components=components.detach().cpu(),
        background='largest_component', affinity_mode='spike', spatial_grid_size=16)
    labels = spatial_components_to_patch_labels(groups, 16, device=spikes.device).reshape(spikes.shape[0], -1)
    hard = [groups_to_onehot(group, device=spikes.device, dtype=spikes.dtype) for group in groups]
    if any(not torch.equal(onehot.argmax(-1), label) for onehot, label in zip(hard, labels)):
        raise AssertionError('RGB hard partition differs from production label conversion')
    return labels, hard, groups


def forward_batch(core, encoder, patcher, mean, std, clip, images, criterion,
                  settle=TRAIN_SETTLE):
    gamma = encode(encoder, patcher, mean, std, clip, images)
    result = _forward_with_plv(core, gamma, criterion, settle, 'phase', 'mean')
    groups, spikes, core_out, plv, theta = result
    components = core.core.last_component_spikes
    if tuple(components.shape) != (images.shape[0], 4, 256, TRAIN_STEPS):
        raise AssertionError('source component history shape mismatch')
    q = affinity(core, settle)
    labels, hard, detected = hard_labels(spikes, components, settle)
    target = rgb_patch_means(images / 255.)
    return gamma, result, q, labels, hard, detected, target


def rgb_losses(q, hard, gamma, target, decoder, assignment_credit=True):
    predictions, losses, diagnostics = [], [], []
    for b in range(q.shape[0]):
        pred, loss, diag = reconstruct_one(q[b], hard[b], gamma[b].transpose(0, 1),
                                           target[b], decoder, assignment_credit)
        predictions.append(pred)
        losses.append(loss)
        diagnostics.append(diag)
    return torch.stack(predictions), torch.stack(losses).mean(), diagnostics


def grad_norm(grads):
    total = 0.0
    for grad in grads:
        if grad is not None:
            if not torch.isfinite(grad).all():
                raise FloatingPointError('nonfinite gradient')
            total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def family_norm(named, grads):
    sums = {}
    for (name, _), grad in zip(named, grads):
        if grad is None:
            continue
        if name.startswith('encoder.'):
            family = 'encoder'
        elif name in ('core.a_d', 'core.a_m', 'core.b'):
            family = name.split('.', 1)[1]
        elif name.startswith('core.core.graph_generator.'):
            family = 'graph'
        elif name.startswith(('core.core.gamma_channel_proj.', 'core.core.gamma_phase_gain.')):
            family = 'oscillator_drive'
        elif name.startswith('core.core.kuramoto.'):
            family = 'kuramoto'
        elif name.startswith('core.core.dendric_layer.'):
            family = 'dendrite'
        elif name.startswith('core.core.membrane_layer.'):
            family = 'membrane'
        else:
            family = 'core_other'
        sums[family] = sums.get(family, 0.) + float(grad.detach().double().square().sum())
    return {key: math.sqrt(value) for key, value in sums.items()}


def joint_parameters(core, encoder):
    integration = [core.a_d, core.a_m, core.b]
    integration_ids = {id(p) for p in integration}
    native = [p for p in core.parameters() if p.requires_grad and id(p) not in integration_ids]
    enc = [p for p in encoder.parameters() if p.requires_grad]
    named = [(f'core.{n}', p) for n, p in core.named_parameters() if p.requires_grad]
    named += [(f'encoder.{n}', p) for n, p in encoder.named_parameters() if p.requires_grad]
    groups = [{'params': native, 'lr': CORE_GRAPH_LR},
              {'params': integration, 'lr': INTEGRATION_LR},
              {'params': enc, 'lr': ENCODER_LR}]
    return named, groups


@torch.no_grad()
def verify_zero_initial_parity(seed, gamma, device):
    """Compare native and both zero-initialized arms at T64 and T1024."""
    checkpoint, _, _, _, _, source_sha = source_contract(seed)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    rows = []
    for steps, settle in ((64, 32), (1024, 512)):
        reference = source97.make_core(device, steps).to(device)
        reference.load_state_dict(state, strict=True)
        reference._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        expected = reference(gamma, return_core_out=True, return_theta=True,
                             num_time_steps=steps)
        expected_components = reference.last_component_spikes.detach().clone()
        expected_membrane = reference.last_component_out.detach().clone()
        expected_q = spike_synchrony_affinity(expected_components.mean(dim=1),
            expected_components, settle=settle, affinity_mode='spike')
        expected_labels, _, _ = hard_labels(expected[1], expected_components, settle)
        criterion = make_criterion()
        expected_plv = phase_locking_value(expected[3], settle=settle)
        expected_primary, _ = criterion(plv=expected_plv, theta=expected[3])
        expected_q_loss, _ = criterion(plv=expected_q)
        expected_old = expected_primary + 5. * expected_q_loss
        for arm in ARMS:
            core = source97.make_core(device, steps).to(device)
            core.load_state_dict(state, strict=True)
            core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
            wrapped = PhaseStateIntegration(core, arm)
            actual = wrapped(gamma, return_core_out=True, return_theta=True,
                             num_time_steps=steps)
            if any(not torch.equal(left, right) for left, right in zip(actual[1:], expected[1:])):
                raise AssertionError(f'{arm} zero-initial source parity failed at T={steps}')
            if not torch.equal(core.last_component_spikes, expected_components):
                raise AssertionError(f'{arm} component spikes differ at T={steps}')
            if not torch.equal(core.last_component_out, expected_membrane):
                raise AssertionError(f'{arm} component membrane differs at T={steps}')
            q = spike_synchrony_affinity(core.last_component_spikes.mean(dim=1),
                core.last_component_spikes, settle=settle, affinity_mode='spike')
            if not torch.equal(q, expected_q):
                raise AssertionError(f'{arm} actual-Q mismatch at T={steps}')
            labels, _, _ = hard_labels(actual[1], core.last_component_spikes, settle)
            if not torch.equal(labels, expected_labels):
                raise AssertionError(f'{arm} hard-H mismatch at T={steps}')
            actual_plv = phase_locking_value(actual[3], settle=settle)
            primary, _ = criterion(plv=actual_plv, theta=actual[3])
            if not torch.equal(primary, expected_primary):
                raise AssertionError(f'{arm} phase-primary loss mismatch at T={steps}')
            actual_q_loss, _ = criterion(plv=q)
            if not torch.equal(primary + 5. * actual_q_loss, expected_old):
                raise AssertionError(f'{arm} old objective mismatch at T={steps}')
            rows.append({'arm': arm, 'time_steps': steps, 'settle': settle,
                         'theta_component_membrane_spikes_q_h_primary_oldloss_exact': True})
    return {'source_core_sha256': source_sha, 'checks': rows}


def disposable_update(seed, arm, device, pool, fixed_lambda, warmup_path):
    """Real throwaway paired-objective step from immutable source/warmup state."""
    wrapped, encoder, patcher, mean, std, clip, decoder, _, _, _ = load_models(seed, device, arm)
    warm = torch.load(warmup_path, map_location=device, weights_only=True)
    decoder.load_state_dict(warm['decoder_state_dict'], strict=True)
    integration_parameters = {'a_d': wrapped.a_d, 'a_m': wrapped.a_m, 'b': wrapped.b}
    decoder_params = list(decoder.parameters())
    named, optimizer_groups = joint_parameters(wrapped, encoder)
    joint_params = [p for group in optimizer_groups for p in group['params']]
    joint_ids = [id(p) for p in joint_params]
    if len(joint_ids) != len(set(joint_ids)):
        raise AssertionError('joint optimizer parameter groups contain duplicates')
    named_ids = {id(p) for _, p in named}
    if set(joint_ids) != named_ids:
        raise AssertionError('joint gradient parameters differ from optimizer parameter groups')
    native_ids = {id(p) for p in wrapped.core.parameters() if p.requires_grad}
    encoder_ids = {id(p) for p in encoder.parameters() if p.requires_grad}
    core_params = [p for p in joint_params if id(p) in native_ids or id(p) in {
        id(wrapped.a_d), id(wrapped.a_m), id(wrapped.b)}]
    enc_params = [p for p in joint_params if id(p) in encoder_ids]
    images = read_batch(np.load(TRAIN_RGB, mmap_mode='r'), pool[:BATCH].tolist(), device)
    criterion = make_criterion()
    gamma, result, q, _, hard, _, target = forward_batch(
        wrapped, encoder, patcher, mean, std, clip, images, criterion)
    _, _, _, plv, theta = result
    primary, _ = criterion(plv=plv, theta=theta)
    q_loss, _ = criterion(plv=q)
    old_loss = primary + 5. * q_loss
    _, rgb_loss, _ = rgb_losses(q, hard, gamma, target, decoder, True)
    rgb_grads = torch.autograd.grad(rgb_loss, [p for _, p in named],
                                    retain_graph=True, allow_unused=True)
    rgb_norms = family_norm(named, rgb_grads)
    required = ('encoder', 'graph', 'oscillator_drive', 'kuramoto', 'dendrite',
                'membrane', 'a_d', 'a_m', 'b')
    if any(rgb_norms.get(family, 0.) <= 0 for family in required):
        raise AssertionError(f'{arm} RGB gradient credit missing by family: {rgb_norms}')
    for name, grad in zip((name for name, _ in named), rgb_grads):
        if name in ('core.a_d', 'core.a_m', 'core.b'):
            if grad is None or not torch.isfinite(grad).all() or bool((grad == 0).any()):
                raise AssertionError(f'{arm} RGB gradient missing for individual {name} components')
    total = old_loss + float(fixed_lambda) * rgb_loss
    optim = torch.optim.Adam(optimizer_groups)
    decoder_optim = torch.optim.Adam(decoder_params, lr=DECODER_LR)
    decoder_optim.load_state_dict(warm['optimizer_state_dict'])
    optim.zero_grad(set_to_none=True); decoder_optim.zero_grad(set_to_none=True)
    joint_grads = torch.autograd.grad(total, joint_params, retain_graph=True, allow_unused=True)
    decoder_grads = torch.autograd.grad(rgb_loss, decoder_params, allow_unused=True)
    if grad_norm(joint_grads) <= 0 or grad_norm(decoder_grads) <= 0:
        raise AssertionError(f'{arm} disposable update has an empty gradient')
    if any(g is not None and not torch.isfinite(g).all() for g in joint_grads + decoder_grads):
        raise FloatingPointError(f'{arm} disposable gradients are nonfinite')
    for parameter, grad in zip(joint_params, joint_grads):
        parameter.grad = None if grad is None else grad.detach().clone()
    for parameter, grad in zip(decoder_params, decoder_grads):
        parameter.grad = None if grad is None else grad.detach().clone()
    before_core = {name: p.detach().clone() for name, p in wrapped.core.named_parameters()}
    before_encoder = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    before_decoder = [p.detach().clone() for p in decoder_params]
    before_integration = {name: p.detach().clone() for name, p in integration_parameters.items()}
    torch.nn.utils.clip_grad_norm_(joint_params, CLIP_NORM)
    torch.nn.utils.clip_grad_norm_(decoder_params, CLIP_NORM)
    optim.step(); decoder_optim.step(); wrapped.project_integrations_()
    changed_core = [name for name, p in wrapped.core.named_parameters()
                    if not torch.equal(before_core[name], p)]
    changed_encoder = [name for name, p in encoder.named_parameters()
                       if not torch.equal(before_encoder[name], p)]
    changed_decoder = [str(i) for i, (before, after) in enumerate(zip(before_decoder, decoder_params))
                       if not torch.equal(before, after)]
    changed_integration = [name for name, p in integration_parameters.items()
                           if not torch.equal(before_integration[name], p)]
    required_core_prefixes = ('graph_generator.', 'gamma_channel_proj.', 'kuramoto.',
                              'dendric_layer.', 'membrane_layer.')
    missing_core = [prefix for prefix in required_core_prefixes
                    if not any(name.startswith(prefix) for name in changed_core)]
    if missing_core or not changed_encoder or not changed_decoder:
        raise AssertionError(f'{arm} disposable update failed to change model families: '
                             f'core_missing={missing_core}, encoder={len(changed_encoder)}, '
                             f'decoder={len(changed_decoder)}')
    if 'b' not in changed_integration:
        raise AssertionError(f'{arm} disposable step did not change threshold adaptation: {changed_integration}')
    if any(not torch.isfinite(p).all() for p in list(wrapped.parameters()) +
           list(encoder.parameters()) + decoder_params):
        raise FloatingPointError(f'{arm} disposable update produced nonfinite parameters')
    return {'arm': arm, 'rgb_gradient_norms_by_family': rgb_norms,
            'joint_gradient_norm': grad_norm(joint_grads),
            'decoder_gradient_norm': grad_norm(decoder_grads),
            'changed_native_core_parameter_count': len(changed_core),
            'changed_encoder_parameter_count': len(changed_encoder),
            'changed_decoder_parameter_count': len(changed_decoder),
            'changed_integration_parameters': changed_integration,
            'optimizer_updates': 1, 'throwaway_only': True}


def preflight(seed, device, output):
    if seed not in SEEDS:
        raise ValueError('seed must be 0, 1, or 2')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'preserve existing preflight record: {output}')
    asset_hashes = validate_rgb_assets()
    train_cache = np.load(TRAIN_RGB, mmap_mode='r')
    gamma_cache, gamma_manifest = source97.validate_gamma_cache(GAMMA_TRAIN, GAMMA_TRAIN_MANIFEST)
    wrapped, encoder, patcher, mean, std, clip, decoder, pool, ids, source_sha = load_models(seed, device, 'phase')
    criterion = make_criterion()
    wrapped.train(); encoder.train(); decoder.train()
    named, joint_groups = joint_parameters(wrapped, encoder)
    joint_params = [p for group in joint_groups for p in group['params']]
    decoder_params = list(decoder.parameters())
    initial_core = {k: v.detach().clone() for k, v in wrapped.core.state_dict().items()}
    initial_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    first_ix = pool[:BATCH].tolist()
    first_images = read_batch(train_cache, first_ix, device)
    with torch.no_grad():
        cached_gamma = gamma_cache[torch.as_tensor(pool[:BATCH], dtype=torch.long)].to(device)
        live_gamma = encode(encoder, patcher, mean, std, clip, first_images)
        gamma_diff = float((live_gamma - cached_gamma).abs().max())
    if gamma_diff > 2e-5:
        raise AssertionError(f'live RGB gamma differs from registered cache by {gamma_diff}')
    parity = verify_zero_initial_parity(seed, cached_gamma[:2], device)

    # Decoder warmup uses native-equivalent zero-initialized phase dynamics.
    warm_optimizer = torch.optim.Adam(decoder_params, lr=DECODER_LR)
    warm_losses = []
    core_mode, enc_mode = wrapped.training, encoder.training
    wrapped.eval(); encoder.eval()
    try:
        for step in range(32):
            indices = pool[step * BATCH:(step + 1) * BATCH].tolist()
            images = read_batch(train_cache, indices, device)
            with torch.no_grad():
                gamma, _, q, _, hard, _, target = forward_batch(
                    wrapped, encoder, patcher, mean, std, clip, images, criterion)
            warm_optimizer.zero_grad(set_to_none=True)
            _, reconstruction, _ = rgb_losses(q, hard, gamma.detach(), target, decoder, True)
            if not torch.isfinite(reconstruction):
                raise FloatingPointError('nonfinite decoder warmup reconstruction')
            reconstruction.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params, CLIP_NORM)
            warm_optimizer.step()
            warm_losses.append(float(reconstruction.detach()))
    finally:
        wrapped.train(core_mode); encoder.train(enc_mode)
    if any(not torch.equal(v, initial_core[k]) for k, v in wrapped.core.state_dict().items()):
        raise AssertionError('decoder warmup changed source core')
    if any(not torch.equal(v, initial_encoder[k]) for k, v in encoder.state_dict().items()):
        raise AssertionError('decoder warmup changed source encoder')
    warmup_path = output.parent / f'preflight_decoder_seed{seed}.pt'
    if warmup_path.exists():
        raise FileExistsError(f'preserve existing warmup artifact: {warmup_path}')
    torch.save({'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': warm_optimizer.state_dict()}, warmup_path)
    warmup_sha = sha(warmup_path)

    # Candidate and same-capacity control are initialized identically at theta=0.
    images = read_batch(train_cache, first_ix, device)
    gamma, result, q, labels, hard, _, target = forward_batch(
        wrapped, encoder, patcher, mean, std, clip, images, criterion)
    _, spikes, _, plv, theta = result
    primary, _ = criterion(plv=plv, theta=theta)
    positive_q, _ = criterion(plv=q)
    old_loss = primary + 5. * positive_q
    _, reconstruction, _ = rgb_losses(q, hard, gamma, target, decoder, True)
    if not torch.isfinite(old_loss + reconstruction):
        raise FloatingPointError('nonfinite source initialization loss')
    # Exercise gradients through native core, graph, encoder, and all twelve terms.
    rgb_gradients = torch.autograd.grad(reconstruction, [p for _, p in named],
                                        retain_graph=True, allow_unused=True)
    rgb_family_norms = family_norm(named, rgb_gradients)
    required_rgb_families = ('encoder', 'graph', 'oscillator_drive', 'kuramoto',
                             'dendrite', 'membrane', 'a_d', 'a_m', 'b')
    for family in required_rgb_families:
        if rgb_family_norms.get(family, 0.) <= 0:
            raise AssertionError(f'RGB assignment gradient missing for {family}')
    for name, grad in zip((n for n, _ in named), rgb_gradients):
        if name in ('core.a_d', 'core.a_m', 'core.b'):
            if grad is None or not torch.isfinite(grad).all() or bool((grad == 0).any()):
                raise AssertionError(f'RGB assignment gradient is missing/nonfinite for a component of {name}')

    # Lambda is calibrated once on seed0 from the first four fixed TRAIN batches.
    ratios, calibration = [], []
    for batch_index in range(4):
        indices = pool[batch_index * BATCH:(batch_index + 1) * BATCH].tolist()
        batch = read_batch(train_cache, indices, device)
        gamma_b, result_b, q_b, _, hard_b, _, target_b = forward_batch(
            wrapped, encoder, patcher, mean, std, clip, batch, criterion)
        _, _, _, plv_b, theta_b = result_b
        primary_b, _ = criterion(plv=plv_b, theta=theta_b)
        q_loss_b, _ = criterion(plv=q_b)
        old_b = primary_b + 5. * q_loss_b
        _, rgb_b, _ = rgb_losses(q_b, hard_b, gamma_b, target_b, decoder, True)
        old_grads = torch.autograd.grad(old_b, [p for _, p in named],
                                        retain_graph=True, allow_unused=True)
        rgb_grads = torch.autograd.grad(rgb_b, [p for _, p in named],
                                       retain_graph=False, allow_unused=True)
        old_norm, rgb_norm = grad_norm(old_grads), grad_norm(rgb_grads)
        if old_norm <= 0 or rgb_norm <= 0:
            raise AssertionError(f'lambda calibration batch {batch_index} has an empty joint gradient')
        rgb_families = family_norm(named, rgb_grads)
        if any(rgb_families.get(key, 0.) <= 0 for key in required_rgb_families):
            raise AssertionError(f'lambda calibration batch {batch_index} has inert RGB gradient family')
        ratio = .25 * old_norm / rgb_norm
        ratios.append(ratio)
        calibration.append({'batch': batch_index, 'old_joint_gradient_norm': old_norm,
                            'rgb_joint_gradient_norm': rgb_norm,
                            'rgb_gradient_norm_by_family': rgb_families, 'ratio': ratio})
    if seed == 0:
        fixed_lambda = float(statistics.median(ratios))
        seed0_lambda_record_sha = None
    else:
        seed0_record = output.parent / 'preflight_seed0.json'
        if not seed0_record.is_file():
            raise RuntimeError('seed1/2 preflight requires the passed seed0 lambda record')
        seed0 = json.loads(seed0_record.read_text())
        if (seed0.get('status') != 'passed' or seed0.get('seed') != 0
                or seed0.get('lambda_source_seed') != 0
                or seed0.get('implementation_fingerprint') != implementation_fingerprint()
                or seed0.get('source_core_sha256') != source97.EXPECTED_SOURCE_SHAS[0]
                or seed0.get('training_ids_sha256') != hashlib.sha256(
                    np.asarray(source_contract(0)[4], dtype='<i8').tobytes()).hexdigest()):
            raise RuntimeError('seed0 shared lambda source is invalid')
        if seed0.get('asset_hashes') != asset_hashes:
            raise RuntimeError('seed0 lambda cache/encoder assets differ from current preflight assets')
        warmup = Path(seed0.get('decoder_warmup_artifact', ''))
        if not warmup.is_file() or sha(warmup) != seed0.get('decoder_warmup_artifact_sha256'):
            raise RuntimeError('seed0 warmed decoder artifact is absent or changed')
        fixed_lambda = float(seed0['lambda'])
        seed0_lambda_record_sha = sha(seed0_record)
    if not math.isfinite(fixed_lambda) or fixed_lambda <= 0:
        raise AssertionError('calibrated shared lambda must be finite and positive')

    # All 64 TRAIN images receive the registered within-image row-scramble guard.
    scramble_deltas = []
    with torch.no_grad():
        for batch_index in range(4):
            ix = pool[batch_index * BATCH:(batch_index + 1) * BATCH].tolist()
            images = read_batch(train_cache, ix, device)
            gamma_b, _, q_b, _, hard_b, _, target_b = forward_batch(
                wrapped, encoder, patcher, mean, std, clip, images, criterion)
            for image_index in range(BATCH):
                gen = torch.Generator(device='cpu').manual_seed(130 + batch_index * BATCH + image_index)
                perm = torch.randperm(256, generator=gen).to(device)
                _, base_loss, _ = reconstruct_one(q_b[image_index], hard_b[image_index],
                    gamma_b[image_index].transpose(0, 1), target_b[image_index], decoder, False)
                _, shuffled_loss, _ = reconstruct_one(q_b[image_index], hard_b[image_index][perm],
                    gamma_b[image_index].transpose(0, 1), target_b[image_index], decoder, False)
                scramble_deltas.append(float((shuffled_loss - base_loss).detach()))
    if float(np.mean(scramble_deltas)) <= 0:
        raise AssertionError('hard-H row scrambling did not increase RGB reconstruction loss')

    # Execute separate same-source/same-warmup throwaway updates for both arms.
    disposable = [disposable_update(seed, arm, device, pool, fixed_lambda, warmup_path)
                  for arm in ARMS]

    report = {
        'status': 'passed', 'experiment': 'SW0130_phase_state_integration', 'seed': seed,
        'arm': 'phase', 'device': str(device), 'created_unix': time.time(),
        'implementation_fingerprint': implementation_fingerprint(),
        'source_core_sha256': source_sha, 'source_manifest_sha256': sha(source_contract(seed)[1]),
        'source_updates': int(source_contract(seed)[2]['steps']),
        'matched_training_ids': ids, 'training_ids_sha256': hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(),
        'pool_indices_sha256': hashlib.sha256(np.asarray(pool, dtype='<i8').tobytes()).hexdigest(),
        'shuffle_seed': 117 + seed, 'batch_size': BATCH, 'updates': UPDATES,
        'train_time_steps': TRAIN_STEPS, 'settle_steps': TRAIN_SETTLE,
        'asset_hashes': asset_hashes, 'registered_gamma_cache_max_abs_diff_first_batch': gamma_diff,
        'gamma_train_cache_sha256': gamma_manifest['gamma_sha256'],
        'gamma_train_manifest_sha256': sha(GAMMA_TRAIN_MANIFEST),
        'native_zero_initial_parity': parity,
        'decoder_warmup_batches': 32, 'decoder_warmup_loss_first_last': [warm_losses[0], warm_losses[-1]],
        'decoder_warmup_source_unchanged': True, 'decoder_warmup_artifact': str(warmup_path.resolve()),
        'decoder_warmup_artifact_sha256': warmup_sha,
        'rgb_gradient_norm_by_family_first_batch': rgb_family_norms,
        'lambda_source_seed': 0, 'lambda': fixed_lambda,
        'seed0_lambda_record_sha256': seed0_lambda_record_sha,
        'lambda_seed0_calibration': calibration if seed == 0 else 'reused immutable seed0 preflight calibration',
        'row_scramble_count': len(scramble_deltas), 'row_scramble_mean_excess': float(np.mean(scramble_deltas)),
        'row_scramble_positive_count': int(sum(value > 0 for value in scramble_deltas)),
        'disposable_updates': disposable,
        'disposable_update_finite': all(item['optimizer_updates'] == 1 for item in disposable),
        'ground_truth_used': False,
        'original_core_checkpoint_modified': False,
    }
    write_once(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight',), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == 'preflight':
        result = preflight(args.seed, torch.device(args.device), args.output)
    else:
        raise AssertionError('unsupported stage')
    print(json.dumps({'status': result['status'], 'stage': args.stage, 'seed': args.seed,
                      'output': str(args.output), 'lambda': result['lambda']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
