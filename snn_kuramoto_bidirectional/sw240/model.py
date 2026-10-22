"""SW0240, including its actual encoder, integration wrapper and native64 readout."""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn

from ..hyperparameter import S2NetHyperparameters
from ..input_layer_generator import CNNFeatureEncoder
from ..gamma_initializer import FeaturePatchGammaInitializer
from ..s2net_cls import S2NetCore
from .context import ContextualFeatureEncoder
from .integration import PhaseStateIntegration
from .encoding import encode64
from .geodesic import geodesic_distance_power2, inverse_softplus
from .rollout import late_rollout
from .capture import capture_source_coupling
from .correlation import native_component_rho_q
from .readout import prediction_from_trace
from .loss import effective_frequency, make_teacher, teacher_objective


def default_config():
    """Architecture configuration; trained weights and statistics come from a bundle."""
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=4096, osc_dim=4,
        gamma_drive_mode='static', num_time_steps=64,
        gamma_phase_mode='standardize_tanh', theta_init='gamma',
        graph_mode='learned', graph_top_k=512, graph_spatial_decay=.35,
        geodesic_steps=3, geodesic_radius=1.5, geodesic_contrast=2.,
        geodesic_temperature=.5, geodesic_cap=16., kuramoto_backend='factorized',
        k=4096., freq_gain=2., membrane_vth=.06, membrane_low_m=-4.,
        membrane_high_m=0., low_n=-4., high_n=0., branch=4,
        gate_mode='raw', spike_per_component=True, dendritic_projection='shared',
        spike_spatial_grid_size=64, gamma_patch_grid_size=64,
    ).validate()
    return dict(format_version=1, model='SW0240', grid_size=64, total_steps=1024,
                settle_steps=512, integration_arm='phase', hparams=vars(hp),
                runtime_attributes={}, feature_clip=3., mask_supervised=True,
                geodesic=dict(effective_contrast=8., squared_dissimilarity=True, row_chunk=1))


def model_sha256(model):
    """The historical full-model digest, unchanged by the package extraction."""
    digest = hashlib.sha256()
    for section, module in (('wrapped', model.wrapped), ('encoder', model.encoder)):
        for name, value in sorted(module.state_dict().items()):
            digest.update((section + '.' + name).encode('utf-8'))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SW240Model(nn.Module):
    def __init__(self, config=None, *, feature_mean=None, feature_std=None):
        super().__init__()
        self.config = copy.deepcopy(default_config() if config is None else config)
        if (self.config['format_version'] != 1 or self.config['model'] != 'SW0240'
                or self.config['grid_size'] != 64):
            raise ValueError('this package requires the SW0240 native64 configuration')
        values = self.config['hparams']
        known = {field.name for field in fields(S2NetHyperparameters)}
        if set(values) - known - {'graph_spatial_grid_size'}:
            raise ValueError('unknown exported hyperparameter')
        hp = S2NetHyperparameters(**{k:v for k,v in values.items() if k in known}).validate()
        if hp.num_regions != 4096 or hp.graph_top_k != 512:
            raise ValueError('SW0240 requires 4096 oscillators and graph top-k 512')
        core = S2NetCore(hp, device='cpu')
        self.wrapped = PhaseStateIntegration(core, self.config['integration_arm'])
        self.encoder = ContextualFeatureEncoder(CNNFeatureEncoder(8, 3, 3, bias=True))
        self.patcher = FeaturePatchGammaInitializer(grid_size=64)
        self.register_buffer('feature_mean', torch.as_tensor(
            torch.zeros(1, 8, 1, 1) if feature_mean is None else feature_mean).clone())
        self.register_buffer('feature_std', torch.as_tensor(
            torch.ones(1, 8, 1, 1) if feature_std is None else feature_std).clone())
        self.feature_clip = float(self.config['feature_clip'])
        self.statistics_loaded = feature_mean is not None and feature_std is not None
        modules = dict(self.wrapped.named_modules())
        for name, attrs in self.config.get('runtime_attributes', {}).items():
            if name not in modules:
                raise ValueError('unknown exported module: ' + name)
            for key, value in attrs.items():
                if not hasattr(modules[name], key) or isinstance(getattr(modules[name], key), (nn.Module, torch.Tensor)):
                    raise ValueError('invalid exported runtime attribute: ' + name + '.' + key)
                setattr(modules[name], key, value)
        settings = self.config['geodesic']
        if settings != dict(effective_contrast=8., squared_dissimilarity=True, row_chunk=1):
            raise ValueError('SW0240 requires the original fixed-8 squared geodesic operator')
        def distance(graph, z, euclidean):
            fixed = torch.as_tensor(inverse_softplus(8.), device=z.device, dtype=z.dtype)
            return geodesic_distance_power2(z, euclidean, fixed, steps=graph.geodesic_steps,
                radius=graph.geodesic_radius, temperature=graph.geodesic_temperature,
                cap=graph.geodesic_cap, chunk=1, log4=True)
        core.graph_generator._geodesic_distance = types.MethodType(distance, core.graph_generator)
        # Construction alone is not a trained SW0240 checkpoint.
        self.checkpoint_loaded = False

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        for module in self.wrapped.modules():
            if hasattr(module, 'device'):
                module.device = device
        return result

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        device = next(self.parameters()).device
        for module in self.wrapped.modules():
            if hasattr(module, 'device'):
                module.device = device
        return result

    def encode(self, rgb):
        if not self.statistics_loaded:
            raise RuntimeError('load a portable checkpoint with the original feature statistics first')
        return encode64(self.encoder, self.patcher, self.feature_mean, self.feature_std, self.feature_clip, rgb)

    def forward(self, rgb, *, live_tail_steps=0):
        """Return the actual gamma, spikes, coupling and Q; no ground truth is accepted."""
        gamma = self.encode(rgb)
        trace, adjacency = capture_source_coupling(self.wrapped.core.kuramoto,
            lambda: late_rollout(self.wrapped, gamma, total_steps=self.config['total_steps'],
                                 live_tail_steps=live_tail_steps))
        rho, q = native_component_rho_q(trace['spikes'], trace['component_spikes'],
                                       settle=self.config['settle_steps'])
        if not all(bool(torch.isfinite(v).all()) for v in
                   (gamma, adjacency, q, rho, *[v for v in trace.values() if torch.is_tensor(v)])):
            raise FloatingPointError('nonfinite SW0240 forward output')
        return dict(gamma=gamma, trace=trace, adjacency=adjacency, q=q, rho=rho)

    @torch.no_grad()
    def predict(self, rgb, *, inference_seeds=None):
        """One image at a time, matching the fixed320 evaluation's RNG/order."""
        if not self.checkpoint_loaded:
            raise RuntimeError('predict requires a trained portable SW0240 checkpoint')
        self.eval()
        rgb = torch.as_tensor(rgb)
        seeds = [193000 + i for i in range(len(rgb))] if inference_seeds is None else list(inference_seeds)
        if len(seeds) != len(rgb):
            raise ValueError('one inference seed per image is required')
        labels = []
        for image, seed in zip(rgb, seeds):
            torch.manual_seed(int(seed))
            if next(self.parameters()).is_cuda:
                torch.cuda.manual_seed(int(seed))
            output = self(image.unsqueeze(0))
            labels.append(prediction_from_trace(self, output['trace'], output['adjacency'], output['q']))
        return np.stack(labels)

    def optimizer_for_training(self):
        """Exact SW0249/240 adaptation groups; freeze all other core parameters."""
        self.wrapped.requires_grad_(False)
        self.wrapped.eval()
        self.encoder.requires_grad_(True)
        self.wrapped.core.gamma_channel_proj.requires_grad_(True)
        return torch.optim.Adam([
            dict(name='source_base', params=list(self.encoder.base_encoder.parameters()), lr=3e-5),
            dict(name='context_encoder', params=list(self.encoder.context.parameters()), lr=3e-4),
            dict(name='sensory_projection', params=list(self.wrapped.core.gamma_channel_proj.parameters()), lr=3e-4)])

    def supervised_loss(self, rgb, labels, teacher):
        """Original SW0240 teacher denoising + gamma anchor; no new objective."""
        with torch.no_grad():
            tg = teacher.encode(torch.as_tensor(rgb).cpu())
            tf = effective_frequency(teacher.wrapped.core, tg)
        targets = make_teacher(tg.cpu().numpy(), tf.cpu().numpy(), labels)
        gamma = self.encode(torch.as_tensor(rgb))
        frequency = effective_frequency(self.wrapped.core, gamma)
        return teacher_objective(gamma, frequency, targets)


def load_model(checkpoint, *, device='cpu', teacher=False):
    """Load a tensor-only export; legacy boundary.ptx requires the one-time exporter."""
    source = checkpoint if hasattr(checkpoint, 'read') else Path(checkpoint)
    bundle = torch.load(source, map_location='cpu', weights_only=True)
    model = SW240Model(bundle['config'], feature_mean=bundle['feature_mean'], feature_std=bundle['feature_std'])
    model.wrapped.load_state_dict(bundle['wrapped_state'], strict=True)
    model.encoder.load_state_dict(bundle['encoder_state'], strict=True)
    if teacher:
        model.encoder.load_state_dict(bundle['teacher_encoder_state'], strict=True)
        model.wrapped.core.gamma_channel_proj.load_state_dict(bundle['teacher_projection_state'], strict=True)
    expected = bundle['teacher_model_sha256'] if teacher else bundle['config']['model_sha256']
    if model_sha256(model) != expected:
        raise ValueError('restored full-model SHA does not match the exported SW0240 model')
    model.checkpoint_loaded = True
    model.eval()
    return model.to(device)
