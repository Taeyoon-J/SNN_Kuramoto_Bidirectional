"""Opt-in phase-conditioned dendrite/membrane state integration.

The production S2NetCore remains untouched. Callers must strict-load the
original source state into ``core`` before wrapping it here.
"""
from __future__ import annotations

import torch
from torch import nn

from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating


COMPONENTS = 4
BASE_THRESHOLD = 0.06


class PhaseStateIntegration(nn.Module):
    """Wrap a strict-loaded native core with four-component integration terms.

    ``arm='phase'`` uses the native continuous gate as integration state ``u``;
    ``arm='constant'`` uses ``u=0.5`` while retaining the exact native gate for
    the carrier, membrane hold predicate, and emitted spike multiplier.
    """

    def __init__(self, core: nn.Module, arm: str):
        super().__init__()
        if arm not in ('phase', 'constant'):
            raise ValueError("arm must be 'phase' or 'constant'")
        self._validate_core(core)
        self.core = core
        self.arm = arm
        device = next(core.parameters()).device
        self.a_d = nn.Parameter(torch.zeros(COMPONENTS, device=device))
        self.a_m = nn.Parameter(torch.zeros(COMPONENTS, device=device))
        self.b = nn.Parameter(torch.zeros(COMPONENTS, device=device))
        self.last_state_trace = None

    @staticmethod
    def _validate_core(core):
        if int(getattr(core, 'osc_dim', -1)) != COMPONENTS:
            raise ValueError('SW0130 requires exactly four oscillator components')
        if not bool(getattr(core, 'spike_per_component', False)):
            raise ValueError('SW0130 requires the native per-component spike layout')
        if getattr(core, 'gamma_drive_mode', None) != 'static':
            raise ValueError('SW0130 supports only the registered static-gamma source')
        graph = getattr(core, 'graph_generator', None)
        if graph is not None and bool(getattr(graph, 'uses_feedback', False)):
            raise ValueError('SW0130 does not support feedback graph dynamics')
        if getattr(core.kuramoto, 'spike_pulse_gain', None) is not None:
            raise ValueError('SW0130 requires the registered no-spike-pulse source')
        if abs(float(core.membrane_layer.vth) - BASE_THRESHOLD) > 1e-12:
            raise ValueError('SW0130 requires source membrane threshold 0.06')

    @staticmethod
    def _dense_dendrite(layer, gamma_wave, prev_spike):
        k_input = torch.cat((gamma_wave.float(), prev_spike.unsqueeze(-1)), dim=-1)
        if layer.dendritic_projection == 'shared':
            return layer.oscillator_dense(k_input)
        dense = torch.einsum('bni,nri->bnr', k_input, layer.oscillator_dense_weight)
        if layer.oscillator_dense_bias is not None:
            dense = dense + layer.oscillator_dense_bias
        return dense

    def forward(self, gamma_seq, return_core_out=False, num_time_steps=None,
                return_theta=False, graph_override=None, capture_state=False):
        core = self.core
        gamma_seq = gamma_seq.to(core.device)
        if gamma_seq.ndim != 3 or gamma_seq.shape[1:] != (core.T, core.in_dim):
            raise ValueError(f'expected static gamma [B,{core.T},{core.in_dim}]')
        batch = gamma_seq.size(0)
        if graph_override is not None:
            expected = (batch, core.in_dim, core.in_dim)
            if tuple(graph_override.shape) != expected or not torch.isfinite(graph_override).all() or (graph_override < 0).any():
                raise ValueError('graph_override must be finite, nonnegative, and have registered shape')
            graph = graph_override.to(device=gamma_seq.device, dtype=gamma_seq.dtype)
        elif core.graph_generator is not None:
            graph = core.graph_generator(gamma_seq)
        else:
            graph = core.sc.to(gamma_seq.device).unsqueeze(0).expand(batch, -1, -1)

        steps = int(num_time_steps if num_time_steps is not None else core.num_time_steps)
        if steps < 1:
            raise ValueError('num_time_steps must be positive')
        drive = core.gamma_to_drive(gamma_seq, core.gamma_channel_proj, core.gamma_phase_gain)
        theta = core._init_theta(drive, batch)
        fold = COMPONENTS
        core.dendric_layer.set_neuron_state(batch * fold)
        core.membrane_layer.set_neuron_state(batch * fold)
        coupling = None
        if core.kuramoto_backend == 'factorized':
            coupling = core.kuramoto.prepare_coupling(
                graph, batch_size=batch, num_units=core.in_dim, device=gamma_seq.device)

        theta_history = []
        membrane_frames, spike_frames = [], []
        trace = {key: [] for key in ('gate', 'integration_gate', 'carrier', 'native_h',
                                      'integrated_h', 'native_membrane', 'integrated_membrane',
                                      'events', 'emitted_spikes')} if capture_state else None
        dendrite = core.dendric_layer
        membrane = core.membrane_layer

        for t in range(steps):
            theta = core.kuramoto(theta, drive, A=graph, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = sinusoidal_gating(theta_history, t, core.phase_delay_steps,
                                               gate_mode=core.gate_mode)
            folded_carrier = carrier.permute(0, 2, 1).reshape(batch * fold, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(fold, dim=0)
            integration_gate = folded_gate if self.arm == 'phase' else torch.full_like(folded_gate, 0.5)

            prev_h = dendrite.h
            prev_mem = membrane.mem
            prev_spike = membrane.spike
            dense = self._dense_dendrite(dendrite, folded_carrier, prev_spike)
            beta_d = torch.sigmoid(dendrite.tau_n).unsqueeze(0)
            native_h = beta_d * prev_h + (1.0 - beta_d) * dense
            component_ids = torch.arange(fold, device=gamma_seq.device).repeat(batch)
            a_d = self.a_d[component_ids].view(batch * fold, 1, 1)
            u = integration_gate.unsqueeze(-1)
            integrated_h = native_h + a_d * (u - 1.0) * (native_h - prev_h)
            dendrite.h = integrated_h
            h_wave = integrated_h.sum(dim=2)

            alpha = torch.sigmoid(membrane.tau_m).unsqueeze(0)
            b = self.b[component_ids].view(batch * fold, 1)
            threshold = BASE_THRESHOLD * torch.exp(b)
            native_mem = (prev_mem * alpha + (1.0 - alpha) * R_m * h_wave
                          - threshold * prev_spike)
            a_m = self.a_m[component_ids].view(batch * fold, 1)
            integrated_mem = native_mem + a_m * (integration_gate - 1.0) * (native_mem - prev_mem)
            integrated_mem = torch.where(folded_gate == 0, prev_mem, integrated_mem)
            events = act_fun_adp(integrated_mem - threshold)
            emitted = events * folded_gate
            membrane.mem, membrane.spike = integrated_mem, emitted
            membrane_frames.append(integrated_mem)
            spike_frames.append(emitted)
            if capture_state:
                trace['gate'].append(folded_gate)
                trace['integration_gate'].append(integration_gate)
                trace['carrier'].append(folded_carrier)
                trace['native_h'].append(native_h)
                trace['integrated_h'].append(integrated_h)
                trace['native_membrane'].append(native_mem)
                trace['integrated_membrane'].append(integrated_mem)
                trace['events'].append(events)
                trace['emitted_spikes'].append(emitted)

        # Match native stack/permute/reshape layout and reduction order exactly.
        component_out = torch.stack(membrane_frames).permute(1, 2, 0).reshape(batch, fold, core.in_dim, steps)
        component_spikes = torch.stack(spike_frames).permute(1, 2, 0).reshape(batch, fold, core.in_dim, steps)
        core_out = component_out.mean(dim=1)
        spikes = component_spikes.mean(dim=1)
        core.last_component_out = component_out
        core.last_component_spikes = component_spikes
        core.last_gate_history = (torch.stack(trace['gate'], dim=-1).reshape(batch, fold, core.in_dim, steps)
                                  if capture_state else None)
        groups = core._detect_object_groups(core_out, spikes)
        theta_stack = torch.stack(theta_history, dim=1)
        self.last_state_trace = ({key: torch.stack(values, dim=-1) for key, values in trace.items()}
                                 if capture_state else None)
        if return_theta:
            if return_core_out:
                return groups, spikes, core_out, theta_stack
            return groups, spikes, theta_stack
        if return_core_out:
            return groups, spikes, core_out
        return groups, spikes

    @torch.no_grad()
    def project_integrations_(self):
        self.a_d.clamp_(0.0, 1.0)
        self.a_m.clamp_(0.0, 1.0)


def strict_load_then_attach(core: nn.Module, source_state: dict, arm: str) -> PhaseStateIntegration:
    """Load the unwrapped source checkpoint strictly before adding SW0130 keys."""
    result = core.load_state_dict(source_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('strict source-core load unexpectedly returned incompatible keys')
    return PhaseStateIntegration(core, arm)

