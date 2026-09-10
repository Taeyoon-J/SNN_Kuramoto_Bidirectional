import math

import torch
import torch.nn as nn
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating
from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeatureMapCNNEncoder, FeaturePatchGammaInitializer
from graph_generator import ImageConditionedGraph
from cluster_readout import ClusterReadout
from gamma_ordering import order_gammas
from spike_classifier import spike_interval, spike_rhythm, spike_spatial_components

class GammaGenerator(nn.Module):
    """Generate gamma sequences from input images."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        self.T = int(hparams.num_feature_maps)
        self.in_dim = int(hparams.num_regions)
        self.device = device
        self.gamma_mode = hparams.gamma_mode

        self.input_layer = CNNFeatureEncoder(
            num_kernels=self.T,
            kernel_size=hparams.kernel_size,
            in_channels=hparams.in_channels,
            bias=True,
        )
        if self.gamma_mode == "patch":
            self.gamma_initializer = FeaturePatchGammaInitializer(
                grid_size=hparams.gamma_patch_grid_size,
                patch_size=hparams.gamma_patch_size,
                stride=hparams.gamma_patch_stride,
                reduction=hparams.gamma_patch_reduction,
            )
        else:
            self.gamma_initializer = FeatureMapCNNEncoder(
                num_osci=self.in_dim,
                in_channels=1,
                dropout=hparams.gamma_dropout,
            )

    def forward(self, x):
        x = x.to(self.device)
        feature_maps = self._image_to_feature_maps(x)
        if self.gamma_mode == "patch":
            gamma_seq = self.gamma_initializer(feature_maps)
            if gamma_seq.size(-1) != self.in_dim:
                raise ValueError(
                    f"Patch gamma produced {gamma_seq.size(-1)} oscillators, "
                    f"but hparams.num_regions is {self.in_dim}."
                )
            return gamma_seq
        B, T, height, width = feature_maps.shape
        return self.gamma_initializer(
            feature_maps.reshape(B * T, 1, height, width)
        ).view(B, T, self.in_dim)

    def _image_to_feature_maps(self, x):
        if x.dim() != 4:
            raise ValueError("x must have shape [B, 3, H, W]. Use B=1 for one image.")
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, but got {x.size(1)}.")

        return self.input_layer(x)


class S2NetCore(nn.Module):
    """Classifier core driven by externally generated gamma sequences."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        self.T = int(hparams.num_feature_maps)
        self.in_dim = int(hparams.num_regions)
        self.osc_dim = int(getattr(hparams, "osc_dim", 4))
        self.phase_delay_steps = 2
        self.device = device
        self.gamma_drive_mode = getattr(hparams, "gamma_drive_mode", "sequence")
        self.num_time_steps = int(getattr(hparams, "num_time_steps", self.T))
        self.gamma_phase_mode = getattr(hparams, "gamma_phase_mode", "none")
        self.theta_init = getattr(hparams, "theta_init", "zeros")
        self.theta_init_noise = float(getattr(hparams, "theta_init_noise", 0.0))
        self.gate_mode = getattr(hparams, "gate_mode", "sigmoid")
        self.spike_classify_method = hparams.spike_classify_method
        self.spike_rhythm_threshold = hparams.spike_rhythm_threshold
        self.spike_rhythm_min_group_size = hparams.spike_rhythm_min_group_size
        self.spike_rhythm_return_all_groups = hparams.spike_rhythm_return_all_groups
        self.spike_interval_size = hparams.spike_interval_size
        self.spike_interval_threshold = hparams.spike_interval_threshold
        self.spike_interval_min_group_size = hparams.spike_interval_min_group_size
        self.spike_interval_include_partial = hparams.spike_interval_include_partial
        self.spike_spatial_grid_size = hparams.spike_spatial_grid_size
        self.spike_spatial_threshold = hparams.spike_spatial_threshold
        self.spike_spatial_min_group_size = hparams.spike_spatial_min_group_size
        self.spike_spatial_activity_source = hparams.spike_spatial_activity_source
        self.spike_spatial_time_aggregate = hparams.spike_spatial_time_aggregate

        graph_mode = getattr(hparams, "graph_mode", "static")
        if hparams.sc is None:
            if graph_mode != "learned":
                raise ValueError(
                    "hparams.sc must be generated before constructing S2NetCore, "
                    'or set graph_mode="learned" to build the graph per image.'
                )
            self.register_buffer("sc", torch.eye(self.in_dim))
        else:
            sc = torch.as_tensor(hparams.sc, dtype=torch.float32)
            expected_shape = (self.in_dim, self.in_dim)
            if tuple(sc.shape) != expected_shape:
                raise ValueError(
                    f"hparams.sc must have shape {expected_shape}, but got {tuple(sc.shape)}."
                )
            self.register_buffer("sc", sc.clone().detach())

        # Extra parameters are created only when the corresponding non-default
        # mode is requested, so a legacy configuration keeps an unchanged
        # state_dict and old checkpoints still load with strict=True.
        self.gamma_channel_proj = None
        if self.gamma_drive_mode == "static":
            self.gamma_channel_proj = nn.Linear(self.T, self.osc_dim, bias=False)
            nn.init.normal_(self.gamma_channel_proj.weight, std=1.0 / max(self.T, 1) ** 0.5)

        self.gamma_phase_gain = None
        if self.gamma_phase_mode != "none":
            self.gamma_phase_gain = nn.Parameter(torch.ones(1))

        # A fixed SC cannot express object membership, which changes per image.
        # In "learned" mode the coupling graph is produced from the sample.
        self.graph_mode = getattr(hparams, "graph_mode", "static")
        self.graph_generator = None
        if self.graph_mode == "learned":
            self.graph_generator = ImageConditionedGraph(
                in_channels=self.T,
                hidden_dim=int(getattr(hparams, "graph_hidden_dim", 16)),
                top_k=int(getattr(hparams, "graph_top_k", 8)),
                coupling_gain=float(getattr(hparams, "graph_coupling_gain", 8.0)),
                temperature=float(getattr(hparams, "graph_temperature", 0.1)),
                grid_size=getattr(hparams, "spike_spatial_grid_size", None),
                spatial_decay=getattr(hparams, "graph_spatial_decay", None),
                feedback_strength=float(getattr(hparams, "graph_feedback_strength", 0.0)),
                feedback_momentum=float(getattr(hparams, "graph_feedback_momentum", 0.9)),
            )

        self.kuramoto = graphVectorKuramoto(
            N=self.in_dim, D=self.osc_dim, K=hparams.k, dt=hparams.dt, alpha_scale=1.0,
            device=device, freq_gain=float(getattr(hparams, "freq_gain", 0.0)),
            spike_pulse_gain=float(getattr(hparams, "spike_pulse_gain", 0.0)),
        )

        self.dendric_layer = DendricLayer(
            input_dim=self.in_dim,
            output_dim=self.in_dim,
            tau_ninitializer='uniform',
            low_n=hparams.low_n,
            high_n=hparams.high_n,
            branch=hparams.branch,
            device=device,
            bias=True,
            # phase_mean hands the dendrite one reduced oscillation per unit
            input_vector_dim=1 if self.gate_mode == "phase_mean" else self.osc_dim,
        )
        self.membrane_layer = MembraneLayer(
            output_dim=self.in_dim,
            tau_minitializer='uniform',
            low_m=float(getattr(hparams, "membrane_low_m", 0.0)),
            high_m=float(getattr(hparams, "membrane_high_m", 4.0)),
            vth=float(getattr(hparams, "membrane_vth", 0.5)),
            dt=1,
            device=device
        )
        self.gate_mode = getattr(hparams, "gate_mode", "sigmoid")

        # The readout that produces the grouping. Built only when asked for, so
        # a checkpoint trained without it still loads with strict=True.
        self.cluster_readout = None
        if int(getattr(hparams, "readout_slots", 0)) > 0:
            self.cluster_readout = ClusterReadout(
                num_slots=int(hparams.readout_slots),
                embed_dim=int(getattr(hparams, "readout_embed_dim", 16)),
                num_iters=int(getattr(hparams, "readout_iters", 3)),
                temperature=float(getattr(hparams, "readout_temperature", 0.5)),
                feature_source=getattr(hparams, "readout_source", "phase"),
                signal_dim=getattr(hparams, "readout_signal_dim", None),
            )

    def forward(
        self,
        gamma_seq,
        return_core_out=False,
        num_time_steps=None,
        return_theta=False,
    ):
        """
        Run the Kuramoto/SNN core.

        In "sequence" drive mode gamma_seq is [B, T, num_regions] and one
        Kuramoto step is taken per visual channel, so T is the channel count.

        In "static" drive mode gamma is a constant sensory drive and the
        recurrent length is decoupled from the channel count:

            [B, num_regions]                -> scalar drive
            [B, num_regions, osc_dim]       -> vector drive
            [B, C, num_regions]             -> channel-projected to [B, N, osc_dim]

        The last form lets already-stored gamma_seq tensors be reused without
        regenerating them.
        """
        gamma_seq = gamma_seq.to(self.device)
        B = gamma_seq.size(0)
        feedback = self.graph_generator is not None and self.graph_generator.uses_feedback
        if self.graph_generator is not None:
            sc = self.graph_generator(gamma_seq)
        else:
            sc = self.sc.to(gamma_seq.device).unsqueeze(0).expand(B, -1, -1)

        if self.gamma_drive_mode == "sequence":
            if gamma_seq.dim() != 3:
                raise ValueError(
                    "gamma_seq must have shape [B, T, num_regions] in sequence drive mode. "
                    "Use B=1 for one sample."
                )
            if num_time_steps is not None:
                raise ValueError(
                    "num_time_steps only applies to static drive mode; in sequence mode "
                    "the recurrent length is fixed by the gamma channel axis."
                )
            drive_seq = self._to_phase(gamma_seq)
            T = drive_seq.size(1)
            drive = None
            theta = self._init_theta(drive_seq[:, 0, :], B)
        else:
            drive_seq = None
            drive = self._to_phase(self._static_drive(gamma_seq))
            T = int(num_time_steps if num_time_steps is not None else self.num_time_steps)
            if T <= 0:
                raise ValueError("num_time_steps must be positive.")
            theta = self._init_theta(drive, B)

        # Oscillators and neurons advance in one interleaved loop so that spikes
        # from step t can act back on the phases at step t+1. Gating only ever
        # looks at theta[t] and theta[t - phase_delay_steps], both already
        # available, so this reproduces the two-loop version exactly whenever
        # pulse coupling is off.
        alignment = (
            self.graph_generator.initial_alignment(B, self.in_dim, theta.device)
            if feedback else None
        )
        self.dendric_layer.set_neuron_state(B)
        self.membrane_layer.set_neuron_state(B)
        pulse_enabled = self.kuramoto.spike_pulse_gain is not None

        # The graph is fixed for the whole rollout unless feedback is on, so the
        # lag kernels are built once. Rebuilding them per step kept T copies of
        # a [B, N, N] pair in the autograd graph for no gain.
        coupling = None if feedback else self.kuramoto.prepare_coupling(
            sc, batch_size=B, num_units=self.in_dim, device=gamma_seq.device
        )

        theta_hist = []
        outputs = []
        spikes_hist = []
        for t in range(T):
            if feedback:
                # Oscillators that have stayed in phase couple more strongly.
                # A graph fixed for the whole rollout cannot sharpen a group
                # once it starts forming; this is what lets one crystallise.
                sc = self.graph_generator(gamma_seq, alignment=alignment)
            drive_t = drive if drive_seq is None else drive_seq[:, t, :]
            theta = self.kuramoto(
                theta,
                drive_t,
                A=sc,
                spike=self.membrane_layer.spike if pulse_enabled else None,
                coupling=coupling,
            )
            if feedback:
                alignment = self.graph_generator.update_alignment(alignment, theta)
            theta_hist.append(theta)

            delayed = theta_hist[max(0, t - self.phase_delay_steps)]
            mask = 0.5 * (1.0 + torch.sin(delayed.mean(dim=-1)))
            if self.gate_mode == "phase_mean":
                gamma_wave_t = torch.sin(theta.mean(dim=-1)).unsqueeze(-1)
            else:
                gamma_wave_t = torch.sin(theta) * mask.unsqueeze(-1)
            g_wave_t = torch.sigmoid(mask) if self.gate_mode == "sigmoid" else mask

            h_wave_t = self.dendric_layer(gamma_wave_t, self.membrane_layer.spike)
            mem_t, spike_t = self.membrane_layer(h_wave_t, g_wave_t)
            spikes_hist.append(spike_t)
            outputs.append(mem_t)

        core_out = torch.stack(outputs).permute(1, 2, 0)
        spikes = torch.stack(spikes_hist).permute(1, 2, 0)
        object_groups = self._detect_object_groups(core_out, spikes)

        if return_theta:
            # [B, T, num_regions, osc_dim]
            theta_stack = torch.stack(theta_hist, dim=1)
            if return_core_out:
                return object_groups, spikes, core_out, theta_stack
            return object_groups, spikes, theta_stack
        if return_core_out:
            return object_groups, spikes, core_out
        return object_groups, spikes

    def _static_drive(self, gamma):
        """Normalize any accepted static-drive layout to [B, N, D] or [B, N]."""
        if gamma.dim() == 2:
            if gamma.size(1) != self.in_dim:
                raise ValueError(
                    f"gamma has {gamma.size(1)} oscillators, but num_regions is {self.in_dim}."
                )
            return gamma
        if gamma.dim() != 3:
            raise ValueError(
                "static gamma must have shape [B, N], [B, N, osc_dim], or [B, C, N]."
            )
        # [B, C, N]: the oscillator axis is last, so project channels to osc_dim.
        if gamma.size(-1) == self.in_dim:
            if self.gamma_channel_proj is None:
                raise ValueError("gamma_channel_proj is unavailable; rebuild the core in static mode.")
            if gamma.size(1) != self.T:
                raise ValueError(
                    f"gamma has {gamma.size(1)} channels, but num_feature_maps is {self.T}."
                )
            return self.gamma_channel_proj(gamma.transpose(1, 2))  # [B, N, osc_dim]
        if gamma.size(1) == self.in_dim and gamma.size(-1) == self.osc_dim:
            return gamma
        raise ValueError(
            f"Cannot interpret static gamma of shape {tuple(gamma.shape)} with "
            f"num_regions={self.in_dim}, num_feature_maps={self.T}, osc_dim={self.osc_dim}."
        )

    def _to_phase(self, gamma):
        """Map raw gamma onto a phase range so sin(gamma - theta) does not wrap."""
        if self.gamma_phase_mode == "none":
            return gamma
        if self.gamma_phase_mode == "standardize_tanh":
            dims = tuple(range(1, gamma.dim()))
            mean = gamma.mean(dim=dims, keepdim=True)
            std = gamma.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-6)
            gamma = (gamma - mean) / std
        return math.pi * torch.tanh(self.gamma_phase_gain * gamma)

    def _init_theta(self, drive, batch_size):
        """Initial oscillator phase [B, num_regions, osc_dim]."""
        if self.theta_init == "zeros":
            theta = torch.zeros(batch_size, self.in_dim, self.osc_dim, device=self.device)
        else:
            theta = drive if drive.dim() == 3 else drive.unsqueeze(-1)
            theta = theta.expand(batch_size, self.in_dim, self.osc_dim).contiguous()
            if self.theta_init == "gamma_noise":
                noise = self.theta_init_noise if self.theta_init_noise > 0 else 0.01
                theta = theta + noise * torch.randn_like(theta)
        return theta

    def _detect_object_groups(self, core_out, spikes):
        if self.spike_classify_method == "spike_rhythm":
            return spike_rhythm(
                spikes,
                threshold=self.spike_rhythm_threshold,
                min_group_size=self.spike_rhythm_min_group_size,
                return_all_groups=self.spike_rhythm_return_all_groups,
            )
        if self.spike_classify_method == "spike_interval":
            return spike_interval(
                core_out,
                interval_size=self.spike_interval_size,
                threshold=self.spike_interval_threshold,
                min_group_size=self.spike_interval_min_group_size,
                include_partial=self.spike_interval_include_partial,
            )
        if self.spike_classify_method == "spatial_components":
            if self.spike_spatial_activity_source == "spikes":
                activity = spikes
            elif self.spike_spatial_activity_source == "membrane":
                activity = core_out
            else:
                activity = torch.sigmoid(core_out)
            return spike_spatial_components(
                activity,
                patch_grid_size=self.spike_spatial_grid_size,
                threshold=self.spike_spatial_threshold,
                min_group_size=self.spike_spatial_min_group_size,
                activity_source=self.spike_spatial_activity_source,
                time_aggregate=self.spike_spatial_time_aggregate,
            )
        raise ValueError(f"Unsupported spike_classify_method: {self.spike_classify_method}")


class S2NetClassifier(nn.Module):
    """End-to-end wrapper: input image -> gamma sequence -> ordered classifier core."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        hparams.validate()
        self.hparams = hparams
        self.device = device
        self.gamma_order_lambda = hparams.gamma_order_lambda
        self.gamma_order_mu = hparams.gamma_order_mu
        self.gamma_order_method = hparams.gamma_order_method
        self.gamma_order_exact_max_steps = hparams.gamma_order_exact_max_steps
        self.gamma_order_local_search_passes = hparams.gamma_order_local_search_passes
        # Patch gamma indices are spatial positions, and static drive has no
        # sequence axis to order in the first place.
        self.gamma_order_enabled = (
            hparams.gamma_mode != "patch"
            and getattr(hparams, "gamma_drive_mode", "sequence") == "sequence"
        )
        self.gamma_generator = GammaGenerator(hparams, device=device)
        self.core = S2NetCore(hparams, device=device)

    @classmethod
    def from_hyperparameters(cls, hparams, device="cuda"):
        """Build S2NetClassifier from S2NetHyperparameters."""
        return cls(hparams, device=device)

    def forward(self, x):
        gamma_seq = self.gamma_generator(x)
        if self.gamma_order_enabled:
            gamma_seq, _, _ = self.order_gamma_sequence(gamma_seq)
        return self.core(gamma_seq)

    def order_gamma_sequence(self, gamma_seq):
        """Order generated gamma sequences before feeding S2NetCore.

        Only valid for latent gamma. In patch mode a gamma index is a fixed
        spatial patch position, so permuting the sequence would break the
        patch-to-image mapping that masks and reconstruction depend on.
        """
        if not self.gamma_order_enabled:
            raise ValueError(
                "gamma ordering is disabled: gamma indices are fixed spatial patches "
                "in patch mode and must never be reordered."
            )
        return order_gammas(
            gamma_seq,
            lambda_smooth=self.gamma_order_lambda,
            mu_similarity=self.gamma_order_mu,
            method=self.gamma_order_method,
            exact_max_steps=self.gamma_order_exact_max_steps,
            local_search_passes=self.gamma_order_local_search_passes,
        )

    def load_input_layer(self, checkpoint_path, map_location=None):
        """Load pretrained GammaGenerator.input_layer parameters."""
        state_dict = torch.load(
            checkpoint_path,
            map_location=self._checkpoint_device(map_location),
        )
        self.gamma_generator.input_layer.load_state_dict(state_dict)
        return self

    def load_gamma_initializer(self, checkpoint_path, map_location=None):
        """Load pretrained GammaGenerator.gamma_initializer parameters."""
        if self.hparams.gamma_mode == "patch":
            return self
        state_dict = torch.load(
            checkpoint_path,
            map_location=self._checkpoint_device(map_location),
        )
        self.gamma_generator.gamma_initializer.load_state_dict(state_dict)
        return self

    def load_core(self, checkpoint_path, map_location=None):
        """Load pretrained S2NetCore parameters."""
        state_dict = torch.load(
            checkpoint_path,
            map_location=self._checkpoint_device(map_location),
        )
        self.core.load_state_dict(state_dict)
        return self

    def load_checkpoints(
        self,
        input_layer_path=None,
        gamma_initializer_path=None,
        core_path=None,
        map_location=None,
        eval_mode=True,
    ):
        """Load any available pretrained component checkpoints."""
        if input_layer_path is not None:
            self.load_input_layer(input_layer_path, map_location=map_location)
        if gamma_initializer_path is not None:
            self.load_gamma_initializer(gamma_initializer_path, map_location=map_location)
        if core_path is not None:
            self.load_core(core_path, map_location=map_location)
        if eval_mode:
            self.eval()
        return self

    def _checkpoint_device(self, map_location):
        return map_location if map_location is not None else torch.device(self.device)
