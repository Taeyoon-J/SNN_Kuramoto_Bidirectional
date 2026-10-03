
if __package__:
    from . import error_bound
else:
    import error_bound

import torch
import torch.nn as nn
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating
from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeatureMapCNNEncoder, FeaturePatchGammaInitializer, GammaToDrive
from graph_generator import ImageConditionedGraph
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
            error_bound.validate_s2net_cls_gamma_generator_forward(self, gamma_seq)
            return gamma_seq
        B, T, height, width = feature_maps.shape
        return self.gamma_initializer(
            feature_maps.reshape(B * T, 1, height, width)
        ).view(B, T, self.in_dim)

    def _image_to_feature_maps(self, x):
        error_bound.validate_s2net_cls_gamma_generator_image_to_feature_maps(x)

        return self.input_layer(x)


# "readout mode" marks README 4.3 training flags plus training parser defaults.
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
            error_bound.validate_s2net_cls_s2net_core_init(graph_mode)
            self.register_buffer("sc", torch.eye(self.in_dim))
        else:
            sc = torch.as_tensor(hparams.sc, dtype=torch.float32)
            expected_shape = (self.in_dim, self.in_dim)
            error_bound.validate_s2net_cls_s2net_core_init_2(expected_shape, sc)
            self.register_buffer("sc", sc.clone().detach())

        # Extra parameters are created only when the corresponding non-default
        # mode is requested, so a legacy configuration keeps an unchanged
        # state_dict and old checkpoints still load with strict=True.
        self.gamma_channel_proj = None
        if self.gamma_drive_mode == "static":  # readout mode
            self.gamma_channel_proj = nn.Linear(self.T, self.osc_dim, bias=False)
            nn.init.normal_(self.gamma_channel_proj.weight, std=1.0 / max(self.T, 1) ** 0.5)

        self.gamma_phase_gain = None
        if self.gamma_phase_mode != "none":  # readout mode: standardize_tanh
            self.gamma_phase_gain = nn.Parameter(torch.ones(1))

        # Keep trainable parameters above under their original checkpoint keys.
        self.gamma_to_drive = GammaToDrive(
            self.T, self.in_dim, self.osc_dim,
            drive_mode=self.gamma_drive_mode, phase_mode=self.gamma_phase_mode,
        )

        # A fixed SC cannot express object membership, which changes per image.
        # In "learned" mode the coupling graph is produced from the sample.
        self.graph_mode = getattr(hparams, "graph_mode", "static")
        self.graph_generator = None
        if self.graph_mode == "learned":  # readout mode
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
            input_vector_dim=1 if self.gate_mode == "phase_mean" else self.osc_dim,  # readout mode: phase_mean -> 1
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
        feedback = self.graph_generator is not None and self.graph_generator.uses_feedback  # readout mode: False (strength=0.0)
        if self.graph_generator is not None:  # readout mode: learned SC, once per forward
            sc = self.graph_generator(gamma_seq)
        else:
            sc = self.sc.to(gamma_seq.device).unsqueeze(0).expand(B, -1, -1)

        if self.gamma_drive_mode == "sequence":
            error_bound.validate_s2net_cls_s2net_core_forward(gamma_seq, num_time_steps)
            drive_seq = self.gamma_to_drive(
                gamma_seq, self.gamma_channel_proj, self.gamma_phase_gain
            )
            T = drive_seq.size(1)
            drive = None
            theta = self._init_theta(drive_seq[:, 0, :], B)
        else:  # readout mode: static
            drive_seq = None
            drive = self.gamma_to_drive(
                gamma_seq, self.gamma_channel_proj, self.gamma_phase_gain
            )
            T = int(num_time_steps if num_time_steps is not None else self.num_time_steps)
            error_bound.validate_s2net_cls_s2net_core_forward_2(T)
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
            )
            if feedback:
                alignment = self.graph_generator.update_alignment(alignment, theta)
            theta_hist.append(theta)

            gamma_wave_t, g_wave_t = sinusoidal_gating(
                theta_hist, t, self.phase_delay_steps, gate_mode=self.gate_mode
            )

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

    def _init_theta(self, drive, batch_size):
        """Initial oscillator phase [B, num_regions, osc_dim]."""
        if self.theta_init == "zeros":
            theta = torch.zeros(batch_size, self.in_dim, self.osc_dim, device=self.device)
        else:  # readout mode: gamma (without noise)
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
        if self.spike_classify_method == "spatial_components":  # readout mode: core's internal groups
            if self.spike_spatial_activity_source == "spikes":
                activity = spikes
            elif self.spike_spatial_activity_source == "membrane":
                activity = core_out
            else:  # readout mode: sigmoid_membrane
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
    """End-to-end wrapper: input image -> gamma sequence -> classifier core."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        hparams.validate()
        self.hparams = hparams
        self.device = device
        self.gamma_generator = GammaGenerator(hparams, device=device)
        self.core = S2NetCore(hparams, device=device)

    @classmethod
    def from_hyperparameters(cls, hparams, device="cuda"):
        """Build S2NetClassifier from S2NetHyperparameters."""
        return cls(hparams, device=device)

    def forward(self, x):
        gamma_seq = self.gamma_generator(x)
        return self.core(gamma_seq)

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
