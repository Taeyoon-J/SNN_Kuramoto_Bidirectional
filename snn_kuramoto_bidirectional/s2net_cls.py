import torch
import torch.nn as nn
from dataclasses import dataclass
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating
from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeaturePatchGammaInitializer
from sc_generator import generate_sc
from spike_classifier import spike_interval, spike_rhythm, spike_spatial_components

class GammaGenerator(nn.Module):
    """Generate gamma sequences from input images."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        self.T = int(hparams.num_feature_maps)
        self.in_dim = int(hparams.num_regions)
        self.device = device
        self.input_layer = CNNFeatureEncoder(
            num_kernels=self.T,
            kernel_size=hparams.kernel_size,
            in_channels=hparams.in_channels,
            bias=True,
        )
        self.gamma_initializer = FeaturePatchGammaInitializer(
            grid_size=hparams.gamma_patch_grid_size,
            patch_size=hparams.gamma_patch_size,
            stride=hparams.gamma_patch_stride,
            reduction=hparams.gamma_patch_reduction,
        )

    def forward(self, x):
        x = x.to(self.device)
        feature_maps = self._image_to_feature_maps(x)
        gamma_seq = self.gamma_initializer(feature_maps)
        if gamma_seq.size(-1) != self.in_dim:
            raise ValueError(
                f"Patch gamma produced {gamma_seq.size(-1)} oscillators, "
                f"but hparams.num_regions is {self.in_dim}."
            )
        return gamma_seq

    def _image_to_feature_maps(self, x):
        if x.dim() != 4:
            raise ValueError("x must have shape [B, 3, H, W]. Use B=1 for one image.")
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, but got {x.size(1)}.")

        return self.input_layer(x)


class S2NetCore(nn.Module):
    """Classifier core driven by gamma and batch-specific connectivity."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        self.T = int(hparams.num_feature_maps)
        self.in_dim = int(hparams.num_regions)
        self.osc_dim = 4
        self.phase_delay_steps = 2
        self.device = device

        self.kuramoto = graphVectorKuramoto(
            N=self.in_dim, D=self.osc_dim, K=hparams.k, dt=hparams.dt, alpha_scale=1.0, device=device
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
            input_vector_dim=self.osc_dim,
            aggregation_mode=hparams.dendritic_aggregation,
        )
        self.membrane_layer = MembraneLayer(
            output_dim=self.in_dim,
            tau_minitializer='uniform',
            low_m=0,
            high_m=4,
            vth=0.5,
            dt=1,
            device=device
        )
    def forward(self, gamma_seq, sc):
        gamma_seq = gamma_seq.to(self.device)
        sc = sc.to(device=gamma_seq.device, dtype=gamma_seq.dtype)
        if gamma_seq.dim() != 3:
            raise ValueError("gamma_seq must have shape [B, T, num_regions]. Use B=1 for one sample.")
        B, T, N = gamma_seq.shape
        if T != self.T or N != self.in_dim:
            raise ValueError(
                f"gamma_seq must have shape [B, {self.T}, {self.in_dim}], "
                f"but got {tuple(gamma_seq.shape)}."
            )
        if tuple(sc.shape) != (B, self.in_dim, self.in_dim):
            raise ValueError(
                f"sc must have shape {(B, self.in_dim, self.in_dim)}, "
                f"but got {tuple(sc.shape)}."
            )

        theta = torch.zeros(B, self.in_dim, self.osc_dim, device=self.device)
        theta_hist = []
        for t in range(T):
            gamma_t = gamma_seq[:, t, :]
            theta = self.kuramoto(theta, gamma_t, A=sc)
            theta_hist.append(theta)

        feats_list, mask_hidden_list = sinusoidal_gating(
            theta_hist,
            T,
            self.phase_delay_steps,
        )

        all_feats = torch.cat(feats_list, dim=1)
        all_mask_hidden = torch.stack(mask_hidden_list, dim=1)

        batch_size, seq_num, _, _ = all_feats.shape
        self.dendric_layer.set_neuron_state(batch_size)
        self.membrane_layer.set_neuron_state(batch_size)

        outputs = []
        spikes_hist = []

        for i in range(seq_num):
            gamma_wave_t = all_feats[:, i, :, :]
            g_wave_t = all_mask_hidden[:, i, :]

            h_wave_t = self.dendric_layer(
                gamma_wave_t,
                self.membrane_layer.spike
            )
            mem_t, spike_t = self.membrane_layer(h_wave_t, g_wave_t)
            spikes_hist.append(spike_t)
            outputs.append(mem_t)

        core_out = torch.stack(outputs).permute(1, 2, 0)
        spikes = torch.stack(spikes_hist).permute(1, 2, 0)
        return spikes, core_out


@dataclass
class S2NetOutput:
    """Detailed outputs from one image-conditioned S2Net forward pass."""

    object_groups: list | None
    spikes: torch.Tensor
    core_out: torch.Tensor
    gamma_seq: torch.Tensor
    sc: torch.Tensor


class S2NetClassifier(nn.Module):
    """End-to-end wrapper: image -> spatial patch gamma -> classifier core."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        hparams.validate()
        self.hparams = hparams
        self.device = device
        self.patch_grid_size = hparams.gamma_patch_grid_size
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
        self.gamma_generator = GammaGenerator(hparams, device=device)
        self.core = S2NetCore(hparams, device=device)

    @classmethod
    def from_hyperparameters(cls, hparams, device="cuda"):
        """Build S2NetClassifier from S2NetHyperparameters."""
        return cls(hparams, device=device)

    def forward(self, x, return_details=False, classify=True):
        x = x.to(self.device)
        gamma_seq = self.gamma_generator(x)
        sc = generate_sc(
            x,
            self.patch_grid_size,
            sigma_color=self.hparams.sc_sigma_color,
            m_min=self.hparams.sc_m_min,
            self_connectivity=self.hparams.sc_self_connectivity,
        )
        spikes, core_out = self.core(
            gamma_seq,
            sc=sc,
        )
        object_groups = (
            self._detect_object_groups(core_out, spikes)
            if classify
            else None
        )
        if return_details:
            return S2NetOutput(
                object_groups=object_groups,
                spikes=spikes,
                core_out=core_out,
                gamma_seq=gamma_seq,
                sc=sc,
            )
        return object_groups, spikes

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

    def load_input_layer(self, checkpoint_path, map_location=None):
        """Load pretrained GammaGenerator.input_layer parameters."""
        state_dict = torch.load(
            checkpoint_path,
            map_location=self._checkpoint_device(map_location),
        )
        self.gamma_generator.input_layer.load_state_dict(state_dict)
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
        core_path=None,
        map_location=None,
        eval_mode=True,
    ):
        """Load any available pretrained component checkpoints."""
        if input_layer_path is not None:
            self.load_input_layer(input_layer_path, map_location=map_location)
        if core_path is not None:
            self.load_core(core_path, map_location=map_location)
        if eval_mode:
            self.eval()
        return self

    def _checkpoint_device(self, map_location):
        return map_location if map_location is not None else torch.device(self.device)
