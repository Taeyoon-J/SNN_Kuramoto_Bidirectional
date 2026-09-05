import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating
from u_net import SharedMultiScaleUNet
from u_net_classifier import classify_hierarchical_spikes
from sc_generator import generate_sc

class GammaGenerator(nn.Module):
    """Generate one spatial-patch gamma sequence per U-Net encoder level."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        self.T = int(hparams.num_feature_maps)
        if self.T != 8:
            raise ValueError(
                "SharedMultiScaleUNet produces exactly 8 feature maps per level, "
                f"but hparams.num_feature_maps is {self.T}."
            )

        self.device = torch.device(device)
        self.patch_size = 8
        self.unet = SharedMultiScaleUNet(feature_channels=self.T).to(self.device)

    def forward(self, x, return_features=False):
        """Return four gamma tensors, optionally together with encoder features."""
        if x.dim() != 4:
            raise ValueError("x must have shape [B, 3, H, W]. Use B=1 for one image.")
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, but got {x.size(1)}.")

        model_device = next(self.unet.parameters()).device
        feature_levels = self.unet.encoder(x.to(model_device))
        gamma_levels = []

        for level_index, feature_maps in enumerate(feature_levels, start=1):
            height, width = feature_maps.shape[-2:]
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    f"U-Net level {level_index} has spatial shape {(height, width)}, "
                    f"which is not divisible by patch size {self.patch_size}."
                )

            pooled = F.avg_pool2d(
                feature_maps,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            gamma_levels.append(pooled.flatten(start_dim=2))

        if return_features:
            return gamma_levels, feature_levels
        return gamma_levels

    def load_unet_checkpoint(self, checkpoint_path, map_location=None):
        """Load a SharedMultiScaleUNet training checkpoint."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location if map_location is not None else self.device,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.unet.load_state_dict(state_dict)
        return self

    def set_unet_trainable(self, trainable):
        """Enable or disable gradient updates for the U-Net."""
        for parameter in self.unet.parameters():
            parameter.requires_grad = bool(trainable)
        return self


class S2NetCore(nn.Module):
    """Classifier core driven by gamma and batch-specific connectivity."""

    def __init__(self, hparams, device="cuda", num_regions=None):
        super().__init__()
        self.T = int(hparams.num_feature_maps)
        self.in_dim = int(
            hparams.num_regions if num_regions is None else num_regions
        )
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
        self.dense_history = None
        self.dendritic_history = None
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
        dense_history = []
        dendritic_history = []

        for i in range(seq_num):
            gamma_wave_t = all_feats[:, i, :, :]
            g_wave_t = all_mask_hidden[:, i, :]

            h_wave_t = self.dendric_layer(
                gamma_wave_t,
                self.membrane_layer.spike
            )
            dense_history.append(self.dendric_layer.dense_i)
            dendritic_history.append(self.dendric_layer.h)
            mem_t, spike_t = self.membrane_layer(h_wave_t, g_wave_t)
            spikes_hist.append(spike_t)
            outputs.append(mem_t)

        core_out = torch.stack(outputs).permute(1, 2, 0)
        spikes = torch.stack(spikes_hist).permute(1, 2, 0)
        self.dense_history = torch.stack(dense_history, dim=2)
        self.dendritic_history = torch.stack(dendritic_history, dim=2)
        return spikes, core_out


@dataclass
class S2NetOutput:
    """Detailed outputs from one multi-level S2Net forward pass."""

    object_masks: torch.Tensor | None
    valid_objects: torch.Tensor | None
    feature_levels: list[torch.Tensor]
    gamma_levels: list[torch.Tensor]
    sc_levels: list[torch.Tensor]
    spike_levels: list[torch.Tensor]
    core_out_levels: list[torch.Tensor]
    dense_i_levels: list[torch.Tensor | None]
    dendritic_h_levels: list[torch.Tensor | None]


class S2NetClassifier(nn.Module):
    """Run four U-Net feature levels through independent S2Net cores."""

    def __init__(self, hparams, device="cuda"):
        super().__init__()
        hparams.validate()
        self.hparams = hparams
        self.device = torch.device(device)
        self.level_grid_sizes = (
            (16, 16),
            (8, 8),
            (4, 4),
            (2, 2),
        )
        self.level_num_regions = tuple(
            grid_height * grid_width
            for grid_height, grid_width in self.level_grid_sizes
        )
        self.gamma_generator = GammaGenerator(hparams, device=device)
        self.level_cores = nn.ModuleList(
            [
                S2NetCore(
                    hparams,
                    device=device,
                    num_regions=num_regions,
                )
                for num_regions in self.level_num_regions
            ]
        )

    @classmethod
    def from_hyperparameters(cls, hparams, device="cuda"):
        """Build S2NetClassifier from S2NetHyperparameters."""
        return cls(hparams, device=device)

    def forward(self, x, return_details=False, classify=True):
        x = x.to(self.device)
        gamma_levels, feature_levels = self.gamma_generator(
            x,
            return_features=True,
        )
        if len(gamma_levels) != len(self.level_cores):
            raise ValueError(
                f"Expected {len(self.level_cores)} gamma levels, "
                f"but received {len(gamma_levels)}."
            )

        sc_levels = []
        spike_levels = []
        core_out_levels = []
        dense_i_levels = []
        dendritic_h_levels = []

        for level_index, (gamma, grid_size, core) in enumerate(
            zip(gamma_levels, self.level_grid_sizes, self.level_cores),
            start=1,
        ):
            expected_regions = self.level_num_regions[level_index - 1]
            if gamma.shape[-1] != expected_regions:
                raise ValueError(
                    f"Level {level_index} gamma has {gamma.shape[-1]} oscillators, "
                    f"but grid {grid_size} requires {expected_regions}."
                )

            sc = generate_sc(
                x,
                grid_size,
                sigma_color=self.hparams.sc_sigma_color,
                m_min=self.hparams.sc_m_min,
                self_connectivity=self.hparams.sc_self_connectivity,
            )
            spikes, core_out = core(gamma, sc=sc)
            sc_levels.append(sc)
            spike_levels.append(spikes)
            core_out_levels.append(core_out)
            dense_i_levels.append(core.dense_history)
            dendritic_h_levels.append(core.dendritic_history)

        if classify:
            object_masks, valid_objects = classify_hierarchical_spikes(
                spike_levels,
                output_size=x.shape[-2:],
                level_grid_sizes=self.level_grid_sizes,
            )
        else:
            object_masks = None
            valid_objects = None

        if return_details:
            return S2NetOutput(
                object_masks=object_masks,
                valid_objects=valid_objects,
                feature_levels=feature_levels,
                gamma_levels=gamma_levels,
                sc_levels=sc_levels,
                spike_levels=spike_levels,
                core_out_levels=core_out_levels,
                dense_i_levels=dense_i_levels,
                dendritic_h_levels=dendritic_h_levels,
            )
        return object_masks, spike_levels

    def load_unet(self, checkpoint_path, map_location=None, trainable=False):
        """Load the trained U-Net and optionally enable fine-tuning."""
        self.gamma_generator.load_unet_checkpoint(
            checkpoint_path,
            map_location=self._checkpoint_device(map_location),
        )
        self.gamma_generator.set_unet_trainable(trainable)
        return self

    def load_cores(self, checkpoint_path, map_location=None):
        """Load one checkpoint containing all four level-core parameters."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self._checkpoint_device(map_location),
        )
        state_dict = checkpoint.get("level_cores_state_dict", checkpoint)
        self.level_cores.load_state_dict(state_dict)
        return self

    def load_checkpoints(
        self,
        unet_path=None,
        cores_path=None,
        map_location=None,
        eval_mode=True,
        unet_trainable=False,
    ):
        """Load the U-Net and/or multi-level core checkpoints."""
        if unet_path is not None:
            self.load_unet(
                unet_path,
                map_location=map_location,
                trainable=unet_trainable,
            )
        if cores_path is not None:
            self.load_cores(cores_path, map_location=map_location)
        if eval_mode:
            self.eval()
        return self

    def _checkpoint_device(self, map_location):
        return map_location if map_location is not None else torch.device(self.device)
