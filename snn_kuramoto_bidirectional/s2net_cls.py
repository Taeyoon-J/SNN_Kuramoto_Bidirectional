import torch
import torch.nn as nn
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating
from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeatureMapCNNEncoder
from gamma_ordering import order_gammas

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
        self.gamma_initializer = FeatureMapCNNEncoder(
            num_osci=self.in_dim,
            in_channels=1,
            dropout=hparams.gamma_dropout,
        )

    def forward(self, x):
        x = x.to(self.device)
        feature_maps = self._image_to_feature_maps(x)
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
        )
        self.membrane_layer = MembraneLayer(
            output_dim=self.in_dim,
            readout_dim=hparams.num_classes,
            tau_minitializer='uniform',
            low_m=0,
            high_m=4,
            vth=0.5,
            dt=1,
            device=device
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, gamma_seq, sc):
        gamma_seq = gamma_seq.to(self.device)
        sc = sc.to(self.device)
        if gamma_seq.dim() != 3:
            raise ValueError("gamma_seq must have shape [B, T, num_regions]. Use B=1 for one sample.")
        B, T, _ = gamma_seq.shape

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

        core_input = torch.cat(feats_list, dim=1)
        all_hidden_masks = torch.stack(mask_hidden_list, dim=1)

        batch_size, seq_num, _, _ = core_input.shape
        self.dendric_layer.set_neuron_state(batch_size)
        self.membrane_layer.set_neuron_state(batch_size)

        outputs = []
        spikes_hist = []

        for i in range(seq_num):
            input_t = core_input[:, i, :, :]
            g_t = all_hidden_masks[:, i, :]

            l_input, mask = self.dendric_layer(
                input_t,
                self.membrane_layer.spike,
                g_t
            )
            mem_readout, spike_t = self.membrane_layer(l_input, mask)
            spikes_hist.append(spike_t)
            outputs.append(mem_readout)

        core_out = torch.stack(outputs).permute(1, 2, 0)
        spikes = torch.stack(spikes_hist).permute(1, 2, 0)
        logits_pooled = torch.mean(core_out, dim=2)

        return self.logsoftmax(logits_pooled), spikes


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
        self.gamma_generator = GammaGenerator(hparams, device=device)
        self.core = S2NetCore(hparams, device=device)

    @classmethod
    def from_hyperparameters(cls, hparams, device="cuda"):
        """Build S2NetClassifier from S2NetHyperparameters."""
        return cls(hparams, device=device)

    def forward(self, x, sc):
        gamma_seq = self.gamma_generator(x)
        ordered_gamma_seq, _, _ = self.order_gamma_sequence(gamma_seq)
        return self.core(ordered_gamma_seq, sc)

    def order_gamma_sequence(self, gamma_seq):
        """Order generated gamma sequences before feeding S2NetCore."""
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
