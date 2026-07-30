"""Complete end-to-end S2Net object-centric autoencoder.

Model path:
    image
      -> CNNFeatureEncoder
      -> FeatureMapCNNEncoder
      -> S2NetCore
      -> SoftMembraneClassifier
      -> SpatialBroadcastDecoder
      -> reconstructed image and object masks
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

try:
    from .decoder import DecoderOutput, SpatialBroadcastDecoder
    from .dendric_layer import DendricLayer
    from .gamma_initializer import FeatureMapCNNEncoder
    from .input_layer_generator import CNNFeatureEncoder
    from .kuramoto_layer import graphVectorKuramoto
    from .membrane_layer import MembraneLayer
    from .sc_generator import DynamicSCGenerator
    from .sinusoidal_gating import sinusoidal_gating
    from .spike_classifier import SoftMembraneClassifier
except ImportError:
    # Retain support for running this file directly from the package folder.
    from decoder import DecoderOutput, SpatialBroadcastDecoder
    from dendric_layer import DendricLayer
    from gamma_initializer import FeatureMapCNNEncoder
    from input_layer_generator import CNNFeatureEncoder
    from kuramoto_layer import graphVectorKuramoto
    from membrane_layer import MembraneLayer
    from sc_generator import DynamicSCGenerator
    from sinusoidal_gating import sinusoidal_gating
    from spike_classifier import SoftMembraneClassifier


class CoreOutput(NamedTuple):
    """Continuous outputs of the SNN core."""

    membrane: Tensor
    spikes: Tensor


class S2NetOutput(NamedTuple):
    """All outputs required for training and inspecting the complete model."""

    core_output: CoreOutput
    reconstruction: Tensor
    masks: Tensor
    object_rgb: Tensor
    mask_logits: Tensor
    object_vectors: Tensor
    spikes: Tensor
    membrane: Tensor
    gamma: Tensor
    feature_maps: Tensor
    sc: Tensor
    batch_sc: Tensor
    running_sc: Tensor


class GammaGenerator(nn.Module):
    """Generate one independent gamma vector from every feature map."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.num_feature_maps = int(hparams.num_feature_maps)
        self.num_oscillators = int(hparams.num_regions)

        self.input_layer = CNNFeatureEncoder(
            num_kernels=self.num_feature_maps,
            kernel_size=hparams.kernel_size,
            in_channels=hparams.in_channels,
            bias=True,
        )
        self.gamma_initializer = FeatureMapCNNEncoder(
            num_osci=self.num_oscillators,
            in_channels=1,
            dropout=hparams.gamma_dropout,
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``gamma [B,T,N]`` and feature maps ``[B,T,H',W']``."""

        self._validate_images(images)
        feature_maps = self.input_layer(images)
        batch_size, num_maps, height, width = feature_maps.shape

        independent_maps = feature_maps.reshape(
            batch_size * num_maps,
            1,
            height,
            width,
        )
        gamma = self.gamma_initializer(independent_maps).view(
            batch_size,
            num_maps,
            self.num_oscillators,
        )
        return gamma, feature_maps

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W].")
        if images.shape[1] != 3:
            raise ValueError(
                f"Expected RGB images with 3 channels, got {images.shape[1]}."
            )
        if not images.is_floating_point():
            raise TypeError("images must be floating-point tensors.")


class S2NetCore(nn.Module):
    """Convert gamma sequences into differentiable membrane and spike traces."""

    def __init__(self, hparams, device=None) -> None:
        super().__init__()
        self.num_steps = int(hparams.num_feature_maps)
        self.num_oscillators = int(hparams.num_regions)
        self.oscillator_vector_dim = 4
        self.phase_delay_steps = 2
        self.requested_device = None if device is None else torch.device(device)

        module_device = (
            str(self.requested_device)
            if self.requested_device is not None
            else "cpu"
        )
        self.kuramoto = graphVectorKuramoto(
            N=self.num_oscillators,
            D=self.oscillator_vector_dim,
            K=hparams.k,
            dt=hparams.dt,
            alpha_scale=1.0,
            device=module_device,
        )
        self.dendric_layer = DendricLayer(
            input_dim=self.num_oscillators,
            output_dim=self.num_oscillators,
            tau_ninitializer="uniform",
            low_n=hparams.low_n,
            high_n=hparams.high_n,
            branch=hparams.branch,
            device=module_device,
            bias=True,
            input_vector_dim=self.oscillator_vector_dim,
        )
        self.membrane_layer = MembraneLayer(
            output_dim=self.num_oscillators,
            tau_minitializer="uniform",
            low_m=0,
            high_m=4,
            vth=0.5,
            dt=1,
            device=module_device,
        )

    def forward(self, gamma: Tensor, sc: Tensor) -> CoreOutput:
        """Process ``gamma [B,T,N]`` and return traces ``[B,N,T]``."""

        self._validate_gamma(gamma)
        self._validate_sc(sc)
        batch_size, num_steps, _ = gamma.shape
        device = gamma.device

        # State-creating legacy layers use their ``device`` attributes.
        self.dendric_layer.device = device
        self.membrane_layer.device = device

        sc = sc.to(device=device, dtype=gamma.dtype)
        if sc.ndim == 2:
            sc = sc.unsqueeze(0).expand(batch_size, -1, -1)
        theta = gamma.new_zeros(
            batch_size,
            self.num_oscillators,
            self.oscillator_vector_dim,
        )

        theta_history = []
        for step in range(num_steps):
            theta = self.kuramoto(theta, gamma[:, step], A=sc)
            theta_history.append(theta)

        gated_features, hidden_masks = sinusoidal_gating(
            theta_history,
            num_steps,
            self.phase_delay_steps,
        )
        all_features = torch.cat(gated_features, dim=1)
        all_hidden_masks = torch.stack(hidden_masks, dim=1)

        self.dendric_layer.set_neuron_state(batch_size)
        self.membrane_layer.set_neuron_state(batch_size)

        membrane_history = []
        spike_history = []
        for step in range(all_features.shape[1]):
            dendritic = self.dendric_layer(
                all_features[:, step],
                self.membrane_layer.spike,
            )
            membrane, spike = self.membrane_layer(
                dendritic,
                all_hidden_masks[:, step],
            )
            membrane_history.append(membrane)
            spike_history.append(spike)

        return CoreOutput(
            membrane=torch.stack(membrane_history, dim=2),
            spikes=torch.stack(spike_history, dim=2),
        )

    def _validate_gamma(self, gamma: Tensor) -> None:
        if gamma.ndim != 3:
            raise ValueError(
                "gamma must have shape [B, num_steps, num_oscillators]."
            )
        if gamma.shape[1] != self.num_steps:
            raise ValueError(
                f"Expected {self.num_steps} gamma steps, got {gamma.shape[1]}."
            )
        if gamma.shape[2] != self.num_oscillators:
            raise ValueError(
                f"Expected {self.num_oscillators} oscillators, "
                f"got {gamma.shape[2]}."
            )

    def _validate_sc(self, sc: Tensor) -> None:
        expected_matrix = (self.num_oscillators, self.num_oscillators)
        if sc.ndim == 2 and tuple(sc.shape) == expected_matrix:
            return
        if (
            sc.ndim == 3
            and tuple(sc.shape[1:]) == expected_matrix
        ):
            return
        raise ValueError(
            "sc must have shape [N,N] or [B,N,N], where N is the "
            f"number of oscillators ({self.num_oscillators})."
        )


class S2NetClassifier(nn.Module):
    """End-to-end object-centric image autoencoder.

    ``num_objects`` controls how many soft oscillator groups are produced by
    differentiable clustering of complete membrane histories.
    """

    def __init__(
        self,
        hparams,
        device=None,
        num_objects=None,
        image_size=(128, 128),
        decoder_broadcast_size=(8, 8),
        decoder_hidden_channels=(64, 64, 64, 64, 64),
        rgb_activation="sigmoid",
        sc_momentum=0.99,
        sc_eps=1e-8,
    ) -> None:
        super().__init__()
        hparams.validate()

        self.hparams = hparams
        self.num_steps = int(hparams.num_feature_maps)
        self.num_oscillators = int(hparams.num_regions)
        self.num_objects = int(
            self.num_steps if num_objects is None else num_objects
        )

        if self.num_objects <= 0:
            raise ValueError("num_objects must be positive.")
        if self.num_objects > self.num_oscillators:
            raise ValueError(
                "num_objects cannot exceed hparams.num_regions because "
                "clustering centers are initialized from oscillator "
                "embeddings."
            )

        self.gamma_generator = GammaGenerator(hparams)
        self.sc_generator = DynamicSCGenerator(
            num_oscillators=self.num_oscillators,
            momentum=sc_momentum,
            eps=sc_eps,
        )
        self.core = S2NetCore(hparams, device=device)
        self.classifier = SoftMembraneClassifier(
            history_length=self.num_steps,
            embedding_dim=hparams.classifier_embedding_dim,
            num_iterations=hparams.classifier_num_iterations,
            temperature=hparams.classifier_temperature,
        )
        self.decoder = SpatialBroadcastDecoder(
            object_dim=self.num_oscillators,
            image_size=image_size,
            broadcast_size=decoder_broadcast_size,
            hidden_channels=decoder_hidden_channels,
            rgb_activation=rgb_activation,
        )

        if device is not None:
            self.to(torch.device(device))

    @classmethod
    def from_hyperparameters(cls, hparams, device=None, **kwargs):
        """Build the complete model from ``S2NetHyperparameters``."""

        return cls(hparams, device=device, **kwargs)

    def forward(self, images: Tensor) -> S2NetOutput:
        gamma, feature_maps = self.gamma_generator(images)
        sc_output = self.sc_generator(gamma)
        core_output = self.core(gamma, sc=sc_output.sc)
        object_vectors = self.classifier(
            core_output.membrane,
            num_centers=self.num_objects,
        )
        decoder_output: DecoderOutput = self.decoder(object_vectors)

        return S2NetOutput(
            core_output=core_output,
            reconstruction=decoder_output.reconstruction,
            masks=decoder_output.masks,
            object_rgb=decoder_output.object_rgb,
            mask_logits=decoder_output.mask_logits,
            object_vectors=object_vectors,
            spikes=core_output.spikes,
            membrane=core_output.membrane,
            gamma=gamma,
            feature_maps=feature_maps,
            sc=sc_output.sc,
            batch_sc=sc_output.batch_sc,
            running_sc=sc_output.running_sc,
        )

    def load_input_layer(self, checkpoint_path, map_location=None):
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.gamma_generator.input_layer.load_state_dict(state_dict)
        return self

    def load_gamma_initializer(self, checkpoint_path, map_location=None):
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.gamma_generator.gamma_initializer.load_state_dict(state_dict)
        return self

    def load_core(self, checkpoint_path, map_location=None):
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.core.load_state_dict(state_dict)
        return self

    def load_decoder(self, checkpoint_path, map_location=None):
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.decoder.load_state_dict(state_dict)
        return self

    def load_model(self, checkpoint_path, map_location=None):
        """Load one checkpoint containing the complete end-to-end model."""

        state_dict = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(state_dict)
        return self


# The architecture is now an autoencoder, but retain the old public class name.
S2NetAutoEncoder = S2NetClassifier
