
if __package__:
    from . import error_bound
else:
    import error_bound

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureEncoder(nn.Module):
    """
    Independent CNN-style image encoder.

    Given D kernels of size N x N, this returns D valid-convolution feature
    maps for RGB image batches shaped [B, 3, H, W].
    """

    def __init__(self, num_kernels, kernel_size, in_channels=3, bias=True):
        super().__init__()
        error_bound.validate_input_layer_generator_c_n_n_feature_encoder_init(num_kernels, kernel_size, in_channels)

        self.num_kernels = int(num_kernels)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)

        self.kernels = nn.Parameter(
            torch.empty(self.num_kernels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.num_kernels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernels, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = fan_in ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, image):
        """
        Args:
            image:
                Tensor shaped [B, 3, H, W].

        Returns:
            Tensor shaped [B, D, H-N+1, W-N+1].
        """
        image = self._prepare_image(image)
        self._validate_image_size(image)

        return F.conv2d(image, self.kernels, bias=self.bias, stride=1, padding=0)

    def _prepare_image(self, image):
        if image.dim() == 4:
            return image
        raise ValueError("image must have shape [B, C, H, W]. Use B=1 for one image.")

    def _validate_image_size(self, image):
        _, channels, width, height = image.shape
        error_bound.validate_input_layer_generator_c_n_n_feature_encoder_validate_image_size(channels, self, width, height)


class CNNFeatureDecoder(nn.Module):
    """
    Decoder paired with CNNFeatureEncoder.

    Given D feature matrices of shape (W - N + 1) x (H - N + 1), this returns a
    reconstructed image batch of shape [B, C, H, W].
    """

    def __init__(self, num_kernels, kernel_size, out_channels=3, bias=True):
        super().__init__()
        error_bound.validate_input_layer_generator_c_n_n_feature_decoder_init(num_kernels, kernel_size, out_channels)

        self.num_kernels = int(num_kernels)
        self.kernel_size = int(kernel_size)
        self.out_channels = int(out_channels)

        self.deconv = nn.ConvTranspose2d(
            in_channels=self.num_kernels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, features):
        features = self._prepare_features(features)
        return self.deconv(features)

    def _prepare_features(self, features):
        if features.dim() == 4:
            return features
        raise ValueError("features must have shape [B, D, H, W].")


class CNNAutoEncoder(nn.Module):
    """
    Autoencoder used to train the encoder without labels.

    Training objective:
        input image -> encoder -> features -> decoder -> reconstructed image
        minimize reconstruction error between reconstructed image and input image.
    """

    def __init__(self, num_kernels, kernel_size, channels=3, bias=True):
        super().__init__()
        self.encoder = CNNFeatureEncoder(
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            in_channels=channels,
            bias=bias,
        )
        self.decoder = CNNFeatureDecoder(
            num_kernels=num_kernels,
            kernel_size=kernel_size,
            out_channels=channels,
            bias=bias,
        )

    def forward(self, image):
        image = self.encoder._prepare_image(image)
        self.encoder._validate_image_size(image)

        features = F.conv2d(
            image,
            self.encoder.kernels,
            bias=self.encoder.bias,
            stride=1,
            padding=0,
        )
        return self.decoder(features)

    def encode(self, image):
        return self.encoder(image)
