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
        if num_kernels <= 0:
            raise ValueError("num_kernels must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

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
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {channels}."
            )
        if width < self.kernel_size or height < self.kernel_size:
            raise ValueError(
                "Image width and height must both be at least as large as kernel_size."
            )


class CNNFeatureDecoder(nn.Module):
    """
    Decoder paired with CNNFeatureEncoder.

    Given D feature matrices of shape (W - N + 1) x (H - N + 1), this returns a
    reconstructed image batch of shape [B, C, H, W].
    """

    def __init__(self, num_kernels, kernel_size, out_channels=3, bias=True):
        super().__init__()
        if num_kernels <= 0:
            raise ValueError("num_kernels must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

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


def train_autoencoder(
    images,
    num_kernels,
    kernel_size,
    channels=3,
    epochs=500,
    lr=1e-3,
    bias=True,
):
    """
    Train an autoencoder and return the trained encoder.

    Args:
        images:
            Tensor shaped [B, 3, H, W].

    Returns:
        trained_encoder, autoencoder, loss_history
    """
    autoencoder = CNNAutoEncoder(
        num_kernels=num_kernels,
        kernel_size=kernel_size,
        channels=channels,
        bias=bias,
    )
    target = autoencoder.encoder._prepare_image(images)
    autoencoder.encoder._validate_image_size(target)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_history = []

    for _ in range(epochs):
        reconstruction = autoencoder(target)
        loss = F.mse_loss(reconstruction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return autoencoder.encoder, autoencoder, loss_history


if __name__ == "__main__":
    W, H = 8, 10
    D, N = 4, 3

    image = torch.randn(1, 3, W, H)
    encoder, autoencoder, losses = train_autoencoder(
        image,
        num_kernels=D,
        kernel_size=N,
        epochs=100,
        lr=1e-2,
    )

    feature_matrices = encoder(image)
    reconstruction = autoencoder(image)

    print(f"initial-to-final reconstruction loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(feature_matrices.shape)
    print(reconstruction.shape)
