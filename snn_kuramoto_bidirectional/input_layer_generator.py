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
