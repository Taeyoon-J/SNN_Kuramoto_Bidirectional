from dataclasses import dataclass


@dataclass
class S2NetHyperparameters:
    """
    Central hyperparameter container for S2Net experiments.

    Shape contract:
        image:     [B, 3, H, W]
        feature:   [B, num_feature_maps, H', W']
        gamma_seq: [B, num_feature_maps, num_regions]
    """

    # Model dimensions
    num_feature_maps: int = 8
    num_regions: int = 90
    num_classes: int = 2

    # RGB input -> feature maps
    in_channels: int = 3
    kernel_size: int = 3

    # Feature map -> gamma vector
    gamma_dropout: float = 0.0

    # Gamma ordering loss
    gamma_order_lambda: float = 1.0
    gamma_order_mu: float = 1.0
    gamma_order_method: str = "auto"
    gamma_order_exact_max_steps: int = 8
    gamma_order_local_search_passes: int = 5

    # Kuramoto dynamics
    k: float = 1.0
    dt: float = 0.1

    # Dendritic SNN layer
    low_n: float = 0.0
    high_n: float = 4.0
    branch: int = 4

    def validate(self):
        if self.num_feature_maps <= 0:
            raise ValueError("num_feature_maps must be positive.")
        if self.num_regions <= 0:
            raise ValueError("num_regions must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.in_channels != 3:
            raise ValueError("in_channels must be 3 because the model is fixed to RGB input.")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if self.gamma_order_method not in {"auto", "exact", "local_search"}:
            raise ValueError('gamma_order_method must be "auto", "exact", or "local_search".')
        return self


DEFAULT_HYPERPARAMETERS = S2NetHyperparameters()
