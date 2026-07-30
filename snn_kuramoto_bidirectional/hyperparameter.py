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
    num_feature_maps: int = 15
    num_regions: int = 90
    num_classes: int = 2

    # Fixed region-to-region connectivity matrix [num_regions, num_regions]
    sc: object = None

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

    # Object-group based classification
    spike_classify_method: str = "spike_rhythm"
    spike_rhythm_threshold: float = 0.8
    spike_rhythm_min_group_size: int = 2
    spike_rhythm_return_all_groups: bool = False
    spike_interval_size: int = 1
    spike_interval_threshold: float = 0.5
    spike_interval_min_group_size: int = 1
    spike_interval_include_partial: bool = True

    # Differentiable membrane-history clustering
    classifier_embedding_dim: int = 16
    classifier_num_iterations: int = 3
    classifier_temperature: float = 0.2

    # End-to-end autoencoder loss
    normal_reconstruction_loss_weight: float = 1.0
    weighted_reconstruction_loss_weight: float = 0.5
    mask_diversity_loss_weight: float = 0.02
    mask_entropy_loss_weight: float = 0.01
    membrane_consistency_loss_weight: float = 0.05
    edge_scale: float = 3.0

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
        if self.spike_classify_method not in {"spike_rhythm", "spike_interval"}:
            raise ValueError('spike_classify_method must be "spike_rhythm" or "spike_interval".')
        if self.spike_rhythm_min_group_size < 2:
            raise ValueError("spike_rhythm_min_group_size must be at least 2.")
        if self.spike_interval_size <= 0:
            raise ValueError("spike_interval_size must be positive.")
        if self.spike_interval_min_group_size <= 0:
            raise ValueError("spike_interval_min_group_size must be positive.")
        if self.classifier_embedding_dim <= 0:
            raise ValueError("classifier_embedding_dim must be positive.")
        if self.classifier_num_iterations <= 0:
            raise ValueError("classifier_num_iterations must be positive.")
        if self.classifier_temperature <= 0.0:
            raise ValueError("classifier_temperature must be positive.")
        return self


DEFAULT_HYPERPARAMETERS = S2NetHyperparameters()
