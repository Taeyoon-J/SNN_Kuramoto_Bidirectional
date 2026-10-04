
if __package__:
    from . import error_bound
else:
    import error_bound

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
    num_regions: int = 256
    num_classes: int = 2
    osc_dim: int = 4

    # SNN recurrent time axis.
    #
    # gamma_drive_mode == "sequence" (legacy): one Kuramoto step per visual
    # channel, so the recurrent length is tied to num_feature_maps.
    # gamma_drive_mode == "static": gamma is a constant sensory drive and the
    # recurrent length is num_time_steps, independent of the channel count.
    # "static" is required for multi-scale/U-Net work, where every level must
    # share one time axis while having a different channel count.
    gamma_drive_mode: str = "static"
    num_time_steps: int = 64

    # Map raw gamma values onto a phase range before sin(gamma - theta).
    # Pooled CNN activations have an arbitrary scale, so "none" lets the drive
    # wrap around the sine and lose image specificity.
    gamma_phase_mode: str = "standardize_tanh"

    # Initial oscillator phase. "zeros" makes every image start from the same
    # state, so only the drive term carries image information.
    theta_init: str = "gamma"
    theta_init_noise: float = 0.0

    # Fixed region-to-region connectivity matrix [num_regions, num_regions]
    sc: object = None

    # RGB input -> feature maps
    in_channels: int = 3
    kernel_size: int = 3

    # Feature map -> gamma vector
    gamma_mode: str = "autoencoder"
    gamma_dropout: float = 0.0
    gamma_patch_grid_size: object = None
    gamma_patch_size: object = None
    gamma_patch_stride: object = None
    gamma_patch_reduction: str = "mean"

    # Coupling graph.
    #
    # "static" uses the fixed sc matrix. A fixed graph cannot express which
    # patches belong to the same object, because that changes with the image,
    # so "learned" builds one graph per sample from its own features.
    # Measured on CLEVR at a 16x16 grid: sparse, feature-only, image-conditioned
    # coupling was the only variant that beat no coupling at all.
    graph_mode: str = "learned"
    graph_hidden_dim: int = 16
    graph_top_k: int = 32
    graph_coupling_gain: float = 8.0
    graph_temperature: float = 0.1

    # Structural prior on the learned graph. Objects are connected regions, and
    # the features do not know that. A hand-written spatial x similarity kernel
    # scored ARI 0.056 against 0.038 for the randomly initialised graph, so this
    # is a better starting point. None disables it.
    graph_spatial_decay: object = 0.55

    # Let the graph track the synchrony it produces, so oscillators that stay in
    # phase couple more strongly. Held fixed for the whole rollout the graph has
    # no way to sharpen a forming group, and the PLV matrix stays near-uniform
    # (measured mean 0.763). 0.0 disables it.
    graph_feedback_strength: float = 0.0
    graph_feedback_momentum: float = 0.9

    # Kuramoto dynamics.
    #
    # freq_gain lets the image set oscillator frequencies. At 0 the input only
    # pins phases, the system settles to a fixed point, and it globally
    # synchronises, which erases every group distinction. Measured optimum for
    # object binding on CLEVR was around 2.0; 0.0 reproduces the original model.
    k: float = 256.0
    dt: float = 0.1
    freq_gain: float = 2.0

    # Pulse coupling: spikes act back on the phases through the same graph.
    # Without it the flow is one-way and the spiking layers receive no gradient
    # at all under the usual loss weights, so they sit at their initialisation
    # and contribute nothing. 0.0 reproduces the original one-way model.
    spike_pulse_gain: float = 0.0

    # Dendritic SNN layer
    low_n: float = -4.0
    high_n: float = 0.0
    branch: int = 4

    # Membrane layer.
    #
    # The defaults were never matched to the signal scale. Measured on a trained
    # checkpoint: the membrane oscillates with a standard deviation of 0.103
    # while vth is 0.5, so the threshold sits about five times the amplitude and
    # the network cannot fire (spike rate 0.0002). Its own binding signal is
    # 7.5x weaker than the phases it is driven by, because tau_m ~ U(0, 4) gives
    # a leak of about 0.85 that low-passes away the oscillation carrying the
    # information. Lowering the tau range shortens the time constant so the
    # membrane tracks the rhythm instead of averaging it.
    membrane_vth: float = 0.06
    membrane_low_m: float = -4.0
    membrane_high_m: float = 0.0

    # Keep oscillator components separate through the dendritic and membrane
    # layers so spike synchrony can be measured per component at readout time.
    spike_per_component: bool = False

    # "sigmoid" reproduces the original gate, compressed to [0.5, 0.731] so it
    # never closes. "raw" leaves it spanning [0, 1]. "phase_mean" reduces the
    # osc_dim axis before the sine, which measured 0.216 against 0.067 for the
    # same information reduced the other way round.
    gate_mode: str = "raw"

    # Object-group based classification
    spike_classify_method: str = "spike_rhythm"
    spike_rhythm_threshold: float = 0.8
    spike_rhythm_min_group_size: int = 2
    spike_rhythm_return_all_groups: bool = False
    spike_interval_size: int = 1
    spike_interval_threshold: float = 0.5
    spike_interval_min_group_size: int = 1
    spike_interval_include_partial: bool = True
    spike_spatial_grid_size: object = 16
    spike_spatial_threshold: float = 0.5
    spike_spatial_min_group_size: int = 2
    spike_spatial_activity_source: str = "sigmoid_membrane"
    spike_spatial_time_aggregate: str = "mean"

    def validate(self):
        error_bound.validate_hyperparameter_s2net_hyperparameters_validate(self)
        return self


DEFAULT_HYPERPARAMETERS = S2NetHyperparameters()
