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
    osc_dim: int = 4

    # SNN recurrent time axis.
    #
    # gamma_drive_mode == "sequence" (legacy): one Kuramoto step per visual
    # channel, so the recurrent length is tied to num_feature_maps.
    # gamma_drive_mode == "static": gamma is a constant sensory drive and the
    # recurrent length is num_time_steps, independent of the channel count.
    # "static" is required for multi-scale/U-Net work, where every level must
    # share one time axis while having a different channel count.
    gamma_drive_mode: str = "sequence"
    num_time_steps: int = 8

    # Map raw gamma values onto a phase range before sin(gamma - theta).
    # Pooled CNN activations have an arbitrary scale, so "none" lets the drive
    # wrap around the sine and lose image specificity.
    gamma_phase_mode: str = "none"

    # Initial oscillator phase. "zeros" makes every image start from the same
    # state, so only the drive term carries image information.
    theta_init: str = "zeros"
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

    # Gamma ordering loss
    gamma_order_lambda: float = 1.0
    gamma_order_mu: float = 1.0
    gamma_order_method: str = "auto"
    gamma_order_exact_max_steps: int = 8
    gamma_order_local_search_passes: int = 5

    # Coupling graph.
    #
    # "static" uses the fixed sc matrix. A fixed graph cannot express which
    # patches belong to the same object, because that changes with the image,
    # so "learned" builds one graph per sample from its own features.
    # Measured on CLEVR at a 16x16 grid: sparse, feature-only, image-conditioned
    # coupling was the only variant that beat no coupling at all.
    graph_mode: str = "static"
    graph_hidden_dim: int = 16
    graph_top_k: int = 8
    graph_coupling_gain: float = 8.0
    graph_temperature: float = 0.1

    # Structural prior on the learned graph. Objects are connected regions, and
    # the features do not know that. A hand-written spatial x similarity kernel
    # scored ARI 0.056 against 0.038 for the randomly initialised graph, so this
    # is a better starting point. None disables it.
    graph_spatial_decay: object = None

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
    k: float = 1.0
    dt: float = 0.1
    freq_gain: float = 0.0

    # Pulse coupling: spikes act back on the phases through the same graph.
    # Without it the flow is one-way and the spiking layers receive no gradient
    # at all under the usual loss weights, so they sit at their initialisation
    # and contribute nothing. 0.0 reproduces the original one-way model.
    spike_pulse_gain: float = 0.0

    # Dendritic SNN layer
    low_n: float = 0.0
    high_n: float = 4.0
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
    membrane_vth: float = 0.5
    membrane_low_m: float = 0.0
    membrane_high_m: float = 4.0

    # "sigmoid" reproduces the original gate, compressed to [0.5, 0.731] so it
    # never closes. "raw" leaves it spanning [0, 1]. "phase_mean" reduces the
    # osc_dim axis before the sine, which measured 0.216 against 0.067 for the
    # same information reduced the other way round.
    gate_mode: str = "sigmoid"

    # Differentiable readout. Spectral clustering runs after training and is
    # not differentiable, so the objective shapes pairwise synchrony while the
    # metric scores a partition, and nothing ties the two together. A soft
    # k-means head over the dynamics puts the partition in the graph, and the
    # relaxed normalized cut scores it directly. 0 disables it and leaves the
    # state_dict unchanged.
    readout_slots: int = 0
    readout_source: str = "phase"
    readout_signal_dim: object = None
    readout_embed_dim: int = 16
    readout_iters: int = 3
    readout_temperature: float = 0.5

    # Object-group based classification
    spike_classify_method: str = "spike_rhythm"
    spike_rhythm_threshold: float = 0.8
    spike_rhythm_min_group_size: int = 2
    spike_rhythm_return_all_groups: bool = False
    spike_interval_size: int = 1
    spike_interval_threshold: float = 0.5
    spike_interval_min_group_size: int = 1
    spike_interval_include_partial: bool = True
    spike_spatial_grid_size: object = None
    spike_spatial_threshold: float = 0.5
    spike_spatial_min_group_size: int = 2
    spike_spatial_activity_source: str = "sigmoid_membrane"
    spike_spatial_time_aggregate: str = "mean"

    def validate(self):
        if self.num_feature_maps <= 0:
            raise ValueError("num_feature_maps must be positive.")
        if self.num_regions <= 0:
            raise ValueError("num_regions must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.osc_dim <= 0:
            raise ValueError("osc_dim must be positive.")
        if self.gamma_drive_mode not in {"sequence", "static"}:
            raise ValueError('gamma_drive_mode must be "sequence" or "static".')
        if self.num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive.")
        if self.gamma_phase_mode not in {"none", "tanh", "standardize_tanh"}:
            raise ValueError(
                'gamma_phase_mode must be "none", "tanh", or "standardize_tanh".'
            )
        if self.theta_init not in {"zeros", "gamma", "gamma_noise"}:
            raise ValueError('theta_init must be "zeros", "gamma", or "gamma_noise".')
        if self.theta_init_noise < 0:
            raise ValueError("theta_init_noise must be non-negative.")
        if self.graph_mode not in {"static", "learned"}:
            raise ValueError('graph_mode must be "static" or "learned".')
        if self.graph_top_k <= 0:
            raise ValueError("graph_top_k must be positive.")
        if self.graph_coupling_gain <= 0:
            raise ValueError("graph_coupling_gain must be positive.")
        if self.graph_temperature <= 0:
            raise ValueError("graph_temperature must be positive.")
        if self.graph_spatial_decay is not None and not 0.0 < float(self.graph_spatial_decay) < 1.0:
            raise ValueError("graph_spatial_decay must lie in (0, 1).")
        if not 0.0 <= self.graph_feedback_momentum < 1.0:
            raise ValueError("graph_feedback_momentum must lie in [0, 1).")
        if self.membrane_vth <= 0:
            raise ValueError("membrane_vth must be positive.")
        if self.membrane_low_m > self.membrane_high_m:
            raise ValueError("membrane_low_m must not exceed membrane_high_m.")
        if self.spike_pulse_gain < 0:
            raise ValueError("spike_pulse_gain must be non-negative.")
        if self.readout_slots < 0 or self.readout_slots == 1:
            raise ValueError("readout_slots must be 0 (disabled) or at least 2.")
        if self.readout_source not in {"phase", "signal"}:
            raise ValueError('readout_source must be "phase" or "signal".')
        if self.readout_slots > 0 and self.readout_source == "signal" \
                and self.readout_signal_dim is None:
            raise ValueError("readout_signal_dim is required when readout_source is signal.")
        if self.readout_iters <= 0:
            raise ValueError("readout_iters must be positive.")
        if self.readout_temperature <= 0:
            raise ValueError("readout_temperature must be positive.")
        if self.gate_mode not in {"sigmoid", "raw", "phase_mean"}:
            raise ValueError('gate_mode must be "sigmoid", "raw", or "phase_mean".')
        if self.in_channels != 3:
            raise ValueError("in_channels must be 3 because the model is fixed to RGB input.")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if self.gamma_order_method not in {"auto", "exact", "local_search"}:
            raise ValueError('gamma_order_method must be "auto", "exact", or "local_search".')
        if self.gamma_mode not in {"autoencoder", "patch"}:
            raise ValueError('gamma_mode must be "autoencoder" or "patch".')
        if self.gamma_patch_reduction not in {"mean", "max"}:
            raise ValueError('gamma_patch_reduction must be "mean" or "max".')
        if self.gamma_mode == "patch" and self.gamma_patch_grid_size is None and self.gamma_patch_size is None:
            raise ValueError("gamma_patch_grid_size or gamma_patch_size is required when gamma_mode is patch.")
        if self.spike_classify_method not in {"spike_rhythm", "spike_interval", "spatial_components"}:
            raise ValueError('spike_classify_method must be "spike_rhythm", "spike_interval", or "spatial_components".')
        if self.spike_rhythm_min_group_size < 2:
            raise ValueError("spike_rhythm_min_group_size must be at least 2.")
        if self.spike_interval_size <= 0:
            raise ValueError("spike_interval_size must be positive.")
        if self.spike_interval_min_group_size <= 0:
            raise ValueError("spike_interval_min_group_size must be positive.")
        if self.spike_classify_method == "spatial_components" and self.spike_spatial_grid_size is None:
            raise ValueError("spike_spatial_grid_size is required when spike_classify_method is spatial_components.")
        if self.spike_spatial_min_group_size <= 0:
            raise ValueError("spike_spatial_min_group_size must be positive.")
        if self.spike_spatial_activity_source not in {"spikes", "membrane", "sigmoid_membrane"}:
            raise ValueError('spike_spatial_activity_source must be "spikes", "membrane", or "sigmoid_membrane".')
        if self.spike_spatial_time_aggregate not in {"max", "mean"}:
            raise ValueError('spike_spatial_time_aggregate must be "max" or "mean".')
        return self


DEFAULT_HYPERPARAMETERS = S2NetHyperparameters()
