"""Central input/configuration guards for the model, losses and classifiers.

Call-site wrappers preserve the original check order and exception messages.
Numerical stabilizers, successful fallbacks and computational branches remain
in their original modules. This module never mutates input tensors or settings.
"""

def validate_gamma_initializer_feature_patch_gamma_initializer_init(grid_size, patch_size, reduction):
    """Validate gamma_initializer.FeaturePatchGammaInitializer.__init__ inputs at the call site."""
    if grid_size is None and patch_size is None:
        raise ValueError("Either grid_size or patch_size must be provided.")
    if grid_size is not None and patch_size is not None:
        raise ValueError("Use either grid_size or patch_size, not both.")
    if reduction not in {"mean", "max"}:
        raise ValueError('reduction must be "mean" or "max".')


def validate_gamma_initializer_feature_patch_gamma_initializer_num_oscillators(out_h, out_w):
    """Validate gamma_initializer.FeaturePatchGammaInitializer.num_oscillators inputs at the call site."""
    if out_h <= 0 or out_w <= 0:
        raise ValueError("patch_size is larger than feature_map_size.")


def validate_gamma_initializer_pair(value, name):
    """Validate gamma_initializer._pair inputs at the call site."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_gamma_initializer_pair_2(first, second, name):
    """Validate gamma_initializer._pair inputs at the call site."""
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive.")


def validate_graph_generator_image_conditioned_graph_init(top_k):
    """Validate graph_generator.ImageConditionedGraph.__init__ inputs at the call site."""
    if int(top_k) < 1:
        raise ValueError("top_k must be at least 1.")


def validate_graph_generator_image_conditioned_graph_init_2(grid_size, spatial_decay):
    """Validate graph_generator.ImageConditionedGraph.__init__ inputs at the call site."""
    if grid_size is None:
        raise ValueError("grid_size is required when spatial_decay is set.")
    if not 0.0 < float(spatial_decay) < 1.0:
        raise ValueError("spatial_decay must lie in (0, 1).")


def validate_graph_generator_image_conditioned_graph_forward(gamma):
    """Validate graph_generator.ImageConditionedGraph.forward inputs at the call site."""
    if gamma.dim() != 3:
        raise ValueError("gamma must have shape [B, C, N] or [B, N, C].")


def validate_hierarchical_spike_classifier_validate_inputs(level_spikes, grid_sizes, min_group_size, time_aggregate):
    """Validate hierarchical_spike_classifier._validate_inputs inputs at the call site."""
    if not isinstance(level_spikes, (list, tuple)) or len(level_spikes) < 2:
        raise ValueError("level_spikes must contain at least two levels.")
    if not isinstance(grid_sizes, (list, tuple)) or len(grid_sizes) != len(level_spikes):
        raise ValueError("grid_sizes must have the same length as level_spikes.")
    if int(min_group_size) <= 0:
        raise ValueError("min_group_size must be positive.")
    if time_aggregate not in {"any", "per_time"}:
        raise ValueError('time_aggregate must be "any" or "per_time".')


def validate_hierarchical_spike_classifier_validate_inputs_2(spikes, level_idx):
    """Validate hierarchical_spike_classifier._validate_inputs inputs at the call site."""
    import torch

    if not torch.is_tensor(spikes):
        raise ValueError(f"level_spikes[{level_idx}] must be a tensor.")
    if spikes.dim() != 3:
        raise ValueError(f"level_spikes[{level_idx}] must have shape [B, N, T].")


def validate_hierarchical_spike_classifier_validate_inputs_3(grid_h, grid_w, spikes, level_idx):
    """Validate hierarchical_spike_classifier._validate_inputs inputs at the call site."""
    if spikes.size(1) != grid_h * grid_w:
        raise ValueError(
            f"level_spikes[{level_idx}] has {spikes.size(1)} oscillators, "
            f"but grid {grid_h}x{grid_w} has {grid_h * grid_w}."
        )


def validate_hierarchical_spike_classifier_parse_grid_size(value):
    """Validate hierarchical_spike_classifier._parse_grid_size inputs at the call site."""
    if value <= 0:
        raise ValueError("grid size must be positive.")


def validate_hierarchical_spike_classifier_parse_grid_size_2(height, width):
    """Validate hierarchical_spike_classifier._parse_grid_size inputs at the call site."""
    if height <= 0 or width <= 0:
        raise ValueError("grid size values must be positive.")


def validate_hyperparameter_s2net_hyperparameters_validate(self):
    """Validate hyperparameter.S2NetHyperparameters.validate inputs at the call site."""
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
    if self.spike_per_component and self.gate_mode == "phase_mean":
        raise ValueError(
            'spike_per_component requires gate_mode "raw", "sigmoid", "centered_raw", "signed_mask", or "phasor_imag_raw"; '
            '"phase_mean" removes the component axis.'
        )
    if self.gate_mode not in {"sigmoid", "raw", "phase_mean", "centered_raw", "signed_mask", "phasor_imag_raw"}:
        raise ValueError('gate_mode must be "sigmoid", "raw", "phase_mean", "centered_raw", "signed_mask", or "phasor_imag_raw".')
    if self.in_channels != 3:
        raise ValueError("in_channels must be 3 because the model is fixed to RGB input.")
    if self.kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
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


def validate_image_reconstruct_decode_oscillator_features(gamma_samples):
    """Validate image_reconstruct.decode_oscillator_features inputs at the call site."""
    if gamma_samples.dim() < 2:
        raise ValueError("gamma_samples must have shape [..., num_osci].")
    if gamma_samples.numel() == 0:
        raise ValueError("gamma_samples must not be empty.")


def validate_image_reconstruct_decode_oscillator_features_2(flat_gamma):
    """Validate image_reconstruct.decode_oscillator_features inputs at the call site."""
    if flat_gamma.size(0) < 2:
        raise ValueError("At least two gamma samples are required for std.")


def validate_image_reconstruct_decode_oscillator_features_3(steps):
    """Validate image_reconstruct.decode_oscillator_features inputs at the call site."""
    if steps.dim() != 1 or steps.numel() == 0:
        raise ValueError("sigma_steps must be a non-empty one-dimensional sequence.")


def validate_image_reconstruct_decode_oscillator_features_from_checkpoint(state_dict):
    """Validate image_reconstruct.decode_oscillator_features_from_checkpoint inputs at the call site."""
    if not isinstance(state_dict, dict):
        raise ValueError("The checkpoint must contain a decoder state_dict.")


def validate_image_reconstruct_decode_oscillator_features_from_checkpoint_2(first_weight, output_weight):
    """Validate image_reconstruct.decode_oscillator_features_from_checkpoint inputs at the call site."""
    if first_weight is None or output_weight is None:
        raise ValueError(
            "The checkpoint does not match a FeatureMapAutoEncoder decoder."
        )


def validate_image_reconstruct_decode_oscillator_features_from_checkpoint_3(decoder_hidden_dim, inferred_hidden_dim, height, width, output_weight):
    """Validate image_reconstruct.decode_oscillator_features_from_checkpoint inputs at the call site."""
    if decoder_hidden_dim is not None and int(decoder_hidden_dim) != inferred_hidden_dim:
        raise ValueError(
            f"decoder_hidden_dim={decoder_hidden_dim} does not match checkpoint "
            f"dimension {inferred_hidden_dim}."
        )
    if int(output_weight.shape[0]) != height * width:
        raise ValueError(
            "input_size does not match the decoder checkpoint output size."
        )


def validate_image_reconstruct_decode_oscillator_features_from_autoencoder_checkpoint(checkpoint):
    """Validate image_reconstruct.decode_oscillator_features_from_autoencoder_checkpoint inputs at the call site."""
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must be a dictionary.")


def validate_image_reconstruct_decode_oscillator_features_from_autoencoder_checkpoint_2(decoder_first_weight, decoder_output_weight, projection_weight):
    """Validate image_reconstruct.decode_oscillator_features_from_autoencoder_checkpoint inputs at the call site."""
    if decoder_first_weight is None or decoder_output_weight is None or projection_weight is None:
        raise ValueError("The checkpoint does not match a FeatureMapAutoEncoder.")


def validate_image_reconstruct_decode_oscillator_features_from_autoencoder_checkpoint_3(flat_output, side):
    """Validate image_reconstruct.decode_oscillator_features_from_autoencoder_checkpoint inputs at the call site."""
    if side * side != flat_output:
        raise ValueError("Cannot infer square decoder output size from checkpoint.")


def validate_image_reconstruct_validate_decoded_feature_maps(expected_batch_size, feature_maps):
    """Validate image_reconstruct._validate_decoded_feature_maps inputs at the call site."""
    if feature_maps.dim() != 4 or feature_maps.size(0) != expected_batch_size:
        raise ValueError(
            "decoder must return feature maps shaped [B, 1, H, W]."
        )
    if feature_maps.size(1) != 1:
        raise ValueError("decoder output must contain exactly one channel.")


def validate_image_reconstruct_validate_input_size(input_size):
    """Validate image_reconstruct._validate_input_size inputs at the call site."""
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 2:
        raise ValueError("input_size must be a (height, width) pair.")


def validate_image_reconstruct_validate_input_size_2(height, width):
    """Validate image_reconstruct._validate_input_size inputs at the call site."""
    if height <= 0 or width <= 0:
        raise ValueError("input_size values must be positive.")


def validate_image_reconstruct_maximize_oscillator_images_from_checkpoint(state_dict):
    """Validate image_reconstruct.maximize_oscillator_images_from_checkpoint inputs at the call site."""
    if not isinstance(state_dict, dict):
        raise ValueError("The checkpoint must contain an encoder state_dict.")


def validate_image_reconstruct_maximize_oscillator_images_from_checkpoint_2(conv_weights, projection_weight):
    """Validate image_reconstruct.maximize_oscillator_images_from_checkpoint inputs at the call site."""
    if not conv_weights or projection_weight is None:
        raise ValueError(
            "The checkpoint does not match a FeatureMapCNNEncoder state_dict."
        )


def validate_image_reconstruct_maximize_oscillator_images(height, width, steps, lr):
    """Validate image_reconstruct.maximize_oscillator_images inputs at the call site."""
    if height <= 0 or width <= 0:
        raise ValueError("input_size values must be positive.")
    if int(steps) <= 0:
        raise ValueError("steps must be positive.")
    if float(lr) <= 0:
        raise ValueError("lr must be positive.")


def validate_image_reconstruct_maximize_oscillator_images_2(value_range):
    """Validate image_reconstruct.maximize_oscillator_images inputs at the call site."""
    if len(value_range) != 2 or value_range[0] >= value_range[1]:
        raise ValueError("value_range must be a (minimum, maximum) pair.")


def validate_image_reconstruct_maximize_oscillator_images_3(gamma, num_osci):
    """Validate image_reconstruct.maximize_oscillator_images inputs at the call site."""
    if gamma.shape != (num_osci, num_osci):
        raise ValueError(
            "gamma_initializer must return [B, num_osci] for input "
            "shaped [B, 1, H, W]."
        )


def validate_image_reconstruct_save_feature_map_grid(feature_maps):
    """Validate image_reconstruct.save_feature_map_grid inputs at the call site."""
    if feature_maps.dim() != 3:
        raise ValueError("feature_maps must have shape [D, H, W] or [D, 1, H, W].")


def validate_input_layer_generator_c_n_n_feature_encoder_init(num_kernels, kernel_size, in_channels):
    """Validate input_layer_generator.CNNFeatureEncoder.__init__ inputs at the call site."""
    if num_kernels <= 0:
        raise ValueError("num_kernels must be positive.")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")


def validate_input_layer_generator_c_n_n_feature_encoder_validate_image_size(channels, self, width, height):
    """Validate input_layer_generator.CNNFeatureEncoder._validate_image_size inputs at the call site."""
    if channels != self.in_channels:
        raise ValueError(
            f"Expected {self.in_channels} input channels, but got {channels}."
        )
    if width < self.kernel_size or height < self.kernel_size:
        raise ValueError(
            "Image width and height must both be at least as large as kernel_size."
        )


def validate_input_layer_generator_c_n_n_feature_decoder_init(num_kernels, kernel_size, out_channels):
    """Validate input_layer_generator.CNNFeatureDecoder.__init__ inputs at the call site."""
    if num_kernels <= 0:
        raise ValueError("num_kernels must be positive.")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    if out_channels <= 0:
        raise ValueError("out_channels must be positive.")


def validate_kuramoto_layer_graph_vector_kuramoto_forward(H, gamma):
    """Validate kuramoto_layer.graphVectorKuramoto.forward inputs at the call site."""
    if gamma.size(1) != H:
        raise ValueError(
            f"gamma has {gamma.size(1)} oscillators, but theta has {H}."
        )


def validate_loss_function_spike_rate_loss(spikes):
    """Validate loss_function.spike_rate_loss inputs at the call site."""
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, N, T].")


def validate_loss_function_sample_activity_diversity_loss(activity):
    """Validate loss_function.sample_activity_diversity_loss inputs at the call site."""
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")


def validate_loss_function_spatial_compactness_loss(grid_h, grid_w, activity):
    """Validate loss_function.spatial_compactness_loss inputs at the call site."""
    if activity.size(1) != grid_h * grid_w:
        raise ValueError(
            f"activity has {activity.size(1)} oscillators, but grid "
            f"{grid_h}x{grid_w} has {grid_h * grid_w}."
        )


def validate_loss_function_activity_area_loss(activity, min_area, max_area):
    """Validate loss_function.activity_area_loss inputs at the call site."""
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if min_area < 0.0 or max_area > 1.0 or min_area > max_area:
        raise ValueError("min_area and max_area must satisfy 0 <= min <= max <= 1.")


def validate_loss_function_activity_contrast_loss(activity, target_std):
    """Validate loss_function.activity_contrast_loss inputs at the call site."""
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if target_std < 0.0:
        raise ValueError("target_std must be non-negative.")


def validate_loss_function_phase_locking_value(theta):
    """Validate loss_function.phase_locking_value inputs at the call site."""
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")


def validate_loss_function_phase_locking_value_2(settle, phase):
    """Validate loss_function.phase_locking_value inputs at the call site."""
    if int(settle) >= phase.size(1):
        raise ValueError("settle must be smaller than the number of steps.")


def validate_loss_function_signal_synchrony(signal):
    """Validate loss_function.signal_synchrony inputs at the call site."""
    if signal.dim() != 3:
        raise ValueError("signal must have shape [B, N, T].")


def validate_loss_function_signal_synchrony_2(settle, trace):
    """Validate loss_function.signal_synchrony inputs at the call site."""
    if int(settle) >= trace.size(2):
        raise ValueError("settle must be smaller than the number of steps.")


def validate_loss_function_plv_spatial_coherence_loss(num_nodes, grid_h, grid_w):
    """Validate loss_function.plv_spatial_coherence_loss inputs at the call site."""
    if grid_h * grid_w != num_nodes:
        raise ValueError(
            f"patch grid {grid_h}x{grid_w} does not match {num_nodes} oscillators."
        )


def validate_loss_function_object_overlap_loss(object_groups):
    """Validate loss_function.object_overlap_loss inputs at the call site."""
    if object_groups is None:
        raise ValueError("object_groups must not be None.")
    if not isinstance(object_groups, (list, tuple)):
        raise ValueError("object_groups must be a list with length B.")


def validate_loss_function_off_diagonal(matrix):
    """Validate loss_function._off_diagonal inputs at the call site."""
    if matrix.dim() != 3:
        raise ValueError("matrix must have shape [B, N, N].")


def validate_loss_function_prepare_sc(batch_size, sc):
    """Validate loss_function._prepare_sc inputs at the call site."""
    if sc.size(0) != batch_size:
        raise ValueError(f"sc batch size {sc.size(0)} does not match spikes batch size {batch_size}.")


def validate_loss_function_parse_grid_size(value):
    """Validate loss_function._parse_grid_size inputs at the call site."""
    if value <= 0:
        raise ValueError("patch_grid_size must be positive.")


def validate_loss_function_parse_grid_size_2(height, width):
    """Validate loss_function._parse_grid_size inputs at the call site."""
    if height <= 0 or width <= 0:
        raise ValueError("patch_grid_size values must be positive.")


def validate_s2net_cls_gamma_generator_forward(self, gamma_seq):
    """Validate s2net_cls.GammaGenerator.forward inputs at the call site."""
    if gamma_seq.size(-1) != self.in_dim:
        raise ValueError(
            f"Patch gamma produced {gamma_seq.size(-1)} oscillators, "
            f"but hparams.num_regions is {self.in_dim}."
        )


def validate_s2net_cls_gamma_generator_image_to_feature_maps(x):
    """Validate s2net_cls.GammaGenerator._image_to_feature_maps inputs at the call site."""
    if x.dim() != 4:
        raise ValueError("x must have shape [B, 3, H, W]. Use B=1 for one image.")
    if x.size(1) != 3:
        raise ValueError(f"Expected RGB input with 3 channels, but got {x.size(1)}.")


def validate_s2net_cls_s2net_core_init(graph_mode):
    """Validate s2net_cls.S2NetCore.__init__ inputs at the call site."""
    if graph_mode != "learned":
        raise ValueError(
            "hparams.sc must be generated before constructing S2NetCore, "
            'or set graph_mode="learned" to build the graph per image.'
        )


def validate_s2net_cls_s2net_core_init_2(expected_shape, sc):
    """Validate s2net_cls.S2NetCore.__init__ inputs at the call site."""
    if tuple(sc.shape) != expected_shape:
        raise ValueError(
            f"hparams.sc must have shape {expected_shape}, but got {tuple(sc.shape)}."
        )


def validate_s2net_cls_s2net_core_forward(gamma_seq, num_time_steps):
    """Validate s2net_cls.S2NetCore.forward inputs at the call site."""
    if gamma_seq.dim() != 3:
        raise ValueError(
            "gamma_seq must have shape [B, T, num_regions] in sequence drive mode. "
            "Use B=1 for one sample."
        )
    if num_time_steps is not None:
        raise ValueError(
            "num_time_steps only applies to static drive mode; in sequence mode "
            "the recurrent length is fixed by the gamma channel axis."
        )


def validate_s2net_cls_s2net_core_forward_2(T):
    """Validate s2net_cls.S2NetCore.forward inputs at the call site."""
    if T <= 0:
        raise ValueError("num_time_steps must be positive.")


def validate_gamma_to_drive_oscillators(self, gamma):
    """Validate GammaToDrive scalar-input oscillator count."""
    if gamma.size(1) != self.in_dim:
        raise ValueError(
            f"gamma has {gamma.size(1)} oscillators, but num_regions is {self.in_dim}."
        )


def validate_gamma_to_drive_dimensions(gamma):
    """Validate GammaToDrive non-scalar input dimensions."""
    if gamma.dim() != 3:
        raise ValueError(
            "static gamma must have shape [B, N], [B, N, osc_dim], or [B, C, N]."
        )


def validate_gamma_to_drive_channels(self, gamma, channel_projection):
    """Validate GammaToDrive channel projection availability and input count."""
    if channel_projection is None:
        raise ValueError("gamma_channel_proj is unavailable; rebuild the core in static mode.")
    if gamma.size(1) != self.T:
        raise ValueError(
            f"gamma has {gamma.size(1)} channels, but num_feature_maps is {self.T}."
        )


def validate_sc_generator_gamma_sampes(images):
    """Validate sc_generator.gamma_sampes inputs at the call site."""
    if images.dim() != 4:
        raise ValueError("images must have shape [num_images, 3, H, W].")


def validate_sc_generator_gamma_sampes_2(gamma_initializer_path):
    """Validate sc_generator.gamma_sampes inputs at the call site."""
    if gamma_initializer_path is None:
        raise ValueError("gamma_initializer_path is required for autoencoder gamma mode.")


def validate_sc_generator_gamma_sampes_3(hparams, gamma_seq):
    """Validate sc_generator.gamma_sampes inputs at the call site."""
    if gamma_seq.size(-1) != hparams.num_regions:
        raise ValueError(
            f"Patch gamma produced {gamma_seq.size(-1)} oscillators, "
            f"but hparams.num_regions is {hparams.num_regions}."
        )


def validate_sc_generator_pearson_cor_sc(gamma_samples):
    """Validate sc_generator.pearson_cor_sc inputs at the call site."""
    if gamma_samples.dim() != 2:
        raise ValueError(
            "gamma_samples must have shape [num_gamma_samples, num_regions]."
        )
    if gamma_samples.size(0) < 2:
        raise ValueError("At least two gamma samples are required.")


def validate_sinusoidal_gating_sinusoidal_gating(gate_mode):
    """Validate sinusoidal_gating.sinusoidal_gating inputs at the call site."""
    if gate_mode not in {"sigmoid", "raw", "phase_mean", "centered_raw", "signed_mask", "phasor_imag_raw"}:
        raise ValueError('gate_mode must be "sigmoid", "raw", "phase_mean", "centered_raw", "signed_mask", or "phasor_imag_raw".')


def validate_spike_classifier_spike_rhythm(spikes, min_group_size):
    """Validate spike_classifier.spike_rhythm inputs at the call site."""
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, num_oscillators, T]. Use B=1 for one sample.")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2.")


def validate_spike_classifier_spike_interval(core_out, interval_size, min_group_size):
    """Validate spike_classifier.spike_interval inputs at the call site."""
    if core_out.dim() != 3:
        raise ValueError("core_out must have shape [B, num_oscillators, T]. Use B=1 for one sample.")
    if interval_size <= 0:
        raise ValueError("interval_size must be positive.")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive.")


def validate_spike_classifier_spike_spatial_components(activity, min_group_size, activity_source, time_aggregate):
    """Validate spike_classifier.spike_spatial_components inputs at the call site."""
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, num_oscillators, T].")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive.")
    if activity_source not in {"spikes", "membrane", "sigmoid_membrane"}:
        raise ValueError('activity_source must be "spikes", "membrane", or "sigmoid_membrane".')
    if time_aggregate not in {"max", "mean"}:
        raise ValueError('time_aggregate must be "max" or "mean".')


def validate_spike_classifier_spike_spatial_components_2(grid_h, grid_w, activity):
    """Validate spike_classifier.spike_spatial_components inputs at the call site."""
    if activity.size(1) != grid_h * grid_w:
        raise ValueError(
            f"activity has {activity.size(1)} oscillators, but patch grid "
            f"{grid_h}x{grid_w} has {grid_h * grid_w}."
        )


