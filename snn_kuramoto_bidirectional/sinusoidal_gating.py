import torch


def sinusoidal_gating(theta_hist, t, phase_delay_steps, gate_mode="sigmoid"):
    """Return dendritic drive and membrane gate for a single time step.

    theta_hist contains phases shaped [B, N, D], including the current step.
    The drive is [B, N, 1] in phase_mean mode, otherwise [B, N, D].
    The membrane gate is shaped [B, N].
    """
    theta = theta_hist[t]
    delayed = theta_hist[max(0, t - phase_delay_steps)]
    mask = 0.5 * (1.0 + torch.sin(delayed.mean(dim=-1)))
    if gate_mode == "phase_mean":  # readout mode: README 4.3 training
        gamma_wave_t = torch.sin(theta.mean(dim=-1)).unsqueeze(-1)
    else:
        gamma_wave_t = torch.sin(theta) * mask.unsqueeze(-1)
    g_wave_t = torch.sigmoid(mask) if gate_mode == "sigmoid" else mask  # readout mode: else -> mask (phase_mean)
    return gamma_wave_t, g_wave_t


# Legacy full-history implementation (not in use).
# def sinusoidal_gating(theta_hist, T, phase_delay_steps, gate_mode="sigmoid"):
#     """
#     Turn oscillator phases into gated drive for the spiking layers.
#
#     gate_mode:
#         "sigmoid" (default) reproduces the original behaviour.
#         "raw" passes the gate through unsquashed.
#         "phase_mean" reduces the osc_dim axis BEFORE taking the sine, and leaves
#         the features unmultiplied.
#
#     Why "phase_mean" exists. The osc_dim components of one oscillator are not in
#     phase with each other; measured mean deviation from their own unit mean is
#     1.57 rad, a quarter cycle. Handing sin(theta) to the dendrite as [B, N, D]
#     therefore means the sines are averaged, and sinusoids a quarter cycle apart
#     cancel rather than reinforce. Reducing the phases first and taking one sine
#     keeps a clean single oscillation:
#
#         sin(theta.mean(D))   binding ARI 0.216
#         sin(theta).mean(D)   binding ARI 0.067      the same information, 3.2x worse
#
#     The mean is arithmetic rather than circular, matching phase_locking_value so
#     that what the loss reads and what the neurons receive agree.
#
#     The gate itself, 0.5 * (1 + sin(theta_mean)), already spans [0, 1]. Squashing
#     it again through a sigmoid compresses it to [0.5, 0.731], which has two
#     measured consequences: MembraneLayer's ``torch.where(g_wave_t == 0, ...)``
#     freeze can never trigger, so the gate never actually gates, and spikes are
#     scaled down by a near-constant factor instead of being switched.
#     """
#     error_bound.validate_sinusoidal_gating_sinusoidal_gating(gate_mode)
#
#     feats_list = []
#     mask_hidden_list = []
#
#     for t in range(T):
#         theta = theta_hist[t]
#         idx = max(0, t - phase_delay_steps)
#         theta_mean = theta_hist[idx].mean(dim=-1)
#         mask = 0.5 * (1.0 + torch.sin(theta_mean))
#
#         if gate_mode == "phase_mean":
#             # reduce first, then one sine; the gate acts only on the membrane
#             phase_feat_gated = torch.sin(theta.mean(dim=-1)).unsqueeze(-1)
#         else:
#             phase_feat = torch.sin(theta)
#             phase_feat_gated = phase_feat * mask.unsqueeze(-1)
#         feats_list.append(phase_feat_gated.unsqueeze(1))
#
#         mask_hidden = torch.sigmoid(mask) if gate_mode == "sigmoid" else mask
#         mask_hidden_list.append(mask_hidden)
#
#     return feats_list, mask_hidden_list
