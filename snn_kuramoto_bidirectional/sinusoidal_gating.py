import torch


def sinusoidal_gating(x, T, f_proj, enc, kuramoto, theta, sc, phase_delay_steps):
    theta_hist = []
    feats_list = []
    mask_hidden_list = []

    for t in range(T):
        x_t = x[:, t, :]
        z_t = f_proj(x_t)
        gamma_t = enc(z_t)

        theta = kuramoto(theta, gamma_t, A=sc)
        theta_hist.append(theta)

        idx = max(0, t - phase_delay_steps)
        theta_mean = theta_hist[idx].mean(dim=-1)
        mask_116 = 0.5 * (1.0 + torch.sin(theta_mean))

        phase_feat = torch.sin(theta)
        phase_feat_gated = phase_feat * mask_116.unsqueeze(-1)
        feats_list.append(phase_feat_gated.unsqueeze(1))

        mask_hidden = torch.sigmoid(mask_116)
        mask_hidden_list.append(mask_hidden)

    return feats_list, mask_hidden_list
