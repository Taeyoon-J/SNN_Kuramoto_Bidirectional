import torch
import torch.nn as nn


class DendricLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tau_ninitializer='uniform', low_n=0, high_n=4, branch=4,
                 device='cpu', bias=True, input_vector_dim=1,
                 per_region=False):
        """
        Rhythm-Modulated Recurrent SNN Layer (External Gating Version)
        """
        super(DendricLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.input_vector_dim = int(input_vector_dim)

        # Dendritic Parameters
        #
        # per_region gives every oscillator its own map from (gated drive,
        # previous spike) into its dendritic branches, instead of all of them
        # sharing one.
        #
        # Measured reason: the spiking path holds 1,292 of the model's 200,114
        # parameters and this map is 8 of them -- (input_vector_dim + 1) x branch,
        # shared across all 256 regions. With the phase path frozen and the model
        # fine-tuned from a trained core, gradients reach the spiking layers
        # (norms 4.4e-4 to 2.8e-2) and an alignment loss against the phase matrix
        # still moves only in its sixth decimal at any weight. The phase readout
        # reaches fg_ari 0.7120 where the spike readout reaches 0.6756, so the
        # information is there and the transduction cannot express it: every unit
        # is forced to weigh drive against recurrence identically.
        #
        # The update is unchanged in form -- still an affine map of the same
        # inputs, then the same leaky accumulation over branches.
        self.per_region = bool(per_region)
        if self.per_region:
            self.oscillator_dense = None
            self.region_weight = nn.Parameter(
                torch.empty(self.output_dim, branch, self.input_vector_dim + 1))
            nn.init.kaiming_uniform_(self.region_weight, a=5 ** 0.5)
            self.region_bias = (
                nn.Parameter(torch.zeros(self.output_dim, branch)) if bias else None)
        else:
            self.oscillator_dense = nn.Linear(self.input_vector_dim + 1, branch, bias=bias)

        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, branch))
        self.branch = branch

        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n, low_n, high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n, low_n)

        self.h = None

    def set_neuron_state(self, batch_size):
        self.h = torch.zeros(batch_size, self.output_dim, self.branch).to(self.device)

    def forward(self, gamma_wave, prev_spike):
        """
        Args:
            gamma_wave: [Batch, Output_Dim, Input_Vector_Dim]
            prev_spike: [Batch, Output_Dim]

        ``oscillator_dense`` is shared by every oscillator, so the per-neuron
        loop is applied as one batched matmul instead. This is numerically the
        same update, and it matters once the oscillator count grows from a
        single 8x8 grid to a multi-level pyramid.
        """
        # 1. Dendritic Integration
        beta = torch.sigmoid(self.tau_n).unsqueeze(0)  # [1, N, branch]

        k_input = torch.cat(
            (gamma_wave.float(), prev_spike.unsqueeze(-1)),
            dim=-1,
        )  # [B, N, input_vector_dim + 1]
        if self.per_region:
            # [B, N, in] x [N, branch, in] -> [B, N, branch]
            dense = torch.einsum("bni,nki->bnk", k_input, self.region_weight)
            if self.region_bias is not None:
                dense = dense + self.region_bias.unsqueeze(0)
        else:
            dense = self.oscillator_dense(k_input)  # [B, N, branch]

        self.h = beta * self.h + (1.0 - beta) * dense
        h_wave = self.h.sum(dim=2, keepdim=False)

        return h_wave
