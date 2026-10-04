import torch
import torch.nn as nn


class DendricLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tau_ninitializer='uniform', low_n=0, high_n=4, branch=4,
                 device='cpu', bias=True, input_vector_dim=1,
                 dendritic_projection='shared'):
        """
        Rhythm-Modulated Recurrent SNN Layer (External Gating Version)
        """
        super(DendricLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.input_vector_dim = int(input_vector_dim)
        if dendritic_projection not in ('shared', 'per_region'):
            raise ValueError("dendritic_projection must be 'shared' or 'per_region'")
        self.dendritic_projection = dendritic_projection

        # Dendritic Parameters
        if dendritic_projection == 'shared':
            # Preserve legacy module name, parameter names, initialization and RNG order.
            self.oscillator_dense = nn.Linear(self.input_vector_dim + 1, branch, bias=bias)
        else:
            # Initialize from the exact shared Linear used by the legacy mode.
            # Constructing it first preserves RNG consumption before tau_n.
            base = nn.Linear(self.input_vector_dim + 1, branch, bias=bias)
            self.oscillator_dense_weight = nn.Parameter(
                base.weight.detach().unsqueeze(0).repeat(self.output_dim, 1, 1)
            )
            if bias:
                self.oscillator_dense_bias = nn.Parameter(base.bias.detach().clone())
            else:
                self.register_parameter('oscillator_dense_bias', None)

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
        if self.dendritic_projection == 'shared':
            dense = self.oscillator_dense(k_input)  # [B, N, branch]
        else:
            dense = torch.einsum(
                'bni,nri->bnr', k_input, self.oscillator_dense_weight
            )
            if self.oscillator_dense_bias is not None:
                dense = dense + self.oscillator_dense_bias

        self.h = beta * self.h + (1.0 - beta) * dense
        h_wave = self.h.sum(dim=2, keepdim=False)

        return h_wave
