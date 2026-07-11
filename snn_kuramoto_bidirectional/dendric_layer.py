import torch
import torch.nn as nn


class DendricLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tau_ninitializer='uniform', low_n=0, high_n=4, branch=4,
                 device='cpu', bias=True, input_vector_dim=1):
        """
        Rhythm-Modulated Recurrent SNN Layer (External Gating Version)
        """
        super(DendricLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.input_vector_dim = int(input_vector_dim)

        # Dendritic Parameters
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
        """
        # 1. Dendritic Integration
        beta = torch.sigmoid(self.tau_n)

        next_h = []
        for i in range(self.output_dim):
            k_input = torch.cat(
                (
                    gamma_wave[:, i, :].float(),
                    prev_spike[:, i:i + 1],
                ),
                dim=1,
            )
            dense_i = self.oscillator_dense(k_input)
            h_i = beta[i] * self.h[:, i, :] + (1 - beta[i]) * dense_i
            next_h.append(h_i)

        self.h = torch.stack(next_h, dim=1)
        h_wave = self.h.sum(dim=2, keepdim=False)

        return h_wave
