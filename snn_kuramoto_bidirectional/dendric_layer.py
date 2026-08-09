import torch
import torch.nn as nn
import torch.nn.functional as F


class DendricLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tau_ninitializer='uniform', low_n=0, high_n=4, branch=4,
                 device='cpu', bias=True, input_vector_dim=1,
                 aggregation_mode='sum'):
        """
        Rhythm-Modulated Recurrent SNN Layer (External Gating Version)
        """
        super(DendricLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.input_vector_dim = int(input_vector_dim)
        if aggregation_mode not in {'sum', 'relu_sum', 'abs_sum'}:
            raise ValueError(
                'aggregation_mode must be "sum", "relu_sum", or "abs_sum".'
            )
        self.aggregation_mode = aggregation_mode

        # Dendritic Parameters
        self.oscillator_dense = nn.Linear(self.input_vector_dim + 1, branch, bias=bias)

        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, branch))
        self.branch = branch

        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n, low_n, high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n, low_n)

        self.h = None
        self.dense_i = None

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
        dense_outputs = []
        for i in range(self.output_dim):
            k_input = torch.cat(
                (
                    gamma_wave[:, i, :].float(),
                    prev_spike[:, i:i + 1],
                ),
                dim=1,
            )
            dense_i = self.oscillator_dense(k_input)
            dense_outputs.append(dense_i)
            h_i = beta[i] * self.h[:, i, :] + (1 - beta[i]) * dense_i
            next_h.append(h_i)

        self.dense_i = torch.stack(dense_outputs, dim=1)
        self.h = torch.stack(next_h, dim=1)
        branch_dim = self.h.dim() - 1
        if self.aggregation_mode == 'sum':
            h_wave = self.h.sum(dim=branch_dim, keepdim=False)
        elif self.aggregation_mode == 'relu_sum':
            h_wave = F.relu(self.h).sum(dim=branch_dim, keepdim=False)
        else:
            h_wave = self.h.abs().sum(dim=branch_dim, keepdim=False)

        return h_wave
