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

        self.d_input = None

    def set_neuron_state(self, batch_size):
        self.d_input = torch.zeros(batch_size, self.output_dim, self.branch).to(self.device)

    def forward(self, input_spike, prev_spike, external_rhythm_mask=None):
        """
        Args:
            input_spike: [Batch, Output_Dim, Input_Vector_Dim]
            external_rhythm_mask: [Batch, Output_Dim]
        """
        # 1. Dendritic Integration
        beta = torch.sigmoid(self.tau_n)

        next_d_input = []
        for i in range(self.output_dim):
            k_input = torch.cat(
                (
                    input_spike[:, i, :].float(),
                    prev_spike[:, i:i + 1],
                ),
                dim=1,
            )
            dense_i = self.oscillator_dense(k_input)
            d_input_i = beta[i] * self.d_input[:, i, :] + (1 - beta[i]) * dense_i
            next_d_input.append(d_input_i)

        self.d_input = torch.stack(next_d_input, dim=1)
        l_input = self.d_input.sum(dim=2, keepdim=False)

        # 2. Somatic Update
        if external_rhythm_mask is None:
            mask = torch.ones_like(prev_spike)
        else:
            mask = external_rhythm_mask

        return l_input, mask
