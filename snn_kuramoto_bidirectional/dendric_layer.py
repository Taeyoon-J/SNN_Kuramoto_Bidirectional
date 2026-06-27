import torch
import torch.nn as nn


class DendricLayer(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tau_ninitializer='uniform', low_n=0, high_n=4, branch=4,
                 device='cpu', bias=True):
        """
        Rhythm-Modulated Recurrent SNN Layer (External Gating Version)
        """
        super(DendricLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Dendritic Parameters
        self.pad = ((input_dim + output_dim) // branch * branch + branch - (input_dim + output_dim)) % branch
        self.dense = nn.Linear(input_dim + output_dim + self.pad, output_dim * branch)

        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, branch))
        self.branch = branch

        #  (Mask for weights, not rhythm)
        self.create_weight_mask()

        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n, low_n, high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n, low_n)

        self.d_input = None

    def create_weight_mask(self):
        input_size = self.input_dim + self.output_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.branch, input_size).to(self.device)
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[
                    i * self.branch + j, seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1

    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask

    def set_neuron_state(self, batch_size):
        self.d_input = torch.zeros(batch_size, self.output_dim, self.branch).to(self.device)

    def forward(self, input_spike, prev_spike, external_rhythm_mask=None):
        """
        Args:
            input_spike: [Batch, Input_Dim]
            external_rhythm_mask: [Batch, Output_Dim]
        """
        # 1. Dendritic Integration
        beta = torch.sigmoid(self.tau_n)

        #  padding
        if self.pad > 0:
            padding = torch.zeros(input_spike.size(0), self.pad).to(self.device)
            k_input = torch.cat((input_spike.float(), prev_spike, padding), 1)
        else:
            k_input = torch.cat((input_spike.float(), prev_spike), 1)

        self.d_input = beta * self.d_input + (1 - beta) * self.dense(k_input).reshape(-1, self.output_dim, self.branch)

        l_input = (self.d_input).sum(dim=2, keepdim=False)

        # 2. Somatic Update
        if external_rhythm_mask is None:
            mask = torch.ones_like(prev_spike)
        else:
            mask = external_rhythm_mask

        return l_input, mask
