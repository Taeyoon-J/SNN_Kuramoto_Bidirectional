import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SNN.SNN_layers.spike_neuron import *
from SNN.SNN_layers.dendric_layer import DendricLayer
# from SNN.SNN_layers.dense_neuron import * 이거는 아예 안쓰이는데? 

# ============================================================
# 1. Rhy_spike_rnn
# ============================================================
class Rhy_spike_rnn_test_denri_wotanh_new(nn.Module):
    def __init__(self, input_dim, output_dim, tau_minitializer='uniform', low_m=0, high_m=4,
                 tau_ninitializer='uniform', low_n=0, high_n=4, vth=0.5, dt=4, branch=4, 
                 device='cpu', bias=True):
        """
        Rhythm-Modulated Recurrent SNN Layer (External Gating Version)
        """
        super(Rhy_spike_rnn_test_denri_wotanh_new, self).__init__()
        self.dendric_layer = DendricLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            tau_minitializer=tau_minitializer,
            low_m=low_m,
            high_m=high_m,
            tau_ninitializer=tau_ninitializer,
            low_n=low_n,
            high_n=high_n,
            vth=vth,
            dt=dt,
            branch=branch,
            device=device,
            bias=bias,
        )

    def set_neuron_state(self, batch_size):
        self.dendric_layer.set_neuron_state(batch_size)

    def forward(self, input_spike, external_rhythm_mask=None):
        l_input, mask = self.dendric_layer(input_spike, external_rhythm_mask)
        self.dendric_layer.mem, self.dendric_layer.spike = mem_update_pra_rhythm(
            l_input, self.dendric_layer.mem, self.dendric_layer.spike,
            self.dendric_layer.v_th, self.dendric_layer.tau_m,
            mask,
            self.dendric_layer.dt, device=self.dendric_layer.device
        )
        return self.dendric_layer.mem, self.dendric_layer.spike
