import torch
import torch.nn as nn
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer

class GraphVectorKuramoto(nn.Module):
    """Wrapper for the Vector Kuramoto implementation."""
    def __init__(self, N, D, K, dt, device):
        super().__init__()
        self.kuramoto = graphVectorKuramoto(
            N=N, D=D, K=K, dt=dt, alpha_scale=1.0, device=device
        )
    
    def forward(self, theta, gamma, A):
        return self.kuramoto(theta, gamma, A=A)
    
class RegionAlignedSNN(nn.Module):
    """
    Bottom-Up Pathway: Rhythm-Modulated Spiking Neural Network.
    Implements the dendritic integration and membrane potential updates described in Eq. 6.
    """
    def __init__(self, T, num_regions, input_feat_dim, num_classes, low_n, high_n, branch, device):
        super().__init__()
        self.input_adapter = nn.Linear(input_feat_dim, 1)
        
        self.dendric_layer = DendricLayer(
            input_dim=num_regions, 
            output_dim=num_regions, 
            tau_ninitializer='uniform', low_n=low_n, high_n=high_n, 
            branch=branch, device=device, bias=True
        )
        self.membrane_layer = MembraneLayer(
            output_dim=num_regions,
            readout_dim=num_classes,
            tau_minitializer='uniform', low_m=0, high_m=4,
            vth=0.5, dt=1, device=device
        )
        self.device = device

    def forward(self, input_4d_seq, gating_signals):
        batch_size, seq_num, _, _ = input_4d_seq.shape
        self.dendric_layer.set_neuron_state(batch_size)
        self.membrane_layer.set_neuron_state(batch_size)
        
        outputs = []
        spikes_hist = [] 

        for i in range(seq_num):
            # Input projection
            input_t = input_4d_seq[:, i, :, :] 
            currents = self.input_adapter(input_t).squeeze(-1) 

            # Rhythm modulation (gating) applied here
            # See Eq. 6: U_i(t) = (1-g_i(t))*U + g_i(t)*V
            g_t = gating_signals[:, i, :]

            l_input, mask = self.dendric_layer(currents, self.membrane_layer.spike, g_t)
            mem_readout, spike_t = self.membrane_layer(l_input, mask)
            spikes_hist.append(spike_t) 

            outputs.append(mem_readout)
            
        outputs = torch.stack(outputs).permute(1, 2, 0) # [B, Classes, T]
        spikes = torch.stack(spikes_hist).permute(1, 2, 0) 

        return outputs, spikes

