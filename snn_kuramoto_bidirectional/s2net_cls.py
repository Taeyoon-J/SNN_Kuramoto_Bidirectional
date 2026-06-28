import torch
import torch.nn as nn
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating

class S2NetClassifier(nn.Module):
    """
    S²-Net for Subject Classification.
    
    Differences from Sequence Labeling Model:
    1. Includes a temporal pooling layer (Global Average Pooling) at the end.
    2. Output shape is [Batch, NumClasses] instead of [Batch, NumClasses, Time].
    """
    def __init__(self, T, num_regions, num_classes, args, device="cuda"):
        super().__init__()
        self.T = T
        self.in_dim = int(num_regions)
        self.osc_dim = 4
        self.phase_delay_steps = 2
        self.latent_dim = args.hidden

        # --- Top-Down Pathway (Kuramoto) ---
        self.enc = nn.Linear(self.latent_dim, self.in_dim)
        
        # Enhanced Projection Head (as per your snippet)
        self.f_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        self.kuramoto = graphVectorKuramoto(
            N=self.in_dim, D=self.osc_dim, K=args.k, dt=args.dt, alpha_scale=1.0, device=device
        )

        self.core = nn.Module()
        self.core.input_adapter = nn.Linear(self.osc_dim, 1)
        self.core.dendric_layer = DendricLayer(
            input_dim=self.in_dim,
            output_dim=self.in_dim,
            tau_ninitializer='uniform',
            low_n=args.low_n,
            high_n=args.high_n,
            branch=args.branch,
            device=device,
            bias=True
        )
        self.core.membrane_layer = MembraneLayer(
            output_dim=self.in_dim,
            readout_dim=num_classes,
            tau_minitializer='uniform',
            low_m=0,
            high_m=4,
            vth=0.5,
            dt=1,
            device=device
        )
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, x, sc):
        B, T, N = x.shape
        x = x.to(self.device)
        sc = sc.to(self.device)

        theta = torch.zeros(B, self.in_dim, self.osc_dim, device=self.device)
        # === 1. Dual-Stream Dynamics ===
        feats_list, mask_hidden_list = sinusoidal_gating(
            x,
            T,
            self.f_proj,
            self.enc,
            self.kuramoto,
            theta,
            sc,
            self.phase_delay_steps,
        )

        # === 2. SNN Processing ===
        core_input = torch.cat(feats_list, dim=1)            
        all_hidden_masks = torch.stack(mask_hidden_list, dim=1) 

        batch_size, seq_num, _, _ = core_input.shape
        self.core.dendric_layer.set_neuron_state(batch_size)
        self.core.membrane_layer.set_neuron_state(batch_size)

        outputs = []
        spikes_hist = []

        for i in range(seq_num):
            input_t = core_input[:, i, :, :]
            currents = self.core.input_adapter(input_t).squeeze(-1)
            g_t = all_hidden_masks[:, i, :]

            l_input, mask = self.core.dendric_layer(
                currents,
                self.core.membrane_layer.spike,
                g_t
            )
            mem_readout, spike_t = self.core.membrane_layer(l_input, mask)
            spikes_hist.append(spike_t)
            outputs.append(mem_readout)

        core_out = torch.stack(outputs).permute(1, 2, 0)
        spikes = torch.stack(spikes_hist).permute(1, 2, 0)
        
        # === 3. Temporal Pooling (For Subject Classification) ===
        # Pooling: [Batch, Classes, Time] -> [Batch, Classes]
        logits_pooled = torch.mean(core_out, dim=2) 
        
        return self.logsoftmax(logits_pooled), spikes
