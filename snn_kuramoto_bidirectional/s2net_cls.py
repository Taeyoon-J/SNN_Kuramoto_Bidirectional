import torch
import torch.nn as nn
from kuramoto_layer import graphVectorKuramoto
from dendric_layer import DendricLayer
from membrane_layer import MembraneLayer
from sinusoidal_gating import sinusoidal_gating
from input_layer_generator import CNNFeatureEncoder
from gamma_initializer import FeatureMapCNNEncoder

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

        self.input_layer = CNNFeatureEncoder(
            num_kernels=self.T,
            kernel_size=getattr(args, "kernel_size", 3),
            in_channels=getattr(args, "in_channels", 1),
            bias=True,
        )
        self.gamma_initializer = FeatureMapCNNEncoder(
            num_osci=self.in_dim,
            in_channels=1,
            dropout=getattr(args, "gamma_dropout", 0.0),
        )

        self.kuramoto = graphVectorKuramoto(
            N=self.in_dim, D=self.osc_dim, K=args.k, dt=args.dt, alpha_scale=1.0, device=device
        )

        self.core = nn.Module()
        self.core.dendric_layer = DendricLayer(
            input_dim=self.in_dim,
            output_dim=self.in_dim,
            tau_ninitializer='uniform',
            low_n=args.low_n,
            high_n=args.high_n,
            branch=args.branch,
            device=device,
            bias=True,
            input_vector_dim=self.osc_dim,
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
        x = x.to(self.device)
        sc = sc.to(self.device)
        feature_maps = self._image_to_feature_maps(x)
        B, T, height, width = feature_maps.shape
        gamma_seq = self.gamma_initializer(
            feature_maps.reshape(B * T, 1, height, width)
        ).view(B, T, self.in_dim)

        theta = torch.zeros(B, self.in_dim, self.osc_dim, device=self.device)
        # === 1. Dual-Stream Dynamics ===
        theta_hist = []
        for t in range(T):
            gamma_t = gamma_seq[:, t, :]
            theta = self.kuramoto(theta, gamma_t, A=sc)
            theta_hist.append(theta)

        feats_list, mask_hidden_list = sinusoidal_gating(
            theta_hist,
            T,
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
            g_t = all_hidden_masks[:, i, :]

            l_input, mask = self.core.dendric_layer(
                input_t,
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

    def _image_to_feature_maps(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError("x must have shape [H, W], [B, H, W], or [B, C, H, W].")

        return self.input_layer(x)
