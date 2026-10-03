
if __package__:
    from . import error_bound
else:
    import error_bound

# kuramoto_layer.py

import torch
import torch.nn as nn
import math

class graphVectorKuramoto(nn.Module):
    """
    Graph-Aware Vector Kuramoto with OT-derived Phase Lags.
    Strictly aligns with Eq. (5) and OT Surrogate mechanics.
    """
    def __init__(self, N, D=2, K=1.0, dt=1.0, alpha_scale=1.0, device="cuda", freq_gain=0.0,
                 spike_pulse_gain=0.0):
        super().__init__()
        self.N = N
        self.D = D
        self.K = K
        self.dt = dt
        self.alpha_scale = alpha_scale # alpha_0 in paper
        self.device = device

        # Sensory frequency modulation.
        #
        # With freq_gain = 0 the input only enters through the phase-pinning
        # drive kappa*sin(gamma - theta), which is an attractor: measured
        # |dtheta| fell to 0.006 by step 31 and the network settled into global
        # synchrony (PLV 1.000, order parameter 0.977). Every oscillator then
        # carries the same phase, so no group is distinguishable.
        #
        # Two oscillators lock when |domega| < K_eff, so letting the image set
        # frequencies is what makes "similar features -> same group" possible.
        # freq_gain = 0 reproduces the original Eq. (5) behaviour exactly, and
        # creates no parameter, so existing checkpoints still load strictly.
        self.freq_gain = (
            nn.Parameter(torch.tensor(float(freq_gain))) if float(freq_gain) != 0.0 else None
        )

        # Pulse coupling: spikes act back on the phases.
        #
        # Without this the flow is one-way. Phases drive the spiking layers and
        # nothing returns, so with the usual loss weights the dendritic and
        # membrane layers receive exactly zero gradient and contribute nothing
        # to the result. Routing spikes back through the same graph closes the
        # loop, which both makes the spiking side causally part of the binding
        # and gives it a learning signal through the spike surrogate gradient.
        #
        # The cos(theta) factor is a phase response curve: an arriving spike
        # advances or delays depending on where the receiver is in its cycle,
        # which is what produces locking rather than a uniform shift. Measured
        # replay on a trained checkpoint improved ARI monotonically with gain up
        # to 1.0, while a densely firing neuron at high gain destabilised the
        # phases, so sparse pulses are the useful regime.
        self.spike_pulse_gain = (
            nn.Parameter(torch.tensor(float(spike_pulse_gain)))
            if float(spike_pulse_gain) != 0.0 else None
        )

        # Natural frequency ω_i (aligns with revised Eq. 5)
        self.omega = nn.Parameter(torch.randn(N, D) * 0.1)

        # Control stiffness κ_i
        self.kappa = nn.Parameter(torch.ones(N, D))
        self.direction_learner = nn.Parameter(torch.randn(N, N) * 0.01)
        # REMOVED: self.alpha = nn.Parameter(...) 
        # Reason: alpha must be derived from A, not learned freely.

    def forward(self, theta_prev, gamma, A=None, spike=None):
        """
        theta_prev : [B, H, D] Oscillator phases
        gamma      : [B, H] scalar drive broadcast over D, or [B, H, D] vector drive
        A          : [B, H, H] Connectivity matrix (Structural priors)
        spike      : [B, H] spikes from the previous step, for pulse coupling
        """
        B, H, D = theta_prev.shape
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(-1)
        elif gamma.dim() != 3 or gamma.size(-1) not in (1, D):
            raise ValueError(
                f"gamma must be [B, {H}] or [B, {H}, {D}], but got {tuple(gamma.shape)}."
            )
        error_bound.validate_kuramoto_layer_graph_vector_kuramoto_forward(H, gamma)
        device = theta_prev.device

        # 1. Handle Graph Structure & OT Surrogate
        if A is None:
            # Fallback if no graph provided
            A_lat = torch.ones(B, H, H, device=device)
            alpha = torch.zeros(B, H, H, 1, device=device)
        else:
            # Symmetrize connectivity
            A_lat = 0.5 * (A + A.transpose(1, 2))
            A_lat = torch.relu(A_lat) + 1e-6 # Avoid div by zero

            # --- OT-Derived Phase Lag (Section 3.3 in Paper) ---
            # Cost C_ij = 1 / A_ij
            cost_matrix = 1.0 / A_lat 
            
            # Normalize cost to [0, 1] per batch to stabilize
            c_min = cost_matrix.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            c_max = cost_matrix.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
            norm_cost = (cost_matrix - c_min) / (c_max - c_min + 1e-6)
            direction = self.direction_learner - self.direction_learner.transpose(0, 1) # [N, N]
            direction_mask = torch.tanh(direction)
            alpha_matrix = direction_mask.unsqueeze(0) * norm_cost # [B, N, N]
            # alpha_ij = alpha_0 * norm(C_ij)
            # Expand to [B, H, H, 1] to broadcast over D dim
            # alpha = (self.alpha_scale * norm_cost).unsqueeze(-1)
            alpha = (self.alpha_scale * alpha_matrix).unsqueeze(-1)

        # 2. Kuramoto Dynamics
        theta_i = theta_prev.unsqueeze(2) # [B, H, 1, D]
        theta_j = theta_prev.unsqueeze(1) # [B, 1, H, D]

        # Interaction term: sin(theta_j - theta_i - alpha_ij)
        phase_diff = theta_j - theta_i - alpha
        
        # Weighted sum by adjacency A_ij
        interaction = torch.sum(A_lat.unsqueeze(-1) * torch.sin(phase_diff), dim=2)
        coupling = (self.K / float(H)) * interaction

        # 3. Sensory Drive (Corrected to Sinusoidal)
        # kappa * sin(gamma - theta); gamma is already [B, H, 1] or [B, H, D]
        drive_term = self.kappa * torch.sin(gamma - theta_prev)

        # 4. Euler Integration
        omega_eff = self.omega
        if self.freq_gain is not None:
            omega_eff = omega_eff + self.freq_gain * gamma
        theta_dot = omega_eff + coupling + drive_term

        # 5. Pulse coupling: spikes arrive through the same graph.
        if spike is not None and self.spike_pulse_gain is not None:
            arriving = torch.bmm(A_lat, spike.unsqueeze(-1)).squeeze(-1)
            theta_dot = theta_dot + (
                self.spike_pulse_gain * arriving.unsqueeze(-1) * torch.cos(theta_prev)
            )
        theta_new = theta_prev + self.dt * theta_dot
        
        return theta_new
