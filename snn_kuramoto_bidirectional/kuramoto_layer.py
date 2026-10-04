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
                 spike_pulse_gain=0.0, center_pulse=True):
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
        self.center_pulse = bool(center_pulse)
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

    def prepare_coupling(self, A, batch_size=None, num_units=None, device=None):
        """
        Symmetrised graph and its two lag kernels, computed once per rollout.

        Returns (A_lat, P, Q) where P = A_lat * cos(alpha) and Q = A_lat *
        sin(alpha). The graph is fixed for the whole rollout unless graph
        feedback is on, so building these once keeps a single copy in the
        autograd graph instead of one per time step.
        """
        if A is None:
            if batch_size is None or num_units is None:
                raise ValueError("batch_size and num_units are required when A is None.")
            A_lat = torch.ones(batch_size, num_units, num_units, device=device)
            return A_lat, A_lat, torch.zeros_like(A_lat)

        A_lat = 0.5 * (A + A.transpose(1, 2))
        A_lat = torch.relu(A_lat) + 1e-6            # Avoid div by zero

        # --- OT-Derived Phase Lag (Section 3.3 in Paper) ---
        cost_matrix = 1.0 / A_lat
        c_min = cost_matrix.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        c_max = cost_matrix.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        norm_cost = (cost_matrix - c_min) / (c_max - c_min + 1e-6)
        direction = self.direction_learner - self.direction_learner.transpose(0, 1)
        alpha_matrix = torch.tanh(direction).unsqueeze(0) * norm_cost
        alpha = self.alpha_scale * alpha_matrix     # [B, H, H]
        return A_lat, A_lat * torch.cos(alpha), A_lat * torch.sin(alpha)

    def forward(self, theta_prev, gamma, A=None, spike=None, coupling=None):
        """
        theta_prev : [B, H, D] Oscillator phases
        gamma      : [B, H] scalar drive broadcast over D, or [B, H, D] vector drive
        A          : [B, H, H] Connectivity matrix (Structural priors)
        spike      : [B, H] spikes from the previous step, for pulse coupling
        coupling   : optional (A_lat, P, Q) from prepare_coupling, to avoid
                     rebuilding the lag kernels at every step
        """
        B, H, D = theta_prev.shape
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(-1)
        elif gamma.dim() != 3 or gamma.size(-1) not in (1, D):
            raise ValueError(
                f"gamma must be [B, {H}] or [B, {H}, {D}], but got {tuple(gamma.shape)}."
            )
        if gamma.size(1) != H:
            raise ValueError(
                f"gamma has {gamma.size(1)} oscillators, but theta has {H}."
            )
        device = theta_prev.device

        # 1. Graph structure and OT surrogate
        if coupling is None:
            coupling = self.prepare_coupling(A, batch_size=B, num_units=H, device=device)
        A_lat, lag_cos, lag_sin = coupling

        # 2. Kuramoto dynamics.
        #
        # sum_j A_ij sin(theta_j - theta_i - alpha_ij) expanded through the angle
        # difference identities, so the pairwise [B, H, H, D] phase difference is
        # never materialised. That tensor was the reason a 32x32 grid could not
        # run: at H=1024 it is 268 MB per step, and the rollout keeps every step
        # for the backward pass. Four matmuls of [B, H, H] against [B, H, D]
        # compute the same quantity in O(H^2) memory.
        sin_theta, cos_theta = torch.sin(theta_prev), torch.cos(theta_prev)
        interaction = (
            cos_theta * torch.bmm(lag_cos, sin_theta)
            - sin_theta * torch.bmm(lag_cos, cos_theta)
            - cos_theta * torch.bmm(lag_sin, cos_theta)
            - sin_theta * torch.bmm(lag_sin, sin_theta)
        )
        coupling_term = (self.K / float(H)) * interaction

        # 3. Sensory Drive (Corrected to Sinusoidal)
        # kappa * sin(gamma - theta); gamma is already [B, H, 1] or [B, H, D]
        drive_term = self.kappa * torch.sin(gamma - theta_prev)

        # 4. Euler Integration
        omega_eff = self.omega
        if self.freq_gain is not None:
            omega_eff = omega_eff + self.freq_gain * gamma
        theta_dot = omega_eff + coupling_term + drive_term

        # 5. Pulse coupling: spikes arrive through the same graph, closing the
        #    loop from the spiking layers back onto the phases. This is the
        #    "bidirectional" the architecture is named for, and it is off unless
        #    spike_pulse_gain is set.
        #
        #    The arriving pulse is centred across oscillators first. Without that
        #    it drove the system straight to global synchrony: when firing is
        #    dense, A @ spike is nearly uniform across units, so the term becomes
        #    a constant times cos(theta) -- a force that pulls every phase the
        #    same way, which is the opposite of a signal about which units belong
        #    together. Measured without centring, firing went from 0.0036 to 0.74
        #    and ARI collapsed to 0.006. Centring removes exactly the uniform
        #    component and leaves the part that differs between oscillators.
        if spike is not None and self.spike_pulse_gain is not None:
            arriving = torch.bmm(A_lat, spike.unsqueeze(-1)).squeeze(-1)
            if self.center_pulse:
                arriving = arriving - arriving.mean(dim=1, keepdim=True)
            theta_dot = theta_dot + (
                self.spike_pulse_gain * arriving.unsqueeze(-1) * torch.cos(theta_prev)
            )
        theta_new = theta_prev + self.dt * theta_dot
        
        return theta_new
