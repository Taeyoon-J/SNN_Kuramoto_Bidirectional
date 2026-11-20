import torch
import torch.nn as nn
import math
import torch.nn.functional as F

R_m = 1
gamma = 0.5
lens = 0.5

surrograte_type = 'MG'


def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        elif surrograte_type == 'rect':
            temp = input.abs() < 0.5
        return grad_input * temp.float()*gamma


act_fun_adp = ActFun_adp.apply


class MembraneLayer(nn.Module):
    def __init__(self, output_dim, tau_minitializer='uniform', low_m=0, high_m=4,
                 vth=0.5, dt=4, device='cpu', threshold_mode='absolute',
                 threshold_k=0.5):
        super(MembraneLayer, self).__init__()
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        # 'absolute' keeps vth as a fixed level. 'population' places it at
        # threshold_k standard deviations above the mean membrane of the whole
        # batch, as one scalar.
        #
        # Why: the steady state of this layer is mem* ~ R_m * h_wave, so the
        # membrane's level is set by the dendritic drive, and nothing constrains
        # that drive's scale. With vth a constant the threshold can end up
        # entirely outside the membrane's range. Across four seeds of one
        # configuration, measured on a fixed batch:
        #
        #   seed   mem mean   above vth=0.06   spike rate
        #   s1       +1.14        91.7%          0.300   healthy
        #   s0       +2.64       100.0%          0.468   threshold inert
        #   s3       +0.26        72.5%          0.284   broken readout
        #   s5       -0.65         0.0%          0.000   dead, mem max -0.046
        #
        # Seed 5 still has structure (membrane std 0.128, theta std 81.9); the
        # threshold simply sits where it cannot reach, so every mask is empty by
        # construction. Seed 0's membrane is entirely above threshold, so its
        # spikes stop carrying membrane information -- which is why it was the
        # weak seed throughout and over-predicted foreground.
        #
        # The shift is GLOBAL, one scalar for the batch, so differences between
        # units are preserved exactly. A per-unit threshold would equalise every
        # unit's rate and destroy precisely the between-unit structure synchrony
        # is read from -- a per-unit median was tried here and did that.
        self.threshold_mode = str(threshold_mode)
        self.threshold_k = float(threshold_k)
        if self.threshold_mode not in ('absolute', 'population'):
            raise ValueError("threshold_mode must be 'absolute' or 'population'")

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

        self.mem = None
        self.spike = None

    def set_neuron_state(self, batch_size):
        self.mem = torch.zeros(batch_size, self.output_dim).to(self.device)
        self.spike = torch.zeros(batch_size, self.output_dim).to(self.device)
        self.v_th = torch.ones(batch_size, self.output_dim).to(self.device) * self.vth

    def forward(self, h_wave_t, g_wave_t):
        alpha = torch.sigmoid(self.tau_m)
        g_wave_t = g_wave_t.expand(self.mem.size(0), -1)
        pre_mem = self.mem
        # One threshold per step, used for BOTH the reset and the comparison.
        #
        # Getting this wrong is not subtle: a first version thresholded on the
        # population level while still resetting by the constant self.v_th, so a
        # unit could fire from a membrane near +2.6 and be reset by 0.06. Trained
        # that way the result collapsed, 3-seed fg_ari 0.6887 -> 0.3029, with
        # groups per image falling to 3.40 / 1.05 / 2.34 against a true 6.20.
        thr = self._threshold(pre_mem)
        self.mem = self.mem * alpha + (1 - alpha) * R_m * h_wave_t - thr * self.spike
        self.mem = torch.where(g_wave_t == 0, pre_mem, self.mem)
        inputs_ = self.mem - self._threshold(self.mem)
        self.spike = act_fun_adp(inputs_) * g_wave_t
        return self.mem, self.spike

    def _threshold(self, mem):
        """Where the threshold sits for this membrane state.

        'absolute' is the fixed v_th. 'population' is threshold_k standard
        deviations above the batch's mean membrane, as one detached scalar, so
        the membrane's absolute scale stops deciding whether anything fires while
        the ordering between units is preserved exactly.
        """
        if self.threshold_mode == 'population':
            ref = mem.detach()
            return ref.mean() + self.threshold_k * ref.std().clamp_min(1e-6)
        return self.v_th
