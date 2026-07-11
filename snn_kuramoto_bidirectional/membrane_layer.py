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
                 vth=0.5, dt=4, device='cpu'):
        super(MembraneLayer, self).__init__()
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

        self.mem = None
        self.spike = None

    def set_neuron_state(self, batch_size):
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)
        self.spike = torch.rand(batch_size, self.output_dim).to(self.device)
        self.v_th = torch.ones(batch_size, self.output_dim).to(self.device) * self.vth

    def forward(self, h_wave_t, g_wave_t):
        alpha = torch.sigmoid(self.tau_m)
        g_wave_t = g_wave_t.expand(self.mem.size(0), -1)
        pre_mem = self.mem
        self.mem = self.mem * alpha  + (1 - alpha) * R_m * h_wave_t-self.v_th*self.spike
        self.mem = torch.where(g_wave_t == 0, pre_mem, self.mem)
        inputs_ = self.mem - self.v_th
        self.spike = act_fun_adp(inputs_) * g_wave_t
        return self.mem, self.spike
