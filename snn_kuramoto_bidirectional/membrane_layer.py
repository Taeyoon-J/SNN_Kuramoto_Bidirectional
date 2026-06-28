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
    def __init__(self, output_dim, readout_dim, tau_minitializer='uniform', low_m=0, high_m=4,
                 vth=0.5, dt=4, device='cpu', bias=True):
        super(MembraneLayer, self).__init__()
        self.output_dim = output_dim
        self.readout_dim = readout_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.readout_dense = nn.Linear(output_dim,readout_dim)
        self.readout_tau_m = nn.Parameter(torch.Tensor(self.readout_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
            nn.init.uniform_(self.readout_tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)
            nn.init.constant_(self.readout_tau_m,low_m)

        self.mem = None
        self.spike = None
        self.readout_mem = None
        self.readout_spike = None

    def set_neuron_state(self, batch_size):
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)
        self.spike = torch.rand(batch_size, self.output_dim).to(self.device)
        self.v_th = torch.ones(batch_size, self.output_dim).to(self.device) * self.vth
        self.readout_mem = torch.rand(batch_size,self.readout_dim).to(self.device)
        self.readout_spike = torch.rand(batch_size,self.readout_dim).to(self.device)
        self.readout_v_th = torch.ones(batch_size,self.readout_dim).to(self.device)*self.vth

    def forward(self, inputs, mask):
        alpha = torch.sigmoid(self.tau_m)
        mask = mask.expand(self.mem.size(0), -1)
        pre_mem = self.mem
        self.mem = self.mem * alpha  + (1 - alpha) * R_m * inputs-self.v_th*self.spike
        self.mem = torch.where(mask == 0, pre_mem, self.mem)
        inputs_ = self.mem - self.v_th
        self.spike = act_fun_adp(inputs_) * mask

        k_input = self.spike.float()

        d_input = self.readout_dense(k_input)
        readout_alpha = torch.sigmoid(self.readout_tau_m)
        self.readout_mem = self.readout_mem * readout_alpha  + (1 - readout_alpha) * R_m * d_input-self.readout_v_th*self.readout_spike
        readout_inputs = self.readout_mem - self.readout_v_th
        self.readout_spike = act_fun_adp(readout_inputs)
        return self.readout_mem, self.spike
