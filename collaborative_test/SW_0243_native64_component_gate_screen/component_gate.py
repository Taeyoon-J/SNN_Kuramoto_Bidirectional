"""Preserve each oscillator component through the delayed raw sinusoidal gate."""
import torch


def sinusoidal_gating(theta_hist,t,phase_delay_steps,gate_mode='raw'):
    if gate_mode!='raw':raise ValueError('registered raw-gate source required')
    theta=theta_hist[t];delayed=theta_hist[max(0,t-phase_delay_steps)]
    if theta.ndim!=3 or theta.shape[-1]!=4 or delayed.shape!=theta.shape:
        raise ValueError('four-component native phase history required')
    mask=.5*(1.+torch.sin(delayed))
    return torch.sin(theta)*mask,mask
