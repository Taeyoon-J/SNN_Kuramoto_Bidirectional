"""Preserve supervised-vs-unsupervised and detached-vs-live final-mask routes."""
import copy
import math
import numpy as np
import torch
from neutral_readout.head import objective as mask_objective
from readout_benchmark.head import activity_features
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss,phase_locking_value
from snn_kuramoto_bidirectional.sw240.loss import effective_frequency,make_teacher,teacher_objective
from full_benchmark.fast_partition import foreground_induced_merge


def core_objective(model,output):
    # The registered94/95 positive-graph core objective, not target masks.
    criterion=UnsupervisedS2NetLoss(spike_rate_weight=0.,spike_smooth_weight=0.,spike_diversity_weight=0.,
        structural_weight=0.,plv_bimodality_weight=6.,plv_balance_weight=10.,plv_coherence_weight=.5,
        plv_collapse_weight=1.,plv_target_density=.867,patch_grid_size=(model.grid,model.grid))
    phase=phase_locking_value(output['trace']['theta'],settle=512,combine='mean')
    primary,_=criterion(plv=phase)
    spike,_=criterion(plv=output['q'])
    return primary+5.*spike,dict(phase_primary=primary,positive_spike=spike)


class InitialTeacher:
    """Frozen shadow of THIS random initialization; no learned checkpoint import."""
    def __init__(self,model):
        if model.variant==253:raise ValueError('unsupervised variant has no mask teacher')
        self.encoder=copy.deepcopy(model.encoder).eval().requires_grad_(False)
        self.projection=copy.deepcopy(model.wrapped.core.gamma_channel_proj).requires_grad_(False)
        self.omega=model.wrapped.core.kuramoto.omega.detach().clone()
        self.gain=model.wrapped.core.kuramoto.freq_gain.detach().clone()
        self.phase_gain=copy.deepcopy(model.wrapped.core.gamma_phase_gain)
        if torch.is_tensor(self.phase_gain):self.phase_gain=self.phase_gain.detach().clone()
        self.mean=model.feature_mean.detach().clone();self.std=model.feature_std.detach().clone()

    @torch.no_grad()
    def targets(self,model,rgb,labels):
        feature=self.encoder(rgb.permute(0,3,1,2).float()/255.)
        gamma=model.patcher(((feature-self.mean)/self.std).clamp(-3.,3.))
        drive=model.wrapped.core.gamma_to_drive(gamma,self.projection,self.phase_gain)
        frequency=(self.omega.unsqueeze(0)+self.gain*drive)/(self.gain.abs()*math.pi)
        return make_teacher(gamma.cpu().numpy(),frequency.cpu().numpy(),np.asarray(labels))


def regions_from_output(model,output):
    with torch.no_grad():
        gain=float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
        return foreground_induced_merge(output['q'][0].detach().cpu().numpy().astype(np.float64),
            output['adjacency'][0].detach().cpu().numpy().astype(np.float64),gain,
            np.ones((model.grid,model.grid),dtype=np.int64),grid_shape=(model.grid,model.grid))['labels']


def objective(model,output,*,rgb,labels=None,teacher=None):
    if model.variant==253 and (labels is not None or teacher is not None):
        raise ValueError('253 training must not receive target masks or a mask teacher')
    base,parts=core_objective(model,output)
    if model.variant!=253:
        if labels is None or teacher is None:raise ValueError('supervised variants need TRAIN masks and own initialization teacher')
        supervised,detail=teacher_objective(output['gamma'],effective_frequency(model.wrapped.core,output['gamma']),
                                          teacher.targets(model,rgb,labels))
        base=base+supervised;parts.update(detail)
    head_loss=None
    if model.head is not None:
        features=activity_features(output['trace'])[0]
        if model.variant==252:features=features.detach()
        regions=regions_from_output(model,output)
        head_loss=mask_objective(model.head(features,regions),labels[0])
        parts['final_mask']=head_loss
    total=base if head_loss is None else base+head_loss
    if not bool(torch.isfinite(total)):raise FloatingPointError('nonfinite full-model loss')
    return total,base,head_loss,parts
