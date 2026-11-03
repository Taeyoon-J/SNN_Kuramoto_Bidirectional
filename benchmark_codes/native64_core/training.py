"""Two disjoint backward paths: phase/spike -> encoder/core; RGB -> detached head."""
import torch
from torch.nn import functional as F


class _DeterministicAverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx,maps,grid):
        # Keep the original CUDA forward exactly; replace only its atomic backward.
        matrices=[]
        for inputs,outputs in zip(maps.shape[-2:],grid):
            positions=torch.arange(outputs,device=maps.device)
            start=torch.div(positions*inputs,outputs,rounding_mode='floor')
            stop=torch.div((positions+1)*inputs+outputs-1,outputs,rounding_mode='floor')
            index=torch.arange(inputs,device=maps.device)[None]
            weights=((index>=start[:,None])&(index<stop[:,None])).to(maps.dtype)/(stop-start)[:,None]
            matrices.append(weights)
        ctx.save_for_backward(*matrices)
        return F.adaptive_avg_pool2d(maps,grid)

    @staticmethod
    def backward(ctx,gradient):
        height,width=ctx.saved_tensors
        return height.T@gradient@width,None


class DeterministicPatchGamma(torch.nn.Module):
    grid_size=(64,64)
    def forward(self,maps):
        return _DeterministicAverage.apply(maps,self.grid_size).flatten(2)


def configure(found, head, *, freeze_core=False, freeze_encoder=False):
    found.patcher=DeterministicPatchGamma()
    found.wrapped.requires_grad_(not freeze_core).train(not freeze_core)
    found.encoder.requires_grad_(not freeze_encoder).train(not freeze_encoder)
    # T uses a fixed contrast8; this legacy Parameter is not in the actual forward.
    found.wrapped.core.graph_generator.geodesic_contrast.requires_grad_(False)
    head.requires_grad_(True).train()
    head.content_encoder.requires_grad_(False)
    head.head.prior_scale.requires_grad_(False)
    head.head.residual_scale.requires_grad_(False)


def make_optimizers(found, head, encoder_lr=3e-6, core_lr=3e-5, head_lr=.001):
    groups=[]
    for name,module,lr in [('encoder',found.encoder,encoder_lr),('core',found.wrapped,core_lr)]:
        params=[p for p in module.parameters() if p.requires_grad]
        if params:groups.append(dict(name=name,params=params,lr=lr))
    body=torch.optim.Adam(groups) if groups else None
    readout=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=head_lr)
    return body,readout


def body_objective(trace,q,settle,grid=64):
    from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss,phase_locking_value
    criterion=UnsupervisedS2NetLoss(spike_rate_weight=0.,spike_smooth_weight=0.,spike_diversity_weight=0.,
        structural_weight=0.,plv_bimodality_weight=6.,plv_balance_weight=10.,plv_coherence_weight=.5,
        plv_collapse_weight=1.,plv_target_density=.867,patch_grid_size=(grid,grid))
    phase=phase_locking_value(trace['theta'],settle=settle,combine='mean')
    phase_loss,_=criterion(plv=phase);spike_loss,_=criterion(plv=q)
    return phase_loss+5.*spike_loss,dict(phase=float(phase_loss.detach()),spike=float(spike_loss.detach()))


def detached_row(found,gamma,trace,adjacency,q,rgb,*,settle,package):
    """The only entry from body to head; detach gamma AND all activity branches."""
    import numpy as np
    from portable import module
    activity=module('train261_activity',package.parent/'vendor/readout_benchmark/head.py')
    neutral=module('train261_candidates',package.parent/'vendor/neutral_readout/head.py')
    solver=module('train261_partition',package/'partition.py')
    with torch.no_grad():
        features=activity.activity_features({k:v.detach() if torch.is_tensor(v) else v for k,v in trace.items()},settle=settle)[0]
        gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
        regions=solver.foreground_induced_merge(q[0].detach().cpu().numpy().astype(np.float64),
            adjacency[0].detach().cpu().numpy().astype(np.float64),gain,np.ones((64,64),dtype=np.int64),grid_shape=(64,64))['labels']
        masks,prior=neutral.candidates(features,regions)
        target=F.avg_pool2d(rgb.permute(0,3,1,2).float()/255.,2)[0].permute(1,2,0).reshape(4096,3)
    row=dict(features=features.detach(),gamma=gamma[0].T.detach(),masks=masks.detach(),prior=prior.detach(),rgb=target.detach())
    assert all(not v.requires_grad and v.grad_fn is None for v in row.values())
    return row


def norm(module):
    gradients=[p.grad.detach() for p in module.parameters() if p.grad is not None]
    if not gradients:return 0.
    if not all(bool(torch.isfinite(g).all()) for g in gradients):raise FloatingPointError('nonfinite gradient')
    return sum(float(g.double().square().sum()) for g in gradients)**.5


def reconstruction_loss(head,row,group_chunk=16):
    """Same context_rgb MSE, checkpoint decoder by groups to bound activation memory."""
    from torch.utils.checkpoint import checkpoint
    if head.objective_config['no_xy'] or head.objective_config['edge_weight']:
        raise ValueError('this trainer preserves the published context_rgb objective only')
    logits,features,_=head.routing(row);probability=(logits/.5).softmax(0)
    with torch.no_grad():content=head.content_encoder(features)
    latent=(probability@content)/probability.sum(1,keepdim=True).clamp_min(1e-6)
    def decode(values,xy):
        batch=values.shape[0]
        return head.decoder(torch.cat([values[:,None].expand(-1,4096,-1),xy[None].expand(batch,-1,-1)],-1))
    reconstructed=torch.zeros_like(row['rgb'])
    for start in range(0,len(latent),group_chunk):
        decoded=checkpoint(decode,latent[start:start+group_chunk],head.xy,use_reentrant=False)
        reconstructed=reconstructed+(probability[start:start+group_chunk,:,None]*decoded).sum(0)
    loss=F.mse_loss(reconstructed,row['rgb'])
    return loss,dict(mse=loss,weighted_mse=loss)


def step(found,head,rgb,body_optimizer,head_optimizer,*,steps=1024,settle=512,tail=64,package,clip=1.,group_chunk=16):
    from collaborative_test.SW_0162_connected_signed_partition import preflight as source
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    if body_optimizer:body_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    gamma=found.encode(rgb)
    trace,a=source.capture_source_coupling(found.wrapped.core.kuramoto,
        lambda:found.rollout(gamma,total_steps=steps,live_tail_steps=tail if body_optimizer else 0))
    q=spike_synchrony_affinity(trace['spikes'],trace['component_spikes'],settle=settle,affinity_mode='spike')
    if q.shape!=(1,4096,4096):raise ValueError('actual native64 affinity required')
    # Build from this SAME pre-update forward. No core cache can be reused across updates.
    row=detached_row(found,gamma,trace,a,q,rgb,settle=settle,package=package)
    if body_optimizer:
        loss,parts=body_objective(trace,q,settle)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite body loss')
        loss.backward()
        if any(p.grad is not None for p in head.parameters()):raise AssertionError('body loss reached head')
        gradients=dict(encoder=norm(found.encoder),core=norm(found.wrapped))
        if any(p.requires_grad for p in found.encoder.parameters()) and gradients['encoder']==0:raise AssertionError('encoder has no body gradient')
        if any(p.requires_grad for p in found.wrapped.parameters()) and gradients['core']==0:raise AssertionError('core has no body gradient')
        torch.nn.utils.clip_grad_norm_([p for g in body_optimizer.param_groups for p in g['params']],clip,error_if_nonfinite=True)
        body_optimizer.step();body_optimizer.zero_grad(set_to_none=True)
        body_value=float(loss.detach());del loss
    else:parts={};gradients=dict(encoder=0.,core=0.);body_value=0.
    del trace,a,q,gamma
    head_loss,head_parts=reconstruction_loss(head,row,group_chunk)
    if not bool(torch.isfinite(head_loss)):raise FloatingPointError('nonfinite head loss')
    head_loss.backward()
    # Actual body grads were cleared above, so this verifies isolation without retain_graph.
    if any(p.grad is not None for module in [found.encoder,found.wrapped] for p in module.parameters()):
        raise AssertionError('head loss leaked into encoder/core')
    gradients['head']=norm(head)
    if gradients['head']==0:raise AssertionError('head has no RGB gradient')
    torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],clip,error_if_nonfinite=True)
    head_optimizer.step()
    return dict(body_loss=body_value,head_loss=float(head_loss.detach()),body_parts=parts,
                head_parts={k:float(v.detach()) for k,v in head_parts.items()},gradients=gradients,
                head_detached_from_body=True,body_detached_from_head=True)
