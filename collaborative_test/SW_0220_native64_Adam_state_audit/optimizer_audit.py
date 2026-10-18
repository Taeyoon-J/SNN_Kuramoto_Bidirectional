"""Paired temporary Adam proposals, always restoring the supplied transaction."""
import math
import numpy as np  # Initialize the workspace NumPy runtime before Torch on Windows.
import torch
from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx

ARMS=('inherited_Adam','reset_Adam')


def compare_steps(found,optimizer,families,gradients,snapshot,measure):
    modules=dict(wrapped=found.wrapped,encoder=found.encoder)
    parameters=[(family,p) for family,entries in families.items() for _,p in entries]
    if len(parameters)!=len({id(p) for _,p in parameters}) or set(gradients)!={id(p) for _,p in parameters}:
        raise ValueError('one immutable clipped gradient per unique family parameter required')
    expected={id(p) for g in optimizer.param_groups for p in g['params']}
    if expected!={id(p) for _,p in parameters}:raise ValueError('optimizer/family parameter mismatch')
    configuration=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]
    results={}
    try:
        for arm in ARMS:
            tx.restore_transaction(snapshot,modules,optimizer,restore_rng=True,clear_grad=True)
            if arm=='reset_Adam':optimizer.state.clear()
            steps_before=sorted({int(s['step'].item()) for s in optimizer.state.values()})
            initial={id(p):p.detach().clone() for _,p in parameters}
            for _,p in parameters:
                g=gradients[id(p)];p.grad=None if g is None else g.to(device=p.device,dtype=p.dtype).clone()
                if g is not None and not torch.equal(p.grad.detach().cpu(),g):raise ValueError('shared gradient copy changed')
            optimizer.step();found.wrapped.project_integrations_()
            deltas={family:0. for family in families};dots={family:0. for family in families};finite=True
            for family,p in parameters:
                delta=p.detach().double()-initial[id(p)].double()
                finite=finite and bool(torch.isfinite(p).all())
                deltas[family]+=float(delta.square().sum().cpu())
                g=gradients[id(p)]
                if g is not None:dots[family]+=float((g.to(device=p.device,dtype=torch.float64)*delta).sum().cpu())
            after=None
            if finite:
                tx.restore_rng_state(snapshot['rng']);after=float(measure())
                if not math.isfinite(after):after=None
            unchanged=[{k:v for k,v in group.items() if k!='params'} for group in optimizer.param_groups]==configuration
            if not unchanged:raise ValueError('optimizer-state contrast changed parameter-group settings')
            results[arm]=dict(after_loss=after,parameters_finite=finite,parameter_group_hyperparameters_unchanged=unchanged,
                shared_clipped_gradient_exactly_copied=True,
                steps_before=steps_before,steps_after=sorted({int(s['step'].item()) for s in optimizer.state.values()}),
                family_parameter_delta_l2={k:math.sqrt(v) if math.isfinite(v) else None for k,v in deltas.items()},
                family_surrogate_gradient_dot_projected_delta={k:v if math.isfinite(v) else None for k,v in dots.items()},
                surrogate_gradient_dot_projected_delta=sum(dots.values()) if all(math.isfinite(v) for v in dots.values()) else None)
    finally:
        tx.restore_transaction(snapshot,modules,optimizer,restore_rng=True,clear_grad=True)
    return results


def decision(before,arms):
    tolerance=max(1e-7,1e-6*abs(before));old=arms['inherited_Adam']['after_loss'];fresh=arms['reset_Adam']['after_loss']
    return dict(tolerance=tolerance,
        inherited_descends=bool(old is not None and old<before-tolerance),
        reset_descends=bool(fresh is not None and fresh<before-tolerance),
        reset_beats_inherited=bool(fresh is not None and old is not None and fresh<old-tolerance),
        TRAIN_only_followup_eligible=bool(fresh is not None and old is not None and fresh<min(before,old)-tolerance),
        segmentation_improvement_verified=False)
