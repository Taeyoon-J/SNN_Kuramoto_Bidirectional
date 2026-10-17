"""SW0191 actual spike rollout/update using the registered composite loss."""
import math
import numpy as np
import torch
from collaborative_test.SW_0179_qce_seed_replication import train as base
from collaborative_test.SW_0191_partition_error_feedback import objective, feedback_context
BATCH, MICRO, CLIP_NORM = base.BATCH, base.MICRO, base.CLIP_NORM
flatten_modal_targets=base.flatten_modal_targets
_vectorized_class_loss=base._vectorized_class_loss
_tensor_sha=base._tensor_sha
_backward_microbatch_losses=base._backward_microbatch_losses

def _loss_for_image(arm,q,rho,labels,predicted):
    if arm!='product_q_ce':raise ValueError('unregistered SW0191 recipe')
    return objective.loss_for_image(q,rho,labels,predicted)


def _train_update(arm, found, optimizer, image_indices, inputs, device, update_number):
    from collaborative_test.SW_0175_critical_component_binding import train_v2 as util
    feedback_context.require_batch(found,image_indices)
    arrays = inputs["arrays"]
    optimizer.zero_grad(set_to_none=True)
    before = {group["name"]: [p.detach().clone() for p in group["params"]]
              for group in optimizer.param_groups}
    losses, q_losses, feedback_losses, emitted_grad_sq, feedback_grad_sq = [], [], [], 0.0, 0.0
    micro_records = []
    aux_family_sq={group["name"]:0. for group in optimizer.param_groups}
    all_params=[p for group in optimizer.param_groups for p in group["params"]]
    for offset in range(0, BATCH, MICRO):
        indices = image_indices[offset:offset+MICRO]
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][indices])).to(device)
        _gamma, trace, adjacency, z, q, rho = util._native_forward(
            found, rgb, live_tail_steps=64)
        labels_batch = flatten_modal_targets(arrays["modal_gt"][indices]).to(device)
        micro_losses = [];micro_feedback=[]
        for local in range(MICRO):
            labels = labels_batch[local]
            predicted=feedback_context.get_prediction(found,int(indices[local]),device)
            feedback=objective.partition_error_loss(q[local],rho[local],labels,predicted)
            qloss = _vectorized_class_loss(q[local], labels, temperature=0.1)
            loss = qloss+feedback
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(qloss)):
                raise FloatingPointError(f"nonfinite SW0191 loss at update {update_number}")
            if update_number <= 2:
                grad = torch.autograd.grad(loss, trace["component_spikes"], retain_graph=True,
                                           allow_unused=True)[0]
                if grad is not None:
                    if not bool(torch.isfinite(grad).all()):
                        raise FloatingPointError("nonfinite SW0191 loss-to-emitted-S gradient")
                    emitted_grad_sq += float(grad.detach().double().square().sum().cpu())
                auxgrad=torch.autograd.grad(feedback,trace["component_spikes"],retain_graph=True,allow_unused=True)[0]
                if auxgrad is not None:
                    if not bool(torch.isfinite(auxgrad).all()):raise FloatingPointError('nonfinite feedback-to-S gradient')
                    feedback_grad_sq += float(auxgrad.detach().double().square().sum().cpu())
            micro_feedback.append(feedback)
            feedback_losses.append(float(feedback.detach().cpu()))
            micro_losses.append(loss)
            losses.append(float(loss.detach().cpu()))
            q_losses.append(float(qloss.detach().cpu()))
        if update_number<=2:
            auxiliary=torch.stack(micro_feedback).sum()/BATCH
            param_grads=torch.autograd.grad(auxiliary,all_params,retain_graph=True,allow_unused=True)
            position=0
            for group in optimizer.param_groups:
                for _parameter in group['params']:
                    value=param_grads[position];position+=1
                    if value is not None:
                        if not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite feedback-only parameter gradient')
                        aux_family_sq[group['name']]+=float(value.detach().double().square().sum().cpu())
        _backward_microbatch_losses(micro_losses)
        micro_records.append({"image_indices": list(map(int, indices)),
                              "q_sha256": _tensor_sha(q), "z_sha256": _tensor_sha(z)})
        del rgb, trace, adjacency, z, q, rho, labels_batch

    family_norms, raw_param_norms = {}, {}
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    for group in optimizer.param_groups:
        sq = 0.0
        for name, parameter in zip(group["parameter_names"], group["params"]):
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"nonfinite SW0191 gradient {group['name']}/{name}")
                norm = float(parameter.grad.detach().double().norm().cpu())
                raw_param_norms[id(parameter)] = norm
                sq += norm * norm
        family_norms[group["name"]] = math.sqrt(sq)
    total_norm = math.sqrt(sum(value * value for value in family_norms.values()))
    clipped = torch.nn.utils.clip_grad_norm_(all_params, CLIP_NORM)
    if not bool(torch.isfinite(clipped)):
        raise FloatingPointError("nonfinite SW0191 global gradient norm")
    optimizer.step()
    found.wrapped.project_integrations_()
    deltas, named = {}, {}
    for group in optimizer.param_groups:
        sq, rows = 0.0, []
        for name, old, parameter in zip(group["parameter_names"], before[group["name"]], group["params"]):
            delta = float((parameter.detach().double() - old.detach().double()).norm().cpu())
            if not math.isfinite(delta):
                raise FloatingPointError("nonfinite SW0191 optimizer delta")
            sq += delta * delta
            rows.append({"name": name, "gradient_l2": raw_param_norms.get(id(parameter), 0.0),
                         "gradient_present": parameter.grad is not None, "parameter_delta_l2": delta})
        deltas[group["name"]], named[group["name"]] = math.sqrt(sq), rows
    emitted_grad = math.sqrt(emitted_grad_sq) if update_number <= 2 else None
    return {"update": int(update_number), "image_indices": list(map(int, image_indices)),
        "image_ids": [int(inputs["ordered_train_ids"][i]) for i in image_indices],
        "images_seen": BATCH, "microbatches": micro_records,
        "mean_registered_loss": float(np.mean(losses)), "mean_q_loss": float(np.mean(q_losses)),
        "mean_feedback_loss":float(np.mean(feedback_losses)),
        "feedback_loss_to_emitted_S_gradient_l2":math.sqrt(feedback_grad_sq) if update_number<=2 else None,
        "feedback_trainable_gradient_rss":math.sqrt(sum(aux_family_sq.values())) if update_number<=2 else None,
        "feedback_family_gradient_rss":{name:math.sqrt(value) for name,value in aux_family_sq.items()} if update_number<=2 else None,
        "feedback_context_sha256":feedback_context.context_sha(found),
        "gradient_norm_before_clip": total_norm,
        "clip_scale": min(1.0, CLIP_NORM / (float(clipped.detach().cpu()) + 1e-12)),
        "family_gradient_norms": family_norms, "family_parameter_delta_l2": deltas,
        "named_parameter_credit": named, "loss_to_emitted_component_spikes_gradient_l2": emitted_grad}

