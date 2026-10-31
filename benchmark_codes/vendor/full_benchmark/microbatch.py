"""Physical batching with unchanged independent per-image scientific objectives."""
import torch
from full_benchmark.loss import objective


def memory_budget_mib(size):
    return 11000 if size==1 else 4000+6000*size


def select_microbatch(free_mib,requested='auto',*,admission=False):
    sizes=(4,2,1) if requested=='auto' else (int(requested),)
    if any(size not in (1,2,4,8) for size in sizes):raise ValueError('invalid physical microbatch')
    for size in sizes:
        actual=min(size,2) if admission else size
        if free_mib>memory_budget_mib(actual):return actual
    return None


def sample_output(output,index):
    """Retain batch dimension; scalar trace metadata is not indexed."""
    return {key:sample_output(value,index) if isinstance(value,dict) else
            value[index:index+1] if torch.is_tensor(value) else value
            for key,value in output.items()}


def backward_microbatch(model,images,labels,teacher,image_ids,epoch,logical_batch_size):
    if len(images)!=len(image_ids) or not 0<len(images)<=logical_batch_size:
        raise ValueError('physical microbatch must belong to one logical batch')
    seeds=[255000+int(image_id)+100000*epoch for image_id in image_ids]
    output=model(images,inference_seeds=seeds,live_tail_steps=64)
    combined=None;loss_value=0.;parts_sum={}
    for index in range(len(images)):
        # Do not apply globally reduced criteria to multiple images: e.g. the
        # PLV balance penalty and variable-size Hungarian head must remain per-image.
        total,base,mask,parts=objective(model,sample_output(output,index),
            rgb=images[index:index+1],labels=labels[index:index+1] if labels is not None else None,
            teacher=teacher)
        scaled=total/logical_batch_size
        combined=scaled if combined is None else combined+scaled
        loss_value+=float(total.detach())/logical_batch_size
        for key,value in parts.items():
            parts_sum[key]=parts_sum.get(key,0.)+float(value.detach())/logical_batch_size
    combined.backward()
    return loss_value,parts_sum
