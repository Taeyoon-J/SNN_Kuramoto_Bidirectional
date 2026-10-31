"""Real native64 CUDA equivalence/throughput check; does not train scientific jobs.

Run on a GPU with sufficient free memory. Numbers are diagnostics, not scores.
Every candidate sees identical images, image seeds, initial weights and loss scaling.
"""
import argparse,gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from portable_runtime import configure,connect_inputs,HERE


def compare(reference,actual):
    dot=aa=bb=err=0.;maximum=0.
    assert reference.keys()==actual.keys()
    for name in reference:
        a=reference[name].double().flatten();b=actual[name].double().flatten()
        assert bool(torch.isfinite(b).all()),name
        dot+=float(a@b);aa+=float(a@a);bb+=float(b@b)
        delta=a-b;err+=float(delta@delta)
        maximum=max(maximum,float(delta.abs().max()))
    return dict(relative_l2=(err/max(aa,1e-30))**.5,
                cosine=dot/max((aa*bb)**.5,1e-30),max_absolute=maximum)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'configs/frontier.json'))
    p.add_argument('--output',required=True);p.add_argument('--model',type=int,choices=(240,252,253,254),default=253)
    p.add_argument('--sizes',default='1,2,4');p.add_argument('--count',type=int,default=4)
    p.add_argument('--checkpoint',help='Optional current native64 weights; read-only diagnostic')
    p.add_argument('--initial-checkpoint',help='Own original teacher/statistics for that checkpoint')
    p.add_argument('--memory-limit-mib',type=int,help='Cap this diagnostic allocator; protect co-resident jobs')
    args=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    if args.memory_limit_mib:
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib*1024**2/total,0)
    config=configure(args.config);root=connect_inputs(config,args.output)
    from full_benchmark.model import FreshModel
    from full_benchmark.data import Dataset
    from full_benchmark.loss import InitialTeacher
    from full_benchmark.microbatch import backward_microbatch
    from full_benchmark.engine import cpu_state,state_digest
    dataset=Dataset('clevr',root);ids=dataset.train[:args.count]
    images=dataset.rgb(ids);labels=dataset.labels(ids) if args.model!=253 else None
    model=FreshModel(args.model,0).to('cuda:0')
    if args.checkpoint:
        if not args.initial_checkpoint:raise ValueError('current weights require own INITIAL teacher')
        initial_saved=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(initial_saved['model'])
    teacher=InitialTeacher(model) if args.model!=253 else None
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model']);del saved,initial_saved
    initial=cpu_state(model)
    sizes=[int(x) for x in args.sizes.split(',')];assert sizes[0]==1 and args.count>=max(sizes)
    proofs=[];reference=None;reference_state=None
    # Exclude one-time CUDA/library initialization from the timing comparison.
    backward_microbatch(model,torch.from_numpy(images[:1]).to('cuda:0'),
        labels[:1] if labels is not None else None,teacher,ids[:1],0,16)
    model.zero_grad(set_to_none=True);torch.cuda.synchronize()
    for size in sizes:
        model.load_state_dict(initial);model.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.Adam(model.parameter_groups())
        try:
            torch.cuda.synchronize();started=time.perf_counter();loss=0.;parts={}
            for start in range(0,len(ids),size):
                image=torch.from_numpy(images[start:start+size]).to('cuda:0')
                value,detail=backward_microbatch(model,image,
                    labels[start:start+size] if labels is not None else None,teacher,
                    ids[start:start+size],0,16)
                loss+=value
                for key,v in detail.items():parts[key]=parts.get(key,0.)+v
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            gradients={n:x.grad.detach().cpu().clone() for n,x in model.named_parameters() if x.requires_grad and x.grad is not None}
            assert len(gradients)==sum(x.requires_grad for x in model.parameters())
            # Match independent clipping of body and head, as in the real engine.
            groups=model.parameter_groups();body=[g for g in groups if g['name']!='mask_head']
            torch.nn.utils.clip_grad_norm_([v for g in body for v in g['params']],1.,error_if_nonfinite=True)
            if model.head is not None:torch.nn.utils.clip_grad_norm_(list(model.head.parameters()),1.,error_if_nonfinite=True)
            optimizer.step();updated=cpu_state(model)
            record=dict(size=size,seconds=seconds,images=len(ids),images_per_second=len(ids)/seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),loss=loss,parts=parts,
                active_gradient_tensors=len(gradients),initial_state_identical=True,
                no_training_masks_read=not dataset.labels_opened,steps=1024,gradient_tail_steps=64)
            if reference is None:reference=gradients;reference_state=updated;record['reference']=True
            else:
                record['gradient_comparison']=compare(reference,gradients)
                record['Adam_state_comparison']=compare(reference_state,updated)
                record['speedup_vs_size1']=proofs[0]['seconds']/seconds
            proofs.append(record)
        except torch.cuda.OutOfMemoryError:
            proofs.append(dict(size=size,status='does_not_fit_measured_GPU_memory'))
            # Do not try a larger candidate on a card that cannot fit this one.
            break
        finally:
            model.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
        (root/f'profile_model{args.model}.json').write_text(json.dumps(dict(status='diagnostic_in_progress',results=proofs),indent=2))
    result=dict(status='actual_native64_microbatch_profile_complete',model=args.model,
        image_ids=ids.tolist(),logical_batch=16,diagnostic_only_not_a_benchmark_score=True,
        current_theta_init=model.wrapped.core.theta_init,results=proofs,
        GPU=torch.cuda.get_device_name(),training_masks_used=args.model!=253)
    result['allocator_memory_limit_MiB']=args.memory_limit_mib
    (root/f'profile_model{args.model}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
