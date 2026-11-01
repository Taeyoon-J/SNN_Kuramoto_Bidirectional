"""SW0256: pretrained fixed core, full35000 x10, physical1 and logical16."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from mod_benchmark import common as c
from readout_benchmark import runtime as r
from frozen_core_benchmark.data import Dataset
from frozen_core_benchmark.model import FrozenCoreModel as FreshModel
from full_benchmark.loss import InitialTeacher,objective
from full_benchmark.evaluation import evaluate
from full_benchmark.microbatch import backward_microbatch


def scientific_sources():
    paths=list(Path(__file__).parent.glob('*.py'))
    import full_benchmark
    paths+=list(Path(full_benchmark.__file__).parent.glob('*.py'))
    paths+=[Path(__file__).parents[1]/'portable_runtime.py']
    paths+=[Path(__file__).parents[1]/'run_frozen_core.py']
    for directory in ('snn_kuramoto_bidirectional','readout_benchmark'):
        paths+=list((c.CODE/directory).rglob('*.py'))
    import neutral_readout.head
    paths.append(Path(neutral_readout.head.__file__))
    return {str(p):c.sha(p) for p in sorted(set(paths))}


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train(dataset_name,variant,seed,root,*,sources,admission=False,microbatch=1):
    if microbatch!=1:raise ValueError('SW0256 keeps physical microbatch1')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA required')
    allocation_limit=os.environ.get('SW0256_ALLOCATOR_LIMIT_MIB')
    if allocation_limit:
        fraction=int(allocation_limit)*1024**2/torch.cuda.get_device_properties(0).total_memory
        if not 0<fraction<=1:raise ValueError('invalid CUDA allocator quota')
        torch.cuda.set_per_process_memory_fraction(fraction,0)
    c.immutable();root=Path(root);dataset=Dataset(dataset_name,root)
    namespace='engine_admission' if admission else 'jobs'
    out=root/namespace/f'model{variant}'/dataset_name/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    epochs=1 if admission else 10;batch=2 if admission else 16
    binding=dict(experiment='SW0256',dataset=dataset.binding,variant=variant,seed=seed,
        initialization='pretrained_SW0097_native16_mapped64_frozen_and_pretrained_encoder',unique_train=35000,epochs=epochs,effective_batch=batch,
        microbatch=microbatch,loss_reduction='mean of per-image objectives; divide by actual last batch length',
        total_rollout_steps=1024,gradient_tail_steps=64,all_active_groups_trained=True,
        pretrained_checkpoint_used=True,mask_training=variant!=253,
        final_mask_to_body_gradient=False,admission_only=admission,
        zero_bias_policy='trained source bias preserved; core/integration fixed',
        source_inventory_sha256=c.sha(sources),core_weight_updates=False,encoder_finetuned=True,
        code=scientific_sources(),benchmark=c.immutable())
    # JSON normalization makes comparison stable across restarts.
    binding=json.loads(json.dumps(binding));binding_path=out/'binding.json'
    effective=dict(code=binding['code'],CUDA_algorithm_policy='FP32/TF32off; physical1, per-image seeds and loss; original intrinsic roundoff retained',
        CPU_partition_policy='exact original energy arithmetic/merge bytecode, blocked independent rows')
    if binding_path.exists():
        previous=json.loads(binding_path.read_bytes())
        if previous!=binding:
            migration=json.loads((out/'runtime_migration.json').read_bytes())
            contract_path=Path(migration['contract_path']);contract=json.loads(contract_path.read_bytes())
            assert c.sha(contract_path)==migration['contract_sha256']
            assert contract['status']=='verified_same_training_with_exact_CPU_partition_acceleration'
            assert {k:v for k,v in previous.items() if k!='code'}=={k:v for k,v in binding.items() if k!='code'}
            assert contract['from_code']==previous['code'] and contract['to_code']==binding['code']
            assert migration['original_binding_sha256']==c.sha(binding_path)
            for proof in contract['proofs']:
                assert c.sha(proof['path'])==proof['sha256']
            effective['migration_sha256']=c.sha(out/'runtime_migration.json')
            binding=previous  # Preserve original dataset/budget binding and checkpoint bytes.
    else:c.write(binding_path,binding)
    binding_sha=c.sha(binding_path)
    effective['original_binding_sha256']=binding_sha
    if (out/'effective_code.json').exists():assert json.loads((out/'effective_code.json').read_bytes())==effective
    else:c.write(out/'effective_code.json',effective)
    effective_sha=c.sha(out/'effective_code.json')
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_bytes());assert complete['binding_sha256']==binding_sha
        return complete
    model=FreshModel(variant,seed,sources=sources).to('cuda:0')
    initial_path=out/'initial.pt'
    if initial_path.exists():
        initial=torch.load(initial_path,map_location='cpu',weights_only=True)
        assert initial['binding_sha256']==binding_sha
        model.load_state_dict(initial['model']);normalization=initial['normalization']
    else:
        normalization=dict(provenance=model.normalization_provenance)
        initial=dict(model=cpu_state(model),binding_sha256=binding_sha,normalization=normalization)
        r.save(initial_path,initial)
    model.normalization_provenance=normalization['provenance']
    teacher=InitialTeacher(model) if variant!=253 else None
    initial_values={name:initial['model'][name] for name,p in model.named_parameters() if p.requires_grad}
    initial_sha=c.sha(initial_path)
    groups=model.parameter_groups();body=[g for g in groups if g['name']!='mask_head']
    optimizer=torch.optim.Adam(body)
    head_optimizer=torch.optim.Adam([g for g in groups if g['name']=='mask_head']) if model.head is not None else None
    coverage={name:0. for name,p in model.named_parameters() if p.requires_grad}
    updates=0;seen=0;next_epoch=0;next_offset=0
    resume_path=out/'resume.pt'
    if resume_path.exists():
        resume=torch.load(resume_path,map_location='cuda:0',weights_only=True)
        assert resume['binding_sha256']==binding_sha and resume['initial_sha256']==initial_sha
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        if head_optimizer:head_optimizer.load_state_dict(resume['head_optimizer'])
        coverage=resume['gradient_coverage'];updates=resume['updates'];seen=resume['images_seen']
        next_epoch=resume['next_epoch'];next_offset=resume['next_offset']
        assert next_epoch*35000+next_offset==seen
        torch.set_rng_state(resume['torch_rng'].cpu());torch.cuda.set_rng_state(resume['cuda_rng'].cpu())
        model.assert_frozen()
    def snapshot(epoch,offset):
        model.assert_frozen()
        r.save(resume_path,dict(model=cpu_state(model),optimizer=optimizer.state_dict(),
            head_optimizer=head_optimizer.state_dict() if head_optimizer else None,
            binding_sha256=binding_sha,initial_sha256=initial_sha,updates=updates,images_seen=seen,
            next_epoch=epoch,next_offset=offset,gradient_coverage=coverage,
            diagnostic_only=admission,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),effective_code_sha256=effective_sha))
        c.write(out/'committed.json',dict(updates=updates,images_seen=seen,next_epoch=epoch,next_offset=offset,
            checkpoint_size=resume_path.stat().st_size,checkpoint_sha256=c.sha(resume_path),binding_sha256=binding_sha))
    started=time.time()
    for epoch in range(next_epoch,epochs):
        order=dataset.train[np.random.RandomState(249000+seed+10000*epoch).permutation(35000)]
        offset=next_offset if epoch==next_epoch else 0
        for start in range(offset,len(order),batch):
            if admission and updates==2:break
            selected=order[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            if head_optimizer:head_optimizer.zero_grad(set_to_none=True)
            images=dataset.rgb(selected)
            labels=dataset.labels(selected) if variant!=253 else None
            accumulated_loss=0.;parts_sum={}
            for index in range(0,len(selected),microbatch):
                chunk=selected[index:index+microbatch]
                c.write(out/'progress.json',dict(stage='training_actual_native_forward_backward',epoch=epoch+1,total_epochs=epochs,
                    optimizer_updates=updates,images_seen=seen,current_image=int(chunk[0]),microbatch=index+1,
                    physical_microbatch=len(chunk),configured_microbatch=microbatch,effective_batch=len(selected)))
                image=torch.from_numpy(images[index:index+len(chunk)]).to('cuda:0')
                value,parts=backward_microbatch(model,image,
                    labels[index:index+len(chunk)] if labels is not None else None,
                    teacher,chunk,epoch,len(selected))
                accumulated_loss+=value
                for key,value in parts.items():parts_sum[key]=parts_sum.get(key,0.)+value
            if variant==253:assert not dataset.labels_opened,'unsupervised training opened target masks'
            for name,p in model.named_parameters():
                if not p.requires_grad or p.grad is None:continue
                if not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('nonfinite gradient: '+name)
                coverage[name]+=float(p.grad.detach().abs().sum())
            norm=torch.nn.utils.clip_grad_norm_([p for g in body for p in g['params']],1.,error_if_nonfinite=True)
            if head_optimizer:torch.nn.utils.clip_grad_norm_(list(model.head.parameters()),1.,error_if_nonfinite=True)
            optimizer.step()
            if head_optimizer:head_optimizer.step()
            updates+=1;seen+=len(selected)
            end=start+len(selected);resume_epoch=epoch+int(end==len(order));resume_offset=0 if end==len(order) else end
            if updates==1 or updates%8==0 or end==len(order) or (admission and updates==2):snapshot(resume_epoch,resume_offset)
            with (out/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=updates,epoch=epoch+1,ids=selected.tolist(),loss=accumulated_loss,parts=parts_sum,
                    gradient_norm=float(norm),images_seen=seen,seconds=time.time()-started),allow_nan=False)+'\n')
        if admission:break
    assert seen==(4 if admission else 350000) and updates==(2 if admission else 21880),(seen,updates)
    inactive=[name for name,value in coverage.items() if value==0]
    assert not inactive,('inactive active parameter tensors',inactive)
    changes={name:float((p.detach().cpu()-initial_values[name]).abs().max()) for name,p in model.named_parameters() if p.requires_grad}
    for group in groups:
        identities={id(p) for p in group['params']}
        assert any(changes[name]>0 for name,p in model.named_parameters() if id(p) in identities),group['name']
    model.assert_frozen()
    # Test actual checkpoint and optimizer restoration before evaluating.
    digest=state_digest(model);restored=torch.load(resume_path,map_location='cuda:0',weights_only=True)
    model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
    if head_optimizer:head_optimizer.load_state_dict(restored['head_optimizer'])
    assert state_digest(model)==digest and {int(s['step'].item()) for s in optimizer.state.values()}=={updates}
    if head_optimizer:assert {int(s['step'].item()) for s in head_optimizer.state.values()}=={updates}
    c.write(out/'training_report.json',dict(binding_sha256=binding_sha,initial_sha256=initial_sha,updates=updates,images_seen=seen,
        physical_microbatch=microbatch,maximum_actual_physical_batch=min(microbatch,batch),effective_batch=batch,
        allocator_limit_MiB=int(allocation_limit) if allocation_limit else None,
        unique_TRAIN=35000,epochs_completed=0 if admission else 10,diagnostic_only=admission,
        active_tensor_gradient_coverage=coverage,parameter_changes=changes,optimizer_resume_verified=True,
        model_sha256=digest,mask_opened_during_training=dataset.labels_opened,seconds=time.time()-started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),no_parameter_checkpoint_reuse=False,frozen_core_initial_and_final_sha256=model.frozen_initial_sha256,
        frozen_core_verified=True,source_provenance=model.source_provenance,effective_code_sha256=effective_sha))
    # Diagnostic evaluation is ONE reserved image, reported explicitly as such.
    result=evaluate(model,dataset,out/'evaluation',count=1 if admission else None)
    complete=dict(status='engine_diagnostic_passed' if admission else 'fixed_core35000_10passes_and_evaluation_complete',
        binding_sha256=binding_sha,training_report_sha256=c.sha(out/'training_report.json'),
        evaluation_sha256=c.sha(out/'evaluation/result.json'),evaluation_count=result['count'],
        variant=variant,seed=seed,dataset=dataset_name,updates=updates,images_seen=seen,diagnostic_only=admission)
    c.write(out/'complete.json',complete);return complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--variant',type=int,choices=(252,253),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--root',required=True);p.add_argument('--admission',action='store_true')
    p.add_argument('--sources',required=True)
    a=p.parse_args();print(json.dumps(train(a.dataset,a.variant,a.seed,a.root,admission=a.admission,sources=a.sources)))
