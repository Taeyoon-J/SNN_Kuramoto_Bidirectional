"""Fixed35000 adaptation; original architectures, objectives and spike readout."""
import argparse,hashlib,json,os,shutil,time
_VISIBLE=os.environ.get('CUDA_VISIBLE_DEVICES')
from pathlib import Path
import numpy as np
import torch
from . import scope,factory
from collaborative_test.SW_0240_native64_70000_cached_io import train as old,data
from collaborative_test.SW_0244_native64_within_precision.loss import groups_from_labels,groups_to_device,objective
if _VISIBLE is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_VISIBLE

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=factory.ARCHIVE
sha=old.sha;write_json=old.write_json;archive=factory.archive

def fingerprint():
    names=('DESIGN_KO.md','scope.py','factory.py','run.py','evaluate.py','queue.py','conftest.py','test_scope.py','test_contract.py')
    return dict(specification=scope.specification(),files={n:sha((HERE/n).read_bytes()) for n in names},
        original_teacher_code=sha(Path(old.teacher_objective.__code__.co_filename).read_bytes()),
        within_loss_code=sha(Path(objective.__code__.co_filename).read_bytes()),
        checkpoint_code=sha(Path(old.checkpoint.__file__).read_bytes()),source_factory=old.source.fingerprint(),
        canonical_input_cache=data.cache_binding())

def prepare(model,seed):
    p=factory.prepare(model,seed)
    p['binding']=dict(fingerprint=fingerprint(),model=model,seed=seed,source=p['source'],dataset=p['dataset'],
        initial_model_sha256=factory.model_sha(p['found']),frozen_nonprojection_core_sha256=p['frozen_core_sha256'],
        input_statistics_sha256=p['statistics_sha256'],teacher_model_sha256=old.source.MODELS[str(seed)],
        within_coefficient=16. if model in ('244','245') else 1.,optimizer_reset=True,DEV_GT_read=False)
    return p

def batch_loss(p,ids,device):
    rgb,labels=data.read_batch(None,ids)
    targets=old.teacher_targets(p['teacher'],rgb,labels)
    gamma=p['found'].encode(torch.from_numpy(rgb).to(device))
    frequency=old.helpers.effective_frequency(p['found'].wrapped.core,gamma)
    targets={k:v.to(device) for k,v in targets.items()}
    groups=groups_from_labels(labels)
    loss,parts=objective(gamma,frequency,targets,groups,p['binding']['within_coefficient'])
    return loss,parts

def params(p):return list(p['found'].encoder.parameters())+old.projection.parameters(p['found'])

def probe(p,device):
    initial=factory.model_sha(p['found'])
    loss,parts=batch_loss(p,scope.batch_ids(p['seed'],0),device)
    gradients=torch.autograd.grad(loss,params(p))
    norms=[float(g.detach().norm()) for g in gradients]
    if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(g).all()) for g in gradients) or norms[-1]<=0:raise ValueError('actual objective/encoder/projection gradient admission failed')
    if factory.model_sha(p['found'])!=initial or p['optimizer'].state:raise ValueError('gradient probe mutated state')
    return dict(loss=float(loss.detach()),projection_gradient_norm=norms[-1],all_gradients_finite=True,optimizer_updates=0)

def admit(model,seed):
    p=prepare(model,seed);a=archive(model,seed)
    proof=probe(p,'cpu')
    result=dict(status='actual_CPU_checkpoint_and_objective_admitted',binding=p['binding'],probe=proof,CUDA_initialized=False)
    path=a/'admission.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('existing CPU admission differs')
    else:write_json(path,result)
    return result

def invariant(p):
    return old.projection.frozen_core_sha(p['found'])==p['binding']['frozen_nonprojection_core_sha256'] and old.helpers.statistics_sha(p['found'])==p['binding']['input_statistics_sha256'] and factory.model_sha(p['teacher'])==p['binding']['teacher_model_sha256']

def history_check(history,seed,terminal=True):
    if terminal and len(history)!=scope.UPDATES:raise ValueError('incomplete35000 training budget')
    exposures=0
    for i,row in enumerate(history):
        ids=scope.batch_ids(seed,i)
        if row['update']!=i+1 or row['TRAIN_ids']!=ids.tolist() or not all(np.isfinite(row[k]) for k in ('loss','frequency_denoising','gamma_anchor','within_variance','gradient_norm')):raise ValueError('actual history/order/finite-loss mismatch')
        exposures+=len(ids)
    if terminal and exposures!=350000:raise ValueError('incomplete unique coverage')
    return exposures

def fit(p,device,runtime):
    a=archive(p['model'],p['seed']);work=a/'encoder_work';opt=p['optimizer'];found=p['found']
    admission=json.loads((a/'admission.json').read_bytes())
    if admission['binding']!=p['binding']:raise ValueError('worker source differs from CPU admission')
    binding=dict(binding=p['binding'],admission_sha256=sha((a/'admission.json').read_bytes()))
    if (work/'boundary.ptx').exists():
        payload=old.checkpoint.load_latest(work,wrapped=old.projection.module(found),encoder=found.encoder,optimizer=opt,expected_bindings=binding,expected_runtime=runtime,device=device,max_updates=scope.UPDATES)
        history=payload['history'];history_check(history,p['seed'],False)
    else:
        torch.manual_seed(249000+p['seed']);torch.cuda.manual_seed(249000+p['seed']);history=[]
    start=time.monotonic();saved_active=sum(float(row.get('active_step_seconds',0)) for row in history)
    for step in range(len(history),scope.UPDATES):
        t=time.monotonic();ids=scope.batch_ids(p['seed'],step);opt.zero_grad(set_to_none=True)
        loss,parts=batch_loss(p,ids,device)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params(p),1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite gradient')
        opt.step()
        if not all(bool(torch.isfinite(v).all()) for v in params(p)):raise FloatingPointError('nonfinite trained weights')
        history.append(dict(update=step+1,TRAIN_ids=ids.tolist(),loss=float(loss.detach()),gradient_norm=float(norm),
            frequency_denoising=float(parts['frequency_denoising'].detach()),gamma_anchor=float(parts['gamma_anchor'].detach()),
            within_variance=float(parts['within_variance'].detach()),active_step_seconds=time.monotonic()-t))
        epoch_end=(step+1)%scope.BATCHES==0
        milestone=(step+1) in (157,625) or epoch_end and (step+1)//scope.BATCHES in (1,2,4,6,8,10)
        if (step+1)%32==0 or milestone or step+1==scope.UPDATES:
            old.checkpoint.save_boundary(work,completed_updates=len(history),wrapped=old.projection.module(found),encoder=found.encoder,optimizer=opt,history=history,bindings=binding,runtime=runtime,device=device)
            if milestone:
                snapshot=a/'milestones'/f'update{step+1:05d}'/'boundary.ptx';snapshot.parent.mkdir(parents=True,exist_ok=True)
                if not snapshot.exists():shutil.copyfile(work/'boundary.ptx',snapshot)
            exposed=sum(len(row['TRAIN_ids']) for row in history)
            write_json(a/'progress.json',dict(model=p['model'],seed=p['seed'],completed_updates=len(history),total_updates=scope.UPDATES,
                complete_passes=len(history)//scope.BATCHES,stage_unique_seen=min(exposed,scope.COUNT),exposures=exposed,
                active_step_seconds=saved_active+time.monotonic()-start,updated_unix=time.time()),replace=True)
        del loss,parts
    exposures=history_check(history,p['seed'])
    if {int(v['step'].item()) for v in opt.state.values()}!={scope.UPDATES} or not invariant(p):raise ValueError('terminal Adam/core/teacher/statistics invariant failed')
    report=dict(status='completed_matched35000_training',binding=p['binding'],checkpoint_binding=binding,runtime=runtime,
        checkpoint_sha256=sha((work/'boundary.ptx').read_bytes()),final_model_sha256=factory.model_sha(found),
        actual_Adam_steps=[scope.UPDATES],unique_TRAIN_images=35000,complete_passes=10,exposures=exposures,
        active_step_seconds=sum(float(row['active_step_seconds']) for row in history),mask_supervised=True)
    path=a/'training_report.json'
    if path.exists():
        if json.loads(path.read_bytes())!=report:raise ValueError('terminal report changed')
    else:write_json(path,report)
    return report

def worker(model,seed,gpu):
    p=prepare(model,seed);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    old.source.source.source.model64.move_to_device(p['found'],device)
    runtime=old.resume_policy.compatible_runtime(old.checkpoint.runtime_signature(device))
    if runtime!=p['source']['runtime']:raise ValueError('source numerical runtime differs')
    proof=probe(p,device)
    # Actual production-spike preflight, before the first new optimizer update.
    rgb,_=data.read_batch(None,scope.batch_ids(seed,0)[:1])
    torch.manual_seed(193000);torch.cuda.manual_seed(193000)
    with torch.no_grad():
        values=old.source.source.full.screen.native._native_forward(p['found'],torch.from_numpy(rgb).to(device),live_tail_steps=0)
        gamma,trace,adj,z,q,rho=values
        labels=old.source.source.full.screen.prediction_from_trace(p['found'],trace,adj,q)
    if np.asarray(labels).shape!=(64,64):raise ValueError('actual spike preflight shape differs')
    write_json(archive(model,seed)/'GPU_preflight.json',dict(binding=p['binding'],proof=proof,physical_gpu=gpu,readout_passed=True),replace=True)
    del values,gamma,trace,adj,z,q,rho
    report=fit(p,device,runtime)
    from . import evaluate
    evaluate.predict(p,report,device)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=scope.MODELS,required=True);parser.add_argument('--seed',type=int,choices=scope.SEEDS,required=True)
    parser.add_argument('--admit',action='store_true');parser.add_argument('--worker',action='store_true');parser.add_argument('--reproduce',action='store_true');parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.admit:print(json.dumps(admit(args.model,args.seed)),flush=True)
    elif args.reproduce:factory.reproduce(args.seed,'cuda:0')
    elif args.worker:worker(args.model,args.seed,args.gpu)
    else:parser.error('select admission, reproduction, or worker')
