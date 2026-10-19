"""Frozen source object-frequency teacher, with original gamma anchoring."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0221_native64_frequency_pretraining import train as parent
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher,teacher_objective
from . import projection,bootstrap
from collaborative_test.SW_0222_native64_frequency_teacher_denoising import train as control

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive'
SEEDS=(0,1,2);COUNTS=(64,);STEPS=512;BATCH=16
source=parent.source;checkpoint=parent.checkpoint;resume_policy=parent.resume_policy
sha=parent.sha;write_json=parent.write_json;state_sha=parent.state_sha;statistics_sha=parent.statistics_sha
save_arrays=parent.save_arrays

def optimizer_for(found):return projection.enable(found,parent.optimizer_for(found))
effective_frequency=parent.effective_frequency


def require_job(seed,count):
    if type(seed) is not int or seed not in SEEDS or type(count) is not int or count not in COUNTS:raise ValueError('registered seed0/1/2 and TRAIN64 required')
    return seed,count


def archive_for(seed,count):require_job(seed,count);return ARCHIVE/f'seed{seed}'/f'TRAIN{count}'


def batch_indices(seed,count,step):
    if count<BATCH or count%BATCH or type(step) is not int or step<0:raise ValueError('complete fixed-size batches required')
    epoch=(step*BATCH)//count;offset=(step*BATCH)%count
    return np.random.RandomState(221000+seed+10000*epoch).permutation(count)[offset:offset+BATCH].tolist()


def fingerprint():
    names=('DESIGN_KO.md','projection.py','train.py','evaluate.py','queue.py','test_projection.py','test_pipeline.py','prefix_guard.py','test_prefix_guard.py','bootstrap.py','test_bootstrap.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},parent=parent.fingerprint(),source_teacher_preparation=control.fingerprint(),matched_128_control=projection.control.fingerprint(),seeds=list(SEEDS),TRAIN_counts=list(COUNTS),
        encoder_updates=STEPS,batch=BATCH,learning_rates=dict(source_base=parent.BASE_LR,context_encoder=parent.CONTEXT_LR,sensory_projection=parent.CONTEXT_LR),
        target='fixed_source_TRAIN_object_frequency_centers',matched_prefix_updates=128,convergence_only=True,resume_actual_accepted_checkpoint=True,additional_updates=384,gamma_anchor_coefficient=1.,frequency_coefficient=1.,
        core_projection_updates=STEPS,other_core_updates=0,native_grid=[64,64],production_readout_unchanged=True)


def prepare_cpu(seed,count):
    require_job(seed,count);p=control.prepare_cpu(seed,count);found=p['found']
    if projection.module(found).weight.shape!=(4,8):raise ValueError('actual 32-weight core projection required')
    p['binding']['fingerprint']=fingerprint()
    p['binding']['source_core_sha256']=p['binding'].pop('frozen_core_sha256')
    p['binding']['frozen_nonprojection_core_sha256']=projection.frozen_core_sha(found)
    p['binding']['initial_projection_sha256']=state_sha(projection.module(found).state_dict())
    p['optimizer']=projection.enable(found,p['optimizer'])
    p['control_labels'],p['binding']['matched_128_control']=projection.control_reference(seed,p['binding'])
    return p


def admit(seed,count):
    p=prepare_cpu(seed,count);a=archive_for(seed,count)
    before=measure(p['found'],p['rgb'],p['targets'],'cpu')
    old_path=parent.archive_for(seed,count)/'initial.json';old=json.loads(old_path.read_bytes())
    if old['binding']['TRAIN_ids']!=p['binding']['TRAIN_ids'] or old['binding']['parent_model_sha256']!=p['binding']['parent_model_sha256']:raise ValueError('different source/TRAIN control')
    error=abs(before['frequency_denoising']-old['TRAIN_phase_loss']['within'])
    if before['gamma_anchor']>1e-12 or error>2e-6:raise ValueError('actual source teacher does not reproduce natural within-object variance')
    gradient_admission=projection.admit_gradient_cpu(p)
    accepted_start=bootstrap.admit_cpu(p)
    save_arrays(a/'baseline.npz',labels=p['baseline']);save_arrays(a/'control.npz',labels=p['control_labels']);save_arrays(a/'teacher.npz',**{k:v.numpy() for k,v in p['targets'].items()})
    result=dict(status='actual_TRAIN64_source_teacher_CPU_admitted',binding=p['binding'],CUDA_initialized=False,
        baseline_file_sha256=sha((a/'baseline.npz').read_bytes()),teacher_file_sha256=sha((a/'teacher.npz').read_bytes()),
        actual_initial_TRAIN_loss=before,SW0221_initial_source_control_sha256=sha(old_path.read_bytes()),natural_within_variance_abs_error=error,
        actual_projection_gradient_admission=gradient_admission,accepted_checkpoint_start=accepted_start,projection_parameter_count=int(projection.module(p['found']).weight.numel()))
    write_json(a/'admission.json',result);return result


def measure(found,rgb,targets,device):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(rgb),BATCH):
            gamma=found.encode(torch.from_numpy(rgb[start:start+BATCH]).to(device));frequency=effective_frequency(found.wrapped.core,gamma)
            loss,parts=teacher_objective(gamma,frequency,{k:v[start:start+BATCH] for k,v in targets.items()})
            rows.append([float(loss.cpu()),float(parts['frequency_denoising'].cpu()),float(parts['gamma_anchor'].cpu())])
    values=np.asarray(rows).mean(0);return dict(total=float(values[0]),frequency_denoising=float(values[1]),gamma_anchor=float(values[2]))


def fit(p,device,runtime):
    found=p['found'];optimizer=p['optimizer'];seed=p['seed'];count=p['count'];a=archive_for(seed,count);device=torch.device(device);targets=p['targets']
    if projection.frozen_core_sha(found)!=p['binding']['frozen_nonprojection_core_sha256'] or state_sha(targets)!=p['binding']['teacher_arrays_sha256']:raise ValueError('source/teacher changed before fit')
    save_arrays(a/'teacher.npz',**{k:v.numpy() for k,v in targets.items()})
    teacher_sha=sha((a/'teacher.npz').read_bytes());dummy=projection.module(found);initial=a/'initial.json';reference=a/'reference_frequency.npz'
    if not initial.exists():
        if state_sha(found.wrapped.state_dict())!=p['binding']['source_core_sha256']:raise ValueError('source projection changed before initial reference')
        before=measure(found,p['rgb'],targets,device);frequency=[]
        with torch.no_grad():
            for image in p['dev']:frequency.append(effective_frequency(found.wrapped.core,found.encode(torch.from_numpy(image[None]).to(device)))[0].cpu().numpy())
        save_arrays(reference,frequency=np.stack(frequency))
        write_json(initial,dict(binding=p['binding'],TRAIN_phase_loss=before,reference_frequency_file_sha256=sha(reference.read_bytes()),teacher_file_sha256=teacher_sha,initial_encoder_sha256=state_sha(found.encoder.state_dict())))
    init=json.loads(initial.read_bytes())
    if init['binding']!=p['binding'] or init['teacher_file_sha256']!=teacher_sha or sha(reference.read_bytes())!=init['reference_frequency_file_sha256']:raise ValueError('frozen teacher/reference changed')
    binding=dict(binding=p['binding'],initial_metadata_sha256=sha(initial.read_bytes()));work=a/'encoder_work'
    if (work/'boundary.ptx').exists():
        loaded=checkpoint.load_latest(work,wrapped=dummy,encoder=found.encoder,optimizer=optimizer,expected_bindings=binding,expected_runtime=runtime,device=device,max_updates=STEPS);history=loaded['history']
        if len(history)>=128:
            from . import prefix_guard
            if len(history)==128 and not (a/'prefix128_proof.json').exists():prefix_guard.verify_and_save(p,history,work/'boundary.ptx')
            else:prefix_guard.validate_saved(p)
    else:
        if state_sha(found.wrapped.state_dict())!=p['binding']['source_core_sha256'] or state_sha(found.encoder.state_dict())!=init['initial_encoder_sha256']:raise ValueError('uncommitted source state changed')
        history=bootstrap.start(p,binding,work,runtime,device)
    for step in range(len(history),STEPS):
        indices=batch_indices(seed,count,step);optimizer.zero_grad(set_to_none=True)
        gamma=found.encode(torch.from_numpy(p['rgb'][indices]).to(device));frequency=effective_frequency(found.wrapped.core,gamma)
        loss,parts=teacher_objective(gamma,frequency,{k:v[indices] for k,v in targets.items()})
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite teacher loss')
        loss.backward();projection_norm=torch.linalg.vector_norm(projection.module(found).weight.grad.detach())
        norm=torch.nn.utils.clip_grad_norm_(list(found.encoder.parameters())+projection.parameters(found),1.)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite encoder gradient')
        optimizer.step()
        if any(not bool(torch.isfinite(v).all()) for v in list(found.encoder.parameters())+projection.parameters(found)):raise FloatingPointError('nonfinite encoder; preserve previous committed boundary')
        history.append(dict(update=step+1,indices=indices,loss=float(loss.detach().cpu()),frequency_denoising=float(parts['frequency_denoising'].detach().cpu()),gamma_anchor=float(parts['gamma_anchor'].detach().cpu()),preclip_gradient_norm=float(norm.cpu()),projection_gradient_norm=float(projection_norm.cpu())))
        checkpoint.save_boundary(work,completed_updates=len(history),wrapped=dummy,encoder=found.encoder,optimizer=optimizer,history=history,bindings=binding,runtime=runtime,device=device)
        if len(history)==128:
            from . import prefix_guard
            prefix_guard.verify_and_save(p,history,work/'boundary.ptx')
        del gamma,frequency,loss,parts
    if any(row['indices']!=batch_indices(seed,count,i) or row['update']!=i+1 for i,row in enumerate(history)):raise ValueError('TRAIN order changed')
    steps=sorted({int(v['step'].item()) for v in optimizer.state.values()})
    if steps!=[STEPS] or projection.frozen_core_sha(found)!=p['binding']['frozen_nonprojection_core_sha256'] or statistics_sha(found)!=p['binding']['input_statistics_sha256']:raise ValueError('counter/core/statistics invariant failed')
    from . import prefix_guard
    if STEPS>=128:prefix_guard.validate_saved(p)
    path=a/'training_report.json'
    if path.exists():
        report=json.loads(path.read_bytes())
        if report['binding']!=p['binding'] or report['checkpoint_binding']!=binding or report['history']!=history or report['checkpoint_file_sha256']!=sha((work/'boundary.ptx').read_bytes()) or report['final_encoder_sha256']!=state_sha(found.encoder.state_dict()) or report['final_model_sha256']!=source.source.full.screen.native._state_sha(found.wrapped,found.encoder) or report['teacher_file_sha256']!=teacher_sha:raise ValueError('completed report changed')
        return report
    report=dict(status='complete_native64_joint_core_sensory_projection',binding=p['binding'],checkpoint_binding=binding,history=history,
        actual_Adam_steps=steps,inherited_TRAIN_stage_optimizer_updates=128,new_TRAIN_stage_optimizer_updates=STEPS-128,additional_TRAIN_exposures=(STEPS-128)*BATCH,encoder_updates=STEPS,core_projection_updates=STEPS,other_core_updates=0,unique_TRAIN_stage_images=count,logical_TRAIN_exposures=STEPS*BATCH,
        initial_TRAIN_phase_loss=init['TRAIN_phase_loss'],final_TRAIN_phase_loss=measure(found,p['rgb'],targets,device),teacher_file_sha256=teacher_sha,
        final_encoder_sha256=state_sha(found.encoder.state_dict()),final_model_sha256=source.source.full.screen.native._state_sha(found.wrapped,found.encoder),
        source_core_sha256=p['binding']['source_core_sha256'],frozen_nonprojection_core_sha256=projection.frozen_core_sha(found),final_core_sha256=state_sha(found.wrapped.state_dict()),
        initial_projection_sha256=p['binding']['initial_projection_sha256'],final_projection_sha256=state_sha(projection.module(found).state_dict()),
        projection_changed=state_sha(projection.module(found).state_dict())!=p['binding']['initial_projection_sha256'],input_statistics_sha256=statistics_sha(found),encoder_changed=state_sha(found.encoder.state_dict())!=init['initial_encoder_sha256'],
        reference_frequency_file_sha256=init['reference_frequency_file_sha256'],checkpoint_file_sha256=sha((work/'boundary.ptx').read_bytes()),DEV_GT_read=False,new70000_training=False,runtime=runtime)
    if not report['projection_changed'] or history[0]['projection_gradient_norm']<=0:raise ValueError('declared actual core projection received no learning credit')
    write_json(path,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS,required=True);parser.add_argument('--count',type=int,choices=COUNTS,required=True);parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    if args.admit:print(json.dumps(admit(args.seed,args.count)))
    else:parser.error('use the admitted exclusive GPU queue')
