"""Matched encoder-only precision stages from the audited best checkpoint."""
import argparse,io,json,os,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0222_native64_frequency_teacher_denoising import train as teacher_stage
from collaborative_test.SW_0221_native64_frequency_pretraining import train as helpers,evaluate as reader
from collaborative_test.SW_0237_native64_converged_full320 import run as reference,evaluate as reference_score
from .loss import groups_from_labels,groups_to_device,objective
from collaborative_test.SW_0244_native64_within_precision import run as previous
from .scope import aggregate_vectors

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ROOT_ARCHIVE=HERE/'results_archive';ARCHIVE=ROOT_ARCHIVE;ARM=None
ARMS={'control':1.,'within16':16.};SEEDS=(0,1,2);COUNT=64;BATCH=16;STEPS=128;CHECKPOINT_EVERY=16
OFFSETS=tuple(range(0,320,5))
sha=helpers.sha;write_json=helpers.write_json;state_sha=helpers.state_sha
checkpoint=helpers.checkpoint;resume_policy=helpers.resume_policy


def require_seed(seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('registered seed1 screen required')
    return seed


def require_arm(arm):
    if arm not in ARMS:raise ValueError('registered control/within16 arm required')
    return arm


def activate(arm):
    global ARM,ARCHIVE
    require_arm(arm)
    if ARM is not None and ARM!=arm:raise ValueError('one immutable arm per process')
    ARM=arm;ARCHIVE=ROOT_ARCHIVE/arm


def archive_for(seed,arm=None):return ROOT_ARCHIVE/require_arm(ARM if arm is None else arm)/f'seed{require_seed(seed)}'


def family_fingerprint():
    names=('DESIGN_KO.md','loss.py','run.py','queue.py','test_loss.py','test_training.py','test_scope.py','conftest.py','scope.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},parent237=reference.fingerprint(),
        screening244=previous.family_fingerprint(),screening244_score_sha256=sha((previous.ROOT_ARCHIVE/'score_v1/summary.json').read_bytes()),
        original_teacher_preparation=teacher_stage.fingerprint(),checkpoint_code_sha256=sha(Path(checkpoint.__file__).read_bytes()),
        arms=ARMS,seeds=list(SEEDS),TRAIN_count=COUNT,batch=BATCH,stage_updates=STEPS,source_checkpoint_updates=512,
        evaluation_offsets=list(OFFSETS),whole_core_including_projection_frozen=True,optimizer_reset=True,
        learning_rates=dict(source_base=helpers.BASE_LR,context=helpers.CONTEXT_LR),native_grid=[64,64],steps=1024,settle=512)


def fingerprint(arm=None):
    arm=require_arm(ARM if arm is None else arm)
    return dict(family=family_fingerprint(),arm=arm,within_coefficient=ARMS[arm])


def prepare_cpu(seed,arm=None):
    require_seed(seed);arm=require_arm(ARM if arm is None else arm)
    screening=json.loads((previous.ROOT_ARCHIVE/'score_v1/summary.json').read_bytes())
    if screening['fingerprint']!=previous.family_fingerprint() or not screening['within16_promising_for_separate_three_seed_confirmation']:
        raise ValueError('accepted positive244 screen required')
    if torch.cuda.is_initialized():raise ValueError('actual CPU source admission before CUDA required')
    p=teacher_stage.prepare_cpu(seed,COUNT);teacher_binding=dict(p['binding']);found=p['found']
    _,source_ref,_,_=reference.scope.restore(found,seed)
    p['optimizer']=helpers.optimizer_for(found)
    accepted=reference.scope.model.projection.control.archive_for(seed,COUNT)
    with np.load(accepted/'teacher.npz',allow_pickle=False) as old:
        if set(old.files)!=set(p['targets']) or any(not np.array_equal(old[k],v.numpy()) for k,v in p['targets'].items()):
            raise ValueError('original accepted teacher arrays changed')
    control=reference.validated_seed(seed);dataset=control['protocol']['binding']['dataset']
    if any(dataset[k]!=teacher_binding['parent']['dataset'][k] for k in ('path','size','mtime_ns')):
        raise ValueError('TRAIN and validation source dataset differ')
    if p['rgb'].shape!=(COUNT,128,128,3) or p['labels'].shape!=(COUNT,64,64):
        raise ValueError('registered native64 TRAIN geometry required')
    import h5py
    with h5py.File(dataset['path'],'r') as stream:rgb=reference.scope.canonical_rgb(np.asarray(stream['image'][1320:1640]))
    if sha(rgb.tobytes())!=control['protocol']['binding']['RGB_sha256']:raise ValueError('canonical validation RGB changed')
    stat=Path(dataset['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(dataset['size'],dataset['mtime_ns']):raise ValueError('dataset changed')
    score_raw=(reference.ARCHIVE/'score_v1/summary.json').read_bytes();audit_raw=(reference.ARCHIVE/'completion_audit_v1/summary.json').read_bytes();audit=json.loads(audit_raw)
    if audit['score_sha256']!=sha(score_raw) or audit['raw_masks_independently_rescored']!=2880 or audit['independent_metrics_means_and_bootstrap_maximum_abs_error']>=1e-12:
        raise ValueError('audited current237 reference required')
    initial=reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)
    if initial!=source_ref['model_sha256'] or p['optimizer'].state:raise ValueError('actual512 checkpoint or fresh Adam differs')
    p.update(arm=arm,groups=groups_from_labels(p['labels']),dev=rgb[list(OFFSETS)].copy(),parent_control=control['labels'][list(OFFSETS)].copy())
    p['binding']=dict(fingerprint=fingerprint(arm),seed=seed,arm=arm,source=source_ref,source_checkpoint_updates=512,
        initial_model_sha256=initial,TRAIN_ids=teacher_binding['TRAIN_ids'],TRAIN_RGB_sha256=sha(p['rgb'].tobytes()),
        TRAIN_labels_sha256=sha(p['labels'].tobytes()),teacher_arrays_sha256=state_sha(p['targets']),
        teacher_preparation=teacher_binding,frozen_core_sha256=state_sha(found.wrapped.state_dict()),
        input_statistics_sha256=helpers.statistics_sha(found),dataset=dataset,DEV_RGB_sha256=sha(p['dev'].tobytes()),
        parent_control_sha256=sha(p['parent_control'].tobytes()),parent_score_sha256=sha(score_raw),parent_audit_sha256=sha(audit_raw),
        mask_supervised=True,DEV_GT_read=False,optimizer_reset=True)
    return p


def batch_indices(seed,count,step):
    epoch,offset=divmod(step,count//BATCH)
    return np.random.RandomState(221000+seed+10000*epoch).permutation(count)[offset*BATCH:(offset+1)*BATCH].tolist()


def batch_loss(p,indices,device,groups):
    gamma=p['found'].encode(torch.from_numpy(p['rgb'][indices]).to(device))
    frequency=helpers.effective_frequency(p['found'].wrapped.core,gamma)
    return objective(gamma,frequency,{k:v[indices] for k,v in p['targets'].items()},
        tuple(groups[i] for i in indices),ARMS[p['arm']])


def admit(seed):
    p=prepare_cpu(seed);found=p['found'];before=reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)
    loss,parts=batch_loss(p,list(range(BATCH)),'cpu',p['groups'])
    gradients=torch.autograd.grad(loss,tuple(found.encoder.parameters()))
    norm=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
    if not bool(torch.isfinite(loss)) or not all(bool(torch.isfinite(g).all()) for g in gradients) or norm<=0:
        raise ValueError('actual CPU objective/encoder gradient preflight failed')
    if reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=before or p['optimizer'].state or any(v.requires_grad for v in found.wrapped.parameters()) or any(v.grad is not None for v in found.encoder.parameters()):
        raise ValueError('CPU admission mutated source or unfroze core')
    a=archive_for(seed);helpers.save_arrays(a/'teacher.npz',**{k:v.numpy() for k,v in p['targets'].items()})
    helpers.save_arrays(a/'parent_control.npz',labels=p['parent_control'])
    result=dict(binding=p['binding'],cuda_initialized=False,optimizer_updates=0,actual_CPU_restore_passed=True,
        encoder_gradient_norm=norm,actual_objective=float(loss.detach()),within_variance=float(parts['within_variance'].detach()),
        teacher_file_sha256=sha((a/'teacher.npz').read_bytes()),parent_control_file_sha256=sha((a/'parent_control.npz').read_bytes()),
        all_core_parameters_including_projection_frozen=True)
    write_json(a/'admission.json',result);return result


def measure(p,device,groups):
    rows=[]
    with torch.no_grad():
        for start in range(0,len(p['rgb']),BATCH):
            value,parts=batch_loss(p,list(range(start,start+BATCH)),device,groups)
            rows.append([float(value),float(parts['frequency_denoising']),float(parts['gamma_anchor']),float(parts['within_variance'])])
    return dict(zip(('total','frequency_denoising','gamma_anchor','within_variance'),map(float,np.mean(rows,axis=0))))


def fit(p,device,runtime):
    device=torch.device(device);a=archive_for(p['seed'],p['arm']);found=p['found'];opt=p['optimizer']
    if json.loads((a/'admission.json').read_bytes())['binding']!=p['binding']:raise ValueError('admitted source differs')
    if state_sha(found.wrapped.state_dict())!=p['binding']['frozen_core_sha256'] or state_sha(p['targets'])!=p['binding']['teacher_arrays_sha256']:
        raise ValueError('initial core or frozen teacher changed')
    groups=groups_to_device(p['groups'],device);params=list(found.encoder.parameters());dummy=torch.nn.Identity().to(device)
    initial=a/'initial.json'
    if not initial.exists():write_json(initial,dict(binding=p['binding'],TRAIN_loss=measure(p,device,groups),initial_encoder_sha256=state_sha(found.encoder.state_dict())))
    init=json.loads(initial.read_bytes())
    if init['binding']!=p['binding']:raise ValueError('initial stage binding changed')
    binding=dict(binding=p['binding'],admission_sha256=sha((a/'admission.json').read_bytes()),initial_sha256=sha(initial.read_bytes()))
    work=a/'encoder_work'
    if (work/'boundary.ptx').exists():
        payload=checkpoint.load_latest(work,wrapped=dummy,encoder=found.encoder,optimizer=opt,expected_bindings=binding,expected_runtime=runtime,device=device,max_updates=STEPS);history=payload['history']
    else:
        torch.manual_seed(244000+p['seed'])
        if device.type=='cuda':torch.cuda.manual_seed(244000+p['seed'])
        history=[];checkpoint.save_boundary(work,completed_updates=0,wrapped=dummy,encoder=found.encoder,optimizer=opt,history=history,bindings=binding,runtime=runtime,device=device)
    for step in range(len(history),STEPS):
        indices=batch_indices(p['seed'],len(p['rgb']),step);opt.zero_grad(set_to_none=True)
        loss,parts=batch_loss(p,indices,device,groups)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite precision objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite encoder gradient')
        opt.step()
        if not all(bool(torch.isfinite(v).all()) for v in params):raise FloatingPointError('nonfinite encoder; committed boundary preserved')
        history.append(dict(update=step+1,indices=indices,loss=float(loss.detach()),frequency_denoising=float(parts['frequency_denoising'].detach()),
            gamma_anchor=float(parts['gamma_anchor'].detach()),within_variance=float(parts['within_variance'].detach()),gradient_norm=float(norm)))
        if (step+1)%CHECKPOINT_EVERY==0 or step+1==STEPS:
            checkpoint.save_boundary(work,completed_updates=len(history),wrapped=dummy,encoder=found.encoder,optimizer=opt,history=history,bindings=binding,runtime=runtime,device=device)
        del loss,parts
    if len(history)!=STEPS or any(row['update']!=i+1 or row['indices']!=batch_indices(p['seed'],len(p['rgb']),i) for i,row in enumerate(history)):
        raise ValueError('actual stage history/order differs')
    steps=sorted({int(v['step'].item()) for v in opt.state.values()})
    if steps!=[STEPS] or state_sha(found.wrapped.state_dict())!=p['binding']['frozen_core_sha256'] or state_sha(p['targets'])!=p['binding']['teacher_arrays_sha256'] or helpers.statistics_sha(found)!=p['binding']['input_statistics_sha256']:
        raise ValueError('actual Adam budget or frozen source invariant differs')
    result=dict(status='complete_matched_encoder_precision_stage',binding=p['binding'],checkpoint_binding=binding,
        actual_Adam_steps=steps,source_checkpoint_updates=512,new_encoder_updates=STEPS,new_core_updates=0,
        unique_TRAIN_stage_images=len(p['rgb']),stage_TRAIN_exposures=STEPS*BATCH,initial_TRAIN_loss=init['TRAIN_loss'],final_TRAIN_loss=measure(p,device,groups),
        final_encoder_sha256=state_sha(found.encoder.state_dict()),final_model_sha256=reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder),
        checkpoint_sha256=sha((work/'boundary.ptx').read_bytes()),history_sha256=sha(json.dumps(history,sort_keys=True,allow_nan=False).encode()),runtime=runtime,
        optimizer_reset=True,mask_supervised=True,DEV_GT_read=False,new70000_training=False)
    path=a/'training_report.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('completed precision report differs')
    else:write_json(path,result)
    return result


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed);device=torch.device('cuda:0')
    torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    found=p['found'];reference.source.source.source.model64.move_to_device(found,device)
    runtime=resume_policy.compatible_runtime(checkpoint.runtime_signature(device))
    if runtime!=p['binding']['source']['expected_runtime']:raise ValueError('accepted numerical runtime differs')
    report=fit(p,device,runtime);found.encoder.eval();found.wrapped.eval()
    binding=dict(binding=p['binding'],training_report_sha256=sha((a/'training_report.json').read_bytes()),model_sha256=report['final_model_sha256'])
    masks=[]
    for index,offset in enumerate(OFFSETS):
        folder=a/'rows'/f'{index:04d}'
        if not folder.exists():
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=reference.source.source.full.screen.native._native_forward(found,torch.from_numpy(p['dev'][index:index+1]).to(device),live_tail_steps=0)
                labels=np.asarray(reference.source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64)
                frequency=helpers.effective_frequency(found.wrapped.core,gamma)[0].cpu().numpy()
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            helpers.save_arrays(stage/'prediction.npz',labels=labels,frequency=frequency)
            write_json(stage/'row.json',dict(binding=binding,index=index,offset=offset,inference_seed=193000+offset,
                data_sha256=sha((stage/'prediction.npz').read_bytes()),DEV_GT_read=False))
            stage.rename(folder);del gamma,trace,adj,z,q,rho,labels,frequency
        labels,_=read_row(folder,binding,index);masks.append(labels)
    if reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=report['final_model_sha256']:
        raise ValueError('prediction changed trained parameters')
    result=dict(binding=binding,row_sha256=[sha((a/'rows'/f'{i:04d}'/'row.json').read_bytes()) for i in range(len(OFFSETS))],
        labels_sha256=sha(np.stack(masks).tobytes()),offsets=list(OFFSETS),DEV_GT_read=False)
    path=a/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('completed precision protocol differs')
    else:write_json(path,result)


def read_row(folder,binding,index):
    row=json.loads((folder/'row.json').read_bytes())
    if row['offset']!=OFFSETS[index] or row['inference_seed']!=193000+OFFSETS[index] or row['DEV_GT_read']:
        raise ValueError('fixed evaluation scope/RNG changed')
    return reader.read_row(folder,binding,index)


def validated_seed(seed,arm=None):
    arm=require_arm(ARM if arm is None else arm);a=archive_for(seed,arm)
    ad=json.loads((a/'admission.json').read_bytes());report=json.loads((a/'training_report.json').read_bytes());protocol=json.loads((a/'protocol.json').read_bytes())
    if ad['binding']['fingerprint']!=fingerprint(arm) or report['binding']!=ad['binding'] or report['actual_Adam_steps']!=[STEPS] or report['new_core_updates']!=0:
        raise ValueError('registered precision training differs')
    expected=dict(binding=ad['binding'],training_report_sha256=sha((a/'training_report.json').read_bytes()),model_sha256=report['final_model_sha256'])
    if protocol['binding']!=expected or protocol['offsets']!=list(OFFSETS) or protocol['DEV_GT_read'] or len(protocol['row_sha256'])!=len(OFFSETS):
        raise ValueError('all registered prediction rows must precede GT')
    meta=checkpoint.inspect_committed_metadata(a/'encoder_work')
    if meta['completed_updates']!=STEPS or meta['bindings']!=report['checkpoint_binding'] or sha((a/'encoder_work/boundary.ptx').read_bytes())!=report['checkpoint_sha256']:
        raise ValueError('actual committed stage checkpoint differs')
    if sha((a/'parent_control.npz').read_bytes())!=ad['parent_control_file_sha256'] or sha((a/'teacher.npz').read_bytes())!=ad['teacher_file_sha256']:
        raise ValueError('teacher or parent controls changed')
    values=[]
    for i,digest in enumerate(protocol['row_sha256']):
        folder=a/'rows'/f'{i:04d}'
        if sha((folder/'row.json').read_bytes())!=digest:raise ValueError('prediction row changed')
        values.append(read_row(folder,expected,i)[0])
    labels=np.stack(values)
    if sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('frozen mask stack changed')
    with np.load(a/'parent_control.npz',allow_pickle=False) as data:parent=data['labels'].copy()
    return dict(labels=labels,parent=parent,binding=ad['binding'],report=report)


def score():
    # Validate every model/seed protocol and all Slot controls before reading GT.
    bundles={arm:{seed:validated_seed(seed,arm) for seed in SEEDS} for arm in ARMS}
    for seed in SEEDS:
        base=bundles['control'][seed]['binding']
        for arm in ARMS:
            bundle=bundles[arm][seed]
            for key in ('TRAIN_ids','TRAIN_RGB_sha256','TRAIN_labels_sha256','teacher_arrays_sha256','initial_model_sha256','frozen_core_sha256','input_statistics_sha256','DEV_RGB_sha256','parent_control_sha256','dataset'):
                if bundle['binding'][key]!=base[key]:raise ValueError('matched seed arms differ beyond objective')
            if not np.array_equal(bundle['parent'],bundles['control'][seed]['parent']):raise ValueError('different parent controls')
    slots=reference_score.load_slot_masks();dataset=bundles['control'][0]['binding']['dataset'];stat=Path(dataset['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(dataset['size'],dataset['mtime_ns']):raise ValueError('evaluation dataset changed')
    if any(v['binding']['dataset']!=dataset for arm in ARMS for v in bundles[arm].values()):raise ValueError('seed datasets differ')
    gt=reference.source.source.full.read_gt(dict(dataset=dataset))[list(OFFSETS)]
    vectors={name:[] for name in ('control','within16','parent237','slot')};maximum=0.
    for seed in SEEDS:
        arrays={arm:bundles[arm][seed]['labels'] for arm in ARMS}
        arrays.update(parent237=bundles['control'][seed]['parent'],slot=slots[seed]['labels64'][list(OFFSETS)])
        for name,labels in arrays.items():
            values,error=reference.source.source.full.metrics.measured_vectors(labels,gt);vectors[name].append(values);maximum=max(maximum,float(error))
    if maximum>=1e-12:raise ValueError('primary and independent metrics differ')
    vectors={k:np.stack(v) for k,v in vectors.items()}
    result=aggregate_vectors(vectors,reference_score.METRICS)
    result.update(status='complete_matched_precision_three_seed64_confirmation',fingerprint=family_fingerprint(),seed_scope=list(SEEDS),image_count_per_seed=len(OFFSETS),
        unique_TRAIN_images=COUNT,new_Adam_updates_per_arm_per_seed=STEPS,source_checkpoint_updates=512,
        frozen_core_and_projection=True,mask_supervised=True,unsupervised_leaderboard_eligible=False,
        full320_improvement_proven=False,native64_full320_FG090_goal_verified=False,new70000_training=False,
        independent_metric_maximum_abs_error=maximum,current_full320_reference='SW0237')
    write_json(ROOT_ARCHIVE/'score_v1/evaluation_private.json',dict(summary=result,vectors={k:v.tolist() for k,v in vectors.items()}))
    write_json(ROOT_ARCHIVE/'score_v1/summary.json',result);return result


def audit_training(seed):
    p=prepare_cpu(seed);a=archive_for(seed);report=json.loads((a/'training_report.json').read_bytes())
    raw=(a/'encoder_work/boundary.ptx').read_bytes();outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if sha(raw)!=report['checkpoint_sha256'] or sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('corrupt precision checkpoint')
    saved=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False);history=saved['history']
    steps=sorted({int(v['step'].item()) for v in saved['optimizer_state']['state'].values()})
    if report['binding']!=p['binding'] or saved['completed_updates']!=STEPS or len(history)!=STEPS or steps!=[STEPS] or saved['bindings']!=report['checkpoint_binding'] or saved['optimizer_schema']!=checkpoint.optimizer_schema(p['optimizer']):
        raise ValueError('actual precision source/Adam/history differs')
    if any(row['indices']!=batch_indices(seed,COUNT,i) or row['update']!=i+1 for i,row in enumerate(history)) or sha(json.dumps(history,sort_keys=True,allow_nan=False).encode())!=report['history_sha256']:
        raise ValueError('actual precision update order differs')
    p['found'].encoder.load_state_dict(saved['encoder_state'],strict=True);p['optimizer'].load_state_dict(saved['optimizer_state'])
    if state_sha(p['found'].encoder.state_dict())!=report['final_encoder_sha256'] or state_sha(p['found'].wrapped.state_dict())!=p['binding']['frozen_core_sha256'] or reference.source.source.full.screen.native._state_sha(p['found'].wrapped,p['found'].encoder)!=report['final_model_sha256']:
        raise ValueError('actual full precision model restoration differs')
    result=dict(status='actual_CPU_precision_checkpoint_Adam_history_and_frozen_core_audited',actual_Adam_steps=steps,
        actual_stage_updates=STEPS,source_checkpoint_updates=512,all_core_and_projection_parameters_unchanged=True,
        actual_final_model_restored_on_CPU=True,optimizer_updates=0,additional_native_forwards=0)
    path=a/'training_audit.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('previous precision audit differs')
    else:write_json(path,result)
    return result


if __name__=='__main__':
    import ctypes,sys
    if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP disable failed')
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=tuple(ARMS),required=True);parser.add_argument('--seed',type=int,choices=SEEDS,default=1)
    parser.add_argument('--admit',action='store_true');parser.add_argument('--score',action='store_true');parser.add_argument('--audit-training',action='store_true');args=parser.parse_args();activate(args.arm)
    if args.admit:print(json.dumps(admit(args.seed)))
    elif args.audit_training:print(json.dumps(audit_training(args.seed)))
    elif args.score:print(json.dumps(score()))
    else:parser.error('use CPU admission, completed scoring, or checkpoint audit')
