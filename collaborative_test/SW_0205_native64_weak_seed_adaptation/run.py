"""Two resumable native64 training proposals before the larger mini comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from collaborative_test.SW_0192_native64_incumbent_mini import model64
from collaborative_test.SW_0197_native64_baseline_replication import run as baseline
from collaborative_test.SW_0193_native64_probe_screen import score as metrics
from collaborative_test.SW_0192_native64_incumbent_mini import run as train
from collaborative_test.SW_0186_full_encoder_qce import train as incumbent
from collaborative_test.SW_0175_critical_component_binding import train_v2 as native
from collaborative_test.SW_0191_partition_error_feedback import source as warm,resume_policy
from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint

HERE=Path(__file__).resolve().parent
ROOT_ARCHIVE=HERE/'results_archive'
ARCHIVE=ROOT_ARCHIVE
SEED=None
ARMS=('candidate',)
BATCH=4
PROPOSALS=2


def sha(raw):return hashlib.sha256(raw).hexdigest()


def write_json(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        if replace:os.replace(temp,path)
        else:os.link(temp,path)
    finally:
        if temp.exists():temp.unlink()


def fingerprint():
    return dict(experiment='SW0205_native64_weak_seed_adaptation',
        files={name:sha((HERE/name).read_bytes()) for name in
            ('DESIGN_KO.md','run.py','queue.py','test_train.py','evaluate.py')},
        parent_factory=warm.fingerprint_sha256(),mapped_model_source_sha256=sha(Path(model64.__file__).read_bytes()),seed=SEED,batch=BATCH,probe_proposals=PROPOSALS,
        objective='product_QCE_temperature0.1',training_geometry=[64,64],
        steps=1024,settle=512,live_tail=64,TRAIN_images_for_probe=8)


def prepare_inputs_cpu():
    import h5py
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    if torch.cuda.is_initialized():raise RuntimeError('admit before CUDA')
    inputs=incumbent.prepare_training_inputs_cpu()
    ids=list(map(int,inputs['ordered_train_ids'][:64]))
    if len(set(ids))!=64 or any(1000<=i<1640 for i in ids):raise ValueError('TRAIN64 must exclude validation')
    from collaborative_test.SW_0130_phase_state_integration import run as source_data
    dataset=Path(source_data.source97.DATASET)
    with h5py.File(dataset,'r') as stream:
        if stream['mask'].shape!=(100000,128,128,1):raise ValueError('canonical128 mask schema')
        pixels=np.stack([stream['mask'][i,...,0] for i in ids])
    modal=clevr_mask_patch(torch.from_numpy(np.ascontiguousarray(pixels)),2)['patch_labels'].cpu().numpy().astype(np.int64)
    if modal.shape!=(64,64,64) or np.any(modal<0):raise ValueError('direct128->64 TRAIN labels invalid')
    found,optimizer,families,proof=model64.build_from_incumbent(SEED,inputs)
    binding=dict(experiment='SW0205_native64_training_admission',fingerprint=fingerprint(),
        factory_proof=proof,ordered_train_ids=ids,TRAIN64_modal_sha256=sha(modal.tobytes()),
        source_dataset_arrays_sha256=inputs['dataset_arrays_sha256'],
        validation_data_read=False)
    old,reference=baseline.validated_pair(SEED)
    if reference['model_sha256']!=proof['native64_initial_model_state_sha256']:raise ValueError('wrong independent seed baseline')
    if native._state_sha(found.wrapped,found.encoder)!=reference['model_sha256']:raise ValueError('actual initial model changed')
    binding.update(reference_binding=reference['reference_binding'],baseline_model_sha256=reference['model_sha256'],
        baseline_protocol_sha256=sha((baseline.archive_for(SEED)/'prediction/protocol.json').read_bytes()),baseline_masks_sha256=sha(old.tobytes()))
    return dict(found=found,optimizer=optimizer,families=families,bindings=binding,
        rgb=inputs['arrays']['rgb'],labels=modal)


def batch_loss(prepared,indices,device,*,live,credit=False):
    from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss
    found=prepared['found'];losses=[];emitted_sq=0.
    for index in indices:
        rgb=torch.from_numpy(np.ascontiguousarray(prepared['rgb'][index:index+1])).to(device)
        with torch.set_grad_enabled(live):
            gamma,trace,adj,z,q,rho=native._native_forward(found,rgb,live_tail_steps=64 if live else 0)
            if gamma.shape!=(1,8,4096) or q.shape!=(1,4096,4096):raise ValueError('actual native64 forward geometry')
            labels=torch.from_numpy(prepared['labels'][index].reshape(-1)).to(device)
            loss=_vectorized_class_loss(q[0],labels,temperature=.1)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite native64 QCE')
            losses.append(float(loss.detach().cpu()))
            if live:
                if credit:
                    gradient=torch.autograd.grad(loss,trace['component_spikes'],retain_graph=True)[0]
                    if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite emitted-S gradient')
                    emitted_sq+=float(gradient.detach().double().square().sum().cpu())
                    del gradient
                (loss/len(indices)).backward()
        del gamma,trace,adj,z,q,rho,rgb,loss,labels
    return float(np.mean(losses)),math.sqrt(emitted_sq)


def run_probe(prepared,output,work,device='cuda:0'):
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('real GPU admission required')
    torch.cuda.set_device(device)
    found=prepared['found'];model64.move_to_device(found,device)
    optimizer=prepared['optimizer'];families=prepared['families']
    optimized={id(p) for g in optimizer.param_groups for p in g['params']}
    if optimized!={id(p) for entries in families.values() for _,p in entries}:raise ValueError('optimizer parameters changed during device move')
    runtime=resume_policy.compatible_runtime(checkpoint.runtime_signature(device))
    work=Path(work);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if (work/'boundary.ptx').exists():
        loaded=checkpoint.load_latest(work,wrapped=found.wrapped,encoder=found.encoder,optimizer=optimizer,
            expected_bindings=prepared['bindings'],expected_runtime=runtime,device=device,max_updates=PROPOSALS)
        history=loaded['history'];cursor=loaded['completed_updates']
    else:
        if work.exists() and any(work.iterdir()) and not checkpoint.is_only_uncommitted_temp(work):raise FileExistsError('preserving unexpected partial work')
        history=[];cursor=0
        checkpoint.save_boundary(work,completed_updates=0,wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,history=history,bindings=prepared['bindings'],runtime=runtime,device=device)
    if cursor!=len(history) or any(row['proposal']!=i+1 for i,row in enumerate(history)):raise ValueError('probe cursor/history mismatch')
    for index in range(cursor,PROPOSALS):
        indices=list(range(index*BATCH,(index+1)*BATCH))
        torch.manual_seed(192000+index);torch.cuda.manual_seed(192000+index)
        optimizer.zero_grad(set_to_none=True)
        modules={'wrapped':found.wrapped,'encoder':found.encoder}
        snapshot=tx.capture_transaction(modules,optimizer,offload_to_cpu=True)
        initial=tx.transaction_digest(snapshot)
        torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
        before,_=batch_loss(prepared,indices,device,live=False)
        tx.restore_rng_state(snapshot['rng'])
        grad_started=time.perf_counter()
        measured,emitted=batch_loss(prepared,indices,device,live=True,credit=True)
        grad_seconds=time.perf_counter()-grad_started
        if abs(measured-before)>=max(1e-7,1e-6*abs(before)):raise RuntimeError('same-RNG full-forward loss parity failed')
        rng_after=tx.capture_rng_state()
        gradients={name:math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu()) for _,p in entries if p.grad is not None)) for name,entries in families.items()}
        if any(not math.isfinite(v) for v in gradients.values()):raise FloatingPointError('nonfinite native64 family gradient')
        clipped=torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],1.)
        if not torch.isfinite(clipped):raise FloatingPointError('nonfinite clipped gradient')
        old={name:[p.detach().clone() for _,p in entries] for name,entries in families.items()}
        optimizer.step();found.wrapped.project_integrations_()
        deltas={name:math.sqrt(sum(float((p.detach().double()-prior.double()).square().sum().cpu()) for (_,p),prior in zip(entries,old[name]))) for name,entries in families.items()}
        del old
        tx.restore_rng_state(snapshot['rng'])
        after,_=batch_loss(prepared,indices,device,live=False)
        accepted,threshold=incumbent._acceptance(before,after)
        failures=[]
        if not math.isfinite(emitted) or emitted<=0:failures.append('no emitted spike credit')
        for name in families:
            if not math.isfinite(deltas[name]) or gradients[name]<=0 or deltas[name]<=0:failures.append(name+' lacks gradient/proposed delta')
        if accepted:
            optimizer.zero_grad(set_to_none=True);tx.restore_rng_state(rng_after)
        else:tx.restore_transaction(snapshot,modules,optimizer,restore_rng=True,clear_grad=True)
        torch.cuda.synchronize(device)
        row=dict(proposal=index+1,indices=indices,accepted=accepted,before_loss=before,after_loss=after,
            parity_error=abs(measured-before),acceptance_threshold=threshold,emitted_S_gradient_l2=emitted,
            family_gradient_norms=gradients,proposed_family_delta_l2=deltas,credit_failures=failures,
            gradient_forward_backward_seconds=grad_seconds,total_proposal_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),start_transaction_sha256=initial)
        history.append(row)
        checkpoint.save_boundary(work,completed_updates=len(history),wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,history=history,bindings=prepared['bindings'],runtime=runtime,device=device)
        print(json.dumps(row),flush=True)
        if failures:break
    failures=[v for row in history for v in row['credit_failures']]
    learned=warm._base_encoder_sha(found)!=prepared['bindings']['factory_proof']['source_CNN_sha256']
    accepted=sum(r['accepted'] for r in history)
    status=('complete_rejected_credit' if failures else
        'complete_rejected_no_committed_learning' if not accepted or not learned else
        'complete_native64_training_admission')
    report=dict(status=status,
        seed=SEED,bindings=prepared['bindings'],history=history,accepted_updates=sum(r['accepted'] for r in history),
        actual_optimizer_steps=sorted({int(state['step'].item()) for state in optimizer.state.values()}),
        final_model_state_sha256=native._state_sha(found.wrapped,found.encoder),source_CNN_changed=learned,
        validation_data_read=False,performance_improvement_claim=False,
        full_native64_mini_comparison_implemented=False,completed_unix=time.time())
    write_json(output,report)
    return report



def configure(seed):
    global SEED,ARCHIVE
    if type(seed) is not int or seed not in (1,2):raise ValueError('independent weak seed1/2 required')
    SEED=seed;ARCHIVE=ROOT_ARCHIVE/f'seed{seed}'


def prepare_cpu():
    prepared=prepare_inputs_cpu()
    return prepared['found'],None,prepared['rgb'],prepared['bindings']


def validate_report(report,bindings):
    if report['seed']!=SEED or report['bindings']!=bindings or report['validation_data_read'] is not False:raise ValueError('training seed/binding changed')
    history=report['history']
    if not 1<=len(history)<=2:raise ValueError('invalid proposal count')
    for i,row in enumerate(history):
        if row['proposal']!=i+1 or row['indices']!=list(range(4*i,4*i+4)):raise ValueError('TRAIN order changed')
        accepted,threshold=incumbent._acceptance(row['before_loss'],row['after_loss'])
        if row['accepted']!=accepted or row['acceptance_threshold']!=threshold or row['parity_error']>=threshold:raise ValueError('descent/parity guard invalid')
    accepted=sum(row['accepted'] for row in history)
    if report['accepted_updates']!=accepted or report['actual_optimizer_steps']!=([accepted] if accepted else []):raise ValueError('actual Adam steps disagree')
    return report


def predict(expected_binding=None):
    prepared=prepare_inputs_cpu()
    if prepared['bindings']!=expected_binding:raise ValueError('CPU admission changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    report_path=ARCHIVE/'training/report.json';work=ARCHIVE/'training_work'
    if report_path.exists():
        model64.move_to_device(prepared['found'],device)
        runtime=resume_policy.compatible_runtime(checkpoint.runtime_signature(device))
        loaded=checkpoint.load_latest(work,wrapped=prepared['found'].wrapped,encoder=prepared['found'].encoder,
            optimizer=prepared['optimizer'],expected_bindings=prepared['bindings'],expected_runtime=runtime,device=device,max_updates=2)
        report=json.loads(report_path.read_bytes())
        if loaded['completed_updates']!=len(report['history']) or loaded['history']!=report['history']:raise ValueError('completed boundary/history changed')
    else:report=run_probe(prepared,report_path,work,device=device)
    validate_report(report,prepared['bindings'])
    actual_steps=sorted({int(s['step'].item()) for s in prepared['optimizer'].state.values()})
    if actual_steps!=report['actual_optimizer_steps']:raise ValueError('actual restored Adam/report mismatch')
    if native._state_sha(prepared['found'].wrapped,prepared['found'].encoder)!=report['final_model_state_sha256']:raise ValueError('actual trained model/report mismatch')
    from .evaluate import predict_candidate
    predict_candidate(prepared,report,device)


def validated_pair():
    from .evaluate import validated_candidate
    return validated_candidate()


def score():
    from .evaluate import score_candidate
    return score_candidate()
