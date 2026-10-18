"""One bounded, read-only native64 optimizer-state diagnosis on TRAIN4."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0218_native64_energy_ranking import train as source
from collaborative_test.SW_0218_native64_energy_ranking import recover_score
from .optimizer_audit import compare_steps,decision

HERE=Path(__file__).resolve().parent;ARCHIVE=HERE/'results_archive';ROOT=HERE.parents[1]
INDICES=(8,9,10,11);sha=source.sha;write_json=source.write_json


def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('DESIGN_KO.md','run.py','optimizer_audit.py','queue.py','test_optimizer.py')},
        source218=source.fingerprint(),recovery_library_sha256=sha(Path(recover_score.__file__).read_bytes()),
        actual218_summary_sha256=sha((source.ARCHIVE/'score_v1/summary.json').read_bytes()),
        indices=list(INDICES),arms=['inherited_Adam','reset_Adam'],clip=1.,committed_updates=0,
        total_steps=1024,settle_steps=512,live_tail_steps=64)


def prepare_cpu():
    if torch.cuda.is_initialized():raise ValueError('CPU admission before CUDA required')
    prepared=source.prepare_inputs_cpu()
    # Explicit real control import and frozen-mask validation; no DEV GT read.
    recover_score.validated_bundle()
    previous=json.loads((source.ARCHIVE/'training/report.json').read_bytes())
    if previous['history'][2]['indices']!=list(INDICES) or not previous['ranking_parameter_credit_verified']:
        raise ValueError('registered first218 proposal/actual auxiliary credit missing')
    binding=dict(fingerprint=fingerprint(),parent_model_sha256=prepared['parent_model_sha256'],
        parent_optimizer_content_sha256=prepared['parent_optimizer_content_sha256'],
        parent_RNG_content_sha256=prepared['parent_RNG_content_sha256'],
        source218_training_report_sha256=sha((source.ARCHIVE/'training/report.json').read_bytes()),
        source_training_bindings=prepared['bindings'],DEV_GT_read=False)
    prepared['audit_binding']=binding;prepared['reference218_first_proposal']=previous['history'][2]
    return prepared


def admit():
    prepared=prepare_cpu();value=dict(status='actual_native64_model_Adam_RNG_and_frozen_control_CPU_admitted',
        binding=prepared['audit_binding'],CUDA_initialized=False)
    write_json(ARCHIVE/'cpu_v1/admission.json',value);return value


def run_audit(prepared,device='cuda:0'):
    started=time.perf_counter();device=torch.device(device);torch.cuda.set_device(device)
    source.probe.model64.move_to_device(prepared['found'],device);found=prepared['found'];optimizer=prepared['optimizer']
    runtime=source.probe.resume_policy.compatible_runtime(source.probe.checkpoint.runtime_signature(device))
    if runtime!=prepared['parent_runtime']:raise ValueError('incompatible actual source runtime')
    loaded=source.probe.checkpoint.load_latest(prepared['parent_path'],wrapped=found.wrapped,encoder=found.encoder,
        optimizer=optimizer,expected_bindings=prepared['parent_bindings'],expected_runtime=runtime,device=device,max_updates=2)
    if loaded['completed_updates']!=2 or source.probe.native._state_sha(found.wrapped,found.encoder)!=prepared['parent_model_sha256']:
        raise ValueError('actual source model cursor/SHA mismatch')
    if source.content_sha(optimizer.state_dict())!=prepared['parent_optimizer_content_sha256'] or source.content_sha(source.probe.checkpoint.capture_rng(device))!=prepared['parent_RNG_content_sha256']:
        raise ValueError('actual source Adam/RNG content mismatch')
    modules=dict(wrapped=found.wrapped,encoder=found.encoder)
    entry=source.probe.tx.capture_transaction(modules,optimizer,offload_to_cpu=True);entry_digest=source.probe.tx.transaction_digest(entry)
    restored=False
    try:
        torch.manual_seed(192002);torch.cuda.manual_seed(192002)
        optimizer.zero_grad(set_to_none=True);prepared['_ranking_contexts']={};prepared['_ranking_credit']={}
        proposal=source.probe.tx.capture_transaction(modules,optimizer,offload_to_cpu=True)
        before,_=source.batch_loss(prepared,INDICES,device,live=False)
        source.probe.tx.restore_rng_state(proposal['rng'])
        measured,emitted=source.batch_loss(prepared,INDICES,device,live=True,credit=True)
        if abs(before-measured)>=max(1e-7,1e-6*abs(before)):raise ValueError('same-RNG gradient/no-grad loss mismatch')
        if not math.isfinite(emitted) or emitted<=0:raise ValueError('actual emitted-spike surrogate credit missing')
        if source.probe.native._state_sha(found.wrapped,found.encoder)!=prepared['parent_model_sha256']:
            raise ValueError('read-only forward changed model state')
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite shared gradient')
        gradients={id(p):None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters}
        arms=compare_steps(found,optimizer,prepared['families'],gradients,proposal,
            lambda:source.batch_loss(prepared,INDICES,device,live=False)[0])
        context={str(k):source.context_record(v) for k,v in prepared['_ranking_contexts'].items()}
        old=prepared['reference218_first_proposal']
        matched_contexts=context==old['ranking_contexts']
        result=dict(status='complete_native64_TRAIN4_Adam_state_diagnosis',binding=prepared['audit_binding'],
            before_loss=before,gradient_forward_loss=measured,parity_error=abs(before-measured),
            emitted_S_surrogate_gradient_l2=emitted,preclip_gradient_norm=float(norm.cpu()),
            arms=arms,decision=decision(before,arms),frozen_contexts=context,
            source218_contexts_match=matched_contexts,
            source218_before_loss_difference=before-old['before_loss'],
            source218_inherited_after_loss_difference=None if arms['inherited_Adam']['after_loss'] is None else arms['inherited_Adam']['after_loss']-old['after_loss'],
            auxiliary_credit=prepared['_ranking_credit'],temporary_optimizer_proposals=2,
            committed_optimizer_updates=0,native_forward_images_completed=16,unique_TRAIN_images=4,
            gradient_images=4,total_steps=1024,settle_steps=512,live_tail_steps=64,
            DEV_GT_read=False,mask_prediction_evaluations=0,new70000_training=False,
            full320_three_seed_FG090_goal_verified=False)
    finally:
        source.probe.tx.restore_transaction(entry,modules,optimizer,restore_rng=True,clear_grad=True)
        now=source.probe.tx.capture_transaction(modules,optimizer,offload_to_cpu=True)
        restored=source.probe.tx.transaction_digest(now)==entry_digest
        if not restored:raise RuntimeError('read-only diagnosis failed exact model/Adam/RNG restoration')
    result.update(source_transaction_exactly_restored=restored,parent_transaction_sha256=entry_digest,
        restored_model_sha256=source.probe.native._state_sha(found.wrapped,found.encoder),
        restored_optimizer_content_sha256=source.content_sha(optimizer.state_dict()),
        restored_RNG_content_sha256=source.content_sha(source.probe.checkpoint.capture_rng(device)),seconds=time.perf_counter()-started)
    write_json(ARCHIVE/'report.json',result);return result


def validate_report():
    value=json.loads((ARCHIVE/'report.json').read_bytes())
    if value['binding']['fingerprint']!=fingerprint() or not value['source_transaction_exactly_restored'] or value['committed_optimizer_updates']!=0:
        raise ValueError('invalid read-only diagnosis binding/restoration')
    if value['restored_model_sha256']!=value['binding']['parent_model_sha256'] or value['restored_optimizer_content_sha256']!=value['binding']['parent_optimizer_content_sha256'] or value['restored_RNG_content_sha256']!=value['binding']['parent_RNG_content_sha256']:
        raise ValueError('source restoration content differs')
    if value['decision']!=decision(value['before_loss'],value['arms']):raise ValueError('diagnosis decision changed')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    if args.admit:print(json.dumps(admit()))
    else:parser.error('GPU worker must use the admitted exclusive queue')
