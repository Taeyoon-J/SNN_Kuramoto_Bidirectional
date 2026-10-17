"""Continue SW0192's real native64 model/Adam/RNG for14 additional proposals."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from collaborative_test.SW_0192_native64_incumbent_mini import run as probe
from collaborative_test.SW_0193_native64_probe_screen import run as screen
from collaborative_test.SW_0193_native64_probe_screen import score as old_score

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
TOTAL=16
BATCH=4
REFERENCE_SHA='6e8e036718c520eec04226fc75e6356644de73bcea74e338bb7c174be9cbd4eb'
write_json=probe.write_json
sha=probe.sha


def content_sha(value):
    digest=hashlib.sha256()
    def visit(prefix,item):
        digest.update(prefix.encode())
        if isinstance(item,torch.Tensor):
            digest.update((str(item.dtype)+str(tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):visit(prefix+'/'+str(key),item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode())
            for i,part in enumerate(item):visit(prefix+'/'+str(i),part)
        else:digest.update(repr(item).encode())
    visit('root',value)
    return digest.hexdigest()


def fingerprint():
    return dict(experiment='SW0194_native64_train64',
        files={name:sha((HERE/name).read_bytes()) for name in ('DESIGN_KO.md','train.py','evaluate.py','queue.py','test_train.py','test_evaluate.py')},
        parent=probe.fingerprint(),readout_screen=screen.fingerprint(),
        source_seed=0,total_proposals=TOTAL,inherited_proposals=2,total_unique_TRAIN=64,batch=4,micro=1,
        objective='unchanged product QCE temperature0.1',mask_supervised=True)


def validate_history(history,parent_history):
    if not 2<=len(history)<=TOTAL or history[:2]!=parent_history:raise ValueError('inherited native64 prefix changed')
    for i,row in enumerate(history):
        if row['proposal']!=i+1 or row['indices']!=list(range(i*BATCH,(i+1)*BATCH)):raise ValueError('TRAIN64 order changed')
        if not math.isfinite(row['parity_error']) or row['parity_error']>=row['acceptance_threshold']:raise ValueError('loss parity proof invalid')


def prepare_inputs_cpu():
    if torch.cuda.is_initialized():raise RuntimeError('CPU admission before CUDA required')
    prepared=probe.prepare_inputs_cpu()
    report_path=probe.ARCHIVE/'probe_seed0/report.json';report=json.loads(report_path.read_bytes())
    screen.validate_parent_report(report,prepared['bindings'])
    source=probe.ARCHIVE/'probe_work_seed0';raw=(source/'boundary.ptx').read_bytes()
    metadata=probe.checkpoint.inspect_committed_metadata(source)
    envelope=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    data=envelope['payload_bytes']
    if sha(data)!=envelope['payload_sha256']:raise ValueError('parent checkpoint payload corrupt')
    payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    if metadata['completed_updates']!=2 or payload['bindings']!=prepared['bindings'] or payload['history']!=report['history']:raise ValueError('parent checkpoint lineage mismatch')
    if payload['optimizer_schema']!=probe.checkpoint.optimizer_schema(prepared['optimizer']):raise ValueError('parent Adam group schema changed')
    found=prepared['found'];found.wrapped.load_state_dict(payload['wrapped_state'],strict=True)
    found.encoder.load_state_dict(payload['encoder_state'],strict=True)
    prepared['optimizer'].load_state_dict(payload['optimizer_state'])
    if probe.native._state_sha(found.wrapped,found.encoder)!=report['final_model_state_sha256']:raise ValueError('parent learned weights not restored on CPU')
    if {int(s['step'].item()) for s in prepared['optimizer'].state.values()}!={report['accepted_updates']}:raise ValueError('parent actual Adam moments/steps not restored')
    references,ref_binding=old_score.validated_pair()
    prior=screen.ARCHIVE/'score_v1/summary.json'
    if sha(prior.read_bytes())!=REFERENCE_SHA:raise ValueError('fixed prior mini reference changed')
    if ref_binding['candidate_model_sha256']!=report['final_model_state_sha256']:raise ValueError('inherited mini model differs from actual parent')
    inherited=prepared['bindings']
    prepared.update(parent_path=source,parent_bindings=inherited,parent_runtime=payload['runtime'],
        parent_history=report['history'],parent_model_sha256=report['final_model_state_sha256'],
        parent_optimizer_content_sha256=content_sha(payload['optimizer_state']),
        parent_RNG_content_sha256=content_sha(payload['rng']),
        reference_binding=ref_binding,references=references)
    prepared['bindings']=dict(experiment='SW0194_native64_train64',fingerprint=fingerprint(),
        parent_boundary_sha256=sha(raw),parent_report_sha256=sha(report_path.read_bytes()),
        parent_model_sha256=report['final_model_state_sha256'],parent_bindings=inherited,
        prior_screen_summary_sha256=REFERENCE_SHA,
        reference_protocol_sha256={arm:sha((screen.ARCHIVE/arm/'protocol.json').read_bytes()) for arm in screen.ARMS},
        ordered_TRAIN64_ids=inherited['ordered_train_ids'],TRAIN64_modal_sha256=inherited['TRAIN64_modal_sha256'],
        validation_GT_read_for_training=False,optimizer_policy='continue exact source Adam and RNG; no reset')
    validate_history(prepared['parent_history'],prepared['parent_history'])
    return prepared


def restore_for_training(prepared,work,device,runtime):
    optimizer=prepared['optimizer'];found=prepared['found'];work=Path(work)
    if (work/'boundary.ptx').is_file():
        loaded=probe.checkpoint.load_latest(work,wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,expected_bindings=prepared['bindings'],expected_runtime=runtime,device=device,max_updates=TOTAL)
    else:
        if work.exists() and any(work.iterdir()) and not probe.checkpoint.is_only_uncommitted_temp(work):raise FileExistsError('preserve unexpected partial continuation')
        if sha((prepared['parent_path']/'boundary.ptx').read_bytes())!=prepared['bindings']['parent_boundary_sha256']:raise ValueError('parent boundary changed before GPU restore')
        if prepared['parent_runtime']!=runtime:raise ValueError('physical GPU lacks exact compatible runtime')
        loaded=probe.checkpoint.load_latest(prepared['parent_path'],wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,expected_bindings=prepared['parent_bindings'],expected_runtime=runtime,device=device,max_updates=2)
        if loaded['completed_updates']!=2 or probe.native._state_sha(found.wrapped,found.encoder)!=prepared['parent_model_sha256']:raise ValueError('GPU continuation did not inherit actual learned weights')
        optimizer_sha=content_sha(optimizer.state_dict())
        rng_sha=content_sha(probe.checkpoint.capture_rng(device))
        if optimizer_sha!=prepared['parent_optimizer_content_sha256'] or rng_sha!=prepared['parent_RNG_content_sha256']:raise ValueError('actual inherited Adam moments or RNG content differs')
        probe.checkpoint.save_boundary(work,completed_updates=2,wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,history=loaded['history'],bindings=prepared['bindings'],runtime=runtime,device=device)
        write_json(work/'inherited_start_receipt.json',dict(status='actual_native64_model_Adam_moments_and_CPU_CUDA_RNG_content_restored',
            parent_boundary_sha256=prepared['bindings']['parent_boundary_sha256'],
            parent_model_sha256=prepared['parent_model_sha256'],optimizer_content_sha256=optimizer_sha,
            RNG_content_sha256=rng_sha,loaded_proposals=2,training_repeated=False))
    history=loaded['history'];validate_history(history,prepared['parent_history'])
    if loaded['completed_updates']!=len(history):raise ValueError('boundary cursor differs from history')
    accepted=sum(row['accepted'] for row in history)
    if {int(row['step'].item()) for row in optimizer.state.values()}!={accepted}:raise ValueError('restored Adam steps differ from inherited history')
    return history


def run_training(prepared,device='cuda:0',work=None,output=None):
    device=torch.device(device);torch.cuda.set_device(device)
    found=probe.model64.move_to_device(prepared['found'],device)
    optimizer=prepared['optimizer'];families=prepared['families']
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for entries in families.values() for _,p in entries}:raise ValueError('optimizer/model parameter references changed')
    runtime=probe.resume_policy.compatible_runtime(probe.checkpoint.runtime_signature(device))
    work=ARCHIVE/'training_work' if work is None else Path(work)
    output=ARCHIVE/'training/report.json' if output is None else Path(output)
    if output.exists():
        report=json.loads(output.read_bytes())
        if report['bindings']!=prepared['bindings'] or report['status']!='complete_native64_TRAIN64':raise ValueError('existing training report binding changed')
        restored=restore_for_training(prepared,work,device,runtime)
        if len(restored)!=TOTAL or restored!=report['history']:raise ValueError('existing report does not contain all completed proposals')
        if probe.native._state_sha(found.wrapped,found.encoder)!=report['final_model_state_sha256']:raise ValueError('completed report/checkpoint mismatch')
        return report
    history=restore_for_training(prepared,work,device,runtime)
    for index in range(len(history),TOTAL):
        indices=list(range(index*BATCH,(index+1)*BATCH))
        torch.manual_seed(192000+index);torch.cuda.manual_seed(192000+index)
        optimizer.zero_grad(set_to_none=True)
        modules={'wrapped':found.wrapped,'encoder':found.encoder}
        snapshot=probe.tx.capture_transaction(modules,optimizer,offload_to_cpu=True)
        initial=probe.tx.transaction_digest(snapshot)
        torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
        before,_=probe.batch_loss(prepared,indices,device,live=False)
        probe.tx.restore_rng_state(snapshot['rng'])
        grad_started=time.perf_counter()
        measured,_=probe.batch_loss(prepared,indices,device,live=True,credit=False)
        grad_seconds=time.perf_counter()-grad_started
        if abs(measured-before)>=max(1e-7,1e-6*abs(before)):raise RuntimeError('same-RNG full-forward loss parity failed')
        rng_after=probe.tx.capture_rng_state()
        gradients={name:math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu()) for _,p in entries if p.grad is not None)) for name,entries in families.items()}
        if any(not math.isfinite(v) for v in gradients.values()):raise FloatingPointError('nonfinite family gradient')
        clipped=torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],1.)
        if not torch.isfinite(clipped):raise FloatingPointError('nonfinite clipped gradient')
        old={name:[p.detach().clone() for _,p in entries] for name,entries in families.items()}
        optimizer.step();found.wrapped.project_integrations_()
        deltas={name:math.sqrt(sum(float((p.detach().double()-prior.double()).square().sum().cpu()) for (_,p),prior in zip(entries,old[name]))) for name,entries in families.items()}
        del old
        if any(not math.isfinite(v) for v in deltas.values()):raise FloatingPointError('nonfinite proposed delta')
        probe.tx.restore_rng_state(snapshot['rng'])
        after,_=probe.batch_loss(prepared,indices,device,live=False)
        accepted,threshold=probe.incumbent._acceptance(before,after)
        if accepted:
            optimizer.zero_grad(set_to_none=True);probe.tx.restore_rng_state(rng_after)
        else:probe.tx.restore_transaction(snapshot,modules,optimizer,restore_rng=True,clear_grad=True)
        torch.cuda.synchronize(device)
        history.append(dict(proposal=index+1,indices=indices,accepted=accepted,before_loss=before,after_loss=after,
            parity_error=abs(measured-before),acceptance_threshold=threshold,emitted_S_gradient_l2=None,
            emitted_credit_scope='source first-two actual proof preserved; not remeasured in later proposals',
            family_gradient_norms=gradients,proposed_family_delta_l2=deltas,credit_failures=[],
            gradient_forward_backward_seconds=grad_seconds,total_proposal_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),start_transaction_sha256=initial))
        validate_history(history,prepared['parent_history'])
        probe.checkpoint.save_boundary(work,completed_updates=len(history),wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,history=history,bindings=prepared['bindings'],runtime=runtime,device=device)
        print(json.dumps(dict(proposal=index+1,accepted=accepted)),flush=True)
    accepted=sum(row['accepted'] for row in history)
    steps=sorted({int(row['step'].item()) for row in optimizer.state.values()})
    if steps!=[accepted]:raise ValueError('actual Adam steps differ from committed proposal count')
    report=dict(status='complete_native64_TRAIN64',bindings=prepared['bindings'],history=history,
        inherited_proposals=2,new_proposals=TOTAL-2,total_unique_TRAIN=64,total_exposures=64,
        accepted_updates=accepted,actual_optimizer_steps=steps,
        final_model_state_sha256=probe.native._state_sha(found.wrapped,found.encoder),
        final_CNN_sha256=probe.warm._base_encoder_sha(found),validation_GT_read=False,
        performance_claim=False,completed_unix=time.time())
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-output',type=Path);args=parser.parse_args()
    if not args.cpu_output:parser.error('CPU output required')
    prepared=prepare_inputs_cpu()
    write_json(args.cpu_output,dict(status='SW0194_exact_native64_weights_and_Adam_CPU_admitted',
        bindings=prepared['bindings'],parent_runtime=prepared['parent_runtime'],
        inherited_actual_Adam_steps=sorted({int(v['step'].item()) for v in prepared['optimizer'].state.values()}),
        cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':main()
