"""Native64 production-energy ranking with exact inherited Adam/RNG transactions."""
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
from collaborative_test.SW_0194_native64_train64 import train as parent
from collaborative_test.SW_0204_native64_component_credit.component_loss import component_objective
from .energy_loss import make_context,ranking_loss,context_record

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
LIVE_TAIL=64
TOTAL=4
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
    return dict(experiment='SW0218_native64_energy_ranking',
        files={name:sha((HERE/name).read_bytes()) for name in
        ('DESIGN_KO.md','train.py','evaluate.py','queue.py','test_policy.py','test_training.py','energy_loss.py','test_energy_loss.py','test_context_pipeline.py')},
        connected_reference_library_sha256=sha(Path(make_context.__globals__['connected'].__code__.co_filename).read_bytes()),
        production_energy_library_sha256=sha(Path(make_context.__globals__['native'].__file__).read_bytes()),
        parent=parent.fingerprint(),component_library_sha256=sha(Path(component_objective.__code__.co_filename).read_bytes()),inherited_proposals=2,new_proposals=2,
        total_unique_TRAIN=16,batch=4,micro=1,objective='SW0204 component QCE + production energy ranking, unit fixed margin and coefficient',update_policy='finite Adam steps, no minibatch descent rejection',
        total_steps=1024,settle_steps=512,live_tail_steps=LIVE_TAIL,mask_supervised=True)


def finite_step_policy(before,after,parameters_finite=True):
    if not math.isfinite(before):raise ValueError('finite pre-step objective required')
    threshold=max(1e-7,1e-6*abs(before))
    finite=bool(parameters_finite and after is not None and math.isfinite(after))
    descent=bool(finite and after<before-threshold)
    return finite,descent,threshold


def validate_history(history,parent_history):
    if not 2<=len(history)<=TOTAL or history[:2]!=parent_history:raise ValueError('inherited native64 prefix changed')
    for i,row in enumerate(history):
        if row['proposal']!=i+1 or row['indices']!=list(range(i*BATCH,(i+1)*BATCH)):raise ValueError('TRAIN64 order changed')
        if not math.isfinite(row['parity_error']) or row['parity_error']>=row['acceptance_threshold']:raise ValueError('loss parity proof invalid')


def prepare_inputs_cpu():
    prepared=parent.prepare_inputs_cpu()
    prior_binding=prepared['bindings']
    prepared['bindings']=dict(prior_binding,experiment='SW0218_native64_finite_Adam',
        fingerprint=fingerprint(),live_tail_steps=LIVE_TAIL,total_unique_TRAIN=16,
        parent_optimizer_content_sha256=prepared['parent_optimizer_content_sha256'],
        parent_RNG_content_sha256=prepared['parent_RNG_content_sha256'])
    prepared['warm_CNN_sha256']=probe.warm._base_encoder_sha(prepared['found'])
    from collaborative_test.SW_0204_native64_component_credit import train as control,evaluate as comparison
    labels,protocol=comparison.validated_candidate()
    if protocol['reference_binding']!=prepared['reference_binding']:raise ValueError('matched control cohort differs')
    prepared['bindings'].update(control_protocol_sha256=sha((control.ARCHIVE/'candidate/protocol.json').read_bytes()),
        control_report_sha256=sha((control.ARCHIVE/'training/report.json').read_bytes()),
        control_model_sha256=protocol['model_sha256'],control_masks_sha256=sha(labels.tobytes()))
    return prepared


def combined_objective(prepared,index,trace,adj,q,rho,*,live):
    found=prepared['found'];contexts=prepared['_ranking_contexts']
    labels=torch.from_numpy(prepared['labels'][index].reshape(-1)).to(q.device)
    gain=found.wrapped.core.graph_generator.log_coupling_gain.exp()
    if index not in contexts:
        if live:raise ValueError('ranking context must be mined in the before-step no-grad pass')
        predicted=screen.prediction_from_trace(found,trace,adj,q)
        contexts[index]=make_context(predicted,prepared['labels'][index],q[0],adj[0],gain)
    auxiliary=ranking_loss(q[0],adj[0],gain,contexts[index])
    loss=component_objective(q[0],rho[0],labels,temperature=.1)+auxiliary
    return loss,auxiliary


def batch_loss(prepared,indices,device,*,live,credit=False):
    found=prepared['found'];losses=[];emitted_sq=0.;contexts=prepared['_ranking_contexts']
    measured_auxiliary_parameters=False
    for index in indices:
        rgb=torch.from_numpy(np.ascontiguousarray(prepared['rgb'][index:index+1])).to(device)
        with torch.set_grad_enabled(live):
            gamma,trace,adj,z,q,rho=probe.native._native_forward(found,rgb,live_tail_steps=LIVE_TAIL if live else 0)
            if gamma.shape!=(1,8,4096) or q.shape!=(1,4096,4096):raise ValueError('actual native64 geometry')
            loss,auxiliary=combined_objective(prepared,index,trace,adj,q,rho,live=live)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite energy ranking objective')
            losses.append(float(loss.detach().cpu()))
            if live:
                if credit:
                    gradient=torch.autograd.grad(loss,trace['component_spikes'],retain_graph=True)[0]
                    if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite emitted-S gradient')
                    emitted_sq+=float(gradient.detach().double().square().sum().cpu());del gradient
                    auxiliary_gradient=torch.autograd.grad(auxiliary,trace['component_spikes'],retain_graph=True,allow_unused=True)[0]
                    auxiliary_sq=0. if auxiliary_gradient is None else float(auxiliary_gradient.detach().double().square().sum().cpu())
                    if not math.isfinite(auxiliary_sq):raise FloatingPointError('nonfinite ranking spike credit')
                    del auxiliary_gradient
                    record=dict(loss=float(auxiliary.detach().cpu()),emitted_S_gradient_l2=math.sqrt(auxiliary_sq),family_parameter_gradient_l2=None)
                    if auxiliary_sq>0 and not measured_auxiliary_parameters:
                        entries=[(family,p) for family,values in prepared['families'].items() for _,p in values]
                        gradients=torch.autograd.grad(auxiliary,[p for _,p in entries],retain_graph=True,allow_unused=True)
                        squares={family:0. for family in prepared['families']}
                        for (family,_),gradient in zip(entries,gradients):
                            if gradient is not None:
                                if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite ranking parameter credit')
                                squares[family]+=float(gradient.detach().double().square().sum().cpu())
                        record['family_parameter_gradient_l2']={family:math.sqrt(value) for family,value in squares.items()}
                        measured_auxiliary_parameters=True;del gradients,entries
                    prepared['_ranking_credit'][index]=record
                (loss/len(indices)).backward()
        del gamma,trace,adj,z,q,rho,rgb,loss,auxiliary
    return float(np.mean(losses)),math.sqrt(emitted_sq)


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
        if report['bindings']!=prepared['bindings'] or report['status']!='complete_native64_finite_Adam':raise ValueError('existing training report binding changed')
        restored=restore_for_training(prepared,work,device,runtime)
        if len(restored)!=TOTAL or restored!=report['history']:raise ValueError('existing report does not contain all completed proposals')
        if probe.native._state_sha(found.wrapped,found.encoder)!=report['final_model_state_sha256']:raise ValueError('completed report/checkpoint mismatch')
        return report
    history=restore_for_training(prepared,work,device,runtime)
    for index in range(len(history),TOTAL):
        indices=list(range(index*BATCH,(index+1)*BATCH))
        prepared['_ranking_contexts']={};prepared['_ranking_credit']={}
        torch.manual_seed(192000+index);torch.cuda.manual_seed(192000+index)
        optimizer.zero_grad(set_to_none=True)
        modules={'wrapped':found.wrapped,'encoder':found.encoder}
        snapshot=probe.tx.capture_transaction(modules,optimizer,offload_to_cpu=True)
        initial=probe.tx.transaction_digest(snapshot)
        torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
        before,_=batch_loss(prepared,indices,device,live=False)
        probe.tx.restore_rng_state(snapshot['rng'])
        grad_started=time.perf_counter()
        measured,emitted=batch_loss(prepared,indices,device,live=True,credit=True)
        grad_seconds=time.perf_counter()-grad_started
        if abs(measured-before)>=max(1e-7,1e-6*abs(before)):raise RuntimeError('same-RNG full-forward loss parity failed')
        rng_after=probe.tx.capture_rng_state()
        gradients={name:math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu()) for _,p in entries if p.grad is not None)) for name,entries in families.items()}
        if any(not math.isfinite(v) for v in gradients.values()):raise FloatingPointError('nonfinite family gradient')
        if not math.isfinite(emitted) or emitted<=0 or any(v<=0 for v in gradients.values()):raise RuntimeError('new objective lacks actual spike/family gradient')
        clipped=torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],1.)
        if not torch.isfinite(clipped):raise FloatingPointError('nonfinite clipped gradient')
        old={name:[p.detach().clone() for _,p in entries] for name,entries in families.items()}
        optimizer.step();found.wrapped.project_integrations_()
        deltas={name:math.sqrt(sum(float((p.detach().double()-prior.double()).square().sum().cpu()) for (_,p),prior in zip(entries,old[name]))) for name,entries in families.items()}
        del old
        parameters_finite=all(math.isfinite(v) for v in deltas.values())
        probe.tx.restore_rng_state(snapshot['rng'])
        after=None
        if parameters_finite:
            try:after,_=batch_loss(prepared,indices,device,live=False)
            except FloatingPointError:after=None
        accepted,descent_accepted,threshold=finite_step_policy(before,after,parameters_finite)
        if after is not None and not math.isfinite(after):after=None
        recorded_deltas={name:value if math.isfinite(value) else None for name,value in deltas.items()}
        if accepted:
            optimizer.zero_grad(set_to_none=True);probe.tx.restore_rng_state(rng_after)
        else:probe.tx.restore_transaction(snapshot,modules,optimizer,restore_rng=True,clear_grad=True)
        torch.cuda.synchronize(device)
        history.append(dict(proposal=index+1,indices=indices,accepted=accepted,finite_step_applied=accepted,
            ranking_contexts={str(k):context_record(v) for k,v in prepared['_ranking_contexts'].items()},
            ranking_credit={str(k):v for k,v in prepared['_ranking_credit'].items()},
            descent_accepted=descent_accepted,update_policy='finite Adam; uphill finite steps retained',
            before_loss=before,after_loss=after,
            parity_error=abs(measured-before),acceptance_threshold=threshold,emitted_S_gradient_l2=emitted,
            emitted_credit_scope='actual full native64 new proposal loss to emitted spikes',
            family_gradient_norms=gradients,proposed_family_delta_l2=recorded_deltas,
            parameters_finite_after_proposal=parameters_finite,credit_failures=[],
            gradient_forward_backward_seconds=grad_seconds,total_proposal_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),start_transaction_sha256=initial))
        validate_history(history,prepared['parent_history'])
        probe.checkpoint.save_boundary(work,completed_updates=len(history),wrapped=found.wrapped,encoder=found.encoder,
            optimizer=optimizer,history=history,bindings=prepared['bindings'],runtime=runtime,device=device)
        print(json.dumps(dict(proposal=index+1,accepted=accepted)),flush=True)
    accepted=sum(row['accepted'] for row in history)
    steps=sorted({int(row['step'].item()) for row in optimizer.state.values()})
    if steps!=[accepted]:raise ValueError('actual Adam steps differ from committed proposal count')
    report=dict(status='complete_native64_finite_Adam',bindings=prepared['bindings'],history=history,
        new_update_policy='finite Adam; accepted means applied finite step, not descent',
        new_descent_steps=sum(r['descent_accepted'] for r in history[2:]),
        inherited_proposals=2,new_proposals=TOTAL-2,total_unique_TRAIN=16,total_exposures=16,
        accepted_updates=accepted,actual_optimizer_steps=steps,
        final_model_state_sha256=probe.native._state_sha(found.wrapped,found.encoder),
        final_CNN_sha256=probe.warm._base_encoder_sha(found),validation_GT_read=False,
        warm_CNN_sha256=prepared['warm_CNN_sha256'],
        new_accepted_updates=sum(r['accepted'] for r in history[2:]),
        committed_model_changed=probe.native._state_sha(found.wrapped,found.encoder)!=prepared['parent_model_sha256'],
        CNN_changed_from_warm=probe.warm._base_encoder_sha(found)!=prepared['warm_CNN_sha256'],
        ranking_parameter_credit_verified=any(r['accepted'] and any(v['emitted_S_gradient_l2']>0 and v['family_parameter_gradient_l2'] is not None and any(g>0 for g in v['family_parameter_gradient_l2'].values()) for v in r.get('ranking_credit',{}).values()) for r in history[2:]),
        ranking_context_frozen_across_before_gradient_after=True,
        performance_claim=False,completed_unix=time.time())
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-output',type=Path);args=parser.parse_args()
    if not args.cpu_output:parser.error('CPU output required')
    prepared=prepare_inputs_cpu()
    write_json(args.cpu_output,dict(status='SW0218_exact_native64_weights_and_Adam_CPU_admitted',
        bindings=prepared['bindings'],parent_runtime=prepared['parent_runtime'],
        inherited_actual_Adam_steps=sorted({int(v['step'].item()) for v in prepared['optimizer'].state.values()}),
        cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':main()
