"""Four-image seed1 screen of a single downstream gate change, no training."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0237_native64_converged_full320 import run as reference
from collaborative_test.SW_0237_native64_converged_full320 import evaluate as reference_score
from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
from collaborative_test.SW_0171_contextual_instance_graph.instance_objective import capture_native_projection
from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import native_component_rho_q
from collaborative_test.SW_0238_native64_spike_event_activity.spike_gate_activity import summarize
from .rollout import late_rollout

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive'
SEEDS=(1,);OFFSETS=(0,2,4,6);sha=reference.sha;write_json=reference.write_json


def require_seed(seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('registered seed1 viability screen only')
    return seed


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    names=('DESIGN_KO.md','component_gate.py','rollout.py','run.py','queue.py','test_component_gate.py','test_scope.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},parent237=reference.fingerprint(),
        native_rollout_source_sha256=sha((ROOT/'collaborative_test/SW_0134_native_spike_binding/rollout.py').read_bytes()),
        seeds=list(SEEDS),offsets=list(OFFSETS),grid=[64,64],steps=1024,settle=512,optimizer_updates=0,
        expected_candidate_forwards=4,expected_original_replay_forwards=1,not_a_full320_or_three_seed_claim=True)


def prepare_cpu(seed):
    require_seed(seed);p=reference.prepare_cpu(seed);found=p['found'];core=found.wrapped.core
    if core.gate_mode!='raw' or found.wrapped.arm!='phase' or core.kuramoto.spike_pulse_gain is not None or core.graph_generator.uses_feedback:
        raise ValueError('fixed raw-gate phase source without global spike/graph feedback required')
    control=reference.validated_seed(seed)
    if not np.array_equal(p['mini'][list(OFFSETS)],control['labels'][list(OFFSETS)]):raise ValueError('matched237 control masks changed')
    score=json.loads((reference.ARCHIVE/'score_v1/summary.json').read_bytes())
    audit_raw=(reference.ARCHIVE/'completion_audit_v1/summary.json').read_bytes();audit=json.loads(audit_raw)
    if audit['score_sha256']!=sha((reference.ARCHIVE/'score_v1/summary.json').read_bytes()) or score['fingerprint']!=reference.fingerprint() or audit['raw_masks_independently_rescored']!=2880 or audit['independent_metrics_means_and_bootstrap_maximum_abs_error']>=1e-12:
        raise ValueError('independently verified237 reference required')
    old=dict(p['binding']);p['parent_binding']=old
    p['binding']=dict(fingerprint=fingerprint(),seed=seed,parent_binding=old,parent_score_sha256=audit['score_sha256'],
        parent_audit_sha256=sha(audit_raw),control_labels_sha256=sha(control['labels'][list(OFFSETS)].tobytes()),
        optimizer_updates=0,DEV_GT_read=False,mask_supervised_weight_lineage=True)
    p['control']=control['labels'][list(OFFSETS)].copy();return p


def admit(seed):
    p=prepare_cpu(seed);a=archive_for(seed)
    reference.helpers.save_arrays(a/'control.npz',labels=p['control'])
    result=dict(binding=p['binding'],cuda_initialized=False,optimizer_updates=0,actual_CPU_restore_passed=True,
        control_file_sha256=sha((a/'control.npz').read_bytes()))
    write_json(a/'admission.json',result);return result


def variant_forward(found,image):
    gamma=found.encode(image);graph=found.wrapped.core.graph_generator
    with capture_native_projection(graph) as projection:
        trace,adj=capture_source_coupling(found.wrapped.core.kuramoto,lambda:late_rollout(found.wrapped,gamma,total_steps=1024,live_tail_steps=0))
    if len(projection)!=1:raise ValueError('one fixed native graph projection required')
    _,q=native_component_rho_q(trace['spikes'],trace['component_spikes'],settle=512)
    for key in ('theta','component_spikes','component_gates','component_membrane'):
        if not bool(torch.isfinite(trace[key]).all()):raise FloatingPointError('nonfinite component gate rollout')
    return trace,adj,q


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed)
    if json.loads((a/'admission.json').read_bytes())['binding']!=p['binding']:raise ValueError('registered CPU source differs')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    found=p['found'];reference.source.source.source.model64.move_to_device(found,device)
    runtime=reference.scope.model.resume_policy.compatible_runtime(reference.scope.model.checkpoint.runtime_signature(device))
    if runtime!=p['parent_binding']['candidate']['expected_runtime']:raise ValueError('accepted native numerical runtime differs')
    rows=[]
    for index,offset in enumerate(OFFSETS):
        folder=a/'rows'/f'{index:04d}'
        if not folder.exists():
            image=torch.from_numpy(p['rgb'][offset:offset+1]).to(device)
            old_theta=old_adj=None
            with torch.no_grad():
                if index==0:
                    torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
                    gamma,old,adj,z,q,rho=reference.source.source.full.screen.native._native_forward(found,image,live_tail_steps=0)
                    mask=np.asarray(reference.source.source.full.screen.prediction_from_trace(found,old,adj,q),dtype=np.int64)
                    if not np.array_equal(mask,p['control'][index]):raise ValueError('original237 replay mask changed')
                    old_theta=old['theta'].clone();old_adj=adj.clone();del gamma,old,adj,z,q,rho,mask
                torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
                trace,adj,q=variant_forward(found,image)
                if index==0 and (not torch.equal(trace['theta'],old_theta) or not torch.equal(adj,old_adj)):
                    raise ValueError('gate change altered upstream theta or graph')
                labels=np.asarray(reference.source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64)
                g=trace['component_gates'][...,512:].cpu().numpy();s=trace['component_spikes'][...,512:].cpu().numpy()
                activity=summarize(s,g);diversity=float(np.max(np.abs(g-g[:,:1])))
                mem=trace['component_membrane'][...,512:]
                summary=dict(gate_mean=trace['component_gates'][...,512:].mean(-1)[0].cpu().numpy(),
                    spike_mean=trace['component_spikes'][...,512:].mean(-1)[0].cpu().numpy(),
                    membrane_mean=mem.mean(-1)[0].cpu().numpy(),membrane_std=mem.std(-1,unbiased=False)[0].cpu().numpy())
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            reference.helpers.save_arrays(stage/'prediction.npz',labels=labels,**summary)
            write_json(stage/'row.json',dict(binding=p['binding'],index=index,offset=offset,inference_seed=193000+offset,
                data_sha256=sha((stage/'prediction.npz').read_bytes()),activity=activity,maximum_component_gate_difference=diversity,
                original_mask_replay_and_upstream_theta_graph_bitwise_equal=(True if index==0 else None),optimizer_updates=0,DEV_GT_read=False))
            stage.rename(folder);del trace,adj,q,image,labels,g,s,mem,summary,old_theta,old_adj
        read_row(folder,p['binding'],index);rows.append(sha((folder/'row.json').read_bytes()))
    if reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['parent_binding']['candidate']['model_sha256']:
        raise ValueError('screen changed frozen model parameters')
    write_json(a/'protocol.json',dict(binding=p['binding'],row_sha256=rows,offsets=list(OFFSETS),optimizer_updates=0,DEV_GT_read=False))


def read_row(folder,binding,index):
    row=json.loads((folder/'row.json').read_bytes())
    if row['binding']!=binding or row['index']!=index or row['offset']!=OFFSETS[index] or row['inference_seed']!=193000+OFFSETS[index] or row['optimizer_updates'] or row['DEV_GT_read']:
        raise ValueError('frozen screen row changed')
    if sha((folder/'prediction.npz').read_bytes())!=row['data_sha256']:raise ValueError('prediction arrays changed')
    with np.load(folder/'prediction.npz',allow_pickle=False) as data:labels=data['labels'].copy()
    if labels.shape!=(64,64) or labels.dtype!=np.int64 or (labels<0).any():raise ValueError('native64 mask required')
    if index==0 and row['original_mask_replay_and_upstream_theta_graph_bitwise_equal'] is not True:raise ValueError('first-image upstream control missing')
    return labels


def validated_seed(seed):
    a=archive_for(seed);ad=json.loads((a/'admission.json').read_bytes());protocol=json.loads((a/'protocol.json').read_bytes())
    if protocol['binding']!=ad['binding'] or ad['binding']['fingerprint']!=fingerprint() or protocol['offsets']!=list(OFFSETS) or len(protocol['row_sha256'])!=4 or protocol['optimizer_updates'] or protocol['DEV_GT_read'] or ad['cuda_initialized']:
        raise ValueError('registered four-image screen changed')
    if sha((a/'control.npz').read_bytes())!=ad['control_file_sha256']:raise ValueError('matched control changed')
    values=[]
    for i,digest in enumerate(protocol['row_sha256']):
        folder=a/'rows'/f'{i:04d}'
        if sha((folder/'row.json').read_bytes())!=digest:raise ValueError('screen row digest changed')
        values.append(read_row(folder,ad['binding'],i))
    with np.load(a/'control.npz',allow_pickle=False) as old:control=old['labels'].copy()
    return dict(labels=np.stack(values),control=control,binding=ad['binding'])


def score():
    bundle=validated_seed(1);slots=reference_score.load_slot_masks()
    gt=reference.source.source.full.read_gt(dict(dataset=bundle['binding']['parent_binding']['dataset']))[list(OFFSETS)]
    metric=reference.source.source.full.metrics.independent;keys=metric.METRICS
    vectors={name:np.stack([metric.image_metrics(p,t) for p,t in zip(labels,gt)]) for name,labels in
        (('component_gate',bundle['labels']),('control237',bundle['control']),('slot_seed1',slots[1]['labels64'][list(OFFSETS)]))}
    means={name:dict(zip(keys,value.mean(0).tolist())) for name,value in vectors.items()};delta=vectors['component_gate'].mean(0)-vectors['control237'].mean(0)
    positive=bool(delta[0]>=.01 and np.all(delta[1:]>=0))
    result=dict(status='complete_component_gate_seed1_four_image_viability_screen',fingerprint=fingerprint(),
        scores=means,delta_from_matched_control=dict(zip(keys,delta.tolist())),promising_screen_for_separate_three_seed_confirmation=positive,
        seed_scope=[1],image_count=4,optimizer_updates=0,full320_or_three_seed_improvement_proven=False,
        native64_FG090_goal_verified=False,mask_supervised_weight_lineage=True,unsupervised_leaderboard_eligible=False,
        parameter_update_or_new70000_training=False,current_full320_reference_unchanged=True)
    write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=result,vectors={k:v.tolist() for k,v in vectors.items()}))
    write_json(ARCHIVE/'score_v1/summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--admit',action='store_true');p.add_argument('--score',action='store_true');args=p.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    else:p.error('choose actual CPU admission or completed screen scoring')
