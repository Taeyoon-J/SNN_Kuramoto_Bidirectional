"""Readout-only control using existing241 arrays; original registered source stays frozen."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from . import run as original
from .readout import gate_only_trace,compare_labels
from .layout_ops import restore_native_spikes,matched_gate_only_trace
from collaborative_test.SW_0221_native64_frequency_pretraining.train import save_arrays

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=original.ARCHIVE/'layout_control_v1'
SEEDS=original.SEEDS;sha=original.sha;write_json=original.write_json;require_seed=original.require_seed


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    names=('LAYOUT_CONTROL.md','layout_control.py','layout_queue.py','layout_ops.py','test_layout_ops.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},original241=original.fingerprint(),
        native_rollout_storage_source_sha256=sha((ROOT/'collaborative_test/SW_0134_native_spike_binding/rollout.py').read_bytes()),
        additional_native_forwards=0,optimizer_updates=0,GT_used=False,seeds=list(SEEDS),cases_per_seed=2)


def prepare_cpu(seed):
    p=original.prepare_cpu(seed);parents=original.validated_seed(seed);old=dict(p['binding'])
    p.update(parent_binding=old,parent_rows=parents)
    p['binding']=dict(fingerprint=fingerprint(),seed=seed,parent_binding=old,
        parent_protocol_sha256=sha((original.archive_for(seed)/'protocol.json').read_bytes()),
        parent_rows_sha256=[sha((original.archive_for(seed)/'rows'/f'{i:04d}'/'row.json').read_bytes()) for i in (0,1)],
        additional_native_forwards=0,GT_used=False,optimizer_updates=0)
    return p


def admit(seed):
    p=prepare_cpu(seed)
    result=dict(binding=p['binding'],cuda_initialized=False,optimizer_updates=0,actual_CPU_restore_passed=True)
    write_json(archive_for(seed)/'admission.json',result);return result


def predict_mask(found,trace,adj):
    _,q=original.native_component_rho_q(trace['spikes'],trace['component_spikes'],settle=512)
    labels=np.asarray(original.reference.source.source.full.screen.prediction_from_trace(found,trace,adj,q),dtype=np.int64)
    return labels,q


def predict(seed):
    p=prepare_cpu(seed);a=archive_for(seed)
    if json.loads((a/'admission.json').read_bytes())['binding']!=p['binding']:raise ValueError('layout admission differs')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    found=p['found'];original.reference.source.source.source.model64.move_to_device(found,device)
    runtime=original.reference.scope.model.resume_policy.compatible_runtime(original.reference.scope.model.checkpoint.runtime_signature(device))
    if runtime!=p['parent_binding']['candidate']['expected_runtime']:raise ValueError('accepted numerical runtime differs')
    row_hashes=[]
    for index in (0,1):
        folder=a/'rows'/f'{index:04d}'
        if not folder.exists():
            parent=original.archive_for(seed)/'rows'/f'{index:04d}'
            with np.load(parent/'raw_trace.npz',allow_pickle=False) as raw:
                s=restore_native_spikes(raw['component_spikes'],device)
                g=torch.from_numpy(np.ascontiguousarray(raw['component_gates'])).to(device)
                adj=torch.from_numpy(np.ascontiguousarray(raw['adjacency'])).to(device)
            if s.shape!=(1,4,4096,1024) or g.shape!=s.shape or adj.shape!=(1,4096,4096):raise ValueError('retained native geometry differs')
            trace=dict(component_spikes=s,component_gates=g,spikes=s.mean(dim=1))
            with torch.no_grad():
                actual,qa=predict_mask(found,trace,adj)
                legacy=gate_only_trace(trace);legacy_mask,ql=predict_mask(found,legacy,adj);del ql
                with np.load(parent/'masks.npz',allow_pickle=False) as masks:
                    if not np.array_equal(actual,masks['actual']) or not np.array_equal(legacy_mask,masks['gate_only']):
                        raise ValueError('actual or legacy parent mask replay is not bitwise exact; no event attribution')
                matched=matched_gate_only_trace(trace);matched_mask,qm=predict_mask(found,matched,adj)
                equal_tail=bool(torch.equal(s[...,512:],g[...,512:]))
                same_q=bool(torch.equal(qa,qm))
                if equal_tail and (not same_q or not np.array_equal(actual,matched_mask)):
                    raise ValueError('identical-tail negative control changed Q or masks')
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True)
            save_arrays(stage/'masks.npz',actual=actual,legacy_gate_only=legacy_mask,matched_gate_only=matched_mask)
            write_json(stage/'row.json',dict(binding=p['binding'],index=index,
                actual_parent_mask_replay_bitwise_equal=True,legacy_parent_mask_replay_bitwise_equal=True,
                actual_component_stride=list(s.stride()),legacy_component_stride=list(legacy['component_spikes'].stride()),
                matched_component_stride=list(matched['component_spikes'].stride()),
                actual_tail_bitwise_equal_to_gates=equal_tail,actual_Q_bitwise_equal_to_matched_Q=same_q,
                actual_vs_matched=compare_labels(actual,matched_mask),legacy_vs_matched=compare_labels(legacy_mask,matched_mask),
                masks_sha256=sha((stage/'masks.npz').read_bytes()),additional_native_forwards=0,optimizer_updates=0,GT_used=False))
            stage.rename(folder);del s,g,adj,trace,actual,legacy,legacy_mask,matched,matched_mask,qa,qm
        read_row(folder,p['binding'],index);row_hashes.append(sha((folder/'row.json').read_bytes()))
    if original.reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=p['parent_binding']['candidate']['model_sha256']:
        raise ValueError('readout replay mutated frozen model')
    result=dict(binding=p['binding'],row_sha256=row_hashes,additional_native_forwards=0,optimizer_updates=0,GT_used=False)
    if (a/'protocol.json').exists():
        if json.loads((a/'protocol.json').read_bytes())!=result:raise ValueError('completed layout protocol differs')
    else:write_json(a/'protocol.json',result)


def read_row(folder,binding,index):
    row=json.loads((folder/'row.json').read_bytes())
    if row['binding']!=binding or row['index']!=index or row['matched_component_stride']!=row['actual_component_stride'] or not row['actual_parent_mask_replay_bitwise_equal'] or not row['legacy_parent_mask_replay_bitwise_equal'] or row['additional_native_forwards'] or row['optimizer_updates'] or row['GT_used']:
        raise ValueError('layout control scope or replay differs')
    if sha((folder/'masks.npz').read_bytes())!=row['masks_sha256']:raise ValueError('retained replay masks changed')
    with np.load(folder/'masks.npz',allow_pickle=False) as masks:
        if row['actual_vs_matched']!=compare_labels(masks['actual'],masks['matched_gate_only']) or row['legacy_vs_matched']!=compare_labels(masks['legacy_gate_only'],masks['matched_gate_only']):raise ValueError('raw masks do not reproduce layout comparisons')
    return row


def validated_seed(seed):
    a=archive_for(seed);ad=json.loads((a/'admission.json').read_bytes());protocol=json.loads((a/'protocol.json').read_bytes())
    if protocol['binding']!=ad['binding'] or ad['binding']['fingerprint']!=fingerprint() or len(protocol['row_sha256'])!=2:raise ValueError('registered layout control changed')
    if sha((original.archive_for(seed)/'protocol.json').read_bytes())!=ad['binding']['parent_protocol_sha256']:raise ValueError('original241 protocol changed')
    parents=original.validated_seed(seed);rows=[]
    for i,digest in enumerate(protocol['row_sha256']):
        folder=a/'rows'/f'{i:04d}'
        if sha((folder/'row.json').read_bytes())!=digest:raise ValueError('layout row changed')
        rows.append(read_row(folder,ad['binding'],i))
    return rows


def score():
    rows=[dict(seed=seed,case_index=i,**{k:r[k] for k in ('actual_vs_matched','legacy_vs_matched','actual_tail_bitwise_equal_to_gates','actual_Q_bitwise_equal_to_matched_Q')}) for seed in SEEDS for i,r in enumerate(validated_seed(seed))]
    result=dict(status='complete_layout_matched_event_readout_control',fingerprint=fingerprint(),cases=rows,
        actual_partition_unchanged_case_count=sum(r['actual_vs_matched']['partition_equal_up_to_foreground_ID_permutation'] for r in rows),
        legacy_partition_changed_by_layout_case_count=sum(not r['legacy_vs_matched']['partition_equal_up_to_foreground_ID_permutation'] for r in rows),
        total_cases=6,all_original_actual_and_legacy_masks_replayed_bitwise=True,
        additional_native_forwards=0,optimizer_updates=0,GT_used=False,
        changes_upstream_dynamics=False,full_membrane_causal_necessity_proven=False,performance_improvement_claim=False)
    write_json(ARCHIVE/'score_v1/summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--admit',action='store_true');p.add_argument('--score',action='store_true');args=p.parse_args()
    if args.score:print(json.dumps(score()))
    elif args.admit and args.seed is not None:print(json.dumps(admit(args.seed)))
    else:p.error('choose CPU admission or completed layout score')
