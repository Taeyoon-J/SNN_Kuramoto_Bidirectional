"""Learn relations between existing components; never replace their interiors."""
import io,json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0211_spike_anchor_center_assignment import run as parent
from . import components,training

HERE=Path(__file__).resolve().parent;ARCHIVE=HERE/'results_archive'
sha=parent.sha;write_json=parent.write_json;ARMS=('primary','geometry_only')


def fingerprint():
    names=('DESIGN_KO.md','components.py','training.py','run.py','queue.py','test_merger.py')
    return dict(files={name:sha((HERE/name).read_bytes()) for name in names},cache_dependency=parent.cache.fingerprint(),
                TRAIN=32,DEV=16,steps=128,pair_channels=548,core_updates=0)


def prepare_cpu():
    if torch.cuda.is_initialized():raise ValueError('prepare cache and TRAIN before CUDA')
    arrays,protocol=parent.cache.load_arrays();mean,scale=parent.training.normalization(arrays['TRAIN_features'])
    tables=[];targets=[];weights=[];counts=[]
    for x,source,gt in zip(arrays['TRAIN_features'],arrays['TRAIN_baseline'],arrays['TRAIN_labels']):
        ids,pairs,features=components.component_pairs(x,source,mean,scale);counts.append(len(pairs))
        if len(pairs):
            tables.append(features);targets.append(components.soft_targets(source,gt,ids,pairs));weights.append(np.full(len(pairs),1/len(pairs),np.float32))
    x=np.concatenate(tables);y=np.concatenate(targets);w=np.concatenate(weights)
    binding=dict(science=fingerprint(),cache_protocol_sha256=sha((parent.ARCHIVE/'cache/protocol.json').read_bytes()),
                 TRAIN_pair_counts=counts,features_sha256=sha(x.tobytes()),targets_sha256=sha(y.tobytes()),weights_sha256=sha(w.tobytes()),DEV_GT_read=False)
    return arrays,protocol,mean,scale,x,y,w,binding


def admit():
    torch.set_num_threads(1);arrays,protocol,mean,scale,x,y,w,binding=prepare_cpu()
    head=training.PairHead();out=head(torch.from_numpy(x))
    if not torch.equal(out,torch.zeros_like(out)):raise ValueError('initial zero-edge identity is not exact')
    value=training.objective(out,torch.from_numpy(y),torch.from_numpy(w));value.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()) or sum(float(p.grad.square().sum()) for p in head.parameters())<=0:raise ValueError('actual pair gradient admission failed')
    for xdev,source in zip(arrays['DEV_features'],arrays['DEV_baseline']):
        ids,pairs,_=components.component_pairs(xdev,source,mean,scale);labels,_=components.merge(source,ids,pairs,np.zeros(len(pairs)))
        if not np.array_equal(labels,source):raise ValueError('zero-edge decoder changed source')
    result=dict(status='actual_component_pairs_native64_CPU_admitted',binding=binding,cuda_initialized=False,
                TRAIN_pairs=len(y),soft_positive_mass=float((y*w).sum()),initial_loss=float(value.detach()),zero_edge_DEV_identity_verified=True)
    if torch.cuda.is_initialized():raise ValueError('CPU admission initialized CUDA')
    write_json(ARCHIVE/'admission.json',result);return result


def worker():
    arrays,protocol,mean,scale,x,y,w,binding=prepare_cpu()
    if binding!=json.loads((ARCHIVE/'admission.json').read_bytes())['binding']:raise ValueError('admitted source/pairs changed')
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.8,0)
    report=training.fit(x,y,w,binding,work=ARCHIVE/'head_work',output=ARCHIVE/'head_training')
    payload=parent.cache.source.full.unpack_checkpoint((ARCHIVE/'head_work/boundary.ptx').read_bytes())
    head=training.PairHead().to('cuda:0');head.load_state_dict(payload['wrapped_state']);head.eval();head.requires_grad_(False)
    if parent.state_sha(head.state_dict())!=report['model_sha256']:raise ValueError('wrong relation endpoint')
    for arm in ARMS:
        labels=[];details=[]
        for features,source in zip(arrays['DEV_features'],arrays['DEV_baseline']):
            ids,pairs,table=components.component_pairs(features,source,mean,scale)
            with torch.no_grad():
                data=torch.from_numpy(table).to('cuda:0')
                if arm=='geometry_only':data=data.clone();data[:,:536]=head.mean[:,:536]
                logits=head(data).cpu().numpy()
            mask,info=components.merge(source,ids,pairs,logits);labels.append(mask);details.append(info)
        values=np.stack(labels);folder=ARCHIVE/'predictions'/arm;folder.mkdir(parents=True,exist_ok=True);path=folder/'labels.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                if not np.array_equal(saved['labels'],values):raise ValueError('immutable merger masks changed')
        else:
            raw=io.BytesIO();np.savez_compressed(raw,labels=values);parent.cache.checkpoint._atomic_bytes(path,raw.getvalue())
        record=dict(binding=binding,arm=arm,model_sha256=report['model_sha256'],labels_sha256=sha(values.tobytes()),file_sha256=sha(path.read_bytes()),details=details,DEV_GT_read=False)
        target=folder/'protocol.json'
        if target.exists():
            if json.loads(target.read_bytes())!=record:raise ValueError('merger prediction provenance changed')
        else:write_json(target,record)
    if parent.state_sha(head.state_dict())!=report['model_sha256']:raise ValueError('relation inference changed weights')


def validated_predictions():
    binding=json.loads((ARCHIVE/'admission.json').read_bytes())['binding']
    if binding['science']!=fingerprint() or binding['cache_protocol_sha256']!=sha((parent.ARCHIVE/'cache/protocol.json').read_bytes()):raise ValueError('merger source changed')
    arrays,protocol=parent.cache.load_arrays();report=json.loads((ARCHIVE/'head_training/report.json').read_bytes())
    payload=parent.cache.source.full.unpack_checkpoint((ARCHIVE/'head_work/boundary.ptx').read_bytes())
    if report['binding']['science']!=binding or report['actual_Adam_steps']!=[128] or len(report['history'])!=128 or payload['completed_updates']!=128 or parent.state_sha(payload['wrapped_state'])!=report['model_sha256']:raise ValueError('wrong actual merger training')
    if sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})!=[128]:raise ValueError('wrong relation Adam count')
    result={'baseline':arrays['DEV_baseline']};records={}
    for arm in ARMS:
        folder=ARCHIVE/'predictions'/arm;record=json.loads((folder/'protocol.json').read_bytes());path=folder/'labels.npz'
        if record['binding']!=binding or record['arm']!=arm or record['model_sha256']!=report['model_sha256'] or record['DEV_GT_read'] is not False or record['file_sha256']!=sha(path.read_bytes()):raise ValueError('invalid merger protocol')
        with np.load(path,allow_pickle=False) as saved:labels=saved['labels'].copy()
        if labels.dtype!=np.int64 or labels.shape!=(16,64,64) or sha(labels.tobytes())!=record['labels_sha256'] or not np.array_equal(labels>0,result['baseline']>0):raise ValueError('invalid source-preserving masks')
        for source,target in zip(result['baseline'],labels):
            for label in np.unique(source[source>0]):
                if len(np.unique(target[source==label]))!=1:raise ValueError('merger split an original component')
        result[arm]=labels;records[arm]=record
    return result,protocol,records,report


def score():
    labels,protocol,records,report=validated_predictions()
    reference=protocol['binding']['source_binding']['training_bindings']['reference_binding']
    gt=parent.cache.source.full.metrics.read_ground_truth(reference)[list(parent.cache.OFFSETS)]
    vectors={};errors=[]
    for arm,values in labels.items():vectors[arm],error=parent.cache.source.full.metrics.measured_vectors(values,gt);errors.append(error)
    paired={arm:parent.paired(vectors[arm],vectors['baseline']) for arm in ARMS};primary=paired['primary']
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors as count_errors
    counts={arm:[count_errors(p,g) for p,g in zip(values,gt)] for arm,values in labels.items()}
    summary=dict(status='completed_source_component_relation_merger',seed_scope=[2],grid=[64,64],images=16,TRAIN_images=32,
        scores={arm:dict(zip(parent.cache.source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in vectors.items()},paired_minus_baseline=paired,
        primary_minus_frozen_geometry=parent.paired(vectors['primary'],vectors['geometry_only']),
        primary_expansion_gate_passed=bool(primary['fg_ari']['mean']>=.01 and primary['fg_ari']['ci95'][0]>0 and primary['matched_object_iou']['mean']>=0),
        mean_component_merges={arm:float(np.mean([row['merges'] for row in record['details']])) for arm,record in records.items()},
        count_diagnostics={arm:{key:float(np.mean([row[key] for row in values])) for key in ('true_object_count','predicted_positive_label_count','raw_count_exact')} for arm,values in counts.items()},
        all_original_components_unsplit=True,source_foreground_preserved=True,actual_Adam_steps=[128],core_optimizer_steps=0,core_forward_passes=0,
        mask_supervised=True,new70000_training=False,full320_three_seed_goal_verified=False,independent_maximum_metric_abs_error=max(errors),
        frozen_geometry_control_is_not_retrained_physical_gating_causality=True,model_sha256=report['model_sha256'])
    write_json(ARCHIVE/'score_v1/summary.json',summary);return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    print(json.dumps(admit() if args.admit else score()),flush=True)
