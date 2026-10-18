"""Fixed512 child endpoint; freeze predictions before opening DEV GT."""
import io,json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0211_spike_anchor_center_assignment import run as parent
from . import engine

HERE=Path(__file__).resolve().parent;ARCHIVE=HERE/'results_archive'
sha=parent.sha;write_json=parent.write_json
ARMS=('primary512','mean_input','moments_only','learned_foreground')


def fingerprint():
    names=('DESIGN_KO.md','engine.py','run.py','queue.py','test_engine.py','test_pipeline.py')
    return dict(files={name:sha((HERE/name).read_bytes()) for name in names},dependency=parent.cache.fingerprint(),
                parent_steps=128,total_steps=512,new_steps=384,TRAIN=32,DEV=16,core_updates=0)


def prepare_cpu():
    if torch.cuda.is_initialized():raise ValueError('prepare before CUDA')
    labels,protocol,report,_=parent.validated_predictions();arrays,_=parent.cache.load_arrays()
    boundary=parent.ARCHIVE/'head_work/boundary.ptx';payload=parent.cache.source.full.unpack_checkpoint(boundary.read_bytes())
    if payload['completed_updates']!=128 or report['actual_Adam_steps']!=[128] or len(report['history'])!=128:raise ValueError('actual parent128 required')
    if parent.state_sha(payload['wrapped_state'])!=report['final_model_sha256']:raise ValueError('parent model mismatch')
    binding=dict(science=fingerprint(),parent_boundary_sha256=sha(boundary.read_bytes()),
                 parent_head_model_sha256=report['final_model_sha256'],
                 parent_report_sha256=sha((parent.ARCHIVE/'head_training/report.json').read_bytes()),
                 cache_protocol_sha256=sha((parent.ARCHIVE/'cache/protocol.json').read_bytes()),DEV_GT_read=False)
    return arrays,protocol,binding


def admit():
    torch.set_num_threads(1);arrays,protocol,binding=prepare_cpu()
    stored=parent.cache.checkpoint.inspect_committed_metadata(parent.ARCHIVE/'head_work')['runtime']
    with torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        current=parent.cache.checkpoint.runtime_signature('cpu')
    for key in ('torch_version','cuda_runtime','cudnn_version','python_version','platform','deterministic_algorithms',
                'cudnn_benchmark','cudnn_deterministic','allow_tf32_matmul','allow_tf32_cudnn','float32_matmul_precision','cublas_workspace_config'):
        if current[key]!=stored[key]:raise ValueError('parent math/software runtime mismatch before GPU: '+key)
    payload=parent.cache.source.full.unpack_checkpoint((parent.ARCHIVE/'head_work/boundary.ptx').read_bytes())
    head=parent.CenterHead();head.load_state_dict(payload['wrapped_state']);before=parent.state_sha(head.state_dict())
    target=engine.library.select_targets(engine.library.prepare_targets(arrays['TRAIN_labels'][:1]),[0],'cpu')
    value,_=engine.library.loss(head(torch.from_numpy(arrays['TRAIN_features'][:1])),target);value.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('actual TRAIN/native64 head gradient admission failed')
    if parent.state_sha(head.state_dict())!=before or torch.cuda.is_initialized():raise ValueError('CPU admission changed source or initialized CUDA')
    result=dict(status='actual_parent128_NATIVE64_TRAIN_gradient_CPU_admitted',binding=binding,loss=float(value.detach()),cuda_initialized=False,parent_math_software_settings_verified=True)
    write_json(ARCHIVE/'admission.json',result);return result


def save_arm(arm,labels,binding,decoder,head_sha):
    folder=ARCHIVE/'predictions'/arm;folder.mkdir(parents=True,exist_ok=True);path=folder/'labels.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if not np.array_equal(saved['labels'],labels):raise ValueError('frozen child masks changed')
    else:
        data=io.BytesIO();np.savez_compressed(data,labels=labels);parent.cache.checkpoint._atomic_bytes(path,data.getvalue())
    protocol=dict(arm=arm,binding=binding,head_model_sha256=head_sha,labels_sha256=sha(labels.tobytes()),
                  file_sha256=sha(path.read_bytes()),decoder=decoder,DEV_GT_read=False)
    target=folder/'protocol.json'
    if target.exists():
        if json.loads(target.read_bytes())!=protocol:raise ValueError('child prediction protocol changed')
    else:write_json(target,protocol)


def worker():
    arrays,protocol,binding=prepare_cpu()
    if binding!=json.loads((ARCHIVE/'admission.json').read_bytes())['binding']:raise ValueError('CPU-admitted parent/source changed')
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.8,0)
    report=engine.extend(arrays['TRAIN_features'],arrays['TRAIN_labels'],protocol,device='cuda:0',
        parent_work=parent.ARCHIVE/'head_work',work=ARCHIVE/'head_work',output=ARCHIVE/'head_training')
    payload=parent.cache.source.full.unpack_checkpoint((ARCHIVE/'head_training/step512/boundary.ptx').read_bytes())
    head=parent.CenterHead().to('cuda:0');head.load_state_dict(payload['wrapped_state']);head.eval();head.requires_grad_(False)
    if parent.state_sha(head.state_dict())!=report['final_model_sha256']:raise ValueError('wrong child head endpoint')
    train_path=ARCHIVE/'TRAIN_detection.json'
    with torch.no_grad(),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
        if not train_path.exists():
            recalls=[];fallback=[]
            for index,features in enumerate(arrays['TRAIN_features']):
                out=head(torch.from_numpy(features[None]).to('cuda:0'));heat=out['heat'][0].cpu().numpy();offset=out['offset'][0].cpu().numpy()
                centers=parent.training.targets(arrays['TRAIN_labels'][index])['centers'];prob=1/(1+np.exp(-np.clip(heat,-80,80)))
                recalls.append(float((prob[centers[:,0],centers[:,1]]>=.3).mean()))
                _,info=parent.decode(heat,offset,arrays['TRAIN_baseline'][index]);fallback.append(info['fallback'])
            write_json(train_path,dict(binding=binding,head_model_sha256=report['final_model_sha256'],
                GT_center_fraction_reaching_fixed_threshold=float(np.mean(recalls)),fallback_fraction=float(np.mean(fallback)),DEV_GT_read=False))
        for arm in ARMS:
            rows=[];info=[]
            for index,features in enumerate(arrays['DEV_features']):
                x=torch.from_numpy(features[None]).to('cuda:0')
                if arm in ('mean_input','moments_only'):x=parent.head_input(x,head,arm)
                out=head(x);support=arrays['DEV_baseline'][index]
                if arm=='learned_foreground':support=(out['foreground'][0].cpu().numpy()>=0).astype(np.int64)
                labels,details=parent.decode(out['heat'][0].cpu().numpy(),out['offset'][0].cpu().numpy(),support)
                rows.append(labels);info.append(details)
            save_arm(arm,np.stack(rows),binding,info,report['final_model_sha256'])
    if parent.state_sha(head.state_dict())!=report['final_model_sha256']:raise ValueError('inference mutated child')


def validated_predictions():
    admission=json.loads((ARCHIVE/'admission.json').read_bytes());binding=admission['binding']
    if binding['science']!=fingerprint():raise ValueError('child scientific source changed')
    old,protocol,parent_report,_=parent.validated_predictions()
    report=json.loads((ARCHIVE/'head_training/report.json').read_bytes())
    if report['actual_Adam_steps']!=[512] or len(report['history'])!=512 or report['history'][:128]!=parent_report['history']:raise ValueError('actual512/parent prefix mismatch')
    if sha((parent.ARCHIVE/'head_work/boundary.ptx').read_bytes())!=binding['parent_boundary_sha256']:raise ValueError('parent source boundary changed')
    payload=parent.cache.source.full.unpack_checkpoint((ARCHIVE/'head_training/step512/boundary.ptx').read_bytes())
    if payload['completed_updates']!=512 or parent.state_sha(payload['wrapped_state'])!=report['final_model_sha256']:raise ValueError('wrong actual child snapshot')
    if sorted({int(v['step'].item()) for v in payload['optimizer_state']['state'].values()})!=[512]:raise ValueError('wrong Adam count')
    predictions={'baseline':old['baseline'],'parent128':old['step128']};records={}
    for arm in ARMS:
        folder=ARCHIVE/'predictions'/arm;record=json.loads((folder/'protocol.json').read_bytes());path=folder/'labels.npz'
        if record['binding']!=binding or record['head_model_sha256']!=report['final_model_sha256'] or record['DEV_GT_read'] is not False or record['file_sha256']!=sha(path.read_bytes()):raise ValueError('wrong child prediction provenance')
        with np.load(path,allow_pickle=False) as saved:labels=saved['labels'].copy()
        if labels.dtype!=np.int64 or labels.shape!=(16,64,64) or (labels<0).any() or sha(labels.tobytes())!=record['labels_sha256']:raise ValueError('invalid immutable masks')
        if arm!='learned_foreground' and not np.array_equal(labels>0,old['baseline']>0):raise ValueError('primary/control foreground changed')
        predictions[arm]=labels;records[arm]=record
    return predictions,protocol,records,report


def score():
    predictions,protocol,records,report=validated_predictions()
    reference=protocol['binding']['source_binding']['training_bindings']['reference_binding']
    gt=parent.cache.source.full.metrics.read_ground_truth(reference)[list(parent.cache.OFFSETS)]
    vectors={};errors=[]
    for arm,labels in predictions.items():vectors[arm],error=parent.cache.source.full.metrics.measured_vectors(labels,gt);errors.append(error)
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors as count_errors
    counts={arm:[count_errors(p,g) for p,g in zip(labels,gt)] for arm,labels in predictions.items()}
    paired={arm:parent.paired(vectors[arm],vectors['baseline']) for arm in ARMS}
    result=dict(status='completed_center_head_single_budget_extension',grid=[64,64],images=16,seed_scope=[2],TRAIN_images=32,
        scores={arm:dict(zip(parent.cache.source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in vectors.items()},
        paired_minus_baseline=paired,primary_expansion_gate_passed=bool(paired['primary512']['fg_ari']['mean']>=.01 and paired['primary512']['fg_ari']['ci95'][0]>0 and paired['primary512']['matched_object_iou']['mean']>=0),
        count_diagnostics={arm:{key:float(np.mean([r[key] for r in rows])) for key in ('true_object_count','predicted_positive_label_count','raw_count_exact')} for arm,rows in counts.items()},
        fallback_fraction={arm:float(np.mean([row['fallback'] for row in record['decoder']])) for arm,record in records.items()},
        TRAIN_detection=json.loads((ARCHIVE/'TRAIN_detection.json').read_bytes()),actual_Adam_steps=[512],new_Adam_steps=384,parent_history128_preserved=True,
        core_forward_passes=0,core_optimizer_steps=0,mask_supervised=True,new70000_training=False,full320_three_seed_goal_verified=False,
        independent_maximum_metric_abs_error=max(errors),binding=records['primary512']['binding'])
    write_json(ARCHIVE/'score_v1/summary.json',result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    print(json.dumps(admit() if args.admit else score()),flush=True)
