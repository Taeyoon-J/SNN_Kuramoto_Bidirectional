"""TRAIN-only readout fit, immutable DEV predictions, then GT-last scoring."""
import json,os
from pathlib import Path
import time
import numpy as np
import torch
from . import cache,training
from .center_head import CenterHead,decode
from collaborative_test.SW_0207_spike_foreground_classifier.classifier import state_sha

ARCHIVE=cache.ARCHIVE
HERE=cache.HERE
sha=cache.sha
write_json=cache.write_json
ARMS=('step32','step64','step128','mean_input','moments_only')

def head_input(x,head,arm):
    if arm not in ARMS:raise ValueError('unknown preregistered arm')
    if arm=='mean_input':return head.mean.expand_as(x).contiguous()
    if arm=='moments_only':
        values=x.clone();values[:,:256]=head.mean[:,:256];return values
    return x

def checkpoint_for(arm):return ARCHIVE/'head_training'/('step'+str(int(arm[4:]) if arm.startswith('step') else 128))

def prediction_binding(arm,index,cache_protocol,head_report):
    if arm not in ARMS or type(index) is not int or index not in range(16):raise ValueError('fixed DEV prediction identity required')
    return dict(arm=arm,index=index,image_id=cache_protocol['binding']['DEV_ids'][index],
        source_model_sha256=cache.source.MODELS['2'],head_model_sha256=head_report['model_sha256'],
        head_boundary_sha256=head_report['boundary_sha256'],
        cache_protocol_sha256=sha((ARCHIVE/'cache/protocol.json').read_bytes()),
        science_sha256=sha(json.dumps(cache.fingerprint(),sort_keys=True).encode()))

def load_prediction(folder,expected):
    record=json.loads((folder/'row.json').read_bytes());raw=(folder/'labels.npz').read_bytes()
    if record['binding']!=expected or record['labels_sha256']!=sha(raw):raise ValueError('head prediction provenance changed')
    import io
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:labels=saved['labels'].copy()
    if labels.shape!=(64,64) or labels.dtype!=np.int64 or (labels<0).any():raise ValueError('invalid center head mask')
    return labels,record['decoder']

def predict(arrays,protocol,*,device='cuda:0'):
    device=torch.device(device)
    training_report=json.loads((ARCHIVE/'head_training/report.json').read_bytes())
    if training_report['actual_Adam_steps']!=[128] or training_report['DEV_GT_read'] is not False or training_report['core_optimizer_steps']!=0:raise ValueError('primary128 TRAIN-only endpoint required')
    for arm in ARMS:
        folder=checkpoint_for(arm);head_report=json.loads((folder/'report.json').read_bytes());raw=(folder/'boundary.ptx').read_bytes()
        metadata=training.checkpoint.inspect_committed_metadata(folder);payload=cache.source.full.unpack_checkpoint(raw)
        expected_step=int(arm[4:]) if arm.startswith('step') else 128
        if head_report['binding']!=training_report['binding'] or metadata['bindings']!=training_report['binding'] or payload['completed_updates']!=expected_step or metadata['completed_updates']!=expected_step or sha(raw)!=head_report['boundary_sha256']:raise ValueError('head endpoint binding mismatch')
        head=CenterHead().to(device);head.load_state_dict(payload['wrapped_state'],strict=True);head.eval();head.requires_grad_(False)
        if state_sha(head.state_dict())!=head_report['model_sha256']:raise ValueError('actual head weights differ')
        rows=[];decoder=[]
        for index in range(16):
            expected=prediction_binding(arm,index,protocol,head_report);row=ARCHIVE/f'DEV_predictions/{arm}/rows/{index:04d}'
            if row.exists():labels,info=load_prediction(row,expected)
            else:
                with torch.no_grad(),torch.backends.cudnn.flags(deterministic=True,benchmark=False):
                    x=torch.from_numpy(arrays['DEV_features'][index:index+1]).to(device)
                    output=head(head_input(x,head,arm))
                    labels,info=decode(output['heat'][0].cpu().numpy(),output['offset'][0].cpu().numpy(),arrays['DEV_baseline'][index])
                stage=row.with_name(row.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
                with (stage/'labels.npz').open('xb') as stream:
                    np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
                write_json(stage/'row.json',dict(binding=expected,labels_sha256=sha((stage/'labels.npz').read_bytes()),decoder=info));stage.rename(row)
            if not np.array_equal(labels>0,arrays['DEV_baseline'][index]>0):raise ValueError('classifier changed source foreground')
            rows.append(labels);decoder.append(info)
        if state_sha(head.state_dict())!=head_report['model_sha256']:raise ValueError('head inference mutated weights')
        frozen=dict(arm=arm,cache_protocol_sha256=sha((ARCHIVE/'cache/protocol.json').read_bytes()),
            head_report_sha256=sha((folder/'report.json').read_bytes()),head_training_report_sha256=sha((ARCHIVE/'head_training/report.json').read_bytes()),
            labels_sha256=sha(np.stack(rows).tobytes()),DEV_image_ids=protocol['binding']['DEV_ids'],DEV_GT_read=False,core_updates=0)
        path=ARCHIVE/f'DEV_predictions/{arm}/protocol.json'
        if path.exists():
            if json.loads(path.read_bytes())!=frozen:raise ValueError('immutable head prediction protocol changed')
        else:write_json(path,frozen)

def worker():
    prepared=cache.prepare_cpu();cache.extract(prepared)
    arrays,protocol=cache.load_arrays()
    if arrays['TRAIN_features'].shape!=(32,268,64,64):raise ValueError('registered TRAIN32/native64 required')
    training.fit(arrays['TRAIN_features'],arrays['TRAIN_labels'],protocol)
    predict(arrays,protocol)

def validated_predictions():
    arrays,protocol=cache.load_arrays();result={'baseline':arrays['DEV_baseline']};decoder={}
    training_raw=(ARCHIVE/'head_training/report.json').read_bytes();report=json.loads(training_raw)
    if (report['status']!='completed_center_head_TRAIN_only' or report['actual_Adam_steps']!=[128] or len(report['history'])!=128
        or report['binding']['science']!=cache.fingerprint() or report['DEV_GT_read'] is not False
        or report['binding']['cache_protocol_sha256']!=sha(json.dumps(protocol,sort_keys=True).encode())
        or report['binding']['features_sha256']!=sha(arrays['TRAIN_features'].tobytes())
        or report['binding']['TRAIN_labels_sha256']!=sha(arrays['TRAIN_labels'].tobytes())):raise ValueError('changed primary endpoint or TRAIN data')
    for arm in ARMS:
        frozen=json.loads((ARCHIVE/f'DEV_predictions/{arm}/protocol.json').read_bytes())
        folder=checkpoint_for(arm);head_report=json.loads((folder/'report.json').read_bytes())
        if (frozen['arm']!=arm or frozen['cache_protocol_sha256']!=sha((ARCHIVE/'cache/protocol.json').read_bytes())
            or frozen['head_report_sha256']!=sha((folder/'report.json').read_bytes()) or frozen['head_training_report_sha256']!=sha(training_raw)
            or frozen['DEV_GT_read'] is not False or frozen['core_updates']!=0 or frozen['DEV_image_ids']!=protocol['binding']['DEV_ids']
            or head_report['binding']!=report['binding'] or head_report['boundary_sha256']!=sha((folder/'boundary.ptx').read_bytes())):raise ValueError('unfrozen DEV head result')
        rows=[];info=[]
        for index in range(16):
            labels,meta=load_prediction(ARCHIVE/f'DEV_predictions/{arm}/rows/{index:04d}',prediction_binding(arm,index,protocol,head_report))
            if not np.array_equal(labels>0,arrays['DEV_baseline'][index]>0):raise ValueError('predicted foreground not exactly preserved')
            rows.append(labels);info.append(meta)
        result[arm]=np.stack(rows);decoder[arm]=info
        if sha(result[arm].tobytes())!=frozen['labels_sha256']:raise ValueError('prediction arrays changed')
    return result,protocol,report,decoder

def paired(candidate,baseline):
    delta=np.asarray(candidate)-np.asarray(baseline)
    if delta.shape!=(16,3) or not np.isfinite(delta).all():raise ValueError('paired16 finite metric vectors required')
    draws=np.random.RandomState(211).randint(0,16,(10000,16));ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    return {k:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,k in enumerate(cache.source.full.metrics.independent.METRICS)}

def score():
    predictions,protocol,training_report,decoder=validated_predictions()
    reference=protocol['binding']['source_binding']['training_bindings']['reference_binding']
    gt=cache.source.full.metrics.read_ground_truth(reference)[list(cache.OFFSETS)]
    vectors={};errors=[]
    for arm,labels in predictions.items():vectors[arm],error=cache.source.full.metrics.measured_vectors(labels,gt);errors.append(error)
    if any(not np.array_equal(v[:,1],vectors['baseline'][:,1]) for v in vectors.values()):raise ValueError('foreground IoU changed despite preserved support')
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors as count_errors
    counts={arm:[count_errors(p,g) for p,g in zip(labels,gt)] for arm,labels in predictions.items()}
    count_summary={arm:{k:float(np.mean([r[k] for r in rows])) for k in ('true_object_count','predicted_positive_label_count','raw_count_exact')} for arm,rows in counts.items()}
    comparisons={arm:paired(v,vectors['baseline']) for arm,v in vectors.items() if arm!='baseline'}
    incremental={arm:paired(vectors['step128'],vectors[arm]) for arm in ('mean_input','moments_only')}
    primary=comparisons['step128'];gate=bool(primary['fg_ari']['mean']>=.01 and primary['fg_ari']['ci95'][0]>0 and primary['matched_object_iou']['mean']>=0)
    summary=dict(status='completed_native64_center_assignment_pilot',seed_scope=[2],images=16,grid=[64,64],TRAIN_images=32,primary_steps=128,
        scores={arm:dict(zip(cache.source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in vectors.items()},
        paired_minus_baseline=comparisons,frozen_input_ablation_differences=incremental,primary_expansion_gate_passed=gate,
        primary_step_fixed_before_DEV_GT=True,exploratory_steps=[32,64],all_source_foreground_bitwise_preserved=True,
        core_optimizer_steps=0,actual_head_Adam_steps=training_report['actual_Adam_steps'],mask_supervised=True,
        count_diagnostics=count_summary,fallback_fraction={arm:float(np.mean([r['fallback'] for r in rows])) for arm,rows in decoder.items()},
        independent_maximum_metric_abs_error=max(errors),cache_protocol_sha256=sha((ARCHIVE/'cache/protocol.json').read_bytes()),
        training_report_sha256=sha((ARCHIVE/'head_training/report.json').read_bytes()),
        source_model_sha256=cache.source.MODELS['2'],full320_three_seed_goal_verified=False,new70000_training=False,
        validation_selected_not_untouched_test=True,
        frozen_ablations_are_not_retrained_gating_causality_proof=True)
    write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}));write_json(ARCHIVE/'score_v1/summary.json',summary);return summary
