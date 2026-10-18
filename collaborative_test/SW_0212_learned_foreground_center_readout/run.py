"""No optimization: validate exact parent replay, freeze all new masks, then GT."""
import io,json,os,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0211_spike_anchor_center_assignment import run as parent
from .readout import assign,require_replay

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
sha=parent.sha


def fingerprint():
    names=('DESIGN_KO.md','readout.py','run.py','queue.py','test_readout.py')
    return dict(files={name:sha((HERE/name).read_bytes()) for name in names},
                parent=parent.cache.fingerprint(),primary_steps=128,
                foreground_logit_threshold=0.,device='cpu',new_optimizer_steps=0)


def save_predictions(labels,binding,decoder):
    path=ARCHIVE/'predictions';path.mkdir(parents=True,exist_ok=True)
    data=io.BytesIO();np.savez_compressed(data,labels=labels)
    if (path/'labels.npz').exists():
        with np.load(path/'labels.npz',allow_pickle=False) as saved:require_replay(saved['labels'],labels)
    else:parent.cache.checkpoint._atomic_bytes(path/'labels.npz',data.getvalue())
    record=dict(binding=binding,labels_sha256=sha(labels.tobytes()),decoder=decoder,DEV_GT_read=False)
    target=path/'protocol.json'
    if target.exists():
        if json.loads(target.read_bytes())!=record:raise ValueError('immutable prediction protocol changed')
    else:parent.write_json(target,record)
    return record


def frozen_predictions():
    admission=json.loads((ARCHIVE/'admission.json').read_bytes())
    if admission['fingerprint']!=fingerprint():raise ValueError('registered readout source changed')
    labels,protocol,report,_=parent.validated_predictions()
    arrays,_=parent.cache.load_arrays()
    folder=parent.checkpoint_for('step128');head_report=json.loads((folder/'report.json').read_bytes())
    payload=parent.cache.source.full.unpack_checkpoint((folder/'boundary.ptx').read_bytes())
    if payload['completed_updates']!=128 or payload['bindings']!=report['binding']:raise ValueError('wrong parent endpoint')
    head=parent.CenterHead();head.load_state_dict(payload['wrapped_state'],strict=True);head.eval();head.requires_grad_(False)
    before=parent.state_sha(head.state_dict())
    if before!=head_report['model_sha256']:raise ValueError('parent weights changed')
    replays=[];new=[];decoder=[]
    with torch.no_grad():
        for index in range(16):
            x=torch.from_numpy(arrays['DEV_features'][index:index+1]);out=head(x)
            heat=out['heat'][0].numpy();offset=out['offset'][0].numpy()
            replay,_=parent.decode(heat,offset,arrays['DEV_baseline'][index]);replays.append(replay)
            predicted,info=assign(out['foreground'][0].numpy(),heat,offset);new.append(predicted);decoder.append(info)
    # All replay masks must pass before any new score is permitted.
    require_replay(np.stack(replays),labels['step128'])
    if parent.state_sha(head.state_dict())!=before:raise ValueError('CPU inference mutated head')
    binding=dict(science=fingerprint(),head_model_sha256=before,
                 parent_summary_sha256=sha((parent.ARCHIVE/'score_v1/summary.json').read_bytes()),
                 parent_protocol_sha256=sha((parent.ARCHIVE/'DEV_predictions/step128/protocol.json').read_bytes()),
                 cache_protocol_sha256=sha((parent.ARCHIVE/'cache/protocol.json').read_bytes()),
                 DEV_ids=protocol['binding']['DEV_ids'],parent_exact_replay=True,new_optimizer_steps=0)
    new=np.stack(new);record=save_predictions(new,binding,decoder)
    with np.load(ARCHIVE/'predictions/labels.npz',allow_pickle=False) as saved:frozen=saved['labels'].copy()
    if sha(frozen.tobytes())!=record['labels_sha256']:raise ValueError('frozen readout labels changed')
    if frozen.dtype!=np.int64 or frozen.shape!=(16,64,64) or (frozen<0).any():raise ValueError('wrong prediction geometry')
    labels['learned_foreground']=frozen
    return {key:labels[key] for key in ('baseline','step128','learned_foreground')},protocol,record


def score():
    predictions,protocol,record=frozen_predictions()
    reference=protocol['binding']['source_binding']['training_bindings']['reference_binding']
    gt=parent.cache.source.full.metrics.read_ground_truth(reference)[list(parent.cache.OFFSETS)]
    vectors={};errors=[]
    for arm,labels in predictions.items():
        vectors[arm],error=parent.cache.source.full.metrics.measured_vectors(labels,gt);errors.append(error)
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors as count_errors
    counts={arm:[count_errors(p,g) for p,g in zip(labels,gt)] for arm,labels in predictions.items()}
    summary=dict(status='completed_frozen_head_learned_foreground_readout',seed_scope=[2],images=16,grid=[64,64],
        scores={arm:dict(zip(parent.cache.source.full.metrics.independent.METRICS,values.mean(0).tolist())) for arm,values in vectors.items()},
        paired_minus={arm:parent.paired(vectors['learned_foreground'],vectors[arm]) for arm in ('baseline','step128')},
        foreground_changed_fraction=float(np.mean((predictions['learned_foreground']>0)!=(predictions['step128']>0))),
        fallback_fraction=float(np.mean([row['fallback'] for row in record['decoder']])),
        mean_predicted_objects=float(np.mean([len(np.unique(row[row>0])) for row in predictions['learned_foreground']])),
        count_diagnostics={arm:{key:float(np.mean([row[key] for row in values])) for key in ('true_object_count','predicted_positive_label_count','raw_count_exact')} for arm,values in counts.items()},
        parent_primary128_replayed_exactly=True,head_weights_unchanged=True,
        new_optimizer_steps=0,mask_supervised=True,new70000_training=False,full320_three_seed_goal_verified=False,
        frozen_readout_binding=record['binding'],independent_maximum_metric_abs_error=max(errors))
    parent.write_json(ARCHIVE/'score_v1/summary.json',summary)
    return summary


def admit():
    if torch.cuda.is_initialized():raise ValueError('CPU-only readout')
    torch.set_num_threads(1)
    head=parent.CenterHead().eval()
    with torch.no_grad():out=head(torch.zeros(1,268,64,64))
    labels,_=assign(out['foreground'][0].numpy(),out['heat'][0].numpy(),out['offset'][0].numpy())
    if labels.shape!=(64,64):raise ValueError('native64 head smoke failed')
    record=dict(status='native64_frozen_readout_CPU_admitted',fingerprint=fingerprint(),cuda_initialized=False)
    parent.write_json(ARCHIVE/'admission.json',record);return record


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    print(json.dumps(admit() if args.admit else score()),flush=True)
