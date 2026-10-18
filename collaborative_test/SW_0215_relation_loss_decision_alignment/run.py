"""Frozen214 head; TRAIN-derived odds correction and exact uncorrected replay."""
import io,json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0214_component_relation_merger import run as parent
from .calibration import training_log_odds_shift,correct

HERE=Path(__file__).resolve().parent;ARCHIVE=HERE/'results_archive'
sha=parent.sha;write_json=parent.write_json


def fingerprint():
    names=('DESIGN_KO.md','calibration.py','run.py','test_alignment.py')
    return dict(files={name:sha((HERE/name).read_bytes()) for name in names},parent=parent.fingerprint(),new_training_updates=0,device='cpu')


def freeze_predictions():
    torch.set_num_threads(1)
    original,protocol,old_records,report=parent.validated_predictions()
    arrays,_,mean,scale,x,y,w,binding=parent.prepare_cpu()
    shift,statistics=training_log_odds_shift(y,w)
    payload=parent.parent.cache.source.full.unpack_checkpoint((parent.ARCHIVE/'head_work/boundary.ptx').read_bytes())
    head=parent.training.PairHead();head.load_state_dict(payload['wrapped_state']);head.eval();head.requires_grad_(False)
    before=parent.parent.state_sha(head.state_dict())
    if before!=report['model_sha256']:raise ValueError('wrong frozen214 head')
    predictions={};details={}
    for arm in parent.ARMS:
        rows=[];replays=[];info=[]
        for features,source in zip(arrays['DEV_features'],arrays['DEV_baseline']):
            ids,pairs,table=parent.components.component_pairs(features,source,mean,scale)
            with torch.no_grad():
                values=torch.from_numpy(table)
                if arm=='geometry_only':values=values.clone();values[:,:536]=head.mean[:,:536]
                logits=head(values).numpy()
            replay,_=parent.components.merge(source,ids,pairs,logits);replays.append(replay)
            labels,record=parent.components.merge(source,ids,pairs,correct(logits,shift));rows.append(labels);info.append(record)
        if not np.array_equal(np.stack(replays),original[arm]):raise ValueError('CPU uncorrected214 replay mismatch; do not score')
        predictions[arm]=np.stack(rows);details[arm]=info
    if parent.parent.state_sha(head.state_dict())!=before:raise ValueError('frozen head changed')
    frozen=dict(science=fingerprint(),parent_model_sha256=before,parent_summary_sha256=sha((parent.ARCHIVE/'score_v1/summary.json').read_bytes()),
                TRAIN_shift=shift,TRAIN_statistics=statistics,parent_exact_CPU_replay=True,
                labels_sha256={arm:sha(values.tobytes()) for arm,values in predictions.items()},details=details,DEV_GT_read=False)
    folder=ARCHIVE/'predictions';folder.mkdir(parents=True,exist_ok=True);path=folder/'labels.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if any(not np.array_equal(saved[arm],values) for arm,values in predictions.items()):raise ValueError('corrected predictions changed')
    else:
        raw=io.BytesIO();np.savez_compressed(raw,**predictions);parent.parent.cache.checkpoint._atomic_bytes(path,raw.getvalue())
    saved=folder/'protocol.json'
    if saved.exists():
        if json.loads(saved.read_bytes())!=frozen:raise ValueError('corrected prediction protocol changed')
    else:write_json(saved,frozen)
    with np.load(path,allow_pickle=False) as source:
        for arm in parent.ARMS:
            if sha(source[arm].tobytes())!=frozen['labels_sha256'][arm]:raise ValueError('saved corrected mask hash differs')
    return original,predictions,protocol,frozen


def score():
    original,predictions,protocol,frozen=freeze_predictions()
    ref=protocol['binding']['source_binding']['training_bindings']['reference_binding']
    gt=parent.parent.cache.source.full.metrics.read_ground_truth(ref)[list(parent.parent.cache.OFFSETS)]
    values=dict(baseline=original['baseline'],uncorrected_primary=original['primary'],uncorrected_geometry=original['geometry_only'],corrected_primary=predictions['primary'],corrected_geometry=predictions['geometry_only'])
    vectors={};errors=[]
    for arm,labels in values.items():vectors[arm],error=parent.parent.cache.source.full.metrics.measured_vectors(labels,gt);errors.append(error)
    summary=dict(status='completed_TRAIN_relation_loss_decision_alignment',grid=[64,64],images=16,seed_scope=[2],TRAIN_images=32,
        scores={arm:dict(zip(parent.parent.cache.source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in vectors.items()},
        paired_minus_baseline={arm:parent.parent.paired(vectors[arm],vectors['baseline']) for arm in ('corrected_primary','corrected_geometry')},
        TRAIN_shift=frozen['TRAIN_shift'],TRAIN_statistics=frozen['TRAIN_statistics'],parent_exact_CPU_replay=True,
        new_training_updates=0,core_forward_passes=0,mask_supervised=True,new70000_training=False,full320_three_seed_goal_verified=False,
        mean_component_merges={arm:float(np.mean([row['merges'] for row in rows])) for arm,rows in frozen['details'].items()},
        independent_maximum_metric_abs_error=max(errors),source_foreground_preserved=True,source_components_unsplit=True,
        correction_interpretation='ideal weighted-BCE log-odds shift; finite classifier probabilities are not guaranteed calibrated')
    write_json(ARCHIVE/'score_v1/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(score()),flush=True)
