"""Freeze/validate both model predictions before GT; publish aggregates only."""
import json
from pathlib import Path
import numpy as np
import torch
from . import run
from collaborative_test.SW_0154_native64_resolution_compare import score_slot64 as slot
from collaborative_test.SW_0185_qce_spatial_connectivity import completion_audit as independent


def validated_pair():
    result={};reference=None
    for arm in run.ARMS:
        protocol=json.loads((run.ARCHIVE/arm/'protocol.json').read_bytes())
        binding=protocol['bindings']
        if binding['fingerprint']!=run.fingerprint() or protocol.get('validation_GT_read') is not False or protocol.get('optimizer_updates')!=0:raise ValueError('invalid frozen prediction protocol')
        if reference is not None and reference!=binding:raise ValueError('prediction pair has different bindings')
        reference=binding
        if protocol['image_ids']!=run.IDS.tolist() or protocol['model_sha256']!=binding[arm+'_model_sha256']:raise ValueError('frozen prediction IDs/model mismatch')
        rows=[]
        for offset,image_id in enumerate(run.IDS):
            folder=run.ARCHIVE/arm/'rows'/f'{offset:04d}'
            record=json.loads((folder/'row.json').read_bytes());expected=record['binding']
            if (expected['arm']!=arm or expected['image_id']!=int(image_id) or expected['inference_seed']!=193000+offset
                or expected['model_sha256']!=protocol['model_sha256']
                or expected['RGB_sha256']!=binding['development_RGB_row_sha256'][offset]
                or expected['source_bindings_sha256']!=run.sha(json.dumps(binding,sort_keys=True).encode())):raise ValueError('row identity changed')
            rows.append(run.validate_saved_row(folder,expected))
        labels=np.stack(rows)
        if run.sha(labels.tobytes())!=protocol['labels_sha256']:raise ValueError('frozen prediction array hash mismatch')
        result[arm]=labels
    return result,reference


def measured_vectors(prediction,gt):
    from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks
    result=evaluate_patch_masks(torch.from_numpy(prediction.astype(np.int64)),torch.from_numpy(gt.astype(np.int64)))
    vectors=np.stack([result['per_image'][name].cpu().numpy() for name in independent.METRICS],axis=1)
    checked=np.stack([independent.image_metrics(p,t) for p,t in zip(prediction,gt)])
    maximum=float(np.max(np.abs(vectors-checked)))
    if not np.isfinite(vectors).all() or maximum>1e-12:raise ValueError('independent actual-mask metrics mismatch')
    return vectors,maximum


def read_ground_truth(binding):
    import h5py
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    dataset=Path(binding['dataset']['path']);stat=dataset.stat()
    if stat.st_size!=binding['dataset']['size'] or stat.st_mtime_ns!=binding['dataset']['mtime_ns']:raise ValueError('canonical dataset changed')
    with h5py.File(dataset,'r') as stream:pixels=np.asarray(stream['mask'][run.IDS[0]:run.IDS[-1]+1,...,0])
    gt=clevr_mask_patch(torch.from_numpy(pixels.copy()),2)['patch_labels'].cpu().numpy().astype(np.int64)
    if gt.shape!=(len(run.IDS),64,64) or not np.array_equal(gt,independent.modal_labels(pixels,patch=2)):raise ValueError('independent native64 modal GT mismatch')
    return gt


def score_pair():
    predictions,binding=validated_pair()  # All candidate bytes checked before GT.
    from collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64 import SLOT_ROOT
    slots={seed:slot._load_slot(SLOT_ROOT,seed) for seed in (0,1,2)}
    for row in slots.values():
        if not np.array_equal(row['ids'][:len(run.IDS)],run.IDS):raise ValueError('Slot mini-image IDs mismatch')
    gt=read_ground_truth(binding)
    vectors={};errors=[]
    for arm,pred in predictions.items():
        vectors[arm],error=measured_vectors(pred,gt);errors.append(error)
    for seed,row in slots.items():
        vectors['slot_seed'+str(seed)],error=measured_vectors(row['labels64'][:len(run.IDS)],gt);errors.append(error)
    vectors['slot_three_seed_mean']=np.mean([vectors['slot_seed'+str(s)] for s in (0,1,2)],axis=0)
    metrics=independent.METRICS
    delta=vectors['candidate']-vectors['baseline']
    draws=np.random.RandomState(193).randint(0,len(run.IDS),size=(10000,len(run.IDS)))
    ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    paired={name:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,name in enumerate(metrics)}
    scores={arm:{name:float(value[:,i].mean()) for i,name in enumerate(metrics)} for arm,value in vectors.items()}
    summary=dict(status='SW0193_native64_development_screen_complete_independently_checked',
        scope='exploratory seed0, predetermined32 development images; no full320/three-seed superiority claim',
        image_count=len(run.IDS),grid=[64,64],training_unique_images=8,training_proposals=2,
        seed_scope=[0],mask_supervised=True,unsupervised_leaderboard_eligible=False,
        scores=scores,paired_candidate_minus_baseline=paired,
        screen_gate_passed=paired['fg_ari']['mean']>=.01 and paired['fg_ari']['ci95'][0]>0,
        native64_FG090_final_goal_achieved=False,independent_maximum_metric_abs_error=max(errors),
        bindings=binding,ground_truth_opened_after_pair_frozen=True,raw_imagewise_score_public=False,
        optimizer_updates_by_this_experiment=0)
    out=run.ARCHIVE/'score_v1'
    run.write_json(out/'evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    run.write_json(out/'summary.json',summary)
    return summary

if __name__=='__main__':print(json.dumps(score_pair()),flush=True)
