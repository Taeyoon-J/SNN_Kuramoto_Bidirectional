"""Preserved-v1 audit successor: load Slot's registered archives explicitly."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from collaborative_test.SW_0185_qce_spatial_connectivity import completion_audit as math_audit

HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'results_archive/audit_v2/completion_audit.json'

def checked_slot_arrays(ids,pixels,*,expected_ids):
    ids=np.asarray(ids)
    pixels=np.asarray(pixels)
    if ids.dtype.kind not in 'iu' or not np.array_equal(ids,np.asarray(expected_ids)):
        raise ValueError('Slot archive IDs differ from registered fixed320 order')
    if pixels.shape!=(320,128,128) or pixels.dtype.kind not in 'iu' or np.any(pixels<0):
        raise ValueError('Slot archive must contain native128 integer label maps')
    return math_audit.modal_labels(pixels)

def load_slot_native32(seed):
    from collaborative_test.SW_0185_qce_spatial_connectivity import score
    contract=score.contract;source=score.sw165
    folder=source.SLOT_ROOT/f'seed{seed}_epoch10'
    archive,protocol=folder/'predictions.npz',folder/'protocol.json'
    baseline_raw=source.SLOT_BASELINE.read_bytes()
    if hashlib.sha256(baseline_raw).hexdigest()!=source.SLOT_BASELINE_SHA256:raise ValueError('Slot baseline changed')
    registered=json.loads(baseline_raw)['seeds'][str(seed)]
    proto=json.loads(protocol.read_text())
    digest=contract.validate_slot_protocol(proto,seed=seed,prediction_path=archive)
    protocol_sha=hashlib.sha256(protocol.read_bytes()).hexdigest()
    if (registered['slot_checkpoint_sha256']!=contract.SLOT_CHECKPOINT_SHA256[seed]
        or registered['slot_prediction_sha256']!=digest or registered['slot_protocol_sha256']!=protocol_sha):
        raise ValueError('Slot archive does not bind the pinned same-seed baseline')
    with np.load(archive,allow_pickle=False) as stored:
        labels=checked_slot_arrays(stored['image_ids'],stored['labels'],expected_ids=contract.IMAGE_IDS)
    return dict(labels=labels,archive=archive,protocol=protocol,prediction_sha256=digest,protocol_sha256=protocol_sha)

def audit_completed():
    import h5py
    from collaborative_test.SW_0185_qce_spatial_connectivity import score,train
    raw=score.SCORE_PATH.read_bytes();report=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    if report.get('status')!='complete' or report.get('experiment')!='SW0185_qce_spatial_connectivity_three_seed_score':
        raise ValueError('unexpected completed score producer')
    prepared=score.prepare_score_inputs_cpu()
    if report['implementation_fingerprint']!=score.implementation_fingerprint():raise ValueError('score source changed')
    slots={seed:load_slot_native32(seed) for seed in (0,1,2)}
    for seed in (0,1,2):
        frozen=prepared['frozen'][seed];published=report['candidate_prediction_hashes'][str(seed)]
        if published['prediction_sha256']!=frozen['result']['prediction_sha256'] or published['protocol_sha256']!=frozen['result']['protocol_sha256']:
            raise ValueError('score does not bind current actual prediction')
    # Both candidate and Slot archives are validated before this GT access.
    with h5py.File(prepared['dataset'],'r') as hdf:
        gt=math_audit.modal_labels(np.asarray(hdf['mask'][1320:1640]))
    errors=[];training={}
    for seed in (0,1,2):
        frozen=prepared['frozen'][seed]
        for endpoint in ('conditional','standalone'):
            errors.append(math_audit.compare_prediction(frozen['result']['labels'][endpoint],gt,report['scores'][f'candidate_{endpoint}'],seed))
        errors.append(math_audit.compare_prediction(slots[seed]['labels'],gt,report['scores']['Slot70k_standalone'],seed))
        artifact=train.OUTPUT_ROOT/f'{score.pe.ARM}_seed{seed}'
        manifest=train.validate_training_artifact(score.pe.ARM,artifact,seed=seed)
        training[str(seed)]=dict(accepted_updates=manifest['accepted_updates'],
            initial_model_state_sha256=manifest['initial_model_state_sha256'],final_model_state_sha256=manifest['final_model_state_sha256'],
            checkpoint_sha256=manifest['checkpoint_sha256'],manifest_sha256=score.sha256_file(artifact/'manifest.json'))
    gate,paired=math_audit.verify_arithmetic(report)
    if hashlib.sha256(score.SCORE_PATH.read_bytes()).hexdigest()!=digest:raise ValueError('score changed during audit')
    for slot in slots.values():
        if score.sha256_file(slot['archive'])!=slot['prediction_sha256'] or score.sha256_file(slot['protocol'])!=slot['protocol_sha256']:
            raise ValueError('Slot archive/protocol changed during audit')
    return dict(status='SW0185_independent_actual_prediction_metrics_GT_modal_means_CI_and_gates_verified',
        audit_version=2,score_sha256=digest,maximum_metric_abs_error=max(errors),candidate_endpoint_image_rows=1920,
        slot_image_rows=960,training=training,gate=gate,paired_bootstrap=paired,
        slot_archives={str(seed):{key:slots[seed][key] for key in ('prediction_sha256','protocol_sha256')} for seed in (0,1,2)},
        means={g:{m:v['mean'] for m,v in rows.items()} for g,rows in report['scores'].items()},
        validation_data_read_for_audit=True,optimizer_updates=0,incumbent_changed=False,
        prior_audit_failure_preserved="KeyError('slot')",observed_unix=time.time(),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        math_helper_source_sha256=hashlib.sha256(Path(math_audit.__file__).read_bytes()).hexdigest())

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args(argv)
    report=audit_completed();math_audit.write_once(args.output,report)
    print(json.dumps(dict(status=report['status'],score_sha256=report['score_sha256'],maximum_metric_abs_error=report['maximum_metric_abs_error'])),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
