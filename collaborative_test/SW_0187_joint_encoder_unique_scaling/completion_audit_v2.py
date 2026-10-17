"""Independent actual-label, paired arithmetic, fork and encoder-learning audit."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import h5py
from collaborative_test.SW_0187_joint_encoder_unique_scaling import score,train
from collaborative_test.SW_0186_full_encoder_qce import completion_audit as shared

HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'results_archive/audit_v2/completion_audit.json'
math_audit=shared.math_audit
write_once=math_audit.write_once
load_slot_native32=shared.load_slot_native32

def verify_arithmetic(report):
    scores=report['scores'];paired={}
    for group,rows in scores.items():
        for metric,row in rows.items():
            matrix=np.asarray([row['per_seed_per_image'][str(seed)] for seed in (0,1,2)],dtype=np.float64)
            if matrix.shape!=(3,320) or not np.isfinite(matrix).all():raise ValueError('invalid metric matrix')
            if (not np.allclose(matrix.mean(0),row['per_image'],rtol=0,atol=1e-12)
                    or abs(float(matrix.mean())-row['mean'])>1e-12
                    or any(abs(float(matrix[i].mean())-row['per_seed_mean'][str(seed)])>1e-12 for i,seed in enumerate((0,1,2)))):
                raise ValueError('reported seed/image mean differs from actual rows')
    sample=np.random.RandomState(178).randint(0,320,size=(10000,320))
    for name,item in report['paired_bootstrap'].items():
        left,right=name.split('_minus_',1)
        choices=[(f'_{ep}_{metric}',ep,metric) for ep in ('conditional','standalone')
                 for metric in ('fg_ari','foreground_iou','matched_object_iou') if right.endswith(f'_{ep}_{metric}')]
        if len(choices)!=1:raise ValueError('unknown paired comparison')
        suffix,endpoint,metric=choices[0];group=right[:-len(suffix)]
        delta=np.asarray(scores[f'{left}_{endpoint}'][metric]['per_image'])-np.asarray(scores[f'{group}_{endpoint}'][metric]['per_image'])
        mean=float(delta.mean());ci=np.percentile(delta[sample].mean(1),[2.5,97.5])
        if (abs(mean-item['mean'])>1e-12 or not np.allclose(ci,item['ci95'],rtol=0,atol=1e-12)
                or item['seed']!=178 or item['replicates']!=10000
                or item['resampling_unit']!='paired_image_after_three_seed_average'):
            raise ValueError('paired arithmetic or resampling contract changed')
        paired[name]=dict(mean=mean,ci95=ci.tolist())
    primary=scores['candidate1024_standalone']
    fg=primary['fg_ari']['mean']
    gates=dict(primary_endpoint='standalone',
        unique_data_fg_mean_above_control=fg>scores['control512_standalone']['fg_ari']['mean'],
        unique_data_fg_paired_ci95_lower_above_zero=paired['candidate1024_minus_control512_standalone_fg_ari']['ci95'][0]>0,
        fg_0_87_goal=fg>=.87,mask_supervised=True,unsupervised_leaderboard_eligible=False)
    for key,group in (('slot','Slot70k'),('sw0165','SW0165_foreground_induced'),('sw0181','SW0181'),('sw0186','SW0186')):
        gates[f'{key}_fg_mean_strictly_exceeded']=fg>scores[f'{group}_standalone']['fg_ari']['mean']
        gates[f'{key}_fg_paired_ci95_lower_above_zero']=paired[f'candidate1024_minus_{group}_standalone_fg_ari']['ci95'][0]>0
    for metric in ('foreground_iou','matched_object_iou'):
        gates[f'slot_{metric}_mean_strictly_exceeded']=primary[metric]['mean']>scores['Slot70k_standalone'][metric]['mean']
    gates['unique_data_gate_passed']=gates['unique_data_fg_mean_above_control'] and gates['unique_data_fg_paired_ci95_lower_above_zero']
    gates['slot_benchmark_passed']=all(gates[k] for k in ('slot_fg_mean_strictly_exceeded','slot_fg_paired_ci95_lower_above_zero',
                                                       'slot_foreground_iou_mean_strictly_exceeded','slot_matched_object_iou_mean_strictly_exceeded'))
    if gates!=report['gates']:raise ValueError('reported gates differ from independent arithmetic')
    return gates,paired

def audit_completed():
    if hashlib.sha256(Path(math_audit.__file__).read_bytes()).hexdigest()!='d81762ea5def3557c65c897fd16432403e4bf27b3ea1f590195f555d07b3440f':
        raise ValueError('independent metric helper source changed')
    raw=score.SCORE_PATH.read_bytes();digest=hashlib.sha256(raw).hexdigest();report=json.loads(raw)
    if report.get('status')!='complete' or report.get('experiment')!='SW0187_shared_prefix_unique_scaling_three_seed_score':
        raise ValueError('unexpected score producer')
    prepared=score.prepare_score_inputs_cpu()
    if report['implementation_fingerprint']!=score.implementation_fingerprint():raise ValueError('score source changed')
    if report['SW0186_control_score_sha256']!=score.SW0186_SCORE_SHA256:raise ValueError('reference changed')
    slots={seed:load_slot_native32(seed) for seed in (0,1,2)}
    training={};lineage=[]
    for seed in (0,1,2):
        item=prepared['frozen'][seed];histories={};manifests={}
        for arm in score.ARMS:
            frozen=item['results'][arm];published=report['candidate_prediction_hashes'][str(seed)][arm]
            if any(published[key]!=frozen[key] for key in ('prediction_sha256','protocol_sha256')):raise ValueError('prediction binding changed')
            artifact=train.validate_training_artifact(arm,train._paths(train.OUTPUT_ROOT,arm,seed)[1],seed=seed)
            manifests[arm]=artifact['manifest'];histories[arm]=artifact['history']
            if artifact['manifest_sha256']!=report['training_manifest_sha256'][str(seed)][arm]:raise ValueError('training binding changed')
            training[f'{seed}_{arm}']={key:artifact['manifest'][key] for key in ('accepted_updates','source_base_encoder_initial_sha256',
                'source_base_encoder_final_sha256','source_base_encoder_changed')}
        left,right=[manifests[arm] for arm in score.ARMS]
        if any(left[key]!=right[key] for key in ('prefix_state_sha256','prefix_rng_sha256')):raise ValueError('fork state/RNG differ')
        if histories['control512'][:32]!=histories['candidate1024'][:32]:raise ValueError('shared prefix histories differ')
        if histories['control512'][32]['start_transaction_sha256']!=histories['candidate1024'][32]['start_transaction_sha256']:
            raise ValueError('first branch model/Adam transaction differs')
        lineage.append(dict(seed=seed,prefix_state_sha256=left['prefix_state_sha256'],prefix_rng_sha256=left['prefix_rng_sha256'],shared_history_equal=True,first_branch_transaction_equal=True))
    # All six candidate archives, their learned-CNN checkpoints, and all Slot archives precede GT access.
    with h5py.File(prepared['dataset'],'r') as hdf:gt=math_audit.modal_labels(np.asarray(hdf['mask'][1320:1640]))
    errors=[]
    for seed in (0,1,2):
        for arm in score.ARMS:
            for endpoint in score.ENDPOINTS:
                errors.append(math_audit.compare_prediction(prepared['frozen'][seed]['results'][arm]['labels'][endpoint],gt,
                    report['scores'][f'{arm}_{endpoint}'],seed))
        errors.append(math_audit.compare_prediction(slots[seed]['labels'],gt,report['scores']['Slot70k_standalone'],seed))
    gates,paired=verify_arithmetic(report)
    if hashlib.sha256(score.SCORE_PATH.read_bytes()).hexdigest()!=digest:raise ValueError('score changed during audit')
    for slot in slots.values():
        if score.sha256_file(slot['archive'])!=slot['prediction_sha256'] or score.sha256_file(slot['protocol'])!=slot['protocol_sha256']:
            raise ValueError('Slot archive changed during audit')
    return dict(status='SW0187_independent_actual_metrics_GT_modal_means_CI_gates_forks_and_encoder_learning_verified',
        audit_version=2,score_sha256=digest,maximum_metric_abs_error=max(errors),candidate_endpoint_image_rows=3840,slot_image_rows=960,
        training=training,lineage=lineage,gates=gates,paired_bootstrap=paired,optimizer_updates=0,incumbent_changed=False,
        means={group:{metric:row['mean'] for metric,row in rows.items()} for group,rows in report['scores'].items()},
        observed_unix=time.time(),audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        math_helper_source_sha256=hashlib.sha256(Path(math_audit.__file__).read_bytes()).hexdigest())

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args(argv)
    report=audit_completed();write_once(args.output,report)
    print(json.dumps(dict(status=report['status'],score_sha256=report['score_sha256'])),flush=True)

if __name__=='__main__':main()
