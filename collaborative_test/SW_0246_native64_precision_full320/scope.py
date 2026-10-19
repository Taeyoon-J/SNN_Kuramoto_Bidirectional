"""Accepted three-seed64 selection and actual frozen trained-model restoration."""
import io,json
import numpy as np
import torch
from collaborative_test.SW_0245_native64_precision_three_seed64 import run as training

SEEDS=(0,1,2);OFFSETS=tuple(range(320));MINI=training.OFFSETS
AUDIT_CODE_SHA256='68a0e46295448f3ac4902f0b72a6d7d988a7059fb87d66b32f5a97081a344711'


def verify_selection(score,audit,score_sha,source_fp,code_sha):
    if score['fingerprint']!=source_fp or audit['fingerprint']!=source_fp or audit['score_sha256']!=score_sha or code_sha!=AUDIT_CODE_SHA256:
        raise ValueError('different registered selection/auditor evidence')
    if score['seed_scope']!=[0,1,2] or score['image_count_per_seed']!=64 or score['full320_improvement_proven']:
        raise ValueError('registered three-seed64 screen required')
    if audit['status']!='completed_six_actual_CPU_checkpoint_audits_and_raw_mask_rescoring' or audit['raw_masks_independently_rescored']!=768 or audit['independent_metrics_per_seed_means_and_deltas_maximum_abs_error']>=1e-12 or not audit['all_prediction_protocols_validated_before_evaluation_GT']:
        raise ValueError('complete independently audited selection required')
    for arm in training.ARMS:
        for seed in SEEDS:
            row=audit['training_audits'][arm][str(seed)]
            if row['actual_Adam_steps']!=[128] or row['actual_stage_updates']!=128 or not row['all_core_and_projection_parameters_unchanged'] or not row['actual_final_model_restored_on_CPU']:
                raise ValueError('all six actual training checkpoint audits required')
    keys=training.reference_score.METRICS
    means={name:np.array([score['scores'][name][k] for k in keys],dtype=float) for name in ('control','within16','parent237','slot')}
    if any(not np.isfinite(v).all() for v in means.values()):raise ValueError('finite selection means required')
    positive=bool(means['within16'][0]-means['control'][0]>=.01 and means['within16'][0]-means['parent237'][0]>=.01 and np.all(means['within16'][1:]>means['slot'][1:]))
    if positive!=score['within16_promising_for_separate_full320_confirmation']:raise ValueError('selection gate differs from frozen means')
    return positive


def qualification():
    a=training.ROOT_ARCHIVE;raw=(a/'score_v1/summary.json').read_bytes();audit_raw=(a/'completion_audit_v1/summary.json').read_bytes()
    score=json.loads(raw);audit=json.loads(audit_raw);code=(a/'completion_audit_v1/audit_runner.py').read_bytes()
    eligible=verify_selection(score,audit,training.sha(raw),training.family_fingerprint(),training.sha(code))
    return dict(eligible=eligible,score_sha256=training.sha(raw),audit_sha256=training.sha(audit_raw),audit_code_sha256=training.sha(code))


def restore_encoder_bytes(found,report,raw):
    if training.sha(raw)!=report['checkpoint_sha256']:raise ValueError('trained checkpoint changed')
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if training.sha(outer['payload_bytes'])!=outer['payload_sha256']:raise ValueError('checkpoint payload corrupted')
    saved=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
    steps=sorted({int(v['step'].item()) for v in saved['optimizer_state']['state'].values()})
    if saved['completed_updates']!=128 or len(saved['history'])!=128 or steps!=[128] or saved['bindings']!=report['checkpoint_binding'] or training.sha(json.dumps(saved['history'],sort_keys=True,allow_nan=False).encode())!=report['history_sha256']:
        raise ValueError('actual Adam/history/bindings differ')
    if training.state_sha(found.wrapped.state_dict())!=report['binding']['frozen_core_sha256']:raise ValueError('frozen core differs before restoration')
    found.encoder.load_state_dict(saved['encoder_state'],strict=True)
    if training.state_sha(found.encoder.state_dict())!=report['final_encoder_sha256'] or training.reference.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=report['final_model_sha256']:
        raise ValueError('actual trained full model restoration differs')
    return saved


def restore_seed(seed):
    training.require_seed(seed)
    if torch.cuda.is_initialized():raise ValueError('restore source on CPU before CUDA')
    p=training.prepare_cpu(seed,'within16');bundle=training.validated_seed(seed,'within16');a=training.archive_for(seed,'within16');report=bundle['report']
    if report['binding']!=p['binding']:raise ValueError('different trained source/input preparation')
    audit=json.loads((a/'training_audit.json').read_bytes())
    if audit['actual_Adam_steps']!=[128] or not audit['actual_final_model_restored_on_CPU'] or not audit['all_core_and_projection_parameters_unchanged']:raise ValueError('source CPU checkpoint audit required')
    restore_encoder_bytes(p['found'],report,(a/'encoder_work/boundary.ptx').read_bytes())
    protocol=json.loads((a/'protocol.json').read_bytes())
    frequency=np.stack([training.read_row(a/'rows'/f'{i:04d}',protocol['binding'],i)[1] for i in range(64)])
    parent=training.reference.validated_seed(seed)
    import h5py
    dataset=p['binding']['dataset']
    with h5py.File(dataset['path'],'r') as stream:rgb=training.reference.scope.canonical_rgb(np.asarray(stream['image'][1320:1640]))
    if rgb.shape!=(320,128,128,3) or training.sha(rgb.tobytes())!=parent['protocol']['binding']['RGB_sha256'] or not np.array_equal(rgb[list(MINI)],p['dev']):raise ValueError('canonical full320 RGB differs from accepted64')
    return dict(found=p['found'],rgb=rgb,baseline=parent['labels'],mini=bundle['labels'],mini_frequency=frequency,report=report,source_binding=p['binding'],protocol_sha256=training.sha((a/'protocol.json').read_bytes()),seed=seed)
