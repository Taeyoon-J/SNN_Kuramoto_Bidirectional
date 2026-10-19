"""Immutable223 model restoration and predetermined secondary validation IDs."""
import io,json
import numpy as np
import torch
from collaborative_test.SW_0232_native64_exact_teacher_resume import train as model
from collaborative_test.SW_0221_native64_frequency_pretraining import evaluate as reader

OFFSETS=tuple(range(64));MINI=tuple(range(0,32,2));NEW=tuple(i for i in OFFSETS if i not in MINI)


def canonical_rgb(values):
    rgb=np.asarray(values)
    if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(128,128,3):raise ValueError('canonical128 RGB geometry required')
    if rgb.dtype!=np.uint8:rgb=np.rint(np.clip(rgb.astype(np.float32),0,1)*255).astype(np.uint8)
    return rgb


def restore(found,seed):
    folder=model.archive_for(seed,64);report_raw=(folder/'training_report.json').read_bytes();report=json.loads(report_raw);protocol_raw=(folder/'protocol.json').read_bytes();protocol=json.loads(protocol_raw)
    score_raw=(model.ARCHIVE/'score_v1/summary.json').read_bytes();score=json.loads(score_raw);audit=json.loads((model.ARCHIVE/'completion_audit_v1/summary.json').read_bytes())
    raw=(folder/'encoder_work/boundary.ptx').read_bytes();meta=model.checkpoint.inspect_committed_metadata(folder/'encoder_work')
    from collaborative_test.SW_0232_native64_exact_teacher_resume import prefix_guard
    prefix_guard.validate_saved(dict(seed=seed,binding=report['binding']))
    if report['binding']['fingerprint']!=model.fingerprint() or score['fingerprint']!=model.fingerprint() or audit['score_sha256']!=model.sha(score_raw):raise ValueError('frozen232 source/score changed')
    if meta['completed_updates']!=512 or meta['bindings']!=report['checkpoint_binding'] or model.sha(raw)!=report['checkpoint_file_sha256'] or model.sha(raw)!=audit['checkpoints'][str(seed)]['checkpoint_sha256']:raise ValueError('frozen232 checkpoint changed')
    if model.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=report['binding']['parent_model_sha256']:raise ValueError('wrong original foundation')
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);payload=torch.load(io.BytesIO(outer['payload_bytes']),map_location='cpu',weights_only=False)
    model.projection.module(found).load_state_dict(payload['wrapped_state'],strict=True);found.encoder.load_state_dict(payload['encoder_state'],strict=True)
    if (model.source.source.full.screen.native._state_sha(found.wrapped,found.encoder)!=report['final_model_sha256']
        or model.projection.frozen_core_sha(found)!=report['frozen_nonprojection_core_sha256'] or model.statistics_sha(found)!=report['input_statistics_sha256']):raise ValueError('actual232 model restoration mismatch')
    expected=dict(training_report_sha256=model.sha(report_raw),binding=report['binding'],model_sha256=report['final_model_sha256'])
    if protocol['binding']!=expected or protocol['DEV_GT_read'] is not False or model.sha(protocol_raw)!=audit['checkpoints'][str(seed)]['protocol_sha256']:raise ValueError('frozen232 mini protocol changed')
    rows=[reader.read_row(folder/'rows'/f'{i:04d}',expected,i) for i in range(16)]
    masks=np.stack([r[0] for r in rows]);frequency=np.stack([r[1] for r in rows])
    if model.sha(masks.tobytes())!=protocol['labels_sha256']:raise ValueError('frozen232 mini masks changed')
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    ref=dict(model_sha256=report['final_model_sha256'],checkpoint_sha256=model.sha(raw),report_sha256=model.sha(report_raw),protocol_sha256=model.sha(protocol_raw),
        source_score_sha256=model.sha(score_raw),mini_masks_sha256=model.sha(masks.tobytes()),mini_frequency_sha256=model.sha(frequency.tobytes()),expected_runtime=report['runtime'])
    return report,ref,masks,frequency
