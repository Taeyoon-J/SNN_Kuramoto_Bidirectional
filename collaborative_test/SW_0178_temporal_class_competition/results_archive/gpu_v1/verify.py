import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from collaborative_test.SW_0178_temporal_class_competition import train

records=[]
for arm in train.ARMS:
    path=train.OUTPUT_ROOT/'work'/(arm+'_seed1')/'boundary.ptx'
    raw=path.read_bytes()
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    data=outer['payload_bytes']
    assert hashlib.sha256(data).hexdigest()==outer['payload_sha256']
    payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    metadata=outer['metadata']
    assert metadata['completed_updates']==payload['completed_updates']
    assert metadata['bindings']==payload['bindings'] and metadata['runtime']==payload['runtime']
    completed=payload['completed_updates'];history=payload['history']
    train._validate_cursor(completed,history)
    assert completed>=2,'wait on the same workers until two committed updates exist'
    bindings=payload['bindings']
    assert bindings['arm']==arm and bindings['initial_model_state_sha256']==train.EXPECTED_FRESH_STATE_SHA256
    assert bindings['mask_supervised'] and not bindings['unsupervised_leaderboard_eligible']
    assert bindings['training_fingerprint']==train.implementation_fingerprint()
    digest=hashlib.sha256()
    for section in ('wrapped','encoder'):
        for name,value in sorted(payload[section+'_state'].items()):
            assert torch.isfinite(value).all()
            digest.update((section+'.'+name).encode());digest.update(value.contiguous().numpy().tobytes())
    assert digest.hexdigest()!=bindings['initial_model_state_sha256']
    families={'context_encoder','graph','native_core','integration'}
    credit=[]
    for row in history[:2]:
        assert set(row['family_gradient_norms'])==families
        assert set(row['family_parameter_delta_l2'])==families
        values=[row['loss_to_emitted_component_spikes_gradient_l2'],row['gradient_norm_before_clip'],*row['family_gradient_norms'].values(),*row['family_parameter_delta_l2'].values()]
        assert all(math.isfinite(float(v)) and float(v)>0 for v in values)
        credit.append({key:row[key] for key in ('update','image_indices','image_ids','mean_registered_loss','mean_q_loss','loss_to_emitted_component_spikes_gradient_l2','family_gradient_norms','family_parameter_delta_l2')})
    steps=[]
    for state in payload['optimizer_state']['state'].values():
        for value in state.values():
            if torch.is_tensor(value):assert torch.isfinite(value).all()
        steps.append(int(state['step']))
    assert steps and min(steps)==max(steps)==completed
    records.append(dict(arm=arm,completed_updates=completed,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),model_state_sha256=digest.hexdigest(),first_two=credit,Adam_step_range=[min(steps),max(steps)]))
assert not torch.cuda.is_initialized()
report=dict(status='SW0178_both_actual_first_two_commits_verified',arms=records,cuda_initialized=False,validation_data_read=False)
private=Path('/Data0/kevinswk/patch_v2_sw/collaborative_testing/collaborative_private_runtime_20261008/SW0178_delivery_20261016')
(private/'first_two_verified.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
