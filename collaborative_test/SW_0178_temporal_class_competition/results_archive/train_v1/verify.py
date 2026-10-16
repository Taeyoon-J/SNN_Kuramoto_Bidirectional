import collections
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0178_temporal_class_competition import train,training_queue

records=[]
for arm in train.ARMS:
    state=json.loads(training_queue._state_path(arm).read_text())
    if state['status']!='complete':
        pid=state.get('supervisor_pid')
        argv=Path('/proc',str(pid),'cmdline').read_bytes().split(bytes([0]))
        records.append(dict(arm=arm,status=state['status'],supervisor_pid=pid,live=any(argv)))
        continue
    artifact=train.OUTPUT_ROOT/(arm+'_seed1')
    manifest=train.validate_training_artifact(arm,artifact)
    manifest_sha=hashlib.sha256((artifact/'manifest.json').read_bytes()).hexdigest()
    assert state['manifest_sha256']==manifest_sha and state['completed_updates']==64
    history=json.loads((artifact/'history.json').read_text())['updates']
    ordered=manifest['bindings']['ordered_train_ids']
    assert len(ordered)==len(set(ordered))==512
    assert len(history)==64 and manifest['exposures']==1024
    for offset in (0,32):
        ids=[image for row in history[offset:offset+32] for image in row['image_ids']]
        assert collections.Counter(ids)==collections.Counter(ordered)
    checkpoint=torch.load(artifact/'checkpoint.pt',map_location='cpu',weights_only=False)
    digest=hashlib.sha256()
    for section in ('wrapped','encoder'):
        for name,value in sorted(checkpoint[section].items()):
            assert torch.isfinite(value).all()
            digest.update((section+'.'+name).encode());digest.update(value.contiguous().numpy().tobytes())
    assert digest.hexdigest()==manifest['final_model_state_sha256']
    assert digest.hexdigest()!=manifest['initial_model_state_sha256']
    optimizer=torch.load(artifact/'optimizer.pt',map_location='cpu',weights_only=False)
    steps=[int(values['step']) for values in optimizer['state'].values()]
    assert steps and min(steps)==max(steps)==64
    raw=(artifact/'boundary.ptx').read_bytes()
    outer=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    data=outer['payload_bytes'];assert hashlib.sha256(data).hexdigest()==outer['payload_sha256']
    boundary=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    assert boundary['completed_updates']==64 and boundary['history']==history
    for section in ('wrapped','encoder'):
        assert set(boundary[section+'_state'])==set(checkpoint[section])
        assert all(torch.equal(value,checkpoint[section][name]) for name,value in boundary[section+'_state'].items())
    records.append(dict(arm=arm,status='actual_64_update_artifact_verified',manifest_sha256=manifest_sha,model_state_sha256=digest.hexdigest(),unique_train_images=512,exposures=1024,Adam_step_range=[min(steps),max(steps)],initial_first_batch_loss=history[0]['mean_registered_loss'],last_batch_loss=history[-1]['mean_registered_loss']))
assert not torch.cuda.is_initialized()
report=dict(observed_unix=time.time(),status='both_training_artifacts_verified' if all(row['status']=='actual_64_update_artifact_verified' for row in records) else 'waiting_on_verified_live_training',arms=records,cuda_initialized=False,validation_ground_truth_read=False,optimizer_updates=0)
private=Path('/Data0/kevinswk/patch_v2_sw/collaborative_testing/collaborative_private_runtime_20261008/SW0178_delivery_20261016')
(private/'completed_validation_latest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
