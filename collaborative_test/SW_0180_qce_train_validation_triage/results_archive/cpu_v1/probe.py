import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0179_qce_seed_replication import train,predict_evaluate as pe
from collaborative_test.SW_0178_temporal_class_competition import train as t178,predict_evaluate as p178
root=Path('/Data0/kevinswk/patch_v2_sw')
private=root/'collaborative_testing/collaborative_private_runtime_20261008/SW0180_delivery_20261016'
private.mkdir(parents=True,exist_ok=True)
inputs=train.prepare_training_inputs_cpu()
positions=list(range(256,264))
rgb=np.ascontiguousarray(inputs['arrays']['rgb'][positions])
gt=np.ascontiguousarray(inputs['arrays']['modal_gt'][positions])
ids=np.asarray(inputs['ordered_train_ids'],dtype=np.int64)[positions]
assert rgb.shape==(8,128,128,3) and gt.shape==(8,32,32) and len(set(ids.tolist()))==8
reference=json.loads((root/'collaborative_testing/collaborative_private_runtime_20261008/SW0179_delivery_20261016/cpu_receipt.json').read_text())
records=[]
for seed in (0,1,2):
    source,optimizer,families,binding,setup=train._make_fresh('product_q_ce',seed,inputs,'cpu')
    initial=train._state_sha(source.wrapped,source.encoder)
    expected=next(row for row in reference['arms'] if row['seed']==seed)
    assert initial==expected['initial_model_state_sha256'] and not optimizer.state
    with torch.no_grad():
        contextual=source.encode(torch.from_numpy(rgb))
        context=source.encoder;source.encoder=setup[2]
        try:plain=source.encode(torch.from_numpy(rgb))
        finally:source.encoder=context
    assert torch.equal(contextual,plain) and train._state_sha(source.wrapped,source.encoder)==initial
    if seed==1:
        final_inputs=t178.prepare_training_inputs_cpu()
        directory=t178.OUTPUT_ROOT/'product_q_ce_seed1'
        manifest=t178.validate_training_artifact('product_q_ce',directory)
        prepared=dict(training_inputs=final_inputs,artifacts={'product_q_ce':directory},manifests={'product_q_ce':manifest})
        final,final_sha=p178._load_model('product_q_ce',prepared,'cpu')
    else:
        directory=train.OUTPUT_ROOT/f'product_q_ce_seed{seed}'
        manifest=train.validate_training_artifact('product_q_ce',directory,seed=seed)
        prepared=dict(seed=seed,training_inputs=inputs,artifacts={'product_q_ce':directory},manifests={'product_q_ce':manifest})
        final,final_sha=pe._load_model('product_q_ce',seed,prepared,'cpu')
    assert final_sha==manifest['final_model_state_sha256'] and final_sha!=initial
    records.append(dict(seed=seed,fresh_model_state_sha256=initial,final_model_state_sha256=final_sha,final_manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),source_binding=binding['source_binding'],zero_context_gamma_exact_on_registered_TRAIN8=True))
    del source,optimizer,families,binding,setup,contextual,plain,context,final
assert not torch.cuda.is_initialized()
report=dict(status='SW0180_actual_CPU_fresh_final_sources_and_registered_TRAIN8_passed',positions=positions,image_ids=ids.tolist(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),modal_gt_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),records=records,cuda_initialized=False,optimizer_updates=0,validation_images_or_ground_truth_read=False,actual_spike_forward_verified=False,observed_unix=time.time())
with (private/'cpu_source_probe.json').open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report),flush=True)
