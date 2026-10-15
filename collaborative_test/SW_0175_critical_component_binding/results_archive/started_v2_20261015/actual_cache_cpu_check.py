import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
root=Path('/Data0/kevinswk/patch_v2_sw');sys.path.insert(0,str(root))
tests=['collaborative_test.SW_0175_critical_component_binding.'+name for name in
       ('test_critical_component_objective','test_train_v2','test_training_queue_v2')]
result=subprocess.run([sys.executable,'-m','unittest',*tests,'-v'],cwd=root,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)
if result.returncode:
    print(json.dumps(dict(status='CPU_tests_failed',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
    sys.exit(result.returncode)
from collaborative_test.SW_0175_critical_component_binding import train_v2 as train
from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as loss_api
admitted=train._load_admitted_inputs();arrays=admitted['prepared']['arrays']
assert not train.torch.cuda.is_initialized()
rows=[]
for position in (0,1,2,3,256,257,258,259):
    support=arrays['source_support'][position];target=arrays['feasible_targets'][position]
    before=support.copy();expected=np.asarray(support)>0
    nodes=support.size
    rho=train.torch.full((4,nodes,nodes),.5,requires_grad=True)
    q=rho.clamp_min(0).prod(dim=0)
    checks={}
    for arm in train.ARMS:
        actual=train._loss_for_arm(arm,q,rho,target,support)
        boolean_actual=train._loss_for_arm(arm,q,rho,target,expected)
        reference=(loss_api.native_q_loss(q,target,expected) if arm=='native_Q' else loss_api.critical_component_loss(q,rho,target,expected))
        assert train.torch.equal(actual,reference) and train.torch.equal(boolean_actual,reference)
        gradient=train.torch.autograd.grad(actual,rho,retain_graph=True)[0]
        assert train.torch.isfinite(actual) and train.torch.isfinite(gradient).all()
        checks[arm]=dict(loss=float(actual.detach()),raw_labels_equal_explicit_boolean=True,finite_gradient=True)
    assert train.torch.equal(train._native_q_loss(q,target,support),loss_api.native_q_loss(q,target,expected))
    assert np.array_equal(support,before)
    rows.append(dict(position=position,support_dtype=str(support.dtype),support_max=int(support.max()),arms=checks))
assert any(row['support_max']>1 for row in rows),'did not exercise real multiclass cache support'
assert not train.torch.cuda.is_initialized()
print(json.dumps(dict(status='V2_CPU_tests_and_actual_cached_caller_contract_passed',returncode=0,
    tests_stderr=result.stderr,test_count=int(re.search(r'Ran (\d+) tests',result.stderr).group(1)),
    source_proof=admitted['proof'],training_fingerprint=train._training_fingerprint(),
    source_admission_report_sha256=admitted['admission_report_sha256'],actual_cache_caller_rows=rows,
    CUDA_initialized=False,validation_data_read=False,optimizer_updates=0)))
