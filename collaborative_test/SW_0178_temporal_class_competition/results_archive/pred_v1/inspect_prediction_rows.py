import hashlib
import json
import time
import numpy as np
from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as evaluate

rows=[]
rgb=None
refs=None
for arm in evaluate.ARMS:
    work=evaluate._work(arm)
    folders=sorted((work/'rows').glob('[0-9][0-9][0-9][0-9]'))
    if not folders:
        rows.append(dict(arm=arm,committed_rows_observed=0,checked=[]));continue
    wrapper=json.loads((work/'bindings.json').read_text())
    binding=wrapper['bindings'];digest=evaluate._json_sha(binding)
    assert wrapper['bindings_sha256']==digest
    assert binding['arm']==arm and binding['implementation_fingerprint']==evaluate.implementation_fingerprint()
    assert binding['image_ids']==evaluate.IMAGE_IDS.tolist()
    assert binding['ground_truth_used_for_prediction'] is False
    if rgb is None:
        rgb=np.load(evaluate.sw130.VAL_RGB,mmap_mode='r',allow_pickle=False)
        refs=evaluate._references()
    checked=[]
    for folder in sorted(set(folders[:1]+folders[-1:])):
        index=int(folder.name)
        rgb_sha=hashlib.sha256(np.ascontiguousarray(rgb[index]).tobytes()).hexdigest()
        labels,support,row=evaluate._read_row(work,index,int(evaluate.IMAGE_IDS[index]),rgb_sha,digest)
        assert np.array_equal(labels['conditional']>0,refs['source_T']['labels'][index]>0)
        assert np.array_equal(labels['standalone']>0,support>0)
        checked.append(dict(index=index,image_id=row['image_id'],row_archive_sha256=row['npz_sha256'],actual_native_Q_and_A_digests_valid=True,conditional_SourceT_support_verified=True,standalone_native_QCC_support_verified=True))
    rows.append(dict(arm=arm,committed_rows_observed=len(folders),checked=checked))
assert not evaluate.torch.cuda.is_initialized()
report=dict(status='actual_committed_rows_verified' if all(row['checked'] for row in rows) else 'waiting_for_committed_rows',observed_unix=time.time(),arms=rows,validation_ground_truth_read=False,optimizer_updates=0,model_forward_performed=False,cuda_initialized=False)
output=evaluate.ROOT/'collaborative_testing/collaborative_private_runtime_20261008/SW0178_evaluation_delivery_20261016/rows_verified.json'
output.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
