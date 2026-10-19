"""CPU-only existing-mask repeat diagnostic; no new training or inference."""

def main():

    import ctypes,os,json
    from pathlib import Path
    assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    os.chdir('/Data0/kevinswk/patch_v2_sw')
    import numpy as np
    from collaborative_test.SW_0244_native64_within_precision import run as old
    from collaborative_test.SW_0245_native64_precision_three_seed64 import run as new

    def canonical(x):
     x=np.asarray(x);ids,first,inverse=np.unique(x.ravel(),return_index=True,return_inverse=True);m=np.zeros(len(ids),np.int64);pos=np.flatnonzero(ids>0);m[pos[np.argsort(first[pos])]]=np.arange(1,len(pos)+1);return m[inverse].reshape(x.shape)
    assert np.array_equal(canonical([[0,3,3],[8,0,8]]),canonical([[0,6,6],[2,0,2]]))
    assert not np.array_equal(canonical([[0,3,3],[8,0,8]]),canonical([[0,6,6],[6,0,6]]))
    rows={};dataset=None
    for arm in ('control','within16'):
     a=old.validated_seed(1,arm);b=new.validated_seed(1,arm);indices=[new.OFFSETS.index(i) for i in old.OFFSETS];left=a['labels'];right=b['labels'][indices]
     for key in ('TRAIN_ids','TRAIN_RGB_sha256','TRAIN_labels_sha256','teacher_arrays_sha256','initial_model_sha256','frozen_core_sha256','input_statistics_sha256','dataset'):assert a['binding'][key]==b['binding'][key]
     dataset=a['binding']['dataset'];stat=Path(dataset['path']).stat();assert (stat.st_size,stat.st_mtime_ns)==(dataset['size'],dataset['mtime_ns'])
     cases=[dict(bitwise_identical=bool(np.array_equal(x,y)),partition_identical_after_ID_normalization=bool(np.array_equal(canonical(x),canonical(y))),foreground_support_changed_patches=int(np.count_nonzero((x>0)!=(y>0)))) for x,y in zip(left,right)]
     rows[arm]=dict(cases=cases,old_masks=left,new_masks=right)
    # Both complete old/new protocols precede any evaluation GT access.
    gt=old.reference.source.source.full.read_gt(dict(dataset=dataset))[list(old.OFFSETS)];keys=old.reference_score.METRICS;maximum=0.;result={}
    for arm,row in rows.items():
     oldv,e=old.reference.source.source.full.metrics.measured_vectors(row['old_masks'],gt);newv,e2=old.reference.source.source.full.metrics.measured_vectors(row['new_masks'],gt);maximum=max(maximum,float(e),float(e2))
     result[arm]=dict(case_count=8,bitwise_identical_mask_cases=sum(v['bitwise_identical'] for v in row['cases']),partition_identical_cases_after_ID_normalization=sum(v['partition_identical_after_ID_normalization'] for v in row['cases']),total_foreground_support_changed_patches=sum(v['foreground_support_changed_patches'] for v in row['cases']),old_eight_image_mean=dict(zip(keys,oldv.mean(0).tolist())),repeated_eight_image_mean=dict(zip(keys,newv.mean(0).tolist())),repeat_minus_old_eight_image_mean=dict(zip(keys,(newv-oldv).mean(0).tolist())))
    assert maximum<1e-12
    print(json.dumps(dict(status='completed_existing_mask_repeat_semantic_audit',seed_scope=[1],image_count=8,arms=result,independent_metric_maximum_abs_error=maximum,optimizer_updates=0,new_native_forwards=0,new_70000_result=False,no_numerical_cause_inferred=True)))

if __name__=='__main__':main()
