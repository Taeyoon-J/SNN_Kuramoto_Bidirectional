"""Describe frozen frequency teacher targets; these are not segmentation scores."""
import numpy as np


def describe(frequency,labels):
    f=np.asarray(frequency);y=np.asarray(labels)
    if y.ndim!=3 or y.dtype.kind not in 'iu' or np.any(y<0) or f.shape!=(len(y),int(np.prod(y.shape[1:])),4) or f.dtype.kind!='f' or not np.isfinite(f).all():
        raise ValueError('finite aligned four-component frequency and integer masks required')
    cases=[]
    for values,grid in zip(f,y):
        flat=grid.reshape(-1);objects=np.unique(flat[flat>0]);centers=[];variance=[];sizes=[]
        for k in objects:
            points=values[flat==k];center=points.mean(0)
            centers.append(center);sizes.append(len(points))
            variance.append(float(np.square(points.astype(np.float64)-center).sum(1).mean()))
        if not len(objects):
            cases.append(dict(foreground_objects=0,foreground_patches=0,patch_accuracy=None,object_balanced_accuracy=None,nearest_center_distances=[],nearest_center_to_within_scale_ratios=[],zero_within_scale_pairs=0));continue
        centers=np.asarray(centers);take=flat>0
        distances=np.square(values[take,None,:].astype(np.float64)-centers[None,:,:]).sum(2)
        predicted=objects[distances.argmin(1)];truth=flat[take]
        balanced=float(np.mean([np.mean(predicted[truth==k]==k) for k in objects]))
        nearest=[];ratios=[];zero=0
        if len(objects)>1:
            d=np.square(centers[:,None,:].astype(np.float64)-centers[None,:,:]).sum(2)
            np.fill_diagonal(d,np.inf)
            for i,j in enumerate(d.argmin(1)):
                distance=float(np.sqrt(d[i,j]));scale=float(np.sqrt(variance[i]+variance[j]));nearest.append(distance)
                if scale>0:ratios.append(distance/scale)
                else:zero+=1
        cases.append(dict(foreground_objects=len(objects),foreground_patches=int(take.sum()),
            patch_accuracy=float(np.mean(predicted==truth)),object_balanced_accuracy=balanced,
            nearest_center_distances=nearest,nearest_center_to_within_scale_ratios=ratios,zero_within_scale_pairs=zero))
    return cases


def aggregate(cases):
    if not cases:raise ValueError('nonempty actual TRAIN cases required')
    def quantiles(values):
        return dict(zip(('min','q25','median','q75','max'),map(float,np.percentile(values,[0,25,50,75,100])))) if values else None
    multi=[r for r in cases if r['foreground_objects']>1]
    return dict(image_count=len(cases),multiple_foreground_object_image_count=len(multi),
        teacher_assisted_TRAIN_patch_accuracy_quantiles=quantiles([r['patch_accuracy'] for r in multi]),
        teacher_assisted_TRAIN_object_balanced_accuracy_quantiles=quantiles([r['object_balanced_accuracy'] for r in multi]),
        nearest_foreground_center_distance_quantiles=quantiles([v for r in cases for v in r['nearest_center_distances']]),
        center_distance_to_within_object_scale_ratio_quantiles=quantiles([v for r in cases for v in r['nearest_center_to_within_scale_ratios']]),
        zero_within_scale_pairs=sum(r['zero_within_scale_pairs'] for r in cases),
        segmentation_evaluation_score=False,teacher_centers_use_TRAIN_GT=True)


def zero_loss_collision_example():
    import torch
    from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher,teacher_objective
    gamma=np.zeros((1,8,4096),dtype=np.float32)
    frequency=np.full((1,4096,4),.5,dtype=np.float32)
    labels=np.zeros((1,64,64),dtype=np.int64);labels[:,8:24,8:24]=1;labels[:,40:56,40:56]=2
    targets=make_teacher(gamma,frequency,labels)
    loss,parts=teacher_objective(torch.from_numpy(gamma),torch.from_numpy(frequency),targets)
    return dict(actual_registered_teacher_loss=float(loss),frequency_denoising=float(parts['frequency_denoising']),
        gamma_anchor=float(parts['gamma_anchor']),distinct_foreground_object_count=2,foreground_frequency_centers_identical=True,
        controlled_counterexample_not_actual_dataset_performance=True)
