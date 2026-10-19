"""CPU-only frozen teacher diagnostic on a preregistered actual TRAIN sample."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0239_native64_best_70000 import train as original
from collaborative_test.SW_0240_native64_70000_cached_io import data
from .statistics import describe,aggregate,zero_loss_collision_example

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive'
SEEDS=(0,1,2);COUNT=128


def fingerprint():
    return dict(files={n:original.sha((HERE/n).read_bytes()) for n in ('DESIGN_KO.md','run.py','statistics.py','test_statistics.py')},
        source239=original.fingerprint(),cache_manifest_sha256=data.cache_binding()['manifest_sha256'],
        seeds=list(SEEDS),TRAIN_images_per_seed=COUNT,sample_seed=242000,TRAIN_GT_teacher=True,
        evaluation_GT_read=False,evaluation_inputs_used_in_statistics=False,shared_restore_may_load_validation_RGB_and_prior_predictions=True,optimizer_updates=0,GPU_used=False)


def sample_ids():
    return data.train_ids()[np.random.RandomState(242000).choice(data.TRAIN_COUNT,COUNT,replace=False)]


def validated_seed(seed):
    folder=ARCHIVE/f'seed{seed}';result=json.loads((folder/'summary.json').read_bytes())
    if result['binding']['fingerprint']!=fingerprint() or result['binding']['seed']!=seed:
        raise ValueError('registered source or seed changed')
    raw=(folder/'teacher_vectors.npz').read_bytes()
    if original.sha(raw)!=result['teacher_vectors_sha256']:raise ValueError('retained vectors changed')
    with np.load(folder/'teacher_vectors.npz',allow_pickle=False) as archive:
        if not np.array_equal(archive['TRAIN_ids'],sample_ids()):raise ValueError('actual TRAIN sample changed')
        cases=describe(archive['frequency'],archive['labels'])
    if cases!=result['cases'] or aggregate(cases)!=result['aggregate']:raise ValueError('raw vectors do not reproduce descriptive statistics')
    return result


def run(seed):
    original.require_seed(seed)
    if torch.cuda.is_initialized():raise ValueError('CPU-only frozen teacher diagnostic required')
    folder=ARCHIVE/f'seed{seed}'
    if (folder/'summary.json').exists():return validated_seed(seed)
    teacher,_,_,parent=original.source.prepare_cpu(seed);ids=sample_ids()
    teacher.wrapped.requires_grad_(False);teacher.encoder.requires_grad_(False)
    teacher.wrapped.eval();teacher.encoder.eval()
    cached=data.cache_binding()
    if any(cached['dataset'][k]!=parent['dataset'][k] for k in ('path','size','mtime_ns')):
        raise ValueError('actual cache and frozen source dataset differ')
    before=original.source.source.full.screen.native._state_sha(teacher.wrapped,teacher.encoder)
    if before!=original.source.MODELS[str(seed)]:raise ValueError('frozen teacher restoration differs')
    rgb,labels=data.read_batch(None,ids);frequencies=[]
    with torch.no_grad():
        for offset in range(0,COUNT,16):
            gamma=teacher.encode(torch.from_numpy(rgb[offset:offset+16]))
            frequency=original.helpers.effective_frequency(teacher.wrapped.core,gamma)
            frequencies.append(frequency.cpu().numpy().copy())
    frequency=np.concatenate(frequencies);cases=describe(frequency,labels)
    if original.source.source.full.screen.native._state_sha(teacher.wrapped,teacher.encoder)!=before or torch.cuda.is_initialized():
        raise ValueError('read-only CPU diagnostic mutated model or initialized CUDA')
    folder.mkdir(parents=True,exist_ok=True)
    original.helpers.save_arrays(folder/'teacher_vectors.npz',frequency=frequency,labels=labels,TRAIN_ids=ids)
    result=dict(status='complete_actual_TRAIN_frozen_teacher_separation_diagnostic',
        binding=dict(fingerprint=fingerprint(),seed=seed,teacher_model_sha256=before,original239_teacher_source_binding=parent,
            RGB_sha256=original.sha(rgb.tobytes())),teacher_vectors_sha256=original.sha((folder/'teacher_vectors.npz').read_bytes()),
        cases=cases,aggregate=aggregate(cases),optimizer_updates=0,GPU_used=False,evaluation_GT_read=False,evaluation_inputs_used_in_statistics=False,shared_restore_may_load_validation_RGB_and_prior_predictions=True)
    original.write_json(folder/'summary.json',result);return validated_seed(seed)


def finish():
    rows=[validated_seed(seed) for seed in SEEDS]
    result=dict(status='complete_three_seed_actual_TRAIN_teacher_separation_audit',
        fingerprint=fingerprint(),per_seed={str(seed):r['aggregate'] for seed,r in zip(SEEDS,rows)},
        pooled=aggregate([case for r in rows for case in r['cases']]),
        controlled_loss_collision_example=zero_loss_collision_example(),
        actual_unique_TRAIN_images=COUNT,seed_image_cases=COUNT*len(SEEDS),
        all_raw_vectors_independently_recomputed=True,optimizer_updates=0,GPU_used=False,
        segmentation_evaluation_score=False,performance_improvement_claim=False,
        explains_observed_large_data_regression_proven=False,current_full320_incumbent_unchanged=True)
    path=ARCHIVE/'diagnostic_v1/summary.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('completed diagnostic differs')
    else:original.write_json(path,result)
    return result


if __name__=='__main__':
    import ctypes,sys
    if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP disable failed')
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--aggregate',action='store_true');args=parser.parse_args()
    if args.aggregate:print(json.dumps(finish()))
    elif args.seed is not None:
        result=run(args.seed);print(json.dumps(dict(status=result['status'],seed=args.seed,aggregate=result['aggregate'])))
    else:parser.error('choose a registered seed or final aggregate')
