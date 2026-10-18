"""CPU-only actual-Q readout ablation; native graph dynamics remain intact."""
import io,json,os,time
from pathlib import Path
import numpy as np
from collaborative_test.SW_0217_native64_energy_refinement import run as source
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as native

HERE=Path(__file__).resolve().parent;ARCHIVE=HERE/'results_archive'
ARMS=('baseline','Q_only');sha=source.sha;write_json=source.write_json


def fingerprint():
    return dict(files={n:sha((HERE/n).read_bytes()) for n in ('DESIGN_KO.md','run.py','test_run.py')},
        parent=source.fingerprint(),readout_library_sha256=sha(Path(native.__file__).read_bytes()),
        parent_summary_sha256=sha((source.ARCHIVE/'score_v1/summary.json').read_bytes()),
        seeds=list(source.SEEDS),offsets=list(source.OFFSETS),coupling_weights=[1,0],core_forwards=0,optimizer_updates=0)


def decode(qf,af,gain,baseline):
    labels=np.asarray(baseline);shape=tuple(labels.shape);f=np.flatnonzero(labels.reshape(-1)>0)
    if labels.ndim!=2 or qf.shape!=(len(f),len(f)) or af.shape!=qf.shape:raise ValueError('exact induced evidence and grid required')
    n=labels.size;q=np.full((n,n),.5,np.float64);a=np.zeros((n,n),np.float64)
    q[np.ix_(f,f)]=qf;a[np.ix_(f,f)]=af
    old=native.foreground_induced_merge(q,a,gain,labels,grid_shape=shape)
    replay=np.asarray(old['labels'],dtype=np.int64);del old
    if not np.array_equal(replay,labels):raise ValueError('actual coupling-weighted production baseline failed bitwise replay')
    a.fill(0.)
    candidate=native.foreground_induced_merge(q,a,gain,labels,grid_shape=shape)
    result=np.asarray(candidate['labels'],dtype=np.int64)
    if not np.array_equal(result>0,labels>0):raise ValueError('source foreground changed')
    return result


def predict():
    # Validate every parent endpoint before this ablation, without reading GT.
    _,_,bindings=source.validated_predictions();science=fingerprint()
    for seed in source.SEEDS:
        records=[];rows={arm:[] for arm in ARMS}
        for index in range(16):
            origin=source.archive_for(seed)/'rows'/f'{index:04d}'
            data,_=source.load_row(origin,source.row_binding(bindings[seed],index))
            folder=ARCHIVE/f'seed{seed}/rows'/f'{index:04d}';started=time.perf_counter()
            expected=dict(fingerprint=science,seed=seed,index=index,image_id=1320+source.OFFSETS[index],
                source_row_sha256=sha((origin/'row.json').read_bytes()))
            if folder.exists():
                meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'labels.npz').read_bytes()
                if meta['binding']!=expected or sha(raw)!=meta['labels_sha256']:raise ValueError('cached CPU ablation changed')
                with np.load(io.BytesIO(raw),allow_pickle=False) as saved:values={arm:saved[arm].copy() for arm in ARMS}
            else:
                candidate=decode(data['q_foreground'],data['a_foreground'],float(data['gain']),data['baseline'])
                values=dict(baseline=data['baseline'],Q_only=candidate)
                stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
                with (stage/'labels.npz').open('xb') as stream:np.savez_compressed(stream,**values);stream.flush();os.fsync(stream.fileno())
                write_json(stage/'row.json',dict(binding=expected,labels_sha256=sha((stage/'labels.npz').read_bytes()),seconds=time.perf_counter()-started));stage.rename(folder)
            if not np.array_equal(values['baseline'],data['baseline']):raise ValueError('wrong source baseline')
            for arm in ARMS:rows[arm].append(values[arm])
            records.append(sha((folder/'row.json').read_bytes()))
        protocol=dict(fingerprint=science,seed=seed,row_metadata_sha256=records,
            labels_sha256={arm:sha(np.stack(value).tobytes()) for arm,value in rows.items()},DEV_GT_read=False,core_forwards=0,optimizer_updates=0)
        path=ARCHIVE/f'seed{seed}/protocol.json'
        if path.exists():
            if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen ablation protocol changed')
        else:write_json(path,protocol)


def validated_predictions():
    results={};science=fingerprint()
    for seed in source.SEEDS:
        protocol=json.loads((ARCHIVE/f'seed{seed}/protocol.json').read_bytes())
        if protocol['fingerprint']!=science or protocol['seed']!=seed or protocol['DEV_GT_read'] is not False:raise ValueError('CPU protocol invalid')
        rows={arm:[] for arm in ARMS};records=[]
        for index in range(16):
            folder=ARCHIVE/f'seed{seed}/rows'/f'{index:04d}';meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'labels.npz').read_bytes()
            origin=source.archive_for(seed)/'rows'/f'{index:04d}'
            expected=dict(fingerprint=science,seed=seed,index=index,image_id=1320+source.OFFSETS[index],source_row_sha256=sha((origin/'row.json').read_bytes()))
            if meta['binding']!=expected or sha(raw)!=meta['labels_sha256']:raise ValueError('CPU row identity changed')
            with np.load(io.BytesIO(raw),allow_pickle=False) as saved:
                for arm in ARMS:
                    value=saved[arm].copy()
                    if value.dtype!=np.int64 or value.shape!=(64,64) or (value<0).any():raise ValueError('invalid ablation mask')
                    rows[arm].append(value)
            if not np.array_equal(rows['baseline'][-1]>0,rows['Q_only'][-1]>0):raise ValueError('support changed')
            records.append(sha((folder/'row.json').read_bytes()))
        results[seed]={arm:np.stack(values) for arm,values in rows.items()}
        if records!=protocol['row_metadata_sha256'] or any(sha(v.tobytes())!=protocol['labels_sha256'][arm] for arm,v in results[seed].items()):raise ValueError('ablation population changed')
    return results


def score():
    predictions=validated_predictions() # all96 masks before GT
    original,_,_=source.validated_predictions()
    for seed in source.SEEDS:
        if not np.array_equal(predictions[seed]['baseline'],original[seed]['baseline']):raise ValueError('matched source masks changed')
    _,binding=source.source.full.validated_all();gt=source.source.full.read_gt(binding)[list(source.OFFSETS)]
    from collaborative_test.SW_0211_spike_anchor_center_assignment.run import paired
    vectors={};errors=[]
    for seed in source.SEEDS:
        vectors[seed]={}
        for arm in ARMS:
            vectors[seed][arm],error=source.source.full.metrics.measured_vectors(predictions[seed][arm],gt);errors.append(error)
    means={arm:np.stack([vectors[s][arm] for s in source.SEEDS]).mean(0) for arm in ARMS}
    comparison=paired(means['Q_only'],means['baseline'])
    result=dict(status='complete_three_seed_native64_Q_readout_ablation_mini',seeds=list(source.SEEDS),images=16,grid=[64,64],
        scores={str(s):{arm:dict(zip(source.source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in vectors[s].items()} for s in source.SEEDS},
        three_seed_mean={arm:dict(zip(source.source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in means.items()},
        paired_mean_Q_only_minus_baseline=comparison,
        per_seed_paired={str(s):paired(vectors[s]['Q_only'],vectors[s]['baseline']) for s in source.SEEDS},
        expansion_gate_passed=bool(comparison['fg_ari']['mean']>=.01 and comparison['fg_ari']['ci95'][0]>0 and comparison['matched_object_iou']['mean']>=0),
        independent_maximum_metric_abs_error=max(errors),source_foreground_preserved=True,original_baseline_bitwise_replayed=True,
        native_graph_dynamics_unchanged=True,not_a_physical_gate_ablation=True,core_forwards=0,optimizer_updates=0,
        full320_FG090_goal_verified=False,new70000_training=False,validation_reused_not_untouched_test=True)
    write_json(ARCHIVE/'score_v1/summary.json',result);return result


if __name__=='__main__':predict();print(json.dumps(score()),flush=True)
