"""Native full-energy refinement of three independently trained checkpoints."""
import io,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0210_native64_full_seed_reference import run as source
from collaborative_test.SW_0162_connected_signed_partition import connected_signed as signed
from . import refine as operator

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ARCHIVE=HERE/'results_archive'
SEEDS=(0,1,2);OFFSETS=tuple(range(0,32,2));ARMS=('baseline','split','refined')
MODELS={'0':source.full.MODELS['warm192'],**source.MODELS}
sha=source.sha;write_json=source.write_json


def require_seed(seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('independent model seed0/1/2 required')
    return seed


def archive_for(seed):return ARCHIVE/f'seed{require_seed(seed)}'


def fingerprint():
    names=('DESIGN_KO.md','refine.py','run.py','queue.py','test_refine.py','test_pipeline.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},source_sha256=sha(json.dumps(source.fingerprint(),sort_keys=True).encode()),
        energy_operator_sha256=sha(Path(signed.__file__).read_bytes()),models=MODELS,seeds=list(SEEDS),offsets=list(OFFSETS),new_optimizer_steps=0)


def prepare_cpu(seed):
    require_seed(seed)
    if torch.cuda.is_initialized():raise ValueError('native CPU admission before CUDA')
    if seed==0:
        prepared=source.full.screen.prepare_inputs_cpu();found=prepared['found'];state=prepared['candidate_state']
        found.wrapped.load_state_dict(state['wrapped'],strict=True);found.encoder.load_state_dict(state['encoder'],strict=True)
        rgb=prepared['rgb'][list(OFFSETS)].copy();all_labels,original=source.full.validated_all();mini=all_labels['warm192'][list(OFFSETS)].copy()
        parent_binding=prepared['bindings'];dataset=original['dataset']
    else:
        found,full_rgb,parent_binding,mini=source.prepare_cpu(seed);rgb=full_rgb[list(OFFSETS)].copy();dataset=parent_binding['dataset']
    if source.full.screen.native._state_sha(found.wrapped,found.encoder)!=MODELS[str(seed)]:raise ValueError('actual source model SHA mismatch')
    if rgb.shape!=(16,128,128,3) or rgb.dtype!=np.uint8 or mini.shape!=(16,64,64):raise ValueError('same16 native geometry required')
    counts=[int((row>0).sum()) for row in mini]
    # Worst case retains both actual matrices as float64, plus masks/indices.
    row_bytes=[16*c*c+3*4096*8+8*c for c in counts];required=sum(row_bytes)
    remaining=sum(size for index,size in enumerate(row_bytes) if not (archive_for(seed)/'rows'/f'{index:04d}'/'data.npz').is_file())
    if shutil.disk_usage(ARCHIVE).free<remaining+1024**3:raise OSError('insufficient remaining private-row disk reserve before GPU execution')
    binding=dict(fingerprint=fingerprint(),seed=seed,model_sha256=MODELS[str(seed)],parent_binding_sha256=sha(json.dumps(parent_binding,sort_keys=True).encode()),
        RGB_sha256=sha(rgb.tobytes()),RGB_rows_sha256=[sha(v.tobytes()) for v in rgb],mini_labels_sha256=sha(mini.tobytes()),
        dataset=dataset,private_bytes_worst_case_this_seed=required,DEV_GT_read=False)
    return found,rgb,mini,binding


def admit(seed):
    _,_,mini,binding=prepare_cpu(seed)
    if any(any(len(operator.connected(g,(64,64)))!=1 for g in operator.groups_of(row)) for row in mini):raise ValueError('native reference lost4-connectivity')
    result=dict(status='actual_native64_source_and_private_disk_CPU_admitted',binding=binding,cuda_initialized=False)
    write_json(archive_for(seed)/'admission.json',result);return result


def row_binding(binding,index):
    if type(index) is not int or index not in range(16):raise ValueError('fixed16 row identity required')
    return dict(seed=binding['seed'],index=index,image_id=1320+OFFSETS[index],inference_seed=193000+OFFSETS[index],
                RGB_sha256=binding['RGB_rows_sha256'][index],binding_sha256=sha(json.dumps(binding,sort_keys=True).encode()))


def load_row(folder,expected,*,replay=False):
    record=json.loads((folder/'row.json').read_bytes());raw=(folder/'data.npz').read_bytes()
    if record['binding']!=expected or sha(raw)!=record['data_sha256']:raise ValueError('native row provenance changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:data={key:saved[key].copy() for key in saved.files}
    for arm in ARMS:
        if data[arm].dtype!=np.int64 or data[arm].shape!=(64,64) or (data[arm]<0).any():raise ValueError('invalid native mask')
    foreground=np.flatnonzero(data['baseline'].reshape(-1)>0)
    if not np.array_equal(foreground,data['foreground']):raise ValueError('wrong induced source membership')
    if any(data[key].shape!=(len(foreground),len(foreground)) or data[key].dtype.kind!='f' or not np.isfinite(data[key]).all() for key in ('q_foreground','a_foreground')) or not np.isfinite(data['gain']).all() or float(data['gain'])<=0:raise ValueError('invalid private source matrices/gain')
    energy=signed.signed_pair_energy(data['q_foreground'].astype(np.float64),data['a_foreground'].astype(np.float64),float(data['gain'])) if len(foreground) else np.zeros((0,0),np.float64)
    if sha(energy.tobytes())!=record['foreground_energy_sha256']:raise ValueError('induced matrices do not exactly reconstruct full-N induced energy')
    def value(labels):
        selected=labels.reshape(-1)[foreground];return float(sum(float(energy[np.ix_(selected==k,selected==k)].sum())*.5 for k in np.unique(selected)))
    values={arm:value(data[arm]) for arm in ARMS};tolerance=record['diagnostic']['tolerance']
    if values['split']<values['baseline']-tolerance or values['refined']<values['split']-tolerance:raise ValueError('stored refinement decreased original energy')
    for arm in ARMS:
        if not np.array_equal(data[arm]>0,data['baseline']>0) or any(len(operator.connected(g,(64,64)))!=1 for g in operator.groups_of(data[arm])):raise ValueError('source foreground/connectivity changed')
    for arm,key in (('baseline','energy_before'),('split','energy_split'),('refined','energy_refined')):
        if abs(values[arm]-record['diagnostic'][key])>max(tolerance,1e-12):raise ValueError('saved objective differs')
    if replay:
        full=np.zeros((4096,4096),np.float64);full[np.ix_(foreground,foreground)]=energy;result=operator.refine(full,data['baseline'])
        if not np.array_equal(result['split'],data['split']) or not np.array_equal(result['refined'],data['refined']):raise ValueError('independent native optimizer replay differs')
    return data,record


def predict(seed):
    found,rgb,mini,binding=prepare_cpu(seed);archive=archive_for(seed)
    if binding!=json.loads((archive/'admission.json').read_bytes())['binding']:raise ValueError('admitted source changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    source.source.model64.move_to_device(found,device);found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    if source.full.screen.native._state_sha(found.wrapped,found.encoder)!=MODELS[str(seed)]:raise ValueError('GPU restore changed source')
    records=[];rows={arm:[] for arm in ARMS}
    for index,offset in enumerate(OFFSETS):
        expected=row_binding(binding,index);folder=archive/'rows'/f'{index:04d}'
        if folder.exists():data,record=load_row(folder,expected)
        else:
            torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset);started=time.perf_counter()
            with torch.no_grad():
                gamma,trace,adj,z,q,rho=source.full.screen.native._native_forward(found,torch.from_numpy(rgb[index:index+1]).to(device),live_tail_steps=0)
                baseline=source.full.screen.prediction_from_trace(found,trace,adj,q)
            if not np.array_equal(baseline,mini[index]):raise ValueError('native baseline did not bitwise replay')
            qraw=q[0].detach().cpu().numpy();araw=adj[0].detach().cpu().numpy();gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
            energy=signed.signed_pair_energy(qraw.astype(np.float64),araw.astype(np.float64),gain);result=operator.refine(energy,baseline);foreground=np.flatnonzero(baseline.reshape(-1)>0)
            data=dict(baseline=baseline,split=result['split'],refined=result['refined'],foreground=foreground,
                      q_foreground=qraw[np.ix_(foreground,foreground)].copy(),a_foreground=araw[np.ix_(foreground,foreground)].copy(),gain=np.asarray(gain))
            stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
            with (stage/'data.npz').open('xb') as stream:np.savez_compressed(stream,**data);stream.flush();os.fsync(stream.fileno())
            record=dict(binding=expected,data_sha256=sha((stage/'data.npz').read_bytes()),full_q_sha256=sha(qraw.tobytes()),full_a_sha256=sha(araw.tobytes()),
                full_energy_sha256=sha(energy.tobytes()),foreground_energy_sha256=sha(energy[np.ix_(foreground,foreground)].tobytes()),
                diagnostic=result['diagnostic'],seconds=time.perf_counter()-started,DEV_GT_read=False)
            write_json(stage/'row.json',record);stage.rename(folder);del gamma,trace,adj,z,q,rho,energy,result,qraw,araw
        if not np.array_equal(data['baseline'],mini[index]):raise ValueError('cached baseline source differs')
        for arm in ARMS:rows[arm].append(data[arm])
        records.append(sha((folder/'row.json').read_bytes()))
    if source.full.screen.native._state_sha(found.wrapped,found.encoder)!=MODELS[str(seed)]:raise ValueError('readout mutated actual source')
    protocol=dict(binding=binding,row_metadata_sha256=records,labels_sha256={arm:sha(np.stack(values).tobytes()) for arm,values in rows.items()},DEV_GT_read=False,optimizer_updates=0,source_unchanged=True)
    path=archive/'protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('native protocol changed')
    else:write_json(path,protocol)


def validated_seed(seed,*,replay=False):
    archive=archive_for(seed);protocol=json.loads((archive/'protocol.json').read_bytes());binding=json.loads((archive/'admission.json').read_bytes())['binding']
    if protocol['binding']!=binding or binding['fingerprint']!=fingerprint() or protocol['DEV_GT_read'] is not False or not protocol['source_unchanged']:raise ValueError('invalid native source protocol')
    rows={arm:[] for arm in ARMS};hashes=[];diag=[]
    for index in range(16):
        data,record=load_row(archive/'rows'/f'{index:04d}',row_binding(binding,index),replay=replay);diag.append(record['diagnostic']);hashes.append(sha((archive/'rows'/f'{index:04d}'/'row.json').read_bytes()))
        for arm in ARMS:rows[arm].append(data[arm])
    values={arm:np.stack(items) for arm,items in rows.items()}
    if hashes!=protocol['row_metadata_sha256'] or any(sha(v.tobytes())!=protocol['labels_sha256'][arm] for arm,v in values.items()) or sha(values['baseline'].tobytes())!=binding['mini_labels_sha256']:raise ValueError('native mask population changed')
    return values,diag,binding


def validated_predictions(*,replay=False):
    result={};diagnostics={};bindings={}
    for seed in SEEDS:result[seed],diagnostics[seed],bindings[seed]=validated_seed(seed,replay=replay)
    return result,diagnostics,bindings


def score():
    predictions,diagnostics,bindings=validated_predictions()
    _,reference=source.full.validated_all();gt=source.full.read_gt(reference)[list(OFFSETS)]
    vectors={};errors=[];scores={}
    for seed,arms in predictions.items():
        vectors[seed]={}
        for arm,pred in arms.items():vectors[seed][arm],error=source.full.metrics.measured_vectors(pred,gt);errors.append(error)
        scores[str(seed)]={arm:dict(zip(source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in vectors[seed].items()}
    means={arm:np.stack([vectors[s][arm] for s in SEEDS]).mean(0) for arm in ARMS}
    from collaborative_test.SW_0211_spike_anchor_center_assignment.run import paired
    comparisons={arm:paired(means[arm],means['baseline']) for arm in ('split','refined')};primary=comparisons['refined']
    from collaborative_test.SW_0189_frozen_error_attribution.run import errors as count_errors
    counts={}
    for seed,arms in predictions.items():
        counts[str(seed)]={}
        for arm,values in arms.items():
            rows=[count_errors(p,g) for p,g in zip(values,gt)]
            counts[str(seed)][arm]={key:float(np.mean([r[key] for r in rows])) for key in ('true_object_count','predicted_positive_label_count','raw_count_exact')}
    summary=dict(status='complete_three_seed_native64_energy_refinement_mini',scores=scores,
        three_seed_mean={arm:dict(zip(source.full.metrics.independent.METRICS,v.mean(0).tolist())) for arm,v in means.items()},
        paired_mean_minus_baseline=comparisons,per_seed_paired={str(s):{arm:paired(vectors[s][arm],vectors[s]['baseline']) for arm in ('split','refined')} for s in SEEDS},
        paired_CI_scope='paired images after averaging the three fixed models, not a population-seed confidence interval',count_diagnostics=counts,
        primary_expansion_gate_passed=bool(primary['fg_ari']['mean']>=.01 and primary['fg_ari']['ci95'][0]>0 and primary['matched_object_iou']['mean']>=0),
        seed_scope=list(SEEDS),images=16,grid=[64,64],optimizer_updates=0,completed_core_forward_rows=48,
        diagnostics={str(s):dict(accepted_splits=float(np.mean([r['accepted_source_splits'] for r in values])),remerges=float(np.mean([r['remerges'] for r in values])),energy_gain=float(np.mean([r['energy_refined']-r['energy_before'] for r in values]))) for s,values in diagnostics.items()},
        independent_maximum_metric_abs_error=max(errors),mask_supervised_lineage=True,new70000_training=False,
        full320_FG090_goal_verified=False,all_source_foreground_preserved=True,all_groups4connected=True)
    write_json(ARCHIVE/'score_v1/summary.json',summary);return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--admit',action='store_true');parser.add_argument('--score',action='store_true');args=parser.parse_args()
    if args.score:print(json.dumps(score()),flush=True)
    elif args.admit:print(json.dumps(admit(args.seed)),flush=True)
    else:predict(args.seed)
