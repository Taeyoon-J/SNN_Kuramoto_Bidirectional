"""Three independent frozen seeds, paired baseline/geometric-Q grouping."""
import io,json,os,time
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0197_native64_baseline_replication import run as baseline
from collaborative_test.SW_0193_native64_probe_screen import run as screen,score as metrics
from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
from .readout import geometric_consensus
train=baseline.train
HERE=Path(__file__).resolve().parent
ROOT_ARCHIVE=HERE/'results_archive'
ARCHIVE=ROOT_ARCHIVE
RUN_SEED=None
ARMS=('prediction',)
OFFSETS=tuple(range(0,32,4))

def configure(seed):
    global RUN_SEED,ARCHIVE
    if type(seed) is not int or seed not in (0,1,2):raise ValueError('invalid seed')
    RUN_SEED=seed;ARCHIVE=ROOT_ARCHIVE/f'seed{seed}'

def fingerprint():
    return dict(files={n:train.sha((HERE/n).read_bytes()) for n in ('run.py','queue.py','readout.py','test_readout.py','DESIGN_KO.md')},
        source=train.fingerprint(),source_baseline=baseline.fingerprint(1),seed=RUN_SEED,offsets=list(OFFSETS),components=4,optimizer_updates=0)

def reference():
    controls,ref=metrics.validated_pair()
    if RUN_SEED==0:return controls['baseline'],ref
    labels,binding=baseline.validated_pair(RUN_SEED)
    if binding['reference_binding']!=ref:raise ValueError('source cohorts differ')
    return labels,ref

def prepare_cpu():
    if torch.cuda.is_initialized():raise RuntimeError('CPU admission must precede CUDA')
    inputs=train.incumbent.prepare_training_inputs_cpu()
    found,_optimizer,_families,proof=train.model64.build_from_incumbent(RUN_SEED,inputs)
    model=train.native._state_sha(found.wrapped,found.encoder)
    if model!=proof['native64_initial_model_state_sha256']:raise ValueError('factory model changed')
    old,ref=reference()
    from collaborative_test.SW_0130_phase_state_integration import run as data
    rgb=np.load(data.VAL_RGB,mmap_mode='r',allow_pickle=False)[:32].copy()
    if train.sha(rgb.tobytes())!=ref['development_RGB_sha256']:raise ValueError('RGB cohort changed')
    binding=dict(fingerprint=fingerprint(),seed=RUN_SEED,model_sha256=model,reference_binding=ref,
        image_ids=screen.IDS[list(OFFSETS)].tolist(),frozen_old_sha256=[train.sha(old[i].tobytes()) for i in OFFSETS],
        validation_GT_used_for_prediction=False,optimizer_updates=0)
    return found,None,rgb[list(OFFSETS)],binding

def read_row(folder,binding,index):
    meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'masks.npz').read_bytes()
    if meta['binding']!=binding or meta['index']!=index or meta['sha256']!=train.sha(raw):raise ValueError('saved row changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as stream:row={k:stream[k].copy() for k in ('old','new')}
    if any(v.dtype!=np.int64 or v.shape!=(64,64) or (v<0).any() for v in row.values()):raise ValueError('invalid masks')
    if train.sha(row['old'].tobytes())!=binding['frozen_old_sha256'][index]:raise ValueError('baseline replay failed')
    if not np.array_equal(row['old']>0,row['new']>0):raise ValueError('foreground support changed')
    return row

def predict(expected_binding=None):
    found,_,rgb,binding=prepare_cpu()
    if binding!=expected_binding:raise ValueError('CPU admission changed')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    train.model64.move_to_device(found,device)
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    for index,offset in enumerate(OFFSETS):
        folder=ARCHIVE/'prediction/rows'/f'{index:04d}'
        if folder.exists():read_row(folder,binding,index);continue
        torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
        with torch.no_grad():
            image=torch.from_numpy(rgb[index:index+1]).to(device)
            gamma,trace,adj,z,q,rho=train.native._native_forward(found,image,live_tail_steps=0)
            if int(rho.shape[1])!=4:raise ValueError('actual component count differs from registered four')
            old=screen.prediction_from_trace(found,trace,adj,q)
            if train.sha(old.tobytes())!=binding['frozen_old_sha256'][index]:raise ValueError('original mask did not replay exactly')
            affinity=geometric_consensus(q[0].detach().cpu().numpy())
            gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
            new=np.asarray(foreground_induced_merge(affinity,adj[0].detach().cpu().numpy().astype(np.float64),gain,old,grid_shape=(64,64))['labels'],dtype=np.int64)
        stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True)
        with (stage/'masks.npz').open('xb') as stream:
            np.savez_compressed(stream,old=old,new=new);stream.flush();os.fsync(stream.fileno())
        train.write_json(stage/'row.json',dict(binding=binding,index=index,sha256=train.sha((stage/'masks.npz').read_bytes())))
        read_row(stage,binding,index);stage.rename(folder)
        del image,gamma,trace,adj,z,q,rho,affinity
    if train.native._state_sha(found.wrapped,found.encoder)!=binding['model_sha256']:raise ValueError('model mutated')
    train.write_json(ARCHIVE/'prediction/protocol.json',dict(binding=binding,rows_sha256=[train.sha((ARCHIVE/'prediction/rows'/f'{i:04d}'/'row.json').read_bytes()) for i in range(8)]))

def validated_pair():
    p=json.loads((ARCHIVE/'prediction/protocol.json').read_bytes());b=p['binding']
    if b['fingerprint']!=fingerprint() or b['validation_GT_used_for_prediction'] is not False:raise ValueError('protocol changed')
    rows=[]
    for i in range(8):
        folder=ARCHIVE/'prediction/rows'/f'{i:04d}'
        if train.sha((folder/'row.json').read_bytes())!=p['rows_sha256'][i]:raise ValueError('row metadata changed')
        rows.append(read_row(folder,b,i))
    return {key:np.stack([r[key] for r in rows]) for key in ('old','new')},b

def score():
    labels,binding=validated_pair();old,ref=reference()
    if ref!=binding['reference_binding'] or not np.array_equal(old[list(OFFSETS)],labels['old']):raise ValueError('source predictions changed')
    gt=metrics.read_ground_truth(ref)[list(OFFSETS)]
    vectors={k:metrics.measured_vectors(v,gt)[0] for k,v in labels.items()}
    if not np.array_equal(vectors['old'][:,1],vectors['new'][:,1]):raise ValueError('foreground IoU changed')
    delta=vectors['new']-vectors['old'];rng=np.random.RandomState(203);draws=rng.randint(0,8,(10000,8))
    ci=np.percentile(delta[draws].mean(1),[2.5,97.5],axis=0)
    names=metrics.independent.METRICS
    summary=dict(status='completed_paired_geometric_consensus_readout',seed=RUN_SEED,grid=[64,64],images=8,
        scores={k:dict(zip(names,v.mean(0).tolist())) for k,v in vectors.items()},
        paired={m:dict(mean=float(delta[:,i].mean()),ci95=ci[:,i].tolist()) for i,m in enumerate(names)},
        binding=binding,optimizer_updates=0,new_training=False,foreground_support_identical=True,
        full320_three_seed_goal_verified=False,causal_gate_contribution_proved=False)
    train.write_json(ARCHIVE/'score_v1/evaluation_private.json',dict(summary=summary,vectors={k:v.tolist() for k,v in vectors.items()}))
    train.write_json(ARCHIVE/'score_v1/summary.json',summary)
    return summary
