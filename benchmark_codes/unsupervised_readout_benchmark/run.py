"""Frozen SW0165 native32; resumable activity cache and mask-free readout fit."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
ROOT=Path('/Data0/kevinswk/patch_v2_sw')
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test'),str(ROOT/'snn_kuramoto_bidirectional'),str(HERE)]
from model import ReconstructionReadout,ARMS,objective
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
from collaborative_test.SW_0162_connected_signed_partition import preflight as source
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver
spec=importlib.util.spec_from_file_location('sw259_activity',HERE.parents[0]/'vendor/readout_benchmark/head.py')
activity=importlib.util.module_from_spec(spec);spec.loader.exec_module(activity)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)


def code():
    paths=[*HERE.glob('*.py'),HERE.parent/'vendor/neutral_readout/head.py',Path(activity.__file__)]
    return {**source.implementation_fingerprint(),**{str(p):sha(p) for p in paths}}


def reference(seed):
    folder=ROOT/('collaborative_test/SW_0164_foreground_preserving_partition' if seed==1 else 'collaborative_test/SW_0165_foreground_partition_replication')/'results_archive/predictions_20261014'/f'foreground_induced_seed{seed}'
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        assert z['labels'].shape==(320,32,32)
        return z['labels'].copy()


def load(seed):
    path,mp,manifest,_,_,digest=sw130.source_contract(seed)
    assert manifest['ground_truth_used_for_training'] is False
    found=load_native32_foundation(seed,'cuda:0',verify_rgb_assets=False)
    found.encoder.eval().requires_grad_(False);found.wrapped.eval().requires_grad_(False)
    geodesic_operator.install_operator(found.wrapped.core.graph_generator,'T')
    return found,dict(core=str(path),core_sha256=digest,manifest_sha256=sha(mp),
        encoder_sha256=sha(sw130.ENCODER_SOURCE),statistics_sha256=sha(sw130.FEATURE_STATS),
        state=source._source_state_digest(found),native_grid=32,steps=1024,settle=512)


@torch.no_grad()
def extract(found,rgb):
    gamma=found.encode(torch.from_numpy(np.asarray(rgb[None]).copy()).cuda())
    trace,a=source.capture_source_coupling(found.wrapped.core.kuramoto,
        lambda:late_rollout(found.wrapped,gamma,total_steps=1024,live_tail_steps=0))
    source._trace_check(trace);source._validate_actual_A(a)
    features=activity.activity_features(trace)[0].cpu()
    q=source._actual_q(trace)[0].cpu().numpy().astype(np.float64)
    adjacency=a[0].cpu().numpy().astype(np.float64)
    gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().cpu())
    support=sw136._qcc_labels(trace)[0]
    baseline=solver.foreground_induced_merge(q,adjacency,gain,support,grid_shape=(32,32))['labels'].astype(np.int64)
    # Like252: full-image proposals; no largest-group background designation.
    regions=solver.foreground_induced_merge(q,adjacency,gain,np.ones((32,32),dtype=np.int64),grid_shape=(32,32))['labels'].astype(np.int64)
    target=F.avg_pool2d(torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).float()[None]/255.,4).squeeze(0).permute(1,2,0).reshape(1024,3)
    assert features.shape==(1024,144) and regions.min()>=0
    return dict(features=features,regions=torch.from_numpy(regions),baseline=torch.from_numpy(baseline),rgb=target)


def bind(path,value):
    if path.exists():assert json.loads(path.read_bytes())==value,'cache binding changed'
    else:write(path,value)


def cache(found,out,identity,split,count):
    folder=out/'cache'/split;folder.mkdir(parents=True,exist_ok=True)
    if split=='train':
        pool=np.r_[np.arange(1000),np.arange(1640,70640)].astype(np.int64)
        positions=np.random.RandomState(259).permutation(70000)[:count]
        ids=pool[positions];images=np.load(sw130.TRAIN_RGB,mmap_mode='r')
    elif split=='validation':
        import h5py
        ids=np.arange(1000,1000+count);positions=np.arange(count)
        with h5py.File(sw130.source97.DATASET,'r') as f:images=np.asarray(f['image'][ids])
    else:
        assert split=='fixed320' and count==320
        ids=np.arange(1320,1640);positions=np.arange(count);images=np.load(sw130.VAL_RGB,mmap_mode='r')
    assert not np.intersect1d(ids,np.arange(1320,1640)).size if split!='fixed320' else True
    binding=dict(source=identity,code=code(),split=split,ids=ids.tolist(),training_masks_used=False)
    bind(folder/'binding.json',binding)
    expected=reference(int(out.name[4:])) if split=='fixed320' else None
    started=time.time()
    for i,(pos,image_id) in enumerate(zip(positions,ids)):
        path=folder/f'{i:04d}.pt';rgb=np.asarray(images[pos]);rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
        if path.exists():
            row=torch.load(path,map_location='cpu',weights_only=True)
            assert row['image_id']==int(image_id) and row['rgb_sha256']==rgbsha
        else:
            row=extract(found,rgb);row.update(image_id=int(image_id),rgb_sha256=rgbsha)
            save(path,row)
        if expected is not None:assert np.array_equal(row['baseline'].numpy(),expected[i]),'SW0165 exact replay failed'
        if i%8==0 or i==count-1:
            write(out/'progress.json',dict(stage='cache_'+split,completed=i+1,total=count,elapsed=time.time()-started,updated=time.time()))
    assert source._source_state_digest(found)==identity['state'],'frozen source changed'
    write(folder/'complete.json',dict(count=count,source_unchanged=True,all_baseline_masks_exact=expected is not None,seconds=time.time()-started))
    return folder


def rows(folder):
    return [torch.load(p,map_location='cpu',weights_only=True) for p in sorted(folder.glob('*.pt'))]


@torch.no_grad()
def validation(model,data):
    losses=[];groups=[]
    for row in data:
        rgb,detail=model(row['features'].cuda(),row['regions'].numpy())
        losses.append(float(F.mse_loss(rgb,row['rgb'].cuda())))
        groups.append(len(np.unique(model.predict(row['features'].cuda(),row['regions'].numpy()))))
    return dict(rgb_mse=float(np.mean(losses)),mean_hard_groups=float(np.mean(groups)))


def fit(out,train,val,seed,arm,epochs):
    folder=out/'arms'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    model=ReconstructionReadout(seed).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    best=float('inf');start=0;updates=0
    if (folder/'resume.pt').exists():
        saved=torch.load(folder/'resume.pt',weights_only=True,map_location='cuda:0')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=int(saved['epoch']);best=float(saved['best']);updates=int(saved['updates'])
    initial={k:v.detach().clone() for k,v in model.head.state_dict().items()}
    for epoch in range(start,epochs):
        order=np.random.RandomState(259000+seed+10000*epoch).permutation(len(train));losses=[]
        model.train()
        for index in order:
            row=train[index];optimizer.zero_grad(set_to_none=True)
            rgb,detail=model(row['features'].cuda(),row['regions'].numpy())
            loss,mse=objective(rgb,row['rgb'].cuda(),detail,arm);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            optimizer.step();updates+=1;losses.append(float(loss.detach()))
        model.eval();v=validation(model,val)
        if v['rgb_mse']<best:
            best=v['rgb_mse'];save(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch+1,validation=v,arm=arm))
        save(folder/'resume.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,updates=updates))
        with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,updates=updates,loss=float(np.mean(losses)),validation=v))+'\n')
        write(out/'progress.json',dict(stage='train_head_decoder',arm=arm,epoch=epoch+1,epochs=epochs,updates=updates,updated=time.time()))
    assert any(not torch.equal(v,model.head.state_dict()[k]) for k,v in initial.items()) if start<epochs else True
    write(folder/'complete.json',dict(best_validation_mse=best,updates=updates,unique_TRAIN=len(train),epochs=epochs,training_masks_used=False,best_sha256=sha(folder/'best.pt')))


def admission(out,seed):
    found,identity=load(seed)
    rgb=np.load(sw130.VAL_RGB,mmap_mode='r')[0]
    row=extract(found,rgb)
    assert np.array_equal(row['baseline'].numpy(),reference(seed)[0]),'first SW0165 replay failed'
    model=ReconstructionReadout(seed).cuda();before={k:v.detach().clone() for k,v in model.head.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    rgb,detail=model(row['features'].cuda(),row['regions'].numpy())
    loss,_=objective(rgb,row['rgb'].cuda(),detail,'rgb');loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.head.parameters() if p.grad is not None)>0
    assert sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)>0
    optimizer.step()
    assert any(not torch.equal(v,model.head.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None and not p.requires_grad for p in found.wrapped.parameters())
    assert source._source_state_digest(found)==identity['state']
    report=dict(seed=seed,source=identity,exact_baseline_first_mask=True,actual_features=list(row['features'].shape),
        mask_free_head_decoder_gradients=True,frozen_source_unchanged=True,seconds=time.time()-START,code=code())
    write(out/'admission.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='code'}),flush=True)


def worker(out,seed,train_count,epochs):
    assert (out/'admission.json').exists(),'actual CUDA admission required'
    found,identity=load(seed)
    admitted=json.loads((out/'admission.json').read_bytes());assert admitted['source']==identity and admitted['code']==code()
    tr=cache(found,out,identity,'train',train_count);va=cache(found,out,identity,'validation',32)
    train=rows(tr);val=rows(va)
    for arm in ARMS:fit(out,train,val,seed,arm,epochs)
    fixed=cache(found,out,identity,'fixed320',320)
    write(out/'complete.json',dict(seed=seed,status='all_arms_trained_fixed_features_ready',source=identity,code=code(),fixed_cache=str(fixed),training_masks_used=False))


def score(output):
    # Choose ONE common arm using RGB-only validation across all seeds, before GT.
    output=Path(output);records={}
    for seed in range(3):
        out=output/f'seed{seed}';assert (out/'complete.json').exists()
        records[seed]={arm:json.loads((out/'arms'/arm/'complete.json').read_bytes()) for arm in ARMS}
    selection={arm:float(np.mean([records[s][arm]['best_validation_mse'] for s in range(3)])) for arm in ARMS}
    chosen=min(selection,key=selection.get)
    write(output/'selection.json',dict(arm=chosen,validation_rgb_mse=selection,GT_used=False))
    for seed in range(3):
        out=output/f'seed{seed}';data=rows(out/'cache/fixed320')
        model=ReconstructionReadout(seed).cuda()
        saved=torch.load(out/'arms'/chosen/'best.pt',map_location='cuda:0',weights_only=True);model.load_state_dict(saved['model']);model.eval()
        pred=np.stack([model.predict(row['features'].cuda(),row['regions'].numpy()) for row in data])
        np.savez_compressed(out/'predictions.npz',labels=pred,image_ids=np.arange(1320,1640))
        write(out/'prediction_protocol.json',dict(arm=chosen,GT_used=False,prediction_sha256=sha(out/'predictions.npz'),code=code()))
    import h5py
    with h5py.File(sw130.source97.DATASET,'r') as f:truth=contract.modal_native32(np.asarray(f['mask'][1320:1640]))
    results=[]
    for seed in range(3):
        out=output/f'seed{seed}'
        with np.load(out/'predictions.npz') as z:pred=z['labels'].copy()
        scored=contract.production_metrics32(pred,truth)
        baseline=np.stack([r['baseline'].numpy() for r in rows(out/'cache/fixed320')])
        base=contract.production_metrics32(baseline,truth)
        result=dict(seed=seed,arm=chosen,metrics={k:v['mean'] for k,v in scored.items()},baseline={k:v['mean'] for k,v in base.items()},prediction_sha256=sha(out/'predictions.npz'))
        write(out/'result.json',result);results.append(result)
    mean={k:float(np.mean([r['metrics'][k] for r in results])) for k in contract.METRICS}
    write(output/'results.json',dict(per_seed=results,mean=mean,selected_arm=chosen,selection=selection,GT_used_for_training=False,baseline_FGARI=.804299,
        exceeded_baseline=mean['fg_ari']>.804299,foreground_label_convention='all groups positive; no BG designation; raw FGIoU is GT foreground fraction'))
    print(json.dumps(dict(mean=mean,selected_arm=chosen)),flush=True)


START=time.time()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['admission','worker','score']);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0,choices=range(3));p.add_argument('--train',type=int,default=256);p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.22,0)
    out=Path(a.output)/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if a.command=='admission':admission(out,a.seed)
    elif a.command=='worker':worker(out,a.seed,a.train,a.epochs)
    else:score(a.output)
