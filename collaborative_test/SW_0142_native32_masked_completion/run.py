"""SW0142 native32 masked-completion paired pilot."""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/"collaborative_test",ROOT/"snn_kuramoto_bidirectional"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract32
from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder,RelativeSlotRGBDecoder,canonical_hard_labels,render_slot_rgb
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval

NAME="SW0142_native32_masked_completion"; SEED=1
STEPS,SETTLE,TAIL=1024,512,64; GRID,N,IMAGE=32,1024,128
TRAIN_COUNT,BATCH,MICRO=4096,16,4; WARM_UPDATES,UPDATES=256,256
ARMS=("encoder_graph","core_trainable")
CACHE=HERE/"masked_cache"/"source1_masked_n4096"; WARM=HERE/"warm_head"/"seed1"
TRAIN=HERE/"training"; RESULTS=HERE/"results"; QUEUE=HERE/"queue_runtime"
SOURCE_ROOT=ROOT/"collaborative_test/SW_0137_native32_source_qcc/results_archive/source_predictions"
SLOT_ROOT=ROOT/"trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE_PATH=ROOT/"collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256="4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
DATASET=Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canonical_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def write_once(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as f: json.dump(row,f,indent=2,allow_nan=False);f.write("\n")
def fingerprint():
    paths=[HERE/"run.py",HERE/"protocol.json"]
    paths += [ROOT/k for k in sw135.implementation_fingerprint()]
    paths += [ROOT/"collaborative_test/SW_0136_native32_transfer/evaluate.py",
              ROOT/"collaborative_test/SW_0137_native32_source_qcc/run.py",
              ROOT/"collaborative_test/SW_0139_native32_representation/run.py",
              ROOT/"collaborative_test/SW_0139_native32_representation/evaluate.py"]
    paths += [ROOT/"snn_kuramoto_bidirectional"/x for x in ("s2net_cls.py","graph_generator.py","dendric_layer.py","membrane_layer.py","kuramoto_layer.py","sinusoidal_gating.py","gamma_initializer.py","loss_function.py","spike_classifier.py","evaluation.py","hyperparameter.py","training/train_gamma_initializer.py")]
    if any(not p.is_file() for p in paths): raise FileNotFoundError("SW0142 fingerprint dependency missing")
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in paths}
def source_foundation(device="cpu",verify_assets=True):
    f=sw135.load_native32_foundation(SEED,device,verify_rgb_assets=verify_assets);src=sw130.source_contract(SEED)
    if f.provenance["source_core_sha256"]!=src[5] or list(map(int,f.image_ids))!=list(map(int,src[4])) or not np.array_equal(f.pool_indices,np.asarray(src[3],dtype=np.int64)):
        raise ValueError("SW0142 requires exact same-seed SW0097 native32 source/order")
    return f

def mask_tiles(image_id,exposure,seed=SEED):
    digest=hashlib.sha256(f"SW0142-mask-v1|{seed}|{int(image_id)}|{int(exposure)}".encode()).digest()
    return np.sort(np.random.RandomState(int.from_bytes(digest[:4],"little")).choice(64,16,replace=False)).astype(np.int16)
def mask_rgb(rgb,ids,exposure,seed=SEED):
    if not torch.is_tensor(rgb): rgb=torch.as_tensor(rgb)
    if rgb.dtype!=torch.uint8 or tuple(rgb.shape[1:])!=(128,128,3) or len(ids)!=rgb.shape[0]: raise ValueError("mask input must be NHWC uint8 RGB with matching IDs")
    result=rgb.clone();tiles=[]
    exps=[int(exposure)]*len(ids) if np.isscalar(exposure) else list(map(int,exposure))
    if len(exps)!=len(ids):raise ValueError("mask exposure batch mismatch")
    for b,image_id in enumerate(ids):
        chosen=mask_tiles(image_id,exps[b],seed);tiles.append(chosen)
        for t in chosen.tolist():
            r,c=divmod(int(t),8);result[b,r*16:(r+1)*16,c*16:(c+1)*16]=128
    return result,np.stack(tiles)
def pixel_mask(tiles,device):
    mask=torch.zeros((len(tiles),128,128),dtype=torch.bool,device=device)
    for b,row in enumerate(tiles):
        for t in np.asarray(row,dtype=np.int64):
            r,c=divmod(int(t),8);mask[b,r*16:(r+1)*16,c*16:(c+1)*16]=True
    return mask
def completion_loss(pred,target,tiles):
    if pred.shape!=target.shape or tuple(pred.shape[1:])!=(128,128,3):raise ValueError("RGB shape mismatch")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():raise FloatingPointError("nonfinite RGB")
    m=pixel_mask(tiles,pred.device);sq=(pred-target.detach()).square()
    lm=(sq*m[...,None]).sum((1,2,3))/(m.sum((1,2)).to(sq.dtype)*3).clamp_min(1)
    lv=(sq*(~m)[...,None]).sum((1,2,3))/((~m).sum((1,2)).to(sq.dtype)*3).clamp_min(1)
    var=target.detach().var((1,2),unbiased=False).mean(-1).clamp_min(1e-6)
    return ((lm+.25*lv)/var).mean()
def masked_forward(f,binder,decoder,rgb,ids,exposure,live_tail=TAIL):
    masked,tiles=mask_rgb(rgb,ids,exposure);gamma=f.encode(masked)
    trace=f.rollout(gamma,total_steps=STEPS,live_tail_steps=live_tail)
    spikes=trace["component_spikes"][...,SETTLE:]
    if tuple(spikes.shape[1:])!=(4,N,512):raise ValueError("readout requires full512 actual spike history")
    pred,P,slots,_,labels=sw135.reconstruct_from_spikes(spikes,binder,decoder)
    target=rgb.to(pred.device,dtype=torch.float32)/255.
    return {"loss":completion_loss(pred,target,tiles),"prediction":pred,"P":P,"slots":slots,"labels":labels,"trace":trace,"gamma":gamma,"tiles":tiles}
def ordered_train():
    src=sw130.source_contract(SEED);ids=list(map(int,src[4][:TRAIN_COUNT]));rows=np.asarray(src[3][:TRAIN_COUNT],dtype=np.int64)
    if len(ids)!=TRAIN_COUNT or len(set(ids))!=TRAIN_COUNT or rows.shape!=(TRAIN_COUNT,):raise ValueError("registered source1 TRAIN order invalid")
    return ids,rows
def _raw_rows(rows,cache,device):
    x=np.asarray(cache[np.asarray(rows,dtype=np.int64)]).copy()
    if x.dtype!=np.uint8 or x.shape[1:]!=(128,128,3):raise ValueError("RGB training asset shape/dtype mismatch")
    return torch.from_numpy(x).to(device)
def cache_dir():return CACHE
def warm_dir():return WARM
def arm_dir(arm):return TRAIN/f"{arm}_seed1"

def create_cache(output=CACHE,device="cuda:0"):
    output=Path(output)
    if output.exists():raise FileExistsError("preserve existing SW0142 masked cache")
    parent=output.parent
    while not parent.exists():parent=parent.parent
    required=TRAIN_COUNT*4*N*512*2 + TRAIN_COUNT*128*128*3 + TRAIN_COUNT*16 + (1<<30)
    if shutil.disk_usage(parent).free < required:raise OSError("insufficient free disk for float16 cache plus 1 GiB reserve")
    f=source_foundation(device);ids,rows=ordered_train();rgbsrc=np.load(sw130.TRAIN_RGB,mmap_mode="r",allow_pickle=False)
    output.mkdir(parents=True,exist_ok=False)
    sp=output/"spikes.f16";rp=output/"rgb.u8";tp=output/"tiles.i16"
    sm=np.memmap(sp,mode="w+",dtype=np.float16,shape=(TRAIN_COUNT,4,N,512));rm=np.memmap(rp,mode="w+",dtype=np.uint8,shape=(TRAIN_COUNT,128,128,3));tm=np.memmap(tp,mode="w+",dtype=np.int16,shape=(TRAIN_COUNT,16))
    f.wrapped.eval();f.encoder.eval();chunk=[];started=time.time();quant_abs=[]
    with torch.no_grad():
        for start in range(0,TRAIN_COUNT,MICRO):
            end=start+MICRO;rgb=_raw_rows(rows[start:end],rgbsrc,device)
            exposures=[0]*MICRO
            masked,tiles=mask_rgb(rgb,ids[start:end],exposures)
            trace=f.rollout(f.encode(masked),total_steps=STEPS,live_tail_steps=0);full=trace["component_spikes"][...,SETTLE:].detach().cpu();x=full.to(torch.float16).contiguous().numpy()
            if x.shape!=(MICRO,4,N,512) or not np.isfinite(x).all():raise ValueError("invalid masked warm cache trace")
            if start==0:
                delta=(full.float()-torch.from_numpy(x).to(torch.float32)).abs();quant_abs=[float(delta.max()),float(delta.mean())]
            sm[start:end]=x;rm[start:end]=rgb.cpu().numpy();tm[start:end]=tiles;chunk.append(hashlib.sha256(x.tobytes()).hexdigest())
            if end%64==0:print(json.dumps({"stage":"cache","completed":end,"total":TRAIN_COUNT}),flush=True)
    sm.flush();rm.flush();tm.flush();del sm,rm,tm
    row={"status":"complete","experiment":NAME,"seed":SEED,"count":TRAIN_COUNT,"shape":[TRAIN_COUNT,4,N,512],"dtype":"float16",
         "image_ids":ids,"pool_rows":rows.tolist(),"training_ids_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest(),
         "source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],
         "rgb_asset_validation":f.provenance["rgb_asset_validation"],"foundation_fingerprint":f.provenance["implementation_fingerprint"],"implementation_fingerprint":fingerprint(),
         "mask_rule":"16/64 row-major 16x16 tiles, SHA256(RandomState) keyed by ID/exposure0; pixels=128",
         "warm_exposures":TRAIN_COUNT,"chunk_count":len(chunk),"chunk_sha256":chunk,"spikes_sha256":sha256(sp),"rgb_sha256":sha256(rp),"tiles_sha256":sha256(tp),
         "first_microbatch_float16_abs_error_max":quant_abs[0],"first_microbatch_float16_abs_error_mean":quant_abs[1],
         "ground_truth_used":False,"optimizer_updates":0,"elapsed_seconds":time.time()-started}
    write_once(output/"manifest.json",row);return row

class MaskCache:
    def __init__(self,path,f):
        self.path=Path(path);m=json.loads((self.path/"manifest.json").read_text(encoding="utf-8"));self.manifest=m
        if m.get("status")!="complete" or m.get("experiment")!=NAME or m.get("seed")!=1 or m.get("shape")!=[TRAIN_COUNT,4,N,512] or m.get("dtype")!="float16" or m.get("implementation_fingerprint")!=fingerprint() or m.get("foundation_fingerprint")!=f.provenance["implementation_fingerprint"] or m.get("rgb_asset_validation")!=f.provenance["rgb_asset_validation"]:raise ValueError("masked cache provenance mismatch")
        ids,rows=ordered_train()
        if m.get("image_ids")!=ids or m.get("pool_rows")!=rows.tolist() or m.get("chunk_count")!=TRAIN_COUNT//MICRO or len(m.get("chunk_sha256",[]))!=TRAIN_COUNT//MICRO or m.get("ground_truth_used") is not False or m.get("optimizer_updates")!=0:raise ValueError("masked cache row ordering/protocol mismatch")
        for name,key in (("spikes.f16","spikes_sha256"),("rgb.u8","rgb_sha256"),("tiles.i16","tiles_sha256")):
            if sha256(self.path/name)!=m[key]:raise ValueError(f"masked cache digest mismatch: {name}")
        self.spikes=np.memmap(self.path/"spikes.f16",mode="r",dtype=np.float16,shape=(TRAIN_COUNT,4,N,512));self.rgb=np.memmap(self.path/"rgb.u8",mode="r",dtype=np.uint8,shape=(TRAIN_COUNT,128,128,3));self.tiles=np.memmap(self.path/"tiles.i16",mode="r",dtype=np.int16,shape=(TRAIN_COUNT,16))
        for row in range(TRAIN_COUNT):
            if not np.array_equal(self.tiles[row],mask_tiles(ids[row],0)):raise ValueError(f"masked cache tile rule mismatch at row {row}")
    def batch(self,start,device):return (torch.from_numpy(np.asarray(self.spikes[start:start+MICRO]).copy()).to(device=device,dtype=torch.float32),torch.from_numpy(np.asarray(self.rgb[start:start+MICRO]).copy()).to(device),np.asarray(self.tiles[start:start+MICRO]).copy())

def warm_head(*,device="cuda:0",updates=WARM_UPDATES):
    if updates!=256:raise ValueError("warmup fixed at256 logical B16 updates")
    if WARM.exists():raise FileExistsError("preserve existing warm head")
    f=source_foundation(device);cache=MaskCache(CACHE,f);binder=NativeSpikeSlotBinder(seed=142).to(device);decoder=RelativeSlotRGBDecoder(seed=106).to(device);params=list(binder.parameters())+list(decoder.parameters());opt=torch.optim.Adam(params,lr=1e-3)
    WARM.mkdir(parents=True,exist_ok=False);hist=[];t0=time.time()
    for u in range(256):
        opt.zero_grad(set_to_none=True);ls=[]
        for mi in range(4):
            pos=u*16+mi*4;sp,rgb,tiles=cache.batch(pos,device)
            P,z,_=binder(sp);pred=render_slot_rgb(P,z,decoder);loss=completion_loss(pred,rgb.float()/255.,tiles);(loss/4).backward();ls.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(params,1.);opt.step();hist.append({"update":u+1,"loss_micro":ls})
        if (u+1)%16==0:print(json.dumps({"stage":"warm","update":u+1,"total":256}),flush=True)
    warm_diagnostic=assignment_diagnostics_from_cache(cache,binder,decoder,device=device)
    cp=WARM/"warm.pt";op=WARM/"warm_optimizer.pt";hp=WARM/"history.json";torch.save({"binder":binder.state_dict(),"decoder":decoder.state_dict(),"optimizer":opt.state_dict()},cp);torch.save(opt.state_dict(),op);hp.write_text(json.dumps(hist,allow_nan=False),encoding="utf-8")
    rec={"status":"warm_complete","experiment":NAME,"updates":256,"logical_batch":16,"exposures":4096,"cache_manifest_sha256":sha256(CACHE/"manifest.json"),"checkpoint_sha256":sha256(cp),"optimizer_sha256":sha256(op),"history_sha256":sha256(hp),"source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],"rgb_asset_validation":f.provenance["rgb_asset_validation"],"assignment_diagnostic_first64_train":warm_diagnostic,"implementation_fingerprint":fingerprint(),"ground_truth_used":False,"elapsed_seconds":time.time()-t0};write_once(WARM/"manifest.json",rec);return rec

def _warm_state(device):
    m=json.loads((WARM/"manifest.json").read_text(encoding="utf-8"));cp=WARM/"warm.pt";op=WARM/"warm_optimizer.pt"
    if m.get("status")!="warm_complete" or m.get("implementation_fingerprint")!=fingerprint() or sha256(cp)!=m.get("checkpoint_sha256") or sha256(op)!=m.get("optimizer_sha256"):raise ValueError("warm state/hash mismatch")
    state=torch.load(cp,map_location=device,weights_only=True);saved_opt=torch.load(op,map_location=device,weights_only=True)
    if not isinstance(state,dict) or not isinstance(saved_opt,dict) or state.get("optimizer") is None:raise ValueError("malformed shared warm state")
    embedded=state["optimizer"]
    if embedded.get("param_groups")!=saved_opt.get("param_groups") or embedded.get("state",{}).keys()!=saved_opt.get("state",{}).keys():raise ValueError("warm Adam moments disagree between checkpoint and optimizer artifact")
    for key,vals in saved_opt["state"].items():
        for field,value in vals.items():
            other=embedded["state"][key][field]
            if torch.is_tensor(value) and (not torch.is_tensor(other) or not torch.equal(value,other)):raise ValueError("warm Adam tensor state mismatch")
            if not torch.is_tensor(value) and value!=other:raise ValueError("warm Adam scalar state mismatch")
    state["optimizer"]=saved_opt
    return m,state

def _trainable(f,arm):
    for name,p in f.wrapped.named_parameters():
        p.requires_grad_(name.startswith("core.graph_generator.") or arm=="core_trainable")
    f.encoder.requires_grad_(True);f.wrapped.train();f.encoder.train()
    params=[p for p in f.wrapped.parameters() if p.requires_grad]+list(f.encoder.parameters())
    if not params or len({id(p) for p in params})!=len(params):raise ValueError("invalid/duplicate backbone parameter group")
    return params
def _optimizers(f,arm,binder,decoder,warm_state):
    backbone=_trainable(f,arm);groups=[]
    graph=[p for p in f.wrapped.core.graph_generator.parameters() if p.requires_grad]
    if graph:groups.append({"params":graph,"lr":3e-5,"name":"graph"})
    if arm=="core_trainable":
        scalars=[];native=[]
        for name,p in f.wrapped.named_parameters():
            if not p.requires_grad or name.startswith("core.graph_generator."):continue
            (scalars if name.endswith(("a_d","a_m","b")) else native).append(p)
        if native:groups.append({"params":native,"lr":3e-5,"name":"native_core"})
        if scalars:groups.append({"params":scalars,"lr":1e-3,"name":"integration_scalars"})
    groups.append({"params":list(f.encoder.parameters()),"lr":3e-6,"name":"encoder"})
    opt_b=torch.optim.Adam(groups);heads=list(binder.parameters())+list(decoder.parameters());opt_h=torch.optim.Adam(heads,lr=1e-3);opt_h.load_state_dict(warm_state["optimizer"])
    return backbone,heads,opt_b,opt_h
def _state_sha(modules):
    h=hashlib.sha256()
    for prefix,module in modules:
        for name,value in sorted(module.state_dict().items()):
            t=value.detach().cpu().contiguous();h.update(prefix.encode()+name.encode()+str(t.dtype).encode());h.update(np.asarray(t.shape,dtype="<i8").tobytes());h.update(t.numpy().tobytes())
    return h.hexdigest()
def _frozen_core_sha(f):
    rows=[(n,p) for n,p in f.wrapped.named_parameters() if not n.startswith("core.graph_generator.")]
    h=hashlib.sha256()
    for n,p in rows:
        t=p.detach().cpu().contiguous();h.update(n.encode()+t.numpy().tobytes())
    return h.hexdigest()

def _optimizer_step_range(opt):
    steps=[]
    for row in opt.state.values():
        value=row.get("step")
        if value is not None:steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return [min(steps),max(steps)] if steps else [0,0]

def _begin_cuda_measurement(device):
    dev=torch.device(device)
    if dev.type!="cuda":return False
    torch.cuda.set_device(dev);torch.cuda.init();torch.cuda.synchronize(dev);torch.cuda.reset_peak_memory_stats(dev)
    return True

def _event_occupancy(trace, b):
    """Describe threshold events and emitted spikes without using gate-inferred events."""
    membrane=trace["component_membrane"].detach()
    gates=trace["component_gates"].detach()
    if membrane.ndim!=4 or gates.shape!=membrane.shape or membrane.shape[1]!=4:
        raise ValueError("event occupancy requires matching [B,4,N,T] membrane and gate traces")
    if tuple(b.shape)!=(4,) or not torch.isfinite(membrane).all() or not torch.isfinite(gates).all():
        raise ValueError("event occupancy received invalid membrane, gates, or thresholds")
    threshold=(0.06*torch.exp(b.detach())).view(1,4,1,1)
    events=membrane>threshold
    emitted=events.to(gates.dtype)*gates
    def rates(value):
        return {"per_component":value.float().mean(dim=(0,2,3)).cpu().tolist(),
                "overall":float(value.float().mean().cpu())}
    return {"event_threshold":"component_membrane > 0.06 * exp(b[component])",
            "all_steps":{"events":rates(events),"emitted_spikes":rates(emitted)},
            "settled_512":{"events":rates(events[...,SETTLE:]),
                            "emitted_spikes":rates(emitted[...,SETTLE:])}}

def _parameter_group_sha(groups):
    result={}
    for name,params in groups.items():
        h=hashlib.sha256()
        for i,p in enumerate(params):
            x=p.detach().cpu().contiguous();h.update(str(i).encode()+str(x.dtype).encode());h.update(np.asarray(x.shape,dtype='<i8').tobytes());h.update(x.numpy().tobytes())
        result[name]=h.hexdigest()
    return result
def _grad_summary(named_parameters):
    result={};allgrads=[]
    for name,params in named_parameters.items():
        grads=[p.grad for p in params if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError(f"nonfinite {name} gradient")
        norm=math.sqrt(float(sum(g.detach().double().square().sum() for g in grads))) if grads else 0.0
        result[name]={"norm":norm,"parameters_with_gradient":len(grads),"parameters_without_gradient":len(params)-len(grads)}
        allgrads.extend(grads)
    if not allgrads:raise FloatingPointError("no gradients in disposable update")
    return result

def _gradient_groups(f,encoder,binder,decoder):
    core=f.wrapped.core
    return {
      "graph":[p for p in core.graph_generator.parameters()],
      "kuramoto":[p for n,p in core.named_parameters() if "kuramoto" in n],
      "dendrite":[p for n,p in core.named_parameters() if "dendrit" in n or "dendric" in n],
      "membrane":[p for n,p in core.named_parameters() if "membrane" in n],
      "integration_scalars":[f.wrapped.a_d,f.wrapped.a_m,f.wrapped.b],
      "encoder":list(encoder.parameters()),
      "binder":list(binder.parameters()),"decoder":list(decoder.parameters())}

@torch.no_grad()
def _assignment_diagnostic_batch(spikes,rgb,tiles,binder,decoder):
    P,slots,_=binder(spikes);target=rgb.float()/255.;base=completion_loss(render_slot_rgb(P,slots,decoder,checkpoint_chunks=False),target,tiles)
    perm=torch.randperm(P.shape[1],generator=torch.Generator(device="cpu").manual_seed(142)).to(P.device)
    row_loss=completion_loss(render_slot_rgb(P[:,perm],slots,decoder,checkpoint_chunks=False),target,tiles)
    # Shuffle assignments only within each patch's coarse RGB-bin group.
    nchw=target.permute(0,3,1,2);means=torch.nn.functional.avg_pool2d(nchw,4,4).permute(0,2,3,1).reshape(P.shape[0],-1,3)
    bins=torch.floor(means*4).clamp(0,3).to(torch.int64);conditional=P.clone()
    for b in range(P.shape[0]):
        keys=bins[b].cpu().numpy();groups={}
        for i,key in enumerate(map(tuple,keys.tolist())):groups.setdefault(key,[]).append(i)
        rng=np.random.RandomState(142+b)
        for indices in groups.values():
            if len(indices)>1:
                order=np.asarray(indices)[rng.permutation(len(indices))]
                conditional[b,indices]=P[b,torch.as_tensor(order,device=P.device)]
    color_loss=completion_loss(render_slot_rgb(conditional,slots,decoder,checkpoint_chunks=False),target,tiles)
    return {"images":int(P.shape[0]),"unshuffled_completion_loss":float(base),"row_permuted_P_loss":float(row_loss),"within_coarse_RGB_bin_permuted_P_loss":float(color_loss),"row_permuted_minus_base":float(row_loss-base),"within_bin_permuted_minus_base":float(color_loss-base),"intervention":"P rows permuted; slot latents held fixed and P-derived geometry recomputed; color condition permutes P only within each target patch coarse RGB bin; no GT"}

def assignment_diagnostics_from_cache(cache,binder,decoder,*,device):
    rows=[]
    for start in range(0,64,MICRO):
        spikes,rgb,tiles=cache.batch(start,device);rows.append(_assignment_diagnostic_batch(spikes,rgb,tiles,binder,decoder))
    return {"scope":"first64 registered TRAIN rows","batches":rows,"mean_unshuffled_loss":float(np.mean([x["unshuffled_completion_loss"] for x in rows])),"mean_row_permuted_loss":float(np.mean([x["row_permuted_P_loss"] for x in rows])),"mean_within_bin_permuted_loss":float(np.mean([x["within_coarse_RGB_bin_permuted_P_loss"] for x in rows]))}

@torch.no_grad()
def assignment_diagnostics_live(f,binder,decoder,ids,rows,raw,*,device,exposure):
    batches=[]
    for start in range(0,64,MICRO):
        rgb=_raw_rows(rows[start:start+MICRO],raw,device);masked,tiles=mask_rgb(rgb,ids[start:start+MICRO],exposure)
        gamma=f.encode(masked);trace=f.rollout(gamma,total_steps=STEPS,live_tail_steps=0)
        batches.append(_assignment_diagnostic_batch(trace["component_spikes"][...,SETTLE:],rgb,tiles,binder,decoder))
    return {"scope":"first64 registered TRAIN rows","mask_exposure":exposure,"batches":batches,"mean_unshuffled_loss":float(np.mean([x["unshuffled_completion_loss"] for x in batches])),"mean_row_permuted_loss":float(np.mean([x["row_permuted_P_loss"] for x in batches])),"mean_within_bin_permuted_loss":float(np.mean([x["within_coarse_RGB_bin_permuted_P_loss"] for x in batches]))}

def preflight(arm,*,device="cuda:0",output_root=TRAIN):
    if arm not in ARMS:raise ValueError("unknown SW0142 arm")
    path=Path(output_root)/f"preflight_{arm}_seed1.json"
    if path.exists():raise FileExistsError("preserve existing preflight")
    warm_manifest,warm_state=_warm_state(device);f=source_foundation(device);src=sw130.source_contract(SEED)
    binder=NativeSpikeSlotBinder(seed=142).to(device);decoder=RelativeSlotRGBDecoder(seed=106).to(device);binder.load_state_dict(warm_state["binder"],strict=True);decoder.load_state_dict(warm_state["decoder"],strict=True)
    if (warm_manifest.get("source_core_sha256")!=src[5] or warm_manifest.get("source_manifest_sha256")!=sha256(src[1]) or warm_manifest.get("source_training_ids_sha256")!=f.provenance["source_training_ids_sha256"] or warm_manifest.get("rgb_asset_validation")!=f.provenance["rgb_asset_validation"]):raise ValueError("warm head/source/assets binding mismatch")
    backbone,heads,opt_b,opt_h=_optimizers(f,arm,binder,decoder,warm_state);before=_frozen_core_sha(f) if arm=="encoder_graph" else None
    family_params=_gradient_groups(f,f.encoder,binder,decoder);family_before=_parameter_group_sha(family_params);integration_before={k:getattr(f.wrapped,k).detach().cpu().tolist() for k in ('a_d','a_m','b')}
    ids,rows=ordered_train();raw=np.load(sw130.TRAIN_RGB,mmap_mode="r",allow_pickle=False);opt_b.zero_grad(set_to_none=True);opt_h.zero_grad(set_to_none=True);losses=[];start=time.perf_counter()
    measured_cuda=_begin_cuda_measurement(device)
    first_rgb=None;event_before=None
    for micro in range(4):
        pos=micro*MICRO;rgb=_raw_rows(rows[pos:pos+MICRO],raw,device)
        result=masked_forward(f,binder,decoder,rgb,ids[pos:pos+MICRO],90000,TAIL)
        if micro==0:
            first_rgb=rgb.detach().clone()
            event_before=_event_occupancy(result["trace"],f.wrapped.b)
        (result["loss"]/4).backward();losses.append(float(result["loss"].detach()))
    # Gradients were accumulated by the four microbatches; clip and step once.
    group_gradients=_grad_summary(_gradient_groups(f,f.encoder,binder,decoder))
    for name,row in group_gradients.items():
        if not math.isfinite(row["norm"]):raise FloatingPointError(f"nonfinite preflight family {name}")
    required=("graph","encoder","binder","decoder")
    missing=[name for name in required if group_gradients[name]["norm"]<=0]
    if missing:raise FloatingPointError(f"preflight has no RGB gradient for required path {missing}")
    norms={}
    for params,name,opt in ((backbone,"backbone",opt_b),(heads,"head",opt_h)):
        grads=[p.grad for p in params if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError(f"preflight {name} gradient invalid")
        if not grads:raise FloatingPointError(f"preflight {name} has no gradients")
        norm=math.sqrt(float(sum(g.detach().double().square().sum() for g in grads)));norms[name]=norm
        if not math.isfinite(norm) or norm<=0:raise FloatingPointError(f"preflight {name} gradient empty")
        torch.nn.utils.clip_grad_norm_(params,1.)
    opt_b.step();opt_h.step()
    if arm=="core_trainable":
        f.wrapped.project_integrations_()
    with torch.no_grad():
        first_masked,_=mask_rgb(first_rgb,ids[:MICRO],90000)
        after_trace=f.rollout(f.encode(first_masked),total_steps=STEPS,live_tail_steps=0)
        event_after=_event_occupancy(after_trace,f.wrapped.b)
    after=_frozen_core_sha(f) if arm=="encoder_graph" else None
    if before is not None and before!=after:raise AssertionError("encoder_graph arm changed frozen native core")
    family_after=_parameter_group_sha(family_params);integration_after={k:getattr(f.wrapped,k).detach().cpu().tolist() for k in ('a_d','a_m','b')}
    if measured_cuda:torch.cuda.synchronize(torch.device(device))
    rec={"status":"preflight_complete","experiment":NAME,"seed":1,"arm":arm,"optimizer_updates":1,"logical_batch":16,
         "source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],"rgb_asset_validation":f.provenance["rgb_asset_validation"],"warm_manifest_sha256":sha256(WARM/"manifest.json"),"gradient_norms":norms,"rgb_gradient_families":group_gradients,"family_parameter_sha256_before":family_before,"family_parameter_sha256_after":family_after,"integration_scalars_before":integration_before,"integration_scalars_after":integration_after,"event_occupancy_before":event_before,"event_occupancy_after_same_masked_train_microbatch":event_after,"loss_micro":losses,"frozen_core_sha256_before":before,"frozen_core_sha256_after":after,
         "elapsed_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(torch.device(device)) if measured_cuda else None,
         "peak_reserved_bytes":torch.cuda.max_memory_reserved(torch.device(device)) if measured_cuda else None,"implementation_fingerprint":fingerprint(),"ground_truth_used":False,"disposable_update_discarded":True}
    write_once(path,rec);return rec

def train_arm(arm,*,device="cuda:0",updates=UPDATES):
    if arm not in ARMS or updates!=UPDATES:raise ValueError("SW0142 pilot is fixed at256 updates")
    pf=TRAIN/f"preflight_{arm}_seed1.json";pr=json.loads(pf.read_text(encoding="utf-8"))
    if pr.get("status")!="preflight_complete" or pr.get("implementation_fingerprint")!=fingerprint():raise ValueError("fresh preflight missing/stale")
    target=arm_dir(arm)
    if target.exists():raise FileExistsError("preserve existing training output")
    warm_manifest,warm_state=_warm_state(device);f=source_foundation(device);src=sw130.source_contract(SEED)
    if (pr.get("warm_manifest_sha256")!=sha256(WARM/"manifest.json") or pr.get("source_core_sha256")!=src[5] or pr.get("source_manifest_sha256")!=sha256(src[1]) or pr.get("source_training_ids_sha256")!=f.provenance["source_training_ids_sha256"] or pr.get("rgb_asset_validation")!=f.provenance["rgb_asset_validation"]):raise ValueError("preflight/source/warm binding mismatch")
    binder=NativeSpikeSlotBinder(seed=142).to(device);decoder=RelativeSlotRGBDecoder(seed=106).to(device);binder.load_state_dict(warm_state["binder"],strict=True);decoder.load_state_dict(warm_state["decoder"],strict=True)
    backbone,heads,opt_b,opt_h=_optimizers(f,arm,binder,decoder,warm_state);ids,rows=ordered_train();raw=np.load(sw130.TRAIN_RGB,mmap_mode="r",allow_pickle=False);hist=[];target.mkdir(parents=True,exist_ok=False);start=time.time()
    frozen_before=_frozen_core_sha(f) if arm=="encoder_graph" else None
    source_initial_sha=_state_sha((("wrapped",f.wrapped),("encoder",f.encoder)))
    for u in range(UPDATES):
        opt_b.zero_grad(set_to_none=True);opt_h.zero_grad(set_to_none=True);losses=[];ordered=[]
        for micro in range(4):
            pos=u*BATCH+micro*MICRO;rgb=_raw_rows(rows[pos:pos+MICRO],raw,device);batch_ids=ids[pos:pos+MICRO];out=masked_forward(f,binder,decoder,rgb,batch_ids,u+1,TAIL);(out["loss"]/4).backward();losses.append(float(out["loss"].detach()));ordered+=batch_ids
        norms={}
        for params,name,opt in ((backbone,"backbone",opt_b),(heads,"head",opt_h)):
            grads=[p.grad for p in params if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError(f"update{u+1}: {name} gradients nonfinite")
            if not grads:raise FloatingPointError(f"update{u+1}: {name} has no gradients")
            norm=math.sqrt(float(sum(g.detach().double().square().sum() for g in grads)));norms[name]=norm
            if not math.isfinite(norm) or norm<=0:raise FloatingPointError(f"update{u+1}: empty {name} gradient")
            torch.nn.utils.clip_grad_norm_(params,1.)
        opt_b.step();opt_h.step()
        if arm=="core_trainable":
            with torch.no_grad():
                f.wrapped.project_integrations_()
        hist.append({"update":u+1,"image_ids":ordered,"loss_micro":losses,"gradient_norms":norms})
        if (u+1)%8==0:print(json.dumps({"stage":"train","arm":arm,"update":u+1,"total":UPDATES}),flush=True)
    f.wrapped.eval();f.encoder.eval();binder.eval();decoder.eval()
    final_diagnostic=assignment_diagnostics_live(f,binder,decoder,ids,rows,raw,device=device,exposure=1)
    frozen_after=_frozen_core_sha(f) if arm=="encoder_graph" else None
    if frozen_before is not None and frozen_before!=frozen_after:raise AssertionError("encoder_graph arm changed frozen native core during training")
    backbone_steps=_optimizer_step_range(opt_b);head_steps=_optimizer_step_range(opt_h)
    if backbone_steps!=[UPDATES,UPDATES] or head_steps!=[WARM_UPDATES+UPDATES,WARM_UPDATES+UPDATES]:raise AssertionError(f"Adam step counts mismatch: backbone={backbone_steps} head={head_steps}")
    cp=target/"checkpoint.pt";bo=target/"backbone_optimizer.pt";ho=target/"head_optimizer.pt";hp=target/"history.json"
    torch.save({"wrapped":f.wrapped.state_dict(),"encoder":f.encoder.state_dict(),"binder":binder.state_dict(),"decoder":decoder.state_dict()},cp);torch.save(opt_b.state_dict(),bo);torch.save(opt_h.state_dict(),ho);hp.write_text(json.dumps(hist,allow_nan=False),encoding="utf-8")
    m={"status":"training_complete","experiment":NAME,"seed":1,"arm":arm,"updates":UPDATES,"logical_batch":BATCH,"microbatch":MICRO,"exposures":4096,"warm_exposures":4096,
       "ordered_training_ids":ids,"ordered_training_ids_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest(),"source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],
       "warm_manifest_sha256":sha256(WARM/"manifest.json"),"warm_checkpoint_sha256":warm_manifest["checkpoint_sha256"],"warm_optimizer_sha256":warm_manifest["optimizer_sha256"],"source_initial_state_sha256":source_initial_sha,"frozen_core_sha256_before":frozen_before,"frozen_core_sha256_after":frozen_after,"backbone_adam_steps":backbone_steps,"head_adam_steps":head_steps,"assignment_diagnostic_first64_train":final_diagnostic,"preflight_sha256":sha256(pf),"implementation_fingerprint":fingerprint(),"checkpoint_sha256":sha256(cp),"backbone_optimizer_sha256":sha256(bo),"head_optimizer_sha256":sha256(ho),"history_sha256":sha256(hp),"ground_truth_used_for_training":False,"elapsed_seconds":time.time()-start}
    write_once(target/"manifest.json",m);write_once(target/"TRAINING_COMPLETED.json",{"status":"training_complete","manifest_sha256":sha256(target/"manifest.json"),"checkpoint_sha256":sha256(cp),"updates":UPDATES,"ground_truth_used":False});return m

def validate_training(arm, root=TRAIN):
    d=Path(root)/f"{arm}_seed1";mp=d/"manifest.json";marker=d/"TRAINING_COMPLETED.json"
    m=json.loads(mp.read_text(encoding="utf-8"));done=json.loads(marker.read_text(encoding="utf-8"))
    if m.get("status")!="training_complete" or m.get("experiment")!=NAME or m.get("seed")!=1 or m.get("arm")!=arm or m.get("updates")!=256 or m.get("logical_batch")!=16 or m.get("microbatch")!=4 or m.get("exposures")!=4096 or m.get("warm_exposures")!=4096 or m.get("ground_truth_used_for_training") is not False or m.get("implementation_fingerprint")!=fingerprint():raise ValueError("training manifest contract mismatch")
    if done.get("manifest_sha256")!=sha256(mp) or done.get("checkpoint_sha256")!=sha256(d/"checkpoint.pt") or done.get("updates")!=256:raise ValueError("training completion marker mismatch")
    for filename,key in (("checkpoint.pt","checkpoint_sha256"),("backbone_optimizer.pt","backbone_optimizer_sha256"),("head_optimizer.pt","head_optimizer_sha256"),("history.json","history_sha256")):
        if sha256(d/filename)!=m.get(key):raise ValueError(f"training artifact changed: {filename}")
    history=json.loads((d/"history.json").read_text(encoding="utf-8"))
    ids, _=ordered_train()
    if len(history)!=256 or any(row.get("update")!=i+1 or row.get("image_ids")!=ids[i*16:(i+1)*16] for i,row in enumerate(history)):raise ValueError("training history does not prove exact ordered4096 exposure")
    src=sw130.source_contract(SEED);warm=json.loads((WARM/"manifest.json").read_text(encoding="utf-8"))
    if (m.get("source_core_sha256")!=src[5] or m.get("source_manifest_sha256")!=sha256(src[1]) or m.get("source_training_ids_sha256")!=hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest() or m.get("warm_manifest_sha256")!=sha256(WARM/"manifest.json") or m.get("warm_checkpoint_sha256")!=warm.get("checkpoint_sha256") or m.get("warm_optimizer_sha256")!=warm.get("optimizer_sha256")):raise ValueError("training source/warm provenance mismatch")
    for filename,key,expected_step in (("backbone_optimizer.pt","backbone_optimizer_sha256",256),("head_optimizer.pt","head_optimizer_sha256",512)):
        state=torch.load(d/filename,map_location="cpu",weights_only=True);steps=[]
        for row in state.get("state",{}).values():
            v=row.get("step")
            if v is not None:steps.append(int(v.item()) if torch.is_tensor(v) else int(v))
        if not steps or min(steps)!=expected_step or max(steps)!=expected_step:raise ValueError(f"{filename} does not prove {expected_step} Adam steps")
    return {"directory":d,"manifest":m,"manifest_sha256":sha256(mp),"checkpoint":d/"checkpoint.pt"}

def _write_npz_once(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:np.savez_compressed(stream,**arrays)

@torch.no_grad()
def predict_arm(arm, *, device="cuda:0", output_root=RESULTS):
    if arm not in ARMS:raise ValueError("unknown arm")
    result_dir=Path(output_root)/arm
    if result_dir.exists():raise FileExistsError("preserve existing SW0142 prediction")
    trained=validate_training(arm);f=source_foundation(device);ckpt=torch.load(trained["checkpoint"],map_location=device,weights_only=True)
    f.wrapped.load_state_dict(ckpt["wrapped"],strict=True);f.encoder.load_state_dict(ckpt["encoder"],strict=True)
    binder=NativeSpikeSlotBinder(seed=142).to(device);decoder=RelativeSlotRGBDecoder(seed=106).to(device);binder.load_state_dict(ckpt["binder"],strict=True);decoder.load_state_dict(ckpt["decoder"],strict=True)
    raw=np.load(sw130.VAL_RGB,mmap_mode="r",allow_pickle=False);labels=np.empty((320,32,32),dtype=np.int64);qcc=np.empty_like(labels)
    ids=np.arange(1320,1640,dtype=np.int64);f.wrapped.eval();f.encoder.eval();binder.eval();decoder.eval()
    for j,image_id in enumerate(ids):
        rgb=torch.from_numpy(np.asarray(raw[int(image_id)-1320:int(image_id)-1319]).copy()).to(device)
        if rgb.dtype!=torch.uint8 or tuple(rgb.shape)!=(1,128,128,3):raise ValueError("registered eval RGB row shape/dtype mismatch")
        gamma=f.encode(rgb);trace=f.rollout(gamma,total_steps=STEPS,live_tail_steps=0)
        actual=trace["component_spikes"][...,SETTLE:]
        P,slots,_=binder(actual);labels[j]=canonical_hard_labels(P)[0].cpu().numpy();qcc[j]=sw136_eval._qcc_labels(trace)[0]
        if not np.isfinite(labels[j]).all() or not np.isfinite(qcc[j]).all():raise FloatingPointError("nonfinite prediction labels")
        if j%40==39:print(json.dumps({"stage":"predict","arm":arm,"predicted":j+1,"count":320}),flush=True)
    result_dir.mkdir(parents=True,exist_ok=False);npz=result_dir/"predictions.npz";_write_npz_once(npz,image_ids=ids,primary_labels=labels,qcc_labels=qcc)
    protocol={"experiment":NAME,"status":"complete","seed":1,"arm":arm,"image_ids":[1320,1639],"count":320,"grid_size":[32,32],"prediction_shape":[320,32,32],"steps":1024,"settle":512,"microbatch_size":1,"primary":"SW0135 P argmax / largest hard slot background","secondary":"SW0136 unchanged QCC .50/min8/largest-component background","source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],"encoder_sha256":f.provenance["encoder_sha256"],"feature_preprocessing_sha256":f.provenance["feature_preprocessing_sha256"],"rgb_asset_validation":f.provenance["rgb_asset_validation"],"training_manifest_sha256":trained["manifest_sha256"],"checkpoint_sha256":sha256(trained["checkpoint"]),"implementation_fingerprint":fingerprint(),"prediction_sha256":sha256(npz),"ground_truth_used_for_prediction":False,"optimizer_updates":0}
    write_once(result_dir/"protocol.json",protocol);return {"directory":str(result_dir),"prediction_sha256":protocol["prediction_sha256"],"protocol_sha256":sha256(result_dir/"protocol.json")}

def _own_prediction(arm):
    d=RESULTS/arm;p=d/"protocol.json";n=d/"predictions.npz";b=p.read_bytes();protocol=json.loads(b.decode("utf-8"));digest=sha256(n)
    trained=validate_training(arm)
    if (protocol.get("experiment")!=NAME or protocol.get("status")!="complete" or protocol.get("arm")!=arm or protocol.get("seed")!=1 or protocol.get("count")!=320 or protocol.get("prediction_shape")!=[320,32,32] or protocol.get("steps")!=1024 or protocol.get("settle")!=512 or protocol.get("ground_truth_used_for_prediction") is not False or protocol.get("prediction_sha256")!=digest or protocol.get("training_manifest_sha256")!=trained["manifest_sha256"] or protocol.get("checkpoint_sha256")!=sha256(trained["checkpoint"]) or protocol.get("implementation_fingerprint")!=fingerprint()):raise ValueError(f"invalid frozen SW0142 prediction {arm}")
    with np.load(n,allow_pickle=False) as z:
        if not {"image_ids","primary_labels","qcc_labels"}.issubset(z.files):raise ValueError("prediction NPZ missing keys")
        contract32.require_fixed_image_ids(z["image_ids"]);primary=np.asarray(z["primary_labels"]);qcc=np.asarray(z["qcc_labels"])
    for arr in (primary,qcc):
        if arr.shape!=(320,32,32) or not np.issubdtype(arr.dtype,np.integer) or np.any(arr<0) or not np.isfinite(arr).all():raise ValueError("invalid prediction label array")
    if sha256(n)!=digest or sha256(p)!=hashlib.sha256(b).hexdigest():raise ValueError("prediction changed during validation")
    return {"primary":primary.astype(np.int64,copy=False),"qcc":qcc.astype(np.int64,copy=False),"ids":np.arange(1320,1640,dtype=np.int64),"npz":n,"protocol":p,"protocol_data":protocol,"npz_sha256":digest,"protocol_sha256":hashlib.sha256(b).hexdigest()}

def _slot32_reference(slot_dir):
    slot_dir=Path(slot_dir);slot_npz=slot_dir/"predictions.npz";protocol_path=slot_dir/"protocol.json"
    protocol=json.loads(protocol_path.read_text(encoding="utf-8-sig"));digest=contract32.validate_slot_protocol(protocol,seed=1,prediction_path=slot_npz)
    with np.load(slot_npz,allow_pickle=False) as z:
        if not {"image_ids","labels"}.issubset(z.files):raise ValueError("Slot NPZ missing labels/IDs")
        ids=contract32.require_fixed_image_ids(z["image_ids"]);pixels=np.asarray(z["labels"])
    labels=contract32.slot_native_pixels_to32(pixels)
    protocol_sha=sha256(protocol_path)
    if sha256(slot_npz)!=digest or sha256(protocol_path)!=protocol_sha:raise ValueError("Slot reference changed while validating")
    return {"labels":labels,"ids":ids,"prediction_sha256":digest,"protocol_sha256":protocol_sha,"prediction_path":slot_npz,"protocol_path":protocol_path,"protocol":protocol}

def score(*, output=HERE/"results_archive"/"score_seed1.json", dataset=DATASET, source_root=SOURCE_ROOT, slot_root=SLOT_ROOT):
    output=Path(output)
    if output.exists():raise FileExistsError("preserve existing SW0142 scoring output")
    # Validate all predictions and references, and bind hashes before opening GT.
    own={arm:_own_prediction(arm) for arm in ARMS}
    if own[ARMS[0]]["protocol_data"]["training_manifest_sha256"]==own[ARMS[1]]["protocol_data"]["training_manifest_sha256"]:raise ValueError("paired arms unexpectedly share one training manifest")
    own_manifests=[validate_training(arm)["manifest"] for arm in ARMS]
    if (own_manifests[0].get("source_initial_state_sha256")!=own_manifests[1].get("source_initial_state_sha256") or own_manifests[0].get("warm_checkpoint_sha256")!=own_manifests[1].get("warm_checkpoint_sha256") or own_manifests[0].get("warm_optimizer_sha256")!=own_manifests[1].get("warm_optimizer_sha256")):raise ValueError("paired arms do not share the registered initial source and warm state")
    source=sw137.validate_prediction(Path(source_root)/"seed1",seed=1)
    slotdir=Path(slot_root)/"seed1_epoch10";slot=_slot32_reference(slotdir);slot_npz=slot["prediction_path"];slot_sha=slot["prediction_sha256"]
    slot_ids=slot["ids"];slot32=slot["labels"]
    if sha256(SLOT_BASELINE_PATH)!=SLOT_BASELINE_SHA256:raise ValueError("registered three-seed Slot32 baseline SHA mismatch")
    slot_baseline=json.loads(SLOT_BASELINE_PATH.read_text(encoding="utf-8"))["scores"]["slot32"]
    slot_mean={metric:float(slot_baseline[metric]["three_seed_mean"]) for metric in contract32.METRICS}
    if not np.array_equal(slot_ids,own[ARMS[0]]["ids"]) or not np.array_equal(source["ids"],own[ARMS[0]]["ids"]):raise ValueError("GT-last prediction references have different IDs")
    frozen={"own":{a:(r["npz_sha256"],r["protocol_sha256"]) for a,r in own.items()},"source":(source["prediction_sha256"],source["protocol_sha256"]),"slot":(slot_sha,slot["protocol_sha256"])}
    for arm,r in own.items():
        if (sha256(r["npz"]),sha256(r["protocol"]))!=frozen["own"][arm]:raise ValueError("own prediction changed before GT")
    if (sha256(source["archive_path"]),sha256(source["protocol_path"]))!=frozen["source"] or (sha256(slot_npz),sha256(slot["protocol_path"]))!=frozen["slot"]:raise ValueError("reference prediction changed before GT")
    import h5py
    with h5py.File(dataset,"r") as h:
        if "mask" not in h or h["mask"].shape[0]<1640:raise ValueError("GT dataset does not contain fixed320 mask rows")
        gt=contract32.modal_native32(np.asarray(h["mask"][1320:1640]))
    result={"status":"scored_complete","experiment":NAME,"seed":1,"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,"frozen_prediction_hashes":frozen,"scores":{}}
    for arm,r in own.items():result["scores"][arm]={"primary":contract32.production_metrics32(r["primary"],gt),"qcc":contract32.production_metrics32(r["qcc"],gt)}
    result["scores"]["source97_qcc"]={"qcc":contract32.production_metrics32(source["labels"],gt)};result["scores"]["slot32"]={"primary":contract32.production_metrics32(slot32,gt)}
    rng=np.random.RandomState(139);samples=rng.randint(0,320,size=(10000,320));paired={}
    for candidate,reference in (("core_trainable","encoder_graph"),("core_trainable","source97_qcc")):
        paired[reference]={}
        left=result["scores"][candidate]["primary"];right=result["scores"][reference].get("primary",result["scores"][reference].get("qcc"))
        for metric in contract32.METRICS:
            delta=np.asarray(left[metric]["per_image"],dtype=np.float64)-np.asarray(right[metric]["per_image"],dtype=np.float64)
            low,high=np.quantile(delta[samples].mean(1),[.025,.975]);paired[reference][metric]={"mean_difference":float(delta.mean()),"ci95":[float(low),float(high)]}
    candidate=result["scores"]["core_trainable"]["primary"]
    source_fg=result["scores"]["source97_qcc"]["qcc"]["fg_ari"]["mean"]
    control_fg=result["scores"]["encoder_graph"]["primary"]["fg_ari"]["mean"]
    result["paired_comparisons"]=paired;result["bootstrap"]={"method":"paired_image","samples":10000,"random_state":139,"common_indices":True}
    result["registered_slot_baseline"]={"path":str(SLOT_BASELINE_PATH.relative_to(ROOT)),"sha256":SLOT_BASELINE_SHA256,"three_seed_mean":slot_mean}
    gate={"foreground_ari_above_source_plus_0_01":candidate["fg_ari"]["mean"]>source_fg+.01,
      "foreground_ari_above_encoder_graph_plus_0_01":candidate["fg_ari"]["mean"]>control_fg+.01,
      "foreground_ari_paired_ci_lower_positive_vs_source":paired["source97_qcc"]["fg_ari"]["ci95"][0]>0,
      "foreground_ari_paired_ci_lower_positive_vs_encoder_graph":paired["encoder_graph"]["fg_ari"]["ci95"][0]>0,
      "foreground_iou_above_slot_plus_0_05":candidate["foreground_iou"]["mean"]>=slot_mean["foreground_iou"]+.05,
      "object_iou_above_slot_plus_0_05":candidate["matched_object_iou"]["mean"]>=slot_mean["matched_object_iou"]+.05}
    gate["may_justify_seed_expansion"]=all(gate.values())
    result["promotion_gate"]=gate
    write_once(output,result);return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("cache","warm","preflight","train","predict","score"),required=True);p.add_argument("--arm",choices=ARMS);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path);p.add_argument("--dataset",type=Path,default=DATASET)
    a=p.parse_args(argv)
    if a.stage in {"preflight","train","predict"} and not a.arm: p.error("--arm required")
    if a.stage=="cache":r=create_cache(device=a.device)
    elif a.stage=="warm":r=warm_head(device=a.device)
    elif a.stage=="preflight":r=preflight(a.arm,device=a.device)
    elif a.stage=="train":r=train_arm(a.arm,device=a.device)
    elif a.stage=="predict":r=predict_arm(a.arm,device=a.device)
    else:r=score(output=a.output or HERE/"results_archive"/"score_seed1.json",dataset=a.dataset)
    print(json.dumps(r,allow_nan=False),flush=True);return 0

if __name__=="__main__":raise SystemExit(main())
