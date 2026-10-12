"""SW0146 paired seed1 boundary-Q pilot; all results remain local until reviewed."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test"),str(ROOT/"snn_kuramoto_bidirectional")]
import h5py, numpy as np, torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135, evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding.foundation import late_rollout
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145
from collaborative_test.SW_0142_native32_masked_completion import run as sw142

NAME="SW0146_native32_boundary_q_pilot"; SEED=1; ARMS=("encoder_graph","joint")
STEPS,SETTLE,TAIL=1024,512,64; N,GRID,COUNT=1024,32,320
BATCH,MICRO,UPDATES=16,4,256; EXPOSED=4096
REPORT145=ROOT/"collaborative_test/SW_0145_native32_boundary_credit_probe/results_archive/source0_boundary_credit_probe.json"
SCORE143=sw143.SUMMARY
ADMISSION=HERE/"admission.json"
TRAIN=HERE/"training"; PREFLIGHT=HERE/"preflight"; PRED=HERE/"predictions"; ARCHIVE=HERE/"results_archive"
SOURCE137=sw137.OUTPUT_ROOT/"seed1"; SLOT_ROOT=ROOT/"trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE=ROOT/"collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA="4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
DATASET=Path(sw130.source97.DATASET)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def write_once(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(obj,f,indent=2,allow_nan=False);f.write("\n")
def fingerprint():
 merged={}; merged.update(sw135.implementation_fingerprint()); merged.update(sw137.implementation_fingerprint()); merged.update(sw143.implementation_fingerprint())
 paths=[HERE/"run.py",HERE/"protocol.json",ROOT/"collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",ROOT/"collaborative_test/SW_0145_native32_boundary_credit_probe/protocol.json",ROOT/"collaborative_test/SW_0143_native32_selective_geodesic/run.py",ROOT/"collaborative_test/SW_0136_native32_transfer/evaluate.py"]
 for p in paths:
  if not p.is_file():raise FileNotFoundError(p)
  merged[p.relative_to(ROOT).as_posix()]=sha(p)
 return merged
def implementation_sha():return hashlib.sha256(json.dumps(fingerprint(),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def source(seed=SEED):return sw130.source_contract(seed)
def validated_assets():return sw130.validate_rgb_assets()
def _report_sources(report):return report.get("source",{})
def validate_admission(path=ADMISSION):
 """Fail closed on exact upstream report bytes and their registered source lineage."""
 path=Path(path); admission=json.loads(path.read_text(encoding="utf-8"))
 if admission.get("status")!="admitted" or admission.get("experiment")!=NAME:raise ValueError("SW0146 admission record is not admitted")
 p145=Path(admission.get("sw0145_report_path", "")); p143=Path(admission.get("sw0143_score_path", ""))
 if p145.resolve()!=REPORT145.resolve() or p143.resolve()!=SCORE143.resolve():raise ValueError("admission paths must be the registered SW0145 and SW0143 artifacts")
 if sha(p145)!=admission.get("sw0145_report_sha256") or sha(p143)!=admission.get("sw0143_score_sha256"):raise ValueError("upstream report bytes differ from sealed admission digests")
 if sha(ROOT/"collaborative_test/SW_0145_native32_boundary_credit_probe/run.py")!="9b3db438992b55260ebdf6bbafe421bdf1b54c5522624f6c66a0b7b363180fbe":raise ValueError("SW0145 probe implementation changed")
 r=json.loads(p145.read_text(encoding="utf-8")); src0=source(0); report_source=_report_sources(r)
 assets=validated_assets(); ratios=r.get("gradient_comparison_after_disposable_step",{}).get("per_logical_batch_gradient_norms",[])
 vals=[row.get("ratio_B_over_A") for row in ratios]
 bn=[row.get("boundary_encoder_graph_norm") for row in ratios]; an=[row.get("source_q_encoder_graph_norm") for row in ratios]
 if not (r.get("update",{}).get("initial_source_q_baseline_exact_on_first16") is True and report_source.get("training_ids")==list(map(int,src0[4]))):raise ValueError("SW0145 did not bind exact source0 ordered IDs and first16 source-Q parity")
 for b,a,ratio in zip(bn,an,vals):
  if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (b,a,ratio)) or not math.isclose(ratio,b/a,rel_tol=1e-12,abs_tol=1e-15):raise ValueError("SW0145 B/A norms or ratio are invalid")
 changed=r.get("update",{}).get("changed_parameter_groups",{})
 if not (changed.get("encoder") or changed.get("graph")) or not (changed.get("native_core") or changed.get("integration_scalars")):raise ValueError("SW0145 source families lack required actual parameter credit")
 if (r.get("status")!="disposable_probe_complete" or r.get("experiment")!="SW0145_native32_boundary_credit_probe" or r.get("seed")!=0 or r.get("training_admission") is not False or r.get("ground_truth_used") is not False or r.get("implementation_sha256")!=sha(ROOT/"collaborative_test/SW_0145_native32_boundary_credit_probe/run.py") or admission.get("sw0145_protocol_sha256")!=sha(ROOT/"collaborative_test/SW_0145_native32_boundary_credit_probe/protocol.json")):
  raise ValueError("SW0145 report is not the exact completed disposable source0 probe")
 if len(vals)!=4 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in vals):raise ValueError("SW0145 report lacks four finite positive B/A ratios")
 lambda_a=.25*float(np.median(vals)); reported=r.get("gradient_comparison_after_disposable_step",{}).get("proposed_lambda_A")
 if not math.isfinite(lambda_a) or lambda_a<=0 or reported!=lambda_a:raise ValueError("SW0145 fixed lambda does not reproduce its four registered ratios")
 if (report_source.get("checkpoint_sha256")!=src0[5] or report_source.get("manifest_sha256")!=sha(src0[1]) or report_source.get("training_ids_sha256")!=hashlib.sha256(np.asarray(src0[4],dtype="<i8").tobytes()).hexdigest() or report_source.get("rgb_asset_validation")!=assets):raise ValueError("SW0145 source0 checkpoint/assets/ordered-ID provenance mismatch")
 s143=json.loads(p143.read_text(encoding="utf-8"))
 if s143.get("experiment")!="SW0143_native32_selective_geodesic" or s143.get("status")!="complete" or s143.get("seed_scope")!=[0,1,2] or s143.get("implementation")!=sw143.implementation_fingerprint():raise ValueError("SW0143 must have a valid terminal three-seed scientific score")
 scale=r.get("scale",{}).get("value")
 if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:raise ValueError("SW0145 boundary scale is invalid")
 return {"lambda_A":lambda_a,"scale":float(scale),"sw0145_report_sha256":sha(p145),"sw0143_score_sha256":sha(p143),"sw0145_source0_manifest_sha256":sha(src0[1]),"rgb_assets":assets,"admission_sha256":sha(path)}
def ordered_train():
 src=source(1); ids=list(map(int,src[4])); rows=np.asarray(src[3],dtype=np.int64)
 if len(ids)<EXPOSED or len(set(ids[:EXPOSED]))!=EXPOSED or rows.shape[0]<EXPOSED:raise ValueError("source1 ordered TRAIN prefix must contain4096 unique rows")
 return ids[:EXPOSED],rows[:EXPOSED],src
def arm_dir(arm,root=TRAIN):
 if arm not in ARMS:raise ValueError("unregistered arm")
 return Path(root)/f"{arm}_seed1"
def preflight_path(arm,root=PREFLIGHT):
 if arm not in ARMS:raise ValueError("unregistered arm")
 return Path(root)/f"{arm}_seed1.json"
def _source_foundation(device):
 f=sw135.load_native32_foundation(SEED,device,verify_rgb_assets=True); src=source(SEED)
 if (f.provenance.get("source_core_sha256")!=src[5] or list(map(int,f.image_ids))!=list(map(int,src[4])) or not np.array_equal(f.pool_indices,np.asarray(src[3],dtype=np.int64))):raise ValueError("native32 foundation does not match source1 checkpoint/order")
 if f.provenance.get("rgb_asset_validation")!=validated_assets():raise ValueError("native32 assets differ from current verified RGB assets")
 return f,src

def _groups(f,arm):
 core=f.wrapped.core; graph=list(core.graph_generator.parameters()); enc=list(f.encoder.parameters())
 native=[p for n,p in core.named_parameters() if not n.startswith("graph_generator.")]
 scalars=[f.wrapped.a_d,f.wrapped.a_m,f.wrapped.b]
 if arm=="encoder_graph":return {"encoder":enc,"graph":graph}
 return {"encoder":enc,"graph":graph,"native_core":native,"integration_scalars":scalars}
def _set_trainable(f,groups):
 members={id(p) for ps in groups.values() for p in ps}
 for p in f.wrapped.parameters():p.requires_grad_(id(p) in members)
 for p in f.encoder.parameters():p.requires_grad_(id(p) in members)
 f.wrapped.train();f.encoder.train()
def _optimizer(groups):
 lr={"encoder":3e-6,"graph":3e-5,"native_core":3e-5,"integration_scalars":1e-3}
 seen=[]; optgroups=[]
 for name,params in groups.items():
  optgroups.append({"params":params,"lr":lr[name],"name":name});seen+=params
 if len({id(p) for p in seen})!=len(seen):raise ValueError("duplicate optimizer parameters")
 return torch.optim.Adam(optgroups),seen
def _q(f,rgb):
 gamma=f.encode(rgb); tr=late_rollout(f.wrapped,gamma,total_steps=STEPS,live_tail_steps=TAIL)
 q=sw145._actual_q(tr)
 return q,tr
def _event_rate(tr,f):return sw145._event_rates(tr,f.wrapped)
def _param_sha(groups):return sw142._parameter_group_sha(groups)
def _norms(groups):
 out={}
 for name,ps in groups.items():
  vals=[p.grad.detach().double().square().sum() for p in ps if p.grad is not None]
  out[name]=math.sqrt(float(torch.stack(vals).sum().cpu())) if vals else 0.0
 if any(not math.isfinite(v) for v in out.values()):raise FloatingPointError("nonfinite source-family gradient")
 return out

def _admission_context(path=ADMISSION):
 ctx=validate_admission(path); ctx["admission_path"]=str(Path(path).resolve()); return ctx

def _rgb_rows(cache,rows,device):
 x=np.asarray(cache[np.asarray(rows,dtype=np.int64)]).copy()
 if x.dtype!=np.uint8 or x.shape[1:]!=(128,128,3):raise ValueError("registered native128 RGB cache mismatch")
 return torch.from_numpy(x).to(device)

def _initial_source_parity(student,teacher,cache,rows,device):
 max_abs=0.0; seen=0
 for start in range(0,BATCH,MICRO):
  rgb=_rgb_rows(cache,rows[start:start+MICRO],device)
  with torch.no_grad():
   q_student,_=_q(student,rgb); q_teacher,_=_q(teacher,rgb)
  if not torch.equal(q_student,q_teacher):raise AssertionError("fresh source student differs from frozen float32 teacher Q")
  max_abs=max(max_abs,float((q_student-q_teacher).abs().max().cpu()));seen+=rgb.shape[0]
 return {"exact":True,"images":seen,"shape":[seen,4,N,STEPS],"max_abs_difference":max_abs}

def _loss_micro(student,teacher,rgb,scale,lambda_a):
 q,tr=_q(student,rgb)
 with torch.no_grad(): tq,_=_q(teacher,rgb)
 edges=sw145.boundary_energy(rgb.detach().cpu().numpy()); target=torch.as_tensor(np.exp(-edges/scale),device=rgb.device,dtype=q.dtype)
 pairs=torch.as_tensor(sw145.boundary_pairs(),device=rgb.device,dtype=torch.long)
 b=sw145.boundary_q_loss(q,target,pairs); a=sw145.source_q_preservation_loss(q,tq)
 loss=b+float(lambda_a)*a
 if not torch.isfinite(loss):raise FloatingPointError("nonfinite SW0146 joint objective")
 return loss,b,a,tr

def _first64_boundary_diagnostic(student,teacher,raw,rows,scale,device):
 """Describe clean TRAIN seam-Q/event behavior; diagnostic cannot select or tune."""
 student.wrapped.eval();student.encoder.eval();teacher.wrapped.eval();teacher.encoder.eval()
 live=[p.requires_grad for p in student.wrapped.parameters()];live_e=[p.requires_grad for p in student.encoder.parameters()]
 for p in student.wrapped.parameters():p.requires_grad_(False)
 for p in student.encoder.parameters():p.requires_grad_(False)
 pairs=sw145.boundary_pairs();pi=torch.as_tensor(pairs,dtype=torch.long,device=device)
 keys=("edge_energy","student_q","source_q","student_boundary_mse","source_boundary_mse","student_event","source_event")
 rows_data={name:{key:[] for key in keys} for name in ("e_zero","zero_lt_e_lt_s","e_ge_s")}
 try:
  for start in range(0,64,MICRO):
   rgb=_rgb_rows(raw,rows[start:start+MICRO],device)
   with torch.no_grad():
    qs,ts=_q(student,rgb);qt,tt=_q(teacher,rgb)
    energy=sw145.boundary_energy(rgb.detach().cpu().numpy());target=np.exp(-energy/scale)
    threshold_s=(.06*torch.exp(student.wrapped.b.detach())).view(1,4,1,1)
    threshold_t=(.06*torch.exp(teacher.wrapped.b.detach())).view(1,4,1,1)
    es=(ts["component_membrane"].detach()>threshold_s)[...,SETTLE:].float().mean((1,3))
    et=(tt["component_membrane"].detach()>threshold_t)[...,SETTLE:].float().mean((1,3))
    qs=qs[:,pi[:,0],pi[:,1]].cpu().numpy();qt=qt[:,pi[:,0],pi[:,1]].cpu().numpy()
    es=(es[:,pi[:,0]]+es[:,pi[:,1]]).mul(.5).cpu().numpy();et=(et[:,pi[:,0]]+et[:,pi[:,1]]).mul(.5).cpu().numpy()
   masks={"e_zero":energy==0,"zero_lt_e_lt_s":(energy>0)&(energy<scale),"e_ge_s":energy>=scale}
   for name,mask in masks.items():
    d=rows_data[name]
    for key,val in (("edge_energy",energy),("student_q",qs),("source_q",qt),
                    ("student_boundary_mse",(qs-target)**2),("source_boundary_mse",(qt-target)**2),
                    ("student_event",es),("source_event",et)):
     d[key].append(np.asarray(val)[mask])
   del rgb,qs,qt,ts,tt
 finally:
  for p,flag in zip(student.wrapped.parameters(),live):p.requires_grad_(flag)
  for p,flag in zip(student.encoder.parameters(),live_e):p.requires_grad_(flag)
 records={}
 for name,columns in rows_data.items():
  arrays={key:(np.concatenate(values) if values else np.empty(0,dtype=np.float64)) for key,values in columns.items()}
  n=len(arrays["edge_energy"])
  records[name]={"boundary_count":n,**{key:float(value.mean()) if n else None for key,value in arrays.items()},
   "student_minus_source_q":float((arrays["student_q"]-arrays["source_q"]).mean()) if n else None,
   "student_minus_source_event":float((arrays["student_event"]-arrays["source_event"]).mean()) if n else None}
 return {"scope":"first64 registered clean source1 TRAIN images; descriptive only; no GT, tuning, or selection",
         "scale":float(scale),"records":records}

def _frozen_native_sha(f):
 ps=[(n,p) for n,p in f.wrapped.named_parameters() if not n.startswith("core.graph_generator.")]
 return sw142._parameter_group_sha({"frozen_native_core":[p for _,p in ps]})["frozen_native_core"]

def _valid_preflight(arm,path,ctx):
 row=json.loads(Path(path).read_text(encoding="utf-8"))
 ids,_,src=ordered_train()
 if (row.get("status")!="preflight_complete" or row.get("experiment")!=NAME or row.get("arm")!=arm or row.get("seed")!=SEED or row.get("implementation_sha256")!=implementation_sha() or row.get("lambda_A")!=ctx["lambda_A"] or row.get("scale")!=ctx["scale"] or row.get("source_core_sha256")!=src[5] or row.get("source_manifest_sha256")!=sha(src[1]) or row.get("source_training_ids_sha256")!=hashlib.sha256(np.asarray(src[4],dtype="<i8").tobytes()).hexdigest() or row.get("train_ids")!=ids or row.get("train_ids_sha256")!=hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest()):
  raise ValueError(f"invalid SW0146 {arm} preflight contract")
 if row.get("viable_for_training") is not True or row.get("initial_source_Q_parity",{}).get("exact") is not True or row.get("ground_truth_used") is not False:
  raise ValueError(f"SW0146 {arm} preflight did not establish registered viability")
 return row

def preflight_arm(arm,*,device="cuda:0",admission=ADMISSION,output_root=PREFLIGHT):
 if arm not in ARMS:raise ValueError("unregistered SW0146 arm")
 out=preflight_path(arm,output_root)
 if out.exists():raise FileExistsError(f"preserve existing SW0146 preflight {out}")
 ctx=_admission_context(admission); dev=torch.device(device)
 if dev.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("SW0146 preflight requires an exclusive CUDA device")
 torch.cuda.set_device(dev);torch.cuda.init();torch.cuda.synchronize(dev);torch.cuda.reset_peak_memory_stats(dev);started=time.perf_counter()
 f,src=_source_foundation(dev);teacher,_=_source_foundation(dev)
 for p in teacher.wrapped.parameters():p.requires_grad_(False)
 for p in teacher.encoder.parameters():p.requires_grad_(False)
 teacher.wrapped.eval();teacher.encoder.eval()
 groups=_groups(f,arm);_set_trainable(f,groups);opt,params=_optimizer(groups)
 ids,rows,_=ordered_train();cache=np.load(sw130.TRAIN_RGB,mmap_mode="r")
 parity=_initial_source_parity(f,teacher,cache,rows,dev)
 pre_sha=_param_sha(groups);frozen_before=_frozen_native_sha(f)
 opt.zero_grad(set_to_none=True);losses=[];event=[]
 for i in range(0,BATCH,MICRO):
  rgb=_rgb_rows(cache,rows[i:i+MICRO],dev); loss,b,a,tr=_loss_micro(f,teacher,rgb,ctx["scale"],ctx["lambda_A"])
  (loss/(BATCH//MICRO)).backward();losses.append({"joint":float(loss.detach().cpu()),"boundary":float(b.detach().cpu()),"source_Q":float(a.detach().cpu())});event.append(sw145._event_rates(tr,f.wrapped));del rgb,loss,b,a,tr
 grads=_norms(groups)
 if any(not math.isfinite(x) for x in grads.values()):raise FloatingPointError("preflight gradients are nonfinite")
 torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
 if arm=="joint":
  with torch.no_grad():f.wrapped.a_d.clamp_(0,1);f.wrapped.a_m.clamp_(0,1)
 post_sha=_param_sha(groups);changed={k:pre_sha[k]!=post_sha[k] for k in pre_sha}
 frozen_after=_frozen_native_sha(f)
 if arm=="encoder_graph" and frozen_after!=frozen_before:raise AssertionError("encoder_graph preflight modified frozen native core")
 if arm=="encoder_graph": viable=(grads.get("encoder",0)>0 or grads.get("graph",0)>0) and any(changed.values())
 else: viable=(grads.get("native_core",0)>0 or grads.get("integration_scalars",0)>0) and (changed.get("native_core",False) or changed.get("integration_scalars",False))
 torch.cuda.synchronize(dev)
 record={"status":"preflight_complete","experiment":NAME,"arm":arm,"seed":SEED,"updates":1,"disposable_only":True,"viable_for_training":bool(viable),"initial_source_Q_parity":parity,"lambda_A":ctx["lambda_A"],"scale":ctx["scale"],"source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],"rgb_asset_validation":f.provenance["rgb_asset_validation"],"train_ids_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest(),"train_ids":ids,"gradient_norms":grads,"pre_parameter_sha256":pre_sha,"post_parameter_sha256":post_sha,"changed_groups":changed,"frozen_native_core_sha256_before":frozen_before,"frozen_native_core_sha256_after":frozen_after,"microbatch_losses":losses,"event_rates":event,"admission":ctx,"implementation_sha256":implementation_sha(),"ground_truth_used":False,"cuda":{"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(dev)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(dev)),"elapsed_seconds":time.perf_counter()-started}}
 write_once(out,record);return record

def _require_both_preflights(ctx,root=PREFLIGHT):
 rows={arm:_valid_preflight(arm,preflight_path(arm,root),ctx) for arm in ARMS}
 for row in rows.values():
  if row.get("source_core_sha256")!=sw130.source97.EXPECTED_SOURCE_SHAS[SEED] or row.get("source_manifest_sha256")!=sha(source(SEED)[1]):raise ValueError("paired preflights do not bind current source1 lineage")
 return rows

def train_arm(arm,*,device="cuda:0",admission=ADMISSION,preflight_root=PREFLIGHT,output_root=TRAIN,updates=UPDATES):
 if arm not in ARMS or updates!=UPDATES:raise ValueError("SW0146 arms/updates are fixed")
 ctx=_admission_context(admission);preflights=_require_both_preflights(ctx,preflight_root)
 out=arm_dir(arm,output_root)
 if out.exists():raise FileExistsError(f"preserve existing SW0146 training output {out}")
 dev=torch.device(device);f,src=_source_foundation(dev);teacher,_=_source_foundation(dev)
 for p in teacher.wrapped.parameters():p.requires_grad_(False)
 for p in teacher.encoder.parameters():p.requires_grad_(False)
 teacher.wrapped.eval();teacher.encoder.eval()
 groups=_groups(f,arm);_set_trainable(f,groups);opt,params=_optimizer(groups)
 ids,rows,_=ordered_train();cache=np.load(sw130.TRAIN_RGB,mmap_mode="r")
 parity=_initial_source_parity(f,teacher,cache,rows,dev);before_frozen=_frozen_native_sha(f);before_sha=_param_sha(groups)
 out.mkdir(parents=True,exist_ok=False);history=[];started=time.perf_counter()
 for update in range(UPDATES):
  opt.zero_grad(set_to_none=True); parts=[]
  for micro in range(4):
   start=update*BATCH+micro*MICRO;rgb=_rgb_rows(cache,rows[start:start+MICRO],dev)
   loss,b,a,_tr=_loss_micro(f,teacher,rgb,ctx["scale"],ctx["lambda_A"])
   (loss/4).backward();parts.append({"loss":float(loss.detach().cpu()),"boundary":float(b.detach().cpu()),"source_Q":float(a.detach().cpu())})
   del loss,b,a,_tr,rgb
  grad=_norms(groups);torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
  if arm=="joint":
   with torch.no_grad():f.wrapped.a_d.clamp_(0,1);f.wrapped.a_m.clamp_(0,1)
  history.append({"update":update+1,"image_ids":ids[update*BATCH:(update+1)*BATCH],"microbatches":parts,"gradient_norms":grad})
  if (update+1)%16==0:print(json.dumps({"stage":"train","arm":arm,"update":update+1,"total":UPDATES}),flush=True)
 diagnostic=_first64_boundary_diagnostic(f,teacher,cache,rows,ctx["scale"],dev)
 diagnostic_path=out/"first64_boundary_diagnostic.json"
 write_once(diagnostic_path,diagnostic)
 after_frozen=_frozen_native_sha(f);after_sha=_param_sha(groups)
 if arm=="encoder_graph" and before_frozen!=after_frozen:raise AssertionError("encoder_graph training changed frozen native core")
 ck=out/"checkpoint.pt";op=out/"optimizer.pt";hp=out/"history.json"
 torch.save({"wrapped":f.wrapped.state_dict(),"encoder":f.encoder.state_dict()},ck);torch.save(opt.state_dict(),op)
 with hp.open("x",encoding="utf-8",newline="\n") as stream:json.dump(history,stream,allow_nan=False);stream.write("\n")
 manifest={"status":"training_complete","experiment":NAME,"seed":SEED,"arm":arm,"updates":UPDATES,"logical_batch":BATCH,"microbatch":MICRO,"exposures":UPDATES*BATCH,"lambda_A":ctx["lambda_A"],"scale":ctx["scale"],"checkpoint_sha256":sha(ck),"optimizer_sha256":sha(op),"history_sha256":sha(hp),"first64_boundary_diagnostic_path":diagnostic_path.name,"first64_boundary_diagnostic_sha256":sha(diagnostic_path),"source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],"rgb_asset_validation":f.provenance["rgb_asset_validation"],"train_ids_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest(),"train_ids":ids,"preflight_sha256":{a:sha(preflight_path(a,preflight_root)) for a in ARMS},"preflight_artifact_sha256":{a:preflights[a]["post_parameter_sha256"] for a in ARMS},"initial_source_Q_parity":parity,"trainable_groups":list(groups),"parameter_sha256_before":before_sha,"parameter_sha256_after":after_sha,"frozen_native_core_sha256_before":before_frozen,"frozen_native_core_sha256_after":after_frozen,"optimizer_step_range":_optimizer_range(opt),"implementation_fingerprint":fingerprint(),"implementation_sha256":implementation_sha(),"admission":ctx,"ground_truth_used":False,"elapsed_seconds":time.perf_counter()-started}
 write_once(out/"manifest.json",manifest);return manifest

def _optimizer_range(opt):
 vals=[]
 for state in opt.state.values():
  step=state.get("step")
  if step is not None:vals.append(int(step.item()) if torch.is_tensor(step) else int(step))
 return [min(vals),max(vals)] if vals else [0,0]
def validate_training(arm,root=TRAIN):
 d=arm_dir(arm,root);mp=d/"manifest.json";m=json.loads(mp.read_text(encoding="utf-8"));ck=d/"checkpoint.pt";op=d/"optimizer.pt";hp=d/"history.json"
 if (m.get("status")!="training_complete" or m.get("experiment")!=NAME or m.get("arm")!=arm or m.get("seed")!=SEED or m.get("updates")!=UPDATES or m.get("exposures")!=EXPOSED or m.get("implementation_sha256")!=implementation_sha() or m.get("checkpoint_sha256")!=sha(ck) or m.get("optimizer_sha256")!=sha(op) or m.get("history_sha256")!=sha(hp) or m.get("optimizer_step_range")!=[UPDATES,UPDATES]):raise ValueError(f"invalid training artifact {arm}")
 ctx=_admission_context()
 if m.get("admission")!=ctx:raise ValueError(f"training admission lineage changed for {arm}")
 src=source(SEED)
 if (m.get("source_core_sha256")!=src[5] or m.get("source_manifest_sha256")!=sha(src[1]) or m.get("source_training_ids_sha256")!=hashlib.sha256(np.asarray(src[4],dtype="<i8").tobytes()).hexdigest() or m.get("rgb_asset_validation")!=validated_assets() or m.get("train_ids")!=ordered_train()[0]):raise ValueError(f"training source/assets/ordered IDs changed for {arm}")
 return {"manifest":m,"manifest_sha256":sha(mp),"checkpoint":ck,"optimizer":op,"history":hp}

def _train_manifest_hash(arm):return sha(arm_dir(arm)/"manifest.json")
def _prediction_dir(arm,root=PRED):
 if arm not in ARMS:raise ValueError("unregistered arm")
 return Path(root)/f"{arm}_seed1"
def predict_arm(arm,*,device="cuda:0",training_root=TRAIN,output_root=PRED):
 trained=validate_training(arm,training_root); out=_prediction_dir(arm,output_root)
 if out.exists():raise FileExistsError(f"preserve existing prediction directory {out}")
 f,_=_source_foundation(device); state=torch.load(trained["checkpoint"],map_location=device,weights_only=True)
 f.wrapped.load_state_dict(state["wrapped"],strict=True);f.encoder.load_state_dict(state["encoder"],strict=True);f.wrapped.eval();f.encoder.eval()
 cache=np.load(sw130.VAL_RGB,mmap_mode="r",allow_pickle=False)
 if cache.dtype!=np.uint8 or cache.shape[0]<COUNT or cache.shape[1:]!=(128,128,3):raise ValueError("registered fixed320 validation RGB cache mismatch")
 labels=np.empty((COUNT,GRID,GRID),dtype=np.int64)
 with torch.no_grad():
  for i in range(COUNT):
   rgb=torch.from_numpy(np.asarray(cache[i:i+1]).copy()).to(device);gamma=f.encode(rgb);_,qcc=sw136_eval._predict_one(f,None,gamma,primary=False)
   qcc=np.asarray(qcc)
   if qcc.shape!=(1,GRID,GRID) or not np.isfinite(qcc).all():raise ValueError("invalid actual-spike QCC prediction")
   labels[i]=qcc[0]
   if (i+1)%40==0:print(json.dumps({"stage":"predict","arm":arm,"count":i+1}),flush=True)
 ids=np.asarray(contract.IMAGE_IDS,dtype=np.int64);out.mkdir(parents=True,exist_ok=False);npz=out/"predictions.npz"
 with npz.open("xb") as fobj:np.savez_compressed(fobj,image_ids=ids,qcc_labels=labels)
 protocol={"experiment":NAME,"status":"complete","arm":arm,"seed":SEED,"image_ids":[1320,1639],"count":COUNT,"grid_size":[GRID,GRID],"prediction_shape":[COUNT,GRID,GRID],"steps":STEPS,"settle":SETTLE,"live_tail":TAIL,"ground_truth_used_for_prediction":False,"optimizer_updates":0,"training_manifest_sha256":trained["manifest_sha256"],"checkpoint_sha256":trained["manifest"]["checkpoint_sha256"],"source_core_sha256":f.provenance["source_core_sha256"],"source_manifest_sha256":f.provenance["source_manifest_sha256"],"source_training_ids_sha256":f.provenance["source_training_ids_sha256"],"rgb_asset_validation":f.provenance["rgb_asset_validation"],"prediction_sha256":sha(npz),"implementation_fingerprint":fingerprint()}
 write_once(out/"protocol.json",protocol);return protocol

def validate_prediction(arm,root=PRED,training_root=TRAIN):
 d=_prediction_dir(arm,root);n=d/"predictions.npz";p=d/"protocol.json";raw=p.read_bytes();psha=hashlib.sha256(raw).hexdigest();protocol=json.loads(raw.decode("utf-8"));trained=validate_training(arm,training_root)
 if (protocol.get("experiment")!=NAME or protocol.get("status")!="complete" or protocol.get("arm")!=arm or protocol.get("seed")!=SEED or protocol.get("image_ids")!=[1320,1639] or protocol.get("count")!=COUNT or protocol.get("prediction_shape")!=[COUNT,GRID,GRID] or protocol.get("steps")!=STEPS or protocol.get("settle")!=SETTLE or protocol.get("live_tail")!=TAIL or protocol.get("ground_truth_used_for_prediction") is not False or protocol.get("training_manifest_sha256")!=trained["manifest_sha256"] or protocol.get("checkpoint_sha256")!=trained["manifest"]["checkpoint_sha256"] or protocol.get("implementation_fingerprint")!=fingerprint() or protocol.get("prediction_sha256")!=sha(n)):
  raise ValueError(f"invalid frozen SW0146 {arm} prediction protocol")
 with np.load(n,allow_pickle=False) as z:
  if not {"image_ids","qcc_labels"}.issubset(z.files):raise ValueError("SW0146 prediction missing arrays")
  ids=contract.require_fixed_image_ids(z["image_ids"]);labels=np.asarray(z["qcc_labels"])
 if labels.shape!=(COUNT,GRID,GRID) or not np.issubdtype(labels.dtype,np.integer) or np.issubdtype(labels.dtype,np.bool_) or np.any(labels<0) or np.any(labels>1024):raise ValueError("invalid SW0146 fixed320 QCC labels")
 if sha(n)!=protocol["prediction_sha256"] or sha(p)!=psha:raise ValueError("SW0146 prediction bytes changed during validation")
 return {"ids":ids,"labels":labels.astype(np.int64,copy=False),"protocol":protocol,"npz":n,"protocol_path":p,"npz_sha256":protocol["prediction_sha256"],"protocol_sha256":psha}

def _slot_reference():
 d=SLOT_ROOT/"seed1_epoch10";npz=d/"predictions.npz";protocol_path=d/"protocol.json";raw=protocol_path.read_bytes();protocol=json.loads(raw.decode("utf-8-sig"));digest=contract.validate_slot_protocol(protocol,seed=1,prediction_path=npz)
 with np.load(npz,allow_pickle=False) as z:
  if not {"image_ids","labels"}.issubset(z.files):raise ValueError("Slot prediction archive missing labels/IDs")
  ids=contract.require_fixed_image_ids(z["image_ids"]);pixels=np.asarray(z["labels"])
 labels=contract.slot_native_pixels_to32(pixels);protocol_sha=hashlib.sha256(raw).hexdigest()
 if sha(npz)!=digest or sha(protocol_path)!=protocol_sha:raise ValueError("Slot reference changed during validation")
 return {"ids":ids,"labels":labels,"npz":npz,"protocol_path":protocol_path,"npz_sha256":digest,"protocol_sha256":protocol_sha,"protocol":protocol}

def _bootstrap_delta(left,right,draws):
 d=np.asarray(left,dtype=np.float64)-np.asarray(right,dtype=np.float64)
 low,high=np.quantile(d[draws].mean(axis=1),[.025,.975]);return {"mean_difference":float(d.mean()),"ci95":[float(low),float(high)]}
def score_all(*,output=ARCHIVE/"evaluation_seed1.json",dataset=DATASET):
 output=Path(output)
 if output.exists():raise FileExistsError("preserve existing SW0146 score output")
 # Complete validation/hash freeze of both arms and fixed source/Slot references before GT.
 own={a:validate_prediction(a) for a in ARMS};source_row=sw137.validate_prediction(SOURCE137,seed=SEED);slot=_slot_reference()
 for a in ARMS:
  if not np.array_equal(own[a]["ids"],source_row["ids"]) or not np.array_equal(own[a]["ids"],slot["ids"]):raise ValueError("SW0146/source/Slot fixed image IDs differ")
 if sha(SLOT_BASELINE)!=SLOT_BASELINE_SHA:raise ValueError("registered Slot32 baseline SHA changed")
 baseline=json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))
 for row in own.values():
  if sha(row["npz"])!=row["npz_sha256"] or sha(row["protocol_path"])!=row["protocol_sha256"]:raise ValueError("SW0146 prediction changed before GT")
 if sha(source_row["archive_path"])!=source_row["prediction_sha256"] or sha(source_row["protocol_path"])!=source_row["protocol_sha256"]:raise ValueError("source97 prediction changed before GT")
 if sha(slot["npz"])!=slot["npz_sha256"] or sha(slot["protocol_path"])!=slot["protocol_sha256"] or sha(SLOT_BASELINE)!=SLOT_BASELINE_SHA:raise ValueError("Slot references changed before GT")
 with h5py.File(dataset,"r") as h5:
  if "mask" not in h5 or h5["mask"].shape[0]<=1639:raise ValueError("fixed320 target masks unavailable")
  pixels=np.asarray(h5["mask"][1320:1640])
 gt=contract.modal_native32(pixels)
 scores={a:contract.production_metrics32(own[a]["labels"],gt) for a in ARMS};scores["source97_seed1"]=contract.production_metrics32(source_row["labels"],gt);scores["Slot32_seed1"]=contract.production_metrics32(slot["labels"],gt)
 slot_means={m:float(baseline["scores"]["slot32"][m]["three_seed_mean"]) for m in contract.METRICS}
 draws=np.random.RandomState(139).randint(0,COUNT,size=(10000,COUNT));paired={}
 for ref,scorekey in (("source97","source97_seed1"),("encoder_graph","encoder_graph")):
  paired[f"joint_minus_{ref}"]={m:_bootstrap_delta(scores["joint"][m]["per_image"],scores[scorekey][m]["per_image"],draws) for m in contract.METRICS}
 means={a:{m:float(scores[a][m]["mean"]) for m in contract.METRICS} for a in (*ARMS,"source97_seed1","Slot32_seed1")}
 gate={"joint_fg_at_least_source_plus_0_01":means["joint"]["fg_ari"]>=means["source97_seed1"]["fg_ari"]+.01,"joint_fg_at_least_control_plus_0_01":means["joint"]["fg_ari"]>=means["encoder_graph"]["fg_ari"]+.01,"joint_control_fg_ci_lower_positive":paired["joint_minus_encoder_graph"]["fg_ari"]["ci95"][0]>0,"joint_source_fg_ci_lower_positive":paired["joint_minus_source97"]["fg_ari"]["ci95"][0]>0,"joint_foreground_iou_above_slot_three_seed_plus_0_05":means["joint"]["foreground_iou"]>slot_means["foreground_iou"]+.05,"joint_matched_object_iou_above_slot_three_seed_plus_0_05":means["joint"]["matched_object_iou"]>slot_means["matched_object_iou"]+.05}
 result={"experiment":NAME,"status":"complete","seed":SEED,"scores":scores,"means":means,"registered_slot32_three_seed_means":slot_means,"paired_bootstrap":paired,"bootstrap":{"samples":10000,"rng":"numpy RandomState","seed":139,"common_image_indices":True},"promotion_gate":gate,"may_promote_seed_expansion":all(gate.values()),"provenance":{"predictions":{a:own[a]["npz_sha256"] for a in ARMS},"source97":source_row["prediction_sha256"],"slot":slot["npz_sha256"],"slot_baseline_sha256":SLOT_BASELINE_SHA},"ground_truth_opened_after_all_predictions_validated":True,"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,"limits":"Paired seed1 native32 pilot only; no scaling or universal gate claim."}
 write_once(output,result);return result

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("preflight","train","predict","score"),required=True);p.add_argument("--arm",choices=ARMS);p.add_argument("--device",default="cuda:0");p.add_argument("--admission",type=Path,default=ADMISSION);p.add_argument("--output",type=Path);a=p.parse_args(argv)
 if a.stage!="score" and a.arm is None:p.error("--arm required")
 if a.stage=="preflight":r=preflight_arm(a.arm,device=a.device,admission=a.admission)
 elif a.stage=="train":r=train_arm(a.arm,device=a.device,admission=a.admission)
 elif a.stage=="predict":r=predict_arm(a.arm,device=a.device)
 else:r=score_all(output=a.output or ARCHIVE/"evaluation_seed1.json")
 print(json.dumps({"experiment":NAME,"status":r.get("status"),"arm":r.get("arm"),"training_admission":r.get("viable_for_training")},allow_nan=False),flush=True);return r
if __name__=="__main__":main()
