"""Validated seed1 discovery plus prospective seed0/2 summary, common-image bootstrap."""
from __future__ import annotations
import hashlib,json,os,uuid
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
SEEDS=(0,1,2); BOOTSTRAP_SEED=165; REPLICATES=10000
SEED1_SCORE=ROOT/"collaborative_test/SW_0164_foreground_preserving_partition/results_archive/evaluation_seed1_20261014.json"
SEED1_SCORE_SHA256="978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"
OUTPUT=HERE/"results_archive"/"replication_summary_20261014.json"
METRICS=("fg_ari","foreground_iou","matched_object_iou")

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def paired_seed_bootstrap(differences):
 d=np.asarray(differences,dtype=np.float64)
 if d.ndim!=2 or d.shape[0] not in (2,3) or d.shape[1]!=320 or not np.isfinite(d).all(): raise ValueError("bootstrap expects two or three seed x fixed320 paired deltas")
 rng=np.random.RandomState(BOOTSTRAP_SEED); ix=rng.randint(0,320,size=(REPLICATES,320))
 # Same sampled image columns are applied to all three seeds in every replicate.
 vals=d[:,ix].mean(axis=(0,2))
 return {"mean":float(d.mean()),"ci95":[float(x) for x in np.percentile(vals,[2.5,97.5])],"seed":BOOTSTRAP_SEED,"replicates":REPLICATES,"common_image_indices_across_seeds":True}

def _per_image(report,arm,metric):
 value=np.asarray(report["scores"][arm][metric]["per_image"],dtype=np.float64)
 if value.shape!=(320,) or not np.isfinite(value).all(): raise ValueError("invalid per-image score vector")
 return value

def summarize(validated_reports,*,output=OUTPUT):
 """Consume already validated score reports keyed exactly by seeds 0,1,2."""
 if set(validated_reports)!={0,1,2}: raise ValueError("all three validated seed reports are required")
 reports={int(k):v for k,v in validated_reports.items()}
 for seed,r in reports.items():
  if r.get("status")!="complete" or r.get("seed_scope")!=[seed] or r.get("ground_truth_used_for_prediction") is not False or r.get("ground_truth_used_for_scoring") is not True: raise ValueError("seed score provenance mismatch")
 means={str(seed):r["means"] for seed,r in reports.items()}
 per_seed={str(seed):{m:{"candidate_minus_source_T":float(r["means"]["foreground_induced"][m]-r["means"]["source_T"][m]),
  "candidate_minus_slot":float(r["means"]["foreground_induced"][m]-r["means"]["Slot70k"][m]),
  "source_T_minus_slot":float(r["means"]["source_T"][m]-r["means"]["Slot70k"][m])} for m in METRICS} for seed,r in reports.items()}
 aggregate={}; paired={}; prospective={}
 for metric in METRICS:
  candidate=np.stack([_per_image(reports[s],"foreground_induced",metric) for s in SEEDS])
  source=np.stack([_per_image(reports[s],"source_T",metric) for s in SEEDS])
  slot=np.stack([_per_image(reports[s],"Slot70k",metric) for s in SEEDS])
  aggregate[metric]={"candidate_mean_across_three_seeds":float(candidate.mean()),"source_T_mean_across_three_seeds":float(source.mean()),"slot_mean_across_three_seeds":float(slot.mean()),
   "candidate_minus_source_T":paired_seed_bootstrap(candidate-source),"candidate_minus_slot":paired_seed_bootstrap(candidate-slot),"source_T_minus_slot":paired_seed_bootstrap(source-slot)}
  # Prospective-only contrast uses both replication seeds with common fixed320 indices.
  d02=np.concatenate([(candidate[0]-source[0])[None,:],(candidate[2]-source[2])[None,:]],axis=0)
  prospective[metric]={"seed0_mean_delta":float(d02[0].mean()),"seed2_mean_delta":float(d02[1].mean()),
    "mean_delta_over_seed0_seed2":float(d02.mean()),"paired_bootstrap":paired_seed_bootstrap(d02),"bootstrap_seed":BOOTSTRAP_SEED,"replicates":REPLICATES}
 all3_fg=aggregate["fg_ari"]
 exact_all=all(reports[s].get("source_T_foreground_iou_exact") is True for s in SEEDS)
 gate={"candidate_fg_ari_three_seed_mean_above_source_T":all3_fg["candidate_mean_across_three_seeds"]>all3_fg["source_T_mean_across_three_seeds"],
  "three_seed_paired_fg_ari_ci_lower_above_zero":all3_fg["candidate_minus_source_T"]["ci95"][0]>0,
  "at_least_two_of_three_seed_fg_ari_gains_positive":sum(per_seed[str(s)]["fg_ari"]["candidate_minus_source_T"]>0 for s in SEEDS)>=2,
  "foreground_iou_exact_to_each_source_T":exact_all,
  "three_seed_mean_object_iou_not_below_source_T":aggregate["matched_object_iou"]["candidate_mean_across_three_seeds"]>=aggregate["matched_object_iou"]["source_T_mean_across_three_seeds"]}
 gate["replication_gate_passed"]=all(gate.values())
 slot_goal={m:{"candidate_mean_above_slot_mean":aggregate[m]["candidate_mean_across_three_seeds"]>aggregate[m]["slot_mean_across_three_seeds"]} for m in METRICS}
 slot_goal["fg_ari_at_least_0_87"] = aggregate["fg_ari"]["candidate_mean_across_three_seeds"]>=.87
 record={"experiment":"SW0165_foreground_partition_replication","status":"complete","seed_roles":{"1":"discovery, immutable SW0164 result","0":"prospective replication","2":"prospective replication"},
  "seed_score_sha256":{str(s):reports[s]["score_sha256"] for s in SEEDS},"per_seed_means":means,"per_seed_deltas":per_seed,
  "prospective_seed0_2":prospective,"three_seed_aggregate":aggregate,"replication_gate":gate,"slot_goal_and_0_87_target":slot_goal,
  "bootstrap":{"method":"paired fixed320 image bootstrap; identical image indices across seeds","random_state_seed":BOOTSTRAP_SEED,"replicates":REPLICATES},
  "interpretation":"Two prospective seeds are reported separately from discovery seed1. Fixed three-seed aggregate is not a seed-population confidence interval.",
  "ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True}
 output=Path(output)
 if output.exists(): raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_name(output.name+".tmp-"+uuid.uuid4().hex)
 with tmp.open("x",encoding="utf-8",newline="\n") as f: json.dump(record,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,output)
 return record

def validate_and_summarize(*,output=OUTPUT):
 from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as seed1eval
 from collaborative_test.SW_0165_foreground_partition_replication import predict_evaluate as eval165
 if sha256_file(SEED1_SCORE)!=SEED1_SCORE_SHA256: raise ValueError("immutable SW0164 discovery score changed")
 seed1=seed1eval.validate_score(SEED1_SCORE)
 r1=json.loads(SEED1_SCORE.read_text(encoding="utf-8")); r1["score_sha256"]=seed1["score_sha256"]
 reports={1:r1}
 for seed in (0,2):
  path=eval165.score_path(seed); checked=eval165.validate_score(path,seed=seed); report=json.loads(path.read_text(encoding="utf-8")); report["score_sha256"]=checked["score_sha256"]; reports[seed]=report
 return summarize(reports,output=output)

if __name__=="__main__":
 r=validate_and_summarize(); print(json.dumps({"status":r["status"],"gate":r["replication_gate"],"output":str(OUTPUT)},allow_nan=False))
