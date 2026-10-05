#!/usr/bin/env python3
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("--seed0",type=Path,required=True); p.add_argument("--seed2",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
s0=json.loads(a.seed0.read_text()); s2=json.loads(a.seed2.read_text())
assert s0["seed"]==0 and s2["seed"]==2 and s0["ids"]==s2["ids"]==[1320,1351]
stages=["gamma","graph","kuramoto_plv_early","kuramoto_plv_middle","kuramoto_plv_late","gate_early","gate_middle","gate_late","dendritic_early","dendritic_middle","dendritic_late","membrane_early","membrane_middle","membrane_late","spike_early","spike_middle","spike_late"]
rows={}
for stage in stages:
 a0=s0["pair_affinity"][stage]["object_margin"]; a2=s2["pair_affinity"][stage]["object_margin"]
 rows[stage]={"seed0":a0,"seed2":a2,"seed0_minus_seed2":a0-a2}
out={"experiment":"SW0064 best-vs-worst seed activation flow","ids":[1320,1351],"metric":"same-foreground-object affinity minus different-foreground-object affinity","stages":rows,"ground_truth_role":"diagnostic only"}
if a.output.exists(): raise FileExistsError(a.output)
a.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
