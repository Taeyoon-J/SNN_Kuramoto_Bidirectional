import json, pathlib
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
r252 = json.load(open(B/'recovered252/run252/jobs/model252/clevr/seed0/training_report.json'))
p253 = B/'run/jobs/model253/clevr/seed0/training_report.json'
if not p253.exists(): p253 = pathlib.Path('/work/USERS/tkim1/share/native64_core_253/training_report.json')
r253 = json.load(open(p253))
print("KEYS 252:", sorted(k for k in r252 if not isinstance(r252[k], dict)))
print("KEYS 253:", sorted(k for k in r253 if not isinstance(r253[k], dict)))
print()
for k in ('updates','images_seen','unique_TRAIN','epochs_completed','train_preset','total_exposures'):
    print(f"  {k:20s} 252={r252.get(k)}   253={r253.get(k)}")
print()
print("dict keys 252:", [k for k in r252 if isinstance(r252[k], dict)])
print("dict keys 253:", [k for k in r253 if isinstance(r253[k], dict)])

# core vs head parameter movement
def split(rep):
    pc = rep.get('parameter_changes', {})
    core = {k: v for k, v in pc.items() if k.startswith('wrapped.')}
    enc  = {k: v for k, v in pc.items() if k.startswith('encoder.')}
    head = {k: v for k, v in pc.items() if k.startswith('head.')}
    return core, enc, head

for name, rep in (('252', r252), ('253', r253)):
    core, enc, head = split(rep)
    f = lambda d: (sum(d.values())/len(d) if d else float('nan'))
    print(f"\n{name}: mean relative parameter change")
    print(f"   core  n={len(core):2d}  mean={f(core):.5f}   max={max(core.values()) if core else 0:.5f}")
    print(f"   encoder n={len(enc):2d}  mean={f(enc):.5f}   max={max(enc.values()) if enc else 0:.5f}")
    print(f"   head  n={len(head):2d}  mean={f(head):.5f}   max={max(head.values()) if head else 0:.5f}")
    gc = rep.get('active_tensor_gradient_coverage', {})
    gcore = {k: v for k, v in gc.items() if k.startswith('wrapped.')}
    genc  = {k: v for k, v in gc.items() if k.startswith('encoder.')}
    ghead = {k: v for k, v in gc.items() if k.startswith('head.')}
    print(f"   gradient mass: core={sum(gcore.values()):.3e}  encoder={sum(genc.values()):.3e}  head={sum(ghead.values()):.3e}")
