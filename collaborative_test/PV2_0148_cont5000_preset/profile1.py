"""Where do the extra 19 s per update go on Tetrominoes, and is it GPU or CPU?

This decides whether renting a faster GPU helps. The forward is encode -> 1,024-step rollout
-> native_component_rho_q over the spike components. The fragmented datasets produce far more
components, and if rho_q scales with that, the cost is CPU-side and a 4090 buys little.
"""
import sys, time, numpy as np, torch
B = '/work/USERS/tkim1/sw0255b'
sys.path.insert(0, B+'/benchmark_codes/vendor'); sys.path.insert(0, B+'/benchmark_codes')
import portable_runtime as pr
cfg = pr.configure(B+'/benchmark_codes/configs/frontier.json')
out = pr.connect_inputs(cfg, B+'/run')
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
J = '/work/USERS/tkim1/share/native64_core_253'
resume = torch.load(J+'/resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(J+'/initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)

for ds, split in (('clevr', 'train'), ('tetrominoes', 'train')):
    d = Dataset(ds, out)
    rgb = np.asarray(d.rgb(np.asarray(d.train)[:1], split))[0]
    x = torch.from_numpy(rgb[None].copy()).cuda()
    with torch.no_grad():
        model(x, inference_seed=0, live_tail_steps=0)      # warm up
        torch.cuda.synchronize()
        t = time.time(); o = model(x, inference_seed=0, live_tail_steps=0)
        torch.cuda.synchronize(); full = time.time()-t
        sp, cs = o['trace']['spikes'], o['trace']['component_spikes']
        ncomp = cs.shape[1] if cs.dim() > 1 else int(cs.max())+1
        # re-time just the component statistics on the captured trace
        from full_benchmark.model import native_component_rho_q
        torch.cuda.synchronize(); t = time.time()
        native_component_rho_q(sp, cs, settle=512)
        torch.cuda.synchronize(); rq = time.time()-t
    print(f'{ds:14s} full forward {full:6.2f} s   native_component_rho_q {rq:6.2f} s '
          f'({rq/full*100:4.1f}%)   component_spikes shape {tuple(cs.shape)}')
