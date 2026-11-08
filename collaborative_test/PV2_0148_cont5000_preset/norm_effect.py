"""How much does using CLEVR's feature normalization on another dataset actually cost?

model.py:73 standardises the encoder's features with the stored (mean, std) and then
clamps to +-3. If the target dataset's features sit at a different scale, CLEVR's statistics
push values outside that range, where the clamp flattens them and the gradient dies. This
measures the fraction clamped, and what each dataset's own statistics would be.
"""
import sys, numpy as np, torch
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
cm = float(model.feature_mean.mean()); cs = float(model.feature_std.mean())
print(f"CLEVR core's stored normalization: mean {cm:+.6f}  std {cs:.6f}\n")

print(f"{'dataset':16s} {'own mean':>10s} {'own std':>9s} {'std ratio':>10s} "
      f"{'% clamped w/ CLEVR norm':>24s} {'% clamped w/ own':>17s}")
for ds, split in (('clevr', 'train'), ('tetrominoes', 'train'), ('multi_dsprites', 'train')):
    d = Dataset(ds, out)
    ids = np.asarray(d.train)[:64]
    rgb = np.asarray(d.rgb(ids, split))
    x = torch.from_numpy(rgb.copy()).cuda()
    with torch.no_grad():
        f = model.encoder(x.permute(0, 3, 1, 2).float()/255.)
    own_m, own_s = float(f.mean()), float(f.std())
    z_clevr = (f - model.feature_mean)/model.feature_std
    z_own = (f - own_m)/own_s
    pc = float((z_clevr.abs() > 3).float().mean())*100
    po = float((z_own.abs() > 3).float().mean())*100
    print(f'{ds:16s} {own_m:+10.6f} {own_s:9.6f} {own_s/cs:10.2f}x '
          f'{pc:23.2f}% {po:16.2f}%')
