"""Open the rollout length T for a sweep, strictly backward-compatibly.

T is hardcoded in three places that must move together:

  full_benchmark/model.py:81   late_rollout(..., total_steps=1024)
  full_benchmark/model.py:82   native_component_rho_q(..., settle=512)
  readout_benchmark/head.py:12 activity_features(trace, settle=512, bins=32)

`settle` is the point the readout starts reading from, and 512 of 1024 is the second half of the
rollout. A fixed 512 is impossible for T < 512, so the sweep keeps the *rule* rather than the
number: **settle = T // 2**, the second half at every T. `bins=32` needs `settle % 32 == 0`,
which holds down to T = 64.

Every default is left at today's value, so any existing command produces bit-identical results.
Backups are written once.
"""
import pathlib, shutil

C = pathlib.Path('/work/USERS/tkim1/sw0255b/benchmark_codes')

# ---------------------------------------------------------------- model.py
p = C/'vendor/full_benchmark/model.py'
bak = p.with_suffix('.py.bak_steps')
if not bak.exists():
    shutil.copy(p, bak)
s = p.read_text()

old = """    def forward(self,rgb,*,inference_seed,live_tail_steps=64):
        torch.manual_seed(int(inference_seed))
        if next(self.parameters()).is_cuda:torch.cuda.manual_seed(int(inference_seed))
        gamma=self.encode(rgb)
        trace,adjacency=capture_source_coupling(self.wrapped.core.kuramoto,
            lambda:late_rollout(self.wrapped,gamma,total_steps=1024,live_tail_steps=live_tail_steps))
        rho,q=native_component_rho_q(trace['spikes'],trace['component_spikes'],settle=512)"""
new = """    def forward(self,rgb,*,inference_seed,live_tail_steps=64,total_steps=1024):
        # total_steps opens the rollout length T for the PV2_0154 sweep. The default is 1024,
        # today's value, so every existing call is bit-identical. `settle` is the step the
        # readout starts reading from and was hardcoded at 512 of 1024, i.e. the second half;
        # that rule is kept as total_steps//2 rather than the number, because a fixed 512 is
        # impossible for T<512. activity_features needs settle%32==0, true down to T=64.
        torch.manual_seed(int(inference_seed))
        if next(self.parameters()).is_cuda:torch.cuda.manual_seed(int(inference_seed))
        gamma=self.encode(rgb)
        steps=int(total_steps);settle=steps//2
        if steps<64 or steps%2 or settle%32:raise ValueError('total_steps must be even, >=64, with total_steps//2 divisible by 32')
        trace,adjacency=capture_source_coupling(self.wrapped.core.kuramoto,
            lambda:late_rollout(self.wrapped,gamma,total_steps=steps,live_tail_steps=live_tail_steps))
        rho,q=native_component_rho_q(trace['spikes'],trace['component_spikes'],settle=settle)"""
assert s.count(old) == 1, 'model.py forward not found verbatim'
s = s.replace(old, new, 1)

old = "        return dict(gamma=gamma,trace=trace,adjacency=adjacency,rho=rho,q=q)"
new = "        return dict(gamma=gamma,trace=trace,adjacency=adjacency,rho=rho,q=q,total_steps=steps,settle=settle)"
assert s.count(old) == 1, 'model.py return not found verbatim'
s = s.replace(old, new, 1)
p.write_text(s)
print('model.py: total_steps opened, settle = T//2')

# ---------------------------------------------------------------- cache_any.py
p = C/'cache_any.py'
bak = p.with_suffix('.py.bak_steps')
if not bak.exists():
    shutil.copy(p, bak)
s = p.read_text()

old = "ap.add_argument('--out', required=True); ap.add_argument('--min-group', type=int, default=8)"
new = ("ap.add_argument('--out', required=True); ap.add_argument('--min-group', type=int, default=8)\n"
       "# PV2_0154: the rollout length. 1024 is today's value, so omitting it changes nothing.\n"
       "ap.add_argument('--steps', type=int, default=1024,\n"
       "                help='rollout length T; the readout settles at T//2')")
assert s.count(old) == 1, 'cache_any args not found verbatim'
s = s.replace(old, new, 1)

old = "        o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=seed, live_tail_steps=0)"
new = ("        o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=seed,\n"
       "                  live_tail_steps=0, total_steps=a.steps)")
assert s.count(old) == 1, 'cache_any model call not found verbatim'
s = s.replace(old, new, 1)

old = "    torch.save(dict(image_id=int(image_id), features=activity_features(o['trace'])[0].cpu(),"
new = "    torch.save(dict(image_id=int(image_id), features=activity_features(o['trace'], settle=o['settle'])[0].cpu(),"
assert s.count(old) == 1, 'cache_any save not found verbatim'
s = s.replace(old, new, 1)
p.write_text(s)
print('cache_any.py: --steps added, settle threaded through to activity_features')
