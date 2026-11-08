"""Verify the chained targets without spending 9 h per stage: fabricate a stage-1 checkpoint
at 9,380/150,000 and confirm stage 2 computes 12,510/200,000 rather than aborting."""
import torch, pathlib, json
B = pathlib.Path('/work/USERS/tkim1/sw0255b')
src = pathlib.Path('/work/USERS/tkim1/share/native64_core_253')
for stage, (upd, seen, want_u, want_s) in enumerate(
        [(6250, 100000, 9380, 150000), (9380, 150000, 12510, 200000)], start=1):
    d = torch.load(src/'resume.pt', map_location='cpu', weights_only=True)
    d['updates'], d['images_seen'] = upd, seen
    out = B/f'chaincheck_s{stage}/jobs/model253/multi_dsprites/seed0'
    out.mkdir(parents=True, exist_ok=True)
    torch.save(d, out/'resume.pt')
    import shutil; shutil.copy(src/'initial.pt', out/'initial.pt')
    print(f'stage {stage}: seeded a checkpoint at {upd} updates / {seen} images '
          f'-> should target {want_u} / {want_s}')
