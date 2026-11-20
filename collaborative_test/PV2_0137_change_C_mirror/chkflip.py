import torch
GRID = 64
def unflip_nodes(t):
    return t.reshape(*t.shape[:-1], GRID, GRID).flip(-1).reshape(*t.shape[:-1], GRID*GRID) \
        if t.dim() > 1 else t.reshape(GRID, GRID).flip(-1).reshape(-1)
for i in (0, 1, 2):
    d = torch.load(f"/work/USERS/tkim1/sw0255b/ev/clevr_train/{i:04d}.pt",
                   map_location="cpu", weights_only=True)
    if 'features_f' not in d:
        print(f"img {int(d['image_id'])}: not mirrored yet"); continue
    fa, ff = d["features"].float(), d["features_f"].float()
    ga, gf = d["gamma"].float(), d["gamma_f"].float()
    ra, rf = d["regions_253"], d["regions_253_f"]
    # compare in the ORIGINAL frame: un-mirror the mirrored run
    ff_u = ff.T.reshape(-1, GRID, GRID).flip(-1).reshape(ff.shape[1], -1).T
    gf_u = gf.T.reshape(-1, GRID, GRID).flip(-1).reshape(gf.shape[1], -1).T
    rf_u = rf.reshape(GRID, GRID).flip(-1)
    rel = float((fa-ff_u).abs().mean()/fa.abs().mean())
    print(f"img {int(d['image_id'])}:")
    print(f"   features  identical after un-mirroring = {torch.equal(fa, ff_u)}   "
          f"mean|diff| = {float((fa-ff_u).abs().mean()):.5f}   relative = {rel:.4f}")
    print(f"   gamma     identical after un-mirroring = {torch.equal(ga, gf_u)}   "
          f"mean|diff| = {float((ga-gf_u).abs().mean()):.5f}")
    print(f"   regions   identical after un-mirroring = {bool((ra == rf_u).all())}   "
          f"differ on {float((ra != rf_u).float().mean()):.4f} of patches   "
          f"n = {len(ra.unique())} vs {len(rf_u.unique())}")
    print(f"   foreground agrees on {float(((ra>0) == (rf_u>0)).float().mean()):.4f} of patches")
