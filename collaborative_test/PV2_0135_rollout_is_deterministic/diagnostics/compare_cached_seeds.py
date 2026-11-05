import torch
for i in [0, 1, 2]:
    d = torch.load(f"/work/USERS/tkim1/sw0255b/ev/clevr_train/{i:04d}.pt",
                   map_location="cpu", weights_only=True)
    fa, fb = d["features"].float(), d["features_b"].float()
    ga, gb = d["gamma"].float(), d["gamma_b"].float()
    ra, rb = d["regions_253"], d["regions_b"]
    iid = int(d["image_id"])
    print(f"img {iid}: features identical={torch.equal(fa, fb)}  "
          f"mean|df|={float((fa-fb).abs().mean()):.6f}  "
          f"relative={float((fa-fb).abs().mean()/fa.abs().mean()):.4f}")
    print(f"   gamma identical={torch.equal(ga, gb)}  "
          f"mean|dg|={float((ga-gb).abs().mean()):.6f}")
    print(f"   regions identical={bool((ra == rb).all())}  "
          f"differ on {float((ra != rb).float().mean()):.4f} of patches  "
          f"n_regions {len(ra.unique())} vs {len(rb.unique())}")
