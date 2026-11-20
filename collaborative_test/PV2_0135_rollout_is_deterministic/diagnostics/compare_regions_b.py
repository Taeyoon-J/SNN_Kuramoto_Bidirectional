import torch
for i in [0, 1, 2]:
    d = torch.load(f"/work/USERS/tkim1/sw0255b/ev/clevr_train/{i:04d}.pt",
                   map_location="cpu", weights_only=True)
    rb, ra, r253 = d["regions_b"], d["regions_all"], d["regions_253"]
    print(f"img {int(d['image_id'])}: regions_b == regions_all -> {bool((rb == ra).all())}   "
          f"regions_b == regions_253 -> {bool((rb == r253).all())}   "
          f"n: b={len(rb.unique())} all={len(ra.unique())} 253={len(r253.unique())}")
