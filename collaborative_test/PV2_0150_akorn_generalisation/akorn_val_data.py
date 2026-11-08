"""Build AKOrN-format npz for our validation split, so its background rule is chosen there."""
import numpy as np, sys, pathlib
B = pathlib.Path("/work/USERS/tkim1/sw0255b")
sys.path.insert(0, str(B/"benchmark_codes/vendor")); sys.path.insert(0, str(B/"benchmark_codes"))
import portable_runtime as pr
cfg = pr.configure(str(B/"benchmark_codes/configs/frontier.json"))
out_root = pr.connect_inputs(cfg, str(B/"run"))
from full_benchmark.data import Dataset
OUT = pathlib.Path("/work/USERS/tkim1/baselines/_clone/akorn/data/ours")
for ds, name in (("multi_dsprites", "dsprites"), ("tetrominoes", "tetrominoes")):
    d = Dataset(ds, out_root)
    for split, ids in (("val", np.asarray(d.validation)), ("eval320", np.asarray(d.test)[:320])):
        rgb = np.asarray(d.rgb(ids, "validation" if split == "val" else "test"))
        lab = d.labels(ids, "validation" if split == "val" else "test")
        images = (rgb.astype(np.float32)/255.).transpose(0, 3, 1, 2)
        f = OUT/f"{name}_{split}.npz"
        np.savez_compressed(f, images=images, labels=lab.astype(np.int32), ids=ids.astype(np.int64))
        print(f"{name}_{split}: {images.shape} {lab.shape} ids[0,-1]={ids[0]},{ids[-1]} -> {f}", flush=True)
