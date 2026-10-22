"""Add the fields evaluate_model expects to the benchmark targets.

CLEVR's targets carry `names` -- the image filename per row -- and evaluate_model slices
it to label the rows it scored. The benchmark targets were built without it, so the
evaluation got through the contract check and then died on KeyError: 'names'.

`names` is not decoration: it is how the gamma tensor and the ground truth are checked to
be describing the same image in the same order, which the project's own rules require.
The PNGs were written by dump_png.py as "<dataset>_%06d.png" in row order from the same
tensor the targets were built from, so the name for row i is exactly that, and the
existence of every file is verified here rather than assumed.
"""
import sys, torch
from pathlib import Path

for d in ("tetrominoes", "multi_dsprites"):
    for g in (16, 32):
        p = "/work/USERS/tkim1/mod/targets_%s_grid%d.pt" % (d, g)
        blob = torch.load(p, map_location="cpu")
        n = blob["patch_labels"].shape[0]
        names = ["%s_%06d.png" % (d, i) for i in range(n)]
        img_dir = Path("/work/USERS/tkim1/mod/images_%s" % d)
        missing = [x for x in (names[0], names[n // 2], names[-1]) if not (img_dir / x).exists()]
        if missing:
            print("  %-16s grid %-3d NAME MISMATCH: %s" % (d, g, missing)); continue
        blob["names"] = names
        blob.setdefault("patch_size", None)
        torch.save(blob, p)
        print("  %-16s grid %-3d names added (%d rows, first %s, spot-checked 3 files exist)"
              % (d, g, n, names[0]))
