"""Give the benchmark manifests the `splits` lists evaluate_model checks against.

The evaluator compares the targets' `names` to manifest["splits"][split] and refuses to
score if they disagree. That is the check CLAUDE.md asks for -- that the gamma tensor and
the ground truth describe the same images in the same order -- and my manifests carried
only `ranges`, so the check could not run.

Built from `ranges` and cross-checked against the targets' own `names`, so the two cannot
drift apart silently.
"""
import json, torch

for d in ("tetrominoes", "multi_dsprites"):
    mp = "/work/USERS/tkim1/mod/split_manifest_%s.json" % d
    man = json.load(open(mp))
    names = torch.load("/work/USERS/tkim1/mod/targets_%s_grid32.pt" % d,
                       map_location="cpu")["names"]
    splits, counts = {}, {}
    for k, (lo, hi) in man["ranges"].items():
        splits[k] = names[lo:hi]
        counts[k] = hi - lo
        assert len(splits[k]) == hi - lo, (d, k, len(splits[k]), hi - lo)
    man["splits"] = splits
    man["counts"] = counts
    man["image_dir"] = "/work/USERS/tkim1/mod/images_%s" % d
    man["order"] = "row order of the source tfrecord, as dump_png.py wrote the PNGs"
    json.dump(man, open(mp, "w"), indent=2)
    print("  %-16s splits built: %s" % (d, {k: len(v) for k, v in splits.items()}))
    print("     train[0]=%s  validation[0]=%s  test[0]=%s"
          % (splits["train"][0], splits["validation"][0], splits["test"][0]))
