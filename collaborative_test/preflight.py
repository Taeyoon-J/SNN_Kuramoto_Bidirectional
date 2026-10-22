"""Check a dataset's targets and manifest against the reference BEFORE evaluating.

Written after four consecutive evaluation failures in one session, all the same class:
the benchmark targets and manifest did not carry fields the evaluator requires, and each
missing field was discovered by running the evaluator and reading the traceback --
contract_version, then names, then patch_size, then the manifest's splits. One comparison
against the CLEVR artefacts would have found all four at once.

Checks, in the order the evaluator hits them:
  1. every key CLEVR's targets carry is present
  2. contract_version is a string the caller can pass to --expect-contract
  3. names has one entry per row and the files exist
  4. manifest has splits, and splits agrees with ranges AND with the targets' names
  5. the gamma tensor has the same row count and the right node count for the grid

Exit code is non-zero if anything fails, so a runner can gate on it.
"""
import json, sys, argparse
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--grid", type=int, required=True)
ap.add_argument("--src", default="/work/USERS/tkim1/mod")
ap.add_argument("--reference",
                default="/work/USERS/tkim1/clevr/with_masks/targets_v1.pt")
args = ap.parse_args()

fails = []
def check(ok, label, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label, "" if ok else "   " + detail))
    if not ok:
        fails.append(label)

tp = "%s/targets_%s_grid%d.pt" % (args.src, args.dataset, args.grid)
mp = "%s/split_manifest_%s.json" % (args.src, args.dataset)
gp = "%s/gamma_%s_grid%d.pt" % (args.src, args.dataset, args.grid)
print("preflight: %s grid %d" % (args.dataset, args.grid))

blob = torch.load(tp, map_location="cpu")
ref = torch.load(args.reference, map_location="cpu")
missing = sorted(set(ref.keys()) - set(blob.keys()))
check(not missing, "targets carry every key the reference has", "missing %s" % missing)

cv = blob.get("contract_version")
check(isinstance(cv, str) and cv != "",
      "contract_version is a usable string", "got %r -- pass it as --expect-contract" % cv)

n = blob["patch_labels"].shape[0]
names = blob.get("names")
check(isinstance(names, list) and len(names) == n,
      "names has one entry per row", "rows %d, names %s" % (n, len(names) if names else None))
if isinstance(names, list) and names:
    img = Path("%s/images_%s" % (args.src, args.dataset))
    absent = [x for x in (names[0], names[n // 2], names[-1]) if not (img / x).exists()]
    check(not absent, "named image files exist (first, middle, last)", "absent %s" % absent)

man = json.load(open(mp))
check("splits" in man, "manifest has splits",
      "evaluate_model compares names to manifest['splits'][split]")
if "splits" in man and "ranges" in man:
    bad = []
    for k, (lo, hi) in man["ranges"].items():
        s = man["splits"].get(k, [])
        if len(s) != hi - lo:
            bad.append("%s len %d != %d" % (k, len(s), hi - lo))
        elif isinstance(names, list) and s != names[lo:hi]:
            bad.append("%s disagrees with targets' names" % k)
    check(not bad, "splits agrees with ranges and with the targets' names", "; ".join(bad))

g = torch.load(gp, map_location="cpu")
check(g.shape[0] == n, "gamma row count matches the targets",
      "gamma %d, targets %d" % (g.shape[0], n))
check(g.shape[-1] == args.grid * args.grid, "gamma node count matches the grid",
      "gamma %d, expected %d" % (g.shape[-1], args.grid * args.grid))

print("  -> %s" % ("ready" if not fails else "NOT ready: %d check(s) failed" % len(fails)))
sys.exit(1 if fails else 0)
