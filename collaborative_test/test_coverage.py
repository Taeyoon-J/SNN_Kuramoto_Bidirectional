"""Local unit test for the coverage stage. No GPU, no checkpoint, no server.

A 16x16 grid holds three 3x3 blocks. The primary readout is handed only two of
them, so the third is a missed object. The stage must recover exactly that third
block as ONE new group and leave the first two byte-identical -- which is the
constraint REVIEW_0007 took from SW_0207 against SW_0234.
"""
import sys, torch
sys.path.insert(0, ".")
from evaluate_model import _coverage_groups

G, N, T = 16, 256, 8
def block(r0, c0, k=3):
    return [ (r0+dr)*G + (c0+dc) for dr in range(k) for dc in range(k) ]

A_, B_, C_ = block(2, 2), block(2, 11), block(11, 6)
act = torch.full((1, N, T), 0.02)
for blk in (A_, B_, C_):
    act[0, blk, :] = 0.90

groups = [[tuple(sorted(A_)), tuple(sorted(B_))]]        # C_ is missed
out = _coverage_groups(groups, act, None, G, quantile=0.85,
                       synchrony=None, min_size=2, settle=0)[0]

fail = 0
def check(name, cond, extra=""):
    global fail
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else "   " + extra))
    if not cond: fail += 1

check("existing groups untouched", out[0] == tuple(sorted(A_)) and out[1] == tuple(sorted(B_)),
      "got %s" % (out[:2],))
check("exactly one new group", len(out) == 3, "got %d groups" % len(out))
if len(out) == 3:
    check("recovered group IS the missed object", set(out[2]) == set(C_),
          "size %d, expected %d" % (len(out[2]), len(C_)))
allp = [p for g in out for p in g]
check("no patch in two groups", len(allp) == len(set(allp)))

# inert when the quantile admits nothing new
same = _coverage_groups(groups, act, None, G, quantile=0.999, synchrony=None,
                        min_size=2, settle=0)[0]
check("inert at an extreme quantile", [tuple(x) for x in same] == [tuple(x) for x in groups[0]])
check("inert when quantile is None",
      _coverage_groups(groups, act, None, G, None, None, 2, 0) is groups)

# two separated missed objects must come back as TWO groups, not one
D_ = block(11, 6); E_ = block(11, 12)
act2 = torch.full((1, N, T), 0.02)
for blk in (A_, D_, E_): act2[0, blk, :] = 0.90
out2 = _coverage_groups([[tuple(sorted(A_))]], act2, None, G, 0.85, None, 2, 0)[0]
check("two separated misses stay two groups", len(out2) == 3, "got %d" % len(out2))

# an unclaimed patch adjacent to an existing group must NOT join it
adj = [3*G + 5]                                   # sits beside block A_
act3 = torch.full((1, N, T), 0.02); act3[0, A_, :] = 0.90; act3[0, adj, :] = 0.90
out3 = _coverage_groups([[tuple(sorted(A_))]], act3, None, G, 0.85, None, 1, 0)[0]
check("adjacent patch does not widen an existing group",
      out3[0] == tuple(sorted(A_)), "group A changed to %s" % (out3[0],))

print("\n%s" % ("ALL CHECKS PASSED" if fail == 0 else "%d CHECK(S) FAILED" % fail))
sys.exit(1 if fail else 0)
