# PV2_0147 — the head's effect measures how broken the core is

**Status** completed. Assembled from PV2_0138's result files; no new training.
**All rows** grid 64, 10,000 × 10, `min_group` 8, same 320 evaluation images, seed 0.

## The head on a target-trained core

### Tetrominoes — core alone 0.5448 / 0.6217 / 0.4434

| head | arm / epoch | patches changed | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|---|
| **B′** | `local_rgb` / **0** | **16.9%** | 0.5578 **+0.0130** (+2.4%) | 0.6995 **+0.0778** (+12.5%) | 0.6184 **+0.1750** (**+39.5%**) |
| label-free recipe | `local_rgb_trust` / 2 | 8.9% | 0.5293 **−0.0155** (−2.8%) | 0.7668 **+0.1451** (+23.3%) | 0.5878 **+0.1443** (+32.5%) |

### Multi-dSprites — core alone 0.1201 / 0.0597 / 0.0050

| head | arm / epoch | patches changed | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|---|
| **B′** | `local_rgb` / **0** | **23.8%** | 0.3269 **+0.2068** (**+172%**) | 0.2844 **+0.2247** (**+376%**) | 0.5219 **+0.5169** (**+10,291%, i.e. 104×**) |
| label-free recipe | `local_rgb_trust` / 2 | 1.9% | 0.1631 **+0.0429** (+35.8%) | 0.2002 **+0.1404** (+235%) | 0.1255 **+0.1205** (+2,399%, 25×) |

## The same head structure on a core that groups well

| core | dataset | patches changed | fg_ari gain |
|---|---|---|---|
| CLEVR-trained | CLEVR | 0.5% | +0.0047 (+0.56%) |
| CLEVR-trained | Multi-dSprites | 0.3% | +0.0064 (+0.9%) |
| CLEVR-trained | Tetrominoes | 0.6% | **withdrawn** — head-seed mean is negative (`PV2_0146`) |
| **target-trained** | **Tetrominoes** | **16.9%** | **+0.0130 (+2.4%)** |
| **target-trained** | **Multi-dSprites** | **23.8%** | **+0.2068 (+172%)** |

## Three things this says

**1. The head's effect size is a measurement of the core, not of the head.** The identical
structure touches 0.3–0.6% of patches and gains under 1% on a core that groups, and touches
17–24% and gains 172% on one that does not. **The head's value is inversely proportional to the
core's competence.**

**2. The repair comes from the architecture, not from training.** Both B′ selections are
**epoch 0** — the head before any gradient step. So a background candidate plus
locality-radius-constrained reassignment (change A) is what recovers matched-object IoU 104×.
Learned weights contributed nothing on these cores.

**3. One trade runs against the stated preference.** The standing instruction is that a higher
FG-ARI may cost a little Foreground IoU. Tetrominoes' label-free recipe offers the **opposite**:
**−2.8% FG-ARI for +23.3% Foreground IoU** (and +32.5% matched-object IoU). Rule B′ declined it
and took the arm that raises all three. That is the rule behaving as written — but it is worth
flagging that if object IoU mattered more than FG-ARI here, the rule would be the wrong one.
The rule is unchanged pending a decision.

## And the head still cannot close the gap

| dataset | target-trained core + its best head | CLEVR core | CLEVR advantage |
|---|---|---|---|
| Tetrominoes | 0.5578 | **0.8580** (no head at all) | **+0.3002** |
| Multi-dSprites | 0.3269 | **0.7041** (with its head) | **+0.3772** |

## Reproduce

```bash
python3 headtable.py   # reads fit261/*.json on frontier
```
