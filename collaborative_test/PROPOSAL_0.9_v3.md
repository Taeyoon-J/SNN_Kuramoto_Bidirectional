# Proposal to 0.9, v3 — the core cannot de-phase any pair, and that is fixable within it

Supersedes `PROPOSAL_0.9_v2.md`. This is the **how**, which `ROUTE_MAP_0.9_v4.md` stopped
short of: v4 established that route C cannot reach 0.9 by arithmetic and that the deficit
is CLEVR-specific, then listed measurements. This derives the mechanism from the equations
and names one change, with the measurement that kills it before any GPU time is spent.

Nothing here has been run. It is an analysis of code in this repository plus arithmetic.

## 1. The equations, as configured

`kuramoto_layer.py` integrates, with `A_lat = relu(0.5(A + Aᵀ)) + 1e-6` so coupling is
**non-negative everywhere**:

```
interaction_i  =  Σ_j  A_lat_ij · sin(θ_j − θ_i − α_ij)
α_ij           =  alpha_scale · tanh(direction − directionᵀ)_ij · norm_cost_ij
norm_cost      =  (1/A_lat normalised to [0,1])
```

`α` is antisymmetric: `direction − directionᵀ` is, and `norm_cost` built from `1/A_lat` is
symmetric. For one pair, writing `Δ = θ_j − θ_i`:

| | |
| --- | --- |
| `dθ_i/dt` includes | `A sin(Δ − α)` |
| `dθ_j/dt` includes | `A sin(−Δ + α)` |
| so | **`dΔ/dt = −2 A sin(Δ − α)`** |

Fixed points at `Δ = α`, stable because `d/dΔ = −2A cos(0) < 0`, and at `Δ = α + π`,
unstable.

> **Every coupled pair locks at phase difference exactly `α`.** Anti-phase is never a
> stable state.

## 2. Two bounds that close it

**`alpha_scale = 1.0`, hardcoded** at `s2net_cls.py:177`. There is no CLI flag for it; it
appears nowhere in `train_s2net_core.py`, so **no experiment in this project has ever varied
it**. With `|tanh| < 1` and `norm_cost ∈ [0,1]`:

```
|α| < 1.0 rad = 57.3°        repulsion needs |α| > π/2 = 90°
                             anti-phase needs |α| near π = 180°
```

So the largest phase separation any two oscillators can hold is **57.3°**, and the regime
where the interaction becomes repulsive is **unreachable by construction**.

**`α` is smallest exactly where coupling is strongest.** `norm_cost` is the normalised
`1/A_lat`, so `A_lat` maximal drives `norm_cost → 0` and `α → 0`. Two patches the graph
considers most alike get maximum attraction with **zero** lag.

Two touching, same-material CLEVR objects are precisely that case: high feature cosine
similarity, low geodesic distance because they touch, therefore maximal `A` and minimal `α`
— maximum merging force and no de-phasing whatsoever.

`coupling_balance` does not rescue this. It makes some `adjacency` entries negative, but
`prepare_coupling` then applies `relu`, which clips them to `1e-6`. Its documented mechanism
— "what was attracted less becomes repelled" — does not reach the dynamics; those pairs
become *uncoupled*, and uncoupled oscillators drift under their own `ω`, giving random
relative phase rather than opposition.

## 3. This explains the measurements, retrospectively

| observation | explanation |
| --- | --- |
| nine routes closed (`PV2_0076`–`PV2_0106`) | every one changed *when* the model merges or how the readout cuts the result; none could make the dynamics split a pair, because splitting is not in the reachable set |
| inventing foreground costs 4.7× missing it (`PV2_0114`) | the dynamics have an attractive bias with no opposing force |
| merging routes lose obj-IoU (`PV2_0107b`, peer `SW_0234` with a GT oracle) | merging is the direction the system already over-does |
| Tetrominoes holds **98.63%** of ceiling | three objects *separated* on black: the object/background wall (`PV2_0075`, 5.27×) does all the work and no object-to-object split is ever required |
| CLEVR holds **85.48%** | ten shaded, touching, same-material objects: object-to-object splitting is required and is structurally unavailable |

The last two rows are the same core, same losses, same readout. **The CLEVR deficit is not
a capacity limit. It is a reachability limit.**

## 4. The change

Core-preserving: the Kuramoto update, the image-conditioned graph, the OT-lag structure and
the spike pulse all stay. Two edits, one scalar and one driver.

**(a) Expose `alpha_scale` and let it exceed π/2.** One flag; `1.0` reproduces every
existing checkpoint exactly. Necessary but not sufficient — on its own it widens the locked
offset uniformly, which separates nothing, because `α`'s *pattern* is still wrong.

**(b) Drive the lag by the geodesic-to-Euclidean ratio instead of by `1/A_lat`.**

```
            current:  norm_cost_ij  =  normalise( 1 / A_lat_ij )
          proposed:  r_ij          =  geodesic_ij / (euclidean_ij + eps),  then normalised
```

Within one object a path along the surface is about as long as the straight line, so
`r ≈ 1`. Between two objects that touch, the geodesic must detour around whatever
dissimilarity sits at the seam, so `r > 1` — and it is `> 1` **even when the two objects
look alike**, provided the contact seam carries any shading discontinuity at all. Surface
normals differ across a seam even for identical materials, which is the signal feature
cosine similarity throws away and the geodesic retains.

This inverts the broken coupling: strongly-coupled same-object pairs keep `α ≈ 0` and still
lock in phase, while strongly-coupled *different*-object pairs get large `α` and settle far
apart. The machinery is already built — `_geodesic_distance` is computed and fed to the
logits, and `grid_distance` is the Euclidean term — so this reuses what exists rather than
adding a stage.

## 5. The falsifier, before any GPU time

The proposal rests on one unmeasured claim: that `r` separates same-object from
different-object pairs. `PV2_0075`'s table has three rows and the one that matters is
missing:

| pair type | mean `1 − cosine` |
| --- | ---: |
| within one object | 0.0331 |
| background to background | 0.0054 |
| object to background | 0.1743 |
| **object to a different object** | **never measured** |

**Measurement, CPU only, on saved features and the patch targets:** for CLEVR validation
images, bucket adjacent patch pairs into same-object / different-object / object-background
using the ground-truth patch labels, and report for each bucket the mean `1 − cosine`, the
mean `r = geodesic/euclidean`, and the distributions' overlap.

- **If `r` separates same-object from different-object pairs** — say a ratio comparable to
  the 5.27× the object/background wall achieves — the proposal is worth the training run.
- **If it does not**, the proposal is dead and the geodesic cannot carry object identity.
  Then the only remaining in-core signal is the spike pulse, which already returns through
  `A_lat` and whose `cos(θ)` phase response curve can de-phase, and the question becomes
  whether `spike_pulse_gain` can be made selective rather than global.

Either outcome is informative, and it costs no GPU. **This is the next thing to run.**

## 6. Does it reach 0.9? What would have to be true

From `ROUTE_MAP_0.9_v4.md`: CLEVR needs **≥ 90%** of ceiling and holds **85.48%**, so 4.5
points of fraction, and a grid whose ceiling at 64 is unmeasured. The proposal is aimed at
exactly the deficit that distinguishes CLEVR (85.48%) from Tetrominoes (98.63%), and the
13-point spread between them is what sets the available headroom — the same core already
demonstrates the higher figure on data that needs no object-to-object split.

That is an argument that the headroom exists and that this mechanism is where it sits. It
is **not** a prediction that the change delivers it. Honest statement of the position:

- the diagnosis is derived from the equations and is checkable by reading them
- the retrospective fit to nine closed routes is consistent but is not a test
- the falsifier in §5 decides whether the proposed driver carries the signal
- whether closing 4.5 points of fraction follows from the mechanism is **unknown**, and a
  3-seed CLEVR run under the pixel protocol is the only thing that answers it

## 7. Ablation design, if §5 passes

One variable per arm, grid 32, seed 0, validation, against `PV2_0104`'s grid-32 line.

| arm | change | expected if the diagnosis is right |
| --- | --- | --- |
| control | unchanged | — |
| A | `alpha_scale` 1.0 → 2.5, lag driver unchanged | little or nothing; the pattern is still wrong |
| B | lag driver → `r`, `alpha_scale` 1.0 | partial; the pattern is right but capped at 57.3° |
| C | both | the only arm that can place two objects in opposition |

**Gate:** C must raise matched-object IoU with predicted foreground at or below 0.20, the
calibration gate every route has had to clear. **Falsifier:** if C does not beat the control
on matched-object IoU, the reachability diagnosis is wrong even though the algebra holds,
and the attractive-only property is not what bounds CLEVR.

Diagnostics to read alongside, since the claim is about phase structure: the within-object
and between-object PLV distributions separately, the realised `|Δ|` histogram between
adjacent different-object patches, and the fraction of pairs settling beyond 90°.
