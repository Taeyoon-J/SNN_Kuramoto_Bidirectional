# PV2_0072 / PV2_0073 — the pipeline amplifies incoherent errors, and ours are coherent

**completed. This corrects a claim I made repeatedly this session and replaces it with a
mechanism.** Parent `PV2_0066`. Inference only, `BIM6_s1`, validation 300, contract v1.

## The claim that was wrong

I reported `PV2_0034`'s 0.9754 and `PV2_0066`'s 0.9671 as **"the architecture can reach
0.9, so feasibility is established"**, and repeated it in `STATUS.md` and several
commits. But `make_oracle_gamma.py` builds its tensor as

```python
codes = torch.randn(n_ids, CHANNELS)   # one fixed code per object id
gamma = codes[labels]                  # indexed by the TRUE patch labels
```

so the oracle gamma is **not better features — it is the answer**, written as per-object
codes. The number measures how faithfully the readout **transcribes a grouping it has
been handed**, not whether these dynamics can discover one. "Feasibility established"
does not follow from it, and that phrasing is corrected in `PV2_0066`'s record and in
`STATUS.md`.

## `PV2_0072` — the transfer function, and a surprise

True labels corrupted to a known degree (each foreground patch reassigned to another
object with probability p), fed to the **graph alone** with the phase path on real
gamma, with input and output both scored against ground truth by the same
`evaluate_patch_masks`:

| input fg_ari | output fg_ari | gain |
| --- | --- | --- |
| 1.0000 | 0.9678 | **-0.032** |
| 0.7525 | 0.7766 | **+0.024** |
| 0.5894 | 0.6473 | **+0.058** |
| 0.4205 | 0.5087 | **+0.088** |
| 0.1649 | 0.2900 | **+0.125** |
| 0.0416 | 0.1933 | **+0.152** |

**The map is not the identity: output beats input everywhere below a perfect input**,
most strongly for poor inputs, crossing over where output = input at roughly 0.85-0.88.
Read naively, iterating from our 0.7239 should climb toward that crossing. The pipeline
really does repair a damaged grouping.

## `PV2_0073` — and it does not climb on ours

Dense random codes per **predicted** id, fed to the graph, four rounds, no labels
anywhere:

| iteration | fg_ari | fg_iou | obj_iou | pred_fg | gain |
| --- | --- | --- | --- | --- | --- |
| 0 (real features) | 0.7239 | 0.7251 | 0.4951 | 0.1290 | — |
| 1 | 0.7269 | 0.7247 | 0.4936 | 0.1288 | +0.0030 |
| 2 | 0.7270 | 0.7247 | 0.4931 | 0.1288 | +0.0001 |
| 3 | **0.7274** | 0.7247 | 0.4932 | 0.1288 | +0.0004 |
| 4 | 0.7274 | 0.7247 | 0.4932 | 0.1288 | 0.0000 |
| transcription bound | 0.9678 | 0.9373 | 0.7585 | 0.1120 | |

**Converges immediately to 0.7274, +0.0035 on the baseline** — inside the seed spread of
earlier 3-seed means (0.0022-0.0045). A fixed point, as `PV2_0069` had found; the code
format moved it from -0.0040 to +0.0035 and both are negligible, so `PV2_0069`'s
conclusion stands and the one-hot truncation was not what drove it.

## The mechanism, which is the useful part

`PV2_0072`'s corruption is **random**: isolated patches reassigned independently, so the
errors are spatially incoherent — a wrong patch sits surrounded by right ones. That is
exactly what the generator's geodesic relaxation and distance prior are built to repair,
and the +0.15 at heavy corruption is them doing it.

The model's own errors are **spatially coherent**: whole objects merged or split, which
`PV2_0053` measured as an object pair too close to separate in 38.8% of images. A merged
pair looks like one internally consistent region, so there is nothing locally
inconsistent for the same machinery to fix.

**So the amplification is real but acts on the wrong error type.** That reconciles the
two results, and it says why nineteen routes have failed: every one of them tried to
reduce the amount of error, while the pipeline can only remove error of a kind it does
not make.

## Where part 3 stands now

Not met, and stated more precisely than before:

- the oracle numbers measure **transcription**, not discovery, so they do not establish
  that 0.9 is reachable;
- the pipeline **does** repair groupings, but only spatially incoherent damage;
- its own failures are spatially coherent object merges, and it is a fixed point on them
  at 0.727.

A route to 0.9 therefore needs something that separates two spatially adjacent,
similarly-coloured objects — information absent from a single static render, where the
only cue is the separation itself, which the pipeline already uses.
