# PV2_0135 — the rollout is deterministic, so PV2_0134's change C is withdrawn

**Status** completed
**Parent** PV2_0134
**Effect on earlier results** withdraws one conclusion from PV2_0134 and leaves the rest
standing. See "What this does and does not touch" below.

## What was claimed

PV2_0134 reported change C — training the head to agree with itself across rollout seeds — as
a **negative result**: "with the guards, no arm on any dataset beat its own epoch 0. Cross-seed
agreement does not help this head."

That conclusion is not supported. The experiment never had two seeds.

## What was found

Three things, in the order they came up.

**1. The cached second seed is bitwise identical to the first.** Comparing `features` against
`features_b` and `gamma` against `gamma_b` in `/work/USERS/tkim1/sw0255b/ev/clevr_train`:

```
img  3: features identical=True  mean|df|=0.000000   gamma identical=True
img 24: features identical=True  mean|df|=0.000000   gamma identical=True
img 29: features identical=True  mean|df|=0.000000   gamma identical=True
```

**2. `regions_b` is not a second-seed partition at all — it is `regions_all`.**

```
img  3: regions_b == regions_all -> True   n: b=11 all=11 253=3
img 24: regions_b == regions_all -> True   n: b=30 all=30 253=7
img 29: regions_b == regions_all -> True   n: b=33 all=33 253=8
```

`cache_seed2.py:58-59` passes `np.ones((GRID,GRID), dtype=np.int64)` as the support to
`foreground_induced_merge`, which is the all-ones merge — the same partition change A was
introduced to replace. So the one field that did differ between the two passes differed for the
wrong reason: a different merge support, not a different seed.

**3. The rollout is deterministic in `inference_seed`.** This is the root cause, and it was
checked against the core itself rather than inferred from the cache. `seedtest.py` rolls one
image out three times at seeds 0, 777003 and 12345 with the same checkpoint in eval mode:

| tensor | seed 0 vs 777003 | seed 0 vs 12345 |
|---|---|---|
| `gamma` | identical, max\|diff\| 0.000e+00 | identical, 0.000e+00 |
| `q` | identical, 0.000e+00 | identical, 0.000e+00 |
| `adjacency` | identical, 0.000e+00 | identical, 0.000e+00 |
| `trace['spikes']` | identical, 0.000e+00 | identical, 0.000e+00 |
| `trace['component_spikes']` | identical, 0.000e+00 | identical, 0.000e+00 |

Spike tensor `(1, 4096, 1024)`, mean rate 0.0693, 290,599 spikes — a full rollout, reproduced
exactly. `FreshModel.forward` (`vendor/full_benchmark/model.py:76-78`) does seed both
generators, but nothing downstream draws from them: `encode` is a convolution and `late_rollout`
in eval mode injects no noise.

## Why this makes C impossible rather than failed

The objective compared `logits(features, gamma, masks, prior)` against
`logits(features_b, gamma_b, masks, prior)` with the candidate set deliberately shared so the
index spaces would align. With `features_b == features` and `gamma_b == gamma`, the two sides
were the same tensor, so the symmetric KL was identically zero and carried no information. The
collapse PV2_0134 attributed to the objective's trivial minimum was the head drifting under a
term that measured nothing.

So the correct statement is: **seed invariance is not an available signal for this core, because
its inference is deterministic.** Whether cross-seed agreement would help a stochastic rollout
is untested and remains open.

## What this does and does not touch

**Withdrawn.** Change C's result, and the claim that cross-seed agreement does not help.
`cache_seed2.py`'s premise in its own docstring ("the rollout is stochastic in its inference
seed") is false for inference and should not be reused.

**Unaffected.** Nothing else in PV2_0134 read the second-seed cache:

- **Change A's isolation** — the `regions=all` control, Foreground IoU pinned at 0.2115 with
  `+0.0000` after training versus 0.7830 with 253's partition.
- **Rule B′ and its motivation.** The head that scored +0.0010 FG-ARI and −0.1141 Foreground
  IoU on Tetrominoes was change C's **epoch 0** — the head before any update. Epoch 0 involves
  no training and no second-seed tensor, so the observation that selecting on FG-ARI alone
  accepts that trade is valid independently of C, and B′ stands.
- **The transfer result** — the CLEVR head unchanged loses on all three metrics on both other
  datasets.
- **The recipe result** — label-free on CLEVR and Multi-dSprites, losing 0.0565 FG-ARI on
  Tetrominoes.
- **The Slot Attention rule sweep** and the three-dataset table.

`PV2_0134/README.md` carries a correction notice pointing here.

## Consequences worth keeping

- The `features_b` / `gamma_b` / `regions_b` fields in `/work/USERS/tkim1/sw0255b/ev/*_train`
  are junk and must not be read. They are left in place rather than deleted so this record can
  be re-checked, and `stop.py`'s second-seed path is disabled.
- **Determinism is a useful property**, not only a lost opportunity: every activity cache,
  diagnostic and score in this project is exactly reproducible from the checkpoint and the
  image, with no seed to record and no averaging over draws. That is the opposite of Slot
  Attention, which resamples its slots on every forward pass and needed `--seed` added to
  `run_slot_attention.py` to be reproducible at all.
- A label-free epoch rule is still the open problem from PV2_0134, and it now has one fewer
  candidate. The remaining one is **stability** — the fraction of patches whose predicted
  label changes between epochs, measured on unlabelled training images.

## Reproduce

```bash
cd /work/USERS/tkim1/sw0255b
CUDA_VISIBLE_DEVICES=2 /work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python seedtest.py
/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python diagnostics/compare_cached_seeds.py
/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python diagnostics/compare_regions_b.py
```
