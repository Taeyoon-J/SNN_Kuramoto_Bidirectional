# PV2_0126 — the evaluation batch size was hardcoded, and it is not an inert setting

- **status** completed
- **parent** `PV2_0125`
- **branch** `patch_v2`
- **type** infrastructure fix plus a reproducibility finding. No model change, no new score.

## Why this exists

`PV2_0125` needed Multi-dSprites at grid 64 because the grid-32 ceiling there is 0.5619 and
no win is available below it. The grid-64 evaluation died:

```
torch.OutOfMemoryError: Tried to allocate 2.00 MiB.
GPU has 23.47 GiB of which 2.50 MiB is free; this process has 23.46 GiB in use.
```

## Cause

`evaluate_model.py`'s forward loop had the batch size written in as a literal:

```python
for s in range(0, gamma.size(0), 25):
```

25 is right at grid 32 and cannot work at grid 64. The node count quadruples from 1024 to
4096, but the tensors do not all grow with it at the same rate:

| tensor | shape | grid 32, batch 25 | grid 64, batch 25 |
| --- | --- | ---: | ---: |
| theta | `[B,T,N,D]` | 419 MB | 1.68 GB |
| membrane | `[B,T,N]` | 105 MB | 419 MB |
| spikes | `[B,T,N]` | 105 MB | 419 MB |
| coupling | `[B,N,N]` | 105 MB | **1.68 GB** |

The per-node tensors grow 4x with the node count; the coupling matrix grows with its
**square**, 16x. So the batch that fits is not 4x smaller but about 16x smaller.

## Fix

`--eval-batch`, default 25, so every recorded number is reproduced by the default and only
a run that needs it changes it. Grid 64 runs at 4 and peaked at 5.5 GB.

## The refactor is inert — checked, not assumed

Same checkpoint, same 300 validation images, grid 32, `--eval-batch 25`:

| | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | ---: | ---: | ---: | ---: |
| before the change (`PV2_0125`) | 0.9362 | 0.6914 | 0.7817 | 0.3004 |
| after, `--eval-batch 25` | 0.9362 | 0.6914 | 0.7817 | 0.3004 |

Identical to four decimals on all four.

A first attempt at this check was worthless and is recorded as such: it ran `--limit 50`
against a number measured on 300 images, so the 0.9251 it returned could not be compared
to anything. Different subsets are not a regression test.

## The finding: the score depends on how images are grouped into batches

`membrane_layer._threshold`, in the `population` mode every result on these datasets uses:

```python
ref = mem.detach()
return ref.mean() + self.threshold_k * ref.std().clamp_min(1e-6)
```

`mean()` and `std()` reduce over the **whole tensor, batch dimension included**, so the
firing threshold is one scalar per forward pass — and a forward pass is a batch. Two runs of
the same checkpoint on the same images with different `--eval-batch` therefore threshold at
different levels and can score differently.

In expectation the scalar is the same for exchangeable images, so this is variance rather
than bias, and `--eval-batch 25` reproduced exactly above because the batch *composition*
was also identical. But it means:

- **a score is only reproducible together with its batch size.** Every number in the records
  was measured at 25; grid 64 cannot use 25, so a grid-64 number is not strictly comparable
  to a grid-32 one and that has to be said rather than assumed away.
- the per-image metrics are not per-image in the way they look: image *i*'s mask depends on
  which other images shared its forward pass.

**Not fixed here, deliberately.** The obvious repair is to reduce over all dimensions except
the batch, giving one threshold per image, which removes the dependence entirely and is
arguably what the docstring intends ("so the membrane's absolute scale stops deciding
whether anything fires" — each image has its own scale). That is a core change: it alters
every number this project has produced, so per the scope rule it gets its own test with a
stated ablation, not a quiet edit inside an infrastructure fix.

A partial measurement exists and is reported for what it is: batch 25 gave
0.9362 / 0.6914 / 0.7817; the 8, 4 and 2 arms were cancelled mid-run when the benchmark
table was prioritised, so the size of the effect is **unmeasured**.

## Changes

`collaborative_test/evaluate_model.py` at this commit:

| what | before | after |
| --- | --- | --- |
| forward loop stride | `range(0, gamma.size(0), 25)` | `range(0, gamma.size(0), args.eval_batch)` |
| 4 slice expressions | `[s:s + 25]` | `[s:s + args.eval_batch]` |
| new flag | — | `--eval-batch`, type int, default 25 |

`collaborative_test/build_mod_targets.py`: `names`, `patch_size` and the manifest's
`splits`/`counts` are now written by the builder, with an assert that the named image files
exist. Previously added by `addnames.py` and `fixman.py` afterwards, which meant a rebuild
at a new grid rewrote the manifest from `ranges` alone and silently dropped `splits`,
invalidating grids that were already finished. All five dataset x grid configurations pass
`preflight.py` 8/8 after the rebuild.

## Next

The grid-64 Multi-dSprites number this unblocked, and the CLEVR row at three seeds on the
1000 test images. Both were running when the session's VPN dropped; see `STATUS.md`.
