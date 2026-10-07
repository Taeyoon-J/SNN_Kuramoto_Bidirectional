# PV2_0047 — a properly trained matched Slot Attention beats us on fg_ari

**completed. This corrects the comparison the project has been making.** Parent
`PV2_0044`. Validation 300, contract v1, the same split our own scores use.

## Why it was needed

The Slot column in `baselines/` is one published checkpoint trained on the
original CLEVR render, and the same checkpoint scores FG-ARI **0.6195** under this
contract and **0.8946** under the peer's. A reference that moves by 0.275 with the
measurement cannot settle whether we beat it.

The peer then produced a matched baseline at **0.5711**, which made the comparison
look comfortable. But their budget is 2,500 images at exactly ten exposures each
-- **25,000 presentations, 0.08% of the official recipe**.

## What was run

The **official architecture** from `baselines/slot_attention/model.py`, not a
reimplementation, with the official object-discovery recipe: Adam 4e-4, 10k linear
warmup then halving every 100k, gradient clipping 0.05, 7 slots, 3 iterations,
128x128, reconstruction MSE. Trained on **our** train split, images 0-5999.

**Budget: 60,000 steps at batch 32 = 1,920,000 presentations = 320 epochs over
6,000 images.** The official recipe is 500,000 steps at batch 64 = 32,000,000
presentations, roughly 457 epochs over its own larger set. So this is about 70% of
the official epoch count and **77x the peer's matched budget**.

## Result

| | FG-ARI | foreground IoU | matched-object IoU |
| --- | --- | --- | --- |
| matched SA, seed 0 | 0.7223 | 0.1134 | 0.0918 |
| matched SA, seed 1 | 0.7391 | 0.1147 | 0.0872 |
| **matched SA mean** | **0.7307** | **0.1141** | **0.0895** |
| **ours (3-seed, `PV2_0044`)** | **0.7250** | **0.7188** | **0.4953** |
| difference | **-0.0057** | **+0.6047** | **+0.4058** |

**Slot Attention is ahead of us on fg_ari by 0.006.** We are ahead on foreground
IoU by 6.3x and on matched-object IoU by 5.5x.

So the peer's 0.5711 was an under-trained baseline. Trained properly on the same
split it reaches **0.7307**, above even the transferred checkpoint's 0.6195 --
training on our own data helps it, as expected.

**The goal is not met.** The protocol requires beating the reference on all three
metrics as 3-seed means, and fg_ari is short.

## What the numbers say about the two models

Slot Attention predicts a background fraction of **0.153** where the target is
**0.882** -- it calls about 85% of patches foreground.

That explains the shape of its scores. fg_ari is computed over true-foreground
patches only, so a model that partitions those well scores highly even if it has
no idea which patches are foreground; both IoUs punish exactly that. **Slot
Attention groups well and cannot locate objects; this model does both.** The peer
saw the same structure from their side ("both IoUs exceed the Slot reference"), and
the cause is now explicit.

## Limitations

- **Two seeds, not three.** Seed 2 died twice to CUDA OOM: first because two
  TensorFlow processes were put on one GPU, which does not work since TF claims
  nearly all memory by default, and then because the collaborator took GPUs 2 and
  3. `TF_FORCE_GPU_ALLOW_GROWTH=true` fixed the inference path and would fix the
  training path too.
- **Validation, not test.** Our own test figure remains `PV2_0008`.
- 320 epochs against the official ~457. Shorter, so this is still a lower bound on
  Slot Attention, and the gap on fg_ari would likely widen rather than close with
  a longer run.
