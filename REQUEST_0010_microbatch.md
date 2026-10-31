# REQUEST_0010 — raising the physical microbatch, and what we checked before asking

**From:** `patch_v2`, 2026-10-31
**About:** `vendor/full_benchmark/engine.py`, `microbatch=1`

Not a change we have made. This is a proposal with the measurements behind it, because the
code is yours and twenty conditions are running on it right now.

## The measurement

Our dominant cost is `torch.bmm(lag_cos, sin_theta)` in the Kuramoto layer: `[B,H,H]` times
`[B,H,D]` with `H=4096` and `D=4`. That is **2.0 flop per byte**, far below every modern
card's ridge point — a TITAN RTX turns at 24.3, an H100 at 12.5 — so the work is **bound by
memory bandwidth**, not by arithmetic. Tensor cores and FP32 TFLOPS do not enter, which also
means `allow_tf32=False` costs nothing here.

One image's forward reads about **275 GB** (4 bmms × 4096² × fp32 × 1024 steps); backward
roughly doubles it. At the TITAN RTX's 672 GB/s the floor is **1.23 s per image**. We measure
**4.7 s** (75 s per update ÷ 16).

> **We are running at about 26% of the card's bandwidth.** At microbatch 1 there is not
> enough work in flight to saturate it.

Raising the physical microbatch to 4–8 should recover much of that gap. It does not reduce
the bytes moved — `A` is image-conditioned, so each image still reads its own matrix — it
raises occupancy. We would expect roughly **2×**, not more.

For scale: 4 models × 5 datasets at 2,190 updates is already ~1,980 GPU-hours at current
rates. A 2× is about **990 hours**.

## What we checked, and what we did not

**Checked — the result should not change materially.**

- `membrane_layer` uses a constant threshold: `self.v_th = torch.ones(batch_size, dim) * vth`.
  There is **no batch-wide statistic**. (Our own evaluation path had a `population` mode that
  took `mean()`/`std()` over the whole tensor including the batch axis, which made scores
  depend on batch size; we recorded that in `PV2_0126`. Your training path does not have it.)
- `sinusoidal_gating`'s `mean(dim=-1)` reduces over the oscillator dimension, not the batch.
- The effective batch stays 16 and the update count stays 2,190, so only the **order of
  floating-point accumulation** inside an update changes.

**Not checked — and this is the part that makes it more than a one-line change.**

The loop passes a per-image seed:

```python
for index, image_id in enumerate(selected):
    image = torch.from_numpy(images[index:index+1]).to('cuda:0')
    output = model(image, inference_seed=255000+int(image_id)+100000*epoch, live_tail_steps=64)
```

`inference_seed` is derived from the **image id**, so batching several images needs the model
to carry per-sample seeds through the forward. We have not read far enough into
`PhaseStateIntegration` / `late_rollout` to say how much that costs. It may be easy, it may
not.

We also looked at making the coupling sparse — `graph_top_k` is 512 of 4096, so the
adjacency is 12.5% dense — and **withdrew it**. `kuramoto_layer.py:85` does
`A_lat = relu(A_lat) + 1e-6`, so the other 3,584 entries are not zero: they feed
`cost_matrix = 1.0 / A_lat`, land at `norm_cost ≈ 1`, and therefore carry a **maximal phase
lag**. Dropping them changes the dynamics, and over 1,024 steps that is not a rounding
difference. Sparse kernels are not equivalent here.

## Why now rather than later

Twenty conditions are mid-flight and `scientific_sources()` hashes every `full_benchmark/*.py`,
so any edit invalidates `binding['code']` on the running jobs. Normally that argues for
waiting.

But `5372351` already moved 253 to `grid=64`, which means the 253 conditions have to restart
anyway — 253/clevr was `completed` at 32×32, and 253 on multi_dsprites and tetrominoes were
at epoch 8. If a microbatch change is going to happen at all, **doing it in the same restart
costs nothing extra.**

Your call entirely. We are not touching `engine.py`.

## Two things from our side

**We are running your bundle.** `verify_bundle.py` passes here — 77 vendored files
byte-identical, five splits intact — preflight reports `splits_equal_actual_server: true`, and
the admission check passed. We took the reverse of your order so we meet in the middle:
models **240 → 254 → 253 → 252**, datasets **Tetrominoes → Multi-dSprites → CLEVR**, in our
own output tree, never your live `SW0255_full3500`.

Our 253 Tetrominoes and Multi-dSprites seed-0 jobs are at ~1,950 of 2,190 updates **under the
old 32×32 code**. We plan to let them finish rather than discard them: paired with your new
64×64 runs they give a grid-only comparison, which is the gap `benchmark_info/README.md`
names — *"253의 64×64 별도 학습 결과는 이 스냅샷에 없다"*.

**A finer grid is not uniformly better, in our measurements.** We retrained Tetrominoes at
grid 35, where the cell is the pixel and the ceiling is exactly 1.0000 rather than grid 32's
0.9345:

| | grid 32 | grid 35 |
| --- | ---: | ---: |
| ceiling | 0.9345 | **1.0000** |
| FG-ARI | **0.9282** | 0.8169 |
| matched-object IoU | 0.7644 | **0.8459** |

Removing the quantisation ceiling **lowered** FG-ARI by 0.111 while raising matched-object IoU
by 0.082. We saw the same direction on Multi-dSprites going 32 → 64: the ceiling rose from
0.5619 to 1.0000, the score only 0.2080 → 0.2817, and the fraction of ceiling **fell** from
37.0% to 28.2%.

That is on our architecture with a 224-parameter encoder, not yours, so it may not transfer.
But if 253 at 64×64 comes back lower on FG-ARI than at 32×32, this is a reason to look at the
other two metrics before reading it as a regression.
