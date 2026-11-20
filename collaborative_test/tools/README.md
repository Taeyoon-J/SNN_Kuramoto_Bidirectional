# tools

## `score_validation.py`

Scores a trained checkpoint on the **320 validation images** (CLEVR ids 1000-1319) using
the peer benchmark's own scoring code, unmodified.

**Why.** `full_benchmark/evaluation.py:62` reads `dataset.test` — for CLEVR the 320 images
the peer's own `data.py:19` calls *"fixed320 reserved evaluation, not an independent
original-paper TEST split"*, and `bootstrap.py:150` calls *"reused fixed320 validation, not
untouched test"*. Every configuration compared costs another read of it. The 320 validation
images had never been scored by anything in this benchmark, so selection belongs there.

**How it avoids touching locked code.** `evaluate()` is called as-is. The only change is a
shallow copy of the `Dataset` whose `.test` points at `.validation`. That is sound because
`Dataset.rgb` and `Dataset.labels` index the HDF5 (CLEVR) or the `.npy` (others) **by image
id** — the split name only routes CLEVR's `'train'` branch, which this never takes. So the
predictions, the spike clustering, the merge, the metric module and the `c.immutable()`
check are the originals, byte for byte.

**Verified.** Run against the completed 253 10,000x10 checkpoint it produced
**0.7862 / 0.7513 / 0.5808**, against fixed320's 0.7794 / 0.7308 / 0.5807 — two disjoint
320-image sets agreeing to 0.02 at worst and 0.0001 on object IoU. That agreement is what
makes validation usable for selection: a setting chosen there should carry to fixed320.

```
python -I score_validation.py --config configs/<cfg>.json \
  --run <output root the job trained into> --model 253 --dataset clevr --seed 0
```

Output goes to `<run>/jobs/model<N>/<dataset>/seed<S>/validation/result.json`, tagged
`split: validation` and `NOT_fixed320: true` so it can never be mistaken for a fixed320
number. Needs a GPU for inference only — 320 images, free on frontier.
