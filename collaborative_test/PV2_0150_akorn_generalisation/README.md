# PV2_0150 — AKOrN trained on CLEVR, tested on all three, on our exact budget

**Status** completed. Headline rows are **seed 0 on both sides**; a second AKOrN seed came for
free and is reported as a spread, not as a mean.
**Parent** `PV2_0149` (protocol A, exposure-matched baselines) and `PV2_0137` (our own
CLEVR-trained core evaluated zero-shot on Tetrominoes and Multi-dSprites).

**Hypothesis.** Our headline claim is a *generalisation* claim: a core trained on CLEVR alone
scores well on datasets it never saw. A baseline trained on the target dataset does not test
that claim. So AKOrN — the closest competitor, Kuramoto-based, and the one whose published
3-seed numbers on exactly our three datasets are above ours — is trained on **the same 10,000
CLEVR images**, with no further training, and evaluated on **all three** reserved splits, the
same way our frozen core is.

**Changed variable.** Which dataset the baseline trains on (CLEVR, not the target), holding the
exposure budget, the train ids, the evaluation ids, the grid and the evaluation function fixed.

**Success criterion.** Our CLEVR-trained core beats the CLEVR-trained AKOrN on all three metrics
on all three datasets, under our evaluation contract.

## Result — both models trained on CLEVR only, 100,000 exposures, seed 0

Our evaluation function, grid 64, the same 320 reserved ids per dataset. Background rule chosen
on validation out of the same four rules that chose Slot Attention's (`bg_any.py`).

| dataset | model | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|
| **CLEVR** (trained here) | **253 core + label-free head** | **0.8468** | **0.7574** | **0.6456** |
| | AKOrN (attn), CLEVR | 0.6027 | 0.3327 | 0.3750 |
| **Tetrominoes** (zero-shot) | **253 core** | **0.8580** | **0.7885** | **0.8173** |
| | AKOrN (attn), CLEVR | 0.5676 | 0.7383 | 0.4687 |
| **Multi-dSprites** (zero-shot) | **253 core** | **0.6974** | **0.8177** | **0.7628** |
| | AKOrN (attn), CLEVR | 0.4343 | 0.6019 | 0.5751 |

**Nine cells out of nine go to us**, the narrowest being Tetrominoes Foreground IoU
(0.7885 vs 0.7383, **+0.0502, +6.8%**) and the widest CLEVR Foreground IoU
(0.7574 vs 0.3327, **+0.4247, +128%**).

### How much of that is seed noise

The first AKOrN runs went out without `--seed`, so they used **AKOrN's default 1234**; seed 0 was
then trained to match ours. Keeping both gives a two-seed spread for free, and it is not small:

| dataset | AKOrN seed 0 | AKOrN seed 1234 | spread | our margin over the *better* AKOrN seed |
|---|---|---|---|---|
| CLEVR | 0.6027 | 0.4688 | **0.1339** | +0.2441 |
| Tetrominoes | 0.5676 | 0.5141 | 0.0535 | +0.2904 |
| Multi-dSprites | 0.4343 | 0.4249 | 0.0094 | +0.2631 |

**Every margin is larger than the spread**, on CLEVR by 1.8× and on the other two by 5× and 28×.
That is the most this experiment can say about robustness: it is two seeds of AKOrN against one
of ours, not a 3-seed comparison, and the contract's bar is still unmet on both sides.

### AKOrN trained on the target dataset instead

Also run, because it is the comparison a reader will ask for: AKOrN on Multi-dSprites itself,
same 100,000 exposures, its own published model config, seed 0.

| Multi-dSprites fixed320 | trained on | fg_ari | foreground_iou | matched_object_iou |
|---|---|---|---|---|
| **253 core** | **CLEVR** | **0.6974** | **0.8177** | **0.7628** |
| AKOrN (attn) | Multi-dSprites | 0.4562 | 0.6830 | 0.6084 |
| AKOrN (attn) | CLEVR | 0.4343 | 0.6019 | 0.5751 |
| 253 core | Multi-dSprites | 0.1201 | 0.0597 | 0.0050 |

Two things read off this at once:

1. **Training AKOrN on the target data barely helps it** — 0.4343 → 0.4562 FG-ARI, a gain of
   0.0219, which is inside its own CLEVR seed spread. It does not reach our zero-shot number
   either way. So our margin here is not an artefact of AKOrN being tested out of domain.
2. **Our core is the opposite** — target-trained 0.1201, CLEVR-trained 0.6974, a 5.8× gap in the
   *wrong* direction for ordinary transfer. `PV2_0147` is where that is analysed; it says the
   target-trained cores are broken, not that CLEVR is magic.

### AKOrN's own metric, for comparability

`eval_obj.py` unchanged, its own FG-ARI and mBO implementation, seed 0:

| model | data | its FG-ARI | ours after grid reduction | diff | its mBO |
|---|---|---|---|---|---|
| AKOrN (attn), CLEVR-trained | CLEVR | 0.6035 | 0.6027 | −0.0008 | 0.4175 |
| | Tetrominoes | 0.5680 | 0.5676 | −0.0004 | 0.5465 |
| | Multi-dSprites | 0.4353 | 0.4343 | −0.0010 | 0.5858 |
| AKOrN (attn), dSprites-trained | Multi-dSprites | 0.4562 | 0.4562 | 0.0000 | 0.6160 |
| **published, 3-seed means** | Tetrominoes / dSprites / CLEVR | **88.6±1.7 / 78.3±1.3 / 91.0±0.5** | — | — | 56.4±0.9 / 63.0±1.8 / 45.5±1.4 |

Its own FG-ARI and ours agree to **≤0.0010** on every cell, and exactly on the one row that needs
no grid reduction. That is the check that matters here: the numbers in the headline table are not
an artefact of our scorer. Its mBO is **not** comparable to its published mBO, because the truth
we feed it is the grid quantisation, not the true pixel mask; it is recorded as a diagnostic and
no claim rests on it.

## The honest limit of this result

**Licenses:** "on an identical data budget, from the same 10,000 CLEVR images, our core beats
AKOrN on all three metrics on all three datasets, including two it never trained on, by a margin
larger than AKOrN's own seed spread."

**Does not license** "we beat AKOrN." The gap between 0.6035 and its published 91.0 on CLEVR is
**not a protocol artefact we can wave away — it is the budget.** Its recipe is 300 epochs over
the full ~70,000 CLEVR train set, about **21,000,000 exposures: 210× what we gave it.** A model
given 1/210 of its training is not the model its authors report, exactly as `PV2_0149` says of
Slot Attention at 1/320.

So this is the same **data-efficiency** claim as PV2_0149, now against a much stronger baseline,
and it is measured. Protocol B — AKOrN at its published 300 epochs on our render — is the only
version that would support the stronger sentence, and it has not been run. Its published numbers
are also on the pixel protocol and on the full official test splits, so they are not directly
comparable to the table above in either direction.

Our side is **one seed**. The contract judges 3-seed means and that bar is unmet.

## Method notes, and the two choices that could have gone wrong

- **The truth tensor is shared.** AKOrN's loaders take `images` and `labels` from an `.npz`. We
  write `labels` as **our own grid-64 truth, `np.repeat`-ed ×2 to 128** — not the raw CLEVR h5
  pixel mask. That makes the tensor AKOrN scores itself against the identical tensor
  `evaluate_patch_masks` uses, and it removes any question of whether `Dataset.pixels` (which
  indexes the h5 globally) and `Dataset.labels` (which goes through the split manifest) agree on
  row order. The ×2 lift is exact, so it costs AKOrN nothing.
- **Predictions are reduced to the grid, never the reverse.** AKOrN emits a hard partition at
  image resolution 128. `to_grid64.py` reduces it by the contract's own **patch majority vote**,
  ties to the smaller label id. The contract's `np.repeat` lift handles predictions coarser than
  the grid; this is the same rule in the other direction. No prediction is expanded to pixels,
  and the table above bounds what the reduction costs AKOrN at ≤0.0010 FG-ARI.
- **AKOrN is given a fixed cluster count.** Its `eval_obj.py` runs agglomerative clustering with
  `n_clusters` fixed per dataset (4 Tetrominoes, 7 dSprites, 11 CLEVR). That is a dataset-level
  constant, not per-image truth, so it is not an oracle under our contract — but it is
  information our readout does not get, and it favours AKOrN. We left it in place.
- **The evaluation loader shuffles.** `get_loader` passes `shuffle=True` even at eval. That does
  not change AKOrN's own means, but it destroys the alignment between predictions and image ids.
  `eval_dump.py` (our copy) sets `shuffle=False` so the dumped order matches the id order; the
  original `eval_obj.py` is untouched, and its reported numbers are unchanged by the flag
  (0.4514 / 0.4561 / 0.4524 across reruns of the same checkpoint, which is the clustering's own
  run-to-run spread and brackets both settings).
- **Model config is AKOrN's published one.** Only the epoch count changed, to match exposures:
  CLEVR `--ch=256 --psize=8`, 10 epochs instead of 300; Multi-dSprites `--ch=128 --psize=4`,
  10 epochs instead of 50. The exposure budget is 10 passes over 10,000 unique images = 100,000
  exposures; it does not depend on batch size. (`--batchsize` defaults to **256**, so an epoch is
  40 batches with a partial last one, and `train_obj.py:241` notes `PairDataset` feeds two views
  per sample, so the real forward is twice the batch. An earlier version of this file said
  "batch 250 × 40 batches", which was wrong in the arithmetic though not in the exposure count.)
- **Final-epoch checkpoints are saved bare.** `train_obj.py` writes `ema_model.pth` as a plain
  `state_dict` while `eval_obj.py` expects `{"model_state_dict": ...}` (that is what the
  periodic `save_model` writes). We wrap it into `ema_wrapped.pth`; no weights change.

## Diagnostics

Predicted foreground fraction against the truth's, under the selected rule, on fixed320, seed 0:

| dataset | truth fg | AKOrN pred fg |
|---|---|---|
| CLEVR | 0.2115 | 0.6140 |
| Tetrominoes | 0.2368 | 0.2800 |
| Multi-dSprites | 0.0943 | 0.1798 |

On CLEVR AKOrN calls **61% of the image foreground where the truth says 21%**, which is why its
Foreground IoU there is 0.3327 while its FG-ARI is 0.6027 — the same tiling failure mode
`PV2_0149` found in Slot Attention, milder. On the two simpler renders its foreground fraction is
closer to the truth's, and there the gap to us is carried by FG-ARI and matched-object IoU
instead. All four background rules give the **identical FG-ARI** on every dataset, for the reason
recorded in `sa_bg.py`: they only relabel patches the truth calls background, and FG-ARI excludes
those. So the rule choice moves Foreground IoU and matched-object IoU only, and it was still made
on validation.

## Interpretation

AKOrN is the strongest comparison we have — same mechanism family, published 3-seed numbers on
our exact three datasets, and higher than ours there. On our budget it lands at 0.43–0.60 FG-ARI,
and **it does not matter much which dataset it trained on**: CLEVR-trained 0.4343 vs
dSprites-trained 0.4562 on Multi-dSprites, a difference inside its own seed spread. It is not
learning something that transfers and it is not learning something dataset-specific either; at
100,000 exposures it is simply early.

That is the shape of a data-efficiency gap, not a generalisation gap, and it is worth saying
plainly: **this experiment shows our core is far better at 100,000 exposures, and says nothing
about who wins at 21,000,000.** Our own claim is the narrower one and survives: a core trained
on CLEVR alone, frozen, scores 0.8580 and 0.6974 FG-ARI on two renders it never saw, and a
baseline given the same images and the same budget reaches 0.5676 and 0.4343.

## Next

1. **Seeds.** Our side is seed 0 and the contract judges 3-seed means. AKOrN is cheap — 7 min
   per CLEVR run on a TITAN RTX — so its seeds 1 and 2 cost nothing. Our own CLEVR seeds 1 and 2
   remain the expensive, unapproved item, and the two-seed spread above says they are not a
   formality: a 0.134 swing on CLEVR from a baseline reseed is the scale of noise to expect.
2. **Protocol B for AKOrN.** 300 epochs on our 10,000 CLEVR images is ~3.5 h at the measured
   rate, and 300 epochs on the full 70,000 is ~24 h. The first is "its recipe on our data
   budget"; only the second is "its recipe". Decide which, and label it as what it is.
3. The remaining PV2_0149 baselines (Rotating Features, complex autoencoders) are still lower
   value: their published numbers already coincide with our protocol on Multi-dSprites.

## Reproduce

See `commands.sh`. Short form:

```bash
# data: our splits in AKOrN's npz format, truth = our grid-64 labels lifted x2
python akorn_clevr_data.py        # clevr_{train,val,eval320}.npz
python akorn_data128.py           # tetrominoes/dsprites eval frames at 128

# train, CLEVR only, published config, epochs cut to match 100,000 exposures
python -u train_obj.py --exp_name=s0_clevr_akorn_attn --data=clevr --seed=0 \
  --model=akorn --J=attn --L=1 --ch=256 --psize=8 --epochs=10 --c_norm=none

# predictions, order-preserving, at 128
DUMP=s0C_clevr_eval320.npz python -u eval_dump.py --data=clevr --data_imsize=128 \
  --model=akorn --model_path=runs/s0_clevr_akorn_attn/ema_wrapped.pth \
  --J=attn --L=1 --T=8 --ch=256 --psize=8 --c_norm=none --model_imsize=128

# reduce to the grid, then our evaluation with the rule chosen on validation
python benchmark_codes/to_grid64.py --inp s0C_clevr_eval320.npz --out ..._g64.npz
python benchmark_codes/bg_any.py --val ..._val_g64.npz --eval ..._eval320_g64.npz \
  --val-ids clevr_val.npz --eval-ids clevr_eval320.npz --dataset clevr \
  --code benchmark_codes --config benchmark_codes/configs/frontier.json \
  --run /work/USERS/tkim1/sw0255b/run --out s0_clevr_bg.json
```
