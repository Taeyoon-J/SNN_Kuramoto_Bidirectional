# PV2_0151 — the whole baseline field at our budget, and we beat the best of it in every cell

**Status** completed, seed 0 everywhere.
**Parent** `PV2_0150` (AKOrN attn, CLEVR-trained, all three datasets) and `PV2_0149`
(protocol A, Slot Attention).

**Hypothesis.** PV2_0150 beat one baseline. A single baseline can be the wrong one — it might be
the one that happens to suit a 100,000-exposure budget badly. Adding the two models that live in
the same repo costs nothing (same data, same environment, same evaluation) and asks a harder
question: **is there any baseline in this field that beats us in any cell at this budget?**

**Changed variable.** The baseline architecture. Everything else held: the same 10,000 CLEVR
train images, 100,000 exposures, seed 0, the same 320 reserved ids per dataset, the same
evaluation function, the same four background rules selected on validation.

**Success criterion.** Our CLEVR-trained core beats the **best baseline per cell** — not the
average, and not a baseline of our choosing — on all three metrics on all three datasets.

## Result — every model trained on CLEVR only, 100,000 exposures, seed 0

Our evaluation function, grid 64, the same reserved 320 ids. `fg_ari / foreground_iou /
matched_object_iou`. Our rows are the **core alone, no head**, which is what the transfer rows
have always been (`PV2_0137`).

| model | CLEVR | Tetrominoes (zero-shot) | Multi-dSprites (zero-shot) |
|---|---|---|---|
| **ours, 253 core** | **0.8421 / 0.7545 / 0.6410** | **0.8580 / 0.7885 / 0.8173** | **0.6974 / 0.8177 / 0.7628** |
| AKOrN (attn) | 0.6027 / 0.3327 / 0.3750 | 0.5676 / 0.7383 / 0.4687 | 0.4343 / 0.6019 / 0.5751 |
| AKOrN (conv) | 0.6046 / 0.4219 / 0.2769 | 0.7202 / 0.7540 / 0.5913 | 0.4055 / 0.6834 / 0.5434 |
| ItrSA | 0.5794 / 0.2787 / 0.3853 | 0.6319 / 0.7137 / 0.5687 | 0.4817 / 0.2489 / 0.6093 |
| Slot Attention | 0.2872 / 0.2220 / 0.0801 | not run | not run |
| **best baseline, per cell** | 0.6046 / 0.4219 / 0.3853 | 0.7202 / 0.7540 / 0.5913 | 0.4817 / 0.6834 / 0.6093 |
| **our margin over the best** | **+0.2375 / +0.3326 / +0.2557** | **+0.1378 / +0.0345 / +0.2260** | **+0.2157 / +0.1343 / +0.1535** |

**Nine cells out of nine, against the best baseline in each cell separately.** The best-per-cell
row is a composite of three different models — no single baseline achieves it — so this is a
harder bar than beating any one of them.

The **narrowest margin in the whole table is Tetrominoes Foreground IoU, +0.0345** over AKOrN
(conv). That is the one cell a reseed could plausibly touch: PV2_0150 measured AKOrN's own
two-seed FG-ARI spread on Tetrominoes at 0.0535, and we have no Foreground IoU spread for the
conv variant at all. **Treat that single cell as unsettled.** The other eight margins are
0.134–0.333, well clear of any spread we have measured.

### Where each baseline is strong tells you what it learned

- **AKOrN (conv)** is the best baseline on **Tetrominoes** (0.7202) and the *worst* of the three
  on Multi-dSprites (0.4055). Its published table has the same shape in miniature — conv is
  AKOrN's weaker variant overall (59.0 vs 91.0 on CLEVR) yet it holds up on the simplest render.
- **ItrSA** is the best baseline on **Multi-dSprites** FG-ARI (0.4817) and matched-object IoU
  (0.6093), and the worst on CLEVR Foreground IoU (0.2787) — it calls **72% of the CLEVR image
  foreground where the truth says 21%**, and 50% on Multi-dSprites against a truth of 9%.
- **AKOrN (attn)** is the best baseline on CLEVR matched-object IoU after ItrSA and middling
  everywhere else, despite being the variant with the best published numbers.

So the field does not have a consistent second place: **which baseline is second depends on the
dataset**, which is the behaviour of models that have not yet learned anything transferable.
Ours is the only row that is first in all nine.

### Each model's own metric, for comparability

`eval_obj.py` unchanged, its own FG-ARI, against ours after the grid reduction:

| model | dataset | its FG-ARI | ours | diff | its mBO |
|---|---|---|---|---|---|
| ItrSA | CLEVR | 0.5807 | 0.5794 | −0.0013 | 0.4157 |
| | Tetrominoes | 0.6324 | 0.6319 | −0.0005 | 0.6162 |
| | Multi-dSprites | 0.4826 | 0.4817 | −0.0009 | 0.6191 |
| AKOrN (conv) | CLEVR | 0.6053 | 0.6046 | −0.0007 | 0.3316 |
| | Tetrominoes | 0.7205 | 0.7202 | −0.0003 | 0.6428 |
| | Multi-dSprites | 0.4067 | 0.4055 | −0.0012 | 0.5533 |

Agreement is **≤0.0013** everywhere, the same as PV2_0150 found for attn. The grid reduction is
not where any of these numbers come from. mBO is a diagnostic only and is not comparable to the
published mBO (the truth we feed is the grid quantisation, not the true pixel mask).

### Published numbers, for the budget caveat

| model | Tetrominoes | dSprites | CLEVR |
|---|---|---|---|
| ItrSA, published 3-seed | 85.8±0.8 | 68.1±1.4 | 82.5±1.7 |
| ItrSA, **ours at 1/210 budget** | 63.2 | 48.3 | 58.1 |
| AKOrN conv, published 3-seed | 76.4±0.8 | 63.8±7.7 | 59.0±4.3 |
| AKOrN conv, **ours at 1/210 budget** | 72.1 | 40.7 | 60.5 |
| AKOrN attn, published 3-seed | 88.6±1.7 | 78.3±1.3 | 91.0±0.5 |
| AKOrN attn, **ours at 1/210 budget** | 56.8 | 43.5 | 60.4 |

Two things worth noticing. **AKOrN (conv) at 1/210 of its budget already matches its published
CLEVR number** (60.5 vs 59.0±4.3) and nearly its Tetrominoes one (72.1 vs 76.4±0.8) — so for
that model the budget is *not* the binding constraint, and its row here is close to the model its
authors report. **AKOrN (attn) and ItrSA are far from theirs**, which is where the 1/210 caveat
bites. The published figures are on the pixel protocol and the full official test splits, so
they are not directly comparable to our columns; the comparison above is between each model and
*itself*, which is what the caveat needs.

That sharpens what this experiment licenses. For AKOrN (conv) specifically, the margin
**+0.1378 / +0.0345 / +0.2260 on Tetrominoes** is not obviously a budget artefact, because that
model is near its published ceiling at this budget. For attn and ItrSA it still is.

## What this licenses, and what it does not

**Licenses:** "on an identical data budget, from the same 10,000 CLEVR images, our core beats the
best of four baselines in every one of nine metric cells, on two datasets it never trained on" —
with Tetrominoes Foreground IoU flagged as the one narrow cell, and with AKOrN (conv) being the
one baseline for which the budget excuse does not apply.

**Does not license** "we beat the field." Still missing:
- **3-seed means on our side.** Everything here is seed 0. The contract judges 3-seed means.
- **Protocol B** for attn, ItrSA and Slot Attention, which are all far from their published
  numbers at this budget.
- **Slot Attention on Tetrominoes and Multi-dSprites**, never run; its CLEVR row is the only one.
- **Any seed spread** for ItrSA and AKOrN (conv). One seed each.

## Deviations from the published recipes, and why

Both models use their published CLEVR config (`--ch=256 --psize=8`) with two changes:

1. **`--epochs` 300 → 10**, to match 100,000 exposures. This is the protocol A deviation and
   applies to every model in PV2_0149/0150/0151.
2. **`--batchsize` 256 → 128.** ItrSA at 256 raised
   `torch.OutOfMemoryError: Tried to allocate 1024.00 MiB` with 20.73 GiB already in use on a
   23.47 GiB TITAN RTX. The published recipe is `accelerate launch --num_processes=$NUM_GPUS`,
   i.e. multi-GPU, and `train_obj.py:224` divides the batch by the process count, so 256 was
   never meant for one card. `train_obj.py:241` also notes the `PairDataset` yields two views
   per sample, so the real forward is twice the batch. **The exposure budget is unaffected** —
   it is `epochs × 10,000` regardless of batch size — and AKOrN (conv) was given the same 128 so
   the two rows are mutually comparable.

**A correction to PV2_0150's record**, found while setting this up: it says
"Batch 250 × 40 batches × 10 epochs = 100,000 exposures exactly". The argument default is
**256**, not 250, with a partial final batch (40 batches over 10,000 images), and `PairDataset`
feeds two views per sample. **The exposure count is right** — 10 passes over 10,000 unique
images — but the arithmetic as written was not. PV2_0150's numbers are unaffected; the sentence
is fixed in this commit.

## Diagnostics

Predicted foreground fraction against the truth's, selected rule, fixed320:

| dataset | truth fg | ours | AKOrN attn | AKOrN conv | ItrSA |
|---|---|---|---|---|---|
| CLEVR | 0.2115 | 0.2001 | 0.6140 | 0.4355 | 0.7218 |
| Tetrominoes | 0.2368 | 0.2842 | 0.2800 | 0.2997 | 0.2787 |
| Multi-dSprites | 0.0943 | 0.0841 | 0.1798 | 0.1070 | 0.4988 |

**Only our core gets the foreground fraction right on CLEVR and Multi-dSprites** (0.2001 against
0.2115, and 0.0841 against 0.0943 — the only row that ever *under*-claims); every baseline
over-claims foreground by 2–3.4× on CLEVR, and that is the whole story of the Foreground IoU
column there. On Tetrominoes all four are close to the truth, which is exactly why that is the column
with our narrowest margin. ItrSA's 0.4988 on Multi-dSprites against a truth of 0.0943 is the
most extreme over-claim in the table and explains its 0.2489 Foreground IoU there while its
FG-ARI is the best of the baselines — a reminder of why the three metrics are never combined.

## Interpretation

The field at this budget sits at **0.40–0.72 FG-ARI** and has no stable ordering: AKOrN (conv)
wins Tetrominoes, ItrSA wins Multi-dSprites, AKOrN (attn) wins nothing outright. Our core is
first in all nine cells, and the only one whose predicted foreground tracks the truth on CLEVR.

The honest reading is still **data efficiency**, with one real exception: AKOrN (conv) is already
at its published CLEVR number here, so on Tetrominoes we are beating a model that is not starved.
That is the first cell in this whole line of work where the "give it its full budget" objection
does not apply, and it is also where our margin is thinnest on Foreground IoU (+0.0345). Both
facts belong in the same sentence.

## Next

1. **Seeds, on our side.** Nothing in PV2_0149–0151 changes the fact that our side is one seed.
2. **A second seed for AKOrN (conv)**, because it is the one baseline that is not budget-limited
   and it owns our narrowest margin. 7 min.
3. **Slot Attention on Tetrominoes and Multi-dSprites**, to complete its row.
4. Rotating Features and the complex autoencoders still need their own environments and loaders.

## Reproduce

```bash
cd /work/USERS/tkim1/baselines/_clone/akorn
P=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python

# ItrSA
$P -u train_obj.py --exp_name=s0_clevr_itrsa --data=clevr --seed=0 --batchsize=128 \
  --model=vit --L=1 --gta=False --T=8 --ch=256 --psize=8 --epochs=10
# AKOrN (conv)
$P -u train_obj.py --exp_name=s0_clevr_akorn_conv --data=clevr --seed=0 --batchsize=128 \
  --model=akorn --J=conv --L=1 --ksize=7 --ch=256 --psize=8 --epochs=10 --c_norm=none

# then, per model and per dataset: wrap ema_model.pth, eval_dump.py at --data_imsize=128,
# to_grid64.py, bg_any.py -- exactly as in PV2_0150/commands.sh.
```
