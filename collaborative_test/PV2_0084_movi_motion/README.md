# PV2_0084 — MOVi-A prepared, and what the motion signal actually measures

**stage report, honest and incomplete.** The dataset and the diagnostics are done; the
training comparison is not, and the reason is recorded so the next attempt starts from the
right design. Parent `PROPOSAL_0.9.md`. **New contract — nothing here is comparable with
contract v1, and the test result 0.7603 stands untouched.**

## What is in place

MOVi-A 128x128 from `gs://kubric-public/tfds/movi_a`, 60 train shards and all 16
validation shards, 9 GB on a volume with 19 TB free.

| | MOVi-A pilot | contract v1 |
| --- | --- | --- |
| train scenes | 1,000 | 1,000 images |
| validation | 250 | 300 |
| foreground fraction | 0.1027 / 0.1061 | 0.118 |
| objects per scene | 5.97 / 6.11 | 6.32 |
| patch grid | 16x16, 8x8 px on 128x128 | 16x16, 15x20 px on 240x320 |
| labelling rule | **the same** — majority per patch, ties to smallest id | |

The statistics are close enough that the pipeline's operating point should transfer, which
is why MOVi-A was chosen over CATER or CLEVRER.

Motion by frame gap, measured as the share of pixels whose instance label changes:
**8.8%** at gap 2, **16.8%** at 6, **18.8%** at 12, **18.1%** at 23. It saturates around
12; gap 6 was used, near saturation with tighter correspondence.

## What the motion signal is worth, measured before training

The descriptor is label-free and flow-free: the per-patch RGB change between frame t and
t+6. Patches of one rigid object change alike; two objects moving differently do not.

| descriptor | cos_within | cos_between | ratio |
| --- | --- | --- | --- |
| appearance, raw RGB patch means | 0.0021 | 0.0544 | 0.0380 |
| motion, raw | 0.1040 | **0.7137** | 0.1457 |
| **appearance, standardised** | 0.0419 | **0.5012** | 0.0837 |
| motion, standardised | 0.1056 | **0.7170** | 0.1473 |
| appearance + 0.25 x motion | 0.0440 | 0.5140 | 0.0856 |
| contract v1's gamma, for scale | 0.0142 | 0.2483 | 0.0571 |
| oracle codes | 0.0000 | 1.0000 | 0.0000 |

**Two corrections to how this looked at first.**

A magnitude gate barely helps: 20.2% of patches hardly change, and their direction is
noise, but thresholding at 0.08 only moves within-object spread 0.1318 -> 0.1131 while
keeping 94% of the foreground.

More important, **standardisation changes the comparison completely.** Against raw RGB
means motion's margin looks 13x larger; against *standardised* appearance it is **1.4x**
(0.717 against 0.501). And MOVi-A's standardised appearance already has a margin of
**0.5012, twice contract v1's 0.2483** — so this render separates by colour better than
CLEVR does, and **the dataset change confounds the motion test.** Any claim about motion
has to be made within MOVi-A, not across datasets.

On the ratio, adding motion is monotonically worse (0.0837 -> 0.1368 as its weight grows).
That is weak evidence, since the ratio has failed to predict the score four times
(`PV2_0058`, `PV2_0062`, `PV2_0064`, `PV2_0078`), but it is not encouraging either.

## Where it stopped, and why that is informative

The controlled comparison — same pipeline, same scenes, appearance against
appearance+motion — needs motion inside the model. I took the shortcut of concatenating
it as extra channels, and it hit the wall `PV2_0054` documented:

| arm | first-epoch loss | `spike_plv_collapse` |
| --- | --- | --- |
| appearance, 3 channels | **28.25** | 2.11 |
| appearance + motion, 6 channels | 89.93 | **9.210340** |
| appearance + motion, motion to the **graph only** | 85.15 | 8.30 |

9.210340 is exactly log(10^4), the clamp floor: **no spikes at all**, the dead-membrane
regime. Routing motion to the graph alone — the fix built for `PV2_0055`, and where
`PV2_0066` says features act — does not rescue it here, because the appearance block is
only 3 channels wide so adding 3 more doubles it, a far larger relative change than the
8 -> 10 that `PV2_0055` survived.

**And concatenation was never the proposal.** `PROPOSAL_0.9.md` specifies motion as the
external target of a consistency term, in the `--recon-target-path` pattern, not as extra
gamma channels. I drifted from my own design and the smoke test caught it before a
training run was wasted.

## The precise next step

Build a per-patch **motion affinity** — cosine similarity between patches' change vectors,
masked where the change is below ~0.08 — and add a term pulling the model's PLV toward it,
exactly as `slot_reconstruction_loss` pulls toward pooled RGB. The plumbing exists:
`--recon-target-path` already threads an external per-patch target through training with
frozen features, and `--view-consistency-weight` already computes a PLV agreement term.

The arm to beat is **appearance-only on MOVi-A**, which trains healthily at first-epoch
loss 28.25. Judge it with the gates this cycle earned: a regression row first, gains only
below predicted foreground 0.20, do not read the loss as progress, three seeds before any
claim.

## Files

`/work/USERS/tkim1/movi_a/`: `movi_train.npz`, `movi_val.npz` (frames, patch labels),
`gamma_app_{train,val}.pt`, `gamma_appmot_{train,val}.pt`, `targets_movi_val.pt`,
`manifest_movi.json`. Standardisation statistics come from train rows only.
