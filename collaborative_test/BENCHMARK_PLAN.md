# Benchmark plan for a paper table

Written 2026-10-19. Sources are web-retrieved published numbers, marked below; nothing here
was reproduced on our hardware yet. Numbers we have not re-run are **other people's numbers
under other people's protocols** and may not be placed in a column with ours until the
protocols are made to match.

## 0. The finding that decides the whole table

**There are two incompatible conventions both called "FG-ARI", and we are using the rarer one.**

Slot Attention reports ARI-FG **98.8** on CLEVR6; SynCx reports ARI-FG **0.59** on CLEVR.
Both are "foreground ARI". They differ because of the data variant, the model class and —
decisively — because the field scores **per pixel**, while `evaluation_contract.md` scores
**per patch** and forbids expanding a predicted mask to pixels.

Our 0.7603 is therefore **not comparable to any published number in this field.** Putting it
in a table beside 98.8 would be rejected on sight, and rightly: it is the same class of error
as `PV2_0090`, where our own Slot baseline was depressed 0.1446 by a preprocessing mismatch
we had not noticed.

**This is a decision for the user, not an implementation detail.** Either:

- **(A) Adopt the field's protocol for the paper table.** Score at pixel level with ARI-FG and
  MBO on the standard datasets. Our patch contract stays as an internal diagnostic. This is
  the only route to a table that a reviewer will accept at face value. It requires a contract
  version bump and re-evaluating both models, per `evaluation_contract.md`.
- **(B) Keep the patch contract and re-run every baseline ourselves under it.** Defensible and
  much more work: every row in the table has to be a model we ran, not a number we cited.
- **(C) Report both**, with the patch table as the main result and a pixel-protocol table for
  comparability. Most honest, largest effort.

### DECIDED, 2026-10-19: (A) the field's pixel protocol, at 32x32 or finer

Measured first, on 100 validation images, by quantising the **pixel** ground truth to our
grid and expanding it straight back — a perfect patch segmenter, zero model error:

| pixel protocol | FG-ARI | fg-IoU | obj-IoU |
| --- | ---: | ---: | ---: |
| ceiling, perfect patch model 16x16 | 0.6973 | 0.6965 | 0.5814 |
| **ours 16x16** | 0.6125 | 0.6179 | 0.3803 |
| ceiling, perfect patch model 32x32 | **0.8355** | 0.8259 | 0.7627 |
| **ours 32x32** | **0.7142** | 0.7327 | 0.6190 |
| ours 16x16, **our patch contract**, same images | 0.7460 | 0.7194 | 0.4893 |

Two things settle it.

**The grid costs more than the model does.** The same predictions score 0.7460 on our
contract and 0.6125 on the field's. Of that 0.1335, only 0.0848 is model error — the rest
is quantisation. Going to 32x32 lifts the ceiling by 0.1382 and the actual score by 0.1017.

**At 32x32 the pixel number is competitive where it matters.** Against the synchrony-binding
family on CLEVR — CAE 0.04, CAE++ 0.27, CtCAE 0.54, SynCx 0.59, Rotating Features 0.65 —
our 0.7142 sits at the top. It is Slot Attention's 98.8 on CLEVR**6** that is out of reach,
and that is a different task: at most six objects, decoder reconstruction, their variant.

So the main table is **pixel protocol, 32x32 or finer, against the synchrony family**, and
the patch table goes in the appendix with the ceiling stated as a limitation. Choosing (B)
would mean citing nothing and re-running CAE, CtCAE, RF, SynCx and AKOrN ourselves.

**This does not change `evaluation_contract.md`.** The contract and the standing goal — three
patch metrics, 0.9 FG-ARI — are unchanged; this is a second, clearly labelled reporting path
for the paper. Mixing the two in one table is the error `PV2_0090` already cost us once.

**Tier 1 is technically unblocked.** Our pure-python reader (no TensorFlow) parses all three
suites, and the shapes are as published: Tetrominoes 35x35x3 with 4 entities, Multi-dSprites
64x64x3 with 5, CLEVR 240x320x3 with 11. Both are downloaded to `/work/USERS/tkim1/mod/`.

Nothing below is worth doing until this is chosen.

## 1. Who we are actually competing with

The paper's related work is **not** the slot family. It is the synchrony-binding family, and
it already exists:

| line | model | idea |
| --- | --- | --- |
| 2021 | Complex AutoEncoder (CAE) | complex-valued activations; phase encodes binding |
| 2023 | CAE++ / CtCAE | gating and a contrastive loss on top of CAE |
| 2023 | Rotating Features (Löwe et al., NeurIPS oral) | complex features generalised to higher dimensions, cosine binding |
| 2024 | SynCx | recurrent **complex-weighted** autoencoder, no gating or contrastive loss |
| 2025 | **AKOrN** (Miyato et al., ICLR) | **Artificial Kuramoto Oscillatory Neurons** |

**AKOrN is literally Kuramoto oscillators for object binding, published at ICLR 2025.** It is
the closest prior work that exists and the paper cannot avoid it. Any claim of novelty has to
be stated against AKOrN specifically, not against Slot Attention.

What we have that AKOrN does not, on the face of it: **spikes**. AKOrN clusters oscillator
phases; our contract requires the score to come from a classifier on spikes or membrane, with
phase clustering as a diagnostic only. That is a real architectural difference and it is the
honest place to stand.

## 2. Published numbers, by protocol

### 2a. Pixel ARI-FG on the small multi-object suite — the synchrony family's table

(From SynCx, arXiv 2405.17283. Tetrominoes 35x35, dSprites 64x64, CLEVR filtered 96x96.)

| model | Tetrominoes | dSprites | CLEVR |
| --- | ---: | ---: | ---: |
| CAE | 0.00 ± 0.00 | 0.05 ± 0.02 | 0.04 ± 0.03 |
| CAE++ | 0.78 ± 0.07 | 0.51 ± 0.08 | 0.27 ± 0.13 |
| CtCAE | 0.84 ± 0.09 | 0.56 ± 0.11 | 0.54 ± 0.02 |
| Rotating Features | 0.42 ± 0.09 | 0.84 ± 0.03 | 0.65 ± 0.01 |
| SynCx | 0.89 ± 0.01 | 0.82 ± 0.01 | 0.59 ± 0.03 |

### 2b. The slot family on the same suite (different data variants)

(From Slot Attention, arXiv 2006.15055.) Tetrominoes **99.5 ± 0.2**, Multi-dSprites
**91.3 ± 0.3**, CLEVR6 **98.8 ± 0.3**; IODINE and MONet below it.

**Read 2a against 2b carefully.** The synchrony family scores 0.4-0.9 where the slot family
scores 0.91-0.995. They are not the same task: the slot numbers use easier variants
(CLEVR**6**, i.e. at most six objects) and decoder-based reconstruction. **A table mixing 2a
and 2b rows is wrong**, and several papers in this area are careless about it.

### 2c. Texture-rich — where a binding model should win

CLEVRTex, plus its **CAMO** (camouflaged) and **OOD** generalisation sets. AKOrN reports
**FG-ARI 88.5 ± 0.9, MBO 59.7 ± 0.9** on full CLEVRTex at 128x128. This is the dataset
designed to break slot models, so it is the strongest place for a binding argument.

### 2d. Real images

PascalVOC and COCO2017, metric **MBO_i / MBO_c**:

| model | VOC MBO_i | VOC MBO_c | COCO MBO_i | COCO MBO_c |
| --- | ---: | ---: | ---: | ---: |
| Slot Attention | 22.2 | 23.7 | 24.6 | 24.9 |
| SLATE | 35.9 | 41.5 | 29.1 | 33.6 |
| DINOSAUR | 44.0 | 51.2 | 31.6 | 39.7 |
| Slot-diffusion | 50.4 | 55.3 | 31.0 | 35.0 |
| SPOT | 48.3 | 55.6 | 35.0 | 44.7 |
| AKOrN | **52.0** | **60.3** | 31.3 | 40.3 |

AKOrN's own framing: the first non-slot model competitive on real-image object discovery.

## 3. MBO is the metric our result is strong in

**Mean Best Overlap** is, up to the matching rule, our `patch_matched_object_iou`: per true
object, the best-overlapping predicted mask's IoU. The modern papers report it precisely
because ARI-FG is insensitive to over-prediction — the same hole `PV2_0090` found when Slot
Attention called 85% of patches foreground and still scored 0.8102 FG-ARI.

Our obj-IoU is 0.6598 where our aligned Slot baseline is 0.1025. **MBO is the column to build
the paper on**, not FG-ARI. It also happens to be the column we cannot buy by over-predicting,
which is why it survives scrutiny.

## 4. Tiered plan

**Tier 1 — comparability, required.** Tetrominoes, Multi-dSprites, CLEVR6 from
`multi_object_datasets` (the same source as our `clevr_with_masks`). Every baseline has
published numbers; small images; cheap. The point of this tier is not to win, it is to show
the model works on the standard suite and to establish that our protocol matches theirs.

**Tier 2 — the differentiator.** CLEVRTex + CAMO + OOD, 128x128, FG-ARI **and** MBO. This is
where texture breaks slot models and where a binding argument has room. AKOrN's 88.5 is the
number to beat or to lose to honestly.

**Tier 3 — reach, probably out of scope.** VOC / COCO with MBO_i/MBO_c. Needs a
DINO-class encoder and far more compute; only worth it if Tier 2 lands.

## 5. Rules for the table, from our own scars

1. **Never mix a cited number with one we ran** unless the protocol is identical and stated.
   Every row says which it is.
2. **Re-run at least one baseline ourselves per dataset** and check it reproduces the
   published value. If our Slot Attention does not reproduce published Slot Attention, the
   harness is wrong — that is exactly what `PV2_0090` caught, and it cost us a claimed win.
3. **Three independently trained seeds, per metric, with std.** A published checkpoint
   evaluated three times is a single checkpoint and is labelled as one — the rule already in
   `baselines/slot_attention.md`, and the peer applies it too.
4. **Report FG-ARI and MBO together, always.** FG-ARI alone has given the wrong verdict six
   times in this project's own logs.
5. **State the foreground fraction** beside every row. It is what exposes a model that buys
   FG-ARI by calling everything foreground.

## 6. What we cannot claim

On easy CLEVR the slot family is at 98.8 ARI-FG. We will not beat that, and our 0.7603 is not
even measured the same way. The paper's claim has to be one of:

- **MBO / instance quality** at comparable FG-ARI — our actual measured strength;
- **the trained-parameter count** — 1,325 of 69,042 match full training (`PV2_0081`), which
  is a genuinely unusual number in this field and is already verified;
- **spikes as the readout** rather than phase clustering, which is the architectural
  difference from AKOrN.

Picking the axis before running Tier 1 is the difference between a table that argues something
and a table that merely reports.

## Sources

AKOrN: arXiv 2410.13821, ICLR 2025, project page takerum.github.io/akorn_project_page,
code github.com/autonomousvision/akorn. SynCx: arXiv 2405.17283. Rotating Features:
NeurIPS 2023, arXiv 2306.00600. Slot Attention: arXiv 2006.15055. CLEVRTEX: arXiv 2111.10265.
