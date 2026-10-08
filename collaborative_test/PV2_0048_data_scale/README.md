# PV2_0048 — more training images, at matched compute, made it worse

**exhausted. Negative result.** Parent `PV2_0039`. Validation 300, contract v1,
end-to-end encoder with the reconstruction term — the only feature route that
trains at all.

## Why it was tried

Seven routes to better features were closed (DINOv2, label-free clustering,
self-bootstrapping, encoder capacity, slot-term defaults, end-to-end alone, frozen
core), and **all seven moved the features inside 6,000 training images**. The
tfrecord holds at least 51,536 and the dataset is nominally 100,000, so we had been
using 6-12% of it; the matched Slot baseline was trained on the same 6,000.
Object-centric learning is data-hungry, so this changes the conditions of feature
learning rather than rearranging them.

The split was never touched: extra images come from IDs 8000+, while validation is
6000-6999 and test 7000-7999, so every existing number stays comparable.

## Design, and the confound it introduced

Updates were held constant at ~240k presentations so that data diversity would be
the only variable: 6k x 40 epochs, 20k x 12, 40k x 6. Keeping 40 epochs instead
would have multiplied compute by the same factor as the data and confounded "more
images" with "more training".

**But holding updates constant necessarily cut exposures per image from 40 to 6.**
One confound was removed and another introduced. This result therefore says only
that more data does not help *at fixed compute*.

## Result

| training data | epochs | fg_ari | fg_iou | obj_iou | groups | final loss |
| --- | --- | --- | --- | --- | --- | --- |
| 6,000 (bar) | 40 | **0.6779** | — | — | — | — |
| 20,000 | 12 | 0.5983 | 0.6453 | 0.3455 | 4.00 | 15.55 |
| 40,000 | 6 | 0.4790 | 0.5302 | 0.3487 | 5.53 | 16.01 |

Monotonically worse, and far below the frozen-feature 0.7250. The loss went the
wrong way with more data — 16.01 at 6 epochs against 15.55 at 12 — and the 40k run
was still descending when it stopped (18.52 -> 16.01). Rising groups per image
(5.53 against 4.00) is over-segmentation, also a sign of under-training.

## How it was closed properly

The expensive follow-up — 40,000 images at the full 40 epochs, 1.6M presentations,
about 27 hours — was queued and then **cancelled**. `PV2_0051` answered the same
question from the encoders this test had already written to disk, at zero GPU cost:
the scale-invariant separation ratio improved by only 7% relative while fg_ari
fell. Data scale is closed on the mechanism as well as the score.
