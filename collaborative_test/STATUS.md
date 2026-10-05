# STATUS

Updated 2026-10-04 18:05. Branch `patch_v2`. Contract v1, split v1.

## Where things stand

Active goal: `patch_fg_ari` at or above 0.75, the spike readout beating the phase
readout, no model bugs, and the peer branch's positive results applied.

Best on validation (300 images, 3 seeds, readout window 1024/512, spatial kernel
sigma 1.0, from classifier masks on the model's own spikes):

| metric | value |
| --- | --- |
| `patch_fg_ari` | **0.7043** |
| `patch_foreground_iou` | 0.6740 |
| `patch_matched_object_iou` | **0.4914** |

Last confirmed **test** result is still `PV2_0008`: foreground IoU 0.6755 +-
0.0660, fg_ari 0.6075, matched-object IoU 0.4298. Everything since is validation
only; test has been read once and is not touched again until a setting is settled.

| goal condition | state |
| --- | --- |
| fg_ari 0.75 | 0.7043, short by 0.049 |
| spike beats phase | gap 0.0044 at sigma 1.5; the matched sigma 1.0 comparison is running |
| no model bugs | three fixed, one open (seeds 3/4 over-predict foreground) |
| apply peer's positive results | spatial kernel applied, +0.0156; spectral-k readout outstanding |

## What the search has settled

Axes exhausted or refuted, so they need not be re-run:

| axis | outcome |
| --- | --- |
| coupling graph | spent; a ground-truth graph is worth only +0.06 |
| classifier knobs (settle, min_group, background) | at their best already |
| synchrony threshold | flat across 0.05-0.20 |
| readout window | saturates near 1024; worth +0.18 fg_ari from 64 |
| training window | refuted -- matching it to the readout collapses the result |
| `--spike-plv-weight` | 5 is the peak, measured at 3 seeds |
| `--plv-bimodality-weight` | 6 is the peak, at two different readout windows |
| phase-to-spike imitation | five attempts failed: MSE, contrast, frozen path, per-region capacity, tied init |

## The central finding

`PV2_0019` explains all five imitation failures at once. The gap between the
readouts is a **sampling limit**: with window length alone, nothing retrained, the
spike readout goes 0.512 -> 0.688 while the phase readout saturates at 0.711 by
window 128. Spikes sample a continuous phase relationship as sparse binary events
and the per-component product multiplies that noise four times.

Binding-by-synchrony does survive the oscillator dynamics and reach the spikes.
Reading it out of spikes costs 8-16x the window the phases need.

## Bugs

Fixed:

1. **Membrane threshold absolute, membrane scale free.** Fix applies at
   **inference only** -- training with it collapses all three seeds (fg_ari
   0.5445 / 0.0442 / 0.0325), because a threshold tracking the population mean is
   a moving target that puts half the units above it at every step. `mem* ~ R_m * h_wave`
   while `v_th` is 0.06, so across seeds the membrane sat at means +1.14, +2.64
   (100% above, threshold inert), +0.26 and -0.65 with maximum -0.046 -- firing
   nothing. Re-thresholding the dead checkpoint at the population mean, no weight
   changed, took it from foreground IoU 0.0000 to 0.6353. Fixed with
   `--membrane-threshold-mode population`, one global scalar so between-unit
   ordering is untouched.
2. **Reset used a different threshold from the comparison** (introduced while
   fixing 1). Trained that way: 0.6887 -> 0.3029. Now one `_threshold()`;
   regression confirms absolute mode unchanged to four decimals.
3. **`--help` crashed** on `"83% of scenes"` parsing as an octal conversion. Had
   been broken for a while and an empty `--help | grep` was wrongly read as line
   wrapping.

Open:

4. **Seeds 3 and 4 call 0.50-0.58 of patches foreground** with membranes in a
   healthy range. The threshold fix does not touch them (foreground IoU 0.169,
   0.208). Separate cause, not yet diagnosed.

## Peer branch

Reviewed `origin/patch_v2_sw` through `7e3ce3d` (SW_0001-0038).

Applied: **SW_0028's spatial kernel**, `exp(-d^2/2 sigma^2)` into the affinity.
Worth +0.0156 fg_ari and +0.0204 matched-object IoU here, best at sigma 1.0.

Applied and settled: **SW_0024/0027's spectral-k readout**. It does not beat
connected components on fg_ari (about 0.61 against 0.7059) so it is not adopted,
but it raises matched-object IoU 0.4807 to 0.5240 and, more importantly, it made
their control runnable.

**That control now answers the open question in this branch's favour.** Same
readout, same k, only the affinity's source changing: the spatial kernel alone
reaches fg_ari 0.3902 and spike synchrony reaches 0.6068, so the spikes add
**+0.217** over pure geometry. The peer's finding -- that spatial-only nearly
matched membrane-times-spatial on their data -- does not reproduce here.

SW0047's permutation control agrees: keeping the kernel's values and scrambling
which patch each row belongs to takes seed 0 from 0.6639 to **0.0033** and seed 1
to **-0.0050**, below having no kernel at all. The kernel acts entirely through
grid geometry and must be correctly registered to help.

Independent corroboration: their SW_0034 found the binary spike threshold
"constant 1 after settle" and SW_0036 repaired it by changing vth -- the same
saturation found here as bug 1, reached separately. Their REVIEW_0002 also
reproduced the `PV2_0007` failure: full affinity MSE distillation fails by dense
fusion.

## Resume point

Detached on frontier, surviving the loss of a driving session:

| what | where |
| --- | --- |
| matched-kernel phase comparison, threshold re-selected per sigma | `$R/decide2_results.txt` |

The peer (`kevinswk`) is verifying BIM6 on this machine under
`/Data0/kevinswk/peer_verify_20261004`. Leave it alone: GPU 3 is theirs, and a
wait loop matching `train_s2net_core.py` by name will block on their process --
match by save path instead.

## Standing cautions

- Single-seed validation numbers are candidates, never findings. Five have had to
  be corrected this way, the latest being "fg_ari prefers a lower spike weight",
  which reversed at three seeds.
- Compare only matched conditions, and re-select the threshold whenever the
  measure changes. A new spike score was once held against a phase score from
  older checkpoints and read as a win; later, sweeping the spatial kernel with the
  phase threshold left at 0.75 clipped every phase row to zero groups, because the
  kernel rescales the affinity. The same error once produced the false conclusion
  that per-component spikes did not help.
- Gains that fix the same error substitute rather than add. Hybrid background with
  the spike loss, and now the membrane fix with the spatial kernel, both came out
  below the better single change.
- Keep stderr in sweeps and assert the mechanism is live. A suppressed argparse
  error once returned an empty table that looked like a result; smoke tests have
  since caught a missing flag, a missing import, a wrong parts type and a frozen
  path with no gradient.
