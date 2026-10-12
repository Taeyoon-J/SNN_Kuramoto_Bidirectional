# SW0142 completion, 2026-10-12

The registered seed1 masked-completion pilot failed its promotion gate and is closed without seed expansion or hyperparameter sweeps. Both 256-update/4096-exposure arms and both 320-image prediction stages completed; the CPU scorer froze references before reading GT. All 18 metric means and six paired bootstrap intervals were independently recomputed from the archived per-image arrays.

| Seed1 model/readout | FG-ARI | Foreground IoU | Matched object IoU |
|---|---:|---:|---:|
| Encoder/graph trained, P | .184229 | .277616 | .083670 |
| Encoder/graph/core trained, P | .291697 | .412178 | .151097 |
| Encoder/graph trained, QCC | .748880 | .607114 | .580991 |
| Encoder/graph/core trained, QCC | .749859 | .604018 | .578109 |
| Source seed1, QCC | .754135 | .606900 | .582201 |
| Comparable Slot seed1 | .841819 | .203417 | .222115 |

The joint P assignment improved over the frozen-core control (FG +.107468, paired 95% CI [.078425, .135617]) but remains substantially below source QCC (FG -.462438, CI [-.490126, -.434414]). It also fails the registered object-IoU margin. The unchanged QCC readout remains near source; this points toward a primary assignment/objective mismatch rather than disappearance of all core separation.

GT-free first64 TRAIN completion diagnostics also show weak instance-specific pressure: permuting P within coarse RGB bins reduces loss from .937748 to .907844 in the encoder/graph arm and from .974090 to .945565 in the joint arm. These descriptive interventions suggest color-compatible assignment can satisfy the objective. They do not establish the sole cause, prove gate necessity, or justify another blind reconstruction sweep.

This is a one-seed native32 continuation pilot, not a three-seed or fresh native32 70,000-image result. SW0137 remains the incumbent; its FG advantage over comparable Slot is only .000267 and is not a robust or substantial victory. SW0143 independently tests selective graph boundaries with unchanged spike QCC and all three seeds; its queue remains active. No checkpoint, dataset, mask array or key is included in this record.

Post-hoc descriptive mask statistics on the same frozen 320 images show under-separation: joint P predicts 3.328 foreground labels per image versus 6.313 in GT and 5.784 in joint QCC. The largest P object contains 61.87% of its predicted foreground, versus 40.17% in GT. P foreground occupies 36.43% of patches versus 21.84% in GT. These observations locate merging/background confusion in the primary readout; they do not justify changing the evaluation contract or selecting a new threshold from validation masks. See mask_diagnostics.json.
