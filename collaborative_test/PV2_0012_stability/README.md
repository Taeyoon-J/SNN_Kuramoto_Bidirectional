# PV2_0012 — the setting fails on half of a wider seed draw

**completed. This is the most important limitation on record.** Parent `PV2_0008`.
Validation 300, unchanged `PV2_0008` configuration, seeds 3/4/5 added.

| seed | final loss | fg_ari | foreground_iou | predicted_fg |
| --- | --- | --- | --- | --- |
| 0 | 15.31 | 0.481 | 0.603 | 0.124 |
| 1 | 16.67 | 0.643 | **0.732** | 0.127 |
| 2 | 15.47 | 0.698 | 0.692 | 0.144 |
| 3 | 42.23 | 0.508 | **0.163** | **0.582** |
| 4 | 15.40 | 0.443 | 0.430 | 0.212 |
| 5 | **85.93** | **0.003** | **0.000** | 0.000 |

Seed 5 did not train at all -- its loss moved 87.42 to 86.08 and it emits nothing.
Seed 3 calls 58% of patches foreground. Two of six are unusable and a third is
poor.

**`PV2_0008`'s reported 0.6755 +- 0.0660 is a lucky draw of seeds.** The protocol
fixes seeds 0, 1, 2, so that report stands as written, but the setting is not
reliable and any future report of it must carry this.

## It also retracts the anti-correlation PV2_0009 was built on

Seeds 0-2 span 15.3 to 16.7, a narrow band where the loss ordering was noise.
Over six points high loss means failure, which is the ordinary direction.
Instability was the right reading, was raised, was then wrongly retracted in
favour of "the objective is misaligned", and is now supported strongly.

Neither bimodality 3 (`PV2_0010`) nor bimodality 6 (`PV2_0013`) rescues these
seeds: seed 3 stays at foreground IoU 0.166-0.171 and seed 5 stays at 0.000.
