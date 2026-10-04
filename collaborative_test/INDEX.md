# Experiment index

| id | status | hypothesis | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- | --- | --- |
| PV2_0001 | planned | 3-seed baseline of the current best configuration under contract v1 | | | |

Peer experiments from `patch_v2_sw` use the `SW_` prefix and are reviewed under
`peer_updates/`.

## Reference

| | fg_ari | fg_iou | obj_iou |
| --- | --- | --- | --- |
| Slot Attention, ckpt-500, single published checkpoint | 0.620 | 0.124 | 0.092 |

Not a 3-seed training mean. Trained on the original CLEVR render and evaluated
here on `clevr_with_masks`; see `baselines/slot_attention.md`.
