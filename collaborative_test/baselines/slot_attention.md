# Slot Attention reference

Saved prediction: `/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/patch_masks.pt`.

These 320 saved predictions correspond to CLEVR IDs 1000-1319. They were re-evaluated with `evaluation.py` against the HDF5 mask converted to 16x16 patch labels:

| Metric | Mean |
|---|---:|
| Patch FG-ARI | 0.8901145722 |
| Patch foreground IoU | 0.2122510283 |
| Patch matched object IoU | 0.2354868700 |

The same official checkpoint was also evaluated on the aligned SW0042
validation IDs 1320-1639 (all 320 valid):

| Metric | Mean |
|---|---:|
| Patch FG-ARI | 0.8946414575 |
| Patch foreground IoU | 0.2235111789 |
| Patch matched object IoU | 0.2459682554 |

The close results across the two slices suggest the evaluation path and split
shift are stable. Against this single-checkpoint reference, the current SW0042
seed0 gap is concentrated in FG-ARI; both IoUs already exceed the reference.
This comparison does not provide a three-seed Slot Attention mean and does not
establish goal success.

The stored predictions represent one available Slot Attention run. A comparable three-seed Slot Attention mean and predictions on an untouched test split are not yet available. Do not present a repeated evaluation of this same checkpoint as three independent seeds.

Server provenance inspection on 2026-10-04 shows this is Google's official
pretrained object-discovery checkpoint `gs://gresearch/slot-attention/object-discovery/ckpt-500`
(local model SHA-256 `96c2b12d8b28c22fd2605eccf9027bda304f38bb9332f8c1620898f471c67ec4`),
not a model trained on our IDs 0-999. Inference used seed 0, 11 slots and three
Slot Attention iterations. The published checkpoint was trained with seven
slots on CLEVR scenes containing at most six objects, while this evaluation
uses CLEVR10 and 11 inference slots. Images were scaled to [-1,1] without an
extra crop; the background slot was selected by most hard-assigned perimeter
patches. Re-running this deterministic checkpoint with different random seeds
would not supply a comparable three-seed training mean. A formal final goal
claim therefore still requires a matched three-seed Slot baseline or an
explicitly documented single-checkpoint limitation.
