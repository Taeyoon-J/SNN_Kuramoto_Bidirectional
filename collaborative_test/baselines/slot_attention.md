# Slot Attention reference

Saved prediction: `/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/patch_masks.pt`.

These 320 saved predictions correspond to CLEVR IDs 1000–1319. They were re-evaluated with `evaluation.py` against the HDF5 mask converted to 16×16 patch labels:

| Metric | Mean |
|---|---:|
| Patch FG-ARI | 0.8901145722 |
| Patch foreground IoU | 0.2122510283 |
| Patch matched object IoU | 0.2354868700 |

The stored predictions represent one available Slot Attention run. A comparable three-seed Slot Attention mean and predictions on an untouched test split are not yet available. Do not present a repeated evaluation of this same checkpoint as three independent seeds.
