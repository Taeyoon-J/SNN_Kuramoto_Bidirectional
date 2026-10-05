# SW0055 - unique training-data scale

This experiment separates additional unique scenes from additional passes over
the same scenes. The fixed validation/reference range 1000-1639 is excluded.
The expanded 2,500-scene training set is IDs 0-999 plus 1640-3139, in that
order. It reuses the exact frozen input encoder and scalar standardization
statistics that produced the original 1,000-row training gamma.

Before any training, `build_gamma.py` must reproduce the original first 1,000
gamma rows within the recorded tolerance, produce finite `[2500,8,256]`
output, and write an ID/checksum manifest. The controlled comparison is:

| Condition | Unique scenes | Epochs | Image exposures |
|---|---:|---:|---:|
| current selected recipe | 1,000 | 25 | 25,000 |
| unique-data control | 2,500 | 10 | 25,000 |
| data + compute | 2,500 | 25 | 62,500 |

The 2,500x10 arm isolates unique-scene diversity at matched exposure. The
2,500x25 arm is launched only after the matched-exposure result. This folder
initially builds and validates the gamma asset; it does not launch training.

```bash
python collaborative_test/SW_0055_unique_data_scale/build_gamma.py \
  --output data/SW_0055_unique_data_scale/gamma_train_2500.pt \
  --manifest data/SW_0055_unique_data_scale/manifest.json
```

The real-asset preflight regenerated all original 1,000 rows with maximum
absolute difference `0.0`. The full expanded asset then passed shape
`[2500,8,256]`, finite-value, held-out-exclusion, and provenance checks. Its
SHA-256 is `7139439232d5717b66ffd1749572ab3faff9d0337d49c28b1602dce4d6c742aa`;
the versioned manifest is under `results/`. Training remains unlaunched until
the matched-exposure 2,500x10 runner receives its own one-update preflight.
