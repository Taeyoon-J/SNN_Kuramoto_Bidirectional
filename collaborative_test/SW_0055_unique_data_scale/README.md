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
2,500x25 arm is launched only after the matched-exposure result. With batch
size 16, the selected 1,000x25 run has 1,575 optimizer updates and this
2,500x10 control has 1,570; every scene is still seen for exactly the stated
number of epochs, so image exposure is exact and update count differs by five.

```bash
python collaborative_test/SW_0055_unique_data_scale/build_gamma.py \
  --output data/SW_0055_unique_data_scale/gamma_train_2500.pt \
  --manifest data/SW_0055_unique_data_scale/manifest.json
```

The real-asset preflight regenerated all original 1,000 rows with maximum
absolute difference `0.0`. The full expanded asset then passed shape
`[2500,8,256]`, finite-value, held-out-exclusion, and provenance checks. Its
SHA-256 is `7139439232d5717b66ffd1749572ab3faff9d0337d49c28b1602dce4d6c742aa`;
the versioned manifest is under `results/`. The matched-exposure runner's
real-gamma one-update preflight passes and is bound to the gamma and
training-code checksums. `launch_parallel.sh` refuses occupied GPUs and
existing seed output directories before starting seeds 0/1/2, followed by
fixed short and long aligned validation. Training is ready but remains
unlaunched while the requested GPUs are occupied by another user.
