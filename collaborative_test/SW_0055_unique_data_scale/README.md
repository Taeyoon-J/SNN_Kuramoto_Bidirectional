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
training-code checksums. The current preflight marker is V2; it intentionally
invalidates V1 because the checksum now includes, in a fixed order, the S2Net,
graph, Kuramoto, dendritic, membrane, gating, hyperparameter, loss, and training
implementation files plus the run/preflight scripts.

For the three-seed report, the primary endpoint is the aligned validation long
window (T=1024, settle=512) at the fixed threshold 0.50. Report the mean across
seeds 0/1/2 for FG-ARI, foreground IoU, and matched-object IoU. Threshold 0.35
is a prespecified secondary endpoint. Do not select or describe a post-hoc
best threshold. Short-window results are diagnostic only.

Each seed output writes an exclusive `manifest.json` before training. It
records the gamma and ordered code SHA-256 values, exact command arguments,
recipe, git commit when available, training IDs, validation IDs, 25,000 image
exposures, and 1,570 optimizer updates. Existing seed output or manifest paths
are refused. `launch_parallel.sh` also rejects duplicate GPU IDs, checks all
three devices for active compute processes, and refuses existing output
directories before starting. No server run or evaluation was launched as part
of this preparation.

To wait for an available GPU and run the SW0054 gate-invariant preflight plus
its 32-image short pilot before launching SW0055 on GPUs 1/2/3, use:

```bash
bash collaborative_test/SW_0055_unique_data_scale/wait_for_idle_and_launch.sh
```

The wrapper polls GPUs 1/2/3 every 30 seconds, takes a single-instance lock,
logs timestamps, validates and reuses an existing successful SW0054 pilot, and
does not retry a pilot that leaves a failed evaluator artifact. It refuses
existing SW0055 seed outputs and writes a completion marker after successful
launch. Inspect its log and marker before any later restart.

Once seed0/1/2 short and long validation JSON files exist, create the fixed
threshold three-seed JSON/Markdown summary with:

```bash
python collaborative_test/SW_0055_unique_data_scale/summarize.py \
  --model-root trained_models \
  --output collaborative_test/SW_0055_unique_data_scale/results/three_seed_summary.json
```

The summarizer requires all six reports and seed manifests, checks the
checkpoint/code/gamma hashes, aligned IDs/counts and rollout windows, and
reports long .50 as primary and long .35 as secondary against the fixed SW0053
baselines. It refuses to overwrite the JSON/Markdown pair unless `--overwrite`
is provided. It performs no threshold reselection.
