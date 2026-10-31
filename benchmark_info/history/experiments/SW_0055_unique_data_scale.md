# SW_0055_unique_data_scale

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `95dad1e937940f32dd9816339e395f1617774380f4599151b06a3200dbe68dfb`

````markdown
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
are refused. The scheduler below uses only GPUs 0 and 1, checks occupancy
before and after preflight, runs seeds 0 and 1 concurrently, then runs seed 2
on GPU 0 after both first seeds complete. Existing seed outputs and partial
state are refused rather than restarted. No server run or evaluation was
launched as part of this preparation.

To wait for an available GPU and run the SW0054 gate-invariant preflight plus
its 32-image short pilot before launching SW0055 on GPUs 0/1, use:

```bash
bash collaborative_test/SW_0055_unique_data_scale/wait_for_idle_and_launch.sh
```

The wrapper polls GPUs 0/1 every 30 seconds, takes a single-instance lock,
logs timestamps and worker PIDs, validates and reuses an existing successful
SW0054 pilot, and does not retry a pilot that leaves a failed evaluator
artifact. It refuses existing SW0055 seed outputs and partial scheduler state.
It writes the overall completion marker only after every seed has successful
short/long evaluation artifacts. A verified dead wrapper PID permits safe
stale-lock cleanup; a live or ambiguous PID blocks a second instance. The old
`launch_parallel.sh` entry point is deprecated and forwards to this scheduler.
Inspect its log, PIDs, phase markers, and outputs before any manual recovery.

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

The three-seed run completed. At the primary long-window threshold `.50`, the
mean changed from the SW0053 baseline `.562902/.381328/.356857` to
`.626733/.423340/.440119` for FG-ARI/foreground IoU/matched-object IoU. All
three means improved, by `.063831/.042012/.083262`. Seed FG-ARI values were
`.817116/.614845/.448238`, so this is a real mean improvement with substantial
initialization sensitivity. The validated summary is in
`results/three_seed_summary.{json,md}`.

````

</details>

## results/three_seed_summary.md

<details>
<summary>three_seed_summary.md</summary>

Source SHA256: `22f3a80c40fe172c12344320618345b46acea75b524107b7e52ec9bb7e272065`

````markdown
# SW0055 three-seed summary

Validation IDs 1320-1639 (320 images). Primary endpoint: long T1024/settle512 at fixed threshold .50; .35 is secondary. No posthoc threshold selection.

| Window | Threshold | Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | SW0053 baseline | Delta vs baseline |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| long | 0.50 | fg_ari | 0.817116 | 0.614845 | 0.448238 | 0.626733 | 0.150828 | 0.562902 | +0.063831 |
| long | 0.50 | foreground_iou | 0.730071 | 0.338969 | 0.200981 | 0.423340 | 0.224088 | 0.381328 | +0.042012 |
| long | 0.50 | matched_object_iou | 0.643890 | 0.454188 | 0.222280 | 0.440119 | 0.172409 | 0.356857 | +0.083262 |
| long | 0.35 | fg_ari | 0.810630 | 0.607738 | 0.450793 | 0.623054 | 0.147302 | 0.560105 | +0.062949 |
| long | 0.35 | foreground_iou | 0.730350 | 0.335704 | 0.201251 | 0.422435 | 0.224541 | 0.381852 | +0.040583 |
| long | 0.35 | matched_object_iou | 0.633323 | 0.441718 | 0.220506 | 0.431849 | 0.168676 | 0.349261 | +0.082588 |
| short | 0.50 | fg_ari | 0.765027 | 0.589180 | 0.435789 | 0.596665 | 0.134515 | n/a | n/a |
| short | 0.50 | foreground_iou | 0.769182 | 0.384011 | 0.198056 | 0.450416 | 0.237842 | n/a | n/a |
| short | 0.50 | matched_object_iou | 0.554573 | 0.408971 | 0.189466 | 0.384337 | 0.150069 | n/a | n/a |
| short | 0.35 | fg_ari | 0.740321 | 0.560759 | 0.432097 | 0.577726 | 0.126402 | n/a | n/a |
| short | 0.35 | foreground_iou | 0.771371 | 0.385611 | 0.204249 | 0.453743 | 0.236486 | n/a | n/a |
| short | 0.35 | matched_object_iou | 0.522353 | 0.375422 | 0.177882 | 0.358552 | 0.141135 | n/a | n/a |

Code SHA-256: `7a9f59b4be5ecce1024c5e127b43bd3bdc510797a619e2728be2e2bed176805f`  
Training gamma SHA-256: `7139439232d5717b66ffd1749572ab3faff9d0337d49c28b1602dce4d6c742aa`

````

</details>
