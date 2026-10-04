# Standing instructions

Collaborative goal-mode. Branch `patch_v2`; peer branch `patch_v2_sw`. Push only
to `patch_v2`, never modify or force-push the peer branch.

## Goal

Beat Slot Attention on all three patch metrics, as 3-seed means, strictly:

- `patch_fg_ari`
- `patch_foreground_iou`
- `patch_matched_object_iou`

Seeds 0, 1, 2 on one fixed test split. Never combine the three into one score;
report per-seed values and standard deviation alongside the mean.

**The final score comes from masks produced by a classifier on spikes or
membrane.** theta/PLV clustering is a diagnostic and is never the success score.
A PLV loss is allowed; a phase readout as the final mask is not.

If Slot Attention has no independent 3-seed checkpoints, say the reference is a
single checkpoint. Never report one checkpoint evaluated three times as a 3-seed
training mean.

## Signal flow to verify

```
image/features -> gamma -> phase mapping -> image-conditioned graph
-> Kuramoto theta -> sinusoidal gating -> dendritic state
-> membrane -> spikes -> classifier -> patch object masks -> evaluation
```

Check that theta forms real synchrony structure (PLV and its distribution), that
it reaches gating, dendrite, membrane and spikes (activations and gradients), and
that the score is measured on the classifier's masks.

## Scope of code changes

Keep the core computation and overall structure as far as possible.
Hyperparameters, losses and classifiers are free to change, and a loss that
improves the spikes directly comes first. A core change is analysed before it is
made -- what in the equations or the state updates changes -- and validated with a
small ablation. One variable, or one clearly meaningful group, per experiment.

## Data and evaluation contract

Same train/validation/test image IDs and order for every experiment and for both
models. The split manifest is fixed; never draw new test images per experiment.
Verify that the gamma tensor and the ground-truth masks agree on image ID and
order.

Hyperparameters, losses, classifiers and thresholds are selected on validation. A
split repeatedly used for tuning is not called an independent test split.

Convert ground truth to the same patch grid and score per patch. Never expand a
predicted mask to pixels to score it. Slot Attention masks go through the same
patch conversion and the same evaluation function. Anything that used the true
object count or the true masks to pick a cluster count or a threshold is an
oracle result and is reported separately as a diagnostic.

`evaluation_contract.md` fixes background ID, patch majority voting and ties,
mask overlap, unassigned patches, empty masks, the foreground definition, object
matching, unmatched objects, and per-image and per-dataset aggregation -- before
the first baseline. A change to it raises its version and both models are
re-evaluated.

## Experiment loop

1. Check repository state, branch, running work, available compute.
2. Reserve the next test number; record parent, hypothesis, changed variables,
   success criterion.
3. After a code change, run a small shape / gradient / CLI check.
4. Train and evaluate; save the full reproducible command, environment, seed,
   commit and data identifiers.
5. Read the three patch metrics together with the theta-to-spike diagnostics.
6. Analyse why it improved or failed; write the next hypothesis.
7. Add results and insight to `collaborative_test/`, commit and push to `patch_v2`.
8. Fetch `patch_v2_sw`, read its `collaborative_test` and code diff since the last
   review, and port only changes that hold up against the data, the evaluation
   definitions, the actual readout and the diff itself. Never overwrite peer files
   wholesale; port selectively into the current structure. Keep what the peer
   reported separate from what reproduced here.

Repeat until the goal is met. If a session ends or compute runs out, record the
state and the resume point. Report an improvement as soon as it is confirmed, and
record experiments that did not improve as well.

## Reporting

Exploration may compare at seed 0 alone. A validation metric above its previous
best is a candidate: report it with the other two metrics and say it is one seed.
Promising settings are then retrained at seeds 0, 1, 2. Success is judged only on
3-seed means of all three metrics. Every report carries the previous value, the
new value, the absolute change, the relative change, the standard deviation, and
the data and evaluation versions.

## collaborative_test

Numbering `PV2_0001` upward here, `SW_0001` on the peer branch. Numbers are never
reused, including for failed or interrupted runs. Status is one of planned,
running, completed, failed, interrupted.

```
collaborative_test/
  README.md  INDEX.md  STATUS.md  evaluation_contract.md
  data/split_manifest.json
  baselines/slot_attention.md  baselines/slot_attention_metrics.csv
  peer_updates/REVIEW_0001.md
  PV2_0001_baseline/
    README.md config.json commands.sh changes.md code.patch
    provenance.json metrics.csv diagnostics/ figures/ artifacts.json
```

Each test records: test ID, status, hypothesis, parent and comparison; base and
resulting commit, branch, and the source commit and peer test ID of anything
ported; data identifier, split manifest version, evaluation contract version;
every effective hyperparameter and seed; the full commands and environment;
per-image and per-seed values of the three metrics with mean and std; PLV,
collapse, membrane threshold-crossing rate, spike rate, spikes per image, active
oscillator count; where useful the dense/dendritic/membrane statistics, each
loss's raw and weighted value, gradient norms and cosine similarities;
checkpoint, log and result locations with hashes; interpretation, limitations and
the next proposal.

`changes.md` is specific: file path, class/function/config name, line numbers at
that commit, the previous computation or value, the new one, the reason and the
expected effect, and the related test ID with its verification. `code.patch`
holds that experiment's diff alone. Large checkpoints, raw data and full logs stay
out of git; `artifacts.json` records their location and hash.

## Running

Confirm the user's paths, server, scheduler, GPUs and storage policy first. Use
the one most available permitted GPU and never interrupt another user's process.
Run long training detached from the session and record the PID or job ID and the
log path. On an error, find the cause, fix the command, and record the failure and
the fix in the same test. After a session ends, resume from `STATUS.md` without
re-running completed work.

## On reaching the goal

Report both models' three metrics side by side as per-seed values, means and
standard deviations, with the final config, data split, evaluation version, code
commit, reproduction commands, checkpoint identifiers and known limitations --
and the evidence that the final score came from classifier masks on
spikes or membrane.
