# STATUS

Updated 2026-10-19 00:45. Branch `patch_v2`. Contract v1 and v2, split v1.

> ## INCIDENT, 2026-10-19 — `/export_home/tkim1` is faulted
>
> `ls`, `stat` and `df` on `/export_home/tkim1` all return **`Input/output error`**. This is a
> filesystem fault on the server's home mount, not a permissions or VPN problem: `/work` is
> healthy and answers normally in the same session. **It needs the machine's administrator.**
>
> ### What survives
>
> | | where | state |
> | --- | --- | --- |
> | every checkpoint | `/work/USERS/tkim1/runs/` | safe |
> | gamma tensors, targets, split manifest copy | `/work/USERS/tkim1/` | safe |
> | all result files (`pv2_*_results.txt`) | `/work/USERS/tkim1/runs/` | safe |
> | `collaborative_test/` records and scripts | git, `origin/patch_v2` | safe |
> | `PV2_0098` and `PV2_0110` training | GPUs 2 and 3 | **still running** — already resident in memory, reading from `/work` and writing checkpoints to `/work` |
>
> ### What is at risk
>
> **`patch_v2` has never tracked `snn_kuramoto_bidirectional/`.** Only `collaborative_test/`
> is in the branch. The peer branch `patch_v2_sw` *does* track the model package; ours does
> not. So the only current copy of our model source was on the faulted mount.
>
> The local working tree has a copy, but it is dated **2026-10-03** — sixteen days stale, and
> missing `geodesic_chunk`/`_relax` and `coupling_balance`.
>
> Recent model-source changes are recoverable from their experiment diffs, which are in git:
> `PV2_0088_grid32_enablers/code.patch` (geodesic chunking) and
> `PV2_0092_signed_coupling/code.patch` (signed coupling). They need a correct base.
>
> ### Restored into git from this session, before they could be lost
>
> - `evaluate_model.py`: `_merge_groups` plus `--merge-threshold` and `--merge-rounds`. These
>   had never been committed and existed only on the faulted mount.
> - `figparts.py`: the row x column phase-alignment panels.
> - `panels.py` and `peer_data.py` are **not** restored and are not in git.
>
> ### Recovery attempted and FAILED — the stale copy cannot evaluate
>
> The local tree was staged to `/work/USERS/tkim1/code/` (via a tar pipe; `rsync` cannot be
> used while the remote login shell fails to read its `.bashrc` on the faulted mount) and the
> mandated regression row was run: `BIM6_s1`, grid 16, validation 300, expecting
> `0.7239 0.7251 0.4951`. It did not reach a number:
>
> ```
> TypeError: S2NetHyperparameters.__init__() got an unexpected keyword argument
>            'graph_extra_channels'
> ```
>
> The evaluator passes 20 hyperparameters; the 2026-10-03 package accepts 68 fields and is
> missing **7** of the 20: `coupling_balance`, `dendrite_per_region`, `geodesic_base`,
> `geodesic_chunk`, `geodesic_power`, `membrane_threshold_k`, `membrane_threshold_mode`.
>
> These are not inert flags. `geodesic_base` and `geodesic_power` (`PV2_0076`, `PV2_0078`),
> the population membrane threshold (`PV2_0094`) and `dendrite_per_region` all changed what
> the model computes, and `BIM6_s1` was trained under them. Only two of the seven have a diff
> in git — `PV2_0088_grid32_enablers/code.patch` and `PV2_0092_signed_coupling/code.patch`.
> **The other five exist nowhere but the faulted mount.** Reconstructing them by guessing
> would load the weights into a model that computes something else, which is the exact
> failure the regression row is there to catch.
>
> So: evaluation cannot be restored until `/export_home` comes back. Training checkpoints are
> unaffected and keep accumulating on `/work`.
>
> ### Two findings while blocked
>
> **`PV2_0098` is interrupted, not completed.** Both arms trained; neither could be scored,
> because the evaluator is on the faulted mount. DS_1000 ended at loss 16.673826 / collapse
> 2.154499, DS_6000 at **14.429044 / 1.954825** — six times the data gives a clearly better
> training signal and *no patch metric at all*. The checkpoints are intact on `/work`; the
> test resumes by scoring them.
>
> **Graph feedback costs ~27x the wall clock.** `PV2_0110`'s control ran 40 epochs in 23 min
> (0.58 min/epoch). The `--graph-feedback-strength 0.5` arm is at epoch 5 after 78 min
> (15.6 min/epoch) — `update_alignment` runs at every one of the 64 timesteps. Projected
> ~10 h for this arm alone, against 23 min for the control. It is training healthily
> (`spike_plv_collapse` 2.293 → 2.283 → 2.262, falling), so it is left running: GPU 3 is ours
> and idle, and the mount being down means nothing else can use it. But the cost is itself a
> result — if graph feedback needs 27x the compute, it has to earn far more than a marginal
> gain to be worth keeping, and that judgement can be made before the numbers arrive.
>
> ### `REVIEW_0007` changes the next experiment
>
> The peer branch moved 81 tests past what `REVIEW_0006` covered (SW_0157..SW_0236 at
> `790ec4f`). Two things matter here.
>
> **`SW_0234` reproduced our route map on their own data.** Merge .885762, coverage .883696,
> both **.911458** — superadditive past .90, as `PV2_0106` found at .9329. Different grid,
> different split, different supervision. The decomposition is a property of the readout.
>
> **Their oracle merge also loses Object IoU** (.674313 → .667059), and theirs chooses pairs
> *with the truth*. Our `PV2_0107b` lost .0285 with a label-free merger and could be blamed on
> pair selection; theirs cannot. **`PV2_0111` as scoped — complete linkage against mean
> linkage — is withdrawn.** The hypothesis behind it is dead.
>
> **Coverage moves first, and its shape is now constrained.** `SW_0207`, their label-free
> foreground head, widened the support of existing groups: foreground IoU +.171275, FG-ARI
> −.053863, Object IoU down, closed. `SW_0234`'s oracle recovery instead creates **new groups
> for objects missed entirely** and gains all three. So: a coverage stage may instantiate
> groups for unclaimed regions and may not widen groups that already exist.
>
> Their headline — SW0227 beating Slot on all three as three-seed means — is on their declared
> **mask-supervised** leaderboard, on a reused validation split, on their contract. Nothing
> from it is portable to our column.
>
> ### `PV2_0112` is specified, implemented and offline-verified — it needs only a server
>
> The third part of the standing goal is no longer a plan sentence. `PV2_0112` exists:
> `_coverage_groups` in `evaluate_model.py` with `--coverage-quantile`, and
> `test_coverage.py` passing **8/8 on CPU torch** against a synthetic grid. The pre-registered
> gate is in its README — all three metrics up, `patch_matched_object_iou` specifically up,
> predicted foreground at or below 0.20 — together with the falsification that closes it.
>
> The test earned its keep before any GPU time: the first implementation admitted patches at
> `act >= quantile_value`, and because the background is 87.5% of an image the quantile value
> *is* the background value, so it recovered **238 of 256 patches as one object**. Strict `>`
> fixes it and is equivalent where activity is continuous.
>
> It runs **after** the merge stage, never before — covering first would let the merger absorb
> the recovered groups into existing ones, which is `SW_0207`'s failure by another route.
>
> ### Resume point
>
> 1. When `/export_home` returns, **first** copy `snn_kuramoto_bidirectional/` off it and commit
>    it to `patch_v2`. Never work another experiment with the model source untracked.
> 2. If it does not return, rebuild the package from the 2026-10-03 local copy plus the two
>    `code.patch` files, and re-verify against a known checkpoint by re-scoring `PV2_0104`
>    seed 0 — it must reproduce 0.7699 / 0.7531 / 0.7017 before anything below it is read.
> 3. `PV2_0098` (data scale) and `PV2_0110` (bidirectional feedback) finish on their own; read
>    `/work/USERS/tkim1/runs/pv2_0098_results.txt` and `pv2_0110_results.txt`.
>

> ## Resume point, 2026-10-14 — access lost, not compute
>
> **The VPN tunnel to the 152.19.x.x subnet died.** Both `raptor` (152.19.10.112) and
> `frontier` (152.19.10.80) fail with `Can't assign requested address` and lose 100% of
> pings, while the open internet is fine and the `utun8` route entry is still present.
> Nothing is wrong on the server; the jobs below are detached with `nohup` and keep
> running. Reconnect the VPN and the results are one command away.
>
> ### What is waiting on the server
>
> | job | file to read | state when contact was lost |
> | --- | --- | --- |
> | `PV2_0092` signed coupling | `/work/USERS/tkim1/runs/pv2_0092_results.txt` | balance 0.0 and 0.25 done, 0.5 and 1.0 running |
> | `PV2_0093` grid-32 spike path | `/work/USERS/tkim1/runs/pv2_0093_results.txt` | queued; polls for `signed.sh` to end, then runs three 8-epoch arms |
>
> Both append as they go, so nothing is lost. `PV2_0092`'s control is healthy — balance
> 0.0 trained 23.52 -> 17.00 at `spike_plv_collapse` 2.157, balance 0.25 23.66 -> 17.05 at
> 2.142 — so the comparison can carry a verdict, unlike `PV2_0085`'s.
>
> ### The one command to run on reconnection
>
> ```
> ssh frontier 'cat /work/USERS/tkim1/runs/pv2_0092_results.txt;
>               cat /work/USERS/tkim1/runs/pv2_0093_results.txt'
> ```
>
> `PV2_0092` prints its own gate verdict at the end. The gate was registered before the
> run (`PROPOSAL_0.9_v2.md`): the fg_ari gain must appear at predicted foreground at or
> below 0.20, must not be bought with foreground IoU, and `patch_matched_object_iou` must
> move too.
>
> ### Then
>
> 1. If signed coupling passes at seed 0, retrain at seeds 1 and 2, select on validation,
>    read test once. If it fails, record it as the fifth mechanism-level improvement that
>    did not convert, and the remaining route is `PV2_0086`'s motion term at seeds 1 and 2
>    with extent matched.
> 2. `PV2_0093` decides which knob keeps the 32x32 spike path alive. Hypothesis B — that
>    `--graph-spatial-decay` is in grid units and 0.35 is twice the intended decay at this
>    grid — is **untested, not refuted**; `PV2_0089`'s eval-time test of it was a no-op
>    because `spatial_rate` is an `nn.Parameter` restored by `load_state_dict`.
> 3. `PV2_0091`, our extent dial against aligned Slot Attention's operating point
>    (0.8102 fg_ari at predicted foreground 0.8531), was stopped at one row for being too
>    slow under GPU contention. Redo it cheaply: `--limit 150 --num-time-steps 512`,
>    spectral k from 6 to 32. It is the measurement that would settle whether the readout
>    alone can close the gap.
> 4. Nothing portable is pending from the peer; `REVIEW_0006` covers through `ce923f1`.
>    Their SW_0156 is signed coupling too, inference-only on a frozen checkpoint, with no
>    outcome yet. A split verdict between their arm and ours would itself be the finding.

> ## Correction, `PV2_0090`
>
> **The claim that we beat Slot Attention on all three metrics does not survive.**
> The baseline was run through the published centre crop, which on a 240x320 frame
> is the centre 240x240, so its 16x16 prediction map and the 16x16 target map
> described **different regions of the image** -- column `c` up to 40 px, two target
> columns, apart. Correcting it, on test, three seeds x three seeded draws:
>
> | metric | ours (best) | Slot, crop | Slot, aligned | verdict |
> | --- | --- | --- | --- | --- |
> | `patch_fg_ari` | 0.7603 | 0.6656 | **0.8102** | **Slot leads by 0.0499** |
> | `patch_foreground_iou` | **0.7132** | 0.1151 | 0.1151 | ours, +0.5981 |
> | `patch_matched_object_iou` | **0.6598** | 0.0874 | 0.1025 | ours, +0.5573 |
>
> Our own numbers are unchanged; the error was in the baseline harness and ran in
> our favour. The 0.75 bar is still cleared in absolute terms (0.7603). The fg_ari
> **lead** is gone. The two IoU results stand and are the part of this project that
> does not depend on a tie-break: Slot Attention predicts 85% foreground against an
> 11.8% target under every preprocessing tried.
>
> Everything below this box predates the correction. Read the Slot Attention column
> in it as the **crop** column above.

## Where things stand

Active goal, three parts: exceed **0.75** `patch_fg_ari`; review and apply the peer's
results; and if 0.75 is unreachable, find a route to **0.9+** keeping SNN + Kuramoto
bidirectional.

**Parts 1 and 2 are met. Part 3 is not.**

| goal condition | state |
| --- | --- |
| fg_ari 0.75 | **met** — 0.7533 on test, 3 seeds (`PV2_0061`) |
| beat Slot Attention on all three | **NOT met** -- `PV2_0090` corrected the baseline; aligned Slot leads fg_ari 0.8102 vs 0.7603. The two IoU wins stand |
| apply the peer's positive results | **done** — graph freeze ported (`PV2_0044`), 0.7059 -> 0.7250 |
| a demonstrated route to 0.9 | **not met**; 21 routes closed, the 0.9671 'feasibility' withdrawn (`PV2_0072`), failure explained arithmetically (`PV2_0075`) |

### The test result (`PV2_0061`), read once

Test IDs 7000-7999, **all 1000 images**, from classifier masks on the model's own
spikes, spectral k=14 with one background cluster.

| metric | ours k=14, power 2 | matched Slot | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7603** +- 0.0003 | 0.6663 +- 0.1108 | +0.0940 | +14.1% |
| `patch_foreground_iou` | **0.3373** +- 0.0207 | 0.1177 +- 0.0101 | +0.2196 | +186.6% |
| `patch_matched_object_iou` | **0.6598** +- 0.0061 | 0.0871 +- 0.0040 | +0.5727 | +657.6% |

Per seed, ours: 0.7600 / 0.7605 / 0.7604. Slot: 0.7265 / 0.7340 / 0.5385. The
superlinear geodesic step cost (`PV2_0076`) adds +0.0070 / +0.0015 / +0.0049 over
`PV2_0061`'s 0.7533 / 0.3358 / 0.6549 at the same readout, and the connected-component
configuration improves with it too, 0.7258 / 0.7124 / 0.5029 -> **0.7300 / 0.7132 /
0.5050**. This is the second test read of the session; the first was `PV2_0061`.

The same checkpoints read through connected components give **0.7258 / 0.7124 /
0.5029** on the same images — also beating Slot on all three, foreground IoU twice as
high, fg_ari below 0.75.

**What k=14 costs:** -0.3767 foreground IoU against that configuration, and predicted
foreground of 0.556 against a target of ~0.118. fg_ari is computed over true-foreground
patches only, so a wider predicted foreground can only help it; `PV2_0059` and
`PV2_0060` map the whole dial (k=12 fg_ari 0.7381 at fg_iou 0.4413; k=20 fg_ari 0.7718
at fg_iou 0.1680). k was chosen on validation, which the contract permits.

**Evidence the score comes from spikes:** the spectral readout clusters
`_spike_affinity(spikes, components)`, `--readout` stays at its default `spikes`, and
`PV2_0043` measured binary spike timing carrying that affinity at r = 0.9822
(0.9808 / 0.9807 on these checkpoints). `--spectral-k 14` is a fixed group count;
`--spectral-oracle-k` is the oracle variant and is not used.

**Limitations:** the three seeds share one graph through `PV2_0044`'s freeze, so they
are not three fully independent runs; the Slot baseline ran 320 epochs against the
official 457 and is a lower bound, with its weak seed 2 (0.5385) dragging the mean,
so a better-trained Slot could close the fg_ari gap while leaving both IoU gaps intact.

## The Slot Attention comparison

`PV2_0049`, the official architecture trained on our own train split, three seeds,
each evaluated under three seeded inference draws:

| metric | ours (3-seed) | matched SA (3-seed) | absolute | relative |
| --- | --- | --- | --- | --- |
| `patch_fg_ari` | **0.7250** | 0.6635 +- 0.1090 | +0.0615 | +9.3% |
| `patch_foreground_iou` | **0.7188** | 0.1184 +- 0.0093 | +0.6004 | +507% |
| `patch_matched_object_iou` | **0.4953** | 0.0869 +- 0.0037 | +0.4084 | +470% |

Ahead on all three. Three caveats travel with it: the baseline ran 320 epochs
against the official 457, so it is a **lower bound** on Slot Attention; our three
seeds **share one graph** through `PV2_0044`'s freeze, so the variance the graph
contributes is removed by construction; and this is validation, not test.

Inference noise is ~0.005 per checkpoint against 0.1090 across checkpoints, so the
weak seed is the checkpoint and not the draw. `PV2_0047` claimed the opposite from
two seeds with unseeded inference and is superseded.

## What is closed

**The architecture is not the limit.** `PV2_0034` ran oracle features through this
pipeline unchanged and reached **0.9754**; the encoder costs 0.27. `PV2_0035/0036`
split that loss: within-object spread 0.465 -> 0.000 is worth +0.170, the
between-object margin carries the rest.

**Graph quality is spent** — a ground-truth coupling graph is worth +0.06
(`PV2_0005`).

**Seven feature routes**: DINOv2 worse, label-free clustering worse,
self-bootstrapping circular (+0.006), encoder capacity diverges, slot-term defaults
already optimal, end-to-end alone 0.6779, frozen core collapses.

**Data scale**, both ways. `PV2_0048`: more images at matched compute gives 0.6779
-> 0.5983 -> 0.4790. `PV2_0051`: the scale-invariant separation ratio improved only
7% relative, so the encoder did not get better either, and the 27-hour
full-convergence run was cancelled rather than run.

**The integration window.** `PV2_0052`: +0.0005 going from 1024 to 4096, inside the
seed spread. It also confirms 0.7250 is not an artifact of the window.

## What is open — part 3, the route to 0.9

**Not met, but the target is now exact and a correction matters more than the rest.**

### `PV2_0005` was wrong, and I leaned on it all session

`PV2_0066` split the two paths features take into the dynamics. Feeding oracle features
to the **graph alone** gives **0.9671 / 0.9353 / 0.7584 at predicted foreground 0.1121**
— 94% of the total oracle gain, and the first fg_ari gain in this project that does
**not** come with foreground over-prediction. Through theta initialisation alone it is
only +0.0922.

`PV2_0005` had reported an oracle coupling graph worth +0.06, and I repeated "the graph
is spent" throughout. That test forced the **final coupling matrix**, bypassing the
generator's projection, cosine, top-k, geodesic relaxation and spatial decay. Given
oracle **features** instead, that machinery amplifies them. **Nearly all the headroom
is in the graph**, not in theta initialisation and not outside the graph.

### What the 0.9671 actually measures — corrected by `PV2_0072`

`make_oracle_gamma.py` builds its tensor as `codes[true_labels]`, so the oracle gamma is
the **answer** rather than better features. 0.9671 / 0.9353 / 0.7584 therefore measures
how faithfully the readout **transcribes a grouping it is handed**, not whether these
dynamics can discover one. I had written it as "the 0.9 target is reachable by this
architecture unmodified"; **that does not follow and is withdrawn.**

What `PV2_0072` does establish is sharper and more useful. Feeding the graph a grouping
corrupted to a known degree:

| input fg_ari | output fg_ari | gain |
| --- | --- | --- |
| 1.0000 | 0.9678 | -0.032 |
| 0.7525 | 0.7766 | +0.024 |
| 0.4205 | 0.5087 | +0.088 |
| 0.0416 | 0.1933 | **+0.152** |

**The pipeline repairs damaged groupings**, strongly for badly damaged ones, with the
crossing where output = input near 0.85-0.88.

But `PV2_0073` iterates the model's **own** grouping through the same path and it
converges at **0.7274** in one step (+0.0035, inside the seed spread). The two results
reconcile through the error type: `PV2_0072`'s corruption is random, so its errors are
spatially **incoherent** -- a wrong patch among right ones, exactly what the geodesic
relaxation and distance prior are built to repair. The model's own errors are spatially
**coherent**: whole objects merged, which `PV2_0053` measured in 38.8% of images. A
merged pair is internally consistent, so there is nothing local to fix.

**The amplification is real but acts on the wrong error type.** That is why nineteen
routes failed: each tried to reduce the amount of error, while the pipeline can only
remove error of a kind it does not make.

### Fourteen routes have failed to produce it, and they fail in two opposite ways

`PV2_0067` sent every transform to the graph alone, with the phase path on real gamma:

| graph gamma | cos_between | fg_ari | fg_iou | pred_fg |
| --- | --- | --- | --- | --- |
| real | 0.2483 | 0.7250 | 0.7188 | **0.1299** |
| whitened eps=0.1 | **0.6900** | 0.7602 | 0.2256 | 0.4861 |
| smoothed | 0.2113 | 0.7196 | **0.7618** | **0.1178** |
| oracle (diagnostic) | **1.0000** | **0.9671** | **0.9353** | **0.1121** |

The margin can be enlarged (whitening, 0.2483 -> 0.7693) but only by widening predicted
foreground to 0.45-0.49 — **and that happens even confined to the graph**, so the phase
path was never the obstacle. Calibration can be kept (smoothing, 0.1178) but only
without a margin gain. Oracle codes are one-hot per object, orthogonal between objects
by construction; no fixed transform of appearance features reproduces that, and a
trained one would need object identity.

Three separate transforms also improved a separation statistic without improving fg_ari
in the calibrated regime (`PV2_0058`, `PV2_0062`, `PV2_0064`), so separation statistics
are not sufficient statistics for the score — they bound the problem without pointing
at a solution.

### Sixteen routes, and what each kind of failure says

| route | result |
| --- | --- |
| eleven on feature geometry | closed; three improved a separation statistic without moving fg_ari (`PV2_0058`, `PV2_0062`, `PV2_0064`) |
| transforms confined to the graph (`PV2_0067`) | margin without calibration, or calibration without margin |
| the matched Slot encoder's features (`PV2_0065`) | within-object cosine 0.0019, near the oracle's zero; between-object 0.0316, eight times too small; fg_ari 0.0604 |
| reconstruction training the graph, features frozen (`PV2_0068`) | optimisation worked (loss 18.06 vs 17.86) but best calibrated 0.7175, -0.0064 |
| the graph fed its own groups as one-hot (`PV2_0069`) | **0.7239 in, 0.7199 out** -- a fixed point, no amplification |

`PV2_0069` is the sharpest of these. Oracle one-hot codes into the graph give 0.9671;
one-hot codes built from the model's own 0.7239-accurate groups give 0.7199. The graph
does not amplify its own estimate, so no iterative scheme climbs, and the errors in the
predicted groups are exactly the errors that survive it.

What the oracle supplies is **instance identity**. Every label-free route tried either
reproduces its geometry without the information, or is a fixed point at the accuracy it
starts from.

### `PV2_0070`, the last route I had named as untested

Cross-view consistency: view two is view one under per-image per-channel gain/offset
jitter plus noise, so the partition is unchanged and the two views' PLV matrices must
agree. `PV2_0057` had identified shading as the main source of within-object spread, and
a consistency objective is the one kind of signal that does not admit `PV2_0069`'s fixed
point, since agreeing on a wrong grouping is not free when the views differ in the
nuisance that caused the error.

**The term worked and the score fell.** `view_consistency` drops from 0.004486 to
0.001360-0.002252, so the dynamics became 2-3x more invariant to the jitter, and every
arm stayed calibrated (predicted foreground 0.115-0.134) — this failure is not
over-prediction. Best is 0.6757 at weight 200, **-0.0482** on the bar; weight 1000
collapses to 0.3989.

The direction suggests something sharper: **part of what the model uses is
shading-correlated**, so forcing invariance removes usable signal. `PV2_0058`, where
chromaticity removes the same factor analytically, scored below the baseline too.

### The pattern across seventeen routes

Anything that makes the features more regular — lower within-object spread, a bigger
margin, invariance to a nuisance — leaves fg_ari where it was or lowers it. The oracle's
gain comes from its codes **being** instance identity, and the structural reason no
label-free procedure here produces that is in the data: these are single static renders,
so nothing in the input distinguishes two same-coloured objects except spatial
separation, which the pipeline already uses.

### The arithmetic of the failure (`PV2_0074`, `PV2_0075`)

`_geodesic_distance` exists to separate same-coloured objects -- "walking between them
crosses background, so the route is what tells them apart" -- and its strength is a
**learned** parameter, so `PV2_0071` could not vary it. Written directly into the
checkpoint, the learned 2.17 turns out to be the best of 0.5 through 32 (best +0.0007,
noise; 32 gives 0.6406). Groups per image climb 5.46 -> 6.59 as it rises, so objects do
split, and fg_ari falls, so the splits miss the boundaries.

`PV2_0075` says why, in adjacent-pair dissimilarity, which is all the geodesic sees:

| pair type | mean `1 - cosine` |
| --- | --- |
| within one object | 0.0331 |
| **background to background** | **0.0054** |
| object to background | 0.1743 |
| between two objects | 0.1587 |

The wall is real -- crossing into background costs **5.27x** a shading step inside an
object. **The corridor is the problem**: background-to-background is six times *cheaper*
than a within-object step, so the route between two same-coloured objects is two thin
walls with a nearly frictionless corridor. At the learned contrast a wall step costs
1.378, a background step 1.012 and a within-object step 1.072, so reaching an object six
patches away costs about 6.8 while crossing a six-patch-wide object costs 6.4 -- nearly
a tie, which is the merge.

The contrast cannot fix it because it multiplies `(1 - similarity)` **uniformly**:
raising it scales walls and shading steps together, and objects with shading gradients
split before the walls dominate.

### `PV2_0076` is a readout correction, not a training change (`PV2_0077`)

Training under the superlinear cost gives 0.4268 at power 2 and 0.5248 at power 3,
against the 0.7239 bar. The control settles where the damage is: the same checkpoints
evaluated at power 1 give 0.4249 and 0.5254, so the **weights** are bad, not the
evaluation. The loss meanwhile went *down* -- 16.71 and 16.80 against 17.86 -- so the
objective was satisfied better while the score collapsed, the pattern `PV2_0055` recorded
for position in the graph. And training **lowered** the contrast, 2.1732 -> 1.9588,
choosing to rely on the geodesic selectivity less rather than more; the prediction that it
would prefer a stronger one was wrong.

So the sharper cost is a better **readout** of the graph these weights produce and a worse
**objective** to fit them with. `PV2_0076`'s +0.0070 on test stands as a post-hoc
correction.

### The toll, and a fourth mispredicted arithmetic (`PV2_0078`)

`PV2_0075`'s other candidate, a per-step toll on the corridor, turns out to be the
leading `1` already in the cost. Lowering it should have helped: at power 2 and contrast
8 the route to an object six patches away costs 6.48 against 6.05 for crossing a
six-patch object, a ratio of 1.07, and at base 0.1 those become 1.09 and 0.654, ratio
1.66.

It does not. In the calibrated regime base 1.0 is the best of 1.0 down to 0.02, trading a
little fg_ari for a little foreground IoU monotonically. The spectral rows look like a
+0.0140 gain and are not: predicted foreground goes 0.5423 -> 0.6522 while foreground IoU
falls 0.0796, the extent dial again, and the automated check missed it because the
spectral readout never sits below the 0.20 threshold.

That is the **fourth** time a geometric or arithmetic quantity improved without the score
following (`PV2_0058`, `PV2_0062`, `PV2_0064`, here). The common thread: these quantities
are computed on ground-truth object pairs, while the readout acts on whatever the dynamics
produce, and a barrier in the cost matrix still has to survive the rollout, the gate, the
membrane threshold and the component extraction. `PV2_0076` is the only change of this
family that transferred, and `PV2_0077` showed it works only outside the objective.

### The dynamics, and the pattern that now has five instances (`PV2_0079`)

The Kuramoto dynamics were the last untouched axis. `osc_dim` 8 scores 0.5670, 16 never
trains (loss 86, the dead-membrane regime), coupling 64 gives 0.6088 and 1024 diverges.
**osc_dim 4 and k 256 are already the best of these**, and the hypothesis that four phase
dimensions are too few for 6.32 objects is refuted.

The losses explain it, but not the way I first wrote. **osc_dim 8 reached 14.19 against
the baseline's 17.86 and scored 0.5670**, and `PV2_0080` then found **osc_dim 3 reaches
16.21 for 0.6783** — lower loss, worse score, on the *small* side. So the pattern is not
extra capacity creating degenerate solutions; reducing capacity does the same thing.

| setting | loss | fg_ari |
| --- | --- | --- |
| osc_dim 3 | 16.21 | 0.6783 |
| **osc_dim 4** | **17.86** | **0.7239** |
| osc_dim 8 | 14.19 | 0.5670 |

**The operating point has a relatively high loss and the best score**, so the objective
and the metric are **decoupled around it**. Ten instances now fit that (`PV2_0055`,
`PV2_0058`, `PV2_0068`, `PV2_0077`, `PV2_0079`, `PV2_0080` and all four arms of
`PV2_0081`), and the constraint it puts on future work is sharp: **lowering this loss is
not evidence of progress.**

I also read it as reframing `PV2_0044`'s graph freeze into "constraint is the lever".
`PV2_0081` corrects that: at the **same seed** the graph freeze gives 0.7235 against the
untouched 0.7239, so its +0.019 was in the 3-seed mean, by rescuing the weak seed from
0.6639. It reduced **seed variance** rather than improving any seed, and no other
component freeze helps either.

`PV2_0081` also found that **1,325 of 69,042 trainable parameters (1.9%) match full
training** — `kuramoto` and `graph_generator` frozen together, leaving only the dendrite,
membrane and gamma projection, scores 0.7249. The 67,585 `kuramoto` parameters contribute
nothing measurable once their graph is held with them, and freezing `kuramoto` alone drops
to 0.6858 because the graph then drifts from the dynamics it was fitted to. There is very
little here for an optimiser to improve, which is consistent with every capacity and
constraint experiment failing.

### Exhausted, and the obstacle stated precisely

Twenty-eight interventions across twenty-five numbered routes: features 11, graph 8,
geodesic cost 3, readout 2, dynamics 4. One transferred (`PV2_0076`, +0.0070 on test), and
only outside the objective (`PV2_0077`).

1. The oracle numbers measure **transcription**, not discovery (`PV2_0072`).
2. The pipeline repairs spatially **incoherent** damage; its own errors are coherent
   object merges and it is a fixed point on those at 0.727 (`PV2_0073`).
3. The geodesic barrier is two thin walls (5.27x) with a corridor six times cheaper than a
   within-object step, and no shape of the cost fixes that without splitting objects
   (`PV2_0075`, `PV2_0078`).
4. Five changes reached a lower loss and a worse score, so the objective admits degenerate
   solutions that more capacity makes easier to find.

### Start here for part 3

`PROPOSAL_0.9.md` is the one route left, costed and written to be run: independent object
motion as the signal, the existing consistency term pointed at frame pairs, contract v1
and the architecture untouched. **It needs a decision** — it adds data this project does
not use. `PV2_0082` and `PV2_0083` removed the two alternatives that were not the model
(the metric and the input resolution), so there is nothing else outstanding.

`ROUTE_MAP_0.9.md` holds the consolidated map: the five structural findings and all 27
closed routes, with the protocol devices that caught wrong conclusions. It exists because
after 27 routes I nearly re-ran one that was already closed at `PV2_0016-18` — check it
before spending compute.

### Resume here

Within the architecture, the next thing to change is `_geodesic_distance` itself: a step
cost **superlinear** in dissimilarity, or a fixed per-step toll so a long cheap corridor
costs something. Both are core changes rather than parameter choices, and the margin is
tight -- within-object 0.0331 against wall 0.1743, with the usable window for a
selective threshold roughly between 6 and 30, already sampled at 8 for 0.7183.

Outside the architecture, part 3 needs what a single static render does not contain:
motion, or several views of one scene.

## Where the peer stands

Their branch moved to 32x32 = 1024 patches under its own contract. Three-seed
native32: **0.7819 / 0.6054 / 0.5970** against their matched Slot32 **0.7817 /
0.2044 / 0.2032** -- ahead on all three, but the FG-ARI margin is **+0.0003** while
their Slot32 seed spread is **0.1353**, so FG-ARI is a tie and the win is the IoUs.
Scores are not poolable with ours (128x128 vs 240x320, 4px vs 15x20 patches, 70,000
vs 6,000 training IDs). Of 25 new records none carries a positive, reproduced,
promotable result; their incumbent is unchanged at SW0097.

One mechanism does transfer: Slot Attention's foreground IoU is 0.2044 on their
contract and 0.1184 on ours. Across two resolutions, two label rules and two
budgets it still over-predicts foreground, and that is where both branches' real
advantage comes from.

## Resume point

Parts 1 and 2 are done and on the test split. Part 3 is the open work: a training
signal that geometry cannot satisfy, with the peer's SW0139 result the thing to read
first.

Verified and reusable: `--spectral-k` on the graph-frozen checkpoints is the fg_ari /
foreground-IoU dial (`PV2_0059`, `PV2_0060`); `graph_extra_channels` routes trailing
gamma channels to the coupling graph alone and is a no-op at 0; the evaluator reads
both gamma widths from the checkpoint and takes `--save-predictions` on both readout
branches; `run_slot_attention.py` takes a `--seed` and is bitwise reproducible.

**Every experiment script opens with a regression row** re-scoring an existing
checkpoint, and nothing below it is read unless that reproduces its recorded value.
`PV2_0055` is why: a patch broke the graph-selection chain so the learned graph was
replaced by the static one at `graph_extra_channels = 0`, scoring 0.0033 instead of
0.7239 — a bug that fired only in the configuration every existing result uses, while
the new-path smoke test passed. `PV2_0061` ran that gate on validation **before** the
test split was touched.

Compute: GPUs 2 and 3 only, 0 and 1 are the collaborator's — but check actual
occupancy before dispatch, since other users do appear on 2 and 3; an evaluation was
lost to OOM against another user's 22.5 GiB on GPU 3. Heavy runs queue behind a gate
that waits for a clear GPU.
