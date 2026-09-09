# SNN Kuramoto Bidirectional

Experimental unsupervised object-discovery research. The question is whether
spatially grounded image features can drive graph-coupled Kuramoto oscillators
and a recurrent spiking network to discover stable, image-specific, object-like
regions without object labels.

This is a research repository, not a finished segmentation system. Object
grouping now works measurably better than chance and the spiking side finally
carries part of the signal, but the masks are still coarse regions rather than
object boundaries.

## Intended architecture

The division of labour matters and is worth stating up front, because several
design decisions only make sense in this light:

~~~
image -> features -> Kuramoto oscillators        BINDING happens here
                          |
                          v
                     spiking network             TRANSDUCTION happens here
                          |
                          v
              oscillators that spike together    READOUT: co-active units are
              are read as one object             one object candidate
~~~

The oscillators decide what groups with what. The spiking layers are supposed to
carry that decision into spike timing, so that co-firing reveals the grouping.
Much of the work below is about the fact that they were not carrying it: the
spiking readout started at 4% of what the phases hold and is now at 49%.

## Current status

Averaged over four seeds, scored against colour-derived CLEVR object masks:

| readout | original gate | `phase_mean` |
| --- | --- | --- |
| ARI from oscillator phase | **0.329** | 0.304 |
| ARI from membrane | 0.045 | **0.182** |
| ARI from spikes | 0.042 | **0.150** |
| foreground IoU from spikes | 0.122 | **0.179** |

Untrained baseline is ARI 0.033. The `phase_mean` gate improves every spiking
readout on 4 of 4 seeds, at the cost of about 7.6% on the phase readout (3 of 4
seeds), which is the trade this project wants: the spiking side is supposed to
carry the grouping, and it went from 13% to 49% of what the phases hold.

For scale: 0 is chance, and slot-based object-discovery methods on CLEVR
typically report foreground ARI in the 0.9 range. The model finds roughly where
objects are and produces a different grouping for every image, but it does not
recover object boundaries. Two caveats inflate the numbers slightly: the number
of clusters is taken from ground truth (oracle k), and the colour-derived masks
capture only 78% of the scene objects, so objects the model does find can still
be scored as wrong.

`results/result_overview.png` shows images, ground truth, and both readouts side
by side.

---

# Part 1 — What was wrong, and what changed

Every number below is measured. Where a conclusion was later overturned by a
better-controlled measurement, both are kept, because the corrections are the
useful part.

## 1.1 The recurrent length was tied to the channel count

`T` was `num_feature_maps`, so eight CNN channels meant eight SNN updates.

Measured leaks: dendrite and membrane time constants are both `sigmoid(U(0,4))`,
giving a mean of about 14 steps and 38 at the 90th percentile. At `T = 8` most
neurons never completed one time constant, so the sensory drive never reached
the readout at all.

On real CLEVR gamma the legacy configuration reproduced the README's original
failure exactly: specificity 0.000091, and **one** binary mask shared by all 32
images (pairwise mask IoU 1.0).

**Fix.** `gamma_drive_mode="static"` treats gamma as a constant sensory drive and
runs for `num_time_steps`, decoupled from the channel count. Also added
`gamma_phase_mode` (pooled activations have arbitrary scale, so the drive wraps
around the sine) and `theta_init` (with `zeros` every image starts from an
identical state, so only the drive carries image identity).

## 1.2 The system was not oscillating

With the time axis fixed, the deeper problem appeared:

~~~
step 0    |dtheta| = 0.291
step 15   |dtheta| = 0.032
step 31   |dtheta| = 0.006        the phases have frozen

PLV between every pair  = 1.000
global order parameter  = 0.977
~~~

The network converged to a fixed point **in the globally synchronised state**.
Every oscillator ended up at the same phase, which is the most degenerate
possible state for binding by synchrony: if everything is synchronised with
everything, no group is distinguishable.

Cause: `drive = kappa * sin(gamma - theta)` is an attractor that pins each
phase to `gamma`. With `kappa = 1` against `omega ~ 0.1` the pin wins.

This also explains why the coupling graph did nothing. At a fixed point the
coupling only shifts where the equilibrium sits, so swapping in a spatial or an
image-conditioned graph changed nothing.

**Fix — `freq_gain`.** Let the image set oscillator *frequencies*, not just
phases:

~~~
omega_eff = omega + freq_gain * gamma
~~~

Two oscillators lock when `|domega| < K_eff`, so frequency similarity is what
decides who groups with whom. Concretely at `freq_gain = 2.0`:

~~~
gamma after phase mapping     std 2.52 rad
x freq_gain                   std 4.81 rad per unit time
x dt = 0.1                    std 0.48 rad per step
intrinsic omega               std 0.50 rad per unit time     9.6x smaller

measured: two typical units drift a full cycle apart in ~28 steps
~~~

So the image dominates the frequency by an order of magnitude, and units with
dissimilar features separate within the rollout while similar ones stay locked.

## 1.3 The structural connectivity was inert, then wrong, then replaced

`pearson_cor_sc` builds one fixed `[N, N]` matrix from `|Pearson|` over gamma
samples. Measured on CLEVR at 8x8:

~~~
off-diagonal   mean 0.929   std 0.043   min 0.821
corr(SC, grid distance) = +0.146            far corners couple MORE than neighbours
  distance 1-1.5 (adjacent)  0.952
  distance 4-5               0.918
  distance 8-10 (opposite)   0.963
~~~

Nearly uniform, and anti-spatial: CLEVR backgrounds co-vary perfectly, so the
matrix measures "background-ness" rather than object structure.

But replacing it with a *proper spatial* graph did not help either (Moran's I
0.213 vs 0.232). The real problem is deeper:

> A fixed graph cannot express object membership. Which patches belong together
> is a property of the current image, not of the dataset.

**Fix — `ImageConditionedGraph`** (`graph_generator.py`), one graph per sample:

~~~
gamma [B, C, N] -> projection -> cosine similarity -> top-k -> A [B, N, N]
~~~

Measured requirements, each one tested:

| property | evidence |
| --- | --- |
| image conditioning | a purely spatial graph scored ARI -0.001, chance |
| sparsity | dense graphs collapse into global synchrony as K grows (ARI -0.0002 at K=256) |
| no lower bound | a 0.5 floor on the weights cost ARI 0.027 -> -0.0002 |
| feature-only weights | mixing spatial proximity into the weights hurt (+0.091 vs +0.105) |
| learned | 0.293 learned against 0.056 for a hand-written kernel |

`graph_spatial_decay` adds distance as a prior **on the logits** rather than a
multiplier on the weights. Initialised at 0.861 it converged to 0.8415, so the
hand-picked value was already close to optimal.

## 1.4 The loss had a trivial minimum at global synchrony

`plv_bimodality_loss` pushes pairwise synchrony to 0 or 1. It is zero at a real
partition **and** at PLV == 1 everywhere, and the optimiser reaches the second.
Measured per-element gradients at PLV == 1:

~~~
bimodality              1/M
density target 0.867    0.27/M      loses 4x    -> run collapsed (within 0.9993, between 0.9990)
group-count target      0.49/M      loses 2x    -> run collapsed
density target 0.25     1.5/M       wins 1.5x   -> the only configuration that escaped
~~~

**Fix — `plv_collapse_loss` = `-log(Var(PLV))`.** This removes the minimum
instead of out-weighting it: the loss diverges as the variance goes to zero, so
a constant matrix is not a solution at any weight.

~~~
                     collapse   bimodality
global synchrony       9.21       0.00
all unsynchronised     9.21       0.00
7 blocks (target)      2.12       0.00
~~~

With the barrier in place, the ground-truth-derived density target 0.867 works
and beats the accidental 0.25: ARI 0.1729 -> 0.2922.

## 1.5 The spiking layers were receiving no gradient at all

With the usual loss weights and a phase readout, the dendritic and membrane
layers get **exactly zero** gradient. Verified directly:

~~~
spike_pulse_gain = 0.0    SNN gradient norm = 0.000000
spike_pulse_gain = 0.5    SNN gradient norm = 0.018990
spike_pulse_gain = 1.0    SNN gradient norm = 0.039053
~~~

So in every run up to that point the SNN sat at its random initialisation and
contributed nothing to the result. `spike_pulse_gain` closes the loop by routing
spikes back through the same graph with a phase response curve. It did not
survive training (see failures below), but it established the fact.

## 1.6 The membrane could not fire, and could not carry phase

~~~
membrane std 0.10 - 0.19        vth 0.5      the threshold sits ~4 sigma out
spike rate 0.0002
71.6% of units never fire at all
membrane resting level -0.52, oscillating +/- 0.13
~~~

Three fixes, in the order their effect was measured:

**Membrane leak.** `tau_m ~ U(0,4)` gives a leak near 0.85, which low-passes away
the oscillation carrying the information. `U(-4,0)` gives 0.18, and the
membrane's own binding signal rose 3.8x.

**Dendrite leak.** The same problem, untouched until later. Seed-matched, with
the phase readout provably unchanged:

~~~
                       ARI phase   ARI membrane   ARI spike
seed 0  beta 0.83       0.3358       0.0308        0.0058
seed 0  beta 0.17       0.3358       0.0582        0.0606
seed 1  beta 0.83       0.3542       0.0199        0.0110
seed 1  beta 0.17       0.3542       0.0397        0.0342
~~~

The phase ARI is identical to four decimals within each seed, which is the
control: the dendrite cannot influence the phase path, so the spike improvement
is the dendrite's own effect.

**vth.** Lowering it makes neurons fire but does not by itself help, because a
global threshold on a membrane resting at -0.52 only lets through the units whose
baseline happens to be high. A per-unit adaptive threshold makes everything fire
and *destroys* selectivity (co-firing ratio 6.87 -> 1.2), which showed that the
sparse regime's apparent selectivity was a level code, not a timing code.

## 1.7 The osc_dim axis was reduced in the wrong order

This turned out to be the largest single loss in the whole transduction chain,
and it was found only after removing a confound (see 3.3).

The four `osc_dim` components of one oscillator are **not in phase with each
other** — measured mean deviation from their own unit mean is 1.57 rad, a
quarter cycle. The original code hands `sin(theta)` to the dendrite as
`[B, N, 4]`, which the dendrite mixes and sums, so sines a quarter cycle apart
cancel instead of reinforcing.

~~~
sin(theta.mean(D))    reduce phases first, then one sine     ARI 0.216
sin(theta).mean(D)    sine first, then average               ARI 0.067
~~~

The same information, 3.2x worse, purely from the order of operations.

**Fix — `gate_mode="phase_mean"`.** Reduce the phase axis before the sine and
hand the dendrite one clean oscillation per unit. Seed-matched, four seeds:

~~~
          ARI membrane        ARI spike          fgIoU spike       ARI phase
seed   original  phase_mean  original  phase_mean  orig  phase_m  original  phase_mean
  0     0.0582     0.1568     0.0564     0.0913   0.133   0.142    0.3358     0.2891
  1     0.0397     0.2098     0.0420     0.2190   0.119   0.191    0.3542     0.3461
  2     0.0410     0.1690     0.0321     0.1700   0.118   0.194    0.3265     0.2692
  3     0.0400     0.1913     0.0364     0.1193   0.119   0.180    0.3000     0.3106
 mean   0.045      0.182      0.042      0.150    0.122   0.179    0.329      0.304
                    4.1x                  3.6x            1.5x               -7.6%
~~~

Improvement on 4 of 4 seeds for every spiking readout, and the smallest margin
(1.6x, seed 0 spikes) is still well outside the seed noise. Seed 1 essentially
reaches the transduction ceiling of 0.216.

The phase readout drops on 3 of 4 seeds. The mean cost is comparable to the seed
noise so it is not certain, but the direction is consistent, and `gate_mode` can
reach the loss through the membrane so it is not provably harmless the way the
dendrite leak is. The trade is worth taking here because the spiking path is the
point; if the phase readout alone were the goal, the original gate scores
higher.

## 1.8 Smaller fixes

**Gamma ordering broke patch semantics.** `S2NetClassifier.forward` called
`order_gamma_sequence` unconditionally, permuting oscillator indices that are
fixed spatial patch positions. Now guarded by `gamma_order_enabled`, and calling
it in patch or static mode raises instead of silently corrupting the mapping.

**`_pairwise_cosine` is not centred.** `spike_diversity_loss` and
`structural_consistency_loss` both take a raw cosine over all-positive,
near-constant activity, where any two units score ~1. `spike_diversity` sat
pinned at 0.995-0.998 for a whole 100-epoch run while being 46% of the weighted
loss. Still present; those terms are simply left at weight 0.

**The gate never gates.** `mask_hidden = sigmoid(mask)` maps `[0,1]` to
`[0.5, 0.731]`, so `MembraneLayer`'s `torch.where(g_wave_t == 0, ...)` freeze can
never trigger. `gate_mode="raw"` fixes it; measured effect is small (+26%).

**`DendricLayer` per-neuron loop.** Replaced with one batched matmul, identical
to 3e-8. Matters once the oscillator count grows: N=340, T=16 now runs in 0.15 s
on CPU.

**`sc_generator.py` relative imports.** It was the only module using them, which
broke direct execution. Now tries both.

---

# Part 2 — How the model works now

## 2.1 Forward pass

~~~
gamma_seq [B, C, N]                       precomputed patch features
    |
    | _static_drive:  project C -> osc_dim          (static mode)
    | _to_phase:      map onto a phase range
    v
drive [B, N, D]                           constant sensory drive
    |
    | graph_generator(gamma) -> A [B, N, N]         image-conditioned, sparse
    v
for t in range(num_time_steps):                     ONE interleaved loop
    theta = kuramoto(theta, drive, A, spike_prev)
    mask  = 0.5 * (1 + sin(theta[t - delay].mean(D)))
    feats = sin(theta.mean(D))            phase_mean mode
    h     = dendrite(feats, spike_prev)
    mem, spike = membrane(h, mask)
    |
    v
theta [B, T, N, D]   mem [B, N, T]   spikes [B, N, T]
~~~

The loop is interleaved rather than two sequential loops so that spikes from
step t can act back on the phases at step t+1. Gating only ever looks at
`theta[t]` and `theta[t - phase_delay_steps]`, both already available, so this
reproduces the original two-loop computation exactly when pulse coupling is off.

## 2.2 The Kuramoto update

~~~python
omega_eff = omega + freq_gain * gamma                    # image sets frequency
coupling  = (K / N) * sum_j A_ij * sin(theta_j - theta_i - alpha_ij)
drive     = kappa * sin(gamma - theta)                   # original Eq. (5) term
pulse     = spike_pulse_gain * (A @ spike_prev) * cos(theta)

theta += dt * (omega_eff + coupling + drive + pulse)
~~~

`freq_gain = 0` and `spike_pulse_gain = 0` create **no parameters at all**, so
the default reproduces the original model exactly and old checkpoints load with
`strict=True`.

## 2.3 What t is

In static mode `t` is an Euler integration step of the Kuramoto ODE, `dt = 0.1`.
The measured period is about 93 steps, so a 64-step rollout covers roughly 0.7
of a cycle and `plv_settle=32` leaves about a third of a cycle to measure over.

It is **not** a feature index and **not** an object slot. In the legacy
`sequence` mode `t` *was* the CNN channel index, which is what the original
design intended; that mode is preserved but scores far worse (ARI 0.082 vs
0.336) and is not recommended.

## 2.4 Readout

There is **no trained classifier.** The model produces phases; grouping is
post-hoc:

~~~
theta -> phase_locking_value -> PLV [B, N, N] -> spectral clustering -> labels
mem   -> signal_synchrony    -> correlation   -> spectral clustering -> labels
~~~

Three consequences worth knowing:

- **k comes from ground truth.** The number of clusters is the true object count
  plus background. Real performance without that is lower.
- **The clustering is outside the loss.** It is not differentiable, so gradients
  never reach it; training shapes the pairwise PLV structure, not cluster
  quality.
- **`spike_classifier.py` is unused.** `spike_rhythm`, `spike_interval` and
  `spike_spatial_components` are all bypassed. `_detect_object_groups` still runs
  every forward pass and its output is discarded.

`spike_rhythm` is worth revisiting: grouping units by spike-history similarity is
exactly the intended readout, and spikes only recently began carrying enough
signal for it to have a chance.

## 2.5 Loss terms

Trained with the PLV family; the older activity terms are all at weight 0.

| loss | what it shapes |
| --- | --- |
| `plv_bimodality_loss` | pairwise synchrony towards locked or unlocked |
| `plv_collapse_loss` | barrier against a uniform matrix, `-log(Var)` |
| `plv_group_balance_loss` | mean synchrony towards a target density |
| `plv_spatial_coherence_loss` | synchronised groups should be contiguous |
| `plv_group_count_loss` | effective group count via `N^2 / \|\|PLV\|\|_F^2` |
| `phase_quantization_loss` | phases onto k evenly spaced slots, via `R_k` |
| `phase_spread_loss` | penalise every unit sharing one phase, via `R_1` |

The last three all failed in practice (see 3.2); they are kept because they are
cheap and their failure is informative.

Synchrony can be read from four sources via `--plv-source`: `phase` (PLV),
`alignment` (requires phase difference near zero, not merely constant),
`membrane`, `spikes`.

---

# Part 3 — What failed, and what that ruled out

## 3.1 Making the SNN causally involved

Four attempts, all degrading the result:

| attempt | outcome |
| --- | --- |
| graph feedback from synchrony | runaway to global synchrony |
| training on membrane synchrony | ARI 0.0013, firing driven to zero |
| pulse coupling, gain 0.5 / 1.0 | firing 0.0036 -> 0.74, ARI collapsed to 0.006 |
| per-unit adaptive threshold | co-firing selectivity 6.87 -> 1.2 |

All four share a shape: a feedback signal that is not *differential*. When
firing is dense, `A @ spike` is nearly uniform across units, so the pulse term
becomes a constant times `cos(theta)` — a global synchronising force. Centring
the feedback would remove the uniform component; this was designed but never
tested.

## 3.2 Global summary statistics as losses

`plv_group_count_loss` and the `R1`/`R_k` order-parameter pair optimise
beautifully and destroy the task:

~~~
                       target reached?   ARI
group count = 7            yes          0.061       (best run was 0.293)
R1 low, R7 high            yes          0.011 / 0.042
~~~

`R7` high says the phases lie on a 7-fold grid; it says nothing about *which*
units sit on which grid point. A scalar summary is satisfiable by structures
unrelated to objects. Meanwhile the alignment-trained run reached `R7 = 0.613`
without being asked, as a by-product of a pairwise loss.

**Rule that emerged: constrain pairs, not aggregates.** Binding is a statement
about which pairs belong together.

## 3.3 Measurements that misled, and how

Recording these because several conclusions had to be withdrawn.

**No seed control.** `train_s2net_core.py` had no `manual_seed`. Two runs
differing only in a setting the loss provably cannot see still scored 0.244 and
0.382 on phase ARI, so run-to-run noise is about +/- 0.07. Every single-digit
percent comparison made before the seed was added is unreadable, including
"spatial prior +8.6%" and "membrane fix +12.6%". A `--seed` flag now exists and
the affected comparisons were re-run seed-matched.

**A wrong baseline.** The "input control" was computed through the *learned*
channel projection, so it moved with the checkpoint. The true fixed baseline is
raw gamma at +0.0993, not +0.0321. This retracted a claimed 2.75x amplification.

**Ground-truth masks that were mostly background.** Objects were labelled with a
fixed 1.6-patch radius, covering about 8 patches while a small CLEVR object
occupies 1.5, so "within object" was 80% background and within ~ between by
construction. Replaced with masks derived from CLEVR's colour palette, which
capture 78% of scene objects.

**A duty-cycle confound.** Ablations thresholded at a per-unit 85th percentile,
where *every* path scores ~0.067 including the raw phase, so nothing could be
told apart. This produced the wrong conclusion that the gate destroys 73% of the
signal. Reading continuously showed the gate contributes about +26% and the
`osc_dim` reduction order is the real cost.

**The binding gap does not predict ARI.** Mean within-object minus
between-object synchrony diverged from the task metric four separate times, once
with a *higher* gap than the phase readout and 1/10 the ARI. It is descriptive
only.

## 3.4 Hypotheses tested and ruled out

**"t is a feature axis, one object per t."** Requires CNN channels to isolate
objects. They do not: per-channel IoU against ground truth is 0.079 on average,
with two channels at exactly 0.000. The encoder was trained to reconstruct
images; nothing ever asked it to separate objects. Sequence mode scores 0.082.

**"Objects occupy distinct phase windows."** Testable via
`phase_alignment`, which requires the phase difference to be near zero rather
than merely constant. Training for it does produce separation (alignment ARI
0.054 -> 0.175, object phase separation 0.55 rad, measurable for the first time)
but costs the PLV structure (0.358 -> 0.121) and lands well below the phase
route.

---

# Part 4 — How to run it

## 4.1 Environment

~~~bash
# on the UNC ACM cluster
ssh frontier                                   # ProxyJump through raptor
E=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python   # torch 2.13, cu130
export TRITON_CACHE_DIR=/tmp/tkim1_triton_$GPU  # the NFS home gives stale handles
~~~

Data and artefacts:

~~~
/work/USERS/tkim1/clevr/CLEVR_1k/CLEVR_v1.0/images/train   1000 PNGs
/work/USERS/tkim1/gamma_sequences/                          precomputed gamma + SC
/work/USERS/tkim1/runs/                                     checkpoints, logs
~~~

## 4.2 Stages 1 and 2 are already done

`clevr1k_patch_gamma_seq_k8_grid16.pt` `[1000, 8, 256]` and its SC already exist,
so the input encoder and patch-gamma stages do not need rerunning. If you do
need them:

~~~bash
python -m server_train.train_input_layer \
  --dataset-path <dataset.h5> --hdf5-key image \
  --output-dir "$RUN/input_encoder" --num-images 1000 --image-size 128 \
  --num-kernels 8 --kernel-size 3 --epochs 100 --batch-size 32 --device cuda

python snn_kuramoto_bidirectional/training/train_gamma_initializer.py \
  --gamma-mode patch --image-dir "$IMAGE_DIR" \
  --input-encoder-path "$RUN/input_encoder/input_layer_encoder.pt" \
  --gamma-seq-save-path "$RUN/gamma_seq.pt" --patch-grid-size 16 \
  --feature-normalize standardize --feature-clip 3.0 --device cuda
~~~

Note: gamma generation only accepts `--image-dir`, not HDF5.

## 4.3 Training the core — current best configuration

~~~bash
python snn_kuramoto_bidirectional/training/train_s2net_core.py \
  --gamma-seq-path $G/clevr1k_patch_gamma_seq_k8_grid16.pt \
  --sc-save-path   $RUN/sc.pt \
  --save-path      $RUN/core.pt \
  --epochs 50 --batch-size 8 --lr 1e-3 --device cuda --seed 0 \
  \
  --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 \
  --theta-init gamma --gamma-phase-mode standardize_tanh \
  --freq-gain 2.0 \
  \
  --graph-mode learned --graph-top-k 8 --graph-spatial-decay 0.861 --k 256 \
  \
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 \
  --low-n -4 --high-n 0 --gate-mode phase_mean \
  \
  --plv-source phase \
  --plv-collapse-weight 1.0 --plv-bimodality-weight 1.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 \
  --plv-coherence-weight 0.5 \
  --spike-rate-weight 0 --spike-smooth-weight 0 \
  --spike-diversity-weight 0 --structural-weight 0 \
  \
  --spike-classify-method spatial_components --spike-spatial-grid-size 16 \
  --loss-patch-grid-size 16 --grad-clip-norm 1.0 --verbose
~~~

Roughly 50 minutes for 50 epochs on one TITAN RTX at N=256, T=64, batch 8.

**Set `--k` to `num_regions`** so the effective coupling per oscillator equals
the graph's `coupling_gain`; `graph.effective_coupling(k, N)` reports it.

**Always set `--seed`.** Run-to-run ARI variance is about +/- 0.07, which is
larger than most of the effects worth measuring.

## 4.4 Diagnostics

Before interpreting any mask, check whether the dynamics can bind at all:

~~~bash
python snn_kuramoto_bidirectional/training/diagnose_image_specificity.py \
  --gamma-seq-path $RUN/gamma_seq.pt --sc-path $RUN/sc.pt \
  --output-dir $RUN/diagnostics --num-samples 32 --seed 0 \
  --gamma-drive-mode static --num-time-steps 64 \
  --theta-init gamma --gamma-phase-mode standardize_tanh
~~~

`specificity_ratio` near zero means the core is ignoring the image and no loss
weighting will help. A healthy ratio with identical masks points at the readout.

`image_conditioned_sc.py` is a standalone, torch-only file with the graph, a
minimal Kuramoto step, and a full `diagnose()` report. Run it directly:

~~~bash
python snn_kuramoto_bidirectional/image_conditioned_sc.py
~~~

It prints four sections — dynamics, synchrony, graph, spatial — each with the
reading that makes the number actionable, and it catches every failure mode
above without needing labels:

~~~
|dtheta| < 0.01            frozen at a fixed point, nothing can work
mean PLV > 0.95            global synchrony, no groups distinguishable
nonzeros/row > N/4         dense graph, will collapse as coupling grows
differs between images ~0  fixed graph, cannot express objects
~~~

## 4.5 Visualisation

~~~bash
python snn_kuramoto_bidirectional/training/visualize_s2net_objects.py \
  --image-dir <local CLEVR images> \
  --gamma-seq-path ./runs/gamma_seq.pt --sc-path ./runs/sc.pt \
  --checkpoint-path ./runs/core.pt --output-dir ./runs/visualization \
  --num-samples 20 --patch-grid-size 16 --device cpu \
  --gamma-drive-mode static --num-time-steps 64 \
  --theta-init gamma --gamma-phase-mode standardize_tanh
~~~

Flags must match the checkpoint's training configuration, or the state dict will
not load.

The image directory's sort order must match the order used when gamma was
generated. The script only checks counts, so a mismatched directory silently
pairs the wrong image with each mask.

---

# Part 5 — Configuration reference

Every option below defaults to the original behaviour, and the ones that add
parameters create none at their default, so old checkpoints load with
`strict=True`.

| flag | default | effect |
| --- | --- | --- |
| `--gamma-drive-mode` | `sequence` | `static` decouples the recurrent length from the channel count |
| `--num-time-steps` | 8 | recurrent length in static mode |
| `--gamma-phase-mode` | `none` | `standardize_tanh` maps gamma onto a phase range |
| `--theta-init` | `zeros` | `gamma` starts each image from its own state |
| `--freq-gain` | 0.0 | image sets oscillator frequency; ~2.0 |
| `--spike-pulse-gain` | 0.0 | spikes act back on the phases |
| `--graph-mode` | `static` | `learned` builds one graph per image |
| `--graph-top-k` | 8 | graph sparsity |
| `--graph-spatial-decay` | None | distance prior on the logits; 0.861 |
| `--graph-feedback-strength` | 0.0 | graph tracks the synchrony it produces |
| `--membrane-vth` | 0.5 | spike threshold; 0.06 for the measured membrane scale |
| `--membrane-low-m` / `-high-m` | 0 / 4 | membrane tau range; -4 / 0 shortens the constant |
| `--low-n` / `--high-n` | 0 / 4 | dendrite tau range; -4 / 0 |
| `--gate-mode` | `sigmoid` | `raw` unsquashes the gate; `phase_mean` reduces osc_dim first |
| `--plv-source` | `phase` | `alignment`, `membrane`, `spikes` |
| `--seed` | 0 | required for any comparison |

## Repository map

| path | role |
| --- | --- |
| `input_layer_generator.py` | CNN encoder/decoder and image autoencoder |
| `gamma_initializer.py` | learned gamma encoder and patch initializer |
| `sc_generator.py` | gamma sampling and Pearson SC (superseded by the learned graph) |
| `graph_generator.py` | **image-conditioned learned coupling graph** |
| `image_conditioned_sc.py` | **standalone, torch-only: graph + Kuramoto + diagnostics** |
| `s2net_cls.py` | gamma generator, S2Net core, end-to-end wrapper |
| `kuramoto_layer.py` | graph-aware vector Kuramoto, `freq_gain`, pulse coupling |
| `sinusoidal_gating.py` | phase to spiking drive, `gate_mode` |
| `dendric_layer.py`, `membrane_layer.py` | recurrent spiking computation |
| `loss_function.py` | activity, PLV, alignment and phase-slot objectives |
| `spike_classifier.py` | rhythm, interval, spatial grouping (currently unused) |
| `hierarchical_spike_classifier.py` | multi-level same-time matching (unwired) |
| `training/diagnose_image_specificity.py` | **is the core responding to the image at all** |
| `training/train_s2net_core.py` | core training entry point |
| `training/visualize_s2net_objects.py` | visual reports and diagnostics |
| `results/` | figures |

---

# Part 6 — Where the headroom is

Ranked by evidence, not by appeal.

**1. The feature front-end is frozen and was never asked to separate objects.**
`CNNFeatureEncoder` was trained as a reconstruction autoencoder and has not been
updated since. Per-channel IoU against ground truth averages 0.079 with two
channels at exactly 0.000. Yet training a single linear 8->4 projection on top of
those frozen features moved binding +29%, which suggests the encoder itself has
much more to give. This is the ceiling on everything downstream.

Caveat: training the encoder against a binding loss risks a trivial solution
where all features collapse to identical values. It needs a guard.

**2. A learned readout.** Spectral clustering sits outside the loss, so what is
optimised (pairwise PLV structure) and what is measured (cluster quality) are
different objects.

**3. Resolution.** At 16x16 an object spans 2-3 patches, which limits boundary
precision regardless of method. 32x32 costs 4x.

**4. Estimating k.** Currently taken from ground truth. Needed for a real system;
it will lower the reported numbers.

**Not recommended: more loss-weight tuning.** Measured seed noise is +/- 0.07 and
most loss variants tried fell inside it, while the aggregate-statistic family
failed four times for a structural reason.

## Reproducibility notes

- Checkpoints, experiment images, JSON/CSV summaries and caches are excluded from
  Git.
- Record the command line, preprocessing statistics, thresholds, loss weights,
  **the seed**, and diagnostic summaries for every run.
- Do not call the U-Net probe's decoder output a segmentation; it is untrained
  for CLEVR.
- Do not claim success from one attractive mask. Compare across samples, seeds
  and runs, and prefer ARI over the binding gap.
