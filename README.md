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

Scored against `clevr_with_masks`, the dataset's own segmentation (3.10), on 100
images at the true cluster count. Foreground ARI is the metric object-discovery
papers report; the all-patch ARI is kept because this repository's history is in
it, and because the two disagree in informative ways.

| | ARI | FG-ARI | fgIoU |
| --- | --- | --- | --- |
| **16x16 grid, top_k 32, spatial decay 0.55** (3 seeds) | — | **0.591** | — |
| 32x32 grid, top_k 256, spatial decay 0.60 | 0.629 | 0.568 | 0.466 |
| 16x16 grid, top_k 32, spatial decay 0.861 | 0.610 | 0.510 | 0.455 |
| features clustered directly, 32x32 | 0.399 | 0.517 | 0.444 |
| features clustered directly, 16x16 | 0.362 | 0.466 | 0.450 |
| same model with the coupling switched off | 0.747 | 0.281 | — |
| chance | 0.000 | -0.001 | 0.057 |

Three things to read from this.

**The coupling is load-bearing.** Switching it off drops foreground ARI to 0.281,
below the features the drive was built from. The grouping is produced by the
dynamics rather than carried in by the input, which is the claim the architecture
rests on and which had never been measured at this metric.

**The margin over the features is small.** 0.510 against 0.466 at 16x16, 0.565
against 0.517 at 32x32 -- about +0.05 either way. A finer grid raises the absolute
score and the control by the same amount, so resolution buys the task, not the
model.

**Against the literature it is early.** Slot Attention and IODINE report
foreground ARI around 0.99 on CLEVR6. The comparison is not clean in several ways
that mostly disfavour this model -- they segment pixels where this segments a
16x16 or 32x32 patch grid, they train on 50-70k images where this trains on 1000,
and CLEVR6 caps scenes at six objects where these average 6.5 -- but the gap is
large and resolution is the largest single term in it.

Two notes on what the numbers are not. They are measured on images the model
trained on, and separately on images it never saw, and they do not differ (3.12).
The oracle cluster count is not an advantage: a well-chosen fixed k matches it
(3.9).

`results/result_overview.png` shows images, ground truth, and both readouts side
by side. `results/report_figure.png` puts the best case next to the median, the
worst, and the distribution. `results/spike_raster.png` shows the mechanism
itself: oscillators sorted by object, each object firing at its own rhythm.

## Does the model earn its place?

The control that matters: cluster the patch features directly, with no
oscillators at all, and compare. Scored against the dataset's own segmentation
(3.10), on 100 held-out images, at the true cluster count:

| | ARI | FG-ARI | fgIoU |
| --- | --- | --- | --- |
| cluster the CNN features directly | 0.362 | 0.466 | 0.450 |
| through Kuramoto + SNN, averaged phases | 0.812 | 0.430 | 0.416 |
| **through Kuramoto + SNN, per-component synchrony** | **0.816** | **0.501** | 0.445 |
| chance | 0.000 | -0.001 | 0.057 |

Read the two ARI columns together. Over all patches the dynamics are far ahead,
and on foreground only they were behind their own input until the readout stopped
averaging the oscillator components (3.11). So the dynamics buy a great deal of
figure/ground and, now, a modest amount of object/object: FG-ARI 0.501 against
0.466, while boundary precision is a tie.

**Is the coupling doing the work, or is the drive passing the input through?**
A constant drive sets each oscillator's frequency and a phase target, and the
question is whether which patches group together comes from that or from the
coupling. Switch the coupling off on the trained checkpoint:

| | ARI | FG-ARI |
| --- | --- | --- |
| trained, coupling on | 0.610 | **0.510** |
| trained, coupling off (drive alone) | 0.747 | **0.281** |
| trained, coupling x0.25 | 0.509 | 0.507 |
| untrained, coupling on | 0.150 | 0.463 |
| features clustered directly | 0.362 | 0.466 |

Without coupling, foreground ARI falls to 0.281 -- below the features the drive
was built from. The drive alone is a worse description of the objects than its own
input, because with nothing to lock them the phases separate by `|w_i - w_j|`,
which turns a graded feature difference into a step. The grouping is made by the
coupling, not carried in by the drive.

The decomposition: features 0.466, untrained dynamics 0.463 (nothing), trained
0.510 (+0.047), coupling removed 0.281 (-0.229). Coupling is necessary and not
sufficient; the margin over the features comes from training the graph and the
frequencies.

Note the two columns disagree in direction. Removing the coupling *raises* ARI
over all patches, to 0.747. Figure and ground separate on the drive alone; only
telling two objects apart needs the dynamics.

An earlier version of this table reported 0.077 against 0.284 and called it
"3.7x". That was measured against colour-reconstructed masks which labelled 46%
of the foreground as background, and it understated the feature control more
than the model. Both numbers were wrong.

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

- **k comes from ground truth,** and it turns out not to matter. The number of
  clusters is the true object count plus background, which sounds like an
  advantage and is not: a fixed k=3 for every image scores slightly *higher*
  (3.9). Sweeping k also showed the phase readout to be nearly flat across
  k=3..8 on the best checkpoint, so this is not a knife edge.
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

## 3.4 Better features did not help, and the metric that chose them was wrong

DINOv2 patch features separate CLEVR objects far better than the current encoder
by the binding measure, so they looked like the obvious fix:

~~~
                              binding    d'
current 8-channel CNN          0.102     0.84
U-Net resnet34, best level     0.134     0.52
DINOv2 ViT-B/14                0.294     1.49
raw RGB (floor)                0.021     0.92
~~~

Trained through the same pipeline they were far worse: ARI 0.045 against 0.346
for the CNN features on the same seed. The drive entering the oscillators was
still better (binding 0.133 against 0.071 after projection and phase mapping), so
the loss is in the dynamics, and `freq_gain` recalibration made it worse rather
than better, in both directions.

The control above explains it. Clustered directly, DINOv2 scores **0.019** against
the CNN's **0.077** — four times worse at the actual task, despite being three
times better by binding. The two measure different things: binding is a mean
difference, while ARI asks whether a consistent partition exists. DINOv2 patches
of one object are on average more alike without forming coherent blocks.

That is the fifth time in this project that the binding gap failed to predict
ARI, and this time the failure was expensive because the gap is what selected
the features. **Screen feature candidates by direct-clustering ARI, not by
binding.**

## 3.5 Training the encoder end to end

Unfreezing `CNNFeatureEncoder` and training it with the core at the core's own
learning rate cost 23% of the phase readout on both seeds. At `--encoder-lr 1e-5`
the damage disappeared but so did any gain: 0.295 against 0.318 frozen. The
encoder can be trained without harm; it just has nothing to learn from a loss
that does not know what an object is.

## 3.6 Reconstruction as a training signal

`slot_reconstruction_loss` softly assigns oscillators to slots by phase, gives
each slot the mean pooled RGB of its patches, and rebuilds every patch from its
slot. On a synthetic test with equal-sized blocks it behaved exactly as intended
(0.19 when the phase groups matched the content, 0.97 when they cut across it).

On CLEVR it halved performance, 0.318 -> 0.138 on both seeds. The synthetic test
hid the flaw: real backgrounds occupy about 240 of 256 patches, so putting
everything in one slot reconstructs almost perfectly, and the loss actively
pushed towards global synchrony. It can be fixed by normalising the error per
slot rather than per patch, but that is untested.

## 3.7 Readout variants that did not help

Reading the phases scores 0.346 while correlating the same one-dimensional signal
scores 0.216, which suggested a third was being lost in how synchrony is measured
rather than in what survives. Whitening each spike train's Fourier magnitude, so
that only the phase spectrum remains, is PLV's amplitude-blindness in the spike
domain. It made things far worse:

~~~
phases, PLV                     0.3461
spikes, plain correlation       0.2190      current
spikes, phase-only synchrony    0.0356      6x worse
membrane, phase-only synchrony  0.0985
~~~

A binary spike train has noise across most of its spectrum, and whitening
amplifies that noise to the same magnitude as the signal. PLV working well on
continuous phases does not transfer to binary events. Plain correlation is the
better spike readout.

`spike_rhythm`, the repository's own readout, could not be scored at all: its
Bron-Kerbosch maximal-clique search does not terminate in ten minutes on a dense
256-node graph. It was written when the grid was 8x8 (64 oscillators) and is not
usable at 16x16 regardless of whether the spikes now carry signal.

## 3.8 Hypotheses tested and ruled out

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

## 3.9 Putting the readout inside the loss

This was headroom item 1: spectral clustering is not differentiable, so training
shaped the pairwise synchrony while the metric scored a partition. Closing that
gap meant a differentiable readout, scored by an objective that judges the
partition itself, with the gradient running back into the dynamics.

**The objective.** The relaxed normalized cut from MinCutPool, chosen over
anything written here because the loss-design record in this repository was 0
for 5:

~~~
L_cut   = -trace(Y^T A Y) / trace(Y^T D Y)
L_ortho = || Y^T Y / ||Y^T Y||_F - I_K / sqrt(K) ||_F
~~~

**Four things had to be fixed before it could train at all.** Each was found by
measurement, and each is worth knowing independently:

*The balance term is wrong for this data.* Background is 94% of the patch grid,
so the true partition is extremely unbalanced. Scoring known partitions against
the PLV matrix, the objective ranked them exactly backwards:

~~~
partition              cut      ortho    total      ARI
ground truth        -0.9457    0.8972   -0.0485   1.0000     worst
spectral, oracle k  -0.9800    0.8194   -0.1606   0.3300
random balanced     -0.3122    0.0029   -0.3093   0.0001     best
~~~

Replacing it with an occupancy floor -- penalise a slot holding less than 1% of
the units, and nothing else -- put the ground truth first for any weight in
[0.25, 1.0].

*The matrix matters more than the readout.* On the PLV matrix no weighting can
work, because spectral clustering's own partition beats the ground truth on both
terms. On the spike synchrony matrix the cut ranks the ground truth first
(-0.9424 against -0.7983). The readout the architecture actually specifies --
units that spike together are one object -- is the one the objective fits.

*A hard-partition check does not validate a soft objective.* The uniform soft
assignment scores cut = -1.0 exactly and gives every slot 1/K of the mass, so it
satisfies the floor and becomes the optimum. Training sat there at row entropy
1.000. A row-entropy penalty removes it without changing how hard partitions
rank.

*Learned slot queries collapse.* With a diffuse first assignment the centroid
update averages every slot onto the data mean, after which the assignment is
exactly uniform, which is a stationary point. Seeding centroids from the data,
farthest-point style, fixes it. Softmax temperature had to drop from 0.5 to 0.05
as well: on 192-step spike trains, 0.5 leaves the assignment flat.

**With all four fixed, it still fails.** Three runs at T=256, seed 0, 40 epochs,
differing only in the objective:

~~~
checkpoint            phases, oracle k   phases, k=3   trained readout
graph_X1 (T=64)            0.3480          0.3499           --
C1 control (no mincut)     0.3267          0.3423           --
X3 mincut, 4 slots         0.1337          0.1723         0.0450
X2 mincut, 8 slots         0.0687          0.0546         0.0339
~~~

More cut pressure is monotonically worse. The objective does not build group
structure, it destroys it. A confound to state plainly: the criterion holds one
synchrony matrix, so moving the cut onto the spikes meant switching
`--plv-source` and zeroing the phase PLV terms. X2 therefore changed two things
at once, and this measurement cannot separate "the cut hurts" from "removing the
phase terms hurts". Either way the configuration is a loss, not a gain.

Frozen-dynamics runs, training only the head on the trained checkpoint, agree
and add one more comparison: the trained readout scored 0.0869 against 0.0590
for spectral clustering at the same k, which looked like a win until plain
k-means on the raw spike trains -- no learning anywhere -- scored 0.1720.

**What came out of it anyway.** Sweeping k, which the diagnostics needed, showed
the oracle cluster count is not an advantage: a fixed k=3 scores 0.3499 against
0.3480. Two README caveats were wrong. Also `spikes -> k-means on trains` is
nearly flat across k=3..8 (0.168-0.182) where spectral clustering on the same
signal collapses from 0.210 to 0.059, so it is the more robust spiking readout.

The reusable part is the check itself: score known partitions -- ground truth,
what the current method returns, random, degenerate -- under any proposed
objective before training on it. It costs one forward pass. Run on soft
candidates as well as hard ones.

## 3.10 The ground truth was wrong, and so was the metric

CLEVR v1.0 ships images and a scene file but no segmentation, so every number in
this repository up to here was scored against masks reconstructed by matching
each object's declared colour to pixels near its stated centre. Measured against
the real thing:

~~~
                              colour masks      dataset segmentation
objects per image                4.92 of 6.39        6.53 of 6.53
foreground                       7.9% of patches     14.8% of patches
~~~

Not only were a quarter of the objects missing; **46% of the foreground area was
labelled background**, so a model that got an object's edge right was scored
wrong for it. The distortion was not uniform: it compressed differences between
configurations, and it penalised the better model more than the weaker feature
control, which is how the headline control came to read 3.7x in the model's
favour when the true relation is the other way on foreground ARI.

The replacement is DeepMind's `clevr_with_masks`, the version object-discovery
papers evaluate on. It is a different render, so the images come from there too
and gamma is regenerated; `training/prepare_clevr_with_masks.py` converts it
without TensorFlow, which the cluster does not have.

**The metric was also not the field's.** Published CLEVR numbers are foreground
ARI, computed over foreground pixels with background excluded. This repository
reported ARI over all patches with background as one more cluster, and then
compared its 0.33 against slot-based methods' 0.9 as though those were the same
quantity. Both are now reported side by side. Against the roughly 0.95 those
methods reach, this model is at 0.50.

## 3.11 Object identity was being squeezed through one number

With the ground truth fixed, the picture split cleanly in two: the dynamics were
far ahead of the feature control over all patches and slightly behind it on
foreground only. They were buying figure/ground and not object/object. Three
candidate causes, and what each measurement said:

**Not the features.** On foreground patches the drive separates objects at
d' = 2.43, and the single best channel reaches 3.56.

**Not the coupling strength.** Sweeping k at fixed top_k=32, foreground ARI runs
0.337, 0.402, 0.434, 0.430, 0.391 for k = 32, 64, 128, 256, 512. There is a peak
and the untested default was already sitting on it.

**The readout.** `phase_locking_value` averaged the osc_dim components before
measuring synchrony, and `gate_mode="phase_mean"` does the same before the SNN,
so which group a patch belongs to was carried by a single number -- which has to
hold seven distinguishable bands when there are 6.5 objects, from features that
have eight dimensions. Measuring synchrony per component and combining after:

~~~
                                    ARI     FG-ARI   fgIoU
mean over components, then PLV     0.812    0.430    0.416
per-component PLV, product         0.816    0.501    0.445
feature control                    0.362    0.466    0.450
~~~

No retraining: this is the same checkpoint read differently. Comparing two
readouts on one checkpoint also has no seed noise in it, and the only other
source of variation, the chaotic trajectory, is worth at most 0.006.

`phase_locking_value(..., combine="product")` does this. The training loss still
uses the averaged form, and `gate_mode` still averages before the SNN, so the
same compression is still in the path the spiking side sees.

## 3.12 Every score had been measured on the training set

`train_tensor` is the whole gamma tensor and evaluation used images 200-299,
which are inside it, so nothing reported here had ever been scored on an image
the model had not trained on. The same checkpoint, on 100 images from each of
three ranges:

~~~
images        FG-ARI   control   status
200-299       0.5144   0.4657    in the training set
1000-1099     0.5249   0.4598    never seen
5000-9999     0.5169   0.4456    never seen
~~~

Nothing degrades; the unseen ranges score marginally higher, and the margin over
the feature control is slightly wider on them. The reason is structural: every
parameter here is indexed by patch position -- `omega`, `kappa`,
`direction_learner`, the graph projection -- and none is indexed by image, so
there is nowhere for an individual scene to be stored. Worth knowing rather than
assuming, and it means the earlier numbers stand.

## 3.13 osc_dim, and 32x32 reopened

**osc_dim was already right.** If object identity rides on the oscillator
components then four of them is a number nobody chose. Sweeping it:

~~~
osc_dim      2       4       8      16
FG-ARI     0.359   0.510   0.502   0.423
~~~

Two is badly short, four and eight are the same, sixteen is worse. The features
driving it have eight dimensions, so there was never more than that to carry, and
capacity stopped being the constraint once the readout stopped averaging (3.11).

**32x32 was closed on a measurement that no longer holds.** Resolution was ruled
out after 32x32 lost at every coupling density -- but every one of those scores
came from the averaged readout, which was hiding object structure. Rerun with
per-component synchrony:

~~~
32x32, top_k        64      128     256
FG-ARI            0.500   0.546   0.565
control (same grid)      0.517 at every top_k
~~~

0.565 is the best number this project has produced, against 0.510 at 16x16. Two
qualifications. The control rises with the grid as well, 0.466 to 0.517, so a
finer grid flatters every method; the model's own margin is +0.044 at 16x16 and
+0.049 at 32x32, which is the same. And foreground ARI is still climbing at
top_k=256, which is 25% of the grid where the 16x16 optimum was 12.5%, so the
top of that curve has not been found.

The wall that closed this direction the first time was never resolution. It was
a [B, N, N, D] tensor in the coupling (Part 1) that put a 1024-oscillator run out
of memory, and a loss that computed terms it had been given weight 0.

## 3.14 A different gamma at each t

The drive is one vector per patch, held constant for the whole rollout, so the
only thing that varies in time is the oscillator state. The proposal was to vary
the drive instead. `gamma_drive_mode="sequence"` already does a version of this --
channel t at step t -- and scores 0.082 against 0.336, but it differs from the
static drive in three ways at once, and nobody had separated them: the drive
varies in time, it is a scalar rather than a vector, and it is not trained with
the core.

`gamma_time_phases` isolates the first. The learned channel projection emits that
many drive vectors per patch and the rollout cycles through them, so the drive is
time-varying while staying a vector and staying trained with the core. The cycle
length is deliberately not the channel count: tying those together forces T to 8,
and with membrane time constants averaging 14 steps that was measured at
specificity 0.000091 with one mask shared by 32 images (1.1).

~~~
gamma_time_phases      0      2      4      8     16
FG-ARI               0.510  0.452  0.491  0.290  0.376
~~~

**The diagnosis was right and the change does not pay.** Making the drive a
vector and training it with the core recovers nearly all of what the sequence
mode lost -- 0.491 against 0.510, which is a tie -- so those two things were
indeed why the static drive works. Varying the drive in time is worth nothing on
top, and lengthening the cycle costs: 8 and 16 are far below the constant drive,
so the sign of the effect is negative.

There is also a reason to expect that. Patches that receive the same temporal
pattern synchronise because they are driven alike, which puts the grouping in the
input rather than in the coupling -- the opposite of what the coupling ablation
above shows the model currently does. The worry that motivated the proposal, that
a constant drive is "just feeding the input in", is answered by that ablation
rather than by varying the drive.

One part of the proposal is untested: replacing the patch feature itself. Today it
is `adaptive_avg_pool2d` over the feature map, which has no parameters at all, so
everything inside a patch collapses to a mean. Whether a learned per-patch encoder
beats that average is a separate question from whether the drive should vary in
time, and it is worth asking with the drive held constant.

## 3.15 Coupling density at 32x32

The 32x32 sweep, completed: foreground ARI 0.500, 0.546, 0.565, 0.528, 0.429 for
top_k 64, 128, 256, 384, 512. It peaks at 256, which is 25% of the grid, against
12.5% at 16x16. 0.565 remains the best result here.

## 3.16 The coupling graph was wiring same-coloured objects together

The margin over clustering the features directly was about +0.05, which raises a
fair question: if a mechanism is doing the binding, why is it barely ahead of
k-means on its own input? Splitting the evaluation by scene content answered it.

CLEVR has eight colours and these scenes average 6.5 objects, so by the pigeonhole
principle **83% of scenes contain two objects of the same colour**. The graph
chooses its edges by feature cosine, and the features are essentially colour, so
those objects look identical to it. Measured on a trained checkpoint:

~~~
mean learned edge weight between patch pairs
  same object                           0.738
  different object, same colour         0.418     54% of the within-object weight
  different object, different colour    0.041
  object to background                  0.004
~~~

With coupling this strong, an edge at 0.418 pulls two separate objects into one
group. That is the dominant error mode, and it is upstream of everything else:
no loss term and no readout can separate two oscillators the graph has tied
together.

**The fix is the spatial prior, which was far too loose.** It decays as
decay^(patch distance), and at 0.861 a pair eight patches apart still keeps 0.31
of its weight. Same-coloured objects are spatially separated, so tightening the
decay cuts exactly those edges. Three paired seeds, scored on 300 images no run
had trained on:

~~~
decay    FG-ARI (3 seeds)              mean    margin over features   edge ratio
0.861    0.527  0.482  0.520          0.510         +0.049              1.8
0.55     0.553  0.603  0.637          0.598         +0.137              3.8
~~~

The two sets of runs do not overlap, every seed improves, and the margin over the
feature control nearly triples. The subset that improves most is the one the
mechanism predicts: scenes with a repeated colour go from +0.058 to +0.168, while
scenes where every object has its own colour gain less.

Overtightening costs. At 0.40 the ratio reaches 5.0 but within-object edge weight
falls from 0.738 to 0.526 and the score drops back, so the prior is trading one
error against the other and 0.55 is where that trade sits at this grid.

The default is now 0.55. The decay is per unit of patch distance, so a finer grid
wants roughly the square root of it to keep the same reach in the image.

Two corrections made along the way, both from scoring too few images. A first pass
on 18 scenes said the model *lost* to feature clustering where colours repeat
(-0.039); on 500 scenes it wins there by +0.084. And the overall margin, long
quoted as +0.044 from 100 images, is +0.064 on 500.

## 3.17 32x32 closes again, for a different reason, and the SNN is now the leak

**The spatial prior does not transfer to 32x32.** Foreground ARI reads 0.568,
0.527, 0.512, 0.555 for decay 0.60, 0.742, 0.86, 0.928 -- non-monotonic, and the
spread sits inside seed noise. The square-root rule, that a grid twice as fine
wants decay^0.5 to keep the same reach, predicted 0.742 and that came second from
last, so the reasoning was wrong.

**And with the prior tuned on both, 16x16 wins.** The earlier claim that 32x32 was
the best configuration compared two untuned priors, 0.928 against 0.861:

~~~
                              FG-ARI   control   margin
16x16, decay 0.55 (3 seeds)   0.591    0.457     +0.134
32x32, decay 0.60 (1 seed)    0.568    0.509     +0.059
~~~

A finer grid raises the feature control more than it raises the model, 0.457 to
0.509, so resolution is buying the task rather than the mechanism. Resolution is
closed again, now for that reason rather than for the memory wall of Part 1.

**The spiking side is where the loss now is.** The architecture reads the grouping
from which units spike together, and that path has never been in the loss. On 150
unseen images:

~~~
                       decay 0.861   decay 0.55
phase readout             0.503         0.591
spike readout             0.389         0.419
feature control           0.457         0.457
~~~

Fixing the graph gained the phases +0.088 and the spikes +0.030, so the fraction
of the phase result the spikes carry fell from 77% to 71%. More to the point, the
phase readout is now well above the feature control while the spike readout is
**below** it. Everything upstream improves and the transduction falls further
behind, which it cannot correct for because the loss reads the phases and the
spiking layers receive no gradient at all.

That is the next thing to work on, and it is the part the architecture actually
claims.

## 3.18 What the spiking path actually loses: the osc_dim components

The phase readout scores 0.630 and the spike readout 0.374, and the obvious
reading is that the spiking layers destroy the structure. Scoring each stage with
the same measure and the same clustering says otherwise:

~~~
theta, per-component PLV              0.6296
theta, component-mean PLV             0.4278     collapsing components: -0.202
sin(theta) per-component, correlation 0.5847     changing the measure:   -0.045
gate output, component-mean           0.3966
dendrite output                       0.4047
membrane, continuous                  0.3969
spikes, binary                        0.3735
feature control                       0.4566
~~~

Two wrong diagnoses, in order. The first was that the spiking layers are the
bottleneck; from the gate onward the whole path costs 0.058, and even
binarisation costs 0.023. The second was that the measure is the bottleneck,
since PLV on phases and correlation on traces are not the same instrument --
correlation reads two units a quarter cycle apart as unrelated where PLV reads
them as locked. That was tested by taking the analytic signal of the membrane and
computing PLV on its instantaneous phase, which is how this is done on real
membrane recordings, and it was **worse**: 0.375 against 0.397.

What actually costs is collapsing the osc_dim components. Averaging them before
measuring loses 0.202, while keeping them and switching to correlation loses
0.045. Correlation is a perfectly good instrument on this signal; four
components reduced to one is not a perfectly good representation.

**And the SNN performs exactly that collapse.** `gate_mode="raw"` hands the
dendrite all osc_dim components, the dendrite reduces them to one signal per
unit, and from the membrane onward there is nothing left to separate. So the
spiking path is not losing the grouping through leaky integration or
thresholding; it is losing it in its first layer, by design, because that layer
was written to produce one neuron per oscillator.

The measured target: per-component correlation on `sin(theta)` reaches 0.585
against the 0.630 of the full phase readout and the 0.457 of the feature control.
If the spiking layers carried the components as separate channels, the spike
readout has that much room.

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

Score a checkpoint against the object masks with every readout side by side,
including the k sweep, which is what showed the oracle cluster count to be
unnecessary:

~~~bash
PYTHONPATH=/export_home/tkim1/tools:$PWD/snn_kuramoto_bidirectional \
python snn_kuramoto_bidirectional/training/evaluate_binding.py \
  --checkpoint $RUN/core.pt \
  --gamma-seq-path $GAMMA/clevr1k_patch_gamma_seq_k8_grid16.pt \
  --masks $RUNS/clevr1k_object_patches.pt \
  --scenes $CLEVR/scenes/CLEVR_train_scenes.json \
  --num-images 100 --skip 200 --num-time-steps 256 --settle 64
~~~

Add `--readout-slots 8 --readout-source signal --readout-temperature 0.05` for a
checkpoint trained with the differentiable readout. This evaluator used to be
rewritten from scratch in `/tmp` every session, which is how the readout
comparison twice got scored against the wrong baseline.

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
| `--image-dir` | None | train end to end from images instead of frozen gamma |
| `--encoder-lr` | `--lr` | separate rate for the encoder; 1e-5 avoids damaging a pretrained one |
| `--slot-reconstruction-weight` | 0.0 | rebuild each patch from its phase slot (see 3.6) |

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
| `loss_function.py` | activity, PLV, alignment, phase-slot and cut objectives |
| `cluster_readout.py` | **differentiable readout: soft k-means over the dynamics (3.9)** |
| `spike_classifier.py` | rhythm, interval, spatial grouping (currently unused) |
| `hierarchical_spike_classifier.py` | multi-level same-time matching (unwired) |
| `training/diagnose_image_specificity.py` | **is the core responding to the image at all** |
| `training/train_s2net_core.py` | core training entry point |
| `training/evaluate_binding.py` | **ARI/fgIoU for every readout, with a k sweep** |
| `training/visualize_s2net_objects.py` | visual reports and diagnostics |
| `results/` | figures |

---

# Part 6 — Where the headroom is

Ranked by evidence, not by appeal.

Two directions have now been tried and closed, so what remains is narrower than
it was.

**Closed: better off-the-shelf features.** DINOv2 and ImageNet U-Net were both
measured; DINOv2 is three times better by binding and four times worse at the
task (3.4). Screen any future candidate by direct-clustering ARI first.

**Closed: training the encoder end to end.** Harmless at a low learning rate,
useless at any rate, because the loss cannot tell it what an object is (3.5).

**Closed: putting the readout inside the loss.** The one architectural gap that
looked most promising, and it was measured end to end (3.9). A differentiable
readout trained on a relaxed normalized cut destroys the binding structure
rather than sharpening it: the phase readout falls from 0.327 to 0.069. What
survives is the diagnostic, which is reusable and cheap.

**3. Resolution.** At 16x16 an object spans 2-3 patches, which limits boundary
precision regardless of method. 32x32 costs 4x.

**4. Estimating k -- already free.** A fixed k=3 matches the oracle on the best
checkpoint (3.9), so the ground-truth cluster count can simply be dropped. This
entry used to say it would lower the reported numbers. It does not.

**5. Reconstruction, repaired.** The one signal that is known to work in the
literature. It failed here for a diagnosable reason (3.6) and normalising per
slot is the obvious fix, but the loss-design record in this project is 0 for 5.

**Not recommended: more loss-weight tuning.** Measured seed noise is +/- 0.07 and
most loss variants tried fell inside it, while every attempt to encode
object-ness as an aggregate statistic was optimised perfectly and scored at
chance.

## The rollout is chaotic, and that is fine

Rewriting the coupling to avoid the pairwise phase tensor (Part 1) changes only
the order of floating-point accumulation. Measured on the same checkpoint and
inputs, the two forms drift apart anyway:

~~~
step   1   max |dtheta| = 7.2e-06
step  64   max |dtheta| = 2.7e-03
step 255   max |dtheta| = 6.45 rad        a full cycle
~~~

A rounding difference grows by six orders of magnitude over 255 steps, so the
system has a positive Lyapunov exponent and individual phase trajectories are
not reproducible across hardware, library versions, or anything else that
changes reduction order.

The readout is another matter. The PLV matrix differs by 2.6e-04 on average
(0.26 at the worst single pair), and the scores move by at most 0.006:

~~~
readout                        before    after
phases, oracle k               0.3480   0.3481
phases, k=3                    0.3499   0.3505
phases, k=8                    0.3190   0.3253
spikes, spectral, oracle k     0.2097   0.2105
spikes, k-means, k=4           0.1819   0.1855
~~~

Against measured seed noise of +/- 0.07 that is an order of magnitude below the
threshold of interest. This is what the model should look like: binding is
frequency locking, which is a property of the attractor and survives
perturbation, while absolute phase is not and does not. It is also consistent
with the readout being PLV, which is invariant to a constant phase offset.

Practically: do not expect two runs to produce identical phases, do not debug
against a stored trajectory, and treat any single-pair PLV value as noise.

## Reproducibility notes

- Checkpoints, experiment images, JSON/CSV summaries and caches are excluded from
  Git.
- Record the command line, preprocessing statistics, thresholds, loss weights,
  **the seed**, and diagnostic summaries for every run.
- Do not call the U-Net probe's decoder output a segmentation; it is untrained
  for CLEVR.
- Do not claim success from one attractive mask. Compare across samples, seeds
  and runs, and prefer ARI over the binding gap.
