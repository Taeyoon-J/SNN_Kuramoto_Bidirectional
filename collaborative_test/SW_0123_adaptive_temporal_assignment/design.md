# SW0123: adaptive temporal neurons with a spike-derived learned classifier

Design registration only. No implementation, deployment, or training is authorized by this file itself.

SW0122 failed its three-seed gate: candidate FG-ARI .767341 versus source .776348 and paired old-only control .766656. SW0121 improved RGB reconstruction through its head but transferred little change to Q; only about 2.1% of its consistency-gradient mass was blocked by the positive-product Jacobian. Neither recipe is expanded. The SW0108 interventions and retention audit instead motivate testing downstream temporal dynamics: events are nearly always on, and mean dendritic/membrane retention coefficients are only about .14–.18. These observations do not establish the cause of poor scaling.

## Hypothesis and three matched arms

Task-driven spatial reconstruction, a classifier that consumes actual spike history, and nonsaturated recurrent neurons may make downstream processing useful for object grouping. The adaptive intervention is a deliberately coarse bundle; its pilot cannot separately attribute retention initialization and threshold adaptation.

All arms receive a new, identical temporal CNN/assignment head and spatial RGB decoder:

1. `legacy_full`: inherited SW0097 neuronal dynamics; classifier input is actual component spikes.
2. `adaptive_full`: inherited model except dendritic and membrane retention are initialized to .9 and the event selector gets the adaptive threshold below. Classifier input remains actual component spikes. This is the only new incumbent candidate.
3. `gate_only_control`: inherited dynamics are executed, but classifier input is the recorded actual scalar gate repeated over four channels. This counterfactual control asks whether downstream processing adds value. It is not an eligible final model. The head receives no direct RGB, gamma, or theta in any arm.

The source for each seed is its whole SW0097 `positive_frozen` checkpoint and the registered shared encoder/preprocessing. Strictly load the original model before attaching opt-in changes. Keep legacy defaults and historical state dictionaries untouched. Require source spike-pulse coupling and graph feedback to be disabled. At initialization, candidate graph, theta, and raw gate must match legacy; changed dendritic/membrane/spike traces are expected.

## Exact TRAIN-only calibration and neuron update

For each seed, forward the first four original ordered B16 TRAIN batches through the **unchanged legacy source** at T64/settle32. Use actual `last_component_out[B,4,256,64]`; do not reinterpret folded `[B*4,256]` as a different layout. Define, for each component d:

`margin_d = clamp_min(mem_legacy[:,d,:,32:64] - .06, 0)`

`kappa_d = 4 * quantile(margin_d.flatten(), .90)`

All four kappa values must be finite and positive; otherwise stop this recipe. Pool over the same 64 images, 256 patches, and 32 settled frames. Freeze this four-value buffer for warmup, training, evaluation, and later ablations. The factor four targets a coarse submaximal duty-cycle regime; it is not an exact activity guarantee or an object-count target. No threshold search or validation calibration is allowed.

Fit one four-channel RMS from the legacy source's actual spikes over exactly these same axes. Values <=1e-8 use denominator 1. All three arms use this same frozen RMS for that seed. Persist ordered image IDs, cache rows, source/encoder/stats/code SHAs and calibration-array SHAs.

Only `adaptive_full` sets every existing `tau_n` and `tau_m` to `logit(.9)` before warmup; they remain trainable. Preserve the dense weights/bias, gamma projection, graph and Kuramoto weights. Add four trainable `raw_b` values with `softplus(raw_b)=.06` at initialization. Adaptation state `a` and the previous binary event flag start at zero for every image rollout and are not persisted in checkpoints.

At step t, on the existing folded `[B*4,N]` stream:

`a_candidate = .9*a_previous + .1*e_previous`

`a_t = where(g_t == 0, a_previous, a_candidate)`

`mem_candidate = alpha*mem_previous + (1-alpha)*h_t - .06*actual_spike_previous`

`mem_t = where(g_t == 0, mem_previous, mem_candidate)`

`threshold_t = softplus(raw_b_d) + kappa_d*a_t`

`e_t = ActFun_adp(mem_t-threshold_t) * 1[g_t>0]`

`actual_spike_t = e_t*g_t`

The component index is d in folded row `b*4+d`; broadcast `[1,4,1]` through `[B,4,N]` before flattening. Do not use flattened spatial repetition. Adaptation uses the binary flag, not weighted spike amplitude, and remains differentiable through the original surrogate. Dendritic feedback still receives the previous **actual weighted spike**. Reset amplitude stays .06; it is not the adaptive event threshold. Do not globally detach the activation function or internal resets.

On the first four settled TRAIN batches, every adaptive component must have pooled binary occupancy strictly between .01 and .80, and at least 10% of its image-patch units must contain both an event and a non-event over 32 frames. These are fixed activation eligibility checks, not training losses. A failing seed stops the recipe; do not change kappa, retention, threshold initialization, or discard that seed.

## Classifier and decoder: one task path through actual spikes

Input is `[B,4,256,T_settled]`, divided by the frozen four-channel RMS. For each patch independently use Conv1d(4,32,kernel5,padding2), GELU, Conv1d(32,64,kernel5,padding2), GELU, then temporal mean pooling to obtain z[64]. The same function accepts 32 training frames and all 512 settled evaluation frames. A shared Linear(64,11) followed by softmax(logits/.1) gives P. There is no occupancy uniformity/count loss, RGB/gamma/theta bypass, or XY input to this classifier.

`slot_latent_k = sum_i(P_ik*z_i) / clamp_min(sum_i(P_ik),1e-8)`

A shared decoder receives only this 64D latent and the normalized center coordinate of a patch, `(2*(col+.5)/16-1, 2*(row+.5)/16-1)`. Use Linear(66,128), GELU, Linear(128,128), GELU, Linear(128,192), sigmoid. Reshape its output as the slot's RGB 8x8 patch; mix slot outputs using P_ik to reconstruct the full 128x128 RGB image. There is no separate decoder mask or raw-RGB latent pooling.

Optimize per-image full-pixel RGB MSE divided by detached `max(mean_channel(var_over_pixels(RGB)),1e-6)`, averaged over images. No old PLV/Q/density priors, weak color-means loss, Q consistency, activity target loss, or lambda is optimized in any arm. This is a new task/objective and classifier branch, not a single-change causal comparison with SW0117. The three internal arms share the objective and classifier.

Fixed eleven logits may still learn global color categories and merge same-appearance instances. The spatial decoder may also reconstruct while underusing P. Neither failure is solved by assumption. Report slot masses and mask-dependent reconstruction; after training, shuffle P rows with fixed permutation seed12301 while holding slot latents/decoded patches fixed. Reconstruction excess must be positive on the registered first-four TRAIN batches; otherwise do not promote. No additional hypotheses or repairs are registered here.

## Matched schedule and preflight

Run all seeds 0/1/2 and all three arms. Each uses its seed's exact original 4096 ordered TRAIN IDs and shuffle117/118/119. Initialize classifier/decoder with seed12300+source_seed, identically across arms. Warm them for 32 B16 batches on the first 512 ordered TRAIN images while encoder/core/new threshold parameters are frozen. Preserve warm head/decoder Adam state into joint training; use independent deep copies for throwaway tests.

Then train 256 B16 updates at T64/settle32. Fresh Adam uses core/graph/new threshold LR3e-5 and encoder LR3e-6; continued classifier/decoder Adam uses LR3e-4. Both optimize the same reconstruction loss, with separate gradient clips of 1 over their own parameters. No gradient calibration coefficient is needed. Save both optimizer states, all new buffers/weights, 256-row history and completion marker last. Do not overwrite attempts or treat partial runs as complete.

Before training require finite loss, actual input/shape checks, finite nonzero reconstruction gradients into encoder, graph/Kuramoto, dendrite, membrane and head/decoder for both full arms, and a disposable real B16 update changing expected trainable groups. Gate-only must have live upstream/head/decoder credit and unchanged unused dendrite/membrane parameters; it is not required to update those unused branches. Numeric gradients establish execution only, not useful gating. Run one actual B16 memory check before fanout. GPUs0–3 are eligible only when genuinely free, with one owner-aware global lease per GPU.

## Fixed metric contract, new readout and decision

Dataset/preprocessing, validation IDs1320–1639, 320 images, modal 16x16 patch GT, and ordinary FG-ARI/foreground-IoU/matched-object-IoU remain unchanged. Do not substitute SW0118's prediction-dependent allowed-label diagnostic. Evaluate B8/T1024/settle512 using the trained encoder and registered frozen preprocessing; regenerate gamma from the same native RGB images.

The **new primary readout** is argmax(P) on 256 patches. Designate the largest predicted slot as BG, resolving equal areas by first occupied patch index; foreground slots with fewer than two patches become BG. Do not split slots with a new spatial heuristic, tune thresholds, or use true object count. Every arm uses this identical readout. Also report original actual-spike Q>=.50/min2/largest-component QCC as secondary; do not silently replace one with the other in tables.

Report every seed/arm. Adaptive primary FG mean must exceed legacy-full, gate-only, and original SW0097 source means, with positive lower bounds of fixed 10000-draw paired-image 95% CIs against all three references; each draw uses common image indices across seeds. At least two seeds must gain over source. Both IoU means must exceed their matched Slot means by .05. Final adaptive TRAIN activation must still pass the fixed occupancy/variation checks and P-shuffle reconstruction must be positive. Failure ends this recipe without coefficient/threshold tuning or seed selection. No 70k training is committed by this design artifact.

After a positive pilot, preregister full320 frozen interventions for coupling, sinusoidal gate, dendritic retention, membrane retention and events, plus time-shuffled classifier input preserving per-unit counts. Use the same learned classifier and fixed neuron calibration. Each outcome is reported: prediction changes alone or nonzero gradients do not establish useful gating. Positive frozen effects require paired endpoint evidence; matched retrained stage ablations are necessary before claiming learning contributions. If downstream stages remain dispensable, the user's all-stage goal remains unmet even if masks improve.

Only after this evidence should the new recipe undergo matched 70k training and a separate unique-data study. That study starts each condition from a common registered initialization, uses nested 2500/10000/70000 TRAIN subsets with exactly 70000 draws/4375 B16 updates per condition, and discloses inherited pretraining/warmup images. It must distinguish additional unique data from additional updates. Existing SW0109 checkpoint recovery is separate legacy scaling evidence. Reserve IDs90640–90959 stay unread until the final frozen confirmation protocol.
