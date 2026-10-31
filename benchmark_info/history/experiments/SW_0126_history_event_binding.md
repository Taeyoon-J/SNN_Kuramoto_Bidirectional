# SW_0126_history_event_binding

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## design.md

<details>
<summary>design.md</summary>

Source SHA256: `d2725f90d8a515a7a11efbe07d3affff1a59dfb9491fec2f64c6455cb6089c75`

````markdown
# SW0126: history-centered events and imagewise binding

This is a prospective, opt-in mechanism screen. It preserves the production
`MembraneLayer` and `S2NetCore` defaults. A source checkpoint must be strict-
loaded into the original core before `strict_load_source_then_attach` installs
the adapter; the adapter reuses the original `tau_m` parameter and adds only
four trainable component history coefficients.

For each folded batch/component/unit row, the membrane update stays identical
to production. With the registered threshold `0.06`, let
`u_t = m_t - 0.06`, `e_t = H(u_t - a_{t-1})`, and
`a_t = beta_d * a_{t-1} + (1-beta_d) * u_t`, initialized with `a_0=0` and
`beta_d=0.95` for each of the four components. The hard forward event uses the
registered SNN surrogate derivative and the emitted spike is `g_t * e_t`.
Exactly zero gate values preserve both the membrane and adaptation history,
matching the existing membrane's exact-zero preservation condition; no new
gate threshold is introduced. The history state resets whenever the core
starts a new rollout. A truncated-rollout boundary detaches its value while
preserving it, just like the existing membrane and dendritic states.

Because `u-a` is history-centered, it can suppress absolute membrane-level
information that may carry object or amplitude cues. A useful event rate and
nonzero gradients establish only that timing can be learned; they do not show
that event timing helps segmentation. The paired binder screen must measure
that directly. With pulse coupling disabled as in the registered frozen source,
the oscillator phase, carrier, and gate traces should remain unchanged.
Dendritic and membrane traces may change because dendritic updates consume the
previous emitted spike; the history-event binder receives the emitted trace
`S=g*e`, while binary `e` remains available for the recurrence diagnostics.

The paired gate-only diagnostic uses the same event-driven recurrent core and
differs only in the binder's input trace: history-event uses emitted spikes
`S=g*e`;
gate-only uses the continuous gate trace. This separates event timing from
the gate envelope without changing recurrent dynamics.

The planned binder reads the full settled 512-frame trace of the four actual
component events. It must not replace the trace with a time mean and must not
feed theta, RGB, or gamma directly to its classifier. It predicts at most 11
patch groups without ground-truth object counts. It flattens each patch's
four-by-512 trace, projects it to 64 features, then applies 11 learned
imagewise slot queries with three soft-assignment/update iterations. The
registered RGB decoder uses a shared 32-feature coordinate basis and shared
per-slot RGB coefficients. It predicts each slot's RGB field, then
reconstructs through the spike-derived assignment: interpolate the 16x16
assignment maps to native resolution and mix the shared slot RGB predictions
with those maps. An independent decoder mask/alpha head is out of scope because
it could segment the image while ignoring the event representation. Coordinates
may enter the decoder only, identically in both arms. Final labels use the
registered largest-group-as-background and minimum-two-foreground-group rule.

The first data-dependent screen is limited to the first four ordered B16
training batches (64 images) and checks aggregate activity saturation,
per-image/component centered trace variance, and actual gradient credit into
the adaptation, membrane, and dendritic parameter families before a paired
seed-1 pilot. It is a prerequisite, not evidence of improvement. No validation
ground truth may enter prediction or training. The prior SW0125 late-rollout
pilot failed its FG-ARI gate; this mechanism is a new causal hypothesis, not a
repair claim or proof that gating, data scale, or long-horizon credit caused
that failure. If the registered pilot fails, close the branch without tuning
thresholds or expanding seeds.

The opt-in membrane adapter, imagewise binder, and bounded TRAIN-only
activity screen are implemented locally. A conditional seed-1 paired training
runner, evaluator, and owner-aware queue are also implemented locally and have
focused CPU contract tests. The screen and pilot have not been confirmed on
registered server assets; no pilot training or evaluation result is claimed.

````

</details>

## next_direction_review_20261009.md

<details>
<summary>next_direction_review_20261009.md</summary>

Source SHA256: `a58b52cfb3fa1fe5b8515fc59a62cd9e60f4cf074be2e118ec5ff27e252ca460`

````markdown
# Next direction after the failed SW0126 pilot

SW0126 is closed. Both arms learned the RGB objective while their assignments lost instance quality. Neither gradient connectivity nor variable binary events prove an object-binding mechanism. SW0127 first compares frozen checkpoints through the original QCC, including an untouched source97 control whose per-image metrics must replay the archived reference. No SW0126 threshold, learning-rate, or epoch sweep is authorized by this recommendation.

The reviewed next candidate preserves source97 dynamics and the final actual-spike QCC. Each seed uses its own immutable source97 as a confidence-relation teacher. No best-seed teacher selection, external Slot teacher, new RGB decoder, or direct RGB classifier input is introduced. Jointly train the existing encoder, image-conditioned graph, and core against relations preserved under one preregistered global photometric augmentation. This tests a different learning signal; encoder/graph trainability by itself has already failed to guarantee better masks.

Before implementing its training loss, freeze the predicted labels and relation masks on the existing 16 validation IDs1320?1335 for all three sources. A foreground anchor has positives in the same predicted instance with native Q>=.9 and negatives in another predicted group with native Q<=.1. Use distance bins (0,2] and (2,5], and require both classes at each eligible anchor. Only after persisting those masks may GT be read for rejection-only QA. Keep SW0101 acceptance criteria: at least .95 precision per class, 12/16 image coverage, and 50 foreground-relevant decisions per class per bin for every seed. A failed gate stops this recipe; do not tune these thresholds after scoring. Training relations remain entirely GT-free; reserve IDs90640?90959 remain unread.

The native classifier is Q=product_d(max(0,rho_d)), where rho_d is signed centered Pearson correlation of emitted component spikes. It is not a raw signed product. Match the teacher signed component vector on positives rather than forcing every factor to1; match the teacher native Q on negatives. Positive component matching provides credit even when a student factor is nonpositive, while retaining the incumbent temporal pattern. Verify reconstructed rho and Q against the production helper before accepting any update.

If QA passes and SW0127 supports retaining native coding, preregister a seed1 paired pilot: full actual event-times-gate credit versus the identical actual forward with only the event factor detached in the relation loss. Both arms retain actual spikes at inference. Require exact forward parity, finite parameter updates, common data/order/initialization/augmentation, and source/cache hashes before queueing training. Keep useful single-metric outcomes in the tracks required by SELECTION_POLICY; multi-seed or70k promotion requires separately verified evidence.

These are proposals, not completed experiments. Own-source distillation cannot prove independent object discovery, all gating contributions, or scaling. Those require matched retrained ablations and equal-update comparisons of unique dataset sizes after a successful candidate is established. A final success claim still requires all three seed means strictly above the comparable70k Slot baseline and the original fixed evaluation contract.

````

</details>
