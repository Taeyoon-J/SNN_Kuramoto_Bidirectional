# SW0130: phase-conditioned state integration

Status: prospective design, implementation in progress; no GPU job launched.

## Hypothesis and comparison

The native continuous gate generally never invokes the exact-zero membrane hold. Dendritic and membrane retention removal was neutral in the frozen SW0108 intervention, and native events were almost always on. Test whether making phase gate the **state update** improves object binding, rather than assuming that a gradient or a variable event rate proves usefulness.

Start each arm from the same SW0097 seed-specific core and encoder. Keep the original Kuramoto update, sinusoidal carrier, actual emitted spike, and QCC classifier. Do not subtract temporal DC, replace the classifier, impose an event occupancy target, or feed RGB/GT into prediction. SW0126/127 showed that forced history centering and a new binder lost useful native information.

For each of four folded components add integration coefficients a_D and a_M and threshold log-scale b. All 12 added parameters initialize to zero. Project a_D and a_M to [0,1] after optimizer steps. For native dendritic candidate h_c and previous state h_prev:

    h = h_c + a_D * (u - 1) * (h_c - h_prev)

Use this h in the native membrane candidate m_c, with threshold nu = 0.06 * exp(b), including nu in the previous-spike reset:

    m = m_c + a_M * (u - 1) * (m_c - m_prev)
    m = where(g == 0, m_prev, m)
    E = native_surrogate_step(m - nu)
    S = g * E

The phase arm uses u=g; the same-capacity control uses u=0.5. Both receive the same native rhythmic carrier and emitted-spike multiplier. Constant integration has a nonzero coefficient derivative, unlike a centered modulation that makes the control parameter inert. Zero added parameters must reproduce native forward values on identical gamma at both 64 and 1024 steps. Record actual finite gradients and disposable optimizer changes; algebra alone is insufficient.

## Registered pilot budget and objective

Use the original SW0106 shared RGB decoder, hard-QCC assignment STE, detached 8D slot content, decoder coordinates, and phase + 5*Q + lambda*RGB objective. Both arms receive assignment credit; the causal distinction here is phase versus constant state integration, not SW0129's detached assignment-credit control.

Decoder warmup: 32 batches of 16, source core and encoder frozen. Paired arms share that warmed decoder and its Adam moments. Train all core, graph, encoder, and decoder parameters on the same registered 4096 unique TRAIN IDs, order 117+seed, 256 updates of batch16, T64/settle32. Core/graph LR 3e-5, encoder LR 3e-6, decoder LR 3e-4; the 12 new dimensionless parameters LR 1e-2 in both arms. Clip the unique joint parameter union to 1 and decoder separately to 1. No LR/threshold/lambda sweep after an endpoint failure.

Do not reuse the SW0095-derived SW0106/SW0129 coefficient. Calibrate once from source97 seed0 phase initialization after decoder warmup: lambda=0.25*median(old joint gradient norm / RGB joint gradient norm), first four TRAIN batches, including added parameters in the unique joint union. Freeze this coefficient for all seeds and both arms. Record provenance, exact parameter membership, data IDs and warmup artifact SHA.

## Screens before training

All three seeds must pass TRAIN-only screens before any paired pilot:

- Native zero-initialization parity at T64 and T1024, including actual component output/spikes, Q, hard labels and old loss on identical gamma.
- Warmed-decoder assignment use: mean within-image hard-assignment row scramble RGB excess strictly positive across 64 TRAIN images.
- Finite nonzero reconstruction credit to Q, encoder, graph, dendritic, membrane, and the added parameter families; report component-level zeros explicitly.
- Identical paired initial state and disposable finite update; restored native source remains unchanged.

Initial event saturation is the native state and is recorded rather than rejected using a new sparsity target. A gradient screen does not establish useful segmentation. A failure preserves artifacts and closes the registered recipe without changing the guard.

## Endpoint and claim limits

If every screen passes, run the seed1 phase/constant paired pilot first. Evaluate both on IDs1320..1639, 320 images, native actual-spike QCC, T1024/settle512, threshold0.50, minimum group2 and largest-component background. Continue to seeds0/2 only if phase FG-ARI exceeds both the matched control and source97 with positive paired-image 95% interval lower bounds. Always report all three metrics and retain useful tradeoffs under SELECTION_POLICY.

Record learned coefficients, thresholds, binary event occupancy and actual phase-conditioned state-change magnitude. On a completed phase model separately remove dendritic phase integration, remove membrane phase integration, and force **readout events** on while preserving the original recurrent feedback trajectory. These interventions test reliance of this model, not retrained causal necessity of every gating stage. Neutral interventions prohibit a stage-contribution claim even if scores improve.

Neither the pilot nor a three-seed continuation proves data scaling. Only after a validated mechanism/learning candidate should matched-compute and adequately trained increasing-data experiments be frozen. Final success still requires a complete three-seed fixed-contract comparison against comparable Slot Attention and the reserved independent evaluation; the reserved images remain unread during design and pilot selection.

## Peer incorporation

Peer branch was fetched at 697eeb67dc031325b63153e678cea8cbc602da31 in the preceding cycle. Its position-geometry diagnostic does not establish a trained score, and its matched-compute data increase also degraded performance. Our previous XY graph route did not improve FG-ARI. This experiment addresses the implemented temporal mechanism rather than repeating those appearance/position changes.
