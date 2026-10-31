# SW_0146_native32_boundary_q_pilot

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## conditional_membrane_efficacy_review.md

<details>
<summary>conditional_membrane_efficacy_review.md</summary>

Source SHA256: `af93c31f207f22dbaa6745cda9a4d36ebb4210e82d51fdb3eee5f376b68bcc97`

````markdown
# Conditional successor: source-preserving membrane-conditioned spike efficacy

Prospective design review. SW146 is still running; do not modify, stop, restart, or extend it. Prepare the successor implementation and CPU checks locally under the user's existing research authorization. Deployment and GPU training are conditional on the completed SW146 evidence selecting this hypothesis; this is a scientific admission condition, not a request for additional user permission.

## Evidence and selection

SW142's learned P remains far below source QCC; within-color P permutation even lowers completion loss. Another RGB decoder or global slot-count adjustment is not recommended. SW144's cached-source visible-edge diagnostic reports Q/soft-boundary correlation .6295, which motivates the current boundary experiment but is not object-boundary precision. SW143 completed the independent graph correction comparison and failed its registered FG confidence gates.

SW146 joint preflight reports settled event mean occupancy 1.0 for every component in each of four microbatches of the first16 TRAIN images. All-step means are slightly below1 in some components. These reported means do not certify exact event-tensor identity or global saturation. Integration-scalar gradient norm is 5.2556742e-8 versus native-core .0015437614; all parameter groups changed. Finite credit therefore does not establish a useful downstream forward code.

Select this successor only after SW146 completes and fails promotion, with its fixed-TRAIN diagnostics still consistent with gate-dominated output and weak downstream task credit. Prefer it if B improves on the same TRAIN inputs but final grouping does not. If the completed error analysis instead shows that B creates shading fragmentation or foreground/background damage, do not automatically make that proxy more expressive: close this conditional branch pending an objective decision. SW140 already found source membrane/event ablations nearly neutral; it does not prove their retrained necessity or impossibility.

## One architecture change

Preserve original source dynamics and the strong actual-spike QCC. At component d and time t, let

`v_d = .06 exp(b_d)`

`m_d(t) = membrane_d(t) - v_d`

`phi_d(t) = tanh((m_d(t) - mu_d) / s_d)`

`c_d = .5 tanh(rho_d)`

`S_native(t) = E(t) * g(t)`

`S(t) = S_native(t) + c_d * S_native(t) * phi_d(t)`.

There are four new shared rho parameters, initialized to0. E remains the native binary event with the original MG surrogate. The emitted value is therefore an event-gated spike with membrane-dependent efficacy, bounded between .5 and1.5 times its native amplitude. It is not a claim of a strictly binary readout or improved event-selection necessity.

Use this actual S for both recurrent previous-spike/reset input and the final QCC trace. Do not change the threshold rule, carrier, Kuramoto coupling, dendritic recurrence, or state reset. At rho=0 the source arithmetic/output must reproduce the native32 foundation exactly. Keep the native multiply first and add the zero residual; match production output layout. This avoids SW126's immediate destructive history/DC recoding. It also restores a potential component-specific membrane signal that near-always-on E currently suppresses, without passing theta/gamma/RGB to the classifier.

## Matched arms and interpretation

- **Live:** phi carries its normal membrane gradient.
- **Detached-credit control:** use `phi.detach()` only in the added efficacy term. Native E surrogate, internal recurrent states, reset, S_native, and rho gradients remain live.

Both arms have the same forward formula, parameter count, initial state, loss, and optimizer. From an identical checkpoint/input their forward values are the same; learning trajectories may diverge. This comparison tests credit through the new membrane-amplitude branch, not causal forward necessity of membrane or binary events. Such necessity requires later explicit forward interventions and, where warranted, matched retraining.

## TRAIN calibration and admission

For seed1, fit four mu_d values and four positive s_d values from the native source's membrane margins over the first64 clean registered TRAIN images, all1024 nodes and settled512 frames. mu is the componentwise mean; s is population standard deviation. Float64 accumulation, no GT, no centering of recurrent states, no calibration of an event-rate target. If any s is zero/nonfinite, stop; do not introduce a new epsilon or choose another subset. Freeze these constants for both arms. Future seeds apply the identical source-only fitting rule, with all values disclosed.

Preserve SW146's exact B, source-Q A, boundary scale .0007118072634665487 and lambda_A .19366687877861335, bound to its immutable SW145 admission. No lambda recalibration is needed because this is a source-zero-init architecture intervention at the same initial objective. It remains a fixed empirical coefficient, not an assurance of later gradient balance. Never load the SW145 disposable update or completed SW146 training weights as the new initial source.

Preflight on fixed clean TRAIN B4 inputs must prove source graph and all foundation traces/Q parity at rho0 before any update. First disposable update requires finite nonzero rho gradient and an actual rho change. The added membrane derivative is expected to be zero at rho0. On the second update, explicitly measure finite nonzero credit through phi to live-tail membrane and the native downstream parameter path; show that the added path is absent in the detached control while native MG credit remains. Validate the real emitted S, recurrent use, and positive bounded efficacy. If rho never leaves0 or the new live path is inert, close without training. Nonzero credit is feasibility only, not a mask-quality or gating result.

## Bounded pilot and endpoint

Fresh native32 source97 seed1, clean registered4096 TRAIN images, identical order, fresh Adam, 256 logical B16 updates per arm, microB4, full1024/settle512/live-tail64. Prepare live static encoder/graph before burn-in and detach only at the registered boundary. Use SW146 learning rates/clip/projection; rho LR1e-3, shared across arms. No head, decoder, mask corruption, capacity sweep, forced sparsity, or validation selection. Resource-screen one real logical update; runtime/memory should be measured rather than assumed, although this adds no cubic operator or pixel decoder.

Freeze both fixed320 native32 QCC prediction archives before GT. Require live FG to exceed source1 and detached control by .01, with both paired95% CI lower bounds positive; both IoUs must exceed the registered Slot32 means by .05. Report all metrics, same-TRAIN objective strata, actual event changes, rho, and stage contributions regardless of outcome. Failure closes this recipe; no rho/LR/normalization/threshold sweep. Success permits only the preregistered seeds0/2 replication, then all3 Slot comparison and a separate fresh native32 70k/scaling protocol. The reserve remains unread until final freeze.

This is a substantive coding-path intervention, not proof of object boundaries, all-gate usefulness, or scaling. Its rationale is measured downstream underuse plus preservation of the incumbent source, not an assertion that adding membrane information must improve FG-ARI.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `1292e87a64e6414af49f400f6dae6b6526e677ff2bec9bc600bda89548bc859d`

````markdown
# SW0146 native32 boundary-Q pilot

This is a prospective paired seed1 pilot. It compares an encoder/graph-only arm with an arm that also trains the native core and the 12 phase-integration scalars. Both use the same clean 4096-image source1 TRAIN order, 256 logical-B16 updates, microbatch 4, T=1024, settle=512, and a 64-step live tail. The objective is the fixed SW0145 local boundary-Q MSE plus its source-Q preservation term; its scale and lambda are admitted from the exact SW0145 disposable source0 report. Source dynamics, QCC, and Slot evaluation remain fixed.

The staged run order is two disposable preflights, both arm trainings, both GT-free fixed320 predictions, then one CPU-only score. Training records a descriptive first64 clean TRAIN boundary-stratum diagnostic (`e=0`, `0<e<s`, `e>=s`) for Q, boundary loss, and event occupancy. It does not use ground truth or select checkpoints. Scoring validates and rehashes both candidate predictions, source97 seed1, Slot seed1, and the registered three-seed Slot32 baseline before opening fixed320 target masks.

The queue is private until the upstream admission is sealed at `admission.json`. Run the private `probe_queue.py --dry-run` for a no-write DAG preview, or invoke it without `--dry-run` only after the sealed admission exists and the parent workflow has reviewed the bundle. Actual GPU scheduling is owned by the parent workflow; this draft has not been deployed or launched.

Passing the seed1 gates would only justify further seed work. It would not establish scaling, general performance, or necessity of every gating mechanism.

````

</details>
