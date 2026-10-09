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
the oscillator phase and dendritic traces should remain unchanged; membrane
and emitted-event traces are allowed to change under this mechanism.

The paired gate-only diagnostic uses the same event-driven recurrent core and
differs only in the binder's input trace: history-event uses emitted events;
gate-only uses the continuous gate trace. This separates event timing from
the gate envelope without changing recurrent dynamics.

The planned binder reads the full settled 512-frame trace of the four actual
component events. It must not replace the trace with a time mean and must not
feed theta, RGB, or gamma directly to its classifier. It predicts at most 11
patch groups without ground-truth object counts. Any shared RGB decoder must
reconstruct through the spike-derived assignment: interpolate the 16x16
assignment maps to native resolution and mix the shared slot RGB predictions
with those maps. An independent decoder mask/alpha head is out of scope because
it could segment the image while ignoring the event representation. Coordinates
may enter the decoder only, identically in both arms. Final labels use the
registered largest-group-as-background and minimum-two-foreground-group rule.

The first data-dependent screen is limited to 64 training images and checks
activity saturation, centered trace variance, and actual gradient credit into
the adaptation, membrane, and dendritic parameter families before a paired
seed-1 pilot. It is a prerequisite, not evidence of improvement. No validation
ground truth may enter prediction or training. The prior SW0125 late-rollout
pilot failed its FG-ARI gate; this mechanism is a new causal hypothesis, not a
repair claim or proof that gating, data scale, or long-horizon credit caused
that failure. If the registered pilot fails, close the branch without tuning
thresholds or expanding seeds.

The current code contains only the membrane adapter and focused unit tests.
The imagewise binder, TRAIN-only activity screen, full runner, and evaluation
remain unimplemented pending review of this mechanism surface.
