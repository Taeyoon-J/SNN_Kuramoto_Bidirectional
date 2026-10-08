# SW0112: nine-parameter shared bounded gate correction

This is the final nine-parameter shared scalar design. It supersedes any earlier four-component residual proposal. The legacy delayed raw gate receives a zero-initialized bounded correction built from the four phase components' sine and cosine. The corrected scalar is used consistently in carrier drive and membrane gating, preserving the folded core and actual event-times-gate trace.

`protocol.json` fixes the formula, zero-init identity, frozen encoder/legacy graph, matched256 updates and full320 endpoints for all three SW0097 seeds. The unchanged legacy mapping remains the default. A scoped opt-in gate adapter must restore its callable binding and preserve explicit checkpoint state.

This differs from SW100's fixed phasor replacement. It does not establish physical rhythm or representation invariance. Require real nonzero-gradient and exact-forward preflights; never tune the residual recipe using evaluation labels.
