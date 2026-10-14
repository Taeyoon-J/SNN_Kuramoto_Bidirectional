# SW0159 one-shot event regime

This isolated candidate tests one fixed initialization of the existing native `b` parameters using the pinned SW0157 source-T TRAIN margin calibration. The physical threshold is `0.5 * source margin RMS`; native `b` is `log(threshold / 0.06)`. The control keeps `b=0`.

Both paths use the original SW0134 rollout. No SW0158 amplitude/eta code is installed. Since `b` participates in both event threshold and native reset subtraction, this is a joint regime change, not an isolated threshold-only intervention.

The bounded preflight checks exact control parity, finite candidate dynamics, actual `S = E*g`, an event change with gate-positive transitions, and two disposable TRAIN-mask updates. It uses the first four ordered source TRAIN images and positions 2048–2051 for the two updates. It does not inspect validation/test masks or claim performance. No threshold, learning-rate, or seed sweep is registered.
