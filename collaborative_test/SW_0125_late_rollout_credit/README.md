# SW0125: late-rollout credit

This prospective recipe tests whether a 64-frame differentiable tail can
provide useful gradient credit while retaining the original 1024-frame S2Net
forward trajectory and the full 512-frame settled objective window. The prefix
is evaluated without gradients, but its phase and recurrent neuron states are
preserved; static graph and oscillator drive are prepared once live and reused
by the tail. This is truncated BPTT, not full-rollout BPTT.

`late_rollout.py` is the rollout helper. `verify_foundation.py` provides a
read-only actual-data check for one matched B16 training batch per seed; it
records zero optimizer updates and uses no GT. `run.py` registers a 4-batch
same-input preflight and 256-update, full three-seed pilot. `evaluate.py` uses
the unchanged fixed QCC evaluator, and `dispatcher.py` runs the nine stage tasks
only on exclusively leased idle GPUs without automatic retries.

The preflight and pipeline have only local CPU/API validation so far. Real
1024-frame three-seed parity, GPU gradient checks, training, and endpoint
evaluations remain pending. No performance conclusion is registered here.
