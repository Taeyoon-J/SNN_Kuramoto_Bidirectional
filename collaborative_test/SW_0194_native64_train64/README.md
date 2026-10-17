# SW0194 native64 TRAIN64 mini continuation

Continue the exact SW0192 trained native64 weights, Adam and RNG for14 more
B4/micro1 proposals: total16 proposals/64 unique TRAIN images. Same graph,
encoder/gates/QCE, same time horizon and acceptance rule. Reuse SW0193 frozen
baseline/TRAIN8 controls and infer only the new candidate on the same32
development images. This implements the mini budget registered before SW0193;
its earlier+.01 gain gate remains false and is not retrospectively relaxed.
More data and update opportunities both change, so this is not an isolated
causal data-scaling proof. No70k training is queued and no FG.90 result is claimed.

Local/server18 tests and both actual CLI entrypoints passed before queue launch.
CPU admission restored the exact native64 learned model and nonempty Adam steps2.
The actual GPU2 worker (PID1849711, parent1849671) then restored CPU/CUDA RNG and
Adam moments with byte-content hashes equal to source SW0192. An independent
read of its initial committed boundary verified matching model/Adam/RNG tensors
and unchanged first-two history. See results_archive/start_v1.

Training and candidate-only prediction/CPU scoring are registered in one
resumable exclusive-owner queue. All implementation/design sources are frozen
while this run is active. No new performance result is available yet.
