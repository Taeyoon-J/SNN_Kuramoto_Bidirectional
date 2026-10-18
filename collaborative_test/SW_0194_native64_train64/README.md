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

Completed and independently audited: the TRAIN64 candidate scored
**.9001655085 / .8308344896 / .7316582231** on seed0, the same32 development
images. TRAIN8 scored .9185864261 / .8328674431 / .7336862632. The FG change
was-.0184209176 (paired-image95%CI[-.0472437519,.0013289966]). No improvement
was established; the registered gain gate failed. Close this budget/recipe,
retain the incumbent, and use the queued195 generalization diagnostic rather
than a small LR/weight rescue. Do not call this a proven causal overfit result.

All16 proposals and64 unique additional TRAIN exposures completed;12 updates
were accepted and the actual Adam step count was12. The two inherited
proposals/model/Adam/RNG were not repeated. Independent actual-mask scoring
matched within2.23e-16. See RESULT_KO.md and results_archive/score_v1.
Full320/three-seed native64 FG>=.90 remains unverified. The model is supervised
adaptation of an existing trained foundation;64 is not its total data history.
