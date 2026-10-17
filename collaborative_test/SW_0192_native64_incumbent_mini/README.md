# SW0192 native64 incumbent continuation

New primary objective: native64 FG-ARI>=0.90. Start from the exact learned
SW0186 seed0 checkpoint, retaining its graph/encoder/gates/spikes/QCE direction.
Two TRAIN-only native64 proposals first measure actual backward memory, timing,
five-family credit and same-RNG strict-descent acceptance. This is not a new
performance result. See DESIGN_KO.md for the staged64-image mini comparison and
subsequent full320/three-seed confirmation requirements. Joint70k is deferred
until a large, replicated gain justifies its cost.

Local and server9 tests passed, including both-axis learned-state transfer,
native64 encoder gradient, exact geodesic value/gradient parity across memory
chunks, real Adam acceptance/rollback transactions and nested recurrent-device
allocation. Both actual CLI entrypoints passed. Server CPU admission verified
the exact trained SW0186 seed0 weights, preserved learned CNN, native64 mapping
and empty five-family Adam before CUDA. The exclusive-GPU probe supervisor is
registered and verified live (PID1828532). Its actual worker PID1831863 started
on physical GPU2 and saved the initial native64 boundary. GPU process and actual
argv evidence are in results_archive/start_v1/actual_GPU_start.json. Both proposals
subsequently passed actual-spike/five-family gradient credit, full-forward parity,
strict descent, CNN and Adam checks. Mean B4 proposal time was444.77 seconds;
peak allocated memory5.01GiB. See RESULT_KO.md and results_archive/completed_probe_v1.
Mask quality remains unproven here; SW0193 runs the32-image before/after screen.
