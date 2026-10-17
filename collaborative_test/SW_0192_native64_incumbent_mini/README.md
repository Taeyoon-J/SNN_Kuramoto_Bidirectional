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
registered and verified live (PID1828532); it waits for a free GPU. No native64
worker start or quality improvement is claimed yet.
