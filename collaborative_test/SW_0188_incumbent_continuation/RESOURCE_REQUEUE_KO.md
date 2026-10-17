# Resource-only amendment: seed1 fresh requeue

Registered before execution and before any SW0188 validation scoring.
Seeds0/2 have completed training. Seed1 yielded GPU3 to a foreign job and its
unchanged supervisor has waited for that physical device for over an hour.
The same-runtime checkpoint contract binds that partial run to GPU3, so it
cannot use GPU2 after the other queued work releases it.

Fresh-requeue only seed1 from the exact same completed SW0186 seed1 weights and
empty five-group Adam, retaining the original code, dataset/order, loss, RNG
proposal schedule, learning rates,64-proposal budget and queue fingerprint.
The old11 completed proposals/6 accepted updates and full checkpoint/log/state
files will be preserved under the private runtime archive. Those11 proposals
are excluded from delivered model weights and reported as additional compute.
This is a resource-driven resubmission, not a failed-model retry or a choice
based on validation scores. No SW0188 score or validation mask selects it.

Delivered models still use64 proposals each (192 across three seeds); the old11
completed proposals are separate overhead, for203 completed proposals if all
three delivered runs finish. Do not count preserved partial updates as new
training exposures in the delivered seed1 model.

The read-only dry run verified exact current PID/argv/start ticks and our UID,
no child worker, foreign ownership on GPU3, the preserved11-proposal boundary,
all original model/test/DESIGN source hashes and the same CPU/queue identity.
Operational README edits are excluded from immutable model-source checks.
Old queue fingerprint: `4921773107d51bcfa9a4ebc674513514c90a596724071940d42de7f40886eafe`.

Execution briefly pauses only our waiting seed1 prediction parent, stops only
our waiting training parent, verifies the lock is released, moves the partial
files within the checked research workspace without deleting them, registers
one new parent and resumes the same prediction parent. Original launch and GPU
admission receipts remain historical evidence; the replacement needs a fresh
first-two-GPU admission. Never signal or share the foreign process.

Models0/2 and all other queues remain unchanged. A fresh queue may use any
exclusively idle GPU under the existing admission policy. If an observation
times out, inspect the same replacement PID and journal; do not launch again.
