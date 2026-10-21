# User-requested shared GPU execution

On 2026-10-20 the user requested that an occupied GPU be used when sufficient
memory remains, and that our worker no longer yield merely because another
user's process appears. `queue_shared_v2` implements that resource policy.

Admission requires free memory for the unchanged worker's 80% device-memory
allocation cap plus 512 MiB headroom. Other process presence and utilization
do not exclude a GPU. Our existing research GPU lease still prevents duplicate
workers from these cooperating queues. After admission, foreign process arrival
does not terminate our worker. No other user's process is signalled.

The original worker, training source fingerprint, loss, dataset, optimizer,
checkpoint history, seed order and single-worker limit are unchanged. The
successor adopts the existing queue state only after the predecessor and any
recorded child are confirmed absent, under the same exclusive queue lock.
Original shutdown records are preserved. The original acquire loop reports an
intentional SIGTERM as an exception; its exact exit is archived separately in
the controlled handoff receipt before the explicit stopped state is supplied.

Six local and Linux tests cover memory admission boundaries, no yield on other
processes, device identity, live predecessor/child rejection, rejection of
unexpected failures, and sequential single-worker execution. Original workers
perform their existing CPU, runtime and actual GPU preflight before resuming.

This changes resource admission, not model performance. GPU time is shared;
free GPU memory does not guarantee exclusive compute throughput.
