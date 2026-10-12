# SW0146 native32 boundary-Q pilot

This is a prospective paired seed1 pilot. It compares an encoder/graph-only arm with an arm that also trains the native core and the 12 phase-integration scalars. Both use the same clean 4096-image source1 TRAIN order, 256 logical-B16 updates, microbatch 4, T=1024, settle=512, and a 64-step live tail. The objective is the fixed SW0145 local boundary-Q MSE plus its source-Q preservation term; its scale and lambda are admitted from the exact SW0145 disposable source0 report. Source dynamics, QCC, and Slot evaluation remain fixed.

The staged run order is two disposable preflights, both arm trainings, both GT-free fixed320 predictions, then one CPU-only score. Training records a descriptive first64 clean TRAIN boundary-stratum diagnostic (`e=0`, `0<e<s`, `e>=s`) for Q, boundary loss, and event occupancy. It does not use ground truth or select checkpoints. Scoring validates and rehashes both candidate predictions, source97 seed1, Slot seed1, and the registered three-seed Slot32 baseline before opening fixed320 target masks.

The queue is private until the upstream admission is sealed at `admission.json`. Run the private `probe_queue.py --dry-run` for a no-write DAG preview, or invoke it without `--dry-run` only after the sealed admission exists and the parent workflow has reviewed the bundle. Actual GPU scheduling is owned by the parent workflow; this draft has not been deployed or launched.

Passing the seed1 gates would only justify further seed work. It would not establish scaling, general performance, or necessity of every gating mechanism.
