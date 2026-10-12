# Faster exclusive GPU admission

The live v2 queue observed GPU2 idle at 4 MiB, but a foreign workload entered before the 60-second admission window completed. That window was an operational conservative policy, not a scientific requirement or protection against users who do not honor our leases.

V3 changes only the queue docstring, runtime namespace, queue version, and idle window from 60 to 5 seconds. It still requires two idle observations and a final recheck, keeps the exclusive inherited lease, checks foreign owners while running, and pins checkpoint resumes to the same physical GPU. Model, encoder, graph, loss, training budget, source bindings, runtime signature, and evaluation are unchanged. A shorter window increases launch opportunities and may also increase foreign collisions; only our child is stopped on collision and a valid committed boundary is preserved.

The eleven Linux CPU tests passed without skips. They cover the existing v2 admission cases plus the actual stable-idle timing loop and rejection of a newly observed owner. V2's waiting supervisor was identity-checked and replaced before any training child started. V3 supervisor PID551842 was verified live. Evidence is under `results_archive/server_start_v3_20261012/`. Neither a native CUDA checkpoint-resume result nor a new segmentation score exists at launch.

The current entry point is `python -m collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v3`; observe its existing process/state before any launch. Earlier supervisors and their artifacts remain preserved.
