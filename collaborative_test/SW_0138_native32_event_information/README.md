# SW0138 native32 event information

Compare actual emitted spike QCC and retained gate QCC on the same frozen trajectories. Native event E comes directly from saved membrane and the actual wrapped threshold; exact S=E*g is required, including zero gate. Recurrence, feedback and reset are unchanged; this is readout information, not individual gate removal.

Three native32 source seeds, fixed320 images, full1024/settle512, B1. Actual labels must exactly reproduce SW0137; six prediction arrays are frozen before GT.

Local and server CPU tests:14/14. Supervisor2699428, seed0 PID2699504 GPU0 and seed2 PID2699498 GPU1 verified live; seed1 queued. No score yet. Publication remains pending scope approval.
