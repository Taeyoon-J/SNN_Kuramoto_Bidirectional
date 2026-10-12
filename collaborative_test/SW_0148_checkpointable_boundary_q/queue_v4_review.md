SW0148 v4 recovery queue

The reviewed queue preserves the original five-stage training/prediction/scoring DAG and shared inherited GPU leases. Train commands use the sealed parity adapter; prediction and score commands remain original. Only the failed joint arm may start immediately. The original encoder_graph supervisor, child identity and process groups must finish before v4 inspects/adopts/resumes that arm. Permission failures do not mean missing processes; live predecessor helpers are waited on. No old process is signaled.

The adapter's six regression tests and the queue's seventeen focused tests passed on Frontier CPU with zero skips. Both original model/loss/data and original eight source files remain unchanged. Launch receipt verifies supervisor 779598 and records the old encoder still active. At first observation joint child 779656 was live; actual intermediate-update CUDA resume and scientific evaluation remained unverified.
