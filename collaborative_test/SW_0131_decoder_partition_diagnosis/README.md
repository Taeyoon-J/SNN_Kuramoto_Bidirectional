# SW0131 decoder partition dependence diagnosis

SW0130 passed seeds0/1 but failed seed2 because scrambling patch assignments did not increase RGB reconstruction loss. No training started. This separate finite TRAIN-only diagnostic tests the preserved decoder and source states without any optimizer updates.

Compare native assignments, exactly the registered row-shuffled assignments, and an assignment-independent decoder input using the same image mean content for every slot while retaining patch XY. Inspect both initial and warmed32 decoder states on all three seeds. Preserve per-image effects and parameter-family gradient evidence; no ground-truth masks or validation-score tuning.

Implementation and execution are pending. This diagnosis cannot establish improved segmentation, all gating contributions, or data scaling. See protocol.json for the prospective scope and interpretation limits.

Server launch: eight bounded research files and19 dependencies verified, all six CPU tests passed. Supervisor2264104 and actual children2264248/2264249/2264247 verified live on physical GPUs0/2/1 for seeds0/1/2 respectively. No optimizer updates or new segmentation evaluation. See server_launch_20261009.json.

Completed: all three diagnostics passed execution/evidence validation, with supervisor and all three children absent. Warmed32 native vs assignment-independent image-mean RGB differences relative to native MSE: +0.9688% seed0, +0.1545% seed1, -0.2520% seed2. Seed2 warmed decoder reconstructs better without native partition-specific content. This establishes weakness of this RGB supervision across seeds, not failure of source grouping or state gating. Full raw JSON/log hashes and per-image evidence are preserved in results_archive/completed_20261009/summary.json. No optimizer or new segmentation evaluations were performed.
