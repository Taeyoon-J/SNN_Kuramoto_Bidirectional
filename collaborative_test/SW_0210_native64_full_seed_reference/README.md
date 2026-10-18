# SW0210 full native64 independent seed reference

Registered after8 local+8 Linux tests, AST/CLI and actual independent
model/full320 RGB/mini-reference CPU admission. Completes the missing seed1/2 fixed320
reference alongside actual208 warm192 seed0. No209 candidate, training,
checkpoint selection by seed or70k expansion is included. See DESIGN_KO.md.

Supervisors2252470(seed1)/2252471(seed2) verified live. Actual seed2
worker2252587 verified in /proc and nvidia-smi on GPU3; seed1 initially waits.
Each supervisor executes four80-image shards, releasing its exclusive GPU
lease between shards. GPU0..3 eligible; foreign owners retain priority.
Registration and actual process receipts are archived. Read-only waiter2253239
tracks these exact supervisor identities and automatically archives the final
three-seed summary and all eight immutable protocols after GT-last scoring.
No final result, three-seed win or .90 claim exists at registration.

After209 completed and released its lease, seed1 worker2258383 was independently
verified on GPU1 while seed2 worker2252587 continued on GPU3. Exact /proc commands
and NVIDIA observations: results_archive/observed_v1/actual_workers.json.

Completed and independently recomputed: full320/native64 three-seed mean
ours .8486147855/.5925328333/.6720472673 versus comparable Slot
.7869016920/.1982007892/.1952025237. All three strict mean comparisons pass;
the newer FG-ARI>=.90 target remains false. No new training or fresh70k run.
All1920 mask endpoints, bindings, dual metrics and paired CIs were revalidated.
The evaluated models include mask-supervised adaptation. See RESULT_KO.md,
results_archive/completed_v1/completion_audit.json and selected_reference_v1.json.
