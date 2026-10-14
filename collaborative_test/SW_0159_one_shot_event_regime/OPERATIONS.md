# SW0159 execution notes

Keep goal active until the user asks to stop. User instruction (2026-10-14): maintain SSH keepalive and continue bounded remote status checks while tests run; do not send routine status-check messages. Observation timeouts do not establish job termination. Re-poll the same PID and inspect its command line before changing work.

Use SSH ServerAliveInterval=30 and ServerAliveCountMax=3. A disconnected client does not mean a detached server queue stopped. Never restart from an expired observer alone.

Preflight queue uses shared owner leases and a final GPU owner/UUID/memory check. Complete scientific rejection (valid report, return code 2) is terminal and blocks full training. It is not a retryable runtime error.
