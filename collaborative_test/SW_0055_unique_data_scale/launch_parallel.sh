#!/usr/bin/env bash
set -euo pipefail
echo "Deprecated: direct parallel launch is disabled. Use wait_for_idle_and_launch.sh, which schedules only on GPUs 0 and 1." >&2
exec bash /Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0055_unique_data_scale/wait_for_idle_and_launch.sh
