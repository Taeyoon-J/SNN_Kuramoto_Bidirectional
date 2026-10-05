#!/usr/bin/env bash
# Require ten consecutive one-minute samples under all fixed system-load limits.
wait_for_low_load() {
  local required interval consecutive load1 load5 load15 mem_kb sample gate_py
  required="${1:-10}"; interval="${2:-60}"; consecutive=0
  gate_py="${SW0056_GATE_PYTHON:-/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python}"
  [[ "$required" =~ ^[1-9][0-9]*$ && "$interval" =~ ^[1-9][0-9]*$ ]] || return 2
  while (( consecutive < required )); do
    if [[ ! -r /proc/loadavg || ! -r /proc/meminfo ]]; then
      echo "Cannot read /proc load/memory telemetry; no work will start" >&2
      return 2
    fi
    read -r load1 load5 load15 _ < /proc/loadavg
    mem_kb="$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)"
    if [[ -z "$mem_kb" ]]; then echo "MemAvailable missing; no work will start" >&2; return 2; fi
    if "$gate_py" "$(dirname "${BASH_SOURCE[0]}")/load_policy.py" "$load1" "$load5" "$load15" "$mem_kb"; then
      consecutive=$((consecutive + 1))
      sample="PASS $consecutive/$required"
    else
      consecutive=0
      sample="high load (streak reset)"
    fi
    echo "$(date -u +%FT%TZ) load1=$load1 load5=$load5 load15=$load15 MemAvailable_kB=$mem_kb $sample"
    (( consecutive >= required )) || sleep "$interval"
  done
}
