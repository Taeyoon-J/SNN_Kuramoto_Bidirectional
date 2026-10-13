"""Wait for the existing diagnostic, then resume the sealed 151 reservation once."""
from pathlib import Path
import fcntl, hashlib, json, os, time

os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/Data0/kevinswk/patch_v2_sw')
runtime=root/'collaborative_test/SW_0151_source_preserving_event_history/results_archive/prerequisite_resume_20261013'
diagnostic=root/'collaborative_test/SW_0150_spike_spatial_mixture/results_archive/diagnosis_queue_20261012/state.json'

def write_state(status, **fields):
    payload=dict(status=status, supervisor_pid=os.getpid(), updated=time.time(), training_launched=False, **fields)
    temporary=runtime/'state.tmp'
    temporary.write_text(json.dumps(payload,indent=2)+'\n')
    os.replace(temporary,runtime/'state.json')

with (runtime/'watch.lock').open('a+b') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_state('waiting_existing_diagnostic')
    while True:
        state=json.loads(diagnostic.read_text())
        status=state.get('status')
        if status=='passed':
            break
        if status in ('failed','cancelled','interrupted_foreign_owner') or status.endswith('_failed') or status.startswith('blocked_'):
            write_state('diagnostic_terminal_requires_review',diagnostic_status=status)
            raise SystemExit(1)
        time.sleep(30)
    from collaborative_test.SW_0150_spike_spatial_mixture.reconcile_score_queue import requeue_blocked_prerequisite
    write_state('validating_completed_prerequisites')
    receipt=requeue_blocked_prerequisite('sw151_preflight',archive_dir=runtime/'archived_blocked_reservation')
    write_state('original_preflight_queue_relaunched',receipt=receipt)
    print(json.dumps(receipt),flush=True)
