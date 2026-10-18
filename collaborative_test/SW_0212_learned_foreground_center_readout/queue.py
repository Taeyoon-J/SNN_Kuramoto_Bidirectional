"""CPU-only dependency queue; never restart or interfere with the parent."""
import json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=Path(__file__).resolve().parent/'results_archive'
PARENT=ROOT/'collaborative_test/SW_0211_spike_anchor_center_assignment/results_archive'


def parent_alive():
    try:
        path=Path('/proc/2269832');stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        argv=[v.decode() for v in (path/'cmdline').read_bytes().split(bytes([0])) if v]
        return fields[0]!='Z' and int(fields[19])==447963319 and 'collaborative_test.SW_0211_spike_anchor_center_assignment.queue' in argv
    except FileNotFoundError:return False


def main():
    import fcntl
    lock=(ARCHIVE/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=ARCHIVE/'queue.json'
    if state.exists():raise FileExistsError('preserve prior queue; do not duplicate')
    def record(status,**fields):
        raw=json.dumps(dict(status=status,pid=os.getpid(),observed_unix=time.time(),**fields),indent=2).encode()
        stage=state.with_suffix('.json.tmp');stage.write_bytes(raw);stage.replace(state)
    try:
        record('waiting_for_211_frozen_results');deadline=time.monotonic()+12*3600
        while not (PARENT/'score_v1/summary.json').exists():
            if not parent_alive():raise RuntimeError('211 supervisor ended without results; no research job restarted')
            prior=json.loads((PARENT/'queue/state.json').read_bytes())
            if prior['status'] in ('failed','stopped_by_signal'):raise RuntimeError('211 terminal: '+str(prior))
            if time.monotonic()>deadline:raise TimeoutError('dependency wait expired; parent untouched')
            time.sleep(30)
        # Scientific imports occur only after the parent predictions exist.
        from . import run
        run.torch.set_num_threads(1);record('CPU_frozen_head_replay_and_readout')
        summary=run.score();record('complete',summary_sha256=run.sha((ARCHIVE/'score_v1/summary.json').read_bytes()))
        print(json.dumps(summary),flush=True)
    except BaseException as exc:record('failed',error=repr(exc));raise


if __name__=='__main__':main()
