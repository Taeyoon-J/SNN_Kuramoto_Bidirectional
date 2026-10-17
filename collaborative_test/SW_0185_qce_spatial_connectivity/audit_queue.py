"""CPU-only final audit waiting on a verified live SW0185 score producer."""
from __future__ import annotations
import json
import hashlib
import os
import signal
import threading
import time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from collaborative_test.SW_0185_qce_spatial_connectivity import completion_audit as audit

HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'results_archive/audit_queue_20261016'
OUTPUT=HERE/'results_archive/audit_v1/completion_audit.json'
STOP=threading.Event()
MODULE='collaborative_test.SW_0185_qce_spatial_connectivity.audit_queue'
SCORE_MODULE='collaborative_test.SW_0185_qce_spatial_connectivity.score_queue'

def producer_ready(state,proc):
    if state.get('experiment')!='SW0185_score_queue':raise ValueError('score identity mismatch')
    if state.get('status')=='complete':return True
    if state.get('status') not in ('waiting_for_predictions','scoring_gt_last'):raise RuntimeError('score producer terminal')
    argv=list((proc or {}).get('argv') or [])
    if (not proc or proc.get('state')=='Z' or argv.count('-m')!=1 or
        argv[argv.index('-m')+1:][:1]!=[SCORE_MODULE] or '--worker' not in argv):
        raise RuntimeError('score producer is not currently live')
    return False

def run():
    from collaborative_test.SW_0185_qce_spatial_connectivity import score_queue
    owner=score_queue.score.pe.train._module(score_queue.OWNER_MODULE)
    if (RUNTIME/'started.json').exists() or OUTPUT.exists():raise FileExistsError('preserving existing audit')
    audit.write_once(RUNTIME/'started.json',dict(experiment='SW0185_audit_queue',status='waiting_for_score',
        supervisor_pid=os.getpid(),observed_unix=time.time(),validation_data_read=False))
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    try:
        while not STOP.is_set():
            path=score_queue._state_path();state=json.loads(path.read_text())
            proc=owner._proc_record(state.get('supervisor_pid'))
            if producer_ready(state,proc):break
            STOP.wait(10)
        if STOP.is_set():raise RuntimeError('audit stopped before score completion')
        score_path=Path(state['output'])
        if score_path.resolve()!=score_queue.score.SCORE_PATH.resolve():raise ValueError('unexpected score output')
        if hashlib.sha256(score_path.read_bytes()).hexdigest()!=state['score_sha256']:raise ValueError('score receipt SHA mismatch')
        report=audit.audit_completed();audit.write_once(OUTPUT,report)
        audit.write_once(RUNTIME/'terminal.json',dict(experiment='SW0185_audit_queue',status='complete',
            supervisor_pid=os.getpid(),score_sha256=report['score_sha256'],output=str(OUTPUT),
            audit_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()))
        return report
    except BaseException as exc:
        audit.write_once(RUNTIME/'terminal.json',dict(experiment='SW0185_audit_queue',status='failed',
            supervisor_pid=os.getpid(),error=repr(exc),observed_unix=time.time()))
        raise

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    args=parser.parse_args(argv)
    if not args.worker:parser.error('audit is an owned CPU worker')
    result=run();print(json.dumps(dict(status=result['status'],score_sha256=result['score_sha256'])),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
