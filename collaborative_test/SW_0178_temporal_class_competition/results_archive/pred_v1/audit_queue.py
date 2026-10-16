"""Run the private CPU verifier only after the existing score queue completes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Data0/kevinswk/patch_v2_sw')
HERE=ROOT/'collaborative_testing/collaborative_private_runtime_20261008/SW0178_evaluation_delivery_20261016'
SCORE=ROOT/'collaborative_test/SW_0178_temporal_class_competition/results_archive/score_queue_v1_20261016/state.json'
STATE=HERE/'audit_state.json'

def write(status,**fields):
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_unix=time.time(),**fields),indent=2))
    os.replace(temporary,STATE)

def main():
    if STATE.exists():raise FileExistsError('preserve existing audit state; inspect its PID')
    write('waiting_for_score')
    while True:
        state=json.loads(SCORE.read_text())
        assert state['experiment']=='SW0178_score_queue_v1'
        if state['status']=='complete':break
        assert state['status'] in ('waiting_for_predictions','scoring_gt_last'),state
        pid=state['supervisor_pid']
        argv=[part.decode() for part in Path('/proc',str(pid),'cmdline').read_bytes().split(bytes([0])) if part]
        assert argv==[sys.executable,'-u','-m','collaborative_test.SW_0178_temporal_class_competition.score_queue']
        time.sleep(30)
    with (HERE/'audit_stdout.json').open('x') as output,(HERE/'audit_stderr.log').open('x') as errors:
        child=subprocess.Popen([sys.executable,str(HERE/'verify_score.py')],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
            stdin=subprocess.DEVNULL,stdout=output,stderr=errors)
        write('verifying_final_score',child_pid=child.pid)
        code=child.wait()
    assert code==0,dict(verifier_returncode=code,stderr=str(HERE/'audit_stderr.log'))
    result=json.loads((HERE/'audit_stdout.json').read_text())
    assert result['status']=='verified'
    (HERE/'final_score_verification.json').write_text(json.dumps(result,indent=2))
    write('complete',score_report_sha256=result['score_report_sha256'])

if __name__=='__main__':
    if STATE.exists():raise FileExistsError('preserve existing audit state; inspect its PID')
    try:main()
    except BaseException as error:
        write('failed',error=repr(error))
        raise
