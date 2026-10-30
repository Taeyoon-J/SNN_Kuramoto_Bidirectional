"""Linux checks -> immutable preprocessing -> actual smoke -> full registered queue."""
import argparse,fcntl,os,subprocess,sys,time
from pathlib import Path
from . import common as c

def main(archive):
    archive=Path(archive);c.immutable()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(name,command,extra=None):
        c.write(archive/'bootstrap_state.json',dict(stage=name,pid=os.getpid(),time=time.time()))
        with (archive/(name+'.stdout')).open('xb') as out,(archive/(name+'.stderr')).open('xb') as err:
            result=subprocess.run(command,cwd=c.CODE,env=dict(env,**(extra or {})),stdout=out,stderr=err)
        if result.returncode:raise RuntimeError('preserved failure in '+name)
    run('linux_tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','tests','mod_benchmark/test_contract.py'],dict(CUDA_VISIBLE_DEVICES=''))
    run('prepare',[sys.executable,'-u','-B','-m','mod_benchmark.prepare','--archive',str(archive)],dict(CUDA_VISIBLE_DEVICES=''))
    leases=Path('/tmp/kevinswk_sw0113_gpu_leases');leases.mkdir(exist_ok=True);lease=None
    while lease is None:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        for gpu,free in sorted([tuple(map(int,v.split(','))) for v in raw.strip().splitlines()],key=lambda v:-v[1]):
            if free<8192:continue
            candidate=(leases/f'gpu{gpu}.lock').open('a+')
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:candidate.close();continue
            lease=candidate;break
        if lease is None:time.sleep(20)
    try:run('smoke',[sys.executable,'-u','-B','-m','mod_benchmark.smoke','--archive',str(archive)],dict(CUDA_VISIBLE_DEVICES=str(gpu)))
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    run('queue',[sys.executable,'-u','-B','-m','mod_benchmark.queue','--archive',str(archive)])
    c.write(archive/'bootstrap_state.json',dict(stage='coordinator_exited',pid=os.getpid(),time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);main(p.parse_args().archive)
