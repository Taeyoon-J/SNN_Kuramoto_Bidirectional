"""R3: preflight9 objective controls, screen3x3, extend one validation winner3seeds."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ARMS=['context_rgb','context_flat_edge','context_wide_rgb']
def write(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def execute(root,code,tasks,phase):
 running={};done=[]
 while tasks or running:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  free={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()};used={r['gpu'] for r in running.values()}
  for task in list(tasks):
   stage,seed,arm=task
   if stage=='train' and not all((root/f'admission_seed{s}.json').exists() for s in range(3)):continue
   choices=[g for g,m in free.items() if m>=4500 and g not in used]
   if not choices:break
   gpu=max(choices,key=free.get);used.add(gpu);name=f'{stage}_s{seed}_{arm}';log=(root/(name+'.log')).open('a')
   command=[sys.executable,'-B',str(Path(__file__).with_name('context_task.py')),stage,'--root',str(root),'--code',code,'--seed',str(seed)]
   if arm:command+=['--arm',arm]
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   running[task]=dict(stage=stage,seed=seed,arm=arm,pid=proc.pid,gpu=gpu,name=name,started=time.time(),process=proc,stream=log);tasks.remove(task)
  for task,row in list(running.items()):
   result=row['process'].poll()
   if result is None:continue
   row['stream'].close();del running[task]
   if result:write(root/'error.json',dict(stage=row['stage'],seed=row['seed'],arm=row['arm'],exit_code=result,log=row['name']+'.log'));return False
   done.append(task)
  write(root/'status.json',dict(phase=phase,updated=time.time(),running=[{k:v for k,v in r.items() if k not in ['process','stream']} for r in running.values()],pending=tasks,done=done));time.sleep(15)
 return True
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--code',required=True);a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tasks=[('admission',s,'') for s in range(3)]+[('train',s,arm) for arm in ARMS for s in range(3)]
 if not execute(root,a.code,tasks,'screen'):return
 means={arm:sum(json.loads((root/'screen'/f'seed{s}'/arm/'complete.json').read_bytes())['best_validation_FGARI'] for s in range(3))/3 for arm in ARMS}
 arm=max(means,key=means.get);write(root/'screen_results.json',dict(validation_means=means,selected_arm=arm,selection='validation32 only'))
 reference=json.loads((Path('/work/USERS/kevinswk/experiments/SW0260_readout64_r1')/'final_selection.json').read_bytes())
 objective_reference=json.loads((Path('/work/USERS/kevinswk/experiments/SW0260_objective64')/'final_selection.json').read_bytes())
 threshold=max(max(reference['validation_means'].values()),max(objective_reference['validation_means'].values()))
 if means[arm]<=threshold:
  write(root/'rejected_by_validation.json',dict(reference_validation_FGARI=threshold,candidate_validation_FGARI=means[arm],reason='no validation gain; do not spend extended training or query fixed320'))
  write(root/'stage_complete.json',dict(completed=time.time(),trained_screen=9,trained_extension=0));return
 if not execute(root,a.code,[('extend',s,arm) for s in range(3)],'extend'):return
 mean=sum(json.loads((root/'extended'/f'seed{s}'/arm/'complete.json').read_bytes())['best_validation_FGARI'] for s in range(3))/3
 write(root/'final_selection.json',dict(arm=arm,validation_means={arm:mean},selection='validation32 only',routing_variant='contextual_fine_readout',core_grid=32,head_grid=64))
 write(root/'extension_complete.json',dict(completed=time.time(),trained_screen=9,trained_extension=3))
if __name__=='__main__':main()
