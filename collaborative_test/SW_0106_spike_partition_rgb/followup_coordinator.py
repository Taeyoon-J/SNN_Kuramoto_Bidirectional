"""Conditional SW0106 seed expansion, full70k, and frozen-reserve queue.

This is independent of coordinator.py: it waits for PID4012262's seed0 queue to
reach a validated terminal state, and never rewrites that queue or its outputs.
"""
import argparse, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path

from coordinator import eligible_gpus, process_descendants, terminate_owned
from followup_policy import load_eval, pilot_promotion_gate, full_validation_reserve_gate, METRICS

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
OUT=ROOT/'trained_models/SW0106_spike_partition_rgb'
RUNNER=HERE/'run.py'
FINAL=HERE/'final_confirmation.py'
SW107=ROOT/'collaborative_test/SW_0107_official_full70k_transfer'
SW107_PREP=SW107/'prepare_official.py';SW107_RUN=SW107/'run.py'
SW107_SUMMARIZE=SW107/'summarize.py'
SW107_POOL=ROOT/'data/SW_0107_official_full70k_transfer'
SW107_RGB=SW107_POOL/'official_rgb_float16.h5';SW107_GAMMA=SW107_POOL/'official_gamma.pt'
TF_PY='/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python'
UPSTREAM=ARCHIVE/'gpu_queue_state.json'
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
SLOT_SUMMARY=ROOT/'collaborative_test/SW_0092_cross_dataset_training/results/slot_our70000/summary.json'
GPUS=(0,1,2,3)
MAX_WORKERS=4
OWNER_POLL_SECONDS=5
POLL_SECONDS=60
QUEUE_STATE=ARCHIVE/'followup_queue_state.json'
LOCK=OUT/'followup_queue.lock'

def read_json(path):return json.loads(Path(path).read_text())
def write_atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def metric(path):return load_eval(path)
def empty(path):
 if Path(path).exists():raise FileExistsError(f'preserve existing follow-up output; no overwrite: {path}')
def state_has_no_active(d):return not list((d.get('active') or {}).keys())
def pilot_seed0_gate():
 c=metric(OUT/'candidate_seed0/evaluation.json');t=metric(OUT/'control_seed0/evaluation.json')
 b=metric(CONTROL/'seed0_positive_frozen/evaluation.json')
 checks={'candidate_fg_plus_0.01_vs_joint_control':c['metrics']['fg_ari']>=t['metrics']['fg_ari']+.01,
  'candidate_fg_plus_0.01_vs_SW0097':c['metrics']['fg_ari']>=b['metrics']['fg_ari']+.01,
  'foreground_iou_slot_margin':c['metrics']['foreground_iou']>=.25358913,
  'matched_object_iou_slot_margin':c['metrics']['matched_object_iou']>=.25693698}
 return {'status':'passed' if all(checks.values()) else 'failed','candidate':c['metrics'],'joint_control':t['metrics'],
  'sw0097':b['metrics'],'checks':checks}
def pilot_three_seed_gate():
 cand=[metric(OUT/f'candidate_seed{s}/evaluation.json') for s in range(3)]
 ctrl=[metric(OUT/f'control_seed{s}/evaluation.json') for s in range(3)]
 ref=[metric(CONTROL/f'seed{s}_positive_frozen/evaluation.json') for s in range(3)]
 slot=read_json(SLOT_SUMMARY)['epochs']['10']['mean']
 return pilot_promotion_gate(cand,ctrl,ref),cand,ctrl,ref,slot
def full_validation_gate():
 cand=[metric(OUT/f'full70k_candidate_seed{s}/evaluation.json') for s in range(3)]
 ctrl=[metric(OUT/f'full70k_control_seed{s}/evaluation.json') for s in range(3)]
 slot=read_json(SLOT_SUMMARY)['epochs']['10']['seeds']
 slot_rows=[{'metrics':{m:float(slot[str(s)][m]) for m in METRICS}} for s in range(3)]
 gate=full_validation_reserve_gate(cand,ctrl,slot_rows)
 gate['criterion']='candidate mean strictly above Slot epoch10 mean on all3 metrics; controls retained as required paired endpoints'
 return gate,cand,ctrl,slot_rows

def gpu_owners(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
 return [int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
def gpu_used(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 vals=[int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
 if len(vals)!=1:raise RuntimeError(f'bad GPU{gpu} memory response: {r.stdout!r}')
 return vals[0]
def free_devices():return eligible_gpus(GPUS,gpu_owners,gpu_used)
def key(task):return f"{task['stage']}:{task.get('arm','')}:{task.get('seed','')}"
def task(stage,seed=None,arm=None):
 d={'stage':stage}
 if seed is not None:d['seed']=int(seed)
 if arm is not None:d['arm']=arm
 return d
def unique_assignments(free,active,limit=MAX_WORKERS):
 used={j['gpu'] for j in active.values()}
 return [g for g in free if g not in used][:max(0,limit-len(active))]
def worker_command(t):
 s=t.get('seed');a=t.get('arm')
 if t['stage']=='preflight':return [sys.executable,str(RUNNER),'--stage','preflight','--seed',str(s),'--device','cuda:0','--output',str(ARCHIVE/f'preflight_seed{s}.json')]
 if t['stage'] in ('train','eval','full-train','full-eval'):
  return [sys.executable,str(RUNNER),'--stage',t['stage'],'--seed',str(s),'--arm',a,'--device','cuda:0']
 if t['stage']=='reserve-ours':return [sys.executable,str(FINAL),'--stage','reserve-ours','--arm',a,'--seed',str(s)]
 if t['stage']=='reserve-slot':return [sys.executable,str(FINAL),'--stage','reserve-slot','--seed',str(s)]
 if t['stage']=='sw107-export':return [TF_PY,str(SW107_PREP),'--stage','export','--source',str(SW107_RGB),'--manifest',str(SW107_POOL/'official_export_manifest.json')]
 if t['stage']=='sw107-encode':return [sys.executable,str(SW107_PREP),'--stage','encode','--source',str(SW107_RGB),
  '--manifest',str(SW107_POOL/'official_export_manifest.json'),'--gamma',str(SW107_GAMMA),
  '--gamma-manifest',str(SW107_POOL/'official_gamma_manifest.json'),'--device','cuda:0']
 if t['stage'] in ('sw107-preflight','sw107-train','sw107-eval'):
  stage={'sw107-preflight':'preflight','sw107-train':'train','sw107-eval':'eval'}[t['stage']]
  return [sys.executable,str(SW107_RUN),'--stage',stage,'--seed',str(s),'--device','cuda:0']
 raise ValueError(t['stage'])
def task_output(t):
 s=t.get('seed');a=t.get('arm');stage=t['stage']
 if stage=='preflight':return ARCHIVE/f'preflight_seed{s}.json'
 if stage in ('train','eval'):return OUT/f'{a}_seed{s}'
 if stage in ('full-train','full-eval'):return OUT/f'full70k_{a}_seed{s}'
 if stage=='reserve-ours':return ARCHIVE/'final_reserve'/f'ours_{a}_seed{s}'
 if stage=='reserve-slot':return ARCHIVE/'final_reserve'/f'slot_seed{s}'
 if stage=='sw107-export':return SW107_RGB
 if stage=='sw107-encode':return SW107_GAMMA
 if stage=='sw107-preflight':return SW107/'results_archive'/f'preflight_seed{s}.json'
 if stage in ('sw107-train','sw107-eval'):return ROOT/f'trained_models/SW0107_official_full70k_transfer/seed{s}'
 raise ValueError(stage)
def validate_task(t):
 stage=t['stage'];s=t.get('seed');a=t.get('arm');out=task_output(t)
 if stage=='preflight':
  d=read_json(out)
  if d.get('status')!='passed' or d.get('seed')!=s:return False
  if len(d.get('training_ids_first4096',[]))!=4096 or d.get('shuffle_seed')!=117+s:return False
  flags=('warmup_source_core_and_encoder_unchanged','initial_candidate_control_hard_forward_exact','production_labels_exact',
   'control_reconstruction_has_no_core_encoder_gradient','throwaway_candidate_update_finite_and_changed_graph_encoder_parameters',
   'throwaway_decoder_optimizer_state_loaded_from_shared_32_batch_warmup')
  if any(d.get(flag) is not True for flag in flags):return False
  if any(float(d.get('candidate_reconstruction_gradient_norms_by_family',{}).get(f,0.))<=0 for f in ('encoder','graph_generator','kuramoto')):return False
  calibration=d.get('lambda_calibration',[])
  if len(calibration)!=4:return False
  for row in calibration:
   if not math.isfinite(float(row.get('ratio',float('nan')))) or float(row['ratio'])<=0:return False
   fam=row.get('reconstruction_grad_norms_by_family',{})
   if any(not math.isfinite(float(fam.get(f,0.))) or float(fam.get(f,0.))<=0 for f in ('encoder','graph_generator','kuramoto')):return False
  shuffle=d.get('row_scramble_by_batch',[])
  if len(shuffle)!=4 or d.get('row_scramble_images')!=64 or float(d.get('row_scramble_delta_mean',0.))<=0:return False
  if any(x.get('images')!=16 or not math.isfinite(float(x.get('mean_shuffled_minus_original_mse',float('nan')))) for x in shuffle):return False
  artifact=Path(d.get('warmup_artifact',''))
  return artifact.is_file() and file_sha(artifact)==d.get('warmup_artifact_sha256') and math.isfinite(float(d.get('lambda',float('nan')))) and float(d['lambda'])>0
 if stage=='train':
  m=read_json(out/'manifest.json');h=read_json(out/'history.json');op=out/'optimizer_state.pt'
  return (m.get('status')=='training_complete' and m.get('seed')==s and m.get('arm')==a and
   m.get('updates')==256 and len(h)==256 and (out/'TRAINING_COMPLETED').is_file() and op.is_file() and
   file_sha(op)==m.get('optimizer_state_sha256') and m.get('joint_optimizer_steps')==256 and m.get('decoder_optimizer_steps')==288)
 if stage=='eval':
  d=read_json(out/'evaluation.json');load_eval(out/'evaluation.json')
  return (out/'COMPLETED').is_file() and d.get('ground_truth_used_for_prediction') is False
 if stage=='full-train':
  m=read_json(out/'manifest.json');h=read_json(out/'history.json');op=out/'optimizer_state.pt'
  return (m.get('status')=='full70k_training_complete' and m.get('seed')==s and m.get('arm')==a and
   m.get('full_updates')==4375 and m.get('full_images')==70000 and len(h)==4375 and
   (out/'FULL70K_TRAINING_COMPLETED').is_file() and op.is_file() and file_sha(op)==m.get('full_optimizer_state_sha256'))
 if stage=='full-eval':
  d=read_json(out/'evaluation.json');load_eval(out/'evaluation.json')
  return (out/'COMPLETED').is_file() and d.get('ground_truth_used_for_prediction') is False
 if stage=='reserve-ours':
  f=out/'evaluation.json';load_eval(f,ids=(90640,90959));return (out/'RESERVE_COMPLETED').is_file()
 if stage=='reserve-slot':
  d=read_json(out/'evaluation_summary.json');p=read_json(out/'protocol.json')
  if d.get('protocol',{}).get('image_ids')!=[90640,90959] or p.get('image_ids')!=[90640,90959]:return False
  if d.get('protocol',{}).get('count')!=320 or d.get('ground_truth_used_for_prediction') is not False:return False
  if not (out/'SCORING_COMPLETED').is_file() or not (out/'per_image.csv').is_file():return False
  from final_confirmation import slot_csv_metrics
  vals=slot_csv_metrics(out/'per_image.csv');counts=d.get('scores',{}).get('valid_count',{});means=d.get('scores',{}).get('mean',{})
  return all(int(counts.get(m,-1))==320 and abs(float(vals[m].mean())-float(means.get(m,float('nan'))))<=1e-12 for m in METRICS)
 if stage=='sw107-export':
  manifest=read_json(SW107_POOL/'official_export_manifest.json')
  return (out.is_file() and manifest.get('status')=='complete' and manifest.get('images')==70000 and
   manifest.get('source_indices')==[0,69999] and manifest.get('skip')==0 and manifest.get('filter')=='none' and manifest.get('ground_truth_masks_read') is False)
 if stage=='sw107-encode':
  m=read_json(SW107_POOL/'official_gamma_manifest.json')
  return out.is_file() and tuple(m.get('shape',()))==(70000,8,256) and m.get('gamma_sha256')==file_sha(out) and m.get('ground_truth_used') is False
 if stage=='sw107-preflight':
  d=read_json(out)
  return (d.get('status')=='passed' and d.get('seed')==s and d.get('shuffle_seed')==117+s and d.get('batch_size')==16 and
   d.get('time_steps')==64 and d.get('settle')==32 and d.get('ground_truth_used') is False and d.get('graph_bitwise_unchanged') is True and
   d.get('source_hash_unchanged') is True and d.get('throwaway_optimizer_step') is True and math.isfinite(float(d.get('gradient_norm_preclip',float('nan')))) and
   math.isfinite(float(d.get('loss_value',float('nan')))) and bool(d.get('changed_core_keys')) and float(d.get('gradient_norm_preclip',0.))>0)
 if stage=='sw107-train':
  m=read_json(out/'manifest.json');h=read_json(out/'history.json')
  return (m.get('status')=='training_complete' and m.get('seed')==s and m.get('updates')==4375 and m.get('training_unique_images')==70000 and
   len(h)==4375 and m.get('graph_frozen') is True and m.get('encoder_frozen') is True and (out/'TRAINING_COMPLETED').is_file())
 if stage=='sw107-eval':
  d=read_json(out/'evaluation.json');load_eval(out/'evaluation.json')
  return (out/'COMPLETED').is_file() and d.get('ground_truth_used_for_prediction') is False
 return False

def start_sw107(state,reason):
 state['sw0106_outcome']=reason
 if state.get('sw0107_status')=='complete':
  state['phase']='complete';state['status']='complete';state['completed_unix']=time.time();return
 state['phase']='sw107_preflight_all3';state['status']='running'
 state['pending']=[task('sw107-preflight',s) for s in range(3)]
 state['sw0107_status']='queued'

def next_phase(state):
 """Advance only after every worker in the current immutable phase validates."""
 if state['active'] or state['pending']:return
 phase=state['phase'];done=set(state['completed'])
 if phase=='preflight12':
  if not all(f'preflight::{s}' in done for s in (1,2)):raise RuntimeError('seed1/2 preflight phase missing a result')
  state['phase']='pilot_train12';state['pending']=[task('train',s,a) for s in (1,2) for a in ('candidate','control')]
 elif phase=='pilot_train12':
  if not all(f'train:{a}:{s}' in done for s in (1,2) for a in ('candidate','control')):raise RuntimeError('pilot train phase missing a result')
  state['phase']='pilot_eval12';state['pending']=[task('eval',s,a) for s in (1,2) for a in ('candidate','control')]
 elif phase=='pilot_eval12':
  if not all(f'eval:{a}:{s}' in done for s in (1,2) for a in ('candidate','control')):raise RuntimeError('pilot eval phase missing a result')
  gate,_,_,_,_=pilot_three_seed_gate();write_atomic(ARCHIVE/'three_seed_pilot_gate.json',gate);state['three_seed_pilot_gate']=gate
  if gate['status']!='passed':state['failed']={'phase':phase,'gate':gate};start_sw107(state,'three_seed_pilot_gate_failed');return
  seed0_gate=read_json(ARCHIVE/'seed0_expansion_gate.json')
  combined={'status':'passed' if seed0_gate.get('status')=='passed' and gate.get('status')=='passed' else 'failed',
   'seed0_pilot_gate':seed0_gate,'three_seed_pilot_gate':gate}
  write_atomic(ARCHIVE/'full70k_promotion_gate.json',combined)
  state['full70k_promotion_gate']=combined
  state['phase']='full_train_all6';state['pending']=[task('full-train',s,a) for s in range(3) for a in ('candidate','control')]
 elif phase=='full_train_all6':
  if not all(f'full-train:{a}:{s}' in done for s in range(3) for a in ('candidate','control')):raise RuntimeError('full70k training phase incomplete')
  state['phase']='full_eval_all6';state['pending']=[task('full-eval',s,a) for s in range(3) for a in ('candidate','control')]
 elif phase=='full_eval_all6':
  if not all(f'full-eval:{a}:{s}' in done for s in range(3) for a in ('candidate','control')):raise RuntimeError('full70k evaluation phase incomplete')
  gate,_,_,_=full_validation_gate();write_atomic(ARCHIVE/'full70k_reserve_gate.json',gate);state['full70k_reserve_gate']=gate
  if gate['status']!='passed':state['failed']={'phase':phase,'gate':gate};start_sw107(state,'full70k_validation_gate_failed');return
  state['phase']='freeze_before_reserve';state['freeze_required']=True
 elif phase=='reserve_all9':
  if not all(f'reserve-ours:{a}:{s}' in done for s in range(3) for a in ('candidate','control')) or not all(f'reserve-slot::{s}' in done for s in range(3)):
   raise RuntimeError('reserve evaluation phase incomplete')
  subprocess.run([sys.executable,str(FINAL),'--stage','summarize'],cwd=ROOT,check=True)
  start_sw107(state,'sw0106_validation_complete')
 elif phase=='sw107_export':
  if 'sw107-export::' not in done:raise RuntimeError('SW0107 exact TFDS export not validated')
  state['phase']='sw107_encode';state['pending']=[task('sw107-encode')]
 elif phase=='sw107_encode':
  if 'sw107-encode::' not in done:raise RuntimeError('SW0107 registered gamma cache not validated')
  state['phase']='sw107_preflight_all3';state['pending']=[task('sw107-preflight',s) for s in range(3)]
 elif phase=='sw107_preflight_all3':
  if not all(f'sw107-preflight::{s}' in done for s in range(3)):raise RuntimeError('SW0107 B16 actual-gradient preflight phase incomplete')
  state['phase']='sw107_train_all3';state['pending']=[task('sw107-train',s) for s in range(3)]
 elif phase=='sw107_train_all3':
  if not all(f'sw107-train::{s}' in done for s in range(3)):raise RuntimeError('SW0107 three-seed training phase incomplete')
  state['phase']='sw107_eval_all3';state['pending']=[task('sw107-eval',s) for s in range(3)]
 elif phase=='sw107_eval_all3':
  if not all(f'sw107-eval::{s}' in done for s in range(3)):raise RuntimeError('SW0107 three-seed endpoint phase incomplete')
  subprocess.run([sys.executable,str(SW107_SUMMARIZE)],cwd=ROOT,check=True)
  state['sw0107_status']='complete'
  if state.get('sw107_first') and not state.get('upstream_status')=='complete':
   state['phase']='wait_seed0';state['status']='waiting_for_seed0_pilot_terminal'
  else:
   state['phase']='complete';state['status']='complete';state['completed_unix']=time.time()
 else:raise ValueError(f'unknown queue phase {phase}')
 if state['phase']!='freeze_before_reserve':state['pending']=state['pending'] or []

def output_marker_exists(t):
 out=task_output(t)
 if t['stage']=='preflight':return out.exists()
 if t['stage'] in ('eval','full-eval'):return (out/'evaluation.json').exists()
 if t['stage']=='sw107-export':return out.exists() or (SW107_POOL/'official_export_manifest.json').exists()
 if t['stage']=='sw107-encode':return out.exists() or (SW107_POOL/'official_gamma_manifest.json').exists()
 if t['stage']=='sw107-preflight':return out.exists()
 if t['stage'] in ('sw107-train','sw107-eval'):
  return out.exists() if t['stage']=='sw107-train' else (out/'evaluation.json').exists()
 return out.exists()

def save_state(s):
 z={k:v for k,v in s.items() if k!='active'}
 z['active']={k:{a:j[a] for a in ('task','gpu','pid','log','owners_before','started')} for k,j in s.get('active',{}).items()}
 write_atomic(QUEUE_STATE,z)

def run_cpu_preparation(s,t,cmd):
 """Run the independent SW0107 export/encoder preparation without any GPU visibility."""
 if output_marker_exists(t):
  if validate_task(t):
   s['completed'][key(t)]={'task':t,'resource':'cpu','validated_existing':True,'completed_unix':time.time()};save_state(s);return
  raise FileExistsError(f'preserve partial/invalid SW0107 CPU preparation output for {t}')
 logpath=OUT/'followup_queue_logs'/f"{t['stage']}_cpu.log"
 empty(logpath);env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 if t['stage']=='sw107-export':env['TF_CPP_MIN_LOG_LEVEL']='2'
 with logpath.open('w') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  s.setdefault('cpu_preparation',[]).append({'task':t,'pid':proc.pid,'log':str(logpath),'status':'running','started':time.time()});save_state(s)
  try:
   while proc.poll() is None:
    time.sleep(30);save_state(s)
  except BaseException:
   terminate_owned(proc);raise
  rc=proc.returncode
 if rc!=0 or not validate_task(t):
  record={'task':t,'pid':proc.pid,'return_code':rc,'log':str(logpath),'status':'failed','finished':time.time()}
  s['cpu_preparation'][-1].update(record);s['status']='failed';s['failed']=record;save_state(s)
  raise RuntimeError(f"SW0107 CPU preparation failed; preserve {logpath}")
 s['cpu_preparation'][-1].update({'status':'complete','return_code':0,'finished':time.time()})
 s['completed'][key(t)]={'task':t,'pid':proc.pid,'resource':'cpu','validated':True,'completed_unix':time.time()};save_state(s)

def main():
 global UPSTREAM
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--poll-seconds',type=int,default=POLL_SECONDS)
 p.add_argument('--sw107-first',action='store_true',help='Run independent SW0107 before waiting for SW0106; do not repeat it later')
 p.add_argument('--upstream-state',type=Path,default=UPSTREAM)
 p.add_argument('--reuse-completed-sw107',action='store_true',help='Validate and reuse all completed SW0107 workers; never retrain them')
 a=p.parse_args()
 if not a.run:raise SystemExit('readiness only; --run starts the approved conditional follow-up queue')
 if a.poll_seconds<10:raise ValueError('poll interval must be >=10 seconds')
 if not a.upstream_state.resolve().is_relative_to(ARCHIVE.resolve()):raise ValueError('upstream state must be an archived SW0106 queue')
 UPSTREAM=a.upstream_state
 reused=[]
 if a.reuse_completed_sw107:
  if a.sw107_first:raise ValueError('completed SW0107 must not be queued again')
  summary=read_json(SW107_SUMMARIZE.parent/'results_archive/summary.json')
  if summary.get('status')!='complete':raise ValueError('completed SW0107 summary required')
  for seed in range(3):
   for stage in ('sw107-preflight','sw107-train','sw107-eval'):
    t=task(stage,seed)
    if not validate_task(t):raise ValueError(f'completed SW0107 worker failed revalidation: {t}')
    reused.append(t)
 ARCHIVE.mkdir(parents=True,exist_ok=True);(OUT/'followup_queue_logs').mkdir(parents=True,exist_ok=True)
 if QUEUE_STATE.exists():raise FileExistsError(f'preserve prior follow-up state: {QUEUE_STATE}')
 fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 s={'status':'waiting_for_seed0_pilot_terminal','queue_pid':os.getpid(),'created_unix':time.time(),
  'phase':'wait_seed0','eligible_gpus':list(GPUS),'max_workers':MAX_WORKERS,'exclusive_no_foreign_sharing':True,
  'upstream_queue_state':str(UPSTREAM),'pending':[],'active':{},'completed':{},'attempts':[],
  'training_restarted':False,'reserve_read_started':False,'sw107_first':a.sw107_first}
 if reused:
  s['sw0107_status']='complete'
  for t in reused:s['completed'][key(t)]={'task':t,'validated_existing':True,'completed_unix':time.time()}
 save_state(s)
 try:
  s['status']='preparing_sw0107_cpu_data';s['phase']='sw107_cpu_prepare';save_state(s)
  run_cpu_preparation(s,task('sw107-export'),worker_command(task('sw107-export')))
  run_cpu_preparation(s,task('sw107-encode'),[sys.executable,str(SW107_PREP),'--stage','encode','--source',str(SW107_RGB),
   '--manifest',str(SW107_POOL/'official_export_manifest.json'),'--gamma',str(SW107_GAMMA),
   '--gamma-manifest',str(SW107_POOL/'official_gamma_manifest.json'),'--device','cpu'])
  s['status']='waiting_for_seed0_pilot_terminal';s['phase']='wait_seed0';s['sw0107_cpu_cache_status']='complete';save_state(s)
  if a.sw107_first:
   start_sw107(s,'independent_user_requested_priority');save_state(s)
  while s['status'] not in ('complete','failed','complete_validation_gate_failed'):
   if s['phase']=='wait_seed0':
    if not UPSTREAM.is_file():s['upstream_status']='waiting';save_state(s);time.sleep(a.poll_seconds);continue
    upstream=read_json(UPSTREAM)
    if upstream.get('status') in ('failed','error'):
     s['status']='failed';s['failed']={'phase':'wait_seed0','upstream_status':upstream.get('status')};save_state(s);break
    if upstream.get('status')!='complete' or not state_has_no_active(upstream):
     s['upstream_status']='waiting';save_state(s);time.sleep(a.poll_seconds);continue
    gate=pilot_seed0_gate();write_atomic(ARCHIVE/'seed0_expansion_gate.json',gate);s['seed0_expansion_gate']=gate;s['upstream_status']='complete'
    if gate['status']!='passed':s['failed']={'phase':'seed0_expansion_gate','gate':gate};start_sw107(s,'seed0_expansion_gate_failed');save_state(s);continue
    s['phase']='preflight12';s['pending']=[task('preflight',seed) for seed in (1,2)];save_state(s)
   if s['phase']=='freeze_before_reserve':
    freeze=ARCHIVE/'final_reserve'/'frozen_models_manifest.json'
    if freeze.exists():raise FileExistsError('preserve existing reserve freeze manifest; do not overwrite')
    cp=subprocess.run([sys.executable,str(FINAL),'--stage','freeze'],cwd=ROOT,capture_output=True,text=True,check=True)
    result=json.loads(cp.stdout.strip().splitlines()[-1])
    if result.get('status')!='frozen' or not freeze.is_file():raise RuntimeError('reserve freeze did not produce its required manifest')
    s['reserve_freeze_manifest_sha256']=file_sha(freeze);s['reserve_read_started']=False
    s['phase']='reserve_all9';s['pending']=[task('reserve-ours',seed,arm) for seed in range(3) for arm in ('candidate','control')]+[task('reserve-slot',seed) for seed in range(3)]
    save_state(s)
   next_phase(s)
   save_state(s)
   if s['status'] in ('failed','complete_validation_gate_failed','complete'):
    save_state(s);break
   for k,j in list(s['active'].items()):
    proc=j['proc'];rc=proc.poll()
    if rc is not None:
     j['logfile'].close();t=j['task'];valid=False
     if rc==0:valid=validate_task(t)
     attempt={'task':t,'gpu':j['gpu'],'pid':proc.pid,'return_code':rc,'log':j['log'],'status':'complete' if rc==0 and valid else 'failed','finished_unix':time.time()}
     s['attempts'].append(attempt);del s['active'][k]
     if rc!=0 or not valid:s['status']='failed';s['failed']=attempt;s['pending']=[];save_state(s);raise RuntimeError(f"follow-up task failed; preserve {j['log']}")
     s['completed'][key(t)]={'task':t,'gpu':j['gpu'],'pid':proc.pid,'completed_unix':time.time()};save_state(s)
    else:
     owners=gpu_owners(j['gpu']);desc=process_descendants([proc.pid]);foreign=[x for x in owners if x not in desc]
     if foreign:
      terminate_owned(proc);j['logfile'].close();s['status']='failed';s['failed']={'task':j['task'],'gpu':j['gpu'],'pid':proc.pid,'foreign_gpu_pids':foreign,'status':'stopped_on_foreign_owner','log':j['log']};s['pending']=[];del s['active'][k];save_state(s)
      raise RuntimeError('follow-up worker stopped after foreign GPU owner appeared')
   if s['pending'] and len(s['active'])<MAX_WORKERS:
    available=unique_assignments(free_devices(),s['active'])
    for gpu in available:
     if not s['pending']:break
     t=s['pending'][0]
     # Recheck ownership and residual allocation immediately before spawn.
     if gpu_owners(gpu) or gpu_used(gpu)>512:continue
     if output_marker_exists(t):raise FileExistsError(f'preserve existing output for task {t}; no overwrite')
     if t['stage'].startswith('reserve-ours') or t['stage'].startswith('reserve-slot'):
      if not (ARCHIVE/'final_reserve'/'frozen_models_manifest.json').is_file():raise RuntimeError('reserve worker blocked until frozen model/data manifest exists')
      s['reserve_read_started']=True
     logpath=OUT/'followup_queue_logs'/f"{t['stage']}_{t.get('arm','none')}_seed{t.get('seed','none')}_gpu{gpu}.log"
     empty(logpath);logfile=logpath.open('w');before={str(g):gpu_owners(g) for g in GPUS}
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
     if t['stage']=='reserve-slot':env.update(SW0092_ALLOW_GPU='1',TF_FORCE_GPU_ALLOW_GROWTH='true')
     proc=subprocess.Popen(worker_command(t),cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT,env=env,start_new_session=True)
     time.sleep(1);owners=gpu_owners(gpu);desc=process_descendants([proc.pid]);foreign=[x for x in owners if x not in desc]
     if foreign:
      terminate_owned(proc);logfile.close();s['status']='failed';s['failed']={'task':t,'gpu':gpu,'pid':proc.pid,'foreign_gpu_pids':foreign,'status':'launch_race_stopped','log':str(logpath)};s['pending']=[];save_state(s)
      raise RuntimeError('follow-up launch race: stopped owned worker only')
     k=key(t);s['pending'].pop(0);s['active'][k]={'task':t,'gpu':gpu,'pid':proc.pid,'proc':proc,'logfile':logfile,'log':str(logpath),'owners_before':before,'started':time.time()}
     s['attempts'].append({'task':t,'gpu':gpu,'pid':proc.pid,'owners_before':before,'launch_owners':owners,'log':str(logpath),'status':'running'});save_state(s)
   if s['pending'] or s['active']:time.sleep(OWNER_POLL_SECONDS if s['active'] else a.poll_seconds)
 except Exception as exc:
  if s['status']!='failed':s['status']='error';s['error']=f'{type(exc).__name__}: {exc}';s['failed_unix']=time.time();save_state(s)
  raise
 finally:
  for j in s['active'].values():terminate_owned(j['proc'])
  try:LOCK.unlink()
  except FileNotFoundError:pass
 print(json.dumps({k:v for k,v in s.items() if k!='active'},indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
