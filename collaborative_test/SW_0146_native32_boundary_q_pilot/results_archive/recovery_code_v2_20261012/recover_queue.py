"""Fresh-source SW0146 recovery after both arms were interrupted by foreign GPU owners."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, sys, threading, time
from pathlib import Path
import numpy
if not numpy.__version__:raise RuntimeError("NumPy must import before the frozen Torch stack")

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
HERE=Path(__file__).resolve().parent
ACMLAB=HERE.parents[1]
ROOT=Path(os.environ.get("PATCH_V2_REPO",ACMLAB/"patch_v2_sw")).resolve()
os.environ["PATCH_V2_REPO"]=str(ROOT)
EXP=ROOT/"collaborative_test/SW_0146_native32_boundary_q_pilot"
ARCHIVE=EXP/"results_archive"
OLD_RUNTIME=ARCHIVE/"queue_runtime_20261012"
OLD_STATE=OLD_RUNTIME/"queue_state.json"
RUNTIME=ARCHIVE/"recovery_v2_20261012"
STATE=RUNTIME/"recovery_state.json"
LOCK=ARCHIVE/"recovery_v2.lock"
RUN_SHA="9f94c097e0f1765f32247c729523ab306f1319c508cef4160bdefef1c4dea489"
PROTOCOL_SHA="9d86bb93e52e84dee13e160099a3cd91899c9ec16d37dce3f1763b99d3aaca0c"
QUEUE_SHA="f87cb9a138a439c369f9d74d284a6c9f0d0cc87f778510af04c49c01f7030cc2"
OWNER_SHA="b32c1aa5ce8417359aaab40bb9606e6fdee2c8d8314281f365616c24805dbe00"
SUPERVISOR=458960
TERMINAL_STATE_SHA="4423732ec980c1c8dd36baea5565ce9ce83a093ebca556dd4ad0b8835d5704b1"
FAILED={"encoder_graph":(461418,"RuntimeError: foreign GPU owner appeared: [464125]"),
        "joint":(462591,"RuntimeError: foreign GPU owner appeared: [463556]")}
FROZEN_QUEUE=EXP/"results_archive/queue_code_20261012/probe_queue.py"
SERVER_OWNER=ROOT/"collaborative_test/SW_0143_native32_selective_geodesic/results_archive/recovery_code_v3_20261012/recover_queue.py"
LOCAL_OWNER=HERE.parent/"SW0143_recovery_v2_20261012/recover_queue.py"
OWNER=SERVER_OWNER if SERVER_OWNER.is_file() else LOCAL_OWNER

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _load(path,name,expected):
 if not path.is_file() or sha(path)!=expected:raise RuntimeError(f"frozen helper missing/changed: {path}")
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise ImportError(str(path))
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
owner=_load(OWNER,"sw0143_owner_for_sw0146_recovery_v2",OWNER_SHA)
frozen=owner.frozen
_queue_import_path=FROZEN_QUEUE
if not SERVER_OWNER.is_file():
 _base=ACMLAB/".sw146qi"/"patch_v2_sw"/"collaborative_test"/"SW_0146_native32_boundary_q_pilot"/"results_archive"
 _qdir=_base/"queue_code_20261012";_qdir.mkdir(parents=True,exist_ok=True)
 _queue_import_path=_qdir/"probe_queue.py"
 _hdir=_base/"SW0143_recovery_v2_20261012";_hdir.mkdir(parents=True,exist_ok=True)
 _h1src=HERE.parent/"SW0143_recovery_v1_20261012"/"recover_queue.py"
 _h1dst=_base/"SW0143_recovery_v1_20261012"/"recover_queue.py";_h1dst.parent.mkdir(parents=True,exist_ok=True)
 _h1sha="4ad493180d63a89e44bb9f6a27cd79aaaf9e520907a339958d16b4cec38499ed"
 for src,dst,expected in ((FROZEN_QUEUE,_queue_import_path,QUEUE_SHA),(OWNER,_hdir/"recover_queue.py",OWNER_SHA),(_h1src,_h1dst,_h1sha)):
  if dst.exists():
   if sha(dst)!=expected:raise RuntimeError(f"local frozen import copy changed: {dst}")
  else:shutil.copyfile(src,dst)
queue=_load(_queue_import_path,"sw0146_frozen_queue_for_recovery_v2",QUEUE_SHA)
sys.path[:0]=[str(ROOT),str(ROOT/"collaborative_test"),str(ROOT/"snn_kuramoto_bidirectional")]
from collaborative_test.SW_0146_native32_boundary_q_pilot import run

def proc_argv(pid):
 try:
  raw=Path(f"/proc/{int(pid)}/cmdline").read_bytes()
  return [x.decode(errors="replace") for x in raw.split(b"\0") if x] or None
 except (FileNotFoundError,ProcessLookupError):return None
def proc_rows():
 rows=[];p=Path("/proc")
 if p.is_dir():
  for x in p.iterdir():
   if x.name.isdigit():
    a=proc_argv(int(x.name))
    if a:rows.append((int(x.name),a))
 return rows
def _stage_arg(a):
 return any("collaborative_test.SW_0146_native32_boundary_q_pilot.run" in x for x in a) and "--stage" in a
def validate_terminal_bytes(raw):
 if hashlib.sha256(raw).hexdigest()!=TERMINAL_STATE_SHA:raise ValueError("original terminal state SHA changed")
 return json.loads(raw.decode("utf-8"))
def validate_terminal(s,*,pid_alive,process_rows):
 if (s.get("experiment")!="SW0146_native32_boundary_q_pilot" or s.get("status")!="failed"
     or s.get("supervisor_pid")!=SUPERVISOR or s.get("run_sha256")!=RUN_SHA
     or s.get("protocol_sha256")!=PROTOCOL_SHA or s.get("queue_sha256")!=QUEUE_SHA):
  raise ValueError("original queue is not the observed terminal SW0146 attempt")
 if pid_alive(SUPERVISOR):raise ValueError("original supervisor is still alive")
 tasks=s.get("tasks",{})
 for k in ("preflight_encoder_graph","preflight_joint"):
  r=tasks.get(k,{})
  if r.get("status")!="passed" or r.get("artifact_valid") is not True:raise ValueError("original preflight is not valid")
 for arm,(pid,error) in FAILED.items():
  r=tasks.get("train_"+arm,{})
  if r.get("status")!="failed" or r.get("child_pid")!=pid or r.get("error")!=error or r.get("artifact_valid") not in (None,False):
   raise ValueError(f"{arm} interruption differs from recorded foreign-owner failure")
  if pid_alive(pid):raise ValueError(f"failed {arm} process still alive")
 for pid,argv in process_rows:
  if pid in (os.getpid(),SUPERVISOR,*[x[0] for x in FAILED.values()]):continue
  if _stage_arg(argv):raise ValueError(f"another SW0146 run process is alive (pid {pid})")
 return True

def _check_arm_dir(path,training_root):
 p=Path(path)
 if not p.exists():return
 if p.resolve().parent!=Path(training_root).resolve() or not p.is_dir() or any(p.iterdir()):
  raise FileExistsError(f"preserve nonempty or unsafe partial training output: {p}")
def archive_empty(path,destination,training_root):
 p=Path(path);d=Path(destination)
 if not p.exists():return None
 _check_arm_dir(p,training_root)
 if d.exists():raise FileExistsError("recovery archive already exists")
 d.parent.mkdir(parents=True,exist_ok=True);os.replace(p,d);return str(d)

def recovery_plan():
 return ["train_encoder_graph","train_joint","predict_encoder_graph","predict_joint","score"]
def score_dependencies_passed(statuses):
 return all(statuses.get(k)=="passed" for k in ("predict_encoder_graph","predict_joint"))

class RecoveryQueue:
 def __init__(self,*,sample=None,sleep=None,monotonic=None,popen=None):
  if queue.fcntl is None:raise RuntimeError("recovery requires Linux flock and inherited leases")
  if sha(EXP/"run.py")!=RUN_SHA or sha(EXP/"protocol.json")!=PROTOCOL_SHA:raise RuntimeError("frozen SW0146 files changed")
  if RUNTIME.exists() or LOCK.exists():raise FileExistsError("preserve any existing recovery-v2 attempt")
  if sha(OWNER)!=OWNER_SHA:raise RuntimeError("owner-aware helper changed")
  raw=OLD_STATE.read_bytes();old=validate_terminal_bytes(raw)
  validate_terminal(old,pid_alive=lambda p:proc_argv(p)is not None,process_rows=proc_rows())
  ctx=run._admission_context()
  for arm in ("encoder_graph","joint"):
   row=old["tasks"]["preflight_"+arm]
   if not run._valid_preflight(arm,row["task"]["output"],ctx):raise ValueError(f"preserved {arm} preflight failed current validation")
  for arm in FAILED:
   d=run.arm_dir(arm)
   if not d.is_dir() or any(d.iterdir()):raise FileExistsError(f"only the verified empty {arm} output may be archived")
  for arm in FAILED:
   p=run._prediction_dir(arm)
   if p.exists():raise FileExistsError(f"preserve existing prediction output: {p}")
  score=ARCHIVE/"evaluation_seed1.json"
  if score.exists():raise FileExistsError("preserve existing score")
  logbytes={a:Path(old["tasks"]["train_"+a]["log_path"]).read_bytes() for a in FAILED}
  if any(logbytes.values()):raise ValueError("observed failed-arm logs were empty; preserve changed logs and stop")
  self.prior_sha=hashlib.sha256(raw).hexdigest()
  self.lock=LOCK.open("x+")
  try:queue.fcntl.flock(self.lock.fileno(),queue.fcntl.LOCK_EX|queue.fcntl.LOCK_NB)
  except BaseException:self.lock.close();raise RuntimeError("another recovery owns the recovery lock")
  try:
   RUNTIME.mkdir(parents=True,exist_ok=False);self.logs=RUNTIME/"logs";self.logs.mkdir(exist_ok=False)
   (RUNTIME/"prior_attempt_state.json").write_bytes(raw)
   for arm,data in logbytes.items():(RUNTIME/f"prior_train_{arm}.log").write_bytes(data)
   archived={a:archive_empty(run.arm_dir(a),RUNTIME/f"archived_empty_{a}",run.TRAIN) for a in FAILED}
   queue.STATE=RUNTIME/"recovery_state.json"
   self.runtime=RUNTIME;self.stop=threading.Event()
   self.sample=sample or (lambda gpu:frozen.owner._nvidia_gpu_info(gpu))
   self.sleep=sleep or time.sleep;self.monotonic=monotonic or time.monotonic;self.popen=popen or __import__("subprocess").Popen
   self.max_workers=2;self.admission=run.ADMISSION;self.run_module=run;self._state_lock=threading.RLock()
   self.plan=queue.task_plan();self.deps={"train_encoder_graph":["preflight_encoder_graph","preflight_joint"],
    "train_joint":["preflight_encoder_graph","preflight_joint"],"predict_encoder_graph":["train_encoder_graph"],
    "predict_joint":["train_joint"],"score":["predict_encoder_graph","predict_joint"]}
   self.state={"experiment":run.NAME+"_recovery_v2","status":"queued","supervisor_pid":os.getpid(),
    "run_sha256":RUN_SHA,"protocol_sha256":PROTOCOL_SHA,"original_queue_sha256":QUEUE_SHA,
    "recovery_helper_sha256":sha(Path(__file__)),"owner_helper_sha256":OWNER_SHA,
    "original_state_sha256":self.prior_sha,"preflights_reused":True,
    "prior_update_count":"unknown; discarded, never resumed","archived_empty_dirs":archived,
    "dependencies":self.deps,
    "tasks":{k:{"status":"passed","reused_from_original":True} for k in ("preflight_encoder_graph","preflight_joint")}}
   for k in self.deps:self.state["tasks"][k]={"status":"queued"}
   self._save()
  except BaseException:
   queue.fcntl.flock(self.lock.fileno(),queue.fcntl.LOCK_UN);self.lock.close();raise

 def _save(self):
  tmp=queue.STATE.with_name(queue.STATE.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
  with tmp.open("x",encoding="utf-8",newline="\n") as f:
   json.dump(self.state,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
  os.replace(tmp,queue.STATE)
 def _update(self,key,**kw):
  with self._state_lock:self.state["tasks"][key].update(kw);self._save()
 def _reserve(self,key):return queue.PilotQueue._reserve(self,key)
 def _execute(self,task):return queue.PilotQueue._execute(self,task)
 def _run_group(self,rows):return queue.PilotQueue._run_group(self,rows)
 def run(self):
  try:
   self.state["status"]="training_fresh_arms";self._save()
   p={r["task_id"]:r for r in self.plan}
   for keys in (("train_encoder_graph","train_joint"),("predict_encoder_graph","predict_joint"),("score",)):
    if not self._run_group([p[k] for k in keys]):
     self.state["status"]="failed"
     for k in self.deps:
      if self.state["tasks"][k].get("status")=="queued":self.state["tasks"][k]["status"]="blocked_by_prior_failure"
     self._save();return 1
   self.state["status"]="complete";self._save();return 0
  finally:
   if not self.lock.closed:queue.fcntl.flock(self.lock.fileno(),queue.fcntl.LOCK_UN);self.lock.close()

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv)
 if a.dry_run:print(json.dumps({"status":"recovery_prechecked","tasks":recovery_plan(),"preflights_reused":True,
   "fresh_updates_per_arm":256,"prior_partial_updates":"unknown_discarded"},indent=2));return 0
 return RecoveryQueue().run()
if __name__=="__main__":raise SystemExit(main())
