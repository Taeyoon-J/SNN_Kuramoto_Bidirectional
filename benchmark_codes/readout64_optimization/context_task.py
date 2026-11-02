"""R4 factory injection; source encoder/core are still unreachable by backward."""
import argparse,importlib.util,json,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['admission','train','extend']);p.add_argument('--root',required=True);p.add_argument('--code',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--arm');a=p.parse_args()
 sys.path.insert(0,a.code)
 import run as engine
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from context_model import ContextReadout,CONTEXT_ARMS
 loader=engine.original
 def original(seed):
  result=loader(seed);result[0]._seed_hint=seed;return result
 engine.original=original;engine.FineReadout=ContextReadout;engine.ARMS=CONTEXT_ARMS
 torch=engine.torch;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.15,0)
 root=Path(a.root)
 if a.stage=='admission':
  engine.admission(root,a.seed)
  p=root/f'admission_seed{a.seed}.json';value=json.loads(p.read_bytes());value['context_sources']={p.name:engine.sha(p) for p in Path(__file__).resolve().parent.glob('*.py')};engine.write(p,value)
 elif a.stage=='train':engine.train(root,a.seed,a.arm,128,8)
 else:
  parent=Path('/work/USERS/kevinswk/experiments/SW0260_readout64_r1');script=Path(json.loads((parent/'extension_launch.json').read_bytes())['code'])/'extend.py'
  spec=importlib.util.spec_from_file_location('sw0260_context_extend_engine',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  sys.argv=[str(script),'--root',str(root),'--code',a.code,'--seed',str(a.seed),'--arm',a.arm];module.main()
if __name__=='__main__':main()
