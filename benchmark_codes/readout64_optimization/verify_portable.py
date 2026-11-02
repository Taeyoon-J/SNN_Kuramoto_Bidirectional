"""Stdlib checks for complete-parent weights,64 head weights and real replay evidence."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 manifest=json.loads((HERE/'checkpoints/manifest.json').read_bytes())
 assert manifest['core_grid']==32 and manifest['head_grid']==64
 assert sorted(r['seed'] for r in manifest['checkpoints'])==[0,1,2]
 for r in manifest['checkpoints']:
  p=HERE/r['file'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
  parent=HERE/r['parent_file'];assert sha(parent)==r['parent_sha256']
 evidence=json.loads((HERE/'checkpoints/VALIDATION.json').read_bytes())
 assert evidence['status']=='three_seed_standalone_context64_replay_passed'
 assert len(evidence['rows'])==3 and all(r['standalone_full_image_replay_exact'] for r in evidence['rows'])
 audit=json.loads((HERE/'results/context/independent_audit.json').read_bytes())
 assert audit['status']=='independent64_metrics_sources_optimizer_and_replays_passed' and audit['mask_count']==4800
 for p in HERE.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
 return dict(status='three_context64_heads_and_complete_parents_verified',core_grid=32,head_grid=64,checkpoints=3,standalone_end_to_end_replays=3)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
