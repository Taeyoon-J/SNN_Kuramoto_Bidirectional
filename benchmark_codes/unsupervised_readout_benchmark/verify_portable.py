"""Stdlib integrity check for the published source snapshot and three full checkpoints."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):h.update(block)
 return h.hexdigest()
def verify():
 snapshot=json.loads((HERE/'snapshot/SHA256.json').read_bytes())
 for name,expected in snapshot.items():
  p=HERE/'snapshot/python'/name
  assert p.resolve().is_relative_to((HERE/'snapshot/python').resolve())
  assert digest(p)==expected,name
  if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
 manifest=json.loads((HERE/'checkpoints/manifest.json').read_bytes())
 assert manifest['includes_all_core_encoder_head_decoder_weights']
 assert sorted(r['seed'] for r in manifest['checkpoints'])==[0,1,2]
 for r in manifest['checkpoints']:
  p=HERE/r['file'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],r['file']
 validation=json.loads((HERE/'checkpoints/VALIDATION.json').read_bytes())
 assert validation['status']=='three_seed_portable_end_to_end_passed'
 assert len(validation['rows'])==3
 assert all(r['full_image_replay_exact'] and r['all_frozen_state_exact'] and r['fresh_process_snapshot_only'] for r in validation['rows'])
 return dict(status='three_portable_checkpoints_and_exact_source_snapshot_verified',checkpoint_bytes=[r['bytes'] for r in manifest['checkpoints']],snapshot_files=len(snapshot))
if __name__=='__main__':print(json.dumps(verify(),indent=2))
