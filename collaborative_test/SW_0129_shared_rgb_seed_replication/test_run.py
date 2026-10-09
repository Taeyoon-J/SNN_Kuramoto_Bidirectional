import hashlib,json,sys,unittest,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from collaborative_test.SW_0129_shared_rgb_seed_replication import run

class RunnerContractTests(unittest.TestCase):
 def test_fixed_seed0_lambda_provenance_checks_artifact_bytes(self):
  temp=HERE/('tmp_'+uuid.uuid4().hex[:8]);temp.mkdir()
  ref=temp/'seed0.json';artifact=temp/'warm.pt'
  try:
   artifact.write_bytes(b'original-warmup-fixture')
   digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
   row={'status':'passed','seed':0,'lambda':run.SEED0_LAMBDA,
    'source_core_sha256':'aac44697906c835569e3bf900e5930ce900223d1aa95c0bedba899bee5ff155d',
    'warmup_artifact_sha256':digest,'warmup_artifact':str(artifact)}
   ref.write_text(json.dumps(row),encoding='utf-8')
   with unittest.mock.patch.object(run,'SEED0_REFERENCE',ref),unittest.mock.patch.object(run,'SEED0_REFERENCE_SHA256',hashlib.sha256(ref.read_bytes()).hexdigest()),unittest.mock.patch.object(run,'SEED0_WARMUP_SHA256',digest):
    checked=run.validate_seed0_reference()
    self.assertEqual(checked['lambda'],23250.431374718348)
    artifact.write_bytes(b'changed')
    with self.assertRaises(AssertionError):run.validate_seed0_reference()
  finally:
   for child in temp.iterdir():child.unlink()
   temp.rmdir()

 def test_copied_seed0_reference_is_exact_reviewed_record(self):
  self.assertEqual(run.sha(run.SEED0_REFERENCE),run.SEED0_REFERENCE_SHA256)
  row=json.loads(run.SEED0_REFERENCE.read_text(encoding='utf-8-sig'))
  self.assertEqual(row['lambda'],run.SEED0_LAMBDA)
  self.assertEqual(row['warmup_artifact_sha256'],run.SEED0_WARMUP_SHA256)
  self.assertEqual(row['status'],'passed')

 def test_seed0_is_not_a_new_training_target_and_old_gate_is_absent(self):
  self.assertEqual(run.SOURCE_ROOT.name,'SW0095_full70k_aligned_loss')
  self.assertEqual(run.SEED0_LAMBDA,23250.431374718348)
  self.assertEqual(run.EXPECTED_SOURCE95_SHA[1],'b8b5d1d794dc0185b68526b713893f6d842518fde9c21d681bcdbe50549d43f6')
  self.assertEqual(run.EXPECTED_CONTROL97_SHA[2],'798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661')
  self.assertFalse(hasattr(run,'seed0_expansion_gate'))
  self.assertFalse(hasattr(run,'full_train'))
  self.assertFalse(hasattr(run,'full_evaluate'))
  with self.assertRaises(ValueError):run.train(0,'candidate',None)

 def test_registered_pair_has_only_new_seeds_and_dependency_order(self):
  from collaborative_test.SW_0129_shared_rgb_seed_replication import coordinator
  tasks=coordinator.task_plan()
  self.assertEqual(len(tasks),10)
  self.assertEqual({x['seed'] for x in tasks},{1,2})
  for t in tasks:
   if t['stage']=='train':
    self.assertEqual(t['depends_on'],[f"sw0129_preflight_s{t['seed']}"])
   if t['stage']=='evaluate':
    self.assertEqual(t['depends_on'],[f"sw0129_train_s{t['seed']}_{t['arm']}"])
  for task in tasks:
   path=coordinator.artifact_path(task).resolve()
   if task['stage']=='preflight': self.assertEqual(path.parent,run.ARCHIVE.resolve())
   else: self.assertEqual(path.parent.parent,run.OUT.resolve())

 def test_gradient_norm_rejects_nonfinite_values(self):
  import torch
  with self.assertRaises(FloatingPointError):run.grad_norm([torch.tensor([1.0,float('nan')])])

 def test_lambda_reference_rejects_altered_recipe_or_nonpassed_record(self):
  temp=HERE/('tmp_'+uuid.uuid4().hex[:8]);temp.mkdir();ref=temp/'seed0.json';warm=temp/'warm.pt'
  try:
   warm.write_bytes(b'warm');digest=hashlib.sha256(b'warm').hexdigest()
   row={'status':'failed','seed':0,'lambda':run.SEED0_LAMBDA,'source_core_sha256':'aac44697906c835569e3bf900e5930ce900223d1aa95c0bedba899bee5ff155d','warmup_artifact_sha256':digest,'warmup_artifact':str(warm)}
   ref.write_text(json.dumps(row),encoding='utf-8')
   with unittest.mock.patch.object(run,'SEED0_REFERENCE',ref),unittest.mock.patch.object(run,'SEED0_REFERENCE_SHA256',hashlib.sha256(ref.read_bytes()).hexdigest()),unittest.mock.patch.object(run,'SEED0_WARMUP_SHA256',digest):
    with self.assertRaises(AssertionError):run.validate_seed0_reference()
    row['status']='passed';row['lambda']+=1;ref.write_text(json.dumps(row),encoding='utf-8')
    with self.assertRaises(AssertionError):run.validate_seed0_reference()
  finally:
   for child in temp.iterdir():child.unlink()
   temp.rmdir()

if __name__=='__main__':unittest.main()
