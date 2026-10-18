import unittest,uuid
from pathlib import Path
import numpy as np
import torch
from . import engine


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        torch.set_num_threads(1)
        root=engine.library.cache.ROOT/'collaborative_testing/temporary'/('sw0213_'+uuid.uuid4().hex);root.mkdir(parents=True)
        x=np.random.RandomState(213).normal(size=(4,8,8,8)).astype(np.float32)
        y=np.zeros((4,8,8),np.int64);y[:,1:3,1:3]=1;y[:,5:7,5:7]=2
        return root,x,y,dict(fixture='TRAIN_only_no_DEV')

    def test_continued_Adam_matches_uninterrupted_and_parent_is_preserved(self):
        root,x,y,protocol=self.fixture();lib=engine.library
        parent=lib.fit(x,y,protocol,device='cpu',steps=2,snapshots=(2,),work=root/'parent_work',output=root/'parent')
        before=(root/'parent_work/boundary.ptx').read_bytes()
        child=engine.extend(x,y,protocol,device='cpu',parent_steps=2,steps=4,parent_work=root/'parent_work',work=root/'child_work',output=root/'child')
        full=lib.fit(x,y,protocol,device='cpu',steps=4,snapshots=(4,),work=root/'full_work',output=root/'full')
        self.assertEqual(child['final_model_sha256'],full['final_model_sha256'])
        self.assertEqual([r['loss'] for r in child['history']],[r['loss'] for r in full['history']])
        self.assertEqual(child['history'][:2],parent['history']);self.assertEqual(child['actual_Adam_steps'],[4])
        self.assertEqual((root/'parent_work/boundary.ptx').read_bytes(),before)
        resumed=engine.extend(x,y,protocol,device='cpu',parent_steps=2,steps=4,parent_work=root/'parent_work',work=root/'child_work',output=root/'child')
        self.assertEqual(resumed['final_model_sha256'],child['final_model_sha256'])

    def test_changed_TRAIN_data_is_rejected_before_child_creation(self):
        root,x,y,protocol=self.fixture();engine.library.fit(x,y,protocol,device='cpu',steps=2,snapshots=(2,),work=root/'parent_work',output=root/'parent')
        with self.assertRaises(ValueError):engine.extend(x+.1,y,protocol,device='cpu',parent_steps=2,steps=4,parent_work=root/'parent_work',work=root/'child_work',output=root/'child')
        self.assertFalse((root/'child_work').exists())


if __name__=='__main__':unittest.main()
