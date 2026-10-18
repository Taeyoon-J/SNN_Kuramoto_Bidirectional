import unittest,uuid
from unittest.mock import patch
import numpy as np
import torch
from . import components,training,run


class MergerTests(unittest.TestCase):
    def fixture(self):
        mask=np.zeros((8,8),np.int64);mask[1,1:3]=9;mask[3,3:5]=4;mask[6,6]=7
        x=np.random.RandomState(214).normal(size=(268,8,8)).astype(np.float32)
        return mask,x

    def test_zero_edge_readout_preserves_original_mask_exactly(self):
        mask,x=self.fixture();ids,pairs,table=components.component_pairs(x,mask,np.zeros(268),np.ones(268))
        out,info=components.merge(mask,ids,pairs,np.zeros(len(pairs)))
        np.testing.assert_array_equal(out,mask);self.assertEqual(info['merges'],0)
        head=training.PairHead();self.assertTrue(torch.equal(head(torch.from_numpy(table)),torch.zeros(len(table))))

    def test_complete_link_prevents_weak_relation_chaining(self):
        mask,x=self.fixture();ids,pairs,_=components.component_pairs(x,mask,np.zeros(268),np.ones(268))
        out,info=components.merge(mask,ids,pairs,[2.,-2.,1.])
        self.assertEqual(out[1,1],out[3,3]);self.assertNotEqual(out[1,1],out[6,6]);self.assertEqual(info['merges'],1)
        np.testing.assert_array_equal(out>0,mask>0)
        for label in ids:self.assertEqual(len(np.unique(out[mask==label])),1)

    def test_component_ID_permutation_preserves_features_and_partition(self):
        mask,x=self.fixture();other=mask.copy()
        for a,b in ((9,101),(4,5),(7,23)):other[mask==a]=b
        a,p,fa=components.component_pairs(x,mask,np.zeros(268),np.ones(268));b,q,fb=components.component_pairs(x,other,np.zeros(268),np.ones(268))
        np.testing.assert_array_equal(fa,fb);np.testing.assert_array_equal(p,q)
        left,_=components.merge(mask,a,p,[2.,-2.,1.]);right,_=components.merge(other,b,q,[2.,-2.,1.])
        np.testing.assert_array_equal(left.reshape(-1)[:,None]==left.reshape(-1),right.reshape(-1)[:,None]==right.reshape(-1))

    def test_soft_targets_ignore_background_and_account_for_mixed_components(self):
        mask,x=self.fixture();ids,pairs,_=components.component_pairs(x,mask,np.zeros(268),np.ones(268));gt=np.zeros_like(mask)
        gt[1,1]=1;gt[3,3:5]=1
        np.testing.assert_allclose(components.soft_targets(mask,gt,ids,pairs),[.5,0.,0.])

    def test_empty_and_single_component_are_valid(self):
        for source in (np.zeros((8,8),np.int64),np.ones((8,8),np.int64)):
            ids,pairs,table=components.component_pairs(np.zeros((268,8,8),np.float32),source,np.zeros(268),np.ones(268))
            out,_=components.merge(source,ids,pairs,np.zeros(len(pairs)));np.testing.assert_array_equal(source,out)
            self.assertEqual(table.shape,(0,548))

    def test_actual_Adam_resume_matches_full_run_and_changed_data_is_rejected(self):
        torch.set_num_threads(1);root=training.old.cache.ROOT/'collaborative_testing/temporary'/('sw0214_'+uuid.uuid4().hex);root.mkdir(parents=True)
        x=np.random.RandomState(14).normal(size=(8,8)).astype(np.float32);y=np.array([0,1]*4,np.float32);w=np.ones(8,np.float32);science={'fixture':'no_DEV'}
        full=training.fit(x,y,w,science,device='cpu',steps=4,work=root/'full_work',output=root/'full')
        self.assertIsNone(training.fit(x,y,w,science,device='cpu',steps=4,stop_after=2,work=root/'resume_work',output=root/'resume'))
        resumed=training.fit(x,y,w,science,device='cpu',steps=4,work=root/'resume_work',output=root/'resume')
        self.assertEqual(full['model_sha256'],resumed['model_sha256']);self.assertEqual(full['history'],resumed['history']);self.assertEqual(resumed['actual_Adam_steps'],[4])
        with self.assertRaises(ValueError):training.fit(x+.1,y,w,science,device='cpu',steps=4,work=root/'resume_work',output=root/'resume')

    def test_GT_is_not_read_if_prediction_provenance_fails(self):
        with patch.object(run,'validated_predictions',side_effect=ValueError('missing masks')),patch.object(run.parent.cache.source.full.metrics,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()

    def test_worker_requires_an_exclusive_assigned_GPU(self):
        from .queue import worker_command
        self.assertEqual(worker_command(3)[-2:],['--physical-gpu','3'])
        with self.assertRaises(ValueError):worker_command(True)


if __name__=='__main__':unittest.main()
