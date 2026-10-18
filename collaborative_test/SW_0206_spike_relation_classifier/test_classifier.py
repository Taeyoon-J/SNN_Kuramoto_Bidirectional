import io,unittest
import numpy as np
import torch
from .classifier import RelationClassifier,sample_pairs,fit,predict_affinity,state_sha


class ClassifierTests(unittest.TestCase):
    def test_balanced_relations_exclude_self_and_handle_unequal_classes(self):
        gt=np.array([[0,0,0,0],[0,0,1,2]],dtype=np.int64)
        i,j,y=sample_pairs(gt,seed=7,count=100)
        self.assertEqual(int(y.sum()),50);self.assertFalse(np.any(i==j))
        np.testing.assert_array_equal(gt.ravel()[i]==gt.ravel()[j],y.astype(bool))
        renamed=sample_pairs(gt*9+5,seed=7,count=100)
        for a,b in zip((i,j,y),renamed):np.testing.assert_array_equal(a,b)

    def test_invalid_pair_budget_and_one_class(self):
        with self.assertRaises(ValueError):sample_pairs(np.zeros((2,2),dtype=np.int64),seed=0)
        with self.assertRaises(ValueError):sample_pairs(np.array([[0,0],[1,1]]),seed=0,count=3)

    def test_actual_fit_learns_and_preserves_global_RNG(self):
        x=np.tile(np.array([[-.8,-.5,0.,.2],[.8,.5,0.,.2]],dtype=np.float32),(32,1));y=np.tile(np.array([0.,1.],dtype=np.float32),32)
        torch.manual_seed(19);rng=torch.get_rng_state().clone()
        head,opt,report=fit(x,y)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(report['parameters'],97);self.assertEqual(report['actual_Adam_steps'],[200])
        self.assertLess(report['final_loss'],report['initial_loss']*.7)
        self.assertGreater(report['balanced_pair_accuracy'],.95)
        raw=io.BytesIO();torch.save(head.state_dict(),raw);raw.seek(0)
        self.assertEqual(state_sha(torch.load(raw,weights_only=True)),report['final_parameter_sha256'])

    def test_affinity_symmetric_range_and_chunk_equivalence(self):
        torch.manual_seed(4);head=RelationClassifier();rho=torch.rand(4,5,5)*2-1
        a=predict_affinity(head,rho,chunk=3);b=predict_affinity(head,rho,chunk=25)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=0);torch.testing.assert_close(a,a.T,atol=0,rtol=0)
        self.assertTrue(((a>=0)&(a<=1)).all())
        with self.assertRaises(ValueError):predict_affinity(head,rho,chunk=0)

    def test_invalid_training_features(self):
        for x,y in [(np.zeros((3,3),np.float32),np.zeros(3,np.float32)),(np.full((2,4),np.nan,np.float32),np.array([0,1],np.float32))]:
            with self.assertRaises(ValueError):fit(x,y)
