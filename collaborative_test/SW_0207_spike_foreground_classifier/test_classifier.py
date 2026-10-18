import io,unittest
import numpy as np
import torch
from .classifier import ForegroundClassifier,sample_nodes,node_features,fit,predict_foreground,state_sha


class ForegroundTests(unittest.TestCase):
    def test_object_balanced_binary_node_samples(self):
        gt=np.array([[0,0,0,0],[0,1,2,2]],np.int64)
        indices,target=sample_nodes(gt,seed=8,count=100)
        self.assertEqual(int(target.sum()),50)
        np.testing.assert_array_equal(gt.ravel()[indices]>0,target.astype(bool))
        renamed=sample_nodes(np.where(gt>0,gt*7+3,0),seed=8,count=100)
        np.testing.assert_array_equal(indices,renamed[0])
        with self.assertRaises(ValueError):sample_nodes(np.zeros((2,2),np.int64),seed=0)
        with self.assertRaises(ValueError):sample_nodes(gt,seed=0,count=3)

    def test_real_trace_axis_and_settled_statistics(self):
        spikes=torch.full((1,4,3,4),999.);membrane=spikes.clone()
        spikes[...,2]=0;spikes[...,3]=1;membrane[...,2]=1;membrane[...,3]=3
        membrane[...,2:]+=torch.arange(3).reshape(1,1,3,1)*10
        features=node_features(dict(component_spikes=spikes,component_membrane=membrane),settle=2)
        self.assertEqual(tuple(features.shape),(1,3,12))
        torch.testing.assert_close(features[0,0],torch.tensor([.5]*4+[2.]*4+[1.]*4))
        torch.testing.assert_close(features[0,2],torch.tensor([.5]*4+[22.]*4+[1.]*4))
        with self.assertRaises(ValueError):node_features(dict(component_spikes=spikes,component_membrane=membrane),settle=4)

    def test_fit_normalizes_from_TRAIN_and_preserves_CPU_RNG(self):
        x=np.tile(np.array([[-.8]+[.7]*11,[.8]+[.7]*11],np.float32),(32,1));y=np.tile(np.array([0.,1.],np.float32),32)
        torch.manual_seed(11);rng=torch.get_rng_state().clone()
        head,opt,report=fit(x,y)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(report['parameters'],225)
        self.assertEqual(report['actual_Adam_steps'],[200]);self.assertGreater(report['balanced_node_accuracy'],.95)
        self.assertLess(report['final_loss'],report['initial_loss']*.7)
        torch.testing.assert_close(head.mean,torch.from_numpy(x).mean(0))
        self.assertTrue((head.scale>=1e-4).all())
        raw=io.BytesIO();torch.save(head.state_dict(),raw);raw.seek(0)
        self.assertEqual(state_sha(torch.load(raw,weights_only=True)),report['final_parameter_sha256'])

    def test_foreground_probability_shape_range_and_reordering(self):
        head=ForegroundClassifier();x=torch.randn(5,12)
        p=predict_foreground(head,x);self.assertEqual(tuple(p.shape),(5,));self.assertTrue(((p>=0)&(p<=1)).all())
        order=torch.tensor([4,2,0,1,3]);torch.testing.assert_close(predict_foreground(head,x[order]),p[order])
        with self.assertRaises(ValueError):predict_foreground(head,torch.zeros(5,4))

    def test_pre_grouping_support_can_remove_a_background_bridge(self):
        from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
        q=np.full((5,5),.9);a=np.zeros((5,5));full=np.ones((1,5),np.int64)
        before=foreground_induced_merge(q,a,1.,full,grid_shape=(1,5))['labels']
        support=np.array([[1,1,0,1,1]],np.int64)
        postfilter=np.where(support,before,0)
        regroup=foreground_induced_merge(q,a,1.,support,grid_shape=(1,5))['labels']
        self.assertEqual(postfilter[0,0],postfilter[0,4])
        self.assertNotEqual(regroup[0,0],regroup[0,4])
        np.testing.assert_array_equal(regroup>0,support>0)

    def test_reject_bad_feature_training_data(self):
        for x,y in [(np.zeros((3,4),np.float32),np.zeros(3,np.float32)),(np.full((2,12),np.nan,np.float32),np.array([0,1],np.float32))]:
            with self.assertRaises(ValueError):fit(x,y)
