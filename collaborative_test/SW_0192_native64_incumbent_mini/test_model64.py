import unittest
from types import SimpleNamespace
import numpy as np
import torch
from . import model64 as model


class MappingTests(unittest.TestCase):
    def test_parent_layout_and_every_native64_node(self):
        p=model.parent_index().reshape(64,64)
        self.assertEqual(p.shape,(64,64))
        self.assertTrue(torch.equal(p[::2,::2],torch.arange(1024).reshape(32,32)))
        self.assertTrue(torch.equal(torch.bincount(p.flatten()),torch.full((1024,),4)))
        self.assertTrue(torch.equal(p[::2,::2],p[1::2,1::2]))

    def test_both_pair_axes_nodes_shared_identity_distance(self):
        source={'kuramoto.omega':torch.arange(8.).reshape(4,2),
                'kuramoto.direction_learner':torch.arange(16.).reshape(4,4),
                'sc':torch.eye(4),'graph_generator.grid_distance':torch.zeros(4,4),
                'shared':torch.tensor([.123,.456])}
        target={name:torch.empty((16,2) if name=='kuramoto.omega' else
            (2,) if name=='shared' else (16,16)) for name in source}
        result=model.map_state(source,target,old_grid=2,grid=4);p=model.parent_index(2,4)
        self.assertTrue(torch.equal(result['kuramoto.omega'],source['kuramoto.omega'][p]))
        self.assertTrue(torch.equal(result['kuramoto.direction_learner'],source['kuramoto.direction_learner'][p][:,p]))
        self.assertTrue(torch.equal(result['sc'],torch.eye(16)))
        self.assertEqual(result['graph_generator.grid_distance'][0,1],4.)
        self.assertTrue(torch.equal(result['shared'],source['shared']))
        result['shared'].zero_();self.assertGreater(source['shared'].sum(),0)

    def test_unregistered_or_non_nested_geometry_rejected(self):
        with self.assertRaises(ValueError):model.parent_index(32,63)
        with self.assertRaises(ValueError):model.map_state({'unknown':torch.zeros(4)}, {'unknown':torch.zeros(16)},old_grid=2,grid=4)
        with self.assertRaises(ValueError):model.map_state({}, {'missing':torch.ones(1)})

    def test_encode_native64_retains_actual_RGB_gradient(self):
        from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
        encoder=torch.nn.Conv2d(3,8,3,bias=False)
        rgb=torch.randint(0,256,(1,128,128,3),dtype=torch.uint8)
        gamma=model.encode64(encoder,FeaturePatchGammaInitializer(grid_size=64),torch.tensor(0.),torch.tensor(1.),3.,rgb)
        self.assertEqual(gamma.shape,(1,8,4096));gamma.square().mean().backward()
        self.assertGreater(encoder.weight.grad.norm(),0)
        with self.assertRaises(ValueError):model.encode64(encoder,None,0.,1.,3.,rgb.float())

    def test_moving_device_updates_nested_recurrent_allocators(self):
        wrapped=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Linear(1,1))
        wrapped.device='stale';wrapped[1].device='stale'
        found=SimpleNamespace(wrapped=wrapped,encoder=torch.nn.Linear(1,1),patcher=torch.nn.Identity())
        old_ids=[id(p) for p in wrapped.parameters()]
        model.move_to_device(found,'cpu')
        self.assertEqual(wrapped.device,torch.device('cpu'))
        self.assertEqual(wrapped[1].device,torch.device('cpu'))
        self.assertEqual(old_ids,[id(p) for p in wrapped.parameters()])

    def test_geodesic_memory_chunk_preserves_values_and_gradients(self):
        from collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator import geodesic_distance_power2
        from collaborative_test.SW_0135_native32_spike_binding.resolution import grid_distance
        torch.manual_seed(192)
        z=torch.nn.functional.normalize(torch.randn(1,9,3),dim=-1).requires_grad_()
        distance=grid_distance(3,.5)
        a=geodesic_distance_power2(z,distance,torch.tensor(8.),steps=2,chunk=1)
        ga=torch.autograd.grad(a.sum(),z)[0]
        b=geodesic_distance_power2(z,distance,torch.tensor(8.),steps=2,chunk=8)
        gb=torch.autograd.grad(b.sum(),z)[0]
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-5)

if __name__=='__main__':unittest.main()
