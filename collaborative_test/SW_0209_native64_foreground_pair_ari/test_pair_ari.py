import unittest
import numpy as np
import torch
from collaborative_test.SW_0185_qce_spatial_connectivity.completion_audit import image_metrics
from .pair_ari import foreground_pair_ari_loss,foreground_pair_objective
from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss

class PairARITests(unittest.TestCase):
    def test_hard_equivalence_matches_actual_foreground_ARI(self):
        rng=np.random.RandomState(209)
        for _ in range(25):
            target=rng.randint(0,5,25);predicted=rng.randint(0,6,25)
            q=torch.tensor((predicted[:,None]==predicted[None,:]).astype(float),dtype=torch.float64)
            actual=1-float(foreground_pair_ari_loss(q,torch.tensor(target)))
            self.assertAlmostEqual(actual,image_metrics(predicted.reshape(5,5),target.reshape(5,5))[0],places=12)

    def test_perfect_and_degenerate_partitions(self):
        for labels in ([0,1,1,2,2],[1,1,1],[1,2,3],[0,0],[0,1]):
            y=torch.tensor(labels);q=(y[:,None]==y[None,:]).double()
            self.assertEqual(float(foreground_pair_ari_loss(q,y)),0.)

    def test_background_and_diagonal_do_not_change_foreground_pair_score(self):
        y=torch.tensor([0,0,1,1,2,2]);q=torch.full((6,6),.3,dtype=torch.float64)
        expected=foreground_pair_ari_loss(q,y);q[:2]=.9;q[:,:2]=.7;q.fill_diagonal_(.1)
        self.assertEqual(float(foreground_pair_ari_loss(q,y)),float(expected))

    def test_object_label_ids_and_node_order_do_not_matter(self):
        y=torch.tensor([0,1,1,2,2,3]);q=torch.rand((6,6),generator=torch.Generator().manual_seed(209),dtype=torch.float64)
        index=torch.tensor([5,3,0,1,4,2]);renamed=torch.tensor([0,5,5,8,8,11])
        self.assertAlmostEqual(float(foreground_pair_ari_loss(q,y)),float(foreground_pair_ari_loss(q[index][:,index],renamed[index])),places=13)

    def test_analytical_gradient_matches_finite_difference(self):
        y=torch.tensor([0,1,1,2,2]);q=torch.full((5,5),.4,dtype=torch.float64,requires_grad=True)
        gradient=torch.autograd.grad(foreground_pair_ari_loss(q,y),q)[0]
        for i,j in ((1,2),(1,3),(3,4)):
            plus=q.detach().clone();minus=plus.clone();plus[i,j]+=1e-6;minus[i,j]-=1e-6
            numeric=(foreground_pair_ari_loss(plus,y)-foreground_pair_ari_loss(minus,y))/2e-6
            self.assertAlmostEqual(float(gradient[i,j]),float(numeric),places=8)
        self.assertLess(float(gradient[1,2]),0);self.assertGreater(float(gradient[1,3]),0)

    def test_complete_objective_keeps_original_QCE(self):
        y=torch.tensor([0,0,1,1,2,2]);q=torch.full((6,6),.5,dtype=torch.float64)
        self.assertAlmostEqual(float(foreground_pair_objective(q,y)),float(_vectorized_class_loss(q,y)+foreground_pair_ari_loss(q,y)),places=12)

    def test_new_term_has_credit_through_actual_production_correlation_pipeline(self):
        from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import native_component_rho_q
        rng=torch.Generator().manual_seed(209)
        shared=torch.randn((1,4,1,32),generator=rng)
        emitted=(shared+.4*torch.randn((1,4,6,32),generator=rng)).requires_grad_()
        _,q=native_component_rho_q(emitted.mean(1),emitted,settle=16)
        loss=foreground_pair_ari_loss(q[0],torch.tensor([0,0,1,1,2,2]))
        gradient=torch.autograd.grad(loss,emitted)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)
        self.assertEqual(float(gradient[...,:16].abs().sum()),0.)

    def test_loss_can_train_a_pair_affinity_without_masks_at_inference(self):
        y=torch.tensor([0,0,1,1,2,2]);logits=torch.nn.Parameter(torch.zeros((6,6),dtype=torch.float64));optimizer=torch.optim.Adam([logits],lr=.1)
        before=float(foreground_pair_ari_loss(logits.sigmoid(),y).detach())
        for _ in range(30):
            optimizer.zero_grad();loss=foreground_pair_ari_loss(logits.sigmoid(),y);loss.backward();optimizer.step()
        self.assertLess(float(foreground_pair_ari_loss(logits.sigmoid(),y).detach()),before-.4)

    def test_invalid_geometry_values_labels_and_temperature_rejected(self):
        y=torch.tensor([0,1,1]);q=torch.eye(3)
        for invalid in (q[:2],q*float('nan'),q+2):
            with self.assertRaises(ValueError):foreground_pair_ari_loss(invalid,y)
        with self.assertRaises(ValueError):foreground_pair_ari_loss(q,y.float())
        with self.assertRaises(ValueError):foreground_pair_objective(q,y,temperature=0)

if __name__=='__main__':unittest.main()
