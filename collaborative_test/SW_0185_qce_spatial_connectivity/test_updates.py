import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
import torch
from collaborative_test.SW_0185_qce_spatial_connectivity import train,updates,objective
from collaborative_test.SW_0175_critical_component_binding import train_v2 as util
from collaborative_test.SW_0178_temporal_class_competition import preflight

class ToyWrapped(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.05)) for _ in range(4)])
    def project_integrations_(self):pass

class CompositeUpdateIntegration(unittest.TestCase):
    def test_actual_microbatch_loop_composite_full_loss_parity_and_Adam_credit(self):
        wrapped=ToyWrapped();found=SimpleNamespace(wrapped=wrapped,encoder=torch.nn.Identity())
        names=('context_encoder','graph','native_core','integration')
        optimizer=torch.optim.Adam([dict(name=name,params=[p],parameter_names=[name],lr=.001)
            for name,p in zip(names,wrapped.weights)])
        found._sw0185_optimizer=optimizer
        arrays=dict(rgb=np.zeros((16,2,2,3),dtype=np.uint8),
            modal_gt=np.tile(np.array([[0,0],[1,1]],dtype=np.int32),(16,1,1)))
        inputs=dict(arrays=arrays,ordered_train_ids=list(range(16)))
        def native(_found,rgb,*,live_tail_steps):
            n=rgb.shape[0]
            emitted=sum(wrapped.weights).expand(n,4,4,3)
            code=emitted.mean((1,3))
            pattern=torch.tensor([[.7,.7,0.,0.],[.7,.7,0.,0.],[0.,0.,.7,.7],[0.,0.,.7,.7]])
            rho=torch.tanh(code[:,None,:,None]+code[:,None,None,:]+pattern[None,None,:,:]).expand(-1,4,-1,-1)
            q=rho.clamp_min(0).prod(1)
            return None,{'component_spikes':emitted},None,code,q,rho
        original=objective.connectivity_loss
        flat=lambda values:torch.as_tensor(values).reshape(len(values),4).long()
        with mock.patch.object(util,'_native_forward',side_effect=native), \
             mock.patch.object(updates,'flatten_modal_targets',side_effect=flat), \
             mock.patch.object(preflight,'flatten_modal_targets',side_effect=flat), \
             mock.patch.object(objective,'connectivity_loss',side_effect=lambda q,rho,y:original(q,rho,y,grid=2)):
            indices=np.arange(16)
            before=train._full_batch_loss('qce_spatial_connectivity',found,indices,inputs,'cpu')
            row=updates._train_update('qce_spatial_connectivity',found,optimizer,indices,inputs,'cpu',1)
            self.assertAlmostEqual(before,row['mean_registered_loss'],places=6)
            self.assertGreater(row['mean_registered_loss'],row['mean_q_loss'])
            self.assertGreater(row['loss_to_emitted_component_spikes_gradient_l2'],0.)
            self.assertTrue(all(value>0 for value in row['family_gradient_norms'].values()))
            self.assertTrue(all(value>0 for value in row['family_parameter_delta_l2'].values()))
            self.assertEqual({int(value['step']) for value in optimizer.state.values()},{1})
            after=train._full_batch_loss('qce_spatial_connectivity',found,indices,inputs,'cpu')
            self.assertTrue(np.isfinite(after))

    def test_unregistered_loss_cannot_silently_fall_back_to_QCE(self):
        with self.assertRaises(ValueError):updates._loss_for_image('product_q_ce',None,None,None)

if __name__=='__main__':unittest.main()
