import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
import torch
from collaborative_test.SW_0191_partition_error_feedback import train,source,updates,feedback_context
from collaborative_test.SW_0179_qce_seed_replication import train as parent
from collaborative_test.SW_0175_critical_component_binding import train_v2 as native
from collaborative_test.SW_0178_temporal_class_competition import preflight

class Wrapped(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.05)) for _ in range(5)])
        self.core=SimpleNamespace(graph_generator=SimpleNamespace(log_coupling_gain=torch.tensor(0.)))
    def project_integrations_(self):pass

class FiveFamilyIntegration(unittest.TestCase):
    def test_real_composite_update_has_frozen_target_full_forward_parity_and_auxiliary_spike_credit(self):
        wrapped=Wrapped();found=SimpleNamespace(wrapped=wrapped,encoder=torch.nn.Identity())
        names=('context_encoder','graph','native_core','integration','source_base')
        opt=torch.optim.Adam([dict(name=name,params=[p],parameter_names=[name],lr=.001)
            for name,p in zip(names,wrapped.weights)])
        found._sw0191_optimizer=opt
        inputs=dict(arrays=dict(rgb=np.zeros((16,2,2,3),dtype=np.uint8),
            modal_gt=np.tile(np.array([[0,0],[1,1]],dtype=np.int32),(16,1,1))),ordered_train_ids=list(range(16)))
        def forward(_found,rgb,*,live_tail_steps):
            emitted=(sum(wrapped.weights)+torch.rand(())*.001).expand(rgb.shape[0],4,4,3)
            code=emitted.mean((1,3))
            pattern=torch.tensor([[.7,.7,0.,0.],[.7,.7,0.,0.],[0.,0.,.7,.7],[0.,0.,.7,.7]])
            rho=torch.tanh(code[:,None,:,None]+code[:,None,None,:]+pattern[None,None,:,:]).expand(-1,4,-1,-1)
            return None,{'component_spikes':emitted},torch.ones(rgb.shape[0],4,4),code,rho.clamp_min(0).prod(1),rho
        feedback_context.install(found,list(range(16)),{i:np.array([0,1,0,1]) for i in range(16)})
        context_sha=feedback_context.context_sha(found)
        flatten=lambda value:torch.as_tensor(value).reshape(len(value),4).long()
        with mock.patch.object(native,'_native_forward',side_effect=forward), \
             mock.patch.object(parent,'flatten_modal_targets',side_effect=flatten), \
             mock.patch.object(preflight,'flatten_modal_targets',side_effect=flatten), \
             mock.patch.object(updates,'flatten_modal_targets',side_effect=flatten), \
             mock.patch.object(feedback_context.masks,'_qcc_labels',return_value=np.zeros((4,2,2),dtype=np.int32)), \
             mock.patch.object(feedback_context.solver,'foreground_induced_merge',return_value={'labels':np.array([[0,1],[0,1]],dtype=np.int32)}) as partition:
            rng=train.tx.capture_rng_state();caller=torch.get_rng_state().clone()
            before=train._prepare_feedback_and_measure(found,np.arange(16),inputs,'cpu',evaluation_rng=rng)
            self.assertTrue(torch.equal(caller,torch.get_rng_state()))
            self.assertEqual(feedback_context.context_sha(found),context_sha)
            row=updates._train_update('product_q_ce',found,opt,np.arange(16),inputs,'cpu',1)
            after=train._measure_preserving_rng('product_q_ce',found,np.arange(16),inputs,'cpu',evaluation_rng=rng)
            self.assertTrue(np.isfinite(after))
            self.assertEqual(partition.call_count,16)
        self.assertAlmostEqual(before,row['mean_registered_loss'],places=6)
        self.assertGreater(row['mean_feedback_loss'],0.)
        self.assertAlmostEqual(row['mean_registered_loss'],row['mean_q_loss']+row['mean_feedback_loss'],places=6)
        self.assertGreater(row['feedback_loss_to_emitted_S_gradient_l2'],0.)
        self.assertGreater(row['feedback_trainable_gradient_rss'],0.)
        self.assertEqual(set(row['feedback_family_gradient_rss']),set(names))
        self.assertEqual(feedback_context.context_sha(found),context_sha)
        self.assertEqual(source._credit_failures(row),[])
        self.assertEqual(set(row['family_gradient_norms']),set(names))
        self.assertEqual({int(v['step']) for v in opt.state.values()},{1})

if __name__=='__main__':unittest.main()
