import unittest
from types import SimpleNamespace
import numpy as np
import torch
from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
from collaborative_test.SW_0188_incumbent_continuation import source

class SourceEncoderActivation(unittest.TestCase):
    def test_native_encoder_forward_unchanged_but_source_CNN_gets_gradient_and_Adam_update(self):
        encoder=ContextualFeatureEncoder(torch.nn.Conv2d(3,8,3),seed=186001)
        other=[torch.nn.Parameter(torch.tensor(.1)) for _ in range(3)]
        families={'context_encoder':list(encoder.context.named_parameters()),
            **{name:[(name,p)] for name,p in zip(('graph','native_core','integration'),other)}}
        opt=torch.optim.Adam([dict(name=name,parameter_names=[n for n,_ in rows],params=[p for _,p in rows],lr=.001)
            for name,rows in families.items()])
        found=SimpleNamespace(encoder=encoder)
        rgb=torch.rand(1,3,128,128)
        before=encoder(rgb).detach().clone();weights=[p.detach().clone() for p in encoder.base_encoder.parameters()]
        rng=torch.get_rng_state().clone()
        enabled=source.enable_source_encoder(found,opt,families)
        self.assertEqual(set(enabled),source.FAMILIES)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(before,encoder(rgb).detach()))
        self.assertEqual(len(opt.state),0)
        encoder(rgb).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.base_encoder.parameters()))
        self.assertGreater(sum(float(p.grad.norm()) for p in encoder.base_encoder.parameters()),0.)
        opt.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(weights,encoder.base_encoder.parameters())))
        self.assertEqual({int(opt.state[p]['step']) for p in encoder.base_encoder.parameters()},{1})
        self.assertEqual(opt.param_groups[-1]['lr'],3e-5)
        self.assertEqual(source._base_encoder_sha(found),source._base_encoder_sha_from_state(encoder.state_dict()))
        with self.assertRaises(ValueError):source.enable_source_encoder(found,opt,enabled)

    def test_source_group_credit_is_required_in_addition_to_original_four(self):
        row=dict(loss_to_emitted_component_spikes_gradient_l2=.1,gradient_norm_before_clip=1.,
            family_gradient_norms={k:.1 for k in source.FAMILIES},
            family_parameter_delta_l2={k:.01 for k in source.FAMILIES})
        self.assertEqual(source._credit_failures(row),[])
        row['family_gradient_norms']['source_base']=0.
        self.assertTrue(source._credit_failures(row))

    def test_only_RGB_encoder_is_exempt_from_frozen_component_guards(self):
        encoder=ContextualFeatureEncoder(torch.nn.Conv2d(3,8,3),seed=186001)
        found=SimpleNamespace(encoder=encoder,patcher=torch.nn.Identity(),
            wrapped=SimpleNamespace(core=SimpleNamespace(graph_generator=SimpleNamespace(geodesic_contrast=torch.tensor(.5)))),
            feature_mean=torch.zeros(8),feature_std=torch.ones(8),feature_clip=3.)
        before=source._frozen_source_components(found,encoder.base_encoder)
        self.assertNotIn('base_encoder',before)
        self.assertEqual(set(before),{'geodesic_contrast','patcher','feature_mean','feature_std','feature_clip'})
        with torch.no_grad():
            for p in encoder.base_encoder.parameters():p.add_(.1)
        self.assertEqual(source._frozen_source_components(found,encoder.base_encoder),before)
        found.feature_mean.add_(1.)
        self.assertNotEqual(source._frozen_source_components(found,encoder.base_encoder),before)

if __name__=='__main__':unittest.main()
