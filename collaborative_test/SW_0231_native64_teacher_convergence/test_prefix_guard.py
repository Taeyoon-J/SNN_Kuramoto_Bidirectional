import copy,unittest
import numpy as np
import torch
from . import prefix_guard as guard


class PrefixTests(unittest.TestCase):
    def fixture(self):
        return dict(completed_updates=128,wrapped_state={'weight':torch.ones(4,8)},encoder_state={'weight':torch.ones(2,2)},
            optimizer_state={'state':{0:{'step':torch.tensor(128.),'exp_avg':torch.ones(4,8)}},'param_groups':[{'params':[0],'lr':.0003}]},
            optimizer_schema=[{'name':'sensory_projection'}],history=[{'update':i+1,'loss':.01} for i in range(128)],
            rng={'torch_cpu':torch.get_rng_state()},bindings={'experiment':'control'})

    def test_metadata_changes_do_not_substitute_for_numerical_comparison(self):
        original=self.fixture();current=copy.deepcopy(original);current['bindings']={'experiment':'new'}
        guard.compare_payloads(current,original)

    def test_matching_weights_with_reset_Adam_is_rejected(self):
        original=self.fixture();current=copy.deepcopy(original);current['optimizer_state']['state'][0]['step'].zero_()
        with self.assertRaises(ValueError):guard.compare_payloads(current,original)

    def test_changed_encoder_or_history_is_rejected(self):
        for key in ('encoder_state','history'):
            original=self.fixture();current=copy.deepcopy(original)
            if key=='encoder_state':current[key]['weight'][0,0]+=.001
            else:current[key][0]['loss']+=.001
            with self.assertRaises(ValueError):guard.compare_payloads(current,original)

    def test_wrong_prefix_length_or_Torch_RNG_is_rejected(self):
        original=self.fixture();current=copy.deepcopy(original);current['completed_updates']=127
        with self.assertRaises(ValueError):guard.compare_payloads(current,original)
        current=copy.deepcopy(original);current['rng']['torch_cpu'][0]^=1
        with self.assertRaises(ValueError):guard.compare_payloads(current,original)


if __name__=='__main__':unittest.main()
