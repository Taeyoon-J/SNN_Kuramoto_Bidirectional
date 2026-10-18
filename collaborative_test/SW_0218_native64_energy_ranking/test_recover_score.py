import ast,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import recover_score as recovery


class RecoveryTests(unittest.TestCase):
    def test_original_fault_is_control_import_only_and_correct_control_exists(self):
        tree=ast.parse(Path(recovery.evaluate.__file__).read_text(encoding='utf-8'))
        modules=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertIn('collaborative_test.SW_0204_native64_energy_ranking',modules)
        self.assertEqual(recovery.control.HERE.name,'SW_0204_native64_component_credit')
        self.assertIs(recovery.comparison.train,recovery.control)

    def test_no_GT_before_actual_candidate_validation(self):
        with patch.object(recovery.evaluate,'validated_candidate',side_effect=ValueError('pending candidate')),patch.object(recovery.evaluate.screen_score,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):recovery.score()
            reader.assert_not_called()

    def test_control_validation_must_finish_before_GT(self):
        with patch.object(recovery.evaluate,'validated_candidate',return_value=(None,{'bindings':{}})),patch.object(recovery.comparison,'validated_candidate',side_effect=ValueError('control')),patch.object(recovery.evaluate.screen_score,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):recovery.score()
            reader.assert_not_called()

    def test_full_scoring_branch_executes_with_fixed_masks_and_no_training(self):
        gt=np.zeros((32,64,64),np.int64);gt[:,4:28,4:28]=1;gt[:,36:60,36:60]=2
        correct=gt[:16].copy();merged=(correct>0).astype(np.int64)
        predictions=dict(energy_ranking=correct,control=merged,warm=merged)
        report=dict(ranking_parameter_credit_verified=True,new_accepted_updates=2,CNN_changed_from_warm=True,
                    new_descent_steps=1,committed_model_changed=True)
        bundle=(predictions,{},report,{'control_protocol_sha256':'control'},
                {'model_sha256':'candidate','training_report_sha256':'report'})
        with patch.object(recovery,'validated_bundle',return_value=bundle),patch.object(recovery,'file_sha',return_value='sha'),patch.object(recovery.evaluate.screen_score,'read_ground_truth',return_value=gt),patch.object(recovery.train,'write_json') as writer,patch.object(recovery.train,'run_training') as trainer:
            summary=recovery.score()
        self.assertEqual(summary['scores']['energy_ranking']['fg_ari'],1.)
        self.assertGreater(summary['paired_energy_ranking_minus']['control']['fg_ari']['mean'],0)
        self.assertFalse(summary['score_recovery']['training_repeated'])
        self.assertEqual(writer.call_count,2);trainer.assert_not_called()


if __name__=='__main__':unittest.main()
