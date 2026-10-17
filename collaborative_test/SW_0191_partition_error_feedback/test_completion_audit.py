"""Exercise the real audit entry point against the actual score producer ID."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import completion_audit as audit, score


class ProducerContract(unittest.TestCase):
    def test_real_producer_id_reaches_cpu_validation_before_gt(self):
        tree = ast.parse(Path(score.__file__).read_text(encoding='utf-8'))
        ids = {node.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
               for key, node in zip(node.keys, node.values)
               if isinstance(key, ast.Constant) and key.value == 'experiment'
               and isinstance(node, ast.Constant) and isinstance(node.value, str)
               and node.value.endswith('_three_seed_score')}
        self.assertEqual(len(ids), 1)
        path = SimpleNamespace(read_bytes=lambda: json.dumps(dict(status='complete', experiment=ids.pop())).encode())
        with patch.object(score, 'SCORE_PATH', path), patch.object(
                score, 'prepare_score_inputs_cpu', side_effect=RuntimeError('CPU validation reached')):
            with self.assertRaisesRegex(RuntimeError, 'CPU validation reached'):
                audit.audit_completed()

    def test_wrong_producer_still_rejected_before_cpu_or_gt(self):
        path = SimpleNamespace(read_bytes=lambda: json.dumps(dict(status='complete', experiment='other')).encode())
        with patch.object(score, 'SCORE_PATH', path), patch.object(score, 'prepare_score_inputs_cpu') as prepare:
            with self.assertRaisesRegex(ValueError, 'unexpected completed score producer'):
                audit.audit_completed()
            prepare.assert_not_called()


class RealArithmetic(unittest.TestCase):
    def test_archived_seed_vectors_and_current_control_recompute(self):
        import copy
        import numpy as np
        original=score._read_sw0186_score()['payload']
        rows=copy.deepcopy(original['scores'])
        for endpoint in score.ENDPOINTS:
            rows.pop(f'SW0181_{endpoint}')
            rows[f'SW0186_{endpoint}']=copy.deepcopy(rows[f'candidate_{endpoint}'])
        indices=np.random.RandomState(178).randint(0,320,size=(10000,320))
        paired,gate=score._aggregate_paired_and_gate(rows,indices)
        report=dict(scores=rows,paired_bootstrap=paired,gate=gate)
        verified,_=audit.verify_arithmetic(report)
        self.assertFalse(verified['primary_gate_passed'])
        report['gate']['primary_gate_passed']=True
        with self.assertRaisesRegex(ValueError,'gate recomputation'):
            audit.verify_arithmetic(report)

if __name__ == '__main__':
    unittest.main()
