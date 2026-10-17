"""Exercise the real audit entry point against the actual score producer ID."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import completion_audit_v2 as audit, score


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
            with self.assertRaisesRegex(ValueError, 'unexpected score producer'):
                audit.audit_completed()
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
