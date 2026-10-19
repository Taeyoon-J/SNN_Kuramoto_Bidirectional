"""Expand the accepted235 fixed models, preserving the complete64 prefix."""
import json
import numpy as np
from collaborative_test.SW_0235_native64_convergence_validation64 import run as reference

model = reference.scope.model
OFFSETS = tuple(range(320))
MINI = tuple(range(64))
NEW = tuple(range(64, 320))
canonical_rgb = reference.scope.canonical_rgb


def reference_file(name):
    paths = [reference.ARCHIVE / 'score_v1/summary.json', reference.ARCHIVE / 'completed_v1/summary.json'] if name == 'score' else [reference.ARCHIVE / 'completion_audit_v1/summary.json', reference.ARCHIVE / 'completed_v1/completion_audit.json']
    present = [p for p in paths if p.exists()]
    if not present or any(p.read_bytes() != present[0].read_bytes() for p in present):
        raise ValueError('accepted235 evidence missing or inconsistent')
    return present[0]


def selection_fingerprint():
    score_path = reference_file('score'); audit_path = reference_file('audit')
    score = json.loads(score_path.read_bytes()); audit = json.loads(audit_path.read_bytes())
    if (score['fingerprint'] != reference.fingerprint() or not score['balanced_candidate_gate']
            or audit['score_sha256'] != reference.sha(score_path.read_bytes())
            or audit['diagnostic_source_sha256'] != reference.sha((reference.HERE / 'completion_audit.py').read_bytes())
            or not audit['balanced_candidate_gate'] or not audit['original16_gate_failure_preserved']
            or audit['raw_masks_independently_rescored'] != 576
            or audit['independent_metrics_means_and_bootstrap_maximum_abs_error'] > 1e-12):
        raise ValueError('independently accepted fixed225 candidate required')
    return dict(score_sha256=reference.sha(score_path.read_bytes()), audit_sha256=reference.sha(audit_path.read_bytes()),
                source_fingerprint=reference.fingerprint())


def restore(found, seed):
    selection = selection_fingerprint()
    report, old_ref, original16, original_frequency = reference.scope.restore(found, seed)
    cache = reference.validated_seed(seed)
    protocol = cache['protocol']; binding = protocol['binding']
    audit = json.loads(reference_file('audit').read_bytes())
    expected = audit['candidate_references'][str(seed)]
    if (binding['candidate'] != old_ref or expected['candidate'] != old_ref
            or expected['protocol_sha256'] != reference.sha((reference.archive_for(seed) / 'protocol.json').read_bytes())):
        raise ValueError('different accepted235 checkpoint or prediction protocol')
    frequency = np.stack([reference.reader.read_row(reference.archive_for(seed) / 'rows' / f'{i:04d}', binding, i)[1] for i in MINI])
    if not np.array_equal(cache['labels'][list(reference.scope.MINI)], original16) or not np.array_equal(frequency[list(reference.scope.MINI)], original_frequency):
        raise ValueError('original232 mini does not match accepted64 cache')
    ref = dict(old_ref, validation64_score_sha256=selection['score_sha256'], validation64_audit_sha256=selection['audit_sha256'],
               validation64_protocol_sha256=expected['protocol_sha256'], validation64_labels_sha256=protocol['labels_sha256'],
               validation64_RGB_sha256=binding['RGB_sha256'], validation64_baseline_sha256=binding['baseline_sha256'])
    return report, ref, cache['labels'], frequency
