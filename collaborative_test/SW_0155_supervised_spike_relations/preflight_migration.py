"""Narrow AST-proven receipt migration for SW0155 JSON newline serialization only."""
from __future__ import annotations

import ast
import copy
import hashlib
import json

from collaborative_test.SW_0155_supervised_spike_relations import preflight, train

TRAIN_KEY = "collaborative_test/SW_0155_supervised_spike_relations/train.py"
OLD_TRAIN_SHA256 = "e8e177d37609584af02b1c78f12a443eb55982d32458917a06629d8962d92d8d"
NEW_TRAIN_SHA256 = "68306bc5fa658487302dbcb20d0e95676957dbdbd3b04c471292cbfb4126db69"
MIGRATED_REPORT_NAME = "preflight_seed1_serialization_fix.json"
OLD_REPORT_NAME = "preflight_seed1.json"
EXPECTED_LINES = (361, 362, 376, 377)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize_serialization_ast(old_source: bytes, new_source: bytes):
    """Prove the source change is exactly four backslash-n to newline literals."""
    if sha256_bytes(old_source) != OLD_TRAIN_SHA256 or sha256_bytes(new_source) != NEW_TRAIN_SHA256:
        raise ValueError("SW0155 serialization migration source hashes are not the reviewed pair")
    old_tree = ast.parse(old_source.decode("utf-8"))
    new_tree = ast.parse(new_source.decode("utf-8"))
    changed = []
    for node in ast.walk(old_tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.lineno in EXPECTED_LINES and node.value == "\\n":
            changed.append(node.lineno)
            node.value = "\n"
    if tuple(sorted(changed)) != EXPECTED_LINES:
        raise ValueError(f"expected exactly four serialization literals; saw {changed}")
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError("SW0155 training AST has changes beyond newline serialization")
    return {"changed_lines": list(EXPECTED_LINES),
            "normalized_old_ast_sha256": hashlib.sha256(
                ast.dump(old_tree, include_attributes=False).encode()).hexdigest(),
            "new_ast_sha256": hashlib.sha256(
                ast.dump(new_tree, include_attributes=False).encode()).hexdigest()}


def create_migrated_report(old_report, old_train_source: bytes, new_train_source: bytes,
                           current_fingerprint=None):
    """Clone the actual admission receipt, changing only its train.py fingerprint."""
    report = copy.deepcopy(old_report)
    current_fingerprint = current_fingerprint or train.implementation_fingerprint()
    old_fingerprint = report.get("implementation_fingerprint")
    if not isinstance(old_fingerprint, dict) or old_fingerprint.get(TRAIN_KEY) != OLD_TRAIN_SHA256:
        raise ValueError("preflight report is not bound to the known pre-fix trainer")
    if (set(current_fingerprint) != set(old_fingerprint)
            or any(current_fingerprint[key] != old_fingerprint[key]
                   for key in old_fingerprint if key != TRAIN_KEY)
            or current_fingerprint.get(TRAIN_KEY) != NEW_TRAIN_SHA256):
        raise ValueError("current trainer fingerprint differs outside the reviewed serialization edit")
    ast_proof = normalize_serialization_ast(old_train_source, new_train_source)
    if not preflight.validate_report(report, expected_fingerprint=old_fingerprint):
        raise ValueError("original actual-source preflight receipt fails its original validator")
    parent_report_sha = hashlib.sha256(json.dumps(old_report, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    report["implementation_fingerprint"] = current_fingerprint
    report["serialization_migration"] = {
        "status": "metadata_only_ast_equivalent",
        "old_train_sha256": OLD_TRAIN_SHA256,
        "new_train_sha256": NEW_TRAIN_SHA256,
        "parent_report_canonical_sha256": parent_report_sha,
        "ast_proof": ast_proof,
    }
    if not preflight.validate_report(report, expected_fingerprint=current_fingerprint):
        raise ValueError("migrated receipt does not pass the frozen SW0155 report validator")
    return report


def validate_migrated_report(old_report, migrated_report, *, current_fingerprint=None):
    """Check a deployed migration receipt is a strict clone of its parent report."""
    try:
        current_fingerprint = current_fingerprint or train.implementation_fingerprint()
        old_fingerprint = old_report["implementation_fingerprint"]
        migration = migrated_report["serialization_migration"]
        if (migration.get("status") != "metadata_only_ast_equivalent"
                or migration.get("old_train_sha256") != OLD_TRAIN_SHA256
                or migration.get("new_train_sha256") != NEW_TRAIN_SHA256
                or migration.get("parent_report_canonical_sha256") != hashlib.sha256(
                    json.dumps(old_report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                or migration.get("ast_proof", {}).get("changed_lines") != list(EXPECTED_LINES)
                or migrated_report.get("implementation_fingerprint") != current_fingerprint):
            return False
        expected = copy.deepcopy(old_report)
        expected["implementation_fingerprint"] = current_fingerprint
        expected["serialization_migration"] = migration
        if migrated_report != expected:
            return False
        return (preflight.validate_report(old_report, expected_fingerprint=old_fingerprint)
                and preflight.validate_report(migrated_report,
                                              expected_fingerprint=current_fingerprint))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
