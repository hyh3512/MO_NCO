from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mo_nco.pareto_v13_spec import (
    V13_ALGORITHM_ID,
    V13_PROTOCOL_SPECIFICATION_SCHEMA,
    V13_RECEIPT_POLICY,
    load_v13_protocol_specification,
)


class V13ProtocolSpecificationTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "schema": V13_PROTOCOL_SPECIFICATION_SCHEMA,
            "bindings": {
                "instance_sha256": "1" * 64,
                "pareto_smc_specification_sha256": "2" * 64,
                "anchor_certificate_specification_sha256": "3" * 64,
                "full_reference_certificate_specification_sha256": "5" * 64,
                "sparse_compression_certificate_sha256": "4" * 64,
            },
            "identity": {
                "run_id": "run-01",
                "case_id": "case-01",
                "algorithm_id": V13_ALGORITHM_ID,
            },
            "receipt_authority": {
                "signer_key_id": "offline-key-01",
                "ed25519_public_key_hex": "05" * 32,
                "authorization_policy": V13_RECEIPT_POLICY,
            },
            "seed_derivation": {
                "schema": "test-domain-separated-seed-v1",
            },
            "full_sweep_checkpoints": {
                "evaluation_counts": [16, 32],
            },
            "assignment_preflight": {
                "desired_target_mass_lower_bounds_by_anchor_cell": [
                    0.01,
                    0.02,
                ],
                "pilot_failure_budgets_by_anchor_cell": [0.01, 0.01],
                "confirm_failure_budgets_by_anchor_cell": [0.01, 0.01],
                "mutually_exclusive_anchor_cells": True,
            },
        }

    def test_loads_exact_hash_bound_contract(self) -> None:
        payload = self._payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v13.json"
            path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            specification = load_v13_protocol_specification(path)
        self.assertEqual(specification.algorithm_id, V13_ALGORITHM_ID)
        self.assertEqual(
            specification.requested_full_sweep_checkpoints,
            (16, 32),
        )
        self.assertEqual(len(specification.signer_public_key_raw), 32)
        self.assertEqual(len(specification.sha256), 64)

    def test_rejects_unknown_fields_and_noncanonical_grids(self) -> None:
        bad_payloads = []
        extra = self._payload()
        extra["unexpected"] = True
        bad_payloads.append(extra)
        duplicate = self._payload()
        duplicate["full_sweep_checkpoints"] = {
            "evaluation_counts": [16, 16],
        }
        bad_payloads.append(duplicate)
        false_disjoint = self._payload()
        assignment = dict(false_disjoint["assignment_preflight"])
        assignment["mutually_exclusive_anchor_cells"] = False
        false_disjoint["assignment_preflight"] = assignment
        bad_payloads.append(false_disjoint)
        bad_identity = self._payload()
        identity = dict(bad_identity["identity"])
        identity["run_id"] = "run with spaces"
        bad_identity["identity"] = identity
        bad_payloads.append(bad_identity)
        for index, payload in enumerate(bad_payloads):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "bad.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_v13_protocol_specification(path)

    def test_rejects_duplicate_json_fields(self) -> None:
        encoded = json.dumps(self._payload())
        duplicated = (
            '{"schema":"'
            + V13_PROTOCOL_SPECIFICATION_SCHEMA
            + '",'
            + encoded[1:]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(duplicated, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_v13_protocol_specification(path)

    def test_rejects_boolean_and_string_probabilities(self) -> None:
        bad_payloads = []
        boolean_mass = self._payload()
        assignment = dict(boolean_mass["assignment_preflight"])
        assignment[
            "desired_target_mass_lower_bounds_by_anchor_cell"
        ] = [True, 0.02]
        boolean_mass["assignment_preflight"] = assignment
        bad_payloads.append(boolean_mass)
        string_budget = self._payload()
        assignment = dict(string_budget["assignment_preflight"])
        assignment["pilot_failure_budgets_by_anchor_cell"] = [
            "0.01",
            0.01,
        ]
        string_budget["assignment_preflight"] = assignment
        bad_payloads.append(string_budget)
        for index, payload in enumerate(bad_payloads):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"numeric-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "JSON number"):
                    load_v13_protocol_specification(path)


if __name__ == "__main__":
    unittest.main()

