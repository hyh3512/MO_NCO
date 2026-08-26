from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_frozen_cells import (
    FrozenCellManifestError,
    OBJECTIVE_ARITHMETIC_V15,
    canonical_manifest_payload,
)
from mo_nco.pareto_independent_replica_runner import (
    ACCEPTANCE_SEMANTICS_V15,
    INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
    ReplicaTypeConfiguration,
    replica_configuration_sha256,
    replica_stream_plan_sha256,
    run_independent_replica_batch,
)
from mo_nco.pareto_v15_context import V15CertificateContext


class IndependentReplicaRunnerV15Tests(unittest.TestCase):
    def setUp(self) -> None:
        matrix_a = (
            (0.0, 1.0, 2.0, 3.0),
            (1.0, 0.0, 4.0, 2.0),
            (2.0, 4.0, 0.0, 1.0),
            (3.0, 2.0, 1.0, 0.0),
        )
        matrix_b = (
            (0.0, 3.0, 2.0, 1.0),
            (3.0, 0.0, 1.0, 2.0),
            (2.0, 1.0, 0.0, 4.0),
            (1.0, 2.0, 4.0, 0.0),
        )
        self.instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (matrix_a, matrix_b),
            name="v15-replica-test",
        )
        self.configurations = (
            ReplicaTypeConfiguration(
                type_id="balanced",
                reference_direction=(0.5, 0.5),
                beta_schedule=(0.0, 0.5, 1.0),
                mutation_steps_by_stage=(2, 3),
                replica_count=3,
                global_refresh_probability=0.25,
            ),
        )

    def _write_manifest(
        self,
        directory: str,
        *,
        upper: Fraction = Fraction(20),
        observable_cells: tuple[tuple[int, ...], ...] = (
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ),
    ) -> tuple[Path, str]:
        payload = canonical_manifest_payload(
            lower=(0, 0),
            upper=(upper, upper),
            widths=(upper / 2, upper / 2),
            observable_cells=observable_cells,
        )
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        path = Path(directory) / "cells.json"
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def _context(
        self,
        manifest_sha256: str,
        *,
        master_seed: int,
    ) -> V15CertificateContext:
        return V15CertificateContext(
            case_id="v15-replica-test",
            instance_sha256=instance_sha256(self.instance),
            configuration_sha256=replica_configuration_sha256(
                self.configurations
            ),
            cell_manifest_sha256=manifest_sha256,
            reference_sha256="4" * 64,
            type_cell_plan_sha256="5" * 64,
            pilot_plan_sha256=replica_stream_plan_sha256(
                self.configurations,
                master_seed=master_seed,
                stream_role="pilot",
                cell_manifest_sha256=manifest_sha256,
            ),
            confirm_plan_sha256=replica_stream_plan_sha256(
                self.configurations,
                master_seed=master_seed,
                stream_role="confirm",
                cell_manifest_sha256=manifest_sha256,
            ),
        )

    def test_batch_is_named_as_independent_mh_and_charges_exact_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write_manifest(directory)
            result = run_independent_replica_batch(
                self.instance,
                cell_manifest_path=path,
                certificate_context=self._context(digest, master_seed=123),
                configurations=self.configurations,
                master_seed=123,
                stream_role="confirm",
            )

        self.assertEqual(
            result.algorithm_id,
            INDEPENDENT_REPLICA_ALGORITHM_ID_V15,
        )
        self.assertFalse(result.population_interaction_present)
        self.assertFalse(result.resampling_performed)
        self.assertEqual(result.acceptance_semantics, ACCEPTANCE_SEMANTICS_V15)
        self.assertEqual(
            result.endpoint_classification_semantics,
            OBJECTIVE_ARITHMETIC_V15,
        )
        self.assertEqual(result.exact_total_evaluations, 3 * (1 + 2 + 3))
        self.assertTrue(all(endpoint.observable_cell_hit for endpoint in result.endpoints))
        self.assertTrue(
            all(endpoint.evaluations == 6 for endpoint in result.endpoints)
        )

    def test_pilot_and_confirm_domains_have_different_replay_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write_manifest(directory)
            pilot = run_independent_replica_batch(
                self.instance,
                cell_manifest_path=path,
                certificate_context=self._context(digest, master_seed=456),
                configurations=self.configurations,
                master_seed=456,
                stream_role="pilot",
            )
            confirm = run_independent_replica_batch(
                self.instance,
                cell_manifest_path=path,
                certificate_context=self._context(digest, master_seed=456),
                configurations=self.configurations,
                master_seed=456,
                stream_role="confirm",
            )

        self.assertNotEqual(
            [endpoint.derived_seed for endpoint in pilot.endpoints],
            [endpoint.derived_seed for endpoint in confirm.endpoints],
        )
        self.assertIn("ideal_product", result_text := pilot.probability_semantics)
        self.assertIn("replay_only", result_text)

    def test_manifest_observable_family_is_used_and_hash_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write_manifest(
                directory,
                observable_cells=(),
            )
            result = run_independent_replica_batch(
                self.instance,
                cell_manifest_path=path,
                certificate_context=self._context(digest, master_seed=789),
                configurations=self.configurations,
                master_seed=789,
                stream_role="pilot",
            )
            self.assertFalse(
                any(endpoint.observable_cell_hit for endpoint in result.endpoints)
            )
            with self.assertRaises(FrozenCellManifestError):
                run_independent_replica_batch(
                    self.instance,
                    cell_manifest_path=path,
                    certificate_context=self._context(
                        "0" * 64,
                        master_seed=789,
                    ),
                    configurations=self.configurations,
                    master_seed=789,
                    stream_role="pilot",
                )

    def test_exact_endpoint_outside_frozen_box_fails_without_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write_manifest(
                directory,
                upper=Fraction(1),
                observable_cells=((0, 0),),
            )
            with self.assertRaises(FrozenCellManifestError):
                run_independent_replica_batch(
                    self.instance,
                    cell_manifest_path=path,
                    certificate_context=self._context(
                        digest,
                        master_seed=101112,
                    ),
                    configurations=self.configurations,
                    master_seed=101112,
                    stream_role="confirm",
                )

    def test_context_configuration_and_stream_plan_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self._write_manifest(directory)
            context = self._context(digest, master_seed=123)
            for tampered in (
                replace(context, configuration_sha256="9" * 64),
                replace(context, confirm_plan_sha256="8" * 64),
            ):
                with self.subTest(tampered=tampered):
                    with self.assertRaisesRegex(
                        ValueError,
                        "configuration hash|stream-plan hash",
                    ):
                        run_independent_replica_batch(
                            self.instance,
                            cell_manifest_path=path,
                            certificate_context=tampered,
                            configurations=self.configurations,
                            master_seed=123,
                            stream_role="confirm",
                        )


if __name__ == "__main__":
    unittest.main()

