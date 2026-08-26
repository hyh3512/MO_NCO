from __future__ import annotations

import unittest
from dataclasses import replace

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_execution_contract import (
    DOMAIN_SEPARATED_SEED_SCHEMA_V1,
    PARETO_SMC_V13_ALGORITHM_ROLE,
    PARETO_SMC_V13_ALGORITHM_VERSION,
    derive_domain_separated_seed,
    verify_domain_separated_seed,
    verify_full_type_sweep_checkpoints,
)
from mo_nco.pareto_smc import AnnealedParetoSMCOptimizer


class DomainSeparatedSeedTests(unittest.TestCase):
    def _derive(self, **overrides):  # type: ignore[no-untyped-def]
        payload = {
            "case_identity": "case_001",
            "instance_sha256": "a" * 64,
            "paired_seed": 7,
            "algorithm_role": PARETO_SMC_V13_ALGORITHM_ROLE,
            "algorithm_version": PARETO_SMC_V13_ALGORITHM_VERSION,
            "stream_role": "pilot",
            "schema": DOMAIN_SEPARATED_SEED_SCHEMA_V1,
        }
        payload.update(overrides)
        return derive_domain_separated_seed(**payload)

    def test_derivation_is_deterministic_and_uses_full_sha256_integer(self) -> None:
        first = self._derive()
        second = self._derive()
        self.assertEqual(first, second)
        self.assertEqual(first.seed, int(first.derivation_sha256, 16))
        verify_domain_separated_seed(first)

    def test_every_execution_domain_axis_changes_the_seed(self) -> None:
        seeds = {
            self._derive().seed,
            self._derive(case_identity="case_002").seed,
            self._derive(instance_sha256="b" * 64).seed,
            self._derive(paired_seed=8).seed,
            self._derive(algorithm_role="pareto-smc-control").seed,
            self._derive(algorithm_version="v13.1").seed,
            self._derive(stream_role="confirm").seed,
        }
        self.assertEqual(len(seeds), 7)

    def test_strict_domain_validation_rejects_ambiguous_values(self) -> None:
        invalid = (
            {"instance_sha256": "A" * 64},
            {"paired_seed": True},
            {"paired_seed": -1},
            {"paired_seed": 1 << 64},
            {"case_identity": " case_001"},
            {"algorithm_role": "Pareto-SMC"},
            {"algorithm_version": ""},
            {"stream_role": "diagnostic"},
            {"schema": "future_schema"},
        )
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    self._derive(**override)

    def test_forged_seed_preimage_is_rejected(self) -> None:
        seed = self._derive()
        forged = replace(seed, stream_role="confirm")
        with self.assertRaisesRegex(ValueError, "canonical preimage"):
            verify_domain_separated_seed(forged)


class FullTypeSweepCheckpointTests(unittest.TestCase):
    @staticmethod
    def _ledger():  # type: ignore[no-untyped-def]
        return (
            {
                "stage_index": 0,
                "evaluation_start": 0,
                "evaluation_end": 6,
                "evaluations": 6,
                "references": (
                    {"reference_index": 0, "mutation_attempts": 0},
                    {"reference_index": 1, "mutation_attempts": 0},
                    {"reference_index": 2, "mutation_attempts": 0},
                ),
            },
            {
                "stage_index": 1,
                "evaluation_start": 6,
                "evaluation_end": 12,
                "evaluations": 6,
                "references": (
                    {"reference_index": 0, "mutation_attempts": 2},
                    {"reference_index": 1, "mutation_attempts": 2},
                    {"reference_index": 2, "mutation_attempts": 2},
                ),
            },
            {
                "stage_index": 2,
                "evaluation_start": 12,
                "evaluation_end": 18,
                "evaluations": 6,
                "references": (
                    {"reference_index": 0, "mutation_attempts": 2},
                    {"reference_index": 1, "mutation_attempts": 2},
                    {"reference_index": 2, "mutation_attempts": 2},
                ),
            },
        )

    def _verify(self, **overrides):  # type: ignore[no-untyped-def]
        payload = {
            "stage_ledger": self._ledger(),
            "num_reference_types": 3,
            "particles_per_reference": 2,
            "total_evaluations": 18,
            "checkpoint_period": 6,
            "diagnostic_iterations": (6, 12, 18),
        }
        payload.update(overrides)
        return verify_full_type_sweep_checkpoints(**payload)

    def test_exact_stage_end_grid_passes(self) -> None:
        result = self._verify()
        self.assertEqual(result.ledger_gate, "PASS")
        self.assertEqual(result.gate, "PASS")
        self.assertEqual(result.verified_boundaries, (6, 12, 18))
        self.assertEqual(result.non_boundary_checkpoints, ())
        self.assertEqual(result.missing_diagnostic_checkpoints, ())

    def test_mid_type_or_mid_stage_checkpoint_fails_explicitly(self) -> None:
        result = self._verify(
            checkpoint_period=3,
            diagnostic_iterations=(3, 6, 9, 12, 15, 18),
        )
        self.assertEqual(result.ledger_gate, "PASS")
        self.assertEqual(result.gate, "FAIL")
        self.assertEqual(result.non_boundary_checkpoints, (3, 9, 15))
        self.assertIn(
            "REQUESTED_CHECKPOINT_NOT_FULL_TYPE_SWEEP_BOUNDARY",
            result.reasons,
        )

    def test_missing_genuine_diagnostic_fails(self) -> None:
        result = self._verify(diagnostic_iterations=(6, 18))
        self.assertEqual(result.gate, "FAIL")
        self.assertEqual(result.missing_diagnostic_checkpoints, (12,))

    def test_reference_coverage_and_evaluation_tampering_fail_closed(self) -> None:
        ledger = list(self._ledger())
        tampered_stage = dict(ledger[1])
        tampered_stage["references"] = (
            {"reference_index": 0, "mutation_attempts": 2},
            {"reference_index": 0, "mutation_attempts": 2},
            {"reference_index": 2, "mutation_attempts": 1},
        )
        tampered_stage["evaluation_start"] = 5
        ledger[1] = tampered_stage
        result = self._verify(stage_ledger=tuple(ledger))
        self.assertEqual(result.ledger_gate, "FAIL")
        self.assertEqual(result.gate, "FAIL")
        self.assertEqual(result.verified_boundaries, ())
        self.assertIn(
            "STAGE_1_REFERENCE_TYPE_COVERAGE_MISMATCH",
            result.reasons,
        )
        self.assertIn(
            "STAGE_1_NONCONTIGUOUS_EVALUATION_START",
            result.reasons,
        )
        self.assertIn(
            "STAGE_1_MUTATION_EVALUATION_SUM_MISMATCH",
            result.reasons,
        )

    def test_disabled_checkpoint_claim_is_not_run_but_ledger_is_verified(self) -> None:
        result = self._verify(
            checkpoint_period=None,
            diagnostic_iterations=(6, 12, 18),
        )
        self.assertEqual(result.ledger_gate, "PASS")
        self.assertEqual(result.gate, "NOT_RUN")


class ParetoSMCExecutionBindingTests(unittest.TestCase):
    @staticmethod
    def _optimizer(
        *,
        checkpoint_period: int,
        use_domain_seed: bool,
    ) -> AnnealedParetoSMCOptimizer:
        instance = MultiObjectiveTSPInstance.random_biobjective(6, seed=930)
        seed_contract = derive_domain_separated_seed(
            case_identity=instance.name,
            instance_sha256=instance_sha256(instance),
            paired_seed=11,
            algorithm_role=PARETO_SMC_V13_ALGORITHM_ROLE,
            algorithm_version=PARETO_SMC_V13_ALGORITHM_VERSION,
            stream_role="pilot",
        )
        return AnnealedParetoSMCOptimizer(
            instance=instance,
            particles_per_reference=2,
            evaluations=12,
            seed=seed_contract.seed if use_domain_seed else 11,
            beta_schedule=(0.0, 1.0, 2.0),
            reference_directions=((0.9, 0.1), (0.1, 0.9)),
            num_reference_types=2,
            resampling_policy="always",
            mutation_steps_by_stage=(1, 1),
            audit_trace_level="summary",
            anytime_checkpoint_period=checkpoint_period,
            domain_separated_seed=seed_contract if use_domain_seed else None,
        )

    def test_optimizer_binds_seed_and_full_sweep_metadata(self) -> None:
        result = self._optimizer(
            checkpoint_period=4,
            use_domain_seed=True,
        ).run()
        self.assertEqual(result.metadata["domain_separated_seed_gate"], "PASS")
        self.assertEqual(
            result.metadata["formal_full_type_sweep_checkpoint_gate"],
            "PASS",
        )
        self.assertEqual(
            result.metadata["verified_full_type_sweep_boundaries"],
            (4, 8, 12),
        )

    def test_optimizer_marks_mid_sweep_grid_fail(self) -> None:
        result = self._optimizer(
            checkpoint_period=2,
            use_domain_seed=False,
        ).run()
        self.assertEqual(
            result.metadata["formal_full_type_sweep_checkpoint_gate"],
            "FAIL",
        )
        self.assertEqual(
            result.metadata["non_boundary_full_type_sweep_checkpoints"],
            (2, 6, 10),
        )

    def test_optimizer_rejects_seed_bound_to_different_instance(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(6, seed=931)
        seed_contract = derive_domain_separated_seed(
            case_identity=instance.name,
            instance_sha256="a" * 64,
            paired_seed=11,
            algorithm_role=PARETO_SMC_V13_ALGORITHM_ROLE,
            algorithm_version=PARETO_SMC_V13_ALGORITHM_VERSION,
            stream_role="pilot",
        )
        with self.assertRaisesRegex(ValueError, "different instance"):
            AnnealedParetoSMCOptimizer(
                instance=instance,
                particles_per_reference=2,
                evaluations=12,
                seed=seed_contract.seed,
                beta_schedule=(0.0, 1.0, 2.0),
                reference_directions=((0.9, 0.1), (0.1, 0.9)),
                num_reference_types=2,
                resampling_policy="always",
                mutation_steps_by_stage=(1, 1),
                domain_separated_seed=seed_contract,
            )


if __name__ == "__main__":
    unittest.main()

