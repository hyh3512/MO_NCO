from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.pareto_smc_spec import (
    EXACT_INCREMENTAL_TWO_OPT_CONTRACT,
    analytic_objective_box,
    load_pareto_smc_specification,
    original_unit_cell_widths,
)


class ParetoSMCSpecificationTests(unittest.TestCase):
    def test_duplicate_json_fields_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        raw = (
            root / "benchmarks" / "pareto_smc_v1_spec.json"
        ).read_text(encoding="utf-8")
        duplicate = raw.replace(
            '"schema":',
            '"schema":"forged","schema":',
            1,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                load_pareto_smc_specification(
                    path,
                    objective_dimension=2,
                )

    def test_nested_claim_fields_are_exact_and_enforced(self) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads(
            (root / "benchmarks" / "pareto_smc_v1_spec.json").read_text(
                encoding="utf-8"
            )
        )
        variants = []
        false_ess = json.loads(json.dumps(baseline))
        false_ess["resampling"][
            "ess_is_not_a_coverage_certificate"
        ] = False
        variants.append((false_ess, "ess_is_not_a_coverage_certificate"))
        truncated_ledger = json.loads(json.dumps(baseline))
        truncated_ledger["reporting"]["cell_ledger"] = "truncated"
        variants.append((truncated_ledger, "cell_ledger"))
        unknown_nested = json.loads(json.dumps(baseline))
        unknown_nested["objective_box"]["posthoc"] = True
        variants.append((unknown_nested, "unexpected shape"))
        boolean_rho = json.loads(json.dumps(baseline))
        boolean_rho["target"]["chebyshev_rho"] = True
        variants.append((boolean_rho, "JSON number"))
        string_ess = json.loads(json.dumps(baseline))
        string_ess["resampling"]["ess_threshold_fraction"] = "0.9"
        variants.append((string_ess, "JSON number"))
        with tempfile.TemporaryDirectory() as temp:
            for index, (payload, message) in enumerate(variants):
                path = Path(temp) / f"invalid-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_pareto_smc_specification(
                        path,
                        objective_dimension=2,
                    )

    def test_bundled_spec_is_strict_and_converts_normalized_widths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = load_pareto_smc_specification(
            root / "benchmarks" / "pareto_smc_v1_spec.json",
            objective_dimension=2,
        )
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=801)
        lower, upper = analytic_objective_box(instance)
        widths = original_unit_cell_widths(instance, spec)

        self.assertEqual(len(spec.sha256), 64)
        self.assertEqual(len(spec.reference_directions), 8)
        self.assertEqual(spec.beta_schedule[0], 0.0)
        self.assertEqual(spec.mutation_proposal, "uniform_symmetric_two_opt")
        self.assertEqual(spec.mutation_objective_evaluation, "full_tour")
        self.assertEqual(spec.global_refresh_probability, 0.0)
        self.assertIsNone(spec.mutation_steps_by_stage)
        self.assertTrue(
            all(
                abs(width / (high - low) - 0.05) < 1e-15
                for width, low, high in zip(widths, lower, upper)
            )
        )

    def test_analytic_box_uses_the_runtime_outward_rounding_contract(self) -> None:
        matrix = (
            (0.0, 1.0, 4.0),
            (1.0, 0.0, 2.0),
            (4.0, 2.0, 0.0),
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (matrix,),
            name="outward-box-contract",
        )

        lower, upper = analytic_objective_box(instance)

        self.assertEqual(lower, (math.nextafter(3.0, -math.inf),))
        self.assertEqual(upper, (math.nextafter(12.0, math.inf),))

    def test_archive_dependent_cell_spec_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "benchmarks" / "pareto_smc_v1_spec.json").read_text(
                encoding="utf-8"
            )
        )
        payload["epsilon_cells"]["archive_independent"] = False
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "archive_independent"):
                load_pareto_smc_specification(path, objective_dimension=2)

    def test_reference_dimension_mismatch_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "length 3"):
            load_pareto_smc_specification(
                root / "benchmarks" / "pareto_smc_v1_spec.json",
                objective_dimension=3,
            )

    def test_mixture_requires_an_explicit_positive_refresh_probability(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "benchmarks" / "pareto_smc_v1_spec.json").read_text(
                encoding="utf-8"
            )
        )
        payload["mutation"]["proposal"] = (
            "local_two_opt_plus_uniform_global_refresh"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicitly declare"):
                load_pareto_smc_specification(path, objective_dimension=2)

            payload["mutation"]["global_refresh_probability"] = 0.05
            payload["mutation"]["steps_per_stage"] = [1, 2, 3, 4, 5]
            path.write_text(json.dumps(payload), encoding="utf-8")
            specification = load_pareto_smc_specification(
                path,
                objective_dimension=2,
            )
        self.assertEqual(
            specification.mutation_proposal,
            "local_two_opt_plus_uniform_global_refresh",
        )
        self.assertEqual(specification.global_refresh_probability, 0.05)
        self.assertEqual(
            specification.mutation_steps_by_stage,
            (1, 2, 3, 4, 5),
        )

    def test_exact_incremental_contract_is_explicitly_parsed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "benchmarks" / "pareto_smc_v1_spec.json").read_text(
                encoding="utf-8"
            )
        )
        payload["mutation"]["objective_evaluation"] = (
            EXACT_INCREMENTAL_TWO_OPT_CONTRACT
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incremental.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specification = load_pareto_smc_specification(
                path,
                objective_dimension=2,
            )
        self.assertEqual(
            specification.mutation_objective_evaluation,
            EXACT_INCREMENTAL_TWO_OPT_CONTRACT,
        )


if __name__ == "__main__":
    unittest.main()

