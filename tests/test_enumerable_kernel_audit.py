from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from mo_nco.enumerable_kernel_audit import (
    audit_typed_mh_temperature_grid,
    write_enumerable_kernel_audit,
)
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.ips_certified import CertifiedSingleSiteIPSOptimizer
from mo_nco.run_enumerable_kernel_audit import build_parser


def _flat_symmetric_instance(num_cities: int = 4) -> MultiObjectiveTSPInstance:
    matrix = tuple(
        tuple(0.0 if i == j else 1.0 for j in range(num_cities))
        for i in range(num_cities)
    )
    return MultiObjectiveTSPInstance.from_distance_matrices(
        (matrix, matrix),
        name=f"flat_{num_cities}",
    )


class EnumerableKernelAuditTests(unittest.TestCase):
    def test_cli_default_laziness_matches_the_production_kernel(self) -> None:
        self.assertEqual(build_parser().parse_args([]).lazy_probability, 0.05)

    def test_frozen_context_and_typed_energy_match_production_kernel(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=507)
        optimizer = CertifiedSingleSiteIPSOptimizer(
            instance,
            num_particles=2,
            evaluations=2,
            seed=508,
            temperature=0.08,
            lazy_probability=0.4,
        )
        report = audit_typed_mh_temperature_grid(
            instance,
            num_particles=2,
            context_seed=508,
            temperatures=(0.08,),
            evaluation_budget=80,
            lazy_probability=0.4,
            max_states=10,
            max_product_states=100,
        )

        self.assertEqual(
            report["temperature_rows"][0]["production_kernel_context_hash"],
            optimizer.context_hash,
        )
        for state_index, objective in enumerate(
            report["state_space"]["objectives_by_tour_index"]
        ):
            for coordinate in range(2):
                expected = optimizer._typed_single_energy(objective, coordinate)
                actual = report["state_space"]["typed_energies_by_coordinate"][
                    coordinate
                ][state_index]
                self.assertAlmostEqual(actual, expected, places=14)

    def test_exact_rows_irreducibility_and_detailed_balance(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(5, seed=501)
        report = audit_typed_mh_temperature_grid(
            instance,
            num_particles=3,
            context_seed=502,
            temperatures=(0.05, 0.1),
            evaluation_budget=128,
            lazy_probability=0.5,
            max_states=100,
            max_product_states=100,
        )

        self.assertEqual(report["schema"], "enumerable_typed_mh_audit_v1")
        self.assertEqual(report["state_space"]["num_tours"], math.factorial(4))
        self.assertTrue(report["proposal_graph"]["connected"])
        self.assertEqual(len(report["temperature_rows"]), 2)
        for row in report["temperature_rows"]:
            self.assertLessEqual(row["max_row_sum_error_evaluation_clock"], 1e-12)
            self.assertLessEqual(
                row["max_row_sum_error_active_proposal_kernel"],
                1e-12,
            )
            self.assertLessEqual(row["max_db_flow_residual_evaluation_clock"], 1e-12)
            self.assertLessEqual(
                row["max_db_flow_residual_active_proposal_kernel"],
                1e-12,
            )
            self.assertLessEqual(row["max_stationarity_l1_residual_evaluation_clock"], 1e-12)
            self.assertTrue(row["all_coordinate_kernels_irreducible"])
            self.assertGreater(row["product_ordinary_gap_evaluation_clock"], 0.0)
            self.assertGreaterEqual(
                row["product_absolute_gap_evaluation_clock"],
                0.0,
            )

    def test_explicit_product_matrix_confirms_random_scan_gap_formula(self) -> None:
        report = audit_typed_mh_temperature_grid(
            MultiObjectiveTSPInstance.random_biobjective(4, seed=505),
            num_particles=2,
            context_seed=506,
            temperatures=(0.08,),
            evaluation_budget=96,
            lazy_probability=0.5,
            max_states=10,
            max_product_states=100,
        )
        explicit = report["temperature_rows"][0]["explicit_product_audit"]
        self.assertTrue(explicit["performed"])
        self.assertEqual(explicit["num_product_states"], 36)
        self.assertLessEqual(explicit["max_spectral_formula_residual"], 1e-12)
        self.assertLessEqual(
            explicit["evaluation_clock"]["db_max_abs_flow_residual"],
            1e-12,
        )
        self.assertLessEqual(
            explicit["active_proposal_kernel"]["row_sum_max_abs_error"],
            1e-12,
        )

    def test_lazy_evaluation_clock_repairs_active_kernel_periodicity(self) -> None:
        # With n=4 and a flat energy, all non-lazy 2-opt generators are odd
        # permutations. The active-proposal kernel is bipartite, while the
        # evaluated 1/2-lazy production kernel is aperiodic.
        report = audit_typed_mh_temperature_grid(
            _flat_symmetric_instance(),
            num_particles=1,
            context_seed=0,
            temperatures=(0.1,),
            evaluation_budget=64,
            lazy_probability=0.5,
            max_states=10,
            max_relative_stationary_excess=0.01,
            tv_tolerance=0.05,
        )
        row = report["temperature_rows"][0]

        self.assertAlmostEqual(
            row["product_slem_active_proposal_kernel"],
            1.0,
            places=12,
        )
        self.assertGreater(row["product_absolute_gap_evaluation_clock"], 0.0)
        self.assertTrue(row["mixing_gate_evaluation_clock"])
        self.assertIsInstance(row["required_evaluations_tv_bound"], int)
        self.assertEqual(report["h1_grid_verdict"], "NOT_FALSIFIED_ON_GRID")
        self.assertIn("including an explicit lazy identity", report["budget_clock_note"])

    def test_writes_machine_auditable_json_and_csv(self) -> None:
        report = audit_typed_mh_temperature_grid(
            MultiObjectiveTSPInstance.random_biobjective(4, seed=503),
            num_particles=2,
            context_seed=504,
            temperatures=(0.02, 0.08, 0.2),
            evaluation_budget=80,
            lazy_probability=0.5,
            max_states=10,
        )
        with tempfile.TemporaryDirectory() as tmp:
            json_path, csv_path = write_enumerable_kernel_audit(report, Path(tmp))
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["context"]["context_hash"], report["context"]["context_hash"])
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[0]["temperature"]), 0.02)
            self.assertIn("h1_feasible_on_grid", rows[0])

    def test_feasible_grid_points_are_not_reported_as_continuous_intervals(self) -> None:
        report = audit_typed_mh_temperature_grid(
            _flat_symmetric_instance(),
            num_particles=1,
            context_seed=0,
            temperatures=(0.05, 0.1),
            evaluation_budget=64,
            lazy_probability=0.5,
            max_states=10,
        )

        self.assertNotIn("feasible_temperature_grid_intervals", report)
        self.assertEqual(
            report["feasible_temperature_grid_runs"][0]["sampled_temperatures"],
            (0.05, 0.1),
        )
        self.assertIn("not as continuous intervals", report["claim_limit"])


if __name__ == "__main__":
    unittest.main()

