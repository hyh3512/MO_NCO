from __future__ import annotations

import itertools
import json
import math
import unittest

from mo_nco.pareto_bounds import (
    OutOfBoxError,
    _igd_p,
    _igd_plus_p,
    certify_pareto_bounds,
    hypervolume_minimization,
    normalized_cell_index,
)


class ParetoBoundCertificateTests(unittest.TestCase):
    def test_igd_uses_local_p_norm_then_arithmetic_reference_mean(self) -> None:
        approximation = ((0.0, 0.0),)
        reference = ((0.0, 0.0), (2.0, 0.0))

        self.assertEqual(_igd_p(approximation, reference, 2.0), 1.0)
        self.assertEqual(_igd_p(approximation, reference, math.inf), 1.0)
        self.assertEqual(_igd_plus_p(approximation, reference, 2.0), 0.0)

    def test_external_terminal_weights_define_the_empirical_cell_measure(self) -> None:
        report = certify_pareto_bounds(
            feasible_objectives=((0.1, 0.9), (0.9, 0.1)),
            particle_objectives=((0.1, 0.9), (0.9, 0.1)),
            particle_weights=(0.9, 0.1),
            objective_lower=(0.0, 0.0),
            objective_upper=(1.0, 1.0),
            normalized_cell_widths=(0.2, 0.2),
            target_pareto_cell_probabilities={(0, 4): 0.9, (4, 0): 0.1},
            declared_p_min=0.1,
            cellwise_mse_constant_B_L=1e-8,
            confidence_delta=0.05,
        )

        self.assertEqual(
            report["observed"]["particle_weight_source"],
            "externally_supplied_normalized",
        )
        self.assertAlmostEqual(
            report["observed"]["pareto_cell_empirical_masses"]["0,4"],
            0.9,
        )
        self.assertAlmostEqual(
            report["observed"]["pareto_cell_empirical_masses"]["4,0"],
            0.1,
        )

    def test_hit_pareto_cells_certify_geometry_in_both_unit_systems(self) -> None:
        report = certify_pareto_bounds(
            feasible_objectives=(
                (2.0, 16.0),
                (2.4, 15.0),
                (5.0, 8.0),
                (6.0, 9.0),
                (6.0, 18.0),
            ),
            particle_objectives=((2.4, 15.0), (6.0, 9.0)),
            objective_lower=(0.0, 0.0),
            objective_upper=(10.0, 20.0),
            normalized_cell_widths=(0.25, 0.5),
            target_pareto_cell_probabilities={(0, 1): 0.2, (2, 0): 0.3},
            declared_p_min=0.2,
            cellwise_mse_constant_B_L=1e-4,
            confidence_delta=0.05,
            igd_p=2.0,
            normalized_hv_reference=(1.0, 1.0),
        )

        self.assertEqual(report["schema"], "pareto_smc_geometric_bound_certificate_v2")
        self.assertEqual(
            report["metric_semantics"]["p_scope"],
            "local_objective_space_norm_only",
        )
        self.assertEqual(report["pareto"]["cells"], [[0, 1], [2, 0]])
        self.assertTrue(report["observed"]["all_pareto_cells_hit"])
        self.assertTrue(report["observed"]["additive_componentwise_coverage_verified"])
        self.assertLessEqual(
            report["observed"]["igd_p_normalized"],
            report["geometry_bounds"]["igd_p_normalized"],
        )
        self.assertLessEqual(
            report["observed"]["hv_deficit_original"],
            report["geometry_bounds"]["hv_deficit_original"],
        )
        self.assertEqual(
            report["geometry_bounds"]["additive_widths_original"],
            [2.5, 10.0],
        )

    def test_cells_are_half_open_except_for_the_global_upper_boundary(self) -> None:
        widths = (0.25, 0.5)
        self.assertEqual(normalized_cell_index((0.0, 0.0), widths), (0, 0))
        self.assertEqual(
            normalized_cell_index((0.249999999, 0.499999999), widths),
            (0, 0),
        )
        self.assertEqual(normalized_cell_index((0.25, 0.5), widths), (1, 1))
        self.assertEqual(normalized_cell_index((1.0, 1.0), widths), (3, 1))

    def test_out_of_box_particle_fails_instead_of_being_clipped(self) -> None:
        with self.assertRaisesRegex(OutOfBoxError, r"particle_objectives\[0\]\[0\]"):
            certify_pareto_bounds(
                feasible_objectives=((0.1, 0.9), (0.9, 0.1)),
                particle_objectives=((1.01, 0.1),),
                objective_lower=(0.0, 0.0),
                objective_upper=(1.0, 1.0),
                normalized_cell_widths=(0.1, 0.1),
                target_pareto_cell_probabilities={(1, 9): 0.2, (9, 1): 0.2},
                declared_p_min=0.2,
                cellwise_mse_constant_B_L=0.001,
                confidence_delta=0.05,
            )

    def test_missing_cell_is_an_explicit_fail_and_breaks_geometry_bounds(self) -> None:
        # Exhaustively enumerating this three-point feasible set gives three
        # Pareto cells. Repeating only the first point misses two cells. The
        # same-cell geometry theorem must not be applied: both actual IGD and
        # actual HV deficit exceed their cell-hit bounds.
        report = certify_pareto_bounds(
            feasible_objectives=((0.05, 0.95), (0.5, 0.5), (0.95, 0.05)),
            particle_objectives=((0.05, 0.95),) * 20,
            objective_lower=(0.0, 0.0),
            objective_upper=(1.0, 1.0),
            normalized_cell_widths=(0.05, 0.05),
            target_pareto_cell_probabilities={
                (1, 18): 0.2,
                (10, 10): 0.2,
                (18, 1): 0.2,
            },
            declared_p_min=0.2,
            cellwise_mse_constant_B_L=1e-8,
            confidence_delta=0.05,
        )

        self.assertEqual(report["verdict"], "FAIL_OBSERVED_COVERAGE")
        self.assertFalse(report["observed"]["all_pareto_cells_hit"])
        self.assertFalse(report["observed"]["additive_componentwise_coverage_verified"])
        self.assertGreater(
            report["observed"]["igd_p_normalized"],
            report["geometry_bounds"]["igd_p_normalized"],
        )
        self.assertGreater(
            report["observed"]["hv_deficit_normalized"],
            report["geometry_bounds"]["hv_deficit_normalized"],
        )

    def test_external_radius_can_certify_when_mse_gate_is_too_weak(self) -> None:
        report = certify_pareto_bounds(
            feasible_objectives=((0.1, 0.9), (0.9, 0.1)),
            particle_objectives=((0.1, 0.9), (0.9, 0.1)),
            objective_lower=(0.0, 0.0),
            objective_upper=(1.0, 1.0),
            normalized_cell_widths=(0.2, 0.2),
            target_pareto_cell_probabilities={(0, 4): 0.25, (4, 0): 0.25},
            declared_p_min=0.25,
            cellwise_mse_constant_B_L=1.0,
            confidence_delta=0.05,
            declared_cellwise_error_radius=0.1,
            declared_error_failure_probability=0.01,
        )

        assumption = report["finite_particle_assumption"]
        self.assertFalse(assumption["mse_requested_confidence_gate"])
        self.assertTrue(assumption["external_radius_gate"])
        self.assertEqual(assumption["selected_failure_probability_bound"], 0.01)
        self.assertEqual(report["verdict"], "PASS")

    def test_archive_filter_only_preserves_one_sided_not_ordinary_igd_bound(self) -> None:
        # The support witness (0.51, 0.51) shares the true point's cell, but is
        # removed by (0.0, 0.505). The latter still additively covers the true
        # point and has tiny IGD+, yet is far away in ordinary Euclidean IGD.
        report = certify_pareto_bounds(
            feasible_objectives=((0.5, 0.5), (0.51, 0.51), (0.0, 0.505)),
            particle_objectives=((0.51, 0.51), (0.0, 0.505)),
            objective_lower=(0.0, 0.0),
            objective_upper=(1.0, 1.0),
            normalized_cell_widths=(0.1, 0.1),
            target_pareto_cell_probabilities={(0, 5): 0.25, (5, 5): 0.25},
            declared_p_min=0.25,
            cellwise_mse_constant_B_L=1e-8,
            confidence_delta=0.05,
        )

        observed = report["observed"]
        bound = report["geometry_bounds"]["igd_p_normalized"]
        self.assertTrue(observed["all_pareto_cells_hit"])
        self.assertLessEqual(observed["support_ordinary_igd_p_normalized"], bound)
        self.assertGreater(observed["archive_ordinary_igd_p_normalized"], bound)
        self.assertLessEqual(observed["archive_igd_plus_p_normalized"], bound)
        self.assertTrue(
            observed["archive_additive_componentwise_coverage_verified"]
        )
        self.assertNotIn(
            "archive_ordinary_igd_p_at_most",
            report["high_probability_bounds"]["on_that_event"],
        )

    def test_all_nonempty_support_subsets_obey_the_cell_hit_gate(self) -> None:
        front = ((0.0, 0.75), (0.5, 0.5), (0.75, 0.0))
        probabilities = {(0, 3): 0.2, (2, 2): 0.2, (3, 0): 0.2}
        for mask in range(1, 1 << len(front)):
            support = tuple(
                point for index, point in enumerate(front) if mask & (1 << index)
            )
            with self.subTest(mask=mask):
                report = certify_pareto_bounds(
                    feasible_objectives=front,
                    particle_objectives=support,
                    objective_lower=(0.0, 0.0),
                    objective_upper=(1.0, 1.0),
                    normalized_cell_widths=(0.25, 0.25),
                    target_pareto_cell_probabilities=probabilities,
                    declared_p_min=0.2,
                    cellwise_mse_constant_B_L=1e-10,
                    confidence_delta=0.05,
                )
                if mask == (1 << len(front)) - 1:
                    self.assertEqual(report["verdict"], "PASS")
                else:
                    self.assertEqual(
                        report["verdict"],
                        "FAIL_OBSERVED_COVERAGE",
                    )
                self.assertIn(
                    "not a continuous interval claim",
                    report["claim_limit"],
                )
                json.dumps(report, allow_nan=False)

    def test_exact_three_dimensional_hv_matches_inclusion_exclusion(self) -> None:
        points = ((0.2, 0.8, 0.5), (0.6, 0.3, 0.4), (0.4, 0.5, 0.7))
        reference = (1.0, 1.0, 1.0)
        expected = 0.0
        for subset_size in range(1, len(points) + 1):
            sign = 1.0 if subset_size % 2 else -1.0
            for subset in itertools.combinations(points, subset_size):
                corner = tuple(max(point[d] for point in subset) for d in range(3))
                expected += sign * (
                    (1.0 - corner[0])
                    * (1.0 - corner[1])
                    * (1.0 - corner[2])
                )

        self.assertAlmostEqual(
            hypervolume_minimization(points, reference),
            expected,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()

