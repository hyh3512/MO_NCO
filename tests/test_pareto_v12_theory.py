from __future__ import annotations

import math
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

from mo_nco.pareto_regeneration_certificate import (
    AssignmentPilotNonemptinessPreflight,
    assignment_pilot_nonemptiness_preflight,
    confirm_cell_certificate,
    deterministic_target_mass_lower_bound,
    enumerate_equal_dual_stream_schedules,
    evaluate_joint_certificate_design,
    finite_suite_hoeffding_half_width,
    heterogeneous_pilot_confirm_budget,
    minimum_independent_units_for_hoeffding_half_width,
    minimum_pilot_empirical_mass_for_target_bound,
    minimum_refresh_for_assigned_cells,
    pilot_target_mass_lower_bound,
    regeneration_exposure,
    subset_normalizer_lower_bound,
    target_normalizer_lower_bound,
    terminal_residual_weight,
)
from mo_nco.pareto_sparse_reference import (
    SparseReferenceCover,
    doubling_cover_cardinality_bound,
    greedy_maximal_reference_net,
    sparse_reference_metric_bounds,
    sperner_capacity_lower_bound,
    sperner_many_objective_lower_bound,
)


class RegenerationCertificateTests(unittest.TestCase):
    def test_assignment_pilot_simplex_preflight(self) -> None:
        passing = assignment_pilot_nonemptiness_preflight(
            desired_target_mass_lower_bounds_by_cell=(0.1, 0.2, 0.3),
            assigned_type_by_cell=(0, 0, 1),
            pilot_particles_by_type=(1000, 1000),
            pilot_failure_budgets_by_cell=(0.1, 0.1, 0.1),
            pilot_residual_weights_by_type=(0.0, 0.0),
            mutually_exclusive_cells=True,
        )
        self.assertIsInstance(
            passing,
            AssignmentPilotNonemptinessPreflight,
        )
        self.assertTrue(passing.feasible)
        self.assertEqual(passing.gate, "PASS")
        self.assertTrue(all(passing.cell_feasible_by_cell))
        self.assertTrue(all(passing.simplex_feasible_by_type))
        self.assertLessEqual(
            passing.required_empirical_mass_sum_by_type[0],
            1.0,
        )

        simplex_failure = assignment_pilot_nonemptiness_preflight(
            desired_target_mass_lower_bounds_by_cell=(0.3, 0.3, 0.3),
            assigned_type_by_cell=(0, 0, 0),
            pilot_particles_by_type=(1000,),
            pilot_failure_budgets_by_cell=(0.1, 0.1, 0.1),
            pilot_residual_weights_by_type=(0.0,),
            mutually_exclusive_cells=True,
        )
        self.assertTrue(all(simplex_failure.cell_feasible_by_cell))
        self.assertGreater(
            simplex_failure.required_empirical_mass_sum_by_type[0],
            1.0,
        )
        self.assertEqual(
            simplex_failure.gate,
            "FAIL_TYPE_SIMPLEX",
        )
        self.assertFalse(simplex_failure.feasible)

    def test_assignment_pilot_preflight_inputs_fail_closed(self) -> None:
        base = {
            "desired_target_mass_lower_bounds_by_cell": (0.1,),
            "assigned_type_by_cell": (0,),
            "pilot_particles_by_type": (100,),
            "pilot_failure_budgets_by_cell": (0.1,),
            "pilot_residual_weights_by_type": (0.0,),
            "mutually_exclusive_cells": True,
        }
        bad_overrides = (
            {"desired_target_mass_lower_bounds_by_cell": ()},
            {"assigned_type_by_cell": ()},
            {"assigned_type_by_cell": (True,)},
            {"assigned_type_by_cell": (1,)},
            {"pilot_particles_by_type": ()},
            {"pilot_particles_by_type": (0,)},
            {"pilot_failure_budgets_by_cell": ()},
            {"pilot_residual_weights_by_type": ()},
            {"mutually_exclusive_cells": False},
        )
        for override in bad_overrides:
            with self.subTest(override=override):
                kwargs = dict(base)
                kwargs.update(override)
                with self.assertRaises(ValueError):
                    assignment_pilot_nonemptiness_preflight(**kwargs)

    def test_normalizer_lower_bound_accounts_for_product_rounding(self) -> None:
        beta = 54.64834356087315
        potential_upper_bound = 0.9406498851392012
        lower = target_normalizer_lower_bound(
            beta,
            potential_upper_bound,
        )
        with localcontext() as context:
            context.prec = 120
            exact = (
                -Decimal.from_float(beta)
                * Decimal.from_float(potential_upper_bound)
            ).exp()
        self.assertLessEqual(Decimal.from_float(lower), exact)

    def test_normalizer_and_residual(self) -> None:
        lower = target_normalizer_lower_bound(4.0, 1.03)
        self.assertAlmostEqual(lower, math.exp(-4.12), places=15)
        residual = terminal_residual_weight(
            global_refresh_probability=1.0,
            normalizer_lower_bound=lower,
            mutation_steps=9,
        )
        self.assertAlmostEqual(residual, (1.0 - lower) ** 9, places=15)
        with self.assertRaises(ValueError):
            target_normalizer_lower_bound(745.0, 1.0)

    def test_calibration_subset_bounds(self) -> None:
        lower = subset_normalizer_lower_bound(
            beta=2.0,
            subset_base_mass_lower_bound=0.25,
            potential_upper_bound_on_subset=0.3,
        )
        self.assertAlmostEqual(lower, 0.25 * math.exp(-0.6))
        self.assertAlmostEqual(
            deterministic_target_mass_lower_bound(
                beta=2.0,
                subset_base_mass_lower_bound=0.25,
                potential_upper_bound_on_subset=0.3,
            ),
            lower,
        )

    def test_pilot_nonempty_gate(self) -> None:
        requirement = minimum_pilot_empirical_mass_for_target_bound(
            desired_target_mass_lower_bound=0.05,
            pilot_particles=625,
            pilot_failure_budget=0.05 / 64.0,
            pilot_residual_weight=(1.0 - 0.016) ** 9,
        )
        self.assertTrue(requirement.feasible)
        self.assertGreater(requirement.minimum_empirical_terminal_mass, 0.9)
        impossible = minimum_pilot_empirical_mass_for_target_bound(
            desired_target_mass_lower_bound=0.5,
            pilot_particles=4,
            pilot_failure_budget=0.01,
            pilot_residual_weight=0.95,
        )
        self.assertFalse(impossible.feasible)

    def test_zero_desired_pilot_mass_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            minimum_pilot_empirical_mass_for_target_bound(
                desired_target_mass_lower_bound=0.0,
                pilot_particles=10,
                pilot_failure_budget=0.1,
                pilot_residual_weight=0.2,
            )

    def test_pilot_lower_bound_is_algebraically_valid(self) -> None:
        certificate = pilot_target_mass_lower_bound(
            empirical_terminal_mass=0.9,
            pilot_particles=1000,
            pilot_failure_budget=0.01,
            pilot_residual_weight=0.05,
        )
        q_lower = 0.9 - certificate.pilot_hoeffding_radius
        reconstructed_upper = (
            (1.0 - certificate.pilot_residual_weight)
            * certificate.target_mass_lower_bound
            + certificate.pilot_residual_weight
        )
        self.assertLessEqual(q_lower, reconstructed_upper + 1e-15)
        self.assertTrue(certificate.positive_gate)

    def test_pilot_threshold_and_lower_bound_are_decimal_outward(self) -> None:
        empirical = 0.7514952507967816
        particles = 24_814_668
        delta = 0.003624570020291037
        residual = 0.7466064170099472
        desired = 0.02
        certificate = pilot_target_mass_lower_bound(
            empirical_terminal_mass=empirical,
            pilot_particles=particles,
            pilot_failure_budget=delta,
            pilot_residual_weight=residual,
        )
        requirement = minimum_pilot_empirical_mass_for_target_bound(
            desired_target_mass_lower_bound=desired,
            pilot_particles=particles,
            pilot_failure_budget=delta,
            pilot_residual_weight=residual,
        )
        with localcontext() as context:
            context.prec = 120
            empirical_decimal = Decimal.from_float(empirical)
            residual_decimal = Decimal.from_float(residual)
            radius_decimal = Decimal.from_float(
                certificate.pilot_hoeffding_radius
            )
            exact_lower = (
                empirical_decimal
                - radius_decimal
                - residual_decimal
            ) / (Decimal(1) - residual_decimal)
            exact_threshold = (
                Decimal.from_float(requirement.pilot_hoeffding_radius)
                + residual_decimal
                + (Decimal(1) - residual_decimal)
                * Decimal.from_float(desired)
            )
        self.assertLessEqual(
            Decimal.from_float(certificate.target_mass_lower_bound),
            exact_lower,
        )
        self.assertGreaterEqual(
            Decimal.from_float(
                requirement.minimum_empirical_terminal_mass
            ),
            exact_threshold,
        )

    def test_pilot_residual_one_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            pilot_target_mass_lower_bound(
                empirical_terminal_mass=0.5,
                pilot_particles=10,
                pilot_failure_budget=0.1,
                pilot_residual_weight=1.0,
            )

    def test_confirm_miss_bound(self) -> None:
        certificate = confirm_cell_certificate(
            target_mass_lower_bound=0.2,
            confirm_particles=10,
            confirm_residual_weight=0.1,
        )
        self.assertAlmostEqual(certificate.per_particle_hit_lower_bound, 0.18)
        self.assertAlmostEqual(
            certificate.cell_miss_probability_upper_bound,
            0.82**10,
        )

    def test_closed_form_refresh_requirement(self) -> None:
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=(0.25, 0.2),
            cell_failure_budgets=(0.025, 0.025),
            particles=100,
            terminal_steps=20,
            normalizer_lower_bound=0.5,
        )
        self.assertTrue(requirement.feasible)
        gamma = requirement.minimum_global_refresh_probability
        self.assertIsNotNone(gamma)
        residual = (1.0 - float(gamma) * 0.5) ** 20
        for mass, delta in zip((0.25, 0.2), (0.025, 0.025)):
            self.assertLessEqual((1.0 - (1.0 - residual) * mass) ** 100, delta + 1e-14)

    def test_zero_mass_refresh_requirement_is_infeasible(self) -> None:
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=(0.0,),
            cell_failure_budgets=(0.05,),
            particles=100,
            terminal_steps=10,
            normalizer_lower_bound=0.5,
        )
        self.assertFalse(requirement.feasible)
        self.assertEqual(requirement.gate, "FAIL_PARTICLE_MASS_INSUFFICIENT")

    def test_joint_design(self) -> None:
        design = evaluate_joint_certificate_design(
            target_mass_lower_bound_by_type_cell=((0.20, 0.01), (0.05, 0.25)),
            particles_per_type=(100, 100),
            total_mutation_steps_by_type=(10, 10),
            terminal_regeneration_steps_by_type=(10, 10),
            global_refresh_probability_by_type=(1.0, 1.0),
            normalizer_lower_bound_by_type=(0.2, 0.2),
            requested_confirm_failure_budget=0.05,
            success_igd_bound=0.1,
            failure_igd_cap=2.0,
            success_hv_deficit_bound=0.2,
            failure_hv_deficit_cap=5.0,
            mass_bound_failure_probability=0.01,
        )
        self.assertEqual(design.assigned_type_by_cell, (0, 1))
        self.assertEqual(design.confirm_evaluation_cost, 2200)
        self.assertLessEqual(design.expected_igd_upper_bound, 2.0)
        self.assertLessEqual(design.expected_hv_deficit_upper_bound, 5.0)
        self.assertGreaterEqual(
            design.total_metric_failure_probability_upper_bound,
            design.simultaneous_miss_upper_bound + 0.01,
        )
        self.assertEqual(
            design.refresh_application_scope,
            "all_declared_mutation_steps",
        )

    def test_joint_gate_rejects_decimal_strict_false_pass(self) -> None:
        gamma = 0.5131214865398973
        normalizer = 0.4530327429493099
        mass = 0.6762945372098775
        requested = 1.7011882068490918e-30
        design = evaluate_joint_certificate_design(
            target_mass_lower_bound_by_type_cell=((mass,),),
            particles_per_type=(100,),
            total_mutation_steps_by_type=(5,),
            terminal_regeneration_steps_by_type=(5,),
            global_refresh_probability_by_type=(gamma,),
            normalizer_lower_bound_by_type=(normalizer,),
            requested_confirm_failure_budget=requested,
            success_igd_bound=0.0,
            failure_igd_cap=1.0,
            success_hv_deficit_bound=0.0,
            failure_hv_deficit_cap=1.0,
            mass_bound_failure_probability=0.0,
        )
        with localcontext() as context:
            context.prec = 120
            gamma_decimal = Decimal.from_float(gamma)
            normalizer_decimal = Decimal.from_float(normalizer)
            mass_decimal = Decimal.from_float(mass)
            exact = (
                Decimal(1)
                - (
                    Decimal(1)
                    - (
                        Decimal(1)
                        - gamma_decimal * normalizer_decimal
                    )
                    ** 5
                )
                * mass_decimal
            ) ** 100
        self.assertGreater(exact, Decimal.from_float(requested))
        self.assertEqual(design.confirm_failure_gate, "FAIL")
        self.assertGreater(
            design.simultaneous_miss_upper_bound,
            requested,
        )

    def test_positive_underflow_misses_are_rounded_up(self) -> None:
        requested = float.fromhex("0x0.0000000000001p-1022")
        design = evaluate_joint_certificate_design(
            target_mass_lower_bound_by_type_cell=((1.0 / 3.0,) * 3,),
            particles_per_type=(1838,),
            total_mutation_steps_by_type=(1,),
            terminal_regeneration_steps_by_type=(1,),
            global_refresh_probability_by_type=(1.0,),
            normalizer_lower_bound_by_type=(1.0,),
            requested_confirm_failure_budget=requested,
            success_igd_bound=0.0,
            failure_igd_cap=1.0,
            success_hv_deficit_bound=0.0,
            failure_hv_deficit_cap=1.0,
            mass_bound_failure_probability=0.0,
        )
        self.assertTrue(
            all(value > 0.0 for value in design.per_cell_miss_upper_bound)
        )
        self.assertGreater(
            design.simultaneous_miss_upper_bound,
            requested,
        )
        self.assertEqual(design.confirm_failure_gate, "FAIL")

    def test_exact_100k_schedules(self) -> None:
        schedules = enumerate_equal_dual_stream_schedules(
            total_evaluations=100_000,
            type_count=8,
            max_particles_per_stream=10_000,
            checkpoint_period=10_000,
        )
        pairs = {
            (row.particles_per_type, row.total_mutations_per_particle)
            for row in schedules
            if row.particles_per_stream_within_cap and row.checkpoint_aligned
        }
        self.assertEqual(
            pairs,
            {
                (1, 6249), (2, 3124), (5, 1249), (10, 624),
                (25, 249), (50, 124), (125, 49), (250, 24),
                (625, 9), (1250, 4),
            },
        )
        self.assertTrue(all(row.exact_budget_identity for row in schedules))
        self.assertTrue(
            all(
                row.checkpoint_alignment_scope
                == "evaluation_budget_grid_only"
                and not row.checkpoint_full_type_sweep_boundary_verified
                for row in schedules
            )
        )
        self.assertEqual(
            sum(
                row.checkpoint_aligned and row.particles_per_stream_within_cap
                for row in schedules
            ),
            10,
        )
        published_regular_schedules = [
            row
            for row in schedules
            if row.particles_per_stream_within_cap
            and row.checkpoint_aligned
            and row.total_mutations_per_particle >= 54
        ]
        self.assertEqual(
            max(row.particles_per_type for row in published_regular_schedules),
            50,
        )


    def test_stable_refresh_boundary(self) -> None:
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=(0.9,),
            cell_failure_budgets=(0.999999,),
            particles=100_000,
            terminal_steps=10_000,
            normalizer_lower_bound=0.2,
        )
        self.assertTrue(requirement.feasible)
        gamma = float(requirement.minimum_global_refresh_probability)
        self.assertGreater(gamma, 0.0)
        residual = terminal_residual_weight(
            global_refresh_probability=gamma,
            normalizer_lower_bound=0.2,
            mutation_steps=10_000,
        )
        hit = (1.0 - residual) * 0.9
        risk = math.exp(100_000 * math.log1p(-hit))
        self.assertLessEqual(risk, 0.999999 * (1.0 + 1e-10))

    def test_minimum_refresh_repairs_tight_binary64_boundary(self) -> None:
        mass = 0.6517872434456071
        delta = 0.943308174321089
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=(mass,),
            cell_failure_budgets=(delta,),
            particles=827,
            terminal_steps=167,
            normalizer_lower_bound=0.9747979908636953,
        )
        self.assertTrue(requirement.feasible)
        gamma = float(requirement.minimum_global_refresh_probability)
        residual = terminal_residual_weight(
            global_refresh_probability=gamma,
            normalizer_lower_bound=0.9747979908636953,
            mutation_steps=167,
        )
        certified = confirm_cell_certificate(
            target_mass_lower_bound=mass,
            confirm_particles=827,
            confirm_residual_weight=residual,
        )
        self.assertLessEqual(
            certified.cell_miss_probability_upper_bound,
            delta,
        )

    def test_minimum_refresh_uses_public_certificate_predicate(self) -> None:
        masses = (
            0.9640887888122371,
            0.1265967945367811,
            0.04017808211034111,
            0.7144871063317442,
        )
        budgets = (
            0.3087826357775085,
            0.00039880213272078984,
            0.07346423003768494,
            0.09063477463257424,
        )
        particles = 105
        steps = 60
        normalizer = 0.6321236442731819
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=masses,
            cell_failure_budgets=budgets,
            particles=particles,
            terminal_steps=steps,
            normalizer_lower_bound=normalizer,
        )
        self.assertTrue(requirement.feasible)
        gamma = float(requirement.minimum_global_refresh_probability)
        residual = terminal_residual_weight(
            global_refresh_probability=gamma,
            normalizer_lower_bound=normalizer,
            mutation_steps=steps,
        )
        for mass, budget in zip(masses, budgets):
            certified = confirm_cell_certificate(
                target_mass_lower_bound=mass,
                confirm_particles=particles,
                confirm_residual_weight=residual,
            )
            self.assertLessEqual(
                certified.cell_miss_probability_upper_bound,
                budget,
            )

    def test_minimum_refresh_is_smallest_satisfying_binary64(self) -> None:
        mass = 0.12679315091257257
        delta = 1.0325292975878435e-05
        particles = 7646
        steps = 761
        normalizer = 0.00034493212431207696
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=(mass,),
            cell_failure_budgets=(delta,),
            particles=particles,
            terminal_steps=steps,
            normalizer_lower_bound=normalizer,
        )
        self.assertTrue(requirement.feasible)
        gamma = float(requirement.minimum_global_refresh_probability)
        for candidate, should_pass in (
            (gamma, True),
            (math.nextafter(gamma, -math.inf), False),
        ):
            residual = terminal_residual_weight(
                global_refresh_probability=candidate,
                normalizer_lower_bound=normalizer,
                mutation_steps=steps,
            )
            certificate = confirm_cell_certificate(
                target_mass_lower_bound=mass,
                confirm_particles=particles,
                confirm_residual_weight=residual,
            )
            self.assertEqual(
                certificate.cell_miss_probability_upper_bound <= delta,
                should_pass,
            )

    def test_minimum_refresh_does_not_collapse_to_signed_zero(self) -> None:
        requirement = minimum_refresh_for_assigned_cells(
            target_mass_lower_bounds=(1.0,),
            cell_failure_budgets=(0.5,),
            particles=10**18,
            terminal_steps=1,
            normalizer_lower_bound=1.0,
        )
        self.assertTrue(requirement.feasible)
        gamma = float(requirement.minimum_global_refresh_probability)
        self.assertGreater(gamma, 0.0)
        residual = terminal_residual_weight(
            global_refresh_probability=gamma,
            normalizer_lower_bound=1.0,
            mutation_steps=1,
        )
        certificate = confirm_cell_certificate(
            target_mass_lower_bound=1.0,
            confirm_particles=10**18,
            confirm_residual_weight=residual,
        )
        self.assertLessEqual(
            certificate.cell_miss_probability_upper_bound,
            0.5,
        )

    def test_heterogeneous_budget_ledger(self) -> None:
        ledger = heterogeneous_pilot_confirm_budget(
            pilot_particles_by_type=(100,) * 8,
            pilot_mutations_per_particle_by_type=(9,) * 8,
            confirm_particles_by_type=(525,) * 8,
            confirm_mutations_per_particle_by_type=(19,) * 8,
            requested_total_evaluations=92_000,
        )
        self.assertEqual(ledger.pilot_evaluation_cost, 8_000)
        self.assertEqual(ledger.confirm_evaluation_cost, 84_000)
        self.assertTrue(ledger.exact_budget_identity)

    def test_finite_suite_hoeffding_planner(self) -> None:
        half_width_30 = finite_suite_hoeffding_half_width(
            independent_units=30,
            simultaneous_claims=12,
            familywise_alpha=0.05,
        )
        self.assertAlmostEqual(half_width_30, 0.6415494838748833)
        requirement = minimum_independent_units_for_hoeffding_half_width(
            requested_half_width=0.2,
            simultaneous_claims=12,
            familywise_alpha=0.05,
        )
        self.assertEqual(requirement.minimum_independent_units, 309)
        self.assertLessEqual(
            finite_suite_hoeffding_half_width(
                independent_units=309,
                simultaneous_claims=12,
                familywise_alpha=0.05,
            ),
            0.2,
        )
        self.assertGreater(
            finite_suite_hoeffding_half_width(
                independent_units=308,
                simultaneous_claims=12,
                familywise_alpha=0.05,
            ),
            0.2,
        )

    def test_hoeffding_widths_are_decimal_outward_upper_bounds(self) -> None:
        units = 9_004_013
        claims = 9_603
        alpha = 1.958379556346719e-07
        width = 0.8178160991179896
        suite_bound = finite_suite_hoeffding_half_width(
            independent_units=units,
            simultaneous_claims=claims,
            familywise_alpha=alpha,
            range_width=width,
        )
        pilot_units = 849_079_553
        pilot_delta = 0.7397391295281518
        pilot = pilot_target_mass_lower_bound(
            empirical_terminal_mass=0.9,
            pilot_particles=pilot_units,
            pilot_failure_budget=pilot_delta,
            pilot_residual_weight=0.0,
        )
        with localcontext() as context:
            context.prec = 120
            exact_suite = Decimal.from_float(width) * (
                (
                    Decimal(2)
                    * Decimal(claims)
                    / Decimal.from_float(alpha)
                ).ln()
                / (Decimal(2) * Decimal(units))
            ).sqrt()
            exact_pilot = (
                (Decimal(1) / Decimal.from_float(pilot_delta)).ln()
                / (Decimal(2) * Decimal(pilot_units))
            ).sqrt()
        self.assertGreaterEqual(
            Decimal.from_float(suite_bound),
            exact_suite,
        )
        self.assertGreaterEqual(
            Decimal.from_float(pilot.pilot_hoeffding_radius),
            exact_pilot,
        )

    def test_exposure(self) -> None:
        self.assertGreater(
            regeneration_exposure(625, 9, 0.016),
            regeneration_exposure(250, 24, 0.016),
        )


class SparseReferenceTests(unittest.TestCase):
    def test_greedy_cover(self) -> None:
        points = ((0.0, 1.0), (0.05, 0.95), (0.5, 0.5), (1.0, 0.0))
        cover = greedy_maximal_reference_net(points, cover_radius=0.1, p_norm=math.inf)
        self.assertLessEqual(cover.realized_cover_radius, 0.1 + 1e-15)
        self.assertEqual(len(cover.anchor_indices), 3)
        bounds = sparse_reference_metric_bounds(
            points,
            cover,
            cell_width_vector=(0.02, 0.02),
            p_norm=2.0,
        )
        self.assertEqual(bounds.archive_cardinality_bound, 3)
        self.assertGreaterEqual(bounds.ordinary_igd_bound, 0.0)

    def test_sparse_cover_integrity_fails_closed(self) -> None:
        points = ((0.0, 1.0), (1.0, 0.0))
        forged = SparseReferenceCover(
            objective_dimension=2,
            p_norm=math.inf,
            requested_cover_radius=1.0,
            anchor_indices=(),
            assignment_by_reference=(0, 0),
            anchor_objectives=((0.0, 1.0),),
            realized_cover_radius=1.0,
            pairwise_anchor_separation_lower_bound=None,
            cluster_sizes=(),
        )
        with self.assertRaises(ValueError):
            sparse_reference_metric_bounds(
                points,
                forged,
                cell_width_vector=(0.1, 0.1),
            )
        with self.assertRaises(ValueError):
            greedy_maximal_reference_net(
                points,
                cover_radius=0.5,
                p_norm=-math.inf,
            )
        singleton_with_boolean_size = SparseReferenceCover(
            objective_dimension=2,
            p_norm=math.inf,
            requested_cover_radius=0.1,
            anchor_indices=(0,),
            assignment_by_reference=(0,),
            anchor_objectives=((0.0, 1.0),),
            realized_cover_radius=0.0,
            pairwise_anchor_separation_lower_bound=None,
            cluster_sizes=(True,),
        )
        with self.assertRaises(ValueError):
            sparse_reference_metric_bounds(
                ((0.0, 1.0),),
                singleton_with_boolean_size,
                cell_width_vector=(0.1, 0.1),
            )

    def test_sparse_cover_requires_canonical_greedy_assignment(self) -> None:
        points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        noncanonical_tie_assignment = SparseReferenceCover(
            objective_dimension=2,
            p_norm=math.inf,
            requested_cover_radius=1.0,
            anchor_indices=(0, 2),
            assignment_by_reference=(0, 1, 1),
            anchor_objectives=((0.0, 0.0), (2.0, 0.0)),
            realized_cover_radius=1.0,
            pairwise_anchor_separation_lower_bound=2.0,
            cluster_sizes=(1, 2),
        )
        with self.assertRaisesRegex(ValueError, "canonical greedy"):
            sparse_reference_metric_bounds(
                points,
                noncanonical_tie_assignment,
                cell_width_vector=(0.1, 0.1),
            )

    def test_singleton_cover_is_strict_json_serializable(self) -> None:
        cover = greedy_maximal_reference_net(
            ((0.0, 0.0),),
            cover_radius=0.1,
        )
        self.assertIsNone(
            cover.pairwise_anchor_separation_lower_bound
        )
        json.dumps(cover.to_jsonable(), allow_nan=False)

    def test_sperner_exponential_lower_bound(self) -> None:
        lower = sperner_many_objective_lower_bound(
            objective_dimension=20,
            additive_linf_tolerance=0.1,
        )
        self.assertEqual(lower.pareto_point_count, math.comb(20, 10))
        self.assertEqual(
            lower.minimum_required_feasible_representatives,
            math.comb(20, 10),
        )


    def test_sperner_type_and_cell_capacity(self) -> None:
        lower = sperner_capacity_lower_bound(
            objective_dimension=8,
            additive_linf_tolerance=0.5,
            maximum_certified_representatives_per_type=5,
        )
        self.assertEqual(lower.pareto_point_count, 70)
        self.assertEqual(lower.minimum_required_subunit_linf_cells, 70)
        self.assertEqual(lower.minimum_required_types, 14)

    def test_doubling_bound(self) -> None:
        self.assertEqual(
            doubling_cover_cardinality_bound(
                doubling_constant=2,
                diameter=1.0,
                separation_radius=0.25,
            ),
            8,
        )
        self.assertEqual(
            doubling_cover_cardinality_bound(
                doubling_constant=4,
                diameter=0.1,
                separation_radius=1.0,
            ),
            1,
        )

    def test_design_cli_rejects_mass_outside_probability_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mo_nco.run_pareto_smc_v12_design",
                    "--total-evaluations",
                    "100000",
                    "--types",
                    "8",
                    "--max-particles-per-stream",
                    "10000",
                    "--checkpoint-period",
                    "10000",
                    "--final-stage-mutations",
                    "9",
                    "--one-step-minorization",
                    "0.016",
                    "--target-cell-mass-lower-bound",
                    "2.0",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must be finite and lie in (0,1]", completed.stderr)

    def test_design_cli_rejects_vacuous_regeneration_inputs(self) -> None:
        invalid_overrides = (
            ("--one-step-minorization", "0"),
            ("--target-cell-mass-lower-bound", "0"),
            ("--final-stage-mutations", "0"),
        )
        for flag, invalid_value in invalid_overrides:
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "invalid.json"
                arguments = [
                    sys.executable,
                    "-m",
                    "mo_nco.run_pareto_smc_v12_design",
                    "--total-evaluations",
                    "100000",
                    "--types",
                    "8",
                    "--max-particles-per-stream",
                    "10000",
                    "--checkpoint-period",
                    "10000",
                    "--final-stage-mutations",
                    "9",
                    "--one-step-minorization",
                    "0.016",
                    "--target-cell-mass-lower-bound",
                    "0.05",
                    "--output",
                    str(output),
                ]
                arguments[arguments.index(flag) + 1] = invalid_value
                completed = subprocess.run(
                    arguments,
                    capture_output=True,
                    text=True,
                )
                output_created = output.exists()
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output_created)

    def test_design_cli_returns_nonzero_when_no_schedule_is_admissible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "no_schedule.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mo_nco.run_pareto_smc_v12_design",
                    "--total-evaluations",
                    "100000",
                    "--types",
                    "8",
                    "--max-particles-per-stream",
                    "10000",
                    "--checkpoint-period",
                    "10000",
                    "--final-stage-mutations",
                    "10000",
                    "--one-step-minorization",
                    "0.016",
                    "--target-cell-mass-lower-bound",
                    "0.05",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(payload["design_gate"], "FAIL")
        self.assertEqual(payload["admissible_schedules"], [])
        self.assertIsNone(
            payload[
                "best_admissible_for_declared_single_cell_miss_bound"
            ]
        )


if __name__ == "__main__":
    unittest.main()

