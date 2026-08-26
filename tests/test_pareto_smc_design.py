from __future__ import annotations

import math
import unittest
from decimal import Decimal, localcontext

from mo_nco.pareto_smc_design import (
    ParetoSMCDesignError,
    build_frozen_complexity_ledger,
    design_stagewise_global_refresh,
    plan_fixed_schedule_budget,
    plan_single_and_dual_stream_budgets,
)


class StagewiseRefreshDesignTests(unittest.TestCase):
    def test_analytic_minimum_meets_every_stage_cap(self) -> None:
        beta = (0.5, 1.0)
        steps = (4, 8)
        potential_upper = 1.0
        cap = 0.5
        design = design_stagewise_global_refresh(
            beta,
            potential_upper_bound=potential_upper,
            mutation_steps_by_stage=steps,
            target_stage_contraction_cap=cap,
        )

        self.assertEqual(design.feasibility_gate, "PASS")
        self.assertEqual(
            design.numerical_rounding_contract,
            "decimal_outward_scale_and_contraction_then_binary64_"
            "bisection_v2",
        )
        self.assertFalse(design.terminal_quality_optimality_claimed)
        expected_sum = 0.0
        for beta_stage, stage_steps, stage in zip(
            beta,
            steps,
            design.stages,
        ):
            expected_gamma = (
                1.0 - cap ** (1.0 / stage_steps)
            ) * math.exp(beta_stage * potential_upper)
            self.assertTrue(stage.feasible)
            self.assertAlmostEqual(
                stage.minimum_global_refresh_probability,
                expected_gamma,
                places=14,
            )
            self.assertAlmostEqual(
                stage.contraction_at_minimum,
                cap,
                places=14,
            )
            self.assertLessEqual(stage.contraction_at_minimum, cap)
            expected_sum += stage_steps * expected_gamma

            smaller_gamma = expected_gamma - 1e-8
            smaller_contraction = (
                1.0
                - smaller_gamma
                * math.exp(-beta_stage * potential_upper)
            ) ** stage_steps
            self.assertGreater(smaller_contraction, cap)

        self.assertAlmostEqual(
            design.minimum_expected_global_refresh_proposals_per_particle,
            expected_sum,
            places=14,
        )

    def test_infeasible_probability_and_zero_step_are_reported(self) -> None:
        probability_infeasible = design_stagewise_global_refresh(
            (10.0,),
            potential_upper_bound=1.0,
            mutation_steps_by_stage=(1,),
            target_stage_contraction_cap=0.9,
        )
        self.assertEqual(
            probability_infeasible.feasibility_gate,
            "FAIL",
        )
        self.assertIsNone(
            probability_infeasible.stages[
                0
            ].minimum_global_refresh_probability
        )
        self.assertIn(
            "exceeds one",
            probability_infeasible.stages[0].infeasibility_reason,
        )
        self.assertIsNone(
            probability_infeasible.minimum_expected_global_refresh_proposals_per_particle
        )

        zero_step_infeasible = design_stagewise_global_refresh(
            (0.5,),
            potential_upper_bound=1.0,
            mutation_steps_by_stage=(0,),
            target_stage_contraction_cap=0.99,
        )
        self.assertFalse(zero_step_infeasible.stages[0].feasible)
        self.assertIn(
            "zero-step",
            zero_step_infeasible.stages[0].infeasibility_reason,
        )

    def test_contraction_cap_boundaries_are_explicit(self) -> None:
        vacuous = design_stagewise_global_refresh(
            (0.5,),
            potential_upper_bound=2.0,
            mutation_steps_by_stage=(0,),
            target_stage_contraction_cap=1.0,
        )
        self.assertEqual(vacuous.feasibility_gate, "PASS")
        self.assertEqual(
            vacuous.stages[0].minimum_global_refresh_probability,
            0.0,
        )
        self.assertEqual(vacuous.stages[0].contraction_at_minimum, 1.0)

        exact_zero = design_stagewise_global_refresh(
            (0.0,),
            potential_upper_bound=2.0,
            mutation_steps_by_stage=(2,),
            target_stage_contraction_cap=0.0,
        )
        self.assertEqual(exact_zero.feasibility_gate, "PASS")
        self.assertEqual(
            exact_zero.stages[0].minimum_global_refresh_probability,
            1.0,
        )
        self.assertEqual(
            exact_zero.stages[0].contraction_at_minimum,
            0.0,
        )

        impossible_zero = design_stagewise_global_refresh(
            (0.1,),
            potential_upper_bound=1.0,
            mutation_steps_by_stage=(2,),
            target_stage_contraction_cap=0.0,
        )
        self.assertEqual(impossible_zero.feasibility_gate, "FAIL")

    def test_binary64_rounding_never_marks_an_exceeded_cap_pass(self) -> None:
        cap = 0.010596042080244894
        design = design_stagewise_global_refresh(
            (6.830466431895645e-11,),
            potential_upper_bound=0.00540404821047374,
            mutation_steps_by_stage=(1_000_000,),
            target_stage_contraction_cap=cap,
        )
        stage = design.stages[0]
        self.assertEqual(design.feasibility_gate, "PASS")
        self.assertIsNotNone(stage.contraction_at_minimum)
        assert stage.contraction_at_minimum is not None
        self.assertLessEqual(stage.contraction_at_minimum, cap)

    def test_decimal_contraction_boundary_is_fail_closed(self) -> None:
        beta = 8.625505131190627
        cap = 0.99990659345242
        design = design_stagewise_global_refresh(
            (beta,),
            potential_upper_bound=1.0,
            mutation_steps_by_stage=(2,),
            target_stage_contraction_cap=cap,
        )
        stage = design.stages[0]
        self.assertTrue(stage.feasible)
        gamma = stage.minimum_global_refresh_probability
        self.assertIsNotNone(gamma)
        with localcontext() as context:
            context.prec = 120
            exact_scale = (-Decimal.from_float(beta)).exp()
            exact_contraction = (
                Decimal(1)
                - Decimal.from_float(float(gamma)) * exact_scale
            ) ** 2
        self.assertLessEqual(
            exact_contraction,
            Decimal.from_float(cap),
        )
        self.assertIsNotNone(stage.contraction_at_minimum)
        self.assertLessEqual(stage.contraction_at_minimum, cap)

    def test_refresh_inputs_fail_closed(self) -> None:
        bad_calls = (
            lambda: design_stagewise_global_refresh(
                (),
                potential_upper_bound=1.0,
                mutation_steps_by_stage=(),
                target_stage_contraction_cap=0.5,
            ),
            lambda: design_stagewise_global_refresh(
                (1.0, 0.5),
                potential_upper_bound=1.0,
                mutation_steps_by_stage=(1, 1),
                target_stage_contraction_cap=0.5,
            ),
            lambda: design_stagewise_global_refresh(
                (0.5,),
                potential_upper_bound=float("nan"),
                mutation_steps_by_stage=(1,),
                target_stage_contraction_cap=0.5,
            ),
            lambda: design_stagewise_global_refresh(
                (0.5,),
                potential_upper_bound=1.0,
                mutation_steps_by_stage=(True,),
                target_stage_contraction_cap=0.5,
            ),
            lambda: design_stagewise_global_refresh(
                (0.5,),
                potential_upper_bound=1.0,
                mutation_steps_by_stage=(1,),
                target_stage_contraction_cap=1.01,
            ),
        )
        for call in bad_calls:
            with self.subTest(call=call):
                with self.assertRaises(ParetoSMCDesignError):
                    call()


class FixedScheduleBudgetTests(unittest.TestCase):
    def test_single_and_dual_stream_exact_budgets(self) -> None:
        design = plan_single_and_dual_stream_budgets(
            360,
            type_count=3,
            mutation_steps_by_stage=(2, 3),
        )
        single = design.single_stream
        dual = design.two_stream

        self.assertEqual(single.evaluations_per_particle_per_stream, 6)
        self.assertEqual(single.exact_budget_quantum, 18)
        self.assertEqual(single.maximum_particles_per_type, 20)
        self.assertEqual(single.exact_evaluations_per_stream, 360)
        self.assertEqual(single.exact_total_evaluations, 360)
        self.assertEqual(single.leftover_evaluations, 0)
        self.assertEqual(single.budget_gate, "PASS")

        self.assertEqual(dual.exact_budget_quantum, 36)
        self.assertEqual(dual.maximum_particles_per_type, 10)
        self.assertEqual(dual.particles_per_stream, 30)
        self.assertEqual(dual.exact_evaluations_per_stream, 180)
        self.assertEqual(dual.exact_total_evaluations, 360)
        self.assertEqual(dual.leftover_evaluations, 0)
        self.assertEqual(dual.budget_gate, "PASS")

    def test_nondivisible_and_insufficient_budgets_fail_gates(self) -> None:
        design = plan_single_and_dual_stream_budgets(
            378,
            type_count=3,
            mutation_steps_by_stage=(2, 3),
        )
        self.assertEqual(design.single_stream.budget_gate, "PASS")
        self.assertEqual(
            design.two_stream.budget_gate,
            "FAIL_NON_DIVISIBLE",
        )
        self.assertEqual(design.two_stream.leftover_evaluations, 18)
        self.assertFalse(design.two_stream.exact_budget_feasible)

        insufficient = plan_fixed_schedule_budget(
            17,
            type_count=3,
            mutation_steps_by_stage=(2, 3),
            stream_count=1,
        )
        self.assertEqual(insufficient.maximum_particles_per_type, 0)
        self.assertEqual(insufficient.exact_total_evaluations, 0)
        self.assertEqual(insufficient.leftover_evaluations, 17)
        self.assertFalse(insufficient.particle_feasible)
        self.assertEqual(
            insufficient.budget_gate,
            "FAIL_INSUFFICIENT_FOR_ONE_PARTICLE_PER_TYPE",
        )

    def test_budget_inputs_fail_closed(self) -> None:
        bad_calls = (
            lambda: plan_fixed_schedule_budget(
                -1,
                type_count=2,
                mutation_steps_by_stage=(1,),
                stream_count=1,
            ),
            lambda: plan_fixed_schedule_budget(
                10,
                type_count=True,
                mutation_steps_by_stage=(1,),
                stream_count=1,
            ),
            lambda: plan_fixed_schedule_budget(
                10,
                type_count=2,
                mutation_steps_by_stage=(),
                stream_count=1,
            ),
            lambda: plan_fixed_schedule_budget(
                10,
                type_count=2,
                mutation_steps_by_stage=(1,),
                stream_count=0,
            ),
        )
        for call in bad_calls:
            with self.subTest(call=call):
                with self.assertRaises(ParetoSMCDesignError):
                    call()


class FrozenComplexityLedgerTests(unittest.TestCase):
    directions = (
        (0.5, 0.25, 0.25),
        (0.2, 0.3, 0.5),
    )

    def test_ledger_counts_unique_cells_observables_and_archive_floor(
        self,
    ) -> None:
        ledger = build_frozen_complexity_ledger(
            reference_directions=self.directions,
            grid_cell_counts=(4, 5, 6),
            reference_cells=((0, 0, 0), (3, 4, 5), (0, 0, 0)),
            max_type_count=2,
            max_grid_cell_count=120,
            archive_cap=2,
        )

        self.assertEqual(ledger.objective_dimension, 3)
        self.assertEqual(ledger.type_count, 2)
        self.assertEqual(ledger.grid_cell_capacity, 120)
        self.assertEqual(ledger.supplied_reference_cell_count, 3)
        self.assertEqual(ledger.unique_reference_cell_count, 2)
        self.assertEqual(ledger.duplicate_reference_cell_count, 1)
        self.assertEqual(ledger.pilot_type_cell_observable_count, 4)
        self.assertEqual(ledger.confirm_cell_observable_count, 2)
        self.assertEqual(
            ledger.minimum_archive_cap_for_one_per_reference_cell,
            2,
        )
        self.assertEqual(ledger.archive_cap_cardinality_gate, "PASS")
        self.assertEqual(ledger.max_type_max_cell_gate, "PASS")
        self.assertEqual(len(ledger.frozen_structure_sha256), 64)
        self.assertFalse(ledger.unknown_pareto_front_coverage_claimed)
        self.assertFalse(
            ledger.fixed_size_archive_metric_preservation_claimed
        )
        self.assertIn(
            "not sufficient",
            ledger.archive_cap_floor_scope,
        )

    def test_maximum_size_and_archive_gates_fail_closed(self) -> None:
        ledger = build_frozen_complexity_ledger(
            reference_directions=self.directions,
            grid_cell_counts=(4, 5, 6),
            reference_cells=((0, 0, 0), (3, 4, 5)),
            max_type_count=1,
            max_grid_cell_count=100,
            archive_cap=1,
        )
        self.assertEqual(ledger.max_type_gate, "FAIL")
        self.assertEqual(ledger.max_cell_gate, "FAIL")
        self.assertEqual(ledger.max_type_max_cell_gate, "FAIL")
        self.assertEqual(ledger.archive_cap_cardinality_gate, "FAIL")

        exploding_grid = build_frozen_complexity_ledger(
            reference_directions=((0.25, 0.25, 0.25, 0.25),),
            grid_cell_counts=(1000, 1000, 1000, 1000),
            reference_cells=((0, 0, 0, 0),),
            max_type_count=1,
            max_grid_cell_count=10**9,
        )
        self.assertEqual(exploding_grid.grid_cell_capacity, 10**12)
        self.assertEqual(exploding_grid.max_cell_gate, "FAIL")
        self.assertEqual(
            exploding_grid.archive_cap_cardinality_gate,
            "NOT_EVALUATED",
        )

    def test_frozen_hash_uses_canonical_unique_reference_cells(self) -> None:
        first = build_frozen_complexity_ledger(
            reference_directions=self.directions,
            grid_cell_counts=(4, 5, 6),
            reference_cells=((0, 0, 0), (3, 4, 5), (0, 0, 0)),
            max_type_count=2,
            max_grid_cell_count=120,
        )
        second = build_frozen_complexity_ledger(
            reference_directions=self.directions,
            grid_cell_counts=(4, 5, 6),
            reference_cells=((3, 4, 5), (0, 0, 0)),
            max_type_count=2,
            max_grid_cell_count=120,
        )
        self.assertEqual(
            first.frozen_structure_sha256,
            second.frozen_structure_sha256,
        )

    def test_complexity_inputs_fail_closed(self) -> None:
        bad_calls = (
            lambda: build_frozen_complexity_ledger(
                reference_directions=self.directions,
                grid_cell_counts=(4, 5, 6),
                reference_cells=((4, 0, 0),),
                max_type_count=2,
                max_grid_cell_count=120,
            ),
            lambda: build_frozen_complexity_ledger(
                reference_directions=(
                    self.directions[0],
                    self.directions[0],
                ),
                grid_cell_counts=(4, 5, 6),
                reference_cells=((0, 0, 0),),
                max_type_count=2,
                max_grid_cell_count=120,
            ),
            lambda: build_frozen_complexity_ledger(
                reference_directions=((0.5, 0.5, 0.5),),
                grid_cell_counts=(4, 5, 6),
                reference_cells=((0, 0, 0),),
                max_type_count=2,
                max_grid_cell_count=120,
            ),
            lambda: build_frozen_complexity_ledger(
                reference_directions=self.directions,
                grid_cell_counts=(4, 5, 6),
                reference_cells=((0, 0),),
                max_type_count=2,
                max_grid_cell_count=120,
            ),
            lambda: build_frozen_complexity_ledger(
                reference_directions=self.directions,
                grid_cell_counts=(4, 5, 6),
                reference_cells=((0, 0, 0),),
                max_type_count=2,
                max_grid_cell_count=120,
                archive_cap=-1,
            ),
        )
        for call in bad_calls:
            with self.subTest(call=call):
                with self.assertRaises(ParetoSMCDesignError):
                    call()


if __name__ == "__main__":
    unittest.main()

