from __future__ import annotations

import unittest

from mo_nco.evaluation import CountingTSPInstance, evaluation_count
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.ips_certified import CertifiedSingleSiteIPSOptimizer


class CertifiedSingleSiteIPSTests(unittest.TestCase):
    def test_certified_kernel_recomputes_objectives_from_the_tour_state(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=69)

        class NoIncrementalObjectiveWrapper:
            """Expose the source state function but forbid cached delta updates."""

            num_cities = base.num_cities
            num_objectives = base.num_objectives
            objective_scale_estimates = base.objective_scale_estimates
            distance_matrices = base.distance_matrices

            @staticmethod
            def evaluate(tour):  # type: ignore[no-untyped-def]
                return base.evaluate(tour)

            @staticmethod
            def evaluate_two_opt(*_args):  # type: ignore[no-untyped-def]
                raise AssertionError("certified MH must not evolve a cached floating objective state")

        result = CertifiedSingleSiteIPSOptimizer(
            NoIncrementalObjectiveWrapper(),  # type: ignore[arg-type]
            num_particles=4,
            evaluations=16,
            seed=70,
            temperature=0.08,
            lazy_probability=0.25,
        ).run()
        self.assertEqual(result.metadata["objective_evaluation_contract"], "full_tour_state_function")
        for tour, objective in zip(result.particles, result.objectives):
            self.assertEqual(objective, base.evaluate(tour))

    def test_strict_kernel_uses_one_site_positive_temperature_and_full_budget(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(10, seed=71)
        counted = CountingTSPInstance(base, max_evaluations=36)
        optimizer = CertifiedSingleSiteIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=6,
            evaluations=36,
            seed=72,
            temperature=0.08,
            lazy_probability=0.25,
            log_period=6,
        )
        context_hash = optimizer.context_hash
        result = optimizer.run()
        self.assertEqual(evaluation_count(counted), 36)
        self.assertEqual(result.metadata["algorithm_contract"], "theory_certified_single_site_v4")
        self.assertEqual(result.metadata["claim_level"], "certified_mh")
        self.assertEqual(result.metadata["context_hash"], context_hash)
        self.assertEqual(result.metadata["context_refresh_count"], 0)
        self.assertTrue(result.metadata["single_coordinate_transition"])
        self.assertTrue(result.metadata["proposal_symmetric"])
        self.assertTrue(result.metadata["positive_temperature"])
        self.assertTrue(result.metadata["explicit_laziness"])
        self.assertGreater(result.metadata["lazy_self_loops"], 0)
        self.assertEqual(result.metadata["transition_attempts"], 30)
        self.assertEqual(
            result.metadata["proposal_evaluations"] + result.metadata["identity_evaluations"],
            30,
        )
        self.assertFalse(result.metadata["archive_feedback"])
        self.assertFalse(result.metadata["neural_enabled"])
        self.assertLessEqual(float(result.metadata["db_max_abs_log_residual"]), 1e-12)
        self.assertEqual(result.metadata["transition_evaluations"], 30)
        self.assertTrue(all(diag.temperature == 0.08 for diag in result.diagnostics))
        for tour in result.particles:
            base.validate_tour(tour)

    def test_strict_kernel_rejects_zero_temperature(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=73)
        with self.assertRaises(ValueError):
            CertifiedSingleSiteIPSOptimizer(instance, num_particles=4, evaluations=12, temperature=0.0)

    def test_strict_kernel_budget_is_run_local_when_counter_is_precharged(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=74)
        counted = CountingTSPInstance(base, max_evaluations=40, evaluations=5)
        result = CertifiedSingleSiteIPSOptimizer(
            counted,  # type: ignore[arg-type]
            num_particles=4,
            evaluations=20,
            seed=75,
            temperature=0.05,
        ).run()
        self.assertEqual(evaluation_count(counted), 25)
        self.assertEqual(result.metadata["evaluation_counter_start"], 5)
        self.assertEqual(result.metadata["evaluation_budget"], 20)
        self.assertEqual(result.metadata["evaluations_used"], 20)

    def test_strict_kernel_rejects_a_smaller_remaining_counting_budget(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(8, seed=76)
        counted = CountingTSPInstance(base, max_evaluations=12)
        with self.assertRaisesRegex(ValueError, "smaller than the requested certified run"):
            CertifiedSingleSiteIPSOptimizer(
                counted,  # type: ignore[arg-type]
                num_particles=4,
                evaluations=20,
                seed=77,
                temperature=0.05,
            )


if __name__ == "__main__":
    unittest.main()

