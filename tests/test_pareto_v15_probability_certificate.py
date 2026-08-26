from __future__ import annotations

from fractions import Fraction
import json
import unittest

from mo_nco.pareto_independent_replica_certificate import (
    FALSE_PASS_BOUND_SEMANTICS,
    FALSE_PASS_EVENT_LABEL,
    OCCUPANCY_EVENT_LABEL,
    OCCUPANCY_METHOD_BONFERRONI,
    OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION,
    REPLICA_PLAN_CONSERVATIVE_UPPER,
    REPLICA_PLAN_EXACT_MINIMUM,
    REPLICA_PLAN_IMPOSSIBLE,
    ProbabilityCertificateError,
    build_false_pass_certificate,
    canonical_rational_string,
    certify_pilot_power,
    clopper_pearson_lower_bracket,
    exact_binomial_survival,
    minimum_pilot_trials,
    mutually_exclusive_cell_occupancy_lower_bound,
    parse_canonical_probability,
    pilot_pass_probability,
    pilot_success_threshold,
    plan_replica_count,
    simultaneous_pilot_power_lower_bound,
    verify_clopper_pearson_lower_at_least,
)


class ExactProbabilityContractTests(unittest.TestCase):
    def test_canonical_probability_text_rejects_float_and_aliases(
        self,
    ) -> None:
        exact = Fraction(1, 1_000_000_000)
        self.assertEqual(
            canonical_rational_string(exact),
            "1/1000000000",
        )
        self.assertEqual(
            parse_canonical_probability("1/1000000000"),
            exact,
        )
        for invalid in (
            1e-9,
            "0.000000001",
            "1e-9",
            "2/2000000000",
            "01/1000000000",
            True,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ProbabilityCertificateError):
                    parse_canonical_probability(invalid)  # type: ignore[arg-type]

    def test_exact_binomial_survival_and_boundaries(self) -> None:
        self.assertEqual(
            exact_binomial_survival(3, 2, Fraction(1, 2)),
            Fraction(1, 2),
        )
        self.assertEqual(
            exact_binomial_survival(5, 0, Fraction(0)),
            Fraction(1),
        )
        self.assertEqual(
            exact_binomial_survival(5, 1, Fraction(0)),
            Fraction(0),
        )
        self.assertEqual(
            exact_binomial_survival(5, 5, Fraction(1)),
            Fraction(1),
        )
        self.assertEqual(
            exact_binomial_survival(5, 6, Fraction(1, 2)),
            Fraction(0),
        )

    def test_exact_mutually_exclusive_cell_occupancy(self) -> None:
        certificate = mutually_exclusive_cell_occupancy_lower_bound(
            (Fraction(1, 4), Fraction(1, 4)),
            replicas=2,
        )
        self.assertEqual(
            certificate.method,
            OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION,
        )
        self.assertTrue(certificate.exact_for_lower_probability_model)
        self.assertEqual(certificate.lower_bound, Fraction(1, 8))
        payload = certificate.to_jsonable()
        self.assertEqual(payload["event"], OCCUPANCY_EVENT_LABEL)
        self.assertEqual(
            payload["method"],
            OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION,
        )
        self.assertEqual(
            payload["probability_lower_bounds"],
            ["1/4", "1/4"],
        )
        self.assertEqual(
            payload["all_cells_hit_probability_lower_bound"],
            "1/8",
        )
        json.dumps(payload, allow_nan=False)

        empty = mutually_exclusive_cell_occupancy_lower_bound((), 0)
        self.assertEqual(empty.lower_bound, 1)
        twenty = mutually_exclusive_cell_occupancy_lower_bound(
            (Fraction(1),) + (Fraction(0),) * 19,
            5,
        )
        self.assertEqual(
            twenty.method,
            OCCUPANCY_METHOD_EXACT_INCLUSION_EXCLUSION,
        )
        self.assertEqual(twenty.lower_bound, 0)

    def test_large_cell_family_uses_labelled_bonferroni_bound(
        self,
    ) -> None:
        bounds = (Fraction(1, 100),) * 21
        replicas = 1000
        certificate = mutually_exclusive_cell_occupancy_lower_bound(
            bounds,
            replicas,
        )
        expected = max(
            Fraction(0),
            1 - 21 * Fraction(99, 100) ** replicas,
        )
        self.assertEqual(
            certificate.method,
            OCCUPANCY_METHOD_BONFERRONI,
        )
        self.assertFalse(certificate.exact_for_lower_probability_model)
        self.assertEqual(certificate.cell_count, 21)
        self.assertEqual(certificate.lower_bound, expected)
        self.assertGreater(certificate.lower_bound, 0)
        self.assertEqual(
            certificate.to_jsonable()["event"],
            OCCUPANCY_EVENT_LABEL,
        )

    def test_occupancy_inputs_fail_closed(self) -> None:
        invalid_cases = (
            ((Fraction(-1, 10),), 1),
            ((Fraction(3, 5), Fraction(1, 2)), 1),
            ((0.1,), 1),
            ((Fraction(1, 2),), -1),
            ((Fraction(1, 2),), True),
        )
        for bounds, replicas in invalid_cases:
            with self.subTest(bounds=bounds, replicas=replicas):
                with self.assertRaises(ProbabilityCertificateError):
                    mutually_exclusive_cell_occupancy_lower_bound(
                        bounds,  # type: ignore[arg-type]
                        replicas,
                    )

    def test_cp_n_equals_x_equals_one_is_never_rounded_up(
        self,
    ) -> None:
        alpha = Fraction(1, 1_000_000_000)
        binary64_value = Fraction.from_float(1e-9)
        self.assertGreater(binary64_value, alpha)

        bracket = clopper_pearson_lower_bracket(
            successes=1,
            trials=1,
            alpha=alpha,
            precision_bits=128,
        )
        self.assertLessEqual(bracket.lower, alpha)
        self.assertGreaterEqual(bracket.upper, alpha)
        self.assertLessEqual(bracket.survival_at_lower, alpha)
        self.assertGreaterEqual(bracket.survival_at_upper, alpha)
        self.assertLessEqual(
            bracket.upper - bracket.lower,
            Fraction(1, 2**128),
        )
        self.assertLess(bracket.conservative_lower, binary64_value)
        self.assertEqual(
            bracket.to_jsonable()["alpha"],
            "1/1000000000",
        )
        json.dumps(bracket.to_jsonable(), allow_nan=False)

    def test_cp_bracket_has_directed_tail_invariants(self) -> None:
        alpha = Fraction(1, 20)
        bracket = clopper_pearson_lower_bracket(
            successes=5,
            trials=12,
            alpha=alpha,
            precision_bits=96,
        )
        self.assertTrue(bracket.tail_equation_applies)
        self.assertLessEqual(bracket.survival_at_lower, alpha)
        self.assertGreaterEqual(bracket.survival_at_upper, alpha)
        self.assertLessEqual(
            bracket.upper - bracket.lower,
            Fraction(1, 2**96),
        )
        self.assertTrue(
            verify_clopper_pearson_lower_at_least(
                5,
                12,
                alpha,
                bracket.lower,
            )
        )
        if not bracket.exact_endpoint:
            self.assertFalse(
                verify_clopper_pearson_lower_at_least(
                    5,
                    12,
                    alpha,
                    bracket.upper,
                )
            )

        zero_success = clopper_pearson_lower_bracket(
            successes=0,
            trials=12,
            alpha=alpha,
        )
        self.assertEqual(zero_success.lower, 0)
        self.assertEqual(zero_success.upper, 0)
        self.assertFalse(zero_success.tail_equation_applies)

    def test_replica_planner_exact_minimum_and_q_boundaries(self) -> None:
        exact = plan_replica_count(Fraction(1, 2), Fraction(1, 8))
        self.assertEqual(exact.status, REPLICA_PLAN_EXACT_MINIMUM)
        self.assertTrue(exact.is_exact_minimum)
        self.assertEqual(exact.replicas, 3)
        self.assertEqual(exact.exact_miss_probability, Fraction(1, 8))
        self.assertEqual(
            exact.exact_predecessor_miss_probability,
            Fraction(1, 4),
        )

        impossible = plan_replica_count(0, Fraction(1, 20))
        self.assertEqual(impossible.status, REPLICA_PLAN_IMPOSSIBLE)
        self.assertFalse(impossible.feasible)

        certain = plan_replica_count(1, Fraction(1, 20))
        self.assertEqual(certain.status, REPLICA_PLAN_EXACT_MINIMUM)
        self.assertEqual(certain.replicas, 1)
        self.assertEqual(certain.exact_miss_probability, 0)

        zero_budget = plan_replica_count(
            Fraction(1, 2),
            0,
        )
        self.assertEqual(zero_budget.status, REPLICA_PLAN_IMPOSSIBLE)
        self.assertEqual(
            plan_replica_count(0, 1).replicas,
            0,
        )

    def test_tiny_q_terminates_with_labelled_conservative_upper(
        self,
    ) -> None:
        q = Fraction(1, 2**60)
        self.assertLess(q, Fraction(1, 2**53))
        budget = Fraction(1, 1000)
        plan = plan_replica_count(q, budget)
        self.assertEqual(
            plan.status,
            REPLICA_PLAN_CONSERVATIVE_UPPER,
        )
        self.assertFalse(plan.is_exact_minimum)
        self.assertIsNotNone(plan.replicas)
        self.assertIsNone(plan.exact_miss_probability)
        self.assertEqual(plan.dyadic_exponent, 10)
        assert plan.replicas is not None
        self.assertGreaterEqual(q * plan.replicas, 10)
        self.assertLessEqual(
            plan.certified_miss_upper_bound,
            budget,
        )
        self.assertIn(
            "dyadic_exponential_bound",
            plan.proof_method,
        )
        json.dumps(plan.to_jsonable(), allow_nan=False)

    def test_pilot_threshold_and_power_are_exact_and_monotone(
        self,
    ) -> None:
        n = 20
        alpha = Fraction(1, 20)
        threshold_low_target = pilot_success_threshold(
            n,
            Fraction(1, 4),
            alpha,
        )
        threshold_high_target = pilot_success_threshold(
            n,
            Fraction(1, 2),
            alpha,
        )
        self.assertIsNotNone(threshold_low_target)
        self.assertIsNotNone(threshold_high_target)
        assert threshold_low_target is not None
        assert threshold_high_target is not None
        self.assertLessEqual(
            threshold_low_target,
            threshold_high_target,
        )
        self.assertLessEqual(
            exact_binomial_survival(
                n,
                threshold_low_target,
                Fraction(1, 4),
            ),
            alpha,
        )
        if threshold_low_target > 0:
            self.assertGreater(
                exact_binomial_survival(
                    n,
                    threshold_low_target - 1,
                    Fraction(1, 4),
                ),
                alpha,
            )

        power_mid = pilot_pass_probability(
            n,
            threshold_low_target,
            Fraction(1, 2),
        )
        power_high = pilot_pass_probability(
            n,
            threshold_low_target,
            Fraction(3, 4),
        )
        self.assertLessEqual(power_mid, power_high)

        certificate = certify_pilot_power(
            n,
            Fraction(1, 4),
            Fraction(1, 2),
            alpha,
            minimum_acceptable_pass_probability=Fraction(1, 2),
        )
        self.assertEqual(
            certificate.critical_successes,
            threshold_low_target,
        )
        self.assertEqual(
            certificate.pass_probability_lower_bound,
            power_mid,
        )
        json.dumps(certificate.to_jsonable(), allow_nan=False)

    def test_minimum_pilot_trials_and_simultaneous_power(self) -> None:
        certificate = minimum_pilot_trials(
            Fraction(1, 4),
            Fraction(3, 4),
            Fraction(1, 20),
            Fraction(1, 10),
            max_trials=100,
        )
        self.assertGreaterEqual(
            certificate.pass_probability_lower_bound,
            Fraction(9, 10),
        )
        for smaller_n in range(1, certificate.trials):
            smaller = certify_pilot_power(
                smaller_n,
                Fraction(1, 4),
                Fraction(3, 4),
                Fraction(1, 20),
                minimum_acceptable_pass_probability=Fraction(9, 10),
            )
            self.assertLess(
                smaller.pass_probability_lower_bound,
                Fraction(9, 10),
            )
        self.assertEqual(
            simultaneous_pilot_power_lower_bound(
                (Fraction(19, 20), Fraction(9, 10))
            ),
            Fraction(17, 20),
        )

    def test_pilot_power_requires_positive_target_and_frozen_power_gate(self) -> None:
        with self.assertRaises(ProbabilityCertificateError):
            certify_pilot_power(
                10,
                0,
                Fraction(1, 2),
                Fraction(1, 20),
                minimum_acceptable_pass_probability=Fraction(1, 2),
            )

        infeasible = certify_pilot_power(
            1,
            Fraction(9, 10),
            Fraction(19, 20),
            Fraction(1, 100),
            minimum_acceptable_pass_probability=Fraction(1, 2),
        )
        self.assertIsNone(infeasible.critical_successes)
        self.assertFalse(infeasible.power_gate)

    def test_false_pass_label_is_joint_not_conditional(self) -> None:
        certificate = build_false_pass_certificate(
            Fraction(1, 100),
            Fraction(1, 50),
        )
        payload = certificate.to_jsonable()
        self.assertEqual(payload["event"], FALSE_PASS_EVENT_LABEL)
        self.assertEqual(
            payload["semantics"],
            FALSE_PASS_BOUND_SEMANTICS,
        )
        self.assertEqual(payload["alpha_plus_delta"], "3/100")
        self.assertFalse(
            payload["primary_bound_is_conditional_given_pass"]
        )
        self.assertIsNone(
            payload["derived_conditional_failure_upper_bound"]
        )

        conditional = build_false_pass_certificate(
            Fraction(1, 100),
            Fraction(1, 50),
            pass_probability_lower_bound=Fraction(1, 2),
        )
        self.assertEqual(
            conditional.derived_conditional_failure_upper_bound,
            Fraction(3, 50),
        )


if __name__ == "__main__":
    unittest.main()

