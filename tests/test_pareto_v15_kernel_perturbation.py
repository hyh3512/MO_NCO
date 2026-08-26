from __future__ import annotations

import unittest
from fractions import Fraction

from mo_nco.pareto_kernel_perturbation import (
    IndeterminateIntervalDecision,
    RationalInterval,
    certify_kernel_perturbation_bound,
    decide_strict_less,
)


class KernelPerturbationV15Tests(unittest.TestCase):
    def test_conditional_bounds_use_exact_rational_arithmetic(self) -> None:
        certificate = certify_kernel_perturbation_bound(
            beta=3,
            uniform_energy_error=Fraction(1, 100),
            steps=5,
        )

        self.assertEqual(certificate.energy_difference_error_upper, "1/50")
        self.assertEqual(
            certificate.acceptance_probability_error_upper,
            "3/50",
        )
        self.assertEqual(certificate.kernel_row_l1_error_upper, "3/25")
        self.assertEqual(certificate.finite_step_tv_error_upper, "3/10")
        self.assertTrue(
            certificate.requires_verified_uniform_energy_interval
        )
        self.assertFalse(
            certificate.uniform_energy_interval_verified_by_this_module
        )
        self.assertFalse(certificate.implementation_kernel_equality_claimed)

    def test_interval_decision_refines_or_fails_at_boundary(self) -> None:
        self.assertTrue(
            decide_strict_less(
                RationalInterval.make(0, 1),
                RationalInterval.make(2, 3),
            )
        )
        self.assertFalse(
            decide_strict_less(
                RationalInterval.make(2, 3),
                RationalInterval.make(0, 1),
            )
        )
        with self.assertRaises(IndeterminateIntervalDecision):
            decide_strict_less(
                RationalInterval.make(0, 2),
                RationalInterval.make(1, 3),
            )


if __name__ == "__main__":
    unittest.main()

