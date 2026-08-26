from __future__ import annotations

import unittest
from fractions import Fraction

from mo_nco.pareto_reference_fidelity import (
    ReferenceFidelityError,
    certify_reference_fidelity_composition,
)


class ReferenceFidelityV15Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.true_front = (
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(0)),
            (Fraction(1, 2), Fraction(1, 2)),
        )
        self.endpoints = (
            (Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(0)),
        )

    def test_incomplete_reference_cannot_claim_zero_true_front_error(self) -> None:
        with self.assertRaisesRegex(
            ReferenceFidelityError,
            "supplied-front cover",
        ):
            certify_reference_fidelity_composition(
                true_front=self.true_front,
                frozen_reference=self.endpoints,
                approximation=self.endpoints,
                reference_fidelity_vector=(0, 0),
                algorithm_reference_vector=(0, 0),
                p="2",
                hv_reference=(2, 2),
                supplied_front_provenance_note="unverified exact-enumeration note",
            )

    def test_declared_reference_fidelity_composes_with_algorithm_error(self) -> None:
        certificate = certify_reference_fidelity_composition(
            true_front=self.true_front,
            frozen_reference=self.endpoints,
            approximation=self.endpoints,
            reference_fidelity_vector=(Fraction(1, 2), Fraction(1, 2)),
            algorithm_reference_vector=(0, 0),
            p="2",
            hv_reference=(2, 2),
            supplied_front_provenance_note=(
                "exhaustive enumeration of the finite feasible state space"
            ),
        )

        self.assertTrue(certificate.composed_cover_verified)
        self.assertEqual(
            certificate.composed_additive_vector,
            ("1/2", "1/2"),
        )
        self.assertIn("conditional", certificate.scope)
        self.assertIn("supplied_front_relative", certificate.scope)
        self.assertFalse(
            certificate.external_true_front_completeness_verified
        )
        self.assertFalse(certificate.true_front_coverage_claimed)
        self.assertNotEqual(
            certificate.ordinary_igd_supplied_front_upper,
            "0",
        )

    def test_true_front_claim_requires_bound_completeness_evidence(self) -> None:
        with self.assertRaisesRegex(
            ReferenceFidelityError,
            "provenance note",
        ):
            certify_reference_fidelity_composition(
                true_front=self.true_front,
                frozen_reference=self.true_front,
                approximation=self.true_front,
                reference_fidelity_vector=(0, 0),
                algorithm_reference_vector=(0, 0),
                p="1",
                hv_reference=(2, 2),
                supplied_front_provenance_note="",
            )


if __name__ == "__main__":
    unittest.main()

