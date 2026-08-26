from __future__ import annotations

import json
import math
import random
import unittest
from dataclasses import replace

from mo_nco.pareto_bounds import (
    hypervolume_minimization,
    nondominated_points,
)
from mo_nco.pareto_sparse_compression_certificate import (
    SPARSE_COMPRESSION_CERTIFICATE_SCHEMA_V13,
    SparseCompressionCertificateError,
    certify_sparse_finite_reference_compression,
)
from mo_nco.pareto_sparse_reference import (
    SparseReferenceCover,
    greedy_maximal_reference_net,
)


Point = tuple[float, ...]


def _lp(values: tuple[float, ...], p: float) -> float:
    if math.isinf(p):
        return max(values)
    return sum(value**p for value in values) ** (1.0 / p)


def _igd(
    approximation: tuple[Point, ...],
    reference: tuple[Point, ...],
    p: float,
    *,
    plus: bool,
) -> float:
    distances = []
    for reference_point in reference:
        distances.append(
            min(
                _lp(
                    tuple(
                        (
                            max(candidate_value - reference_value, 0.0)
                            if plus
                            else abs(candidate_value - reference_value)
                        )
                        for candidate_value, reference_value in zip(
                            candidate,
                            reference_point,
                        )
                    ),
                    p,
                )
                for candidate in approximation
            )
        )
    return sum(distances) / len(distances)


class SparseFiniteReferenceCompressionCertificateTests(unittest.TestCase):
    def test_metric_helper_uses_arithmetic_reference_mean(self) -> None:
        self.assertEqual(
            _igd(
                ((0.0, 0.0),),
                ((0.0, 0.0), (2.0, 0.0)),
                2.0,
                plus=False,
            ),
            1.0,
        )

    def setUp(self) -> None:
        self.points = (
            (0.0, 1.0),
            (0.05, 0.95),
            (0.5, 0.5),
            (1.0, 0.0),
        )
        self.cover = greedy_maximal_reference_net(
            self.points,
            cover_radius=0.1,
            p_norm=math.inf,
        )

    def _certificate(self, *, p: float = 2.0):
        return certify_sparse_finite_reference_compression(
            self.points,
            self.cover,
            cell_width_vector=(0.02, 0.02),
            igd_p=p,
            hv_reference=(1.2, 1.2),
        )

    def test_certificate_binds_source_cover_and_scopes(self) -> None:
        certificate = self._certificate()
        self.assertEqual(
            certificate.schema,
            SPARSE_COMPRESSION_CERTIFICATE_SCHEMA_V13,
        )
        self.assertEqual(certificate.full_reference_count, 4)
        self.assertEqual(certificate.anchor_indices, self.cover.anchor_indices)
        self.assertEqual(
            certificate.anchor_reference_set,
            self.cover.anchor_objectives,
        )
        self.assertEqual(
            certificate.retained_witness_support_cardinality_bound,
            len(self.cover.anchor_indices),
        )
        self.assertEqual(
            certificate.archive_cardinality_bound,
            len(self.cover.anchor_indices),
        )
        self.assertEqual(
            certificate.nondominated_archive_cardinality_bound,
            len(self.cover.anchor_indices),
        )
        self.assertEqual(len(certificate.source_reference_sha256), 64)
        self.assertEqual(len(certificate.canonical_cover_sha256), 64)
        self.assertIn(
            "before nondominated filtering",
            certificate.ordinary_igd_scope,
        )
        self.assertIn(
            "nondominated view",
            certificate.igd_plus_and_hv_scope,
        )
        self.assertFalse(
            certificate.reference_count_independent_universal_compression_claimed
        )
        json.dumps(certificate.to_jsonable(), allow_nan=False)

    def test_p_one_two_and_infinity_are_supported(self) -> None:
        certificates = {
            p: self._certificate(p=p)
            for p in (1.0, 2.0, math.inf)
        }
        inflation = certificates[1.0].coordinatewise_inflation_vector
        self.assertGreaterEqual(
            certificates[1.0].ordinary_igd_bound,
            sum(inflation),
        )
        self.assertGreaterEqual(
            certificates[2.0].ordinary_igd_bound,
            math.hypot(*inflation),
        )
        self.assertGreaterEqual(
            certificates[math.inf].ordinary_igd_bound,
            max(inflation),
        )
        self.assertEqual(
            certificates[2.0].ordinary_igd_bound,
            certificates[2.0].igd_plus_bound,
        )
        self.assertEqual(
            certificates[math.inf].to_jsonable()["metric_p_norm"],
            "infinity",
        )

    def test_random_same_anchor_witnesses_obey_all_metric_bounds(self) -> None:
        widths = (0.03, 0.04, 0.02)
        for seed in range(20):
            rng = random.Random(seed)
            points = tuple(
                (
                    index / 14.0,
                    1.0 - index / 14.0,
                    0.2 + 0.6 * rng.random(),
                )
                for index in range(15)
            )
            cover = greedy_maximal_reference_net(
                points,
                cover_radius=0.22,
                p_norm=math.inf,
            )
            witnesses = tuple(
                tuple(
                    value + rng.uniform(-width, width)
                    for value, width in zip(anchor, widths)
                )
                for anchor in cover.anchor_objectives
            )
            nondominated_witnesses = nondominated_points(witnesses)
            for p in (1.0, 2.0, math.inf):
                with self.subTest(seed=seed, p=p):
                    certificate = (
                        certify_sparse_finite_reference_compression(
                            points,
                            cover,
                            cell_width_vector=widths,
                            igd_p=p,
                            hv_reference=(1.5, 1.5, 1.5),
                        )
                    )
                    self.assertLessEqual(
                        _igd(
                            witnesses,
                            points,
                            p,
                            plus=False,
                        ),
                        certificate.ordinary_igd_bound + 1e-12,
                    )
                    self.assertLessEqual(
                        _igd(
                            nondominated_witnesses,
                            points,
                            p,
                            plus=True,
                        ),
                        certificate.igd_plus_bound + 1e-12,
                    )
                    hv_deficit = max(
                        0.0,
                        hypervolume_minimization(
                            points,
                            (1.5, 1.5, 1.5),
                        )
                        - hypervolume_minimization(
                            nondominated_witnesses,
                            (1.5, 1.5, 1.5),
                        ),
                    )
                    self.assertLessEqual(
                        hv_deficit,
                        certificate.shifted_front_hv_deficit_bound
                        + 1e-11,
                    )

    def test_exact_radii_and_addition_are_rounded_up(self) -> None:
        points = ((0.1, 0.9), (0.3, 0.7))
        cover = greedy_maximal_reference_net(
            points,
            cover_radius=0.25,
            p_norm=math.inf,
        )
        certificate = certify_sparse_finite_reference_compression(
            points,
            cover,
            cell_width_vector=(0.1, math.nextafter(0.0, math.inf)),
            igd_p=1.0,
            hv_reference=(1.0, 1.0),
        )
        exact_radius = abs(
            points[1][0].as_integer_ratio()[0]
            / points[1][0].as_integer_ratio()[1]
            - points[0][0].as_integer_ratio()[0]
            / points[0][0].as_integer_ratio()[1]
        )
        self.assertGreaterEqual(
            certificate.reference_to_anchor_coordinate_radii[0],
            exact_radius,
        )
        self.assertGreaterEqual(
            certificate.coordinatewise_inflation_vector[0],
            certificate.reference_to_anchor_coordinate_radii[0] + 0.1,
        )
        self.assertGreater(
            certificate.coordinatewise_inflation_vector[1],
            certificate.reference_to_anchor_coordinate_radii[1],
        )

    def test_forged_or_source_mismatched_cover_fails_closed(self) -> None:
        forged_assignment = replace(
            self.cover,
            assignment_by_reference=(0, 0, 0, 2),
            cluster_sizes=(2, 1, 1),
        )
        with self.assertRaisesRegex(
            SparseCompressionCertificateError,
            "canonical integrity",
        ):
            certify_sparse_finite_reference_compression(
                self.points,
                forged_assignment,
                cell_width_vector=(0.02, 0.02),
                igd_p=2.0,
                hv_reference=(1.2, 1.2),
            )

        changed_source = (
            self.points[0],
            (0.06, 0.94),
            self.points[2],
            self.points[3],
        )
        with self.assertRaisesRegex(
            SparseCompressionCertificateError,
            "canonical integrity",
        ):
            certify_sparse_finite_reference_compression(
                changed_source,
                self.cover,
                cell_width_vector=(0.02, 0.02),
                igd_p=2.0,
                hv_reference=(1.2, 1.2),
            )

    def test_valid_but_noncanonical_tie_assignment_fails_closed(self) -> None:
        points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        forged = SparseReferenceCover(
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
        with self.assertRaises(SparseCompressionCertificateError):
            certify_sparse_finite_reference_compression(
                points,
                forged,
                cell_width_vector=(0.1, 0.1),
                igd_p=2.0,
                hv_reference=(3.0, 1.0),
            )

    def test_boolean_cover_scalars_fail_closed(self) -> None:
        singleton = greedy_maximal_reference_net(
            ((0.0, 0.0),),
            cover_radius=0.0,
            p_norm=1.0,
        )
        for forged in (
            replace(singleton, p_norm=True),
            replace(singleton, requested_cover_radius=False),
            replace(singleton, realized_cover_radius=False),
            replace(singleton, anchor_objectives=((False, 0.0),)),
        ):
            with self.subTest(forged=forged):
                with self.assertRaises(SparseCompressionCertificateError):
                    certify_sparse_finite_reference_compression(
                        ((0.0, 0.0),),
                        forged,
                        cell_width_vector=(0.1, 0.1),
                        igd_p=2.0,
                        hv_reference=(1.0, 1.0),
                    )

    def test_input_domain_fails_closed(self) -> None:
        invalid_cases = (
            {
                "reference_points": ((0.0, math.nan),),
                "cover": self.cover,
            },
            {
                "reference_points": ((False, 0.0),),
                "cover": self.cover,
            },
            {"cell_width_vector": (0.1, math.inf)},
            {"cell_width_vector": (-0.1, 0.1)},
            {"cell_width_vector": (True, 0.1)},
            {"igd_p": 3.0},
            {"igd_p": -math.inf},
            {"igd_p": True},
            {"hv_reference": (1.2, math.nan)},
            {"hv_reference": (0.9, 1.2)},
        )
        base = {
            "reference_points": self.points,
            "cover": self.cover,
            "cell_width_vector": (0.02, 0.02),
            "igd_p": 2.0,
            "hv_reference": (1.2, 1.2),
        }
        for override in invalid_cases:
            with self.subTest(override=override):
                kwargs = dict(base)
                kwargs.update(override)
                with self.assertRaises(SparseCompressionCertificateError):
                    certify_sparse_finite_reference_compression(**kwargs)

    def test_nonfinite_reported_bound_is_rejected(self) -> None:
        points = ((0.0, 0.0),)
        cover = greedy_maximal_reference_net(
            points,
            cover_radius=0.0,
        )
        with self.assertRaisesRegex(
            SparseCompressionCertificateError,
            "finite binary64",
        ):
            certify_sparse_finite_reference_compression(
                points,
                cover,
                cell_width_vector=(1e308, 1e308),
                igd_p=math.inf,
                hv_reference=(1e308, 1e308),
            )

    def test_source_hash_binds_order_and_exact_float_values(self) -> None:
        forward = self._certificate()
        reversed_points = tuple(reversed(self.points))
        reversed_cover = greedy_maximal_reference_net(
            reversed_points,
            cover_radius=0.1,
            p_norm=math.inf,
        )
        reverse = certify_sparse_finite_reference_compression(
            reversed_points,
            reversed_cover,
            cell_width_vector=(0.02, 0.02),
            igd_p=2.0,
            hv_reference=(1.2, 1.2),
        )
        self.assertNotEqual(
            forward.source_reference_sha256,
            reverse.source_reference_sha256,
        )
        # The geometric cover can remain equivalent while the source binding
        # changes because original reference indices changed.
        self.assertNotEqual(
            forward.canonical_cover_sha256,
            reverse.canonical_cover_sha256,
        )


if __name__ == "__main__":
    unittest.main()

