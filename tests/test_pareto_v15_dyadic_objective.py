from __future__ import annotations

import math
import unittest
from fractions import Fraction
from unittest import mock

import mo_nco.pareto_dyadic_objective as dyadic
from mo_nco.pareto_dyadic_objective import (
    EXACT_EDGE_SUM_CONTRACT,
    DyadicObjectiveEncoding,
    canonical_binary64_fraction,
    encode_objective_matrices,
    exact_tour_scaled_sums,
    update_two_opt_scaled,
)


def _reverse_segment(
    tour: tuple[int, ...],
    i: int,
    j: int,
) -> tuple[int, ...]:
    proposed = list(tour)
    proposed[i : j + 1] = reversed(proposed[i : j + 1])
    return tuple(proposed)


class CanonicalBinary64EncodingTests(unittest.TestCase):
    def test_contract_is_explicitly_limited_to_edge_sum_exactness(self) -> None:
        self.assertEqual(
            EXACT_EDGE_SUM_CONTRACT,
            "exact_edge_sum_then_binary64_objective_v1",
        )
        self.assertNotIn("mh", EXACT_EDGE_SUM_CONTRACT.lower())

    def test_signed_and_subnormal_values_have_exact_canonical_fractions(
        self,
    ) -> None:
        smallest_subnormal = math.ldexp(1.0, -1074)
        self.assertEqual(
            canonical_binary64_fraction(-1.5),
            Fraction(-3, 2),
        )
        self.assertEqual(
            canonical_binary64_fraction(smallest_subnormal),
            Fraction(1, 1 << 1074),
        )
        self.assertEqual(
            canonical_binary64_fraction(-0.0),
            Fraction(0, 1),
        )

        encoding = encode_objective_matrices(
            (
                (
                    (0.0, -1.5, smallest_subnormal),
                    (-1.5, 0.0, 2.0),
                    (smallest_subnormal, 2.0, -0.0),
                ),
            )
        )
        self.assertEqual(encoding.symmetry_flags, (True,))
        self.assertEqual(
            encoding.edge_fraction(0, 0, 1),
            Fraction(-3, 2),
        )
        self.assertEqual(
            encoding.edge_fraction(0, 0, 2),
            Fraction(1, 1 << 1074),
        )

    def test_rejects_nonfinite_binary64_values(self) -> None:
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite binary64",
                ):
                    canonical_binary64_fraction(bad_value)
                with self.assertRaisesRegex(
                    ValueError,
                    "finite binary64",
                ):
                    encode_objective_matrices(
                        (
                            (
                                (0.0, bad_value, 1.0),
                                (1.0, 0.0, 1.0),
                                (1.0, 1.0, 0.0),
                            ),
                        )
                    )

    def test_nextafter_counterexample_remains_distinct_until_decode(
        self,
    ) -> None:
        base = 1.0e16
        larger = math.nextafter(base, math.inf)
        matrix = [
            [0.0 if i == j else base for j in range(4)]
            for i in range(4)
        ]
        matrix[0][1] = larger
        matrix[1][0] = larger
        encoding = encode_objective_matrices((matrix,))

        x = (0, 1, 2, 3)
        y = (0, 2, 1, 3)
        exact_x = exact_tour_scaled_sums(encoding, x)
        exact_y = exact_tour_scaled_sums(encoding, y)

        self.assertEqual(
            encoding.scaled_as_fraction(exact_x)[0]
            - encoding.scaled_as_fraction(exact_y)[0],
            Fraction(2, 1),
        )
        self.assertNotEqual(exact_x, exact_y)
        self.assertEqual(encoding.decode(exact_x), (4.0e16,))
        self.assertEqual(encoding.decode(exact_y), (4.0e16,))


class ExactTwoOptUpdateTests(unittest.TestCase):
    @staticmethod
    def _symmetric_matrix(scale: float) -> tuple[tuple[float, ...], ...]:
        n = 6
        rows = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                value = scale * ((i + 1) * 7 - (j + 2) * 3) / 8.0
                rows[i][j] = value
                rows[j][i] = value
        return tuple(tuple(row) for row in rows)

    @staticmethod
    def _asymmetric_matrix() -> tuple[tuple[float, ...], ...]:
        n = 6
        return tuple(
            tuple(
                0.0
                if i == j
                else ((i + 2) * 11 - (j + 1) * 5) / 16.0
                for j in range(n)
            )
            for i in range(n)
        )

    def test_symmetric_update_equals_full_exact_recompute(self) -> None:
        encoding = encode_objective_matrices(
            (
                self._symmetric_matrix(1.0),
                self._symmetric_matrix(-0.5),
            )
        )
        self.assertEqual(encoding.symmetry_flags, (True, True))
        tour = (0, 1, 4, 2, 5, 3)
        current = exact_tour_scaled_sums(encoding, tour)

        for i, j in ((1, 1), (1, 4), (2, 5)):
            with self.subTest(i=i, j=j):
                updated = update_two_opt_scaled(
                    encoding,
                    tour,
                    current,
                    i,
                    j,
                )
                expected = exact_tour_scaled_sums(
                    encoding,
                    _reverse_segment(tour, i, j),
                )
                self.assertEqual(updated, expected)

    def test_asymmetric_update_equals_full_exact_recompute(self) -> None:
        encoding = encode_objective_matrices(
            (self._asymmetric_matrix(),)
        )
        self.assertEqual(encoding.symmetry_flags, (False,))
        tour = (0, 5, 1, 4, 2, 3)
        current = exact_tour_scaled_sums(encoding, tour)

        for i, j in ((1, 1), (1, 4), (2, 5)):
            with self.subTest(i=i, j=j):
                updated = encoding.update_two_opt_scaled(
                    tour,
                    current,
                    i,
                    j,
                )
                expected = encoding.exact_tour_scaled_sums(
                    _reverse_segment(tour, i, j)
                )
                self.assertEqual(updated, expected)

    def test_update_uses_cached_symmetry_without_rescanning_matrices(
        self,
    ) -> None:
        matrices = (
            self._symmetric_matrix(1.0),
            self._asymmetric_matrix(),
        )
        original_scan = dyadic._objective_is_symmetric
        with mock.patch.object(
            dyadic,
            "_objective_is_symmetric",
            wraps=original_scan,
        ) as scan:
            encoding = DyadicObjectiveEncoding.from_binary64_matrices(
                matrices
            )
            self.assertEqual(scan.call_count, len(matrices))
            scan.reset_mock()

            tour = (0, 2, 4, 1, 5, 3)
            current = encoding.exact_tour_scaled_sums(tour)
            encoding.update_two_opt_scaled(tour, current, 1, 4)

            self.assertEqual(scan.call_count, 0)

        with mock.patch.object(
            dyadic,
            "_objective_is_symmetric",
            side_effect=AssertionError("hot-path matrix rescan"),
        ):
            encoding.update_two_opt_scaled(tour, current, 1, 4)

    def test_update_argument_checks_are_constant_time_in_tour_length(
        self,
    ) -> None:
        encoding = encode_objective_matrices(
            (self._symmetric_matrix(1.0),)
        )
        tour = (0, 1, 2, 3, 4, 5)
        current = encoding.exact_tour_scaled_sums(tour)

        for i, j in ((0, 2), (1, 6)):
            with self.subTest(i=i, j=j):
                with self.assertRaisesRegex(ValueError, "1 <= i <= j < n"):
                    encoding.update_two_opt_scaled(
                        tour,
                        current,
                        i,
                        j,
                    )
        with self.assertRaisesRegex(
            ValueError,
            "objective dimension",
        ):
            encoding.update_two_opt_scaled(tour, (), 1, 2)


if __name__ == "__main__":
    unittest.main()

