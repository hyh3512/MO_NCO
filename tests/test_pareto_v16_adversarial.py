from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
import unittest

from mo_nco.pareto_adaptive_type_cell import exact_confirm_risk_allocation
from mo_nco.pareto_intrinsic_dimension import (
    IntrinsicDimensionError,
    certify_ordered_bilipschitz_reference_family,
)
from mo_nco.pareto_shared_categorical_design import (
    exact_shared_confirm_allocation,
    rational_pairwise_transportation_lower_bound,
)


class ParetoV16AdversarialTests(unittest.TestCase):
    def test_exact_confirm_greedy_matches_bruteforce_on_117_small_cases(self) -> None:
        def brute(q_values, delta, max_total=80):
            for total in range(max_total + 1):
                for counts in product(range(total + 1), repeat=len(q_values)):
                    if sum(counts) != total:
                        continue
                    risk = sum(
                        (1 - q) ** count
                        for q, count in zip(q_values, counts)
                    )
                    if risk <= delta:
                        return total
            return None

        checked = 0
        pool = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
        for cell_count in (1, 2, 3):
            for probabilities in product(pool, repeat=cell_count):
                for delta in (Fraction(1, 2), Fraction(1, 5), Fraction(1, 10)):
                    certificate = exact_confirm_risk_allocation(
                        {
                            f"c{index}": probability
                            for index, probability in enumerate(probabilities)
                        },
                        union_miss_budget=delta,
                    )
                    self.assertEqual(
                        certificate.total_replicas,
                        brute(probabilities, delta),
                    )
                    checked += 1
        self.assertEqual(checked, 117)

    def test_shared_assignment_and_counts_match_global_bruteforce(self) -> None:
        def brute(matrix, delta, max_total=16):
            type_ids = tuple(sorted(matrix))
            cell_ids = tuple(sorted(next(iter(matrix.values()))))
            best = None
            for assignment in product(range(len(type_ids)), repeat=len(cell_ids)):
                if any(matrix[type_ids[assignment[j]]][cell_ids[j]] <= 0 for j in range(len(cell_ids))):
                    continue
                for total in range(max_total + 1):
                    found = False
                    for counts in product(range(total + 1), repeat=len(type_ids)):
                        if sum(counts) != total:
                            continue
                        risk = sum(
                            (1 - matrix[type_ids[assignment[j]]][cell_ids[j]])
                            ** counts[assignment[j]]
                            for j in range(len(cell_ids))
                        )
                        if risk <= delta:
                            candidate = (total, risk, assignment, counts)
                            if best is None or candidate < best:
                                best = candidate
                            found = True
                    if found:
                        break
            return best

        matrices = (
            {
                "A": {"c0": Fraction(3, 4), "c1": Fraction(1, 4)},
                "B": {"c0": Fraction(1, 4), "c1": Fraction(3, 4)},
            },
            {
                "A": {"c0": Fraction(1, 2), "c1": Fraction(1, 3)},
                "B": {"c0": Fraction(2, 5), "c1": Fraction(1, 2)},
            },
            {
                "A": {"c0": Fraction(1, 2), "c1": Fraction(1, 4), "c2": Fraction(1, 4)},
                "B": {"c0": Fraction(1, 4), "c1": Fraction(1, 2), "c2": Fraction(1, 4)},
            },
        )
        checked = 0
        for matrix in matrices:
            for delta in (Fraction(1, 2), Fraction(1, 5), Fraction(1, 10)):
                certificate = exact_shared_confirm_allocation(
                    matrix,
                    union_miss_budget=delta,
                    max_assignments=1000,
                    max_total_replicas=1000,
                )
                exact = brute(matrix, delta)
                self.assertIsNotNone(exact)
                assert exact is not None
                self.assertEqual(certificate.total_replicas, exact[0])
                self.assertEqual(Fraction(certificate.total_union_miss_upper), exact[1])
                checked += 1
        self.assertEqual(checked, 9)

    def test_rational_transport_bound_is_below_exact_midpoint_ratio(self) -> None:
        certificate = rational_pairwise_transportation_lower_bound(
            cell_id="c",
            best_type="A",
            challenger_type="B",
            best_probability="4/5",
            challenger_probability="2/5",
            error_probability="1/20",
            log_series_terms=64,
        )
        lower = Fraction(certificate.expected_total_samples_lower)
        with localcontext() as context:
            context.prec = 120
            p_star = Decimal(4) / Decimal(5)
            p_other = Decimal(2) / Decimal(5)
            midpoint = (p_star + p_other) / Decimal(2)
            alpha = Decimal(1) / Decimal(20)
            decision = (Decimal(1) - Decimal(2) * alpha) * (
                (Decimal(1) - alpha) / alpha
            ).ln()
            def kl(p, q):
                return p * (p / q).ln() + (Decimal(1) - p) * (
                    (Decimal(1) - p) / (Decimal(1) - q)
                ).ln()
            exact_ratio = decision / max(kl(p_star, midpoint), kl(p_other, midpoint))
            lower_decimal = Decimal(lower.numerator) / Decimal(lower.denominator)
            self.assertLessEqual(lower_decimal, exact_ratio)

    def test_false_bilipschitz_family_is_rejected(self) -> None:
        with self.assertRaises(IntrinsicDimensionError):
            certify_ordered_bilipschitz_reference_family(
                ((0, 0), (1, 0), (1, 100)),
                lower_constant=1,
                upper_constant=1,
                tau=1,
            )


if __name__ == "__main__":
    unittest.main()

