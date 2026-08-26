from __future__ import annotations

import random
import unittest

from mo_nco.evaluation import CountingTSPInstance
from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.moves import random_tour, sample_two_opt_indices, two_opt_at


class IncrementalEvaluationTests(unittest.TestCase):
    def test_nonintegral_symmetric_instance_uses_exact_full_fallback(self) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(12, seed=11)
        rng = random.Random(5)
        tour = random_tour(instance.num_cities, rng)
        objective = instance.evaluate(tour)
        i, j = sample_two_opt_indices(instance.num_cities, rng)

        child = two_opt_at(tour, i, j)
        delta_objective = instance.evaluate_two_opt(tour, objective, i, j)
        full_objective = instance.evaluate(child)

        self.assertFalse(instance.exact_two_opt_delta_in_binary64)
        self.assertEqual(delta_objective, full_objective)

    def test_safe_integer_multiobjective_delta_is_bitwise_exact(self) -> None:
        rng = random.Random(61)
        matrices = []
        for _ in range(3):
            matrix = [[0.0] * 14 for _ in range(14)]
            for left in range(14):
                for right in range(left + 1, 14):
                    value = float(rng.randint(1, 10_000))
                    matrix[left][right] = value
                    matrix[right][left] = value
            matrices.append(tuple(tuple(row) for row in matrix))
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            tuple(matrices),
            name="safe_integer_three_objective",
        )
        self.assertTrue(instance.exact_two_opt_delta_in_binary64)
        for _ in range(100):
            tour = random_tour(instance.num_cities, rng)
            current = instance.evaluate(tour)
            i, j = sample_two_opt_indices(instance.num_cities, rng)
            child = two_opt_at(tour, i, j)
            self.assertEqual(
                instance.evaluate_two_opt(tour, current, i, j),
                instance.evaluate(child),
            )

    def test_cancellation_counterexample_fails_safe_to_full_sum(self) -> None:
        matrix = (
            (0.0, 1e16, 1.0, 1.0),
            (1e16, 0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 0.0),
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (matrix,),
            name="binary64_cancellation_counterexample",
        )
        tour = (0, 1, 2, 3)
        current = instance.evaluate(tour)
        child = two_opt_at(tour, 1, 2)
        self.assertFalse(instance.exact_two_opt_delta_in_binary64)
        self.assertEqual(
            instance.evaluate_two_opt(tour, current, 1, 2),
            instance.evaluate(child),
        )
        self.assertEqual(instance.evaluate(child), (4.0,))

    def test_safe_domain_product_gate_uses_exact_integer_arithmetic(
        self,
    ) -> None:
        n = 107
        large_edge = (2**53 + 1) // n
        matrix = [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]
        for city in range(n):
            nxt = (city + 1) % n
            matrix[city][nxt] = float(large_edge)
            matrix[nxt][city] = float(large_edge)
        instance = MultiObjectiveTSPInstance.from_distance_matrices(
            (tuple(tuple(row) for row in matrix),),
            name="rounded_product_boundary",
        )
        self.assertGreater(n * large_edge, 2**53)
        self.assertFalse(instance.exact_two_opt_delta_in_binary64)
        tour = tuple(range(n))
        child = two_opt_at(tour, 1, 2)
        self.assertEqual(
            instance.evaluate_two_opt(
                tour,
                instance.evaluate(tour),
                1,
                2,
            ),
            instance.evaluate(child),
        )

    def test_counted_two_opt_delta_charges_one_true_evaluation(self) -> None:
        base = MultiObjectiveTSPInstance.random_biobjective(9, seed=3)
        counted = CountingTSPInstance(base, max_evaluations=10)
        rng = random.Random(7)
        tour = random_tour(counted.num_cities, rng)
        objective = counted.evaluate(tour)
        i, j = sample_two_opt_indices(counted.num_cities, rng)

        child_objective = counted.evaluate_two_opt(tour, objective, i, j)
        self.assertEqual(counted.evaluations, 2)
        self.assertEqual(len(child_objective), counted.num_objectives)

    def test_asymmetric_two_opt_falls_back_to_full_exact_evaluation(self) -> None:
        matrix = (
            (0.0, 1.0, 5.0, 7.0),
            (4.0, 0.0, 2.0, 3.0),
            (6.0, 9.0, 0.0, 8.0),
            (2.0, 4.0, 1.0, 0.0),
        )
        instance = MultiObjectiveTSPInstance.from_distance_matrices((matrix,), name="asymmetric")
        tour = (0, 1, 2, 3)
        objective = instance.evaluate(tour)
        child = two_opt_at(tour, 1, 3)

        self.assertEqual(instance.evaluate_two_opt(tour, objective, 1, 3), instance.evaluate(child))


if __name__ == "__main__":
    unittest.main()

