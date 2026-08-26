from __future__ import annotations

from fractions import Fraction
import unittest

from mo_nco.pareto_shared_categorical_experiment import (
    SharedCategoricalPilotError,
    plan_shared_confirm_from_pilot,
    run_shared_categorical_successive_elimination,
)


class SharedCategoricalPilotTests(unittest.TestCase):
    def test_one_endpoint_updates_all_cell_indicators(self) -> None:
        sequences = {
            "A": ["left"] * 90 + ["right"] * 10,
            "B": ["right"] * 90 + ["left"] * 10,
        }
        def sample(type_id: str, index: int) -> str:
            return sequences[type_id][index % len(sequences[type_id])]
        result = run_shared_categorical_successive_elimination(
            type_ids=("A", "B"),
            cell_ids=("left", "right"),
            sample_endpoint_cell=sample,
            familywise_error="1/20",
            max_rounds=100,
        )
        selected = {item.cell_id: item.selected_type for item in result.cell_results}
        self.assertEqual(selected, {"left": "A", "right": "B"})
        self.assertEqual(result.total_endpoint_replicas, 2 * result.final_round)
        self.assertLess(result.total_endpoint_replicas, 4 * result.final_round)
        self.assertTrue(result.optional_stopping_safe)
        self.assertTrue(all(item.anytime_mass_lower_bounds for item in result.cell_results))
        allocation = plan_shared_confirm_from_pilot(
            result, union_miss_budget="1/20"
        )
        self.assertTrue(allocation.exact_single_type_assignment_optimum)
        self.assertLessEqual(Fraction(allocation.total_union_miss_upper), Fraction(1, 20))

    def test_undeclared_cell_fails_closed(self) -> None:
        with self.assertRaisesRegex(SharedCategoricalPilotError, "undeclared"):
            run_shared_categorical_successive_elimination(
                type_ids=("A", "B"),
                cell_ids=("left",),
                sample_endpoint_cell=lambda _t, _i: "other",
                familywise_error=Fraction(1, 20),
                max_rounds=2,
            )


if __name__ == "__main__":
    unittest.main()

