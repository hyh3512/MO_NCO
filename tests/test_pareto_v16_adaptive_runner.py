from __future__ import annotations

import unittest

from mo_nco.pareto_adaptive_replica_experiment import (
    run_cell_separated_successive_elimination,
)


class ParetoV16AdaptiveRunnerTests(unittest.TestCase):
    def test_mechanical_successive_elimination_resolves_two_cells(self) -> None:
        numerators = {
            ("A", "left"): (4, 5),
            ("B", "left"): (1, 5),
            ("A", "right"): (1, 5),
            ("B", "right"): (4, 5),
        }

        def sample(type_id: str, cell_id: str, index: int) -> bool:
            numerator, denominator = numerators[(type_id, cell_id)]
            # A deterministic frequency stream is enough for mechanical tests;
            # it is not presented as an independent random stream certificate.
            return index % denominator < numerator

        result = run_cell_separated_successive_elimination(
            type_ids=("A", "B"),
            cell_ids=("left", "right"),
            sample=sample,
            familywise_identification_error="1/20",
            familywise_cp_error="1/20",
            max_rounds=20_000,
        )
        selected = {item.cell_id: item.selected_type for item in result.cell_results}
        self.assertEqual(selected, {"left": "A", "right": "B"})
        self.assertGreater(result.total_replica_evaluations, 0)


if __name__ == "__main__":
    unittest.main()

