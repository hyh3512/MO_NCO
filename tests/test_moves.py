from __future__ import annotations

import random
import unittest

from mo_nco.moves import random_tour, two_opt_at


class MoveTests(unittest.TestCase):
    def test_random_tour_valid(self) -> None:
        rng = random.Random(1)
        tour = random_tour(8, rng)
        self.assertEqual(tour[0], 0)
        self.assertEqual(sorted(tour), list(range(8)))

    def test_two_opt_is_involution_for_same_indices(self) -> None:
        tour = (0, 1, 2, 3, 4, 5)
        moved = two_opt_at(tour, 2, 4)
        restored = two_opt_at(moved, 2, 4)
        self.assertEqual(restored, tour)


if __name__ == "__main__":
    unittest.main()

