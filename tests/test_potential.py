from __future__ import annotations

import unittest

from mo_nco.archive import ArchiveEntry, ParetoArchive
from mo_nco.potential import HypervolumeArchivePotential, ScalarArchivePotential


class PotentialTests(unittest.TestCase):
    def test_delta_replace_matches_recomputed_energy(self) -> None:
        objectives = [(3.0, 5.0), (4.0, 4.0), (5.0, 3.0)]
        archive = ParetoArchive()
        archive.update(
            [
                ArchiveEntry((0, 1, 2), (3.0, 5.0)),
                ArchiveEntry((0, 2, 1), (5.0, 3.0)),
            ]
        )
        potential = ScalarArchivePotential(diversity_weight=0.1)
        context = potential.build_context(archive, objectives)
        delta = potential.delta_replace(objectives, 1, (3.5, 3.5), context)
        updated = list(objectives)
        updated[1] = (3.5, 3.5)
        expected = potential.empirical_energy(updated, context) - potential.empirical_energy(objectives, context)
        self.assertAlmostEqual(delta, expected)

    def test_hypervolume_delta_matches_recomputed_energy(self) -> None:
        objectives = [(3.0, 5.0), (4.0, 4.0), (5.0, 3.0)]
        archive = ParetoArchive()
        archive.update(
            [
                ArchiveEntry((0, 1, 2), (3.0, 5.0)),
                ArchiveEntry((0, 2, 1), (5.0, 3.0)),
            ]
        )
        potential = HypervolumeArchivePotential()
        context = potential.build_context(archive, objectives)
        delta = potential.delta_replace(objectives, 1, (3.5, 3.5), context)
        updated = list(objectives)
        updated[1] = (3.5, 3.5)
        expected = potential.empirical_energy(updated, context) - potential.empirical_energy(objectives, context)
        self.assertAlmostEqual(delta, expected)


if __name__ == "__main__":
    unittest.main()

