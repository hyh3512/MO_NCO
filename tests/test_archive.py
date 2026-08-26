from __future__ import annotations

import unittest

from mo_nco.archive import ArchiveEntry, ParetoArchive, dominates


class ArchiveTests(unittest.TestCase):
    def test_dominance(self) -> None:
        self.assertTrue(dominates((1.0, 2.0), (2.0, 3.0)))
        self.assertFalse(dominates((1.0, 4.0), (2.0, 3.0)))

    def test_archive_keeps_nondominated_entries(self) -> None:
        archive = ParetoArchive()
        archive.update(
            [
                ArchiveEntry((0, 1, 2, 3), (3.0, 3.0)),
                ArchiveEntry((0, 2, 1, 3), (2.0, 4.0)),
                ArchiveEntry((0, 3, 1, 2), (3.0, 3.0)),
                ArchiveEntry((0, 3, 2, 1), (1.0, 1.0)),
            ]
        )
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.entries[0].objectives, (1.0, 1.0))

    def test_hypervolume_2d(self) -> None:
        archive = ParetoArchive()
        archive.update(
            [
                ArchiveEntry((0, 1, 2, 3), (1.0, 3.0)),
                ArchiveEntry((0, 2, 1, 3), (2.0, 2.0)),
                ArchiveEntry((0, 3, 2, 1), (3.0, 1.0)),
            ]
        )
        hv = archive.hypervolume_2d(reference=(4.0, 4.0))
        self.assertAlmostEqual(hv, 6.0)


if __name__ == "__main__":
    unittest.main()

