from __future__ import annotations

import unittest

from mo_nco.external_paquete_published_tpls_baseline import (
    _parse_entries,
    _parse_kro_case,
    _published_tpls_path,
)


class MatureReferenceTests(unittest.TestCase):
    def test_paquete_kro_case_parser_accepts_public_suite_names(self) -> None:
        self.assertEqual(_parse_kro_case("public_kroA100_kroB100"), ("A", "B", 100))
        self.assertEqual(_parse_kro_case("public_kroB100_kroA100"), ("B", "A", 100))
        self.assertIsNone(_parse_kro_case("paquete_euclidAB100"))

    def test_paquete_tpls_catalog_uses_strongest_ab_archives(self) -> None:
        self.assertEqual(
            _published_tpls_path("public_kroA100_kroB100"),
            "TPLS/KROAB100/points.100.AB.a2000.3.first.ils.tgz",
        )
        self.assertEqual(
            _published_tpls_path("public_kroA150_kroB150"),
            "TPLS/KROAB150/points.150.AB.i200.3.first.ils.tgz",
        )

    def test_paquete_tour_parser_rotates_and_reevaluates(self) -> None:
        rows = ["3 7 3 7 tour: 1 2 0 3 1"]
        matrices = (
            (
                (0.0, 1.0, 5.0, 9.0),
                (1.0, 0.0, 2.0, 8.0),
                (5.0, 2.0, 0.0, 3.0),
                (9.0, 8.0, 3.0, 0.0),
            ),
            (
                (0.0, 7.0, 4.0, 1.0),
                (7.0, 0.0, 6.0, 2.0),
                (4.0, 6.0, 0.0, 5.0),
                (1.0, 2.0, 5.0, 0.0),
            ),
        )
        entries = _parse_entries(rows, matrices, 4)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].tour, (0, 3, 1, 2))
        self.assertEqual(entries[0].objectives, (24.0, 13.0))


if __name__ == "__main__":
    unittest.main()

