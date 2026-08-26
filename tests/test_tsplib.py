from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mo_nco.instance import MultiObjectiveTSPInstance
from mo_nco.tsplib import load_bitsp, load_multiobjective_tsplib, parse_tsplib


DATA_DIR = Path(__file__).resolve().parent / "data"


class TSPLIBTests(unittest.TestCase):
    def test_parse_euc_2d_tsplib(self) -> None:
        problem = parse_tsplib(DATA_DIR / "tsplib" / "demo_obj1.tsp")
        self.assertEqual(problem.dimension, 20)
        self.assertEqual(problem.distance_matrix[0][0], 0.0)
        self.assertGreater(problem.distance_matrix[0][1], 0.0)

    def test_load_multiobjective_tsplib(self) -> None:
        instance = load_multiobjective_tsplib(
            [
                DATA_DIR / "tsplib" / "demo_obj1.tsp",
                DATA_DIR / "tsplib" / "demo_obj2.tsp",
            ]
        )
        self.assertEqual(instance.num_cities, 20)
        self.assertEqual(instance.num_objectives, 2)
        self.assertEqual(len(instance.evaluate(tuple(range(20)))), 2)

    def test_load_bitsp_csv(self) -> None:
        instance = load_bitsp(DATA_DIR / "bitsp" / "demo_bitsp.csv")
        self.assertEqual(instance.num_cities, 20)
        self.assertEqual(instance.num_objectives, 2)

    def test_explicit_full_matrix(self) -> None:
        content = """NAME: explicit_demo
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 1 2
1 0 3
2 3 0
EOF
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "explicit.tsp"
            path.write_text(content, encoding="utf-8")
            problem = parse_tsplib(path)
        self.assertEqual(problem.distance_matrix[0][2], 2.0)


if __name__ == "__main__":
    unittest.main()

