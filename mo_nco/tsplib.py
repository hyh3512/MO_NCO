from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .instance import DistanceMatrix, MultiObjectiveTSPInstance, Point


@dataclass(frozen=True)
class TSPLIBProblem:
    name: str
    dimension: int
    edge_weight_type: str
    edge_weight_format: str
    distance_matrix: DistanceMatrix


def load_multiobjective_tsplib(paths: Sequence[str | Path]) -> MultiObjectiveTSPInstance:
    if len(paths) < 1:
        raise ValueError("At least one TSPLIB file is required.")
    problems = [parse_tsplib(path) for path in paths]
    dimension = problems[0].dimension
    for problem in problems:
        if problem.dimension != dimension:
            raise ValueError("All TSPLIB objective files must have the same DIMENSION.")
    name = "bitsp_" + "_".join(problem.name for problem in problems)
    return MultiObjectiveTSPInstance.from_distance_matrices(
        [problem.distance_matrix for problem in problems],
        name=name,
    )


def load_bitsp(path: str | Path) -> MultiObjectiveTSPInstance:
    """Load a simple bi-objective coordinate CSV.

    Supported rows are either

        id,x1,y1,x2,y2

    or just

        x1,y1,x2,y2

    Header rows are allowed. Distances are Euclidean for each objective.
    """
    path = Path(path)
    coords_a: List[Point] = []
    coords_b: List[Point] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        for row in reader:
            parts = [item.strip() for item in row if item.strip()]
            if not parts:
                continue
            try:
                values = [float(item) for item in parts]
            except ValueError:
                continue
            if len(values) == 5:
                _, x1, y1, x2, y2 = values
            elif len(values) == 4:
                x1, y1, x2, y2 = values
            else:
                raise ValueError(f"Unsupported BITSP row in {path}: {row}")
            coords_a.append((x1, y1))
            coords_b.append((x2, y2))

    if len(coords_a) < 3:
        raise ValueError(f"BITSP file {path} must contain at least three cities.")
    return MultiObjectiveTSPInstance((tuple(coords_a), tuple(coords_b)), name=path.stem)


def parse_tsplib(path: str | Path) -> TSPLIBProblem:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: Dict[str, str] = {}
    section = None
    coord_rows: List[str] = []
    weight_rows: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "EOF":
            break
        if upper == "NODE_COORD_SECTION":
            section = "coords"
            continue
        if upper == "EDGE_WEIGHT_SECTION":
            section = "weights"
            continue
        if upper.endswith("_SECTION"):
            section = "skip"
            continue

        if section == "coords":
            coord_rows.append(line)
        elif section == "weights":
            weight_rows.append(line)
        elif section is None:
            key, value = _split_header(line)
            headers[key.upper()] = value

    name = headers.get("NAME", path.stem)
    dimension = int(headers.get("DIMENSION", "0"))
    if dimension <= 0:
        raise ValueError(f"TSPLIB file {path} is missing a valid DIMENSION.")
    edge_weight_type = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    edge_weight_format = headers.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()

    if edge_weight_type == "EXPLICIT":
        matrix = _parse_explicit_matrix(weight_rows, dimension, edge_weight_format)
    else:
        coords = _parse_coords(coord_rows, dimension)
        matrix = _matrix_from_coords(coords, edge_weight_type)

    return TSPLIBProblem(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        distance_matrix=matrix,
    )


def _split_header(line: str) -> Tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip(), value.strip()
    parts = line.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _parse_coords(rows: Sequence[str], dimension: int) -> Tuple[Point, ...]:
    coords_by_id: Dict[int, Point] = {}
    for row in rows:
        parts = row.split()
        if len(parts) < 3:
            continue
        idx = int(float(parts[0]))
        coords_by_id[idx] = (float(parts[1]), float(parts[2]))
    if len(coords_by_id) != dimension:
        raise ValueError("NODE_COORD_SECTION length does not match DIMENSION.")
    return tuple(coords_by_id[idx] for idx in sorted(coords_by_id))


def _matrix_from_coords(coords: Sequence[Point], edge_weight_type: str) -> DistanceMatrix:
    matrix: List[Tuple[float, ...]] = []
    for a in coords:
        row = []
        for b in coords:
            row.append(float(_distance(a, b, edge_weight_type)))
        matrix.append(tuple(row))
    return tuple(matrix)


def _distance(a: Point, b: Point, edge_weight_type: str) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    value = math.hypot(dx, dy)
    if edge_weight_type in {"EUC_2D", "EUC_3D"}:
        return int(value + 0.5)
    if edge_weight_type == "CEIL_2D":
        return math.ceil(value)
    if edge_weight_type == "ATT":
        rij = math.sqrt((dx * dx + dy * dy) / 10.0)
        tij = int(rij + 0.5)
        return tij if tij >= rij else tij + 1
    if edge_weight_type == "EXACT_2D":
        return value
    if edge_weight_type == "GEO":
        raise ValueError("GEO TSPLIB coordinates are not implemented yet.")
    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")


def _parse_explicit_matrix(rows: Sequence[str], dimension: int, edge_weight_format: str) -> DistanceMatrix:
    values: List[float] = []
    for row in rows:
        values.extend(float(item) for item in row.split())

    fmt = edge_weight_format.upper()
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]

    if fmt == "FULL_MATRIX":
        expected = dimension * dimension
        if len(values) != expected:
            raise ValueError(f"FULL_MATRIX expected {expected} weights, got {len(values)}.")
        cursor = 0
        for i in range(dimension):
            for j in range(dimension):
                matrix[i][j] = values[cursor]
                cursor += 1
    elif fmt == "UPPER_ROW":
        expected = dimension * (dimension - 1) // 2
        if len(values) != expected:
            raise ValueError(f"UPPER_ROW expected {expected} weights, got {len(values)}.")
        cursor = 0
        for i in range(dimension):
            for j in range(i + 1, dimension):
                matrix[i][j] = matrix[j][i] = values[cursor]
                cursor += 1
    elif fmt == "LOWER_ROW":
        expected = dimension * (dimension - 1) // 2
        if len(values) != expected:
            raise ValueError(f"LOWER_ROW expected {expected} weights, got {len(values)}.")
        cursor = 0
        for i in range(1, dimension):
            for j in range(i):
                matrix[i][j] = matrix[j][i] = values[cursor]
                cursor += 1
    elif fmt == "UPPER_DIAG_ROW":
        expected = dimension * (dimension + 1) // 2
        if len(values) != expected:
            raise ValueError(f"UPPER_DIAG_ROW expected {expected} weights, got {len(values)}.")
        cursor = 0
        for i in range(dimension):
            for j in range(i, dimension):
                matrix[i][j] = matrix[j][i] = values[cursor]
                cursor += 1
    elif fmt == "LOWER_DIAG_ROW":
        expected = dimension * (dimension + 1) // 2
        if len(values) != expected:
            raise ValueError(f"LOWER_DIAG_ROW expected {expected} weights, got {len(values)}.")
        cursor = 0
        for i in range(dimension):
            for j in range(i + 1):
                matrix[i][j] = matrix[j][i] = values[cursor]
                cursor += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    return tuple(tuple(row) for row in matrix)
