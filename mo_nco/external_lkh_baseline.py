from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .types import ObjectiveVector, Tour


def solve(input_path: Path, output_path: Path) -> None:
    try:
        import elkai
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("elkai is required for the LKH-derived baseline.") from exc

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    matrices = payload["distance_matrices"]
    if int(payload["num_objectives"]) != 2:
        raise RuntimeError("The LKH-derived scalar baseline currently supports bi-objective TSP only.")
    evaluations = int(payload["evaluations"])
    population = int(payload["population_size"])
    seed = int(payload["seed"])
    rng = random.Random(seed)
    num_weights = max(2, min(population, evaluations))
    weights = [(idx / max(1, num_weights - 1), 1.0 - idx / max(1, num_weights - 1)) for idx in range(num_weights)]
    rng.shuffle(weights)

    entries: List[ArchiveEntry] = []
    diagnostics: List[Tuple[int, Tuple[ArchiveEntry, ...]]] = []
    scale0 = _mean_positive(matrices[0])
    scale1 = _mean_positive(matrices[1])
    elkai_runs = max(1, int(os.environ.get("MO_NCO_LKH_RUNS", "1")))
    for idx, (w0, w1) in enumerate(weights, start=1):
        scalar_matrix = _scalar_matrix(matrices[0], matrices[1], w0, w1, scale0, scale1)
        tour = tuple(elkai.solve_int_matrix(scalar_matrix, runs=elkai_runs, skip_end=True))
        tour = _rotate_to_zero(tour)
        objectives = _evaluate(matrices, tour)
        entries.append(ArchiveEntry(tour, objectives))
        archive = ParetoArchive(max_size=None)
        archive.update(entries)
        step = min(evaluations, max(1, round(idx * evaluations / num_weights)))
        diagnostics.append((step, tuple(archive.entries)))

    archive = ParetoArchive(max_size=None)
    archive.update(entries)
    _write_output(output_path, archive.entries, evaluations)
    _write_diagnostics(output_path.with_suffix(".diagnostics.csv"), diagnostics)


def _scalar_matrix(
    matrix0: Sequence[Sequence[float]],
    matrix1: Sequence[Sequence[float]],
    w0: float,
    w1: float,
    scale0: float,
    scale1: float,
) -> List[List[int]]:
    n = len(matrix0)
    result: List[List[int]] = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                value = w0 * float(matrix0[i][j]) / scale0 + w1 * float(matrix1[i][j]) / scale1
                row.append(max(1, int(round(value * 1_000_000))))
        result.append(row)
    return result


def _mean_positive(matrix: Sequence[Sequence[float]]) -> float:
    values = [float(value) for row in matrix for value in row if float(value) > 0.0]
    return sum(values) / len(values) if values else 1.0


def _rotate_to_zero(tour: Tour) -> Tour:
    if 0 not in tour:
        raise RuntimeError("elkai returned a tour without city 0.")
    idx = tour.index(0)
    return tuple(tour[idx:] + tour[:idx])


def _evaluate(matrices: Sequence[Sequence[Sequence[float]]], tour: Tour) -> ObjectiveVector:
    values = []
    for matrix in matrices:
        total = 0.0
        for idx, city in enumerate(tour):
            total += float(matrix[city][tour[(idx + 1) % len(tour)]])
        values.append(total)
    return tuple(values)


def _write_output(path: Path, entries: Sequence[ArchiveEntry], evaluations: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tour", "evaluations", "objective_0", "objective_1"])
        for entry in entries:
            writer.writerow([" ".join(str(city) for city in entry.tour), evaluations, *entry.objectives])


def _write_diagnostics(path: Path, diagnostics: Sequence[Tuple[int, Tuple[ArchiveEntry, ...]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluations", "tour", "objective_0", "objective_1"])
        for step, entries in diagnostics:
            for entry in entries:
                writer.writerow([step, " ".join(str(city) for city in entry.tour), *entry.objectives])


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mo_nco.external_lkh_baseline input.json output.csv")
    solve(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
