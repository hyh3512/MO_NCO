from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .types import ObjectiveVector, Tour


def solve(input_path: Path, output_path: Path) -> None:
    lkh_path = Path(os.environ.get("MO_NCO_LKH_EXECUTABLE", "")).expanduser()
    if not lkh_path.exists():
        raise RuntimeError(
            "Set MO_NCO_LKH_EXECUTABLE to the official LKH-3 executable, "
            "for example D:\\MO_NCO\\external\\LKH-3.0.14\\LKH-3.exe."
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    matrices = payload["distance_matrices"]
    if int(payload["num_objectives"]) != 2:
        raise RuntimeError("The official LKH scalar baseline currently supports bi-objective TSP only.")
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
    runs = max(1, int(os.environ.get("MO_NCO_OFFICIAL_LKH_RUNS", "1")))
    max_trials = max(1, int(os.environ.get("MO_NCO_OFFICIAL_LKH_MAX_TRIALS", "1000")))
    timeout = float(os.environ.get("MO_NCO_OFFICIAL_LKH_TIMEOUT", "120"))

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for idx, (w0, w1) in enumerate(weights, start=1):
            tsp_path = work_dir / f"scalar_{idx}.tsp"
            par_path = work_dir / f"scalar_{idx}.par"
            tour_path = work_dir / f"scalar_{idx}.tour"
            scalar_matrix = _scalar_matrix(matrices[0], matrices[1], w0, w1, scale0, scale1)
            _write_full_matrix_tsplib(tsp_path, scalar_matrix, f"{payload['name']}_scalar_{idx}")
            _write_parameter_file(
                par_path,
                problem_file=tsp_path.name,
                tour_file=tour_path.name,
                runs=runs,
                max_trials=max_trials,
                seed=seed + idx,
            )
            subprocess.run(
                [str(lkh_path), par_path.name],
                cwd=work_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            tour = _read_tour(tour_path, int(payload["num_cities"]))
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


def _write_full_matrix_tsplib(path: Path, matrix: Sequence[Sequence[int]], name: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write("TYPE: TSP\n")
        handle.write(f"DIMENSION: {len(matrix)}\n")
        handle.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        handle.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            handle.write(" ".join(str(int(value)) for value in row) + "\n")
        handle.write("EOF\n")


def _write_parameter_file(
    path: Path,
    problem_file: str,
    tour_file: str,
    runs: int,
    max_trials: int,
    seed: int,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"PROBLEM_FILE = {problem_file}",
                f"TOUR_FILE = {tour_file}",
                f"RUNS = {runs}",
                f"MAX_TRIALS = {max_trials}",
                f"SEED = {seed}",
                "TRACE_LEVEL = 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _read_tour(path: Path, num_cities: int) -> Tour:
    if not path.exists():
        raise RuntimeError(f"LKH did not create tour file: {path}")
    in_section = False
    values: List[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "TOUR_SECTION":
            in_section = True
            continue
        if not in_section:
            continue
        if line in {"-1", "EOF"}:
            break
        values.append(int(line) - 1)
    if len(values) != num_cities or sorted(values) != list(range(num_cities)):
        raise RuntimeError(f"Invalid LKH tour in {path}: {values[:10]}...")
    return _rotate_to_zero(tuple(values))


def _scalar_matrix(
    matrix0: Sequence[Sequence[float]],
    matrix1: Sequence[Sequence[float]],
    w0: float,
    w1: float,
    scale0: float,
    scale1: float,
) -> List[List[int]]:
    result: List[List[int]] = []
    for i, row0 in enumerate(matrix0):
        row = []
        for j, value0 in enumerate(row0):
            if i == j:
                row.append(0)
            else:
                value = w0 * float(value0) / scale0 + w1 * float(matrix1[i][j]) / scale1
                row.append(max(1, int(round(value * 1_000_000))))
        result.append(row)
    return result


def _mean_positive(matrix: Sequence[Sequence[float]]) -> float:
    values = [float(value) for row in matrix for value in row if float(value) > 0.0]
    return sum(values) / len(values) if values else 1.0


def _rotate_to_zero(tour: Tour) -> Tour:
    if 0 not in tour:
        raise RuntimeError("LKH returned a tour without city 0.")
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
        raise SystemExit("Usage: python -m mo_nco.external_official_lkh_baseline input.json output.csv")
    solve(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
