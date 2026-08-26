from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .external_official_lkh_baseline import (
    _evaluate,
    _mean_positive,
    _read_tour,
    _scalar_matrix,
    _write_diagnostics,
    _write_full_matrix_tsplib,
    _write_output,
    _write_parameter_file,
)
from .moves import sample_two_opt_indices, two_opt_at
from .types import ObjectiveVector, Tour


def solve(input_path: Path, output_path: Path) -> None:
    lkh_path = Path(os.environ.get("MO_NCO_LKH_EXECUTABLE", "")).expanduser()
    if not lkh_path.exists():
        raise RuntimeError(
            "Set MO_NCO_LKH_EXECUTABLE to the official LKH-3 executable before running lkh-2ppls."
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if int(payload["num_objectives"]) != 2:
        raise RuntimeError("lkh-2ppls currently supports bi-objective TSP only.")

    matrices = payload["distance_matrices"]
    evaluations = int(payload["evaluations"])
    population = int(payload["population_size"])
    seed = int(payload["seed"])
    rng = random.Random(seed)
    n = int(payload["num_cities"])

    lkh_weights = max(
        2,
        min(
            population,
            evaluations,
            int(os.environ.get("MO_NCO_2PPLS_LKH_WEIGHTS", str(min(32, population)))),
        ),
    )
    weights = [(idx / max(1, lkh_weights - 1), 1.0 - idx / max(1, lkh_weights - 1)) for idx in range(lkh_weights)]
    rng.shuffle(weights)
    scale0 = _mean_positive(matrices[0])
    scale1 = _mean_positive(matrices[1])
    runs = max(1, int(os.environ.get("MO_NCO_OFFICIAL_LKH_RUNS", "1")))
    max_trials = max(1, int(os.environ.get("MO_NCO_OFFICIAL_LKH_MAX_TRIALS", "1000")))
    timeout = float(os.environ.get("MO_NCO_OFFICIAL_LKH_TIMEOUT", "120"))
    neighbor_sample = max(
        1,
        int(os.environ.get("MO_NCO_2PPLS_NEIGHBOR_SAMPLE", str(max(32, min(4 * population, 256))))),
    )
    log_stride = max(1, int(os.environ.get("MO_NCO_2PPLS_LOG_STRIDE", str(max(1, evaluations // 16)))))
    symmetric = _all_symmetric(matrices)

    entries: List[ArchiveEntry] = []
    diagnostics: List[Tuple[int, Tuple[ArchiveEntry, ...]]] = []
    archive = ParetoArchive(max_size=None)
    used = 0
    last_log = 0
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        for idx, (w0, w1) in enumerate(weights, start=1):
            if used >= evaluations:
                break
            tsp_path = work_dir / f"phase1_{idx}.tsp"
            par_path = work_dir / f"phase1_{idx}.par"
            tour_path = work_dir / f"phase1_{idx}.tour"
            scalar_matrix = _scalar_matrix(matrices[0], matrices[1], w0, w1, scale0, scale1)
            _write_full_matrix_tsplib(tsp_path, scalar_matrix, f"{payload['name']}_2ppls_{idx}")
            _write_parameter_file(
                par_path,
                problem_file=tsp_path.name,
                tour_file=tour_path.name,
                runs=runs,
                max_trials=max_trials,
                seed=seed + 1009 * idx,
            )
            subprocess.run(
                [str(lkh_path), par_path.name],
                cwd=work_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            tour = _read_tour(tour_path, n)
            entry = ArchiveEntry(tour, _evaluate(matrices, tour))
            entries.append(entry)
            archive.update([entry])
            used += 1
            if used - last_log >= log_stride or used >= evaluations:
                diagnostics.append((used, tuple(archive.entries)))
                last_log = used

    visited = {entry.tour for entry in entries}
    weight_cycle = [(idx / max(1, population - 1), 1.0 - idx / max(1, population - 1)) for idx in range(max(2, population))]
    step = 0
    while used < evaluations and archive.entries:
        parent = _select_parent(archive.entries, weight_cycle[step % len(weight_cycle)], rng)
        candidate_pairs = _candidate_pairs(n, neighbor_sample, rng)
        new_entries: List[ArchiveEntry] = []
        ideal, nadir = _ideal_nadir(archive.entries)
        weight = weight_cycle[step % len(weight_cycle)]
        best_scalar = _scalar(parent.objectives, weight, ideal, nadir)
        best_entry = parent
        for i, j in candidate_pairs:
            if used >= evaluations:
                break
            child = two_opt_at(parent.tour, i, j)
            if child in visited:
                continue
            visited.add(child)
            child_obj = _evaluate_two_opt(matrices, parent.tour, parent.objectives, i, j, symmetric)
            used += 1
            child_entry = ArchiveEntry(child, child_obj)
            new_entries.append(child_entry)
            score = _scalar(child_obj, weight, ideal, nadir)
            if score < best_scalar:
                best_scalar = score
                best_entry = child_entry
            if used - last_log >= log_stride or used >= evaluations:
                probe = ParetoArchive(max_size=None)
                probe.update([*archive.entries, *new_entries])
                diagnostics.append((used, tuple(probe.entries)))
                last_log = used
        if new_entries:
            archive.update(new_entries)
        if best_entry is parent and not new_entries:
            step += 1
            if time.perf_counter() - start_time > float(os.environ.get("MO_NCO_2PPLS_TIMEOUT", "3600")):
                break
            continue
        step += 1

    if not diagnostics:
        diagnostics.append((used, tuple(archive.entries)))
    _write_output(output_path, archive.entries, used)
    _write_diagnostics(output_path.with_suffix(".diagnostics.csv"), diagnostics)


def _candidate_pairs(num_cities: int, count: int, rng: random.Random) -> List[Tuple[int, int]]:
    if num_cities <= 90 and count >= (num_cities - 1) * (num_cities - 2) // 2:
        pairs = [(i, j) for i in range(1, num_cities) for j in range(i + 1, num_cities)]
        rng.shuffle(pairs)
        return pairs
    return [sample_two_opt_indices(num_cities, rng) for _ in range(count)]


def _select_parent(entries: Sequence[ArchiveEntry], weight: ObjectiveVector, rng: random.Random) -> ArchiveEntry:
    if len(entries) <= 1:
        return entries[0]
    ideal, nadir = _ideal_nadir(entries)
    elite = min(entries, key=lambda entry: _scalar(entry.objectives, weight, ideal, nadir))
    if rng.random() < 0.85:
        return elite
    return rng.choice(tuple(entries))


def _ideal_nadir(entries: Sequence[ArchiveEntry]) -> Tuple[ObjectiveVector, ObjectiveVector]:
    dim = len(entries[0].objectives)
    ideal = tuple(min(entry.objectives[idx] for entry in entries) for idx in range(dim))
    nadir = tuple(max(entry.objectives[idx] for entry in entries) for idx in range(dim))
    return ideal, nadir


def _scalar(objective: ObjectiveVector, weight: ObjectiveVector, ideal: ObjectiveVector, nadir: ObjectiveVector) -> float:
    normalized = [
        (value - lo) / max(1e-9, hi - lo)
        for value, lo, hi in zip(objective, ideal, nadir)
    ]
    terms = [max(1e-3, w) * value for value, w in zip(normalized, weight)]
    return max(terms) + 0.03 * sum(terms)


def _evaluate_two_opt(
    matrices: Sequence[Sequence[Sequence[float]]],
    tour: Tour,
    current_objectives: ObjectiveVector,
    i: int,
    j: int,
    symmetric: bool,
) -> ObjectiveVector:
    if i > j:
        i, j = j, i
    if symmetric:
        a = tour[i - 1]
        b = tour[i]
        c = tour[j]
        d = tour[(j + 1) % len(tour)]
        values = []
        for current, matrix in zip(current_objectives, matrices):
            removed = float(matrix[a][b]) + float(matrix[c][d])
            added = float(matrix[a][c]) + float(matrix[b][d])
            values.append(current - removed + added)
        return tuple(values)
    return _evaluate(matrices, two_opt_at(tour, i, j))


def _all_symmetric(matrices: Sequence[Sequence[Sequence[float]]]) -> bool:
    for matrix in matrices:
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(float(matrix[i][j]) - float(matrix[j][i])) > 1e-9:
                    return False
    return True


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mo_nco.external_lkh_2ppls_baseline input.json output.csv")
    solve(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
