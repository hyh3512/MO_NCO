from __future__ import annotations

import csv
import json
import re
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence

from .archive import ArchiveEntry, ParetoArchive
from .types import ObjectiveVector, Tour


PAQUETE_BASE = "https://eden.dei.uc.pt/~paquete/tsp/"


def solve(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if int(payload["num_objectives"]) != 2:
        raise RuntimeError("Paquete published TPLS archives are bi-objective TSP results.")
    result_path = _published_tpls_path(payload["name"])
    cache_path = _download(result_path)
    member_rows = _read_seed_member(cache_path, int(payload["seed"]))
    entries = _parse_entries(member_rows, payload["distance_matrices"], int(payload["num_cities"]))
    if not entries:
        raise RuntimeError(f"Paquete published TPLS archive {result_path} yielded no valid tours.")
    archive = ParetoArchive(max_size=None)
    archive.update(entries)
    _write_output(output_path, archive.entries, int(payload["evaluations"]))


def _published_tpls_path(case_name: str) -> str:
    parsed = _parse_kro_case(case_name)
    if parsed is None:
        raise RuntimeError(
            f"{case_name!r} is not one of the Paquete KRO TPLS cases. "
            "Supported examples: public_kroA100_kroB100, public_kroA150_kroB150, public_kroA200_kroB200."
        )
    left, right, size = parsed
    pair = "".join(sorted((left, right)))
    directory = f"TPLS/KRO{pair}{size}"
    if pair == "AB" and size == 100:
        filename = "points.100.AB.a2000.3.first.ils.tgz"
    elif pair == "AB" and size == 150:
        filename = "points.150.AB.i200.3.first.ils.tgz"
    elif pair == "AB" and size == 200:
        filename = "points.200.AB.i300.3.first.ils.tgz"
    elif size == 100 and pair in {"AC", "AD", "AE", "BC", "BD", "BE", "CD", "CE", "DE"}:
        filename = f"points.100.{pair}.3.first.ils.tgz"
    else:
        raise RuntimeError(f"Paquete TPLS published tour archives do not cover KRO pair {pair}{size}.")
    return f"{directory}/{filename}"


def _parse_kro_case(case_name: str) -> tuple[str, str, int] | None:
    matches = re.findall(r"kro([A-E])(\d+)", case_name, flags=re.IGNORECASE)
    if len(matches) < 2:
        return None
    left, left_size = matches[0]
    right, right_size = matches[1]
    if left_size != right_size:
        return None
    return left.upper(), right.upper(), int(left_size)


def _download(relative_path: str) -> Path:
    cache_dir = Path("benchmarks") / "paquete_published_tpls"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / relative_path.replace("/", "__")
    if cache_path.exists():
        return cache_path
    url = PAQUETE_BASE + relative_path
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _read_seed_member(path: Path, seed: int) -> List[str]:
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        members.sort(key=lambda member: _natural_key(member.name))
        if not members:
            return []
        member = members[seed % len(members)]
        handle = archive.extractfile(member)
        if handle is None:
            return []
        return handle.read().decode("utf-8", errors="replace").splitlines()


def _natural_key(value: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _parse_entries(
    rows: Iterable[str],
    matrices: Sequence[Sequence[Sequence[float]]],
    num_cities: int,
) -> List[ArchiveEntry]:
    entries: List[ArchiveEntry] = []
    for row in rows:
        if "tour:" not in row:
            continue
        _, tour_text = row.split("tour:", 1)
        tour_values = [int(value) for value in tour_text.split()]
        if len(tour_values) == num_cities + 1 and tour_values[0] == tour_values[-1]:
            tour_values = tour_values[:-1]
        if len(tour_values) != num_cities or sorted(tour_values) != list(range(num_cities)):
            continue
        tour = _rotate_to_zero(tuple(tour_values))
        entries.append(ArchiveEntry(tour, _evaluate(matrices, tour)))
    return entries


def _rotate_to_zero(tour: Tour) -> Tour:
    idx = tour.index(0)
    return tuple(tour[idx:] + tour[:idx])


def _evaluate(matrices: Sequence[Sequence[Sequence[float]]], tour: Tour) -> ObjectiveVector:
    objectives = []
    for matrix in matrices:
        total = 0.0
        for idx, city in enumerate(tour):
            total += float(matrix[city][tour[(idx + 1) % len(tour)]])
        objectives.append(total)
    return tuple(objectives)


def _write_output(path: Path, entries: Sequence[ArchiveEntry], evaluations: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tour", "evaluations", "objective_0", "objective_1"])
        for entry in entries:
            writer.writerow([" ".join(str(city) for city in entry.tour), evaluations, *entry.objectives])


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mo_nco.external_paquete_published_tpls_baseline input.json output.csv")
    solve(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
