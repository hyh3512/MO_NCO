from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence


ENV_BY_SOLVER = {
    "paquete": "MO_NCO_BRIDGE_PAQUETE",
    "tpls": "MO_NCO_BRIDGE_TPLS",
    "mogls": "MO_NCO_BRIDGE_MOGLS",
}


def solve(solver: str, input_path: Path, output_path: Path) -> None:
    solver_key = solver.lower()
    if solver_key not in ENV_BY_SOLVER:
        raise RuntimeError(f"Unknown MOTSP bridge solver: {solver}.")
    template = os.environ.get(ENV_BY_SOLVER[solver_key], "").strip()
    if not template:
        raise RuntimeError(
            f"Set {ENV_BY_SOLVER[solver_key]} to a real solver wrapper command template. "
            "The command must write the requested output CSV."
        )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    work_dir = output_path.parent / f"{solver_key}_bridge_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    matrices = payload["distance_matrices"]
    if int(payload["num_objectives"]) != 2:
        raise RuntimeError("The generic MOTSP bridge currently supports bi-objective TSP only.")
    obj0_csv = work_dir / "objective_0.csv"
    obj1_csv = work_dir / "objective_1.csv"
    obj0_tsp = work_dir / "objective_0_full_matrix.tsp"
    obj1_tsp = work_dir / "objective_1_full_matrix.tsp"
    _write_matrix_csv(obj0_csv, matrices[0])
    _write_matrix_csv(obj1_csv, matrices[1])
    _write_full_matrix_tsplib(obj0_tsp, matrices[0], f"{payload['name']}_objective_0")
    _write_full_matrix_tsplib(obj1_tsp, matrices[1], f"{payload['name']}_objective_1")

    diagnostics_path = output_path.with_suffix(".diagnostics.csv")
    replacements: Dict[str, str] = {
        "input_json": str(input_path),
        "output_csv": str(output_path),
        "diagnostics_csv": str(diagnostics_path),
        "matrix_dir": str(work_dir),
        "obj0_csv": str(obj0_csv),
        "obj1_csv": str(obj1_csv),
        "obj0_tsp": str(obj0_tsp),
        "obj1_tsp": str(obj1_tsp),
        "name": str(payload["name"]),
        "n": str(payload["num_cities"]),
        "seed": str(payload["seed"]),
        "evaluations": str(payload["evaluations"]),
        "population": str(payload["population_size"]),
    }
    command = [_strip_balanced_quotes(part).format(**replacements) for part in shlex.split(template, posix=False)]
    subprocess.run(command, check=True, cwd=work_dir)
    _validate_output(output_path, int(payload["num_objectives"]))


def _write_matrix_csv(path: Path, matrix: Sequence[Sequence[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def _write_full_matrix_tsplib(path: Path, matrix: Sequence[Sequence[float]], name: str) -> None:
    scale = 1000.0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write("TYPE: TSP\n")
        handle.write(f"DIMENSION: {len(matrix)}\n")
        handle.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        handle.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            handle.write(" ".join(str(int(round(float(value) * scale))) for value in row) + "\n")
        handle.write("EOF\n")


def _validate_output(path: Path, num_objectives: int) -> None:
    if not path.exists():
        raise RuntimeError(f"External solver wrapper did not create {path}.")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tour", "evaluations", *(f"objective_{idx}" for idx in range(num_objectives))}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"External solver output is missing columns: {sorted(missing)}")
        if not any(True for _ in reader):
            raise RuntimeError("External solver output contains no solution rows.")


def _strip_balanced_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python -m mo_nco.external_motsp_bridge <paquete|tpls|mogls> input.json output.csv")
    solve(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
