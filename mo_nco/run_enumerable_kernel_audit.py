from __future__ import annotations

"""CLI for the finite-state strict typed-MH reference audit."""

import argparse
import json
from pathlib import Path

from .enumerable_kernel_audit import (
    audit_typed_mh_temperature_grid,
    write_enumerable_kernel_audit,
)
from .instance import MultiObjectiveTSPInstance


def _positive_temperatures(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperatures must be comma-separated numbers") from exc
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("temperatures must be strictly positive")
    return values


def _load_matrix_instance(path: Path) -> MultiObjectiveTSPInstance:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        matrices = payload.get("distance_matrices", payload.get("matrices"))
        name = str(payload.get("name", path.stem))
    else:
        matrices = payload
        name = path.stem
    if matrices is None:
        raise ValueError(
            "Matrix JSON must be a nested list or contain 'distance_matrices'."
        )
    return MultiObjectiveTSPInstance.from_distance_matrices(matrices, name=name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate a tiny fixed-zero MOTSP state space and audit the strict "
            "positive-temperature typed-MH reference kernel."
        )
    )
    parser.add_argument("--distance-matrices-json", type=Path)
    parser.add_argument("--num-cities", type=int, default=5)
    parser.add_argument("--instance-seed", type=int, default=7101)
    parser.add_argument("--context-seed", type=int, default=7102)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--evaluation-budget", type=int, default=512)
    parser.add_argument("--lazy-probability", type=float, default=0.05)
    parser.add_argument("--chebyshev-rho", type=float, default=0.03)
    parser.add_argument("--minimum-scale-fraction", type=float, default=1e-3)
    parser.add_argument("--absolute-scale-floor", type=float, default=1e-12)
    parser.add_argument("--max-states", type=int, default=720)
    parser.add_argument(
        "--max-product-states",
        type=int,
        default=4096,
        help="Directly enumerate the full product matrix only below this cap.",
    )
    parser.add_argument(
        "--temperatures",
        type=_positive_temperatures,
        default=_positive_temperatures("0.005,0.01,0.02,0.05,0.1,0.2,0.5"),
    )
    parser.add_argument(
        "--max-relative-stationary-excess",
        type=float,
        default=0.05,
    )
    parser.add_argument("--tv-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "enumerable_kernel_audit",
    )
    parser.add_argument(
        "--fail-on-h1-falsified",
        action="store_true",
        help="Return exit code 2 unless the finite-grid H1 gate is not falsified.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance = (
        _load_matrix_instance(args.distance_matrices_json)
        if args.distance_matrices_json is not None
        else MultiObjectiveTSPInstance.random_biobjective(
            args.num_cities,
            seed=args.instance_seed,
        )
    )
    report = audit_typed_mh_temperature_grid(
        instance,
        num_particles=args.num_particles,
        context_seed=args.context_seed,
        temperatures=args.temperatures,
        evaluation_budget=args.evaluation_budget,
        lazy_probability=args.lazy_probability,
        chebyshev_rho=args.chebyshev_rho,
        minimum_scale_fraction=args.minimum_scale_fraction,
        absolute_scale_floor=args.absolute_scale_floor,
        max_states=args.max_states,
        max_product_states=args.max_product_states,
        max_relative_stationary_excess=args.max_relative_stationary_excess,
        tv_tolerance=args.tv_tolerance,
    )
    json_path, csv_path = write_enumerable_kernel_audit(report, args.output_dir)
    print(f"Verdict: {report['h1_grid_verdict']}")
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    for row in report["temperature_rows"]:
        print(
            "T={temperature:g} distortion={distortion:.6g} "
            "SLEM(eval)={slem:.6g} required_eval={required} feasible={feasible}".format(
                temperature=float(row["temperature"]),
                distortion=float(row["max_relative_stationary_excess"]),
                slem=float(row["product_slem_evaluation_clock"]),
                required=row["required_evaluations_tv_bound"],
                feasible=bool(row["h1_feasible_on_grid"]),
            )
        )
    if args.fail_on_h1_falsified and report["h1_grid_verdict"] != "NOT_FALSIFIED_ON_GRID":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
