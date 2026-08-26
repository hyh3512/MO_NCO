from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_suite import BenchmarkSuite
from .budget_sweep import run_budget_sweep
from .run_benchmark import parse_csv_ints, parse_csv_strings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-budget benchmark sweep.")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite_public_motsp_35.json"))
    parser.add_argument(
        "--algorithms",
        default=(
            "ips-theory-certified,ips-heuristic-adaptive,"
            "pymoo-nsga2,pymoo-moead,motsp-pls"
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--budgets", default="512,1024,2048")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/budget_sweep"))
    parser.add_argument("--log-period", type=int, default=256)
    parser.add_argument("--archive-update-period", type=int, default=64)
    args = parser.parse_args()

    suite = BenchmarkSuite.from_json(args.suite)
    run_budget_sweep(
        suite=suite,
        algorithms=parse_csv_strings(args.algorithms),
        seeds=parse_csv_ints(args.seeds),
        budgets=parse_csv_ints(args.budgets),
        output_dir=args.output_dir,
        default_population=args.population,
        log_period=args.log_period,
        archive_update_period=args.archive_update_period,
    )
    print(f"Wrote budget sweep results to {args.output_dir}")


if __name__ == "__main__":
    main()
