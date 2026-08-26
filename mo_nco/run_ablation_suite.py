from __future__ import annotations

import argparse
from pathlib import Path

from .ablation_suite import run_ips_ablation_suite
from .benchmark_suite import BenchmarkSuite
from .run_benchmark import parse_csv_ints


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IPS ablations across a benchmark suite.")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite_public_motsp.json"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablation_suite"))
    parser.add_argument("--log-period", type=int, default=250)
    args = parser.parse_args()

    suite = BenchmarkSuite.from_json(args.suite)
    run_ips_ablation_suite(
        suite=suite,
        seeds=parse_csv_ints(args.seeds),
        output_dir=args.output_dir,
        default_population=args.population,
        default_evaluations=args.evaluations,
        log_period=args.log_period,
    )
    print(f"Wrote cross-instance ablation results to {args.output_dir}")


if __name__ == "__main__":
    main()
