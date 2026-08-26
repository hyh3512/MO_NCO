from __future__ import annotations

"""Freeze a prelaunch algorithm-configuration matrix without running search."""

import argparse
import hashlib
import json
from pathlib import Path

from .benchmark_suite import (
    BenchmarkSuite,
    build_algorithm_configuration_manifest,
)
from .run_benchmark import parse_csv_ints, parse_csv_strings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the complete case x algorithm x seed configuration "
            "matrix before any objective evaluation."
        )
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--algorithms",
        default=(
            "pareto-smc-pilot-confirm-v12,ips-theory-certified,"
            "pymoo-nsga2,pymoo-moead,motsp-pls"
        ),
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--evaluations", type=int, required=True)
    parser.add_argument("--log-period", type=int, required=True)
    parser.add_argument("--archive-update-period", type=int, required=True)
    parser.add_argument(
        "--anytime-checkpoint-period",
        type=int,
        required=True,
    )
    parser.add_argument("--output-archive-limit", type=int, required=True)
    parser.add_argument(
        "--override-case-evaluations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--certified-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_suite = args.suite.read_bytes()
    suite = BenchmarkSuite.from_json(args.suite)
    manifest = build_algorithm_configuration_manifest(
        suite=suite,
        suite_sha256=hashlib.sha256(raw_suite).hexdigest(),
        algorithms=parse_csv_strings(args.algorithms),
        seeds=parse_csv_ints(args.seeds),
        default_population=args.population,
        default_evaluations=args.evaluations,
        log_period=args.log_period,
        archive_update_period=args.archive_update_period,
        override_case_evaluations=args.override_case_evaluations,
        output_archive_limit=args.output_archive_limit,
        certified_traces=args.certified_traces,
        anytime_checkpoint_period=args.anytime_checkpoint_period,
    )
    encoded = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"ROWS {len(manifest['runs'])}")
    print(f"SHA256 {hashlib.sha256(encoded).hexdigest()}")
    print(f"OUTPUT {args.output.resolve()}")


if __name__ == "__main__":
    main()
