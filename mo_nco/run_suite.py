from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .benchmark_suite import (
    BenchmarkSuite,
    build_algorithm_configuration_manifest,
    load_and_verify_algorithm_configuration_manifest,
    load_metric_reference_manifest,
    run_benchmark_suite,
)
from .run_benchmark import parse_csv_ints, parse_csv_strings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-instance MO-NCO benchmark suite.")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite_demo.json"))
    parser.add_argument(
        "--algorithms",
        default=(
            "ips-theory-certified,ips-heuristic-adaptive,"
            "pymoo-nsga2,pymoo-moead,moead,nsga2,random2opt"
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/suite"))
    parser.add_argument("--log-period", type=int, default=200)
    parser.add_argument("--archive-update-period", type=int, default=50)
    parser.add_argument(
        "--anytime-checkpoint-period",
        type=int,
        default=None,
        help=(
            "Frozen common evaluation spacing for exact anytime archive "
            "snapshots. Required by the formal matched protocol."
        ),
    )
    parser.add_argument(
        "--override-case-evaluations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use --evaluations for every case instead of per-case suite values.",
    )
    parser.add_argument(
        "--execution-order",
        choices=("algorithm-major", "seed-major-balanced-v1"),
        default="algorithm-major",
        help="Run order contract. Use seed-major-balanced-v1 for counterbalanced formal timing.",
    )
    parser.add_argument(
        "--metric-reference-manifest",
        type=Path,
        default=None,
        help="Frozen external per-case HV/IGD reference manifest.",
    )
    parser.add_argument(
        "--certified-traces",
        action="store_true",
        help="Emit chained transition traces for strict certified-MH methods.",
    )
    parser.add_argument(
        "--measure-python-memory",
        action="store_true",
        help=(
            "Measure Python allocator peak increments in a separate exact-state "
            "replay. This excludes native and accelerator memory."
        ),
    )
    parser.add_argument(
        "--output-archive-limit",
        type=int,
        default=None,
        help="Apply one deterministic nondominated output cap to every arm.",
    )
    parser.add_argument(
        "--information-contract",
        type=Path,
        default=None,
        help="Frozen JSON contract for information available to every arm.",
    )
    parser.add_argument(
        "--algorithm-configuration-manifest",
        type=Path,
        default=None,
        help=(
            "Prelaunch case x algorithm x seed configuration manifest. "
            "It is verified before any optimizer starts."
        ),
    )
    parser.add_argument(
        "--budget-scope",
        choices=(
            "single_run_objective_evaluations",
            "matched_total_objective_evaluations_including_pilot_confirm",
        ),
        default="single_run_objective_evaluations",
    )
    args = parser.parse_args()

    suite = BenchmarkSuite.from_json(args.suite)
    algorithms = parse_csv_strings(args.algorithms)
    seeds = parse_csv_ints(args.seeds)
    suite_manifest_sha256 = hashlib.sha256(
        args.suite.read_bytes()
    ).hexdigest()
    metric_references = None
    metric_reference_manifest_sha256 = ""
    if args.metric_reference_manifest is not None:
        metric_references, metric_reference_manifest_sha256 = load_metric_reference_manifest(
            args.metric_reference_manifest
        )
    information_contract_sha256 = ""
    if args.information_contract is not None:
        information_contract_sha256 = hashlib.sha256(
            args.information_contract.read_bytes()
        ).hexdigest()
    if (
        args.budget_scope
        == "matched_total_objective_evaluations_including_pilot_confirm"
    ):
        missing = []
        if args.metric_reference_manifest is None:
            missing.append("--metric-reference-manifest")
        if args.information_contract is None:
            missing.append("--information-contract")
        if args.output_archive_limit is None:
            missing.append("--output-archive-limit")
        if args.algorithm_configuration_manifest is None:
            missing.append("--algorithm-configuration-manifest")
        if args.anytime_checkpoint_period is None:
            missing.append("--anytime-checkpoint-period")
        if not args.measure_python_memory:
            missing.append("--measure-python-memory")
        if args.execution_order != "seed-major-balanced-v1":
            missing.append(
                "--execution-order seed-major-balanced-v1"
            )
        if missing:
            raise ValueError(
                "Formal matched budget scope requires: "
                + ", ".join(missing)
            )
    expected_configurations = None
    if args.algorithm_configuration_manifest is not None:
        expected_manifest = build_algorithm_configuration_manifest(
            suite=suite,
            suite_sha256=suite_manifest_sha256,
            algorithms=algorithms,
            seeds=seeds,
            default_population=args.population,
            default_evaluations=args.evaluations,
            log_period=args.log_period,
            archive_update_period=args.archive_update_period,
            override_case_evaluations=args.override_case_evaluations,
            output_archive_limit=args.output_archive_limit,
            certified_traces=args.certified_traces,
            anytime_checkpoint_period=(
                args.anytime_checkpoint_period
            ),
        )
        expected_configurations, _ = (
            load_and_verify_algorithm_configuration_manifest(
                args.algorithm_configuration_manifest,
                expected=expected_manifest,
            )
        )
    run_benchmark_suite(
        suite=suite,
        algorithms=algorithms,
        seeds=seeds,
        output_dir=args.output_dir,
        default_population=args.population,
        default_evaluations=args.evaluations,
        log_period=args.log_period,
        archive_update_period=args.archive_update_period,
        override_case_evaluations=args.override_case_evaluations,
        execution_order=args.execution_order,
        metric_references=metric_references,
        metric_reference_manifest_sha256=metric_reference_manifest_sha256,
        certified_traces=args.certified_traces,
        measure_python_memory=args.measure_python_memory,
        output_archive_limit=args.output_archive_limit,
        suite_manifest_sha256=suite_manifest_sha256,
        information_contract_sha256=information_contract_sha256,
        budget_scope=args.budget_scope,
        expected_algorithm_configurations=expected_configurations,
        anytime_checkpoint_period=args.anytime_checkpoint_period,
    )
    print(f"Wrote suite results to {args.output_dir}")


if __name__ == "__main__":
    main()
