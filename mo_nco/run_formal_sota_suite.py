from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

from .benchmark_suite import BenchmarkSuite
from .budget_sweep import run_budget_sweep
from .instance import MultiObjectiveTSPInstance
from .mature_baselines import ExternalBaselineOptimizer, configured_external_solver, load_external_baseline_from_env
from .run_benchmark import parse_csv_ints, parse_csv_strings
from .strict_sota_audit import build_report


MATURE_EXTERNALS: Dict[str, tuple[str, str]] = {
    "paquete": ("MO_NCO_BASELINE_PAQUETE", "MO_NCO_BRIDGE_PAQUETE"),
    "tpls-external": ("MO_NCO_BASELINE_TPLS", "MO_NCO_BRIDGE_TPLS"),
    "mogls-external": ("MO_NCO_BASELINE_MOGLS", "MO_NCO_BRIDGE_MOGLS"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict 35+ case SOTA MOTSP suite.")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/suite_public_motsp_35.json"))
    parser.add_argument(
        "--algorithms",
        default="ips-neural-quality-relocate,ips-quality-relocate,lkh-scalar,pymoo-nsga2,pymoo-moead,paquete,tpls-external,mogls-external",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--budgets", default="512,1024")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/formal_sota_suite"))
    parser.add_argument("--log-period", type=int, default=256)
    parser.add_argument("--archive-update-period", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-missing-external", action="store_true")
    parser.add_argument(
        "--external-smoke-test",
        action="store_true",
        help="Run a tiny protocol-level solve through each configured external solver before the full suite.",
    )
    args = parser.parse_args()

    algorithms = parse_csv_strings(args.algorithms)
    preflight = preflight_external_solvers(algorithms, smoke_test=args.external_smoke_test)
    missing = [line for line in preflight if line.startswith("FAIL")]
    if preflight:
        print("\n".join(preflight))
    if missing and not args.allow_missing_external:
        raise SystemExit(
            "Missing real external MOTSP solver commands. Set the required MO_NCO_BASELINE_* direct commands "
            "or MO_NCO_BRIDGE_* wrapper templates "
            "or pass --allow-missing-external for a non-SOTA development run."
        )
    if args.preflight_only:
        return

    runnable_algorithms = algorithms
    if missing and args.allow_missing_external:
        missing_names = {name for name in MATURE_EXTERNALS if any(f" {name}:" in line for line in missing)}
        runnable_algorithms = [name for name in algorithms if name not in missing_names]

    suite = BenchmarkSuite.from_json(args.suite)
    rows = run_budget_sweep(
        suite=suite,
        algorithms=runnable_algorithms,
        seeds=parse_csv_ints(args.seeds),
        budgets=parse_csv_ints(args.budgets),
        output_dir=args.output_dir,
        default_population=args.population,
        log_period=args.log_period,
        archive_update_period=args.archive_update_period,
    )
    baselines = [
        name
        for name in [
            "lkh-scalar",
            "lkh-official",
            "lkh-2ppls",
            "paquete-published-tpls",
            "pymoo-nsga2",
            "pymoo-moead",
            *MATURE_EXTERNALS,
        ]
        if name in algorithms
    ]
    report = build_report(
        rows=[{key: str(value) for key, value in row.items()} for row in rows],
        method="ips-neural-quality-relocate",
        baselines=baselines,
        min_cases=35,
        min_seeds=10,
    )
    (args.output_dir / "strict_sota_audit.md").write_text(report, encoding="utf-8")
    print(f"Wrote formal suite outputs to {args.output_dir}")


def preflight_external_solvers(algorithms: Sequence[str], smoke_test: bool = False) -> List[str]:
    messages: List[str] = []
    for algorithm, env_names in MATURE_EXTERNALS.items():
        if algorithm not in algorithms:
            continue
        configured = configured_external_solver(algorithm)
        direct_env, bridge_env = env_names
        if configured is None:
            messages.append(f"FAIL {algorithm}: neither {direct_env} nor {bridge_env} is set.")
            continue
        mode, env_name, command = configured
        executable = _first_command_token(command)
        resolved = shutil.which(executable) if not Path(executable).exists() else executable
        if not resolved:
            messages.append(f"FAIL {algorithm}: executable not found for {env_name}={command!r}.")
            continue
        if mode == "direct":
            probe = subprocess.run([*_split_command(command), "--help"], capture_output=True, text=True, timeout=10)
            if probe.returncode not in {0, 1, 2}:
                messages.append(f"WARN {algorithm}: direct command exists but --help returned {probe.returncode}.")
            else:
                messages.append(f"PASS {algorithm}: direct {env_name}={command}")
        else:
            messages.append(f"PASS {algorithm}: bridge {env_name}={command}")
        if smoke_test:
            messages.extend(_smoke_test_external(algorithm))
    return messages


def _smoke_test_external(algorithm: str) -> List[str]:
    try:
        instance = MultiObjectiveTSPInstance.random_biobjective(8, seed=100 + len(algorithm))
        config = load_external_baseline_from_env(algorithm)
        result = ExternalBaselineOptimizer(
            instance=instance,
            config=config,
            population_size=4,
            evaluations=16,
            seed=0,
            archive_max_size=20,
        ).run()
    except Exception as exc:  # pragma: no cover - depends on external binaries.
        return [f"FAIL {algorithm}: protocol smoke test failed: {exc}"]
    if not result.archive.entries:
        return [f"FAIL {algorithm}: protocol smoke test produced an empty archive."]
    return [f"PASS {algorithm}: protocol smoke test produced {len(result.archive.entries)} nondominated rows."]


def _first_command_token(command: str) -> str:
    parts = _split_command(command)
    return parts[0] if parts else ""


def _split_command(command: str) -> List[str]:
    parts = shlex.split(command, posix=False)
    return [_strip_balanced_quotes(part) for part in parts]


def _strip_balanced_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    main()
