from __future__ import annotations

"""Build a mechanical certificate for the strict single-site MH control.

The report deliberately separates two evidence levels:

1. metadata contract checks, which only inspect self-reported runtime fields;
2. chained transition-trace replay, which independently recomputes every
   recorded 2-opt proposal, energy difference, and accept/reject decision.

Neither level proves mixing, performance, an LDP, or floating-point equivalence
with an ideal real-arithmetic Markov kernel.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .contracts import ClaimLevel, EvidenceLevel
from .benchmark_suite import BenchmarkSuite
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .kernel_trace import verify_certified_trace


def _resolve_trace_path(metadata_dir: Path, raw_path: object) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = (metadata_dir / path, metadata_dir.parent / path, Path.cwd() / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def build_kernel_certificate(
    suite_output: Path,
    method: str,
    output: Path,
    db_tolerance: float = 1e-12,
    *,
    require_trace: bool = False,
    suite_manifest: Path | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> bool:
    source_instances: Dict[str, MultiObjectiveTSPInstance] = {}
    suite_manifest_sha256 = ""
    if suite_manifest is not None:
        raw_suite = suite_manifest.read_bytes()
        suite_manifest_sha256 = hashlib.sha256(raw_suite).hexdigest()
        suite = BenchmarkSuite.from_json(suite_manifest)
        for case in suite.cases:
            source_instances[case.name] = case.load_instance() or MultiObjectiveTSPInstance.random_biobjective(
                case.cities,
                seed=case.instance_seed,
            )

    records: List[Tuple[str, int, Path, Dict[str, object]]] = []
    for metadata_path in sorted(suite_output.rglob("run_metadata.jsonl")):
        case = metadata_path.parent.name
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if str(payload.get("algorithm", "")) != method:
                continue
            records.append(
                (
                    case,
                    int(payload.get("seed", -1)),
                    metadata_path.parent,
                    dict(payload.get("metadata", {})),
                )
            )

    observed_pairs = [(case, seed) for case, seed, _, _ in records]
    observed_pair_set = set(observed_pairs)
    duplicate_pairs = sorted(
        pair for pair in observed_pair_set if observed_pairs.count(pair) > 1
    )
    completeness_requested = suite_manifest is not None or expected_seeds is not None
    expected_pairs: set[tuple[str, int]] = set()
    missing_pairs: list[tuple[str, int]] = []
    extra_pairs: list[tuple[str, int]] = []
    if expected_seeds is not None:
        expected_case_names = (
            set(source_instances)
            if suite_manifest is not None
            else {case for case, _ in observed_pair_set}
        )
        expected_pairs = {
            (case, int(seed))
            for case in expected_case_names
            for seed in expected_seeds
        }
        missing_pairs = sorted(expected_pairs - observed_pair_set)
        extra_pairs = sorted(observed_pair_set - expected_pairs)
        completeness_passed = not missing_pairs and not extra_pairs and not duplicate_pairs
    elif suite_manifest is not None:
        completeness_passed = False
    else:
        completeness_passed = True

    rows = []
    overall = bool(records) and completeness_passed
    for case, seed, metadata_dir, metadata in records:
        source_instance = source_instances.get(case)
        trace_path = _resolve_trace_path(metadata_dir, metadata.get("trace_path"))
        trace_result = (
            verify_certified_trace(
                trace_path,
                instance=source_instance,
                expected_context_hash=str(metadata.get("context_hash", "")),
                expected_final_chain_hash=str(metadata.get("trace_chain_hash", "")),
                expected_records=int(metadata.get("trace_records", -1)),
                expected_transition_attempts=int(metadata.get("transition_attempts", -1)),
                expected_proposal_evaluations=int(metadata.get("proposal_evaluations", -1)),
                expected_seed=seed,
                expected_num_particles=int(metadata.get("initial_population_evaluations", -1)),
                expected_instance_sha256=str(metadata.get("instance_sha256", "")),
            )
            if trace_path is not None and trace_path.exists()
            else None
        )
        checks = {
            "contract": metadata.get("algorithm_contract") == "theory_certified_single_site_v4",
            "implementation_version": metadata.get("implementation_version") == "0.8.0",
            "claim_level": metadata.get("claim_level") == ClaimLevel.CERTIFIED_MH.value,
            "frozen_context": metadata.get("context_frozen") is True
            and int(metadata.get("context_refresh_count", -1)) == 0
            and metadata.get("bounds_frozen") is True,
            "single_site": metadata.get("single_coordinate_transition") is True,
            "symmetric_proposal": metadata.get("proposal") == "uniform_symmetric_two_opt"
            and metadata.get("proposal_symmetric") is True
            and abs(float(metadata.get("proposal_log_ratio", float("inf")))) <= db_tolerance,
            "positive_temperature": metadata.get("temperature_schedule") == "constant"
            and metadata.get("positive_temperature") is True
            and float(metadata.get("temperature_min", 0.0)) > 0.0,
            "no_hidden_feedback": metadata.get("archive_feedback") is False
            and metadata.get("archive_role") == "reporting_only_no_kernel_feedback"
            and metadata.get("mean_field_enabled") is False
            and metadata.get("neural_enabled") is False,
            "no_nonreversible_shortcuts": metadata.get("compiled_polish_enabled") is False
            and metadata.get("crossover_enabled") is False
            and metadata.get("local_refinement_enabled") is False,
            "log_domain_acceptance": metadata.get("acceptance_computation") == "log_uniform_comparison",
            "rng_replay_contract": metadata.get("rng_contract")
            == "python_random_mt19937_seed_replay_v1"
            and int(metadata.get("seed", -1)) == seed,
            "objective_state_function": metadata.get("objective_evaluation_contract")
            == "full_tour_state_function"
            and len(str(metadata.get("instance_sha256", ""))) == 64,
            "source_binding": suite_manifest is None
            or (
                source_instance is not None
                and metadata.get("instance_sha256") == instance_sha256(source_instance)
            ),
            "explicit_laziness": metadata.get("explicit_laziness") is True
            and metadata.get("aperiodicity_mechanism") == "explicit_identity_mixture"
            and metadata.get("evaluation_clock_kernel") == "explicit_lazy_identity_mixture"
            and 0.0 < float(metadata.get("lazy_probability", 0.0)) < 1.0,
            "uniformization_disclaimed": metadata.get("uniformization_role")
            == "declaration_only_not_executed",
            "db_numeric_identity": float(metadata.get("db_max_abs_log_residual", float("inf"))) <= db_tolerance,
            "budget_accounting": int(metadata.get("evaluations_used", -1))
            == int(metadata.get("evaluation_budget", -2))
            and int(metadata.get("initial_population_evaluations", -1))
            + int(metadata.get("transition_evaluations", -1))
            == int(metadata.get("evaluations_used", -3))
            and int(metadata.get("proposal_evaluations", -1))
            + int(metadata.get("identity_evaluations", -1))
            == int(metadata.get("transition_evaluations", -3))
            and int(metadata.get("accepted_single_site_moves", -1))
            + int(metadata.get("rejected_single_site_moves", -1))
            == int(metadata.get("proposal_evaluations", -3))
            and int(metadata.get("lazy_self_loops", -1))
            == int(metadata.get("identity_evaluations", -2))
            and int(metadata.get("transition_attempts", -1))
            == int(metadata.get("transition_evaluations", -2)),
            "trace_replay": bool(trace_result and trace_result.passed),
        }
        metadata_passed = all(value for key, value in checks.items() if key != "trace_replay")
        passed = metadata_passed and (checks["trace_replay"] if require_trace else True)
        overall = overall and passed
        rows.append((case, seed, passed, checks, metadata, trace_path, trace_result))

    requested_evidence_level = (
        EvidenceLevel.SOURCE_REPLAYED
        if require_trace and suite_manifest is not None
        else EvidenceLevel.INTERNAL_TRACE_REPLAY
        if require_trace
        else EvidenceLevel.SELF_REPORTED_METADATA
    )
    strongest_passed_evidence_level = (
        requested_evidence_level
        if overall
        else EvidenceLevel.NO_SUITE_WIDE_LEVEL_PASSED
    )
    lines = [
        "# Strict Single-Site Kernel Certificate",
        "",
        f"Method: `{method}`",
        f"Metadata records: {len(records)}",
        f"DB identity tolerance: `{db_tolerance:.3g}`",
        f"Transition trace required: `{require_trace}`",
        f"Suite source manifest: `{suite_manifest if suite_manifest is not None else 'NOT PROVIDED'}`",
        f"Suite source manifest SHA-256: `{suite_manifest_sha256 or 'NOT PROVIDED'}`",
        (
            "Expected seeds: `"
            + (
                ",".join(str(seed) for seed in expected_seeds)
                if expected_seeds is not None
                else "NOT PROVIDED"
            )
            + "`"
        ),
        (
            f"Suite completeness: `{'PASS' if completeness_passed else 'FAIL'}`"
            if completeness_requested
            else "Suite completeness: `NOT REQUESTED`"
        ),
        f"Requested evidence mode: `{requested_evidence_level.value}`",
        f"Strongest evidence level: `{strongest_passed_evidence_level.value}`",
        "",
        "This report is a mechanical implementation-contract audit. Metadata checks",
        "are self-reported. A trace PASS additionally replays the chained runtime",
        "records, but still does not prove mixing, performance, Sanov/LDP, analytic",
        "assumptions, or ideal-real/floating-point equivalence.",
        "",
        "| case | seed | contract | version | claim | frozen | state function | source binding | budget | RNG | single-site | symmetric Q | T>0 | lazy | no feedback | no shortcuts | log-domain | uniformization disclaimer | DB identity | trace replay | verdict |",
        "|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if not completeness_passed:
        if suite_manifest is not None and expected_seeds is None:
            lines.append(
                "Completeness error: a suite manifest requires explicit `--expected-seeds`."
            )
        if missing_pairs:
            lines.append(f"Missing case/seed pairs: `{missing_pairs[:20]}`")
        if extra_pairs:
            lines.append(f"Unexpected case/seed pairs: `{extra_pairs[:20]}`")
        if duplicate_pairs:
            lines.append(f"Duplicate case/seed pairs: `{duplicate_pairs[:20]}`")
        lines.append("")
    for case, seed, passed, checks, _, trace_path, trace_result in rows:
        mark = lambda key: "PASS" if checks[key] else "FAIL"
        source_mark = mark("source_binding") if suite_manifest is not None else "NOT REQUESTED"
        trace_mark = mark("trace_replay") if trace_path is not None else "NOT PROVIDED"
        lines.append(
            f"| {case} | {seed} | {mark('contract')} | {mark('implementation_version')} | "
            f"{mark('claim_level')} | {mark('frozen_context')} | "
            f"{mark('objective_state_function')} | {source_mark} | {mark('budget_accounting')} | "
            f"{mark('rng_replay_contract')} | "
            f"{mark('single_site')} | {mark('symmetric_proposal')} | {mark('positive_temperature')} | "
            f"{mark('explicit_laziness')} | "
            f"{mark('no_hidden_feedback')} | {mark('no_nonreversible_shortcuts')} | "
            f"{mark('log_domain_acceptance')} | {mark('uniformization_disclaimed')} | "
            f"{mark('db_numeric_identity')} | {trace_mark} | {'PASS' if passed else 'FAIL'} |"
        )
        if trace_result is not None and not trace_result.passed:
            lines.append(
                f"\nTrace errors for `{case}` seed `{seed}`: "
                + "; ".join(trace_result.errors[:8])
            )
    lines.extend(["", f"Overall mechanical certificate: **{'PASS' if overall else 'FAIL'}**", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the strict single-site MH runtime contract.")
    parser.add_argument("--suite-output", type=Path, required=True)
    parser.add_argument("--method", default="ips-theory-certified")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db-tolerance", type=float, default=1e-12)
    parser.add_argument("--require-trace", action="store_true")
    parser.add_argument(
        "--suite-manifest",
        type=Path,
        help="Exact benchmark suite JSON used to reconstruct and bind source instances.",
    )
    parser.add_argument(
        "--expected-seeds",
        help="Comma-separated exact seed set required for every suite case.",
    )
    args = parser.parse_args()
    expected_seeds = (
        tuple(int(item.strip()) for item in args.expected_seeds.split(",") if item.strip())
        if args.expected_seeds
        else None
    )
    passed = build_kernel_certificate(
        suite_output=args.suite_output,
        method=args.method,
        output=args.output,
        db_tolerance=args.db_tolerance,
        require_trace=args.require_trace,
        suite_manifest=args.suite_manifest,
        expected_seeds=expected_seeds,
    )
    print(f"Wrote kernel certificate: {args.output}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
