from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple


ARM_CONTRACTS: Dict[str, str] = {
    "ips-theory-heavy-no-prior": "theory_search_v2:none",
    "ips-theory-endpoint-only": "theory_search_v2:scalar",
    "ips-theory-move-only": "theory_search_v2:move",
    "ips-neural-mv-jitgreedy-targetflow-theory-optimized": "theory_search_v2:full",
}


def load_suite_output(
    suite_output: Path,
) -> Tuple[List[Dict[str, str]], List[Dict[str, object]]]:
    aggregate = suite_output / "aggregate_runs.csv"
    if not aggregate.is_file():
        raise FileNotFoundError(f"Missing aggregate_runs.csv: {aggregate}")
    with aggregate.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    metadata_rows: List[Dict[str, object]] = []
    for path in sorted(suite_output.glob("*/run_metadata.jsonl")):
        case = path.parent.name
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
            payload["case"] = case
            metadata_rows.append(payload)
    if not metadata_rows:
        raise FileNotFoundError(f"No per-case run_metadata.jsonl files under {suite_output}")
    return rows, metadata_rows


def _check(name: str, passed: bool, detail: str) -> Dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _run_key(row: Dict[str, object]) -> Tuple[str, str, int]:
    return str(row["case"]), str(row["algorithm"]), int(row["seed"])


def validate_v2_contract(
    rows: Sequence[Dict[str, object]],
    metadata_rows: Sequence[Dict[str, object]],
    *,
    expected_cases: int,
    expected_seeds: Sequence[int],
    expected_evaluations: int,
    require_execution_order: str = "",
    require_metric_reference: str = "",
) -> Dict[str, object]:
    checks: List[Dict[str, object]] = []
    algorithms = tuple(ARM_CONTRACTS)
    seeds = tuple(int(seed) for seed in expected_seeds)
    cases = sorted({str(row.get("case", "")) for row in rows})
    expected_keys = {
        (case, algorithm, seed)
        for case in cases
        for algorithm in algorithms
        for seed in seeds
    }
    row_keys = [_run_key(row) for row in rows]
    metadata_keys = [_run_key(row) for row in metadata_rows]
    checks.append(
        _check(
            "case_count",
            len(cases) == expected_cases,
            f"observed={len(cases)} expected={expected_cases}",
        )
    )
    checks.append(
        _check(
            "aggregate_cartesian_product",
            len(row_keys) == len(expected_keys)
            and len(set(row_keys)) == len(row_keys)
            and set(row_keys) == expected_keys,
            f"rows={len(row_keys)} unique={len(set(row_keys))} expected={len(expected_keys)}",
        )
    )
    checks.append(
        _check(
            "metadata_cartesian_product",
            len(metadata_keys) == len(expected_keys)
            and len(set(metadata_keys)) == len(metadata_keys)
            and set(metadata_keys) == expected_keys,
            (
                f"metadata_rows={len(metadata_keys)} "
                f"unique={len(set(metadata_keys))} expected={len(expected_keys)}"
            ),
        )
    )
    checks.append(
        _check(
            "forced_evaluation_budget",
            all(int(row.get("evaluations", -1)) == expected_evaluations for row in rows)
            and all(
                int(row.get("evaluations", -1)) == expected_evaluations
                and int(dict(row.get("metadata", {})).get("evaluation_budget", -1))
                == expected_evaluations
                and int(dict(row.get("metadata", {})).get("evaluations_used", -1))
                == expected_evaluations
                for row in metadata_rows
            ),
            f"expected_evaluations={expected_evaluations}",
        )
    )
    contract_ok = True
    for row in metadata_rows:
        metadata = dict(row.get("metadata", {}))
        if metadata.get("ablation_contract") != ARM_CONTRACTS.get(str(row.get("algorithm"))):
            contract_ok = False
            break
    checks.append(_check("v2_arm_contracts", contract_ok, "expected theory_search_v2:none|scalar|move|full"))
    checks.append(
        _check(
            "prior_rng_isolation_enabled",
            all(bool(dict(row.get("metadata", {})).get("prior_loading_rng_isolated")) for row in metadata_rows),
            "prior_loading_rng_isolated must be true for all four arms",
        )
    )
    by_case_seed: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in metadata_rows:
        by_case_seed.setdefault((str(row["case"]), int(row["seed"])), []).append(row)
    initial_hash_ok = all(
        len(group) == len(algorithms)
        and len(
            {
                str(dict(row.get("metadata", {})).get("initial_population_sha256", ""))
                for row in group
            }
        )
        == 1
        and all(
            str(dict(row.get("metadata", {})).get("initial_population_sha256", ""))
            for row in group
        )
        for group in by_case_seed.values()
    )
    base_rng_hash_ok = all(
        len(group) == len(algorithms)
        and len(
            {
                str(
                    dict(row.get("metadata", {})).get(
                        "base_rng_state_after_initialization_sha256",
                        "",
                    )
                )
                for row in group
            }
        )
        == 1
        and all(
            str(
                dict(row.get("metadata", {})).get(
                    "base_rng_state_after_initialization_sha256",
                    "",
                )
            )
            for row in group
        )
        for group in by_case_seed.values()
    )
    checks.append(
        _check(
            "matched_initial_population_hash",
            initial_hash_ok,
            f"groups={len(by_case_seed)}",
        )
    )
    checks.append(
        _check(
            "matched_base_rng_post_initialization_hash",
            base_rng_hash_ok,
            f"groups={len(by_case_seed)}",
        )
    )
    local_bound_ok = all(
        len(
            {
                int(dict(row.get("metadata", {})).get("local_move_check_upper_bound", -1))
                for row in group
            }
        )
        == 1
        for group in by_case_seed.values()
    )
    checks.append(
        _check(
            "matched_local_search_upper_bound",
            local_bound_ok,
            "local_move_check_upper_bound must match within each case x seed",
        )
    )
    checks.append(
        _check(
            "no_accelerator_fallbacks",
            all(not dict(row.get("metadata", {})).get("accelerator_fallbacks") for row in metadata_rows),
            "all accelerator_fallbacks must be empty",
        )
    )
    checks.append(
        _check(
            "context_accounting_complete",
            all(
                bool(dict(row.get("metadata", {})).get("context_jump_accounting_complete"))
                for row in metadata_rows
            ),
            "context_jump_accounting_complete must be true",
        )
    )
    prior_shape_ok = True
    scalar_hashes: set[str] = set()
    full_scalar_hashes: set[str] = set()
    move_hashes: set[str] = set()
    full_move_hashes: set[str] = set()
    for row in metadata_rows:
        metadata = dict(row.get("metadata", {}))
        mode = ARM_CONTRACTS[str(row["algorithm"])].rsplit(":", 1)[1]
        scalar_hash = str(metadata.get("neural_prior_sha256", ""))
        move_hash = str(metadata.get("learned_move_prior_sha256", ""))
        expected_scalar = mode in {"scalar", "full"}
        expected_move = mode in {"move", "full"}
        if bool(scalar_hash) != expected_scalar or bool(move_hash) != expected_move:
            prior_shape_ok = False
        if mode == "scalar":
            scalar_hashes.add(scalar_hash)
        elif mode == "full":
            full_scalar_hashes.add(scalar_hash)
            full_move_hashes.add(move_hash)
        elif mode == "move":
            move_hashes.add(move_hash)
    prior_identity_ok = (
        prior_shape_ok
        and len(scalar_hashes) == 1
        and scalar_hashes == full_scalar_hashes
        and len(move_hashes) == 1
        and move_hashes == full_move_hashes
    )
    checks.append(
        _check(
            "matched_prior_hash_contract",
            prior_identity_ok,
            (
                f"scalar_hashes={sorted(scalar_hashes | full_scalar_hashes)} "
                f"move_hashes={sorted(move_hashes | full_move_hashes)}"
            ),
        )
    )
    if require_execution_order:
        checks.append(
            _check(
                "execution_order_contract",
                all(
                    str(row.get("execution_order_contract", "")) == require_execution_order
                    for row in metadata_rows
                ),
                f"required={require_execution_order}",
            )
        )
    if require_metric_reference:
        manifest_hashes = {
            str(row.get("metric_reference_manifest_sha256", ""))
            for row in metadata_rows
        }
        checks.append(
            _check(
                "metric_reference_contract",
                all(
                    str(row.get("metric_reference_contract", "")) == require_metric_reference
                    for row in metadata_rows
                )
                and len(manifest_hashes) == 1
                and all(manifest_hashes),
                f"required={require_metric_reference} hashes={sorted(manifest_hashes)}",
            )
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "case_count": len(cases),
        "seeds": list(seeds),
        "algorithms": list(algorithms),
        "checks": checks,
    }


def _trimmed_mean(values: Sequence[float], fraction: float = 0.20) -> float:
    ordered = sorted(values)
    cut = int(len(ordered) * fraction)
    body = ordered[cut : len(ordered) - cut] if cut else ordered
    return mean(body)


def _winsorized_mean(values: Sequence[float], fraction: float = 0.20) -> float:
    ordered = sorted(values)
    cut = int(len(ordered) * fraction)
    if not cut:
        return mean(ordered)
    lower = ordered[cut]
    upper = ordered[-cut - 1]
    return mean([lower] * cut + ordered[cut : len(ordered) - cut] + [upper] * cut)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute an empty quantile.")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(
    values: Sequence[float],
    *,
    draws: int,
    rng: random.Random,
) -> Tuple[float, float]:
    if draws <= 0:
        raise ValueError("bootstrap_draws must be positive.")
    size = len(values)
    samples = [
        mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(draws)
    ]
    return _quantile(samples, 0.025), _quantile(samples, 0.975)


def _sign_flip_p(
    values: Sequence[float],
    *,
    draws: int,
    rng: random.Random,
) -> Tuple[float, str]:
    observed = abs(mean(values))
    size = len(values)
    if size <= 20:
        total = 1 << size
        exceed = 0
        for mask in range(total):
            statistic = abs(
                mean(
                    value if mask & (1 << idx) else -value
                    for idx, value in enumerate(values)
                )
            )
            exceed += statistic >= observed - 1e-15
        return exceed / total, "exact"
    if draws <= 0:
        raise ValueError("randomization_draws must be positive.")
    exceed = 0
    for _ in range(draws):
        statistic = abs(
            mean(value if rng.getrandbits(1) else -value for value in values)
        )
        exceed += statistic >= observed - 1e-15
    return (exceed + 1) / (draws + 1), "monte_carlo"


def analyze_case_cluster_contrast(
    rows: Sequence[Dict[str, object]],
    *,
    left: str,
    right: str,
    metric: str,
    higher_is_better: bool,
    bootstrap_draws: int = 20_000,
    randomization_draws: int = 100_000,
    random_seed: int = 20260726,
    tie_tolerance: float = 1e-12,
) -> Dict[str, object]:
    keyed = {
        (str(row["case"]), int(row["seed"]), str(row["algorithm"])): row
        for row in rows
    }
    cases = sorted({str(row["case"]) for row in rows})
    case_deltas: List[Tuple[str, float]] = []
    matched_seed_rows = 0
    for case in cases:
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if str(row["case"]) == case
                and str(row["algorithm"]) in {left, right}
            }
        )
        deltas = []
        for seed in seeds:
            left_row = keyed.get((case, seed, left))
            right_row = keyed.get((case, seed, right))
            if left_row is None or right_row is None:
                continue
            raw = float(left_row[metric]) - float(right_row[metric])
            deltas.append(raw if higher_is_better else -raw)
        if deltas:
            case_deltas.append((case, mean(deltas)))
            matched_seed_rows += len(deltas)
    if not case_deltas:
        raise ValueError(f"No matched rows for contrast {left} versus {right} on {metric}.")
    values = [value for _, value in case_deltas]
    rng = random.Random(random_seed)
    ci_low, ci_high = _bootstrap_ci(values, draws=bootstrap_draws, rng=rng)
    p_value, p_method = _sign_flip_p(values, draws=randomization_draws, rng=rng)
    wins = sum(value > tie_tolerance for value in values)
    losses = sum(value < -tie_tolerance for value in values)
    ties = len(values) - wins - losses
    leave_one_out = []
    if len(values) > 1:
        for index, (case, _) in enumerate(case_deltas):
            leave_one_out.append(
                {
                    "removed_case": case,
                    "mean_delta": mean(value for idx, value in enumerate(values) if idx != index),
                }
            )
    estimate = mean(values)
    most_influential = (
        max(leave_one_out, key=lambda item: abs(float(item["mean_delta"]) - estimate))
        if leave_one_out
        else None
    )
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "orientation": "positive_means_left_better",
        "independent_case_units": len(values),
        "matched_seed_rows": matched_seed_rows,
        "mean_delta": estimate,
        "median_delta": median(values),
        "case_bootstrap_ci95": [ci_low, ci_high],
        "wins_ties_losses": [wins, ties, losses],
        "trimmed_mean_20pct": _trimmed_mean(values),
        "winsorized_mean_20pct": _winsorized_mean(values),
        "sign_flip_p_two_sided": p_value,
        "sign_flip_method": p_method,
        "leave_one_case_out_min": min(
            (float(item["mean_delta"]) for item in leave_one_out),
            default=estimate,
        ),
        "leave_one_case_out_max": max(
            (float(item["mean_delta"]) for item in leave_one_out),
            default=estimate,
        ),
        "most_influential_case": most_influential,
        "case_deltas": [{"case": case, "delta": value} for case, value in case_deltas],
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _mechanism_summary(
    metadata_rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    fields = (
        "neural_gate_eligible_steps",
        "neural_gate_fired_steps",
        "compiled_polish_child_fraction",
        "scalar_proposal_suppressed_by_learned_move",
        "scalar_candidate_decision_observations",
        "scalar_candidate_changed_decision_rate",
        "scalar_candidate_target_margin_vs_analytic",
        "scalar_candidate_target_regret_to_pool_oracle",
        "scalar_parent_changed_decision_rate",
        "scalar_replacement_preference_flip_rate",
        "learned_move_selection_observations",
        "learned_move_selected_reward_margin_vs_pool_uniform",
        "learned_move_selected_reward_regret_to_pool_oracle",
        "neural_generated_children",
        "neural_accepted_children",
    )
    result: Dict[str, Dict[str, object]] = {}
    for algorithm in ARM_CONTRACTS:
        group = [
            dict(row.get("metadata", {}))
            for row in metadata_rows
            if str(row.get("algorithm")) == algorithm
        ]
        values: Dict[str, object] = {"runs": len(group)}
        for field in fields:
            numeric = [
                float(row[field])
                for row in group
                if field in row and isinstance(row[field], (int, float))
            ]
            if numeric:
                values[f"{field}_median"] = median(numeric)
                values[f"{field}_mean"] = mean(numeric)
                values[f"{field}_zero_runs"] = sum(abs(value) <= 1e-15 for value in numeric)
        result[algorithm] = values
    return result


def build_strict_report(
    rows: Sequence[Dict[str, object]],
    metadata_rows: Sequence[Dict[str, object]],
    *,
    bootstrap_draws: int,
    randomization_draws: int,
    random_seed: int,
) -> Dict[str, object]:
    full = "ips-neural-mv-jitgreedy-targetflow-theory-optimized"
    none = "ips-theory-heavy-no-prior"
    scalar = "ips-theory-endpoint-only"
    move = "ips-theory-move-only"
    contrast_pairs = (
        ("full_minus_none", full, none),
        ("scalar_minus_none", scalar, none),
        ("move_minus_none", move, none),
        ("full_minus_scalar", full, scalar),
        ("full_minus_move", full, move),
    )
    metric_specs = (
        ("case_relative_hypervolume_2d", True),
        ("case_relative_anytime_hv_eval_auc", True),
        ("igd_plus", False),
    )
    analyses: Dict[str, Dict[str, Dict[str, object]]] = {}
    hv_summaries = []
    for contrast_index, (label, left, right) in enumerate(contrast_pairs):
        analyses[label] = {}
        for metric_index, (metric, higher_is_better) in enumerate(metric_specs):
            summary = analyze_case_cluster_contrast(
                rows,
                left=left,
                right=right,
                metric=metric,
                higher_is_better=higher_is_better,
                bootstrap_draws=bootstrap_draws,
                randomization_draws=randomization_draws,
                random_seed=random_seed + 100 * contrast_index + metric_index,
            )
            analyses[label][metric] = summary
            if metric == "case_relative_hypervolume_2d":
                hv_summaries.append(summary)
    adjusted = holm_adjust(
        [float(summary["sign_flip_p_two_sided"]) for summary in hv_summaries]
    )
    gates: Dict[str, Dict[str, object]] = {}
    for (label, _, _), summary, adjusted_p in zip(contrast_pairs, hv_summaries, adjusted):
        summary["holm_adjusted_p"] = adjusted_p
        ci_low = float(summary["case_bootstrap_ci95"][0])  # type: ignore[index]
        passed = (
            ci_low > 0.0
            and adjusted_p <= 0.05
            and float(summary["trimmed_mean_20pct"]) > 0.0
            and float(summary["winsorized_mean_20pct"]) > 0.0
            and int(summary["wins_ties_losses"][0]) > int(summary["wins_ties_losses"][2])  # type: ignore[index]
        )
        gates[label] = {
            "passed": passed,
            "rule": (
                "case-bootstrap CI95 lower > 0; Holm sign-flip p <= 0.05; "
                "trimmed and winsorized means > 0; wins > losses"
            ),
        }
    adoption_passed = all(
        bool(gates[label]["passed"])
        for label in ("full_minus_none", "full_minus_scalar", "full_minus_move")
    )
    return {
        "contrasts": analyses,
        "hypervolume_gates": gates,
        "adoption_gate": {
            "passed": adoption_passed,
            "decision": "ADOPT" if adoption_passed else "DO_NOT_ADOPT",
            "scope": "full prior treatment on case-relative final hypervolume",
        },
        "mechanism_summary": _mechanism_summary(metadata_rows),
    }


def _format_number(value: object) -> str:
    return f"{float(value):.6g}"


def write_markdown_report(path: Path, payload: Dict[str, object]) -> None:
    mechanical = dict(payload["mechanical_validation"])
    strict = dict(payload["strict_analysis"])
    lines = [
        "# Theory Ablation v2 Validation",
        "",
        f"Mechanical contract: **{'PASS' if mechanical['passed'] else 'FAIL'}**",
        "",
        "| check | result | detail |",
        "|---|---:|---|",
    ]
    for check in mechanical["checks"]:  # type: ignore[index]
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        )
    lines.extend(
        [
            "",
            f"Efficacy/adoption gate: **{strict['adoption_gate']['decision']}**",
            "",
            "| contrast | mean rel-HV | case CI95 | W/T/L | trim20 | winsor20 | sign-flip p | Holm p | gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, metrics in strict["contrasts"].items():  # type: ignore[union-attr]
        summary = metrics["case_relative_hypervolume_2d"]
        gate = strict["hypervolume_gates"][label]  # type: ignore[index]
        ci = summary["case_bootstrap_ci95"]
        wtl = summary["wins_ties_losses"]
        lines.append(
            f"| {label} | {_format_number(summary['mean_delta'])} | "
            f"[{_format_number(ci[0])}, {_format_number(ci[1])}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} | "
            f"{_format_number(summary['trimmed_mean_20pct'])} | "
            f"{_format_number(summary['winsorized_mean_20pct'])} | "
            f"{_format_number(summary['sign_flip_p_two_sided'])} | "
            f"{_format_number(summary['holm_adjusted_p'])} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Positive deltas always mean the left algorithm is better. Seeds are averaged within each case; cases are the independent bootstrap and sign-flip units.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate theory_search_v2 mechanics and run strict case-cluster inference."
    )
    parser.add_argument("--suite-output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-evaluations", type=int, required=True)
    parser.add_argument("--require-execution-order", default="")
    parser.add_argument("--require-metric-reference", default="")
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--randomization-draws", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=20260726)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--fail-on-efficacy", action="store_true")
    args = parser.parse_args()
    rows, metadata_rows = load_suite_output(args.suite_output)
    mechanical = validate_v2_contract(
        rows,
        metadata_rows,
        expected_cases=args.expected_cases,
        expected_seeds=parse_csv_ints(args.expected_seeds),
        expected_evaluations=args.expected_evaluations,
        require_execution_order=args.require_execution_order,
        require_metric_reference=args.require_metric_reference,
    )
    strict = build_strict_report(
        rows,
        metadata_rows,
        bootstrap_draws=args.bootstrap_draws,
        randomization_draws=args.randomization_draws,
        random_seed=args.random_seed,
    )
    payload = {
        "suite_output": str(args.suite_output.resolve()),
        "mechanical_validation": mechanical,
        "strict_analysis": strict,
    }
    prefix = args.output_prefix or (args.suite_output / "theory_ablation_v2_validation")
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(markdown_path, payload)
    print(f"Mechanical: {'PASS' if mechanical['passed'] else 'FAIL'}")
    print(f"Adoption: {strict['adoption_gate']['decision']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    if not mechanical["passed"]:
        raise SystemExit(2)
    if args.fail_on_efficacy and not strict["adoption_gate"]["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
