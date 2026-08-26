from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS = (
    ("case_relative_hypervolume_2d", "rel HV"),
    ("case_relative_anytime_hv_eval_auc", "rel eval-AUC"),
    ("case_relative_anytime_hv_time_auc", "rel time-AUC"),
    ("case_relative_hypervolume_per_second", "rel HV/sec"),
)

PROPOSAL_KEYS = (
    "learned_move_candidate_count",
    "learned_move_mean_reward",
    "learned_move_positive_reward_rate",
    "learned_move_mean_angle_penalty",
    "learned_move_cone_pass_rate",
    "learned_move_mean_mf_reward",
    "learned_move_good_action_mass",
    "baseline_good_action_mass",
    "learned_move_good_action_mass_margin",
    "learned_move_basin_crossing_mass",
    "baseline_basin_crossing_mass",
    "learned_move_conductance_margin",
    "learned_move_sampled_pool_reward_margin",
    "learned_move_mass_observations",
    "learned_move_calls",
    "learned_move_children",
    "learned_move_updates",
    "learned_move_runtime_target_head_weight",
    "learned_move_runtime_flow_head_weight",
    "learned_move_runtime_mean_field_head_weight",
    "learned_move_runtime_conductance_head_weight",
    "neural_online_training",
    "neural_generated_children",
    "neural_accepted_children",
    "neural_accepted_replacements",
    "neural_scalar_forward_calls",
    "neural_scalar_scored_states",
    "neural_scalar_inference_seconds",
    "local_move_check_upper_bound",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate theorem-side certificates from MO-NCO suite outputs.")
    parser.add_argument("--suite-output", type=Path, default=None, help="Directory containing aggregate_runs.csv.")
    parser.add_argument("--runs", type=Path, default=None, help="Explicit aggregate_runs.csv or budget_runs.csv.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--baselines", required=True, help="Comma-separated comparators.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-quality-margin", type=float, default=0.0)
    args = parser.parse_args()

    runs_path = args.runs or (args.suite_output / "aggregate_runs.csv" if args.suite_output else None)
    if runs_path is None:
        raise SystemExit("Provide --suite-output or --runs.")
    rows = _read_rows(runs_path)
    metadata_rows = _read_metadata(args.suite_output) if args.suite_output else []
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    report = build_report(
        rows,
        metadata_rows,
        method=args.method,
        baselines=baselines,
        min_quality_margin=args.min_quality_margin,
        source=str(runs_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote theorem certificate report to {args.output}")


def build_report(
    rows: Sequence[Dict[str, str]],
    metadata_rows: Sequence[Dict[str, object]],
    method: str,
    baselines: Sequence[str],
    min_quality_margin: float,
    source: str,
) -> str:
    algorithms = sorted({row["algorithm"] for row in rows})
    cases = sorted({row["case"] for row in rows})
    seeds = sorted({int(float(row["seed"])) for row in rows})
    lines = [
        "# Theorem-Side Certificate Report",
        "",
        f"Source: `{source}`",
        f"Method: `{method}`",
        f"Comparators: {', '.join(f'`{item}`' for item in baselines)}",
        "",
        "## Coverage",
        "",
        f"- cases: {len(cases)}",
        f"- seeds: {len(seeds)}",
        f"- algorithms: {', '.join(f'`{item}`' for item in algorithms)}",
        "",
        "## Paired Outcome Margins",
        "",
        "| comparator | pairs | Δrel HV | HV wins-losses | HV p | Δrel eval-AUC | eval wins-losses | eval p | Δrel time-AUC | time wins-losses | time p | Δrel HV/sec | speed wins-losses | speed p | joint nonnegative cases | theorem-outcome gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    outcome_gates: List[bool] = []
    for baseline in baselines:
        deltas = _paired_deltas(rows, method, baseline)
        rows_for_metrics = {key: _sign_row(values) for key, _ in METRICS for values in [deltas[key]]}
        hv = rows_for_metrics["case_relative_hypervolume_2d"]
        eval_auc = rows_for_metrics["case_relative_anytime_hv_eval_auc"]
        time_auc = rows_for_metrics["case_relative_anytime_hv_time_auc"]
        speed = rows_for_metrics["case_relative_hypervolume_per_second"]
        pairs = len(deltas["case_relative_hypervolume_2d"])
        case_deltas = _paired_case_mean_deltas(rows, method, baseline)
        joint_case_passes = sum(
            values["case_relative_hypervolume_2d"] > min_quality_margin
            and values["case_relative_anytime_hv_eval_auc"] >= 0.0
            and values["case_relative_anytime_hv_time_auc"] >= 0.0
            and values["case_relative_hypervolume_per_second"] >= 0.0
            for values in case_deltas.values()
        )
        case_total = len(case_deltas)
        gate = (
            pairs > 0
            and case_total > 0
            and hv[0] > min_quality_margin
            and eval_auc[0] >= 0.0
            and time_auc[0] >= 0.0
            and speed[0] >= 0.0
            and joint_case_passes * 2 > case_total
        )
        outcome_gates.append(gate)
        lines.append(
            "| "
            f"{baseline} | {pairs} | "
            f"{hv[0]:.6g} | {hv[1]}-{hv[2]} | {hv[3]:.4g} | "
            f"{eval_auc[0]:.6g} | {eval_auc[1]}-{eval_auc[2]} | {eval_auc[3]:.4g} | "
            f"{time_auc[0]:.6g} | {time_auc[1]}-{time_auc[2]} | {time_auc[3]:.4g} | "
            f"{speed[0]:.6g} | {speed[1]}-{speed[2]} | {speed[3]:.4g} | "
            f"{joint_case_passes}/{case_total} | "
            f"{_passfail(gate)} |"
        )

    lines.extend(
        [
            "",
            "## Proposal/Internal Certificates",
            "",
            "These quantities estimate the learned proposal assumptions in the theorem. They are diagnostics, not a mathematical proof of performance.",
            "",
            "| algorithm | runs | mean reward | positive reward rate | good-action mass margin | conductance margin | sampled-pool reward margin | mean angle penalty | cone pass rate | mean MF reward | child/call rate | accepted-child rate | neighbor replacements/neural child |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_alg = _metadata_by_algorithm(metadata_rows)
    for algorithm in [method, *baselines]:
        group = by_alg.get(algorithm, [])
        if not group:
            lines.append(
                f"| {algorithm} | 0 | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |"
            )
            continue
        mean_reward = _metadata_mean(group, "learned_move_mean_reward")
        positive_rate = _metadata_mean(group, "learned_move_positive_reward_rate")
        angle = _metadata_mean(group, "learned_move_mean_angle_penalty")
        cone = _metadata_mean(group, "learned_move_cone_pass_rate")
        mf_reward = _metadata_mean(group, "learned_move_mean_mf_reward")
        good_mass_margin = _metadata_mean(group, "learned_move_good_action_mass_margin")
        conductance_margin = _metadata_mean(group, "learned_move_conductance_margin")
        sampled_pool_reward_margin = _metadata_mean(group, "learned_move_sampled_pool_reward_margin")
        calls = _metadata_sum(group, "learned_move_calls")
        children = _metadata_sum(group, "learned_move_children")
        generated = _metadata_sum(group, "neural_generated_children")
        accepted_children = _metadata_sum(group, "neural_accepted_children")
        accepted_replacements = _metadata_sum(group, "neural_accepted_replacements")
        accepted_child_rate = (
            f"{accepted_children / max(1.0, generated):.6g}"
            if all(_metadata_has_key(row, "neural_accepted_children") for row in group)
            else "missing"
        )
        lines.append(
            "| "
            f"{algorithm} | {len(group)} | {mean_reward:.6g} | {positive_rate:.6g} | "
            f"{good_mass_margin:.6g} | {conductance_margin:.6g} | {sampled_pool_reward_margin:.6g} | "
            f"{angle:.6g} | {cone:.6g} | {mf_reward:.6g} | "
            f"{children / max(1.0, calls):.6g} | {accepted_child_rate} | "
            f"{accepted_replacements / max(1.0, generated):.6g} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Proposal-Head Audit",
            "",
            "This table records the weights actually applied after a prior is loaded. It prevents a stored multi-head prior from silently defeating a runtime ablation.",
            "",
            "| algorithm | target/HV | flow | mean-field | conductance | online training | move updates/run |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for algorithm in [method, *baselines]:
        group = by_alg.get(algorithm, [])
        weights = [
            _metadata_mean_if_complete(group, "learned_move_runtime_target_head_weight"),
            _metadata_mean_if_complete(group, "learned_move_runtime_flow_head_weight"),
            _metadata_mean_if_complete(group, "learned_move_runtime_mean_field_head_weight"),
            _metadata_mean_if_complete(group, "learned_move_runtime_conductance_head_weight"),
            _metadata_mean_if_complete(group, "neural_online_training"),
            _metadata_mean_if_complete(group, "learned_move_updates"),
        ]
        lines.append(f"| {algorithm} | {' | '.join(weights)} |")

    lines.extend(
        [
            "",
            "## Runtime Cost Audit",
            "",
            "Local move checks are deterministic upper bounds because accelerated descents may stop before exhausting every configured pass.",
            "",
            "| algorithm | local-move check upper bound/run | scalar forwards/run | scalar states/forward | scalar inference seconds/run |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for algorithm in [method, *baselines]:
        group = by_alg.get(algorithm, [])
        if not group or not all(_metadata_has_key(row, "local_move_check_upper_bound") for row in group):
            lines.append(f"| {algorithm} | missing | missing | missing | missing |")
            continue
        forwards = _metadata_sum(group, "neural_scalar_forward_calls")
        states = _metadata_sum(group, "neural_scalar_scored_states")
        lines.append(
            f"| {algorithm} | {_metadata_mean(group, 'local_move_check_upper_bound'):.6g} | "
            f"{forwards / len(group):.6g} | {states / max(1.0, forwards):.6g} | "
            f"{_metadata_mean(group, 'neural_scalar_inference_seconds'):.6g} |"
        )

    method_group = by_alg.get(method, [])
    contract_required = method.endswith("-targetflow-theory-optimized") or any(
        _metadata_value(row, "neural_endpoint_contract_required") > 0.5
        or _metadata_value(row, "move_target_only_contract_required") > 0.5
        for row in method_group
    )
    contract_runtime_keys = (
        "neural_endpoint_contract_satisfied",
        "move_target_only_contract_satisfied",
        "neural_online_training",
        "learned_move_updates",
        "learned_move_runtime_flow_head_weight",
        "learned_move_runtime_mean_field_head_weight",
        "learned_move_runtime_conductance_head_weight",
    )
    prior_hash_gate = all(
        (
            _metadata_value(row, "neural_endpoint_contract_required") <= 0.5
            or bool(str(_metadata_raw(row, "neural_prior_sha256")))
        )
        and (
            _metadata_value(row, "move_target_only_contract_required") <= 0.5
            or bool(str(_metadata_raw(row, "learned_move_prior_sha256")))
        )
        for row in method_group
    )
    contract_gate = (not contract_required) or (
        bool(method_group)
        and all(all(_metadata_has_key(row, key) for key in contract_runtime_keys) for row in method_group)
        and prior_hash_gate
        and _metadata_mean(method_group, "neural_endpoint_contract_satisfied") == 1.0
        and _metadata_mean(method_group, "move_target_only_contract_satisfied") == 1.0
        and _metadata_mean(method_group, "neural_online_training") == 0.0
        and _metadata_mean(method_group, "learned_move_updates") == 0.0
        and _metadata_mean(method_group, "learned_move_runtime_flow_head_weight") == 0.0
        and _metadata_mean(method_group, "learned_move_runtime_mean_field_head_weight") == 0.0
        and _metadata_mean(method_group, "learned_move_runtime_conductance_head_weight") == 0.0
    )
    method_generated = _metadata_sum(method_group, "neural_generated_children")
    method_accepted_children = _metadata_sum(method_group, "neural_accepted_children")
    proposal_gate = (
        bool(method_group)
        and _metadata_mean(method_group, "learned_move_mean_reward") > 0.0
        and _metadata_mean(method_group, "learned_move_positive_reward_rate") > 0.5
        and _metadata_mean(method_group, "learned_move_good_action_mass_margin") > 0.0
        and _metadata_mean(method_group, "learned_move_conductance_margin") > 0.0
        and method_generated > 0.0
        and method_accepted_children > 0.0
        and all(_metadata_has_key(row, "neural_accepted_children") for row in method_group)
    )
    all_outcomes_gate = bool(outcome_gates) and all(outcome_gates)
    overall_gate = all_outcomes_gate and proposal_gate and contract_gate
    lines.extend(
        [
            "",
            "## Overall Certificate Decision",
            "",
            f"- all-comparator outcome gate: {_passfail(all_outcomes_gate)}",
            f"- learned-proposal internal gate: {_passfail(proposal_gate)}",
            f"- endpoint/target-only runtime contract gate: {_passfail(contract_gate)}",
            f"- overall certificate: {_passfail(overall_gate)}",
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "- A strict theorem certificate needs nonnegative paired outcome deltas against each named comparator and positive proposal-side reward/margin diagnostics.",
            "- The outcome gate also requires a strict majority of independent cases to have positive final-HV margin and nonnegative eval-AUC, time-AUC, and HV/sec margins simultaneously; a single large outlier cannot pass this gate.",
            "- A failed HV/sec row means the method may still be sample-efficient, but it does not satisfy the wall-clock SOTA claim.",
            "- Missing metadata means this run predates proposal-side instrumentation or the comparator is an external one-shot solver.",
            "- A no-MF/no-flow/frozen-prior label is mechanically audited only when the corresponding runtime-head weight is present and zero, online training is zero, and move updates per run is zero where freezing is claimed.",
            "- The theory-optimized alias additionally requires every run to report endpoint_state_v1 and target-only contracts plus nonempty SHA-256 hashes for every required frozen prior; legacy or unauditable priors cannot pass this gate.",
            "- Neighbor replacements per neural child is not an acceptance probability: one accepted child may replace several reference-direction neighbors, so this diagnostic can exceed one.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_metadata(suite_output: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in suite_output.rglob("run_metadata.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload["case"] = path.parent.name
                rows.append(payload)
    return rows


def _metadata_by_algorithm(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("algorithm", ""))].append(row)
    return grouped


def _metadata_value(row: Dict[str, object], key: str) -> float:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0.0
    value = metadata.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metadata_raw(row: Dict[str, object], key: str) -> object:
    metadata = row.get("metadata", {})
    return metadata.get(key, "") if isinstance(metadata, dict) else ""


def _metadata_has_key(row: Dict[str, object], key: str) -> bool:
    metadata = row.get("metadata", {})
    return isinstance(metadata, dict) and key in metadata


def _metadata_mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_metadata_value(row, key) for row in rows) / len(rows)


def _metadata_sum(rows: Sequence[Dict[str, object]], key: str) -> float:
    return sum(_metadata_value(row, key) for row in rows)


def _metadata_mean_if_complete(rows: Sequence[Dict[str, object]], key: str) -> str:
    if not rows or not all(_metadata_has_key(row, key) for row in rows):
        return "missing"
    return f"{_metadata_mean(rows, key):.6g}"


def _paired_deltas(rows: Sequence[Dict[str, str]], method: str, baseline: str) -> Dict[str, List[float]]:
    keyed = {(_pair_key(row), row["algorithm"]): row for row in rows}
    method_rows = [row for row in rows if row["algorithm"] == method]
    deltas = {key: [] for key, _ in METRICS}
    for row in method_rows:
        other = keyed.get((_pair_key(row), baseline))
        if other is None:
            continue
        for key, _ in METRICS:
            deltas[key].append(float(row[key]) - float(other[key]))
    return deltas


def _paired_case_mean_deltas(
    rows: Sequence[Dict[str, str]],
    method: str,
    baseline: str,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    keyed = {(_pair_key(row), row["algorithm"]): row for row in rows}
    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(
        lambda: {key: [] for key, _ in METRICS}
    )
    for row in rows:
        if row["algorithm"] != method:
            continue
        other = keyed.get((_pair_key(row), baseline))
        if other is None:
            continue
        case_key = (row["case"], int(float(row.get("budget", "0"))))
        for key, _ in METRICS:
            grouped[case_key][key].append(float(row[key]) - float(other[key]))
    return {
        case_key: {metric: _mean(values) for metric, values in metrics.items()}
        for case_key, metrics in grouped.items()
    }


def _pair_key(row: Dict[str, str]) -> Tuple[str, int, int]:
    return (row["case"], int(float(row["seed"])), int(float(row.get("budget", "0"))))


def _sign_row(deltas: Sequence[float]) -> Tuple[float, int, int, float]:
    wins = sum(delta > 0.0 for delta in deltas)
    losses = sum(delta < 0.0 for delta in deltas)
    p_value = _sign_test_p_value(wins, losses)
    return _mean(deltas), wins, losses, p_value


def _sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _passfail(value: bool) -> str:
    return "PASS" if value else "FAIL"


if __name__ == "__main__":
    main()
