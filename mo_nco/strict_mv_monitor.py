from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .benchmark import mean, paired_sign_summary


MetricRow = Dict[str, str]


METRICS = (
    ("case_relative_hypervolume_2d", "rel HV"),
    ("case_relative_anytime_hv_eval_auc", "rel eval-AUC"),
    ("case_relative_anytime_hv_time_auc", "rel time-AUC"),
    ("case_relative_hypervolume_per_second", "rel HV/sec"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict monitors for McKean-Vlasov-lite budget sweeps.")
    parser.add_argument("--runs", type=Path, required=True, help="budget_runs.csv from run_budget_sweep.")
    parser.add_argument("--method", default="ips-neural-mv-lite")
    parser.add_argument("--no-mf", default="ips-neural-mv-lite-no-mf")
    parser.add_argument("--no-topk", default="ips-neural-mv-lite-no-topk")
    parser.add_argument("--non-neural", default="ips-quality-relocate")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.runs)
    report = build_report(
        rows=rows,
        method=args.method,
        no_mf=args.no_mf,
        no_topk=args.no_topk,
        non_neural=args.non_neural,
        train_fraction=args.train_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote strict MV-lite monitor report to {args.output}")


def build_report(
    rows: Sequence[MetricRow],
    method: str,
    no_mf: str,
    no_topk: str,
    non_neural: str,
    train_fraction: float,
) -> str:
    cases = sorted({row["case"] for row in rows})
    budgets = sorted({int(float(row["budget"])) for row in rows})
    algorithms = sorted({row["algorithm"] for row in rows})
    split_at = max(1, min(len(cases), int(round(len(cases) * max(0.0, min(1.0, train_fraction))))))
    train_cases = set(cases[:split_at])
    unseen_cases = set(cases[split_at:])

    lines = [
        "# Strict McKean-Vlasov-lite Monitor",
        "",
        f"Method: `{method}`.",
        f"Mean-field ablation: `{no_mf}`.",
        f"Top-K ablation: `{no_topk}`.",
        f"Non-neural baseline: `{non_neural}`.",
        "",
        "## Coverage",
        "",
        f"- cases: {len(cases)}",
        f"- budgets: {', '.join(map(str, budgets))}",
        f"- algorithms present: {', '.join(f'`{name}`' for name in algorithms)}",
        f"- deterministic train/unseen split: {len(train_cases)} train, {len(unseen_cases)} unseen",
        "",
    ]

    lines.extend(_mean_field_collapse_section(rows, method, no_mf, budgets))
    lines.extend(_topk_section(rows, method, no_topk, budgets))
    lines.extend(_generalization_section(rows, method, non_neural, train_cases, unseen_cases))
    lines.extend(_capacity_freezing_section(rows, method, budgets))
    lines.extend(_verdict_section(rows, method, no_mf, non_neural, budgets, unseen_cases))
    return "\n".join(lines).rstrip() + "\n"


def _mean_field_collapse_section(
    rows: Sequence[MetricRow],
    method: str,
    no_mf: str,
    budgets: Sequence[int],
) -> List[str]:
    lines = [
        "## Mean-Field Collapse Monitor",
        "",
        "| budget | Δrel HV vs no-mf | HV wins-losses | HV p | Δrel eval-AUC | eval-AUC wins-losses | eval-AUC p |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        budget_rows = [row for row in rows if int(float(row["budget"])) == budget]
        hv = _paired_metric(budget_rows, method, no_mf, "case_relative_hypervolume_2d")
        auc = _paired_metric(budget_rows, method, no_mf, "case_relative_anytime_hv_eval_auc")
        lines.append(
            f"| {budget} | {hv[0]:.6g} | {hv[1]}-{hv[2]} | {hv[3]:.4g} | "
            f"{auc[0]:.6g} | {auc[1]}-{auc[2]} | {auc[3]:.4g} |"
        )
    lines.extend(
        [
            "",
            "Alarm rule: if the highest-budget Δrel HV turns negative, or if no-mf wins most matched runs at high budget, treat it as possible mean-field over-repulsion/collapse.",
            "",
        ]
    )
    return lines


def _topk_section(
    rows: Sequence[MetricRow],
    method: str,
    no_topk: str,
    budgets: Sequence[int],
) -> List[str]:
    lines = [
        "## Lazy Top-K Monitor",
        "",
        "| budget | Δrel HV/sec vs no-topk | HV/sec wins-losses | HV/sec p | Δrel time-AUC | time-AUC wins-losses | time-AUC p |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        budget_rows = [row for row in rows if int(float(row["budget"])) == budget]
        speed = _paired_metric(budget_rows, method, no_topk, "case_relative_hypervolume_per_second")
        time_auc = _paired_metric(budget_rows, method, no_topk, "case_relative_anytime_hv_time_auc")
        lines.append(
            f"| {budget} | {speed[0]:.6g} | {speed[1]}-{speed[2]} | {speed[3]:.4g} | "
            f"{time_auc[0]:.6g} | {time_auc[1]}-{time_auc[2]} | {time_auc[3]:.4g} |"
        )
    lines.extend(
        [
            "",
            "Alarm rule: Top-K should not materially reduce final HV. A speed win without HV loss supports the lazy-filter design; a speed loss means the Top-K pool is too small or Python overhead dominates.",
            "",
        ]
    )
    return lines


def _generalization_section(
    rows: Sequence[MetricRow],
    method: str,
    non_neural: str,
    train_cases: set[str],
    unseen_cases: set[str],
) -> List[str]:
    lines = [
        "## Train/Unseen Generalization Monitor",
        "",
        "This split is a result stratification split. It is a true zero-shot protocol only when the neural prior is trained on the train cases and frozen on unseen cases.",
        "",
        "| split | algorithm | rel HV | rel eval-AUC | rel time-AUC | rel HV/sec |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split_name, case_set in (("train", train_cases), ("unseen", unseen_cases)):
        split_rows = [row for row in rows if row["case"] in case_set]
        for algorithm in (method, non_neural):
            group = [row for row in split_rows if row["algorithm"] == algorithm]
            lines.append(f"| {split_name} | {algorithm} | {_metric_means(group)} |")
    unseen_rows = [row for row in rows if row["case"] in unseen_cases]
    hv = _paired_metric(unseen_rows, method, non_neural, "case_relative_hypervolume_2d")
    auc = _paired_metric(unseen_rows, method, non_neural, "case_relative_anytime_hv_eval_auc")
    lines.extend(
        [
            "",
            f"Unseen paired Δrel HV vs `{non_neural}`: {hv[0]:.6g} ({hv[1]}-{hv[2]}, p={hv[3]:.4g}).",
            f"Unseen paired Δrel eval-AUC vs `{non_neural}`: {auc[0]:.6g} ({auc[1]}-{auc[2]}, p={auc[3]:.4g}).",
            "Alarm rule: if unseen Δrel eval-AUC or Δrel HV is negative, do not claim neural generalization.",
            "",
        ]
    )
    return lines


def _capacity_freezing_section(rows: Sequence[MetricRow], method: str, budgets: Sequence[int]) -> List[str]:
    lines = [
        "## Capacity Freezing Monitor",
        "",
        "| budget | rejection rate mean | max rejection streak mean | acceptance rate mean |",
        "|---:|---:|---:|---:|",
    ]
    for budget in budgets:
        group = [row for row in rows if row["algorithm"] == method and int(float(row["budget"])) == budget]
        rejection = _safe_mean(row.get("rejection_rate", "0") for row in group)
        streak = _safe_mean(row.get("max_rejection_streak", "0") for row in group)
        acceptance = _safe_mean(row.get("acceptance_rate", "0") for row in group)
        lines.append(f"| {budget} | {rejection:.4g} | {streak:.4g} | {acceptance:.4g} |")
    lines.extend(
        [
            "",
            "Interpretation: late-budget rejection growth is compatible with kinetic freezing only if HV/eval-AUC has already plateaued. High rejection with poor HV is a search-stagnation failure, not theoretical validation.",
            "",
        ]
    )
    return lines


def _verdict_section(
    rows: Sequence[MetricRow],
    method: str,
    no_mf: str,
    non_neural: str,
    budgets: Sequence[int],
    unseen_cases: set[str],
) -> List[str]:
    high_budget = max(budgets) if budgets else 0
    high_rows = [row for row in rows if int(float(row["budget"])) == high_budget]
    unseen_high_rows = [row for row in high_rows if row["case"] in unseen_cases]
    mf_hv = _paired_metric(high_rows, method, no_mf, "case_relative_hypervolume_2d")
    base_auc = _paired_metric(unseen_high_rows, method, non_neural, "case_relative_anytime_hv_eval_auc")
    base_hv = _paired_metric(unseen_high_rows, method, non_neural, "case_relative_hypervolume_2d")
    pass_gate = mf_hv[0] > 0.0 and base_auc[0] >= 0.0 and base_hv[0] >= 0.0
    lines = [
        "## Strict Monitor Verdict",
        "",
        f"Highest budget checked: `{high_budget}`.",
        f"- high-budget mean-field Δrel HV positive: `{_passfail(mf_hv[0] > 0.0)}`.",
        f"- unseen Δrel eval-AUC vs non-neural nonnegative: `{_passfail(base_auc[0] >= 0.0)}`.",
        f"- unseen Δrel HV vs non-neural nonnegative: `{_passfail(base_hv[0] >= 0.0)}`.",
        "",
        f"Monitor gate: `{_passfail(pass_gate)}`.",
        "",
        "This monitor is necessary but not sufficient for SOTA. Mature external baselines and the separate strict SOTA audit still control the final claim.",
    ]
    return lines


def _read_rows(path: Path) -> List[MetricRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _paired_metric(rows: Sequence[MetricRow], method: str, baseline: str, metric: str) -> Tuple[float, int, int, float]:
    keyed = {(row["case"], int(float(row["seed"])), row["algorithm"]): row for row in rows}
    deltas = []
    for row in rows:
        if row["algorithm"] != method:
            continue
        other = keyed.get((row["case"], int(float(row["seed"])), baseline))
        if other is None:
            continue
        deltas.append(float(row.get(metric, 0.0)) - float(other.get(metric, 0.0)))
    wins, losses, p_value = paired_sign_summary(deltas)
    return mean(deltas), wins, losses, p_value


def _metric_means(rows: Sequence[MetricRow]) -> str:
    if not rows:
        return "missing | missing | missing | missing"
    values = [mean([float(row.get(key, 0.0)) for row in rows]) for key, _ in METRICS]
    return f"{values[0]:.4g} | {values[1]:.4g} | {values[2]:.4g} | {values[3]:.4g}"


def _safe_mean(values: Iterable[str]) -> float:
    parsed = [float(value) for value in values]
    return sum(parsed) / len(parsed) if parsed else 0.0


def _passfail(value: bool) -> str:
    return "PASS" if value else "FAIL"


if __name__ == "__main__":
    main()
