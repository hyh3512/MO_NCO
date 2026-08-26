from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .benchmark import mean, paired_sign_summary


METRICS = (
    ("case_relative_hypervolume_2d", "rel HV"),
    ("case_relative_anytime_hv_eval_auc", "rel eval-AUC"),
    ("case_relative_anytime_hv_time_auc", "rel time-AUC"),
    ("case_relative_hypervolume_per_second", "rel HV/sec"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict SOTA gate for budget-sweep results.")
    parser.add_argument("--runs", type=Path, required=True, help="budget_runs.csv from mo_nco.run_budget_sweep.")
    parser.add_argument("--method", required=True, help="Method under review.")
    parser.add_argument("--baselines", required=True, help="Comma-separated mature baselines.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cases", type=int, default=35)
    parser.add_argument("--min-seeds", type=int, default=10)
    args = parser.parse_args()

    rows = _read_rows(args.runs)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    report = build_report(rows, args.method, baselines, args.min_cases, args.min_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote strict SOTA audit to {args.output}")


def build_report(
    rows: Sequence[Dict[str, str]],
    method: str,
    baselines: Sequence[str],
    min_cases: int,
    min_seeds: int,
) -> str:
    cases = sorted({row["case"] for row in rows})
    seeds = sorted({int(float(row["seed"])) for row in rows})
    available_algorithms = sorted({row["algorithm"] for row in rows})
    lines = [
        "# Strict SOTA Audit",
        "",
        f"Method under review: `{method}`.",
        f"Mature baseline set: {', '.join(f'`{baseline}`' for baseline in baselines)}.",
        "",
        "## Coverage Gate",
        "",
        f"- cases: {len(cases)} / required {min_cases}",
        f"- seeds: {len(seeds)} / required {min_seeds}",
        f"- algorithms present: {', '.join(f'`{name}`' for name in available_algorithms)}",
        "",
    ]

    coverage_pass = len(cases) >= min_cases and len(seeds) >= min_seeds
    baseline_presence = all(baseline in available_algorithms for baseline in baselines)
    method_presence = method in available_algorithms
    lines.append(f"Coverage verdict: `{_passfail(coverage_pass)}`.")
    lines.append(f"Baseline presence verdict: `{_passfail(baseline_presence and method_presence)}`.")
    lines.append("")

    if not method_presence:
        lines.append("The reviewed method is absent from the runs file; no metric audit is possible.")
        return "\n".join(lines) + "\n"

    grouped = _group_by_algorithm(rows)
    lines.extend(_summary_table(grouped, [method, *baselines]))
    lines.append("")
    lines.append("## Pairwise Gates")
    lines.append("")
    lines.append(
        "| baseline | pairs | Δrel HV | HV wins-losses | HV p | Δrel eval-AUC | eval-AUC wins-losses | eval-AUC p | Δrel time-AUC | time-AUC wins-losses | time-AUC p | Δrel HV/sec | HV/sec wins-losses | HV/sec p | verdict |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    metric_passes: List[bool] = []
    for baseline in baselines:
        deltas = _paired_deltas(rows, method, baseline)
        hv = _sign_row(deltas["case_relative_hypervolume_2d"])
        eval_auc = _sign_row(deltas["case_relative_anytime_hv_eval_auc"])
        time_auc = _sign_row(deltas["case_relative_anytime_hv_time_auc"])
        speed = _sign_row(deltas["case_relative_hypervolume_per_second"])
        pairs = len(deltas["case_relative_hypervolume_2d"])
        strong = (
            pairs > 0
            and hv[0] >= 0.0
            and eval_auc[0] >= 0.0
            and time_auc[0] >= 0.0
            and speed[0] >= 0.0
            and hv[1] >= hv[2]
        )
        metric_passes.append(strong)
        lines.append(
            "| "
            f"{baseline} | {pairs} | "
            f"{hv[0]:.6g} | {hv[1]}-{hv[2]} | {hv[3]:.4g} | "
            f"{eval_auc[0]:.6g} | {eval_auc[1]}-{eval_auc[2]} | {eval_auc[3]:.4g} | "
            f"{time_auc[0]:.6g} | {time_auc[1]}-{time_auc[2]} | {time_auc[3]:.4g} | "
            f"{speed[0]:.6g} | {speed[1]}-{speed[2]} | {speed[3]:.4g} | {_passfail(strong)} |"
        )

    sota_pass = coverage_pass and baseline_presence and all(metric_passes)
    lines.extend(
        [
            "",
            "## Overall Verdict",
            "",
            f"Strict SOTA claim gate: `{_passfail(sota_pass)}`.",
            "",
            "A pass requires enough public cases and seeds, all requested mature baselines present, and nonnegative paired mean deltas against every mature baseline on final HV, eval-AUC, time-AUC, and HV/sec. This gate is intentionally conservative; failure means the result can still be promising, but should not be written as a settled SOTA claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_by_algorithm(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)
    return grouped


def _summary_table(grouped: Dict[str, List[Dict[str, str]]], algorithms: Sequence[str]) -> List[str]:
    lines = [
        "## Mean Case-Relative Metrics",
        "",
        "| algorithm | rel HV | rel eval-AUC | rel time-AUC | rel HV/sec |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm in algorithms:
        group = grouped.get(algorithm, [])
        if not group:
            lines.append(f"| {algorithm} | missing | missing | missing | missing |")
            continue
        values = [mean([float(row[key]) for row in group]) for key, _ in METRICS]
        lines.append(f"| {algorithm} | {values[0]:.4g} | {values[1]:.4g} | {values[2]:.4g} | {values[3]:.4g} |")
    return lines


def _paired_deltas(
    rows: Sequence[Dict[str, str]],
    method: str,
    baseline: str,
) -> Dict[str, List[float]]:
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


def _pair_key(row: Dict[str, str]) -> Tuple[str, int, int]:
    return (row["case"], int(float(row["seed"])), int(float(row.get("budget", "0"))))


def _sign_row(deltas: Sequence[float]) -> Tuple[float, int, int, float]:
    wins, losses, p_value = paired_sign_summary(deltas)
    return mean(deltas), wins, losses, p_value


def _passfail(value: bool) -> str:
    return "PASS" if value else "FAIL"


if __name__ == "__main__":
    main()
