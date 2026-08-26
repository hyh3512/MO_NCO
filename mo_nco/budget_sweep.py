from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

from .benchmark_suite import BenchmarkSuite, run_benchmark_suite, write_suite_pairwise, write_suite_summary
from .benchmark import RunRecord


def run_budget_sweep(
    suite: BenchmarkSuite,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    budgets: Sequence[int],
    output_dir: Path,
    default_population: int,
    log_period: int,
    archive_update_period: int,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any("jit" in name or "numba" in name for name in algorithms):
        try:
            from .numba_kernels import warmup_numba_kernels

            warmup_numba_kernels()
        except Exception:
            pass
    rows: List[Dict[str, object]] = []
    for budget in budgets:
        budget_suite = BenchmarkSuite(
            suite.name,
            tuple(replace(case, evaluations=budget) for case in suite.cases),
        )
        budget_dir = output_dir / f"eval{budget}"
        records: List[RunRecord] = run_benchmark_suite(
            suite=budget_suite,
            algorithms=algorithms,
            seeds=seeds,
            output_dir=budget_dir,
            default_population=default_population,
            default_evaluations=budget,
            log_period=log_period,
            archive_update_period=archive_update_period,
        )
        aggregate_path = budget_dir / "aggregate_runs.csv"
        if aggregate_path.exists():
            with aggregate_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["budget"] = budget
                    rows.append(row)
        elif records:
            for record in records:
                row = record.__dict__.copy()
                row["budget"] = budget
                rows.append(row)
    write_budget_runs(output_dir / "budget_runs.csv", rows)
    write_budget_summary(output_dir / "budget_summary.md", rows)
    return rows


def write_budget_runs(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["budget", *[field for field in rows[0] if field != "budget"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_budget_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("# Budget Sweep Summary\n\nNo rows.\n", encoding="utf-8")
        return
    lines = [
        "# Budget Sweep Summary",
        "",
        "Each block uses case-relative metrics within the corresponding budget.",
        "",
    ]
    budgets = sorted({int(row["budget"]) for row in rows})
    for budget in budgets:
        budget_rows = [row for row in rows if int(row["budget"]) == budget]
        tmp_summary = path.with_name(f"budget_{budget}_summary.tmp.md")
        tmp_pairwise = path.with_name(f"budget_{budget}_pairwise.tmp.md")
        write_suite_summary(tmp_summary, budget_rows)
        write_suite_pairwise(tmp_pairwise, budget_rows)
        lines.append(f"## Budget {budget}")
        lines.extend(tmp_summary.read_text(encoding="utf-8").splitlines()[2:])
        lines.append("")
        lines.extend(tmp_pairwise.read_text(encoding="utf-8").splitlines()[2:])
        lines.append("")
        tmp_summary.unlink(missing_ok=True)
        tmp_pairwise.unlink(missing_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
