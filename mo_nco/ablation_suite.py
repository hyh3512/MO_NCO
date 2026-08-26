from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

from .ablation import AblationVariant, DEFAULT_IPS_ABLATIONS, run_ips_ablation
from .benchmark import paired_sign_summary
from .benchmark_suite import BenchmarkSuite


def run_ips_ablation_suite(
    suite: BenchmarkSuite,
    seeds: Sequence[int],
    output_dir: Path,
    default_population: int,
    default_evaluations: int,
    log_period: int,
    variants: Sequence[AblationVariant] = DEFAULT_IPS_ABLATIONS,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for case in suite.cases:
        instance = case.load_instance()
        if instance is None:
            from .instance import MultiObjectiveTSPInstance

            instance = MultiObjectiveTSPInstance.random_biobjective(case.cities, seed=case.instance_seed)
        case_rows = run_ips_ablation(
            instance=instance,
            seeds=seeds,
            output_dir=output_dir / case.name,
            population=int(case.population or default_population),
            evaluations=int(case.evaluations or default_evaluations),
            log_period=log_period,
            variants=variants,
        )
        for row in case_rows:
            expanded = dict(row)
            expanded["case"] = case.name
            rows.append(expanded)
    add_case_relative_ablation_metrics(rows)
    write_ablation_suite_csv(output_dir / "ablation_suite_runs.csv", rows)
    write_ablation_suite_summary(output_dir / "ablation_suite_summary.md", rows)
    return rows


def add_case_relative_ablation_metrics(rows: Sequence[Dict[str, object]]) -> None:
    by_case: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    for case_rows in by_case.values():
        best_hv = max(float(row.get("hypervolume_2d", 0.0)) for row in case_rows)
        best_speed = max(float(row.get("hypervolume_per_second", 0.0)) for row in case_rows)
        for row in case_rows:
            row["case_relative_hypervolume_2d"] = (
                float(row.get("hypervolume_2d", 0.0)) / best_hv if best_hv > 0.0 else 0.0
            )
            row["case_relative_hypervolume_per_second"] = (
                float(row.get("hypervolume_per_second", 0.0)) / best_speed if best_speed > 0.0 else 0.0
            )


def write_ablation_suite_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["case", *[field for field in rows[0] if field != "case"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_suite_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("# IPS Cross-Instance Ablation Summary\n\nNo rows.\n", encoding="utf-8")
        return
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    anchor = str(rows[0]["variant"])
    anchor_by_case_seed = {
        (str(row["case"]), int(row["seed"])): row for row in grouped.get(anchor, [])
    }
    lines = [
        "# IPS Cross-Instance Ablation Summary",
        "",
        f"Anchor: `{anchor}`. Deltas are anchor minus comparator over matched `(case, seed)` pairs.",
        "",
        "| variant | cases | runs | rel HV mean | rel HV/sec mean | Δrel HV anchor | rel HV wins-losses | sign p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, group in sorted(grouped.items()):
        hv_values = [float(row.get("case_relative_hypervolume_2d", 0.0)) for row in group]
        speed_values = [float(row.get("case_relative_hypervolume_per_second", 0.0)) for row in group]
        deltas = []
        for row in group:
            key = (str(row["case"]), int(row["seed"]))
            if key in anchor_by_case_seed:
                deltas.append(
                    float(anchor_by_case_seed[key].get("case_relative_hypervolume_2d", 0.0))
                    - float(row.get("case_relative_hypervolume_2d", 0.0))
                )
        wins, losses, p_value = paired_sign_summary(deltas)
        lines.append(
            f"| {variant} | {len({str(row['case']) for row in group})} | {len(group)} | "
            f"{sum(hv_values) / len(hv_values):.6g} | {sum(speed_values) / len(speed_values):.6g} | "
            f"{sum(deltas) / max(1, len(deltas)):.6g} | {wins}-{losses} | {p_value:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
