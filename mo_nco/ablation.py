from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .benchmark import common_reference, paired_sign_summary
from .evaluation import CountingTSPInstance, evaluation_count
from .instance import MultiObjectiveTSPInstance
from .ips_efficient import TheoryAlignedIPSOptimizer
from .sampler import OptimizationResult


@dataclass(frozen=True)
class AblationVariant:
    name: str
    params: Dict[str, object]


DEFAULT_IPS_ABLATIONS: Sequence[AblationVariant] = (
    AblationVariant(
        "full_archive_neural",
        {
            "neighbor_size": 8,
            "proposal": "two_opt",
            "extra_two_opt_probability": 0.0,
            "archive_parent_probability": 0.12,
            "archive_parent_sample": 6,
            "crossover_probability": 0.0,
            "archive_update_period": 64,
            "archive_conditioning_weight": 3.0,
            "neural_scalar_weight": 0.08,
            "neural_training_epochs": 4,
            "neural_archive_repeats": 4,
            "neural_proposal_probability": 0.55,
            "neural_proposal_weight": 0.35,
            "neural_candidate_pool": 4,
            "neural_proposal_min_samples": 64,
            "initialization": "mixed_scalar_greedy",
            "greedy_candidate_pool": 3,
            "initial_2opt_passes": 3,
            "proposal_2opt_passes": 1,
            "initial_temperature": 0.0,
            "final_temperature": 0.0,
        },
    ),
    AblationVariant(
        "core_lowtemp_no_archive_no_neural",
        {
            "archive_parent_probability": 0.0,
            "archive_conditioning_weight": 0.0,
            "neural_scalar_weight": 0.0,
            "neural_proposal_probability": 0.0,
            "neural_proposal_weight": 0.0,
            "initialization": "random",
            "initial_2opt_passes": 0,
            "proposal_2opt_passes": 0,
            "archive_update_period": None,
        },
    ),
    AblationVariant("no_archive_scalar", {"archive_conditioning_weight": 0.0}),
    AblationVariant("no_neural_scalar", {"neural_scalar_weight": 0.0}),
    AblationVariant("no_neural_proposal", {"neural_proposal_probability": 0.0, "neural_proposal_weight": 0.0}),
    AblationVariant("random_initialization", {"initialization": "random"}),
    AblationVariant("no_scalar_descent", {"initial_2opt_passes": 0, "proposal_2opt_passes": 0}),
    AblationVariant("no_archive_parent", {"archive_parent_probability": 0.0}),
    AblationVariant("neighbor6", {"neighbor_size": 6}),
    AblationVariant("mixed_proposal", {"proposal": "mixed"}),
    AblationVariant("finite_temperature", {"initial_temperature": 0.02, "final_temperature": 0.001}),
)


def run_ips_ablation(
    instance: MultiObjectiveTSPInstance,
    seeds: Sequence[int],
    output_dir: Path,
    population: int,
    evaluations: int,
    log_period: int,
    variants: Sequence[AblationVariant] = DEFAULT_IPS_ABLATIONS,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: List[Dict[str, object]] = []
    for variant in variants:
        params = dict(DEFAULT_IPS_ABLATIONS[0].params)
        params.update(variant.params)
        if params.get("archive_update_period") is None:
            params["archive_update_period"] = max(evaluations + 1, 1)
        for seed in seeds:
            counted = CountingTSPInstance(instance, max_evaluations=evaluations)
            start = time.perf_counter()
            result = TheoryAlignedIPSOptimizer(
                instance=counted,  # type: ignore[arg-type]
                num_particles=population,
                evaluations=evaluations,
                seed=seed,
                log_period=log_period,
                **params,
            ).run()
            runtime = time.perf_counter() - start
            evals = evaluation_count(counted)
            raw_rows.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "runtime_seconds": runtime,
                    "evaluations": evals,
                    "archive_size": len(result.archive),
                    "result": result,
                }
            )
    reference = common_reference([row["result"] for row in raw_rows if isinstance(row["result"], OptimizationResult)])
    rows: List[Dict[str, object]] = []
    for row in raw_rows:
        result = row.pop("result")
        if not isinstance(result, OptimizationResult):
            continue
        hv = result.archive.hypervolume_2d(reference=reference) if reference is not None else 0.0
        runtime = float(row["runtime_seconds"])
        evals = int(row["evaluations"])
        row["hypervolume_2d"] = hv
        row["hypervolume_per_second"] = hv / max(runtime, 1e-12)
        row["hypervolume_per_evaluation"] = hv / max(evals, 1)
        rows.append(row)
    write_ablation_csv(output_dir / "ablation_runs.csv", rows)
    write_ablation_summary(output_dir / "ablation_summary.md", rows)
    return rows


def write_ablation_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    anchor = str(rows[0]["variant"]) if rows else "full_archive_neural"
    anchor_by_seed = {int(row["seed"]): row for row in grouped.get(anchor, [])}
    lines = [
        "# IPS Ablation Summary",
        "",
        f"Anchor: `{anchor}`. Positive deltas mean the anchor is better.",
        "",
        "| variant | HV mean | HV/sec mean | evals mean | ΔHV anchor | HV wins-losses | sign p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, group in sorted(grouped.items()):
        hv_values = [float(row["hypervolume_2d"]) for row in group]
        speed_values = [float(row["hypervolume_per_second"]) for row in group]
        eval_values = [float(row["evaluations"]) for row in group]
        deltas = []
        for row in group:
            seed = int(row["seed"])
            if seed in anchor_by_seed:
                deltas.append(float(anchor_by_seed[seed]["hypervolume_2d"]) - float(row["hypervolume_2d"]))
        wins, losses, p_value = paired_sign_summary(deltas)
        lines.append(
            f"| {variant} | {sum(hv_values) / len(hv_values):.6g} | "
            f"{sum(speed_values) / len(speed_values):.6g} | {sum(eval_values) / len(eval_values):.4g} | "
            f"{sum(deltas) / max(1, len(deltas)):.6g} | {wins}-{losses} | {p_value:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
