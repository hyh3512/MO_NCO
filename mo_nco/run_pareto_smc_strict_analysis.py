from __future__ import annotations

"""Cluster-aware strict paired analysis for the Pareto-SMC matched suite."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


METRICS = {
    "relative_final_hv": (
        "case_relative_hypervolume_2d",
        "higher",
    ),
    "relative_eval_auc": (
        "case_relative_anytime_hv_eval_auc",
        "higher",
    ),
    "igd_plus": ("igd_plus", "lower"),
    "additive_epsilon": ("additive_epsilon", "lower"),
    "runtime_seconds": ("runtime_seconds", "lower"),
}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _trimmed_mean(values: Sequence[float], fraction: float = 0.10) -> float:
    ordered = sorted(values)
    trim = int(math.floor(fraction * len(ordered)))
    kept = ordered[trim : len(ordered) - trim] if trim else ordered
    return sum(kept) / len(kept)


def _winsorized_mean(
    values: Sequence[float],
    fraction: float = 0.10,
) -> float:
    ordered = sorted(values)
    trim = int(math.floor(fraction * len(ordered)))
    if trim == 0:
        return sum(ordered) / len(ordered)
    low = ordered[trim]
    high = ordered[-trim - 1]
    winsorized = [max(low, min(high, value)) for value in ordered]
    return sum(winsorized) / len(winsorized)


def _cluster_bootstrap_ci(
    case_deltas: dict[str, Sequence[float]],
    *,
    repetitions: int,
    rng: random.Random,
) -> tuple[float, float]:
    cases = sorted(case_deltas)
    case_means = {
        case: sum(case_deltas[case]) / len(case_deltas[case])
        for case in cases
    }
    samples = []
    for _ in range(repetitions):
        draw = [rng.choice(cases) for _ in cases]
        samples.append(sum(case_means[case] for case in draw) / len(draw))
    return _quantile(samples, 0.025), _quantile(samples, 0.975)


def _cluster_randomization_p(
    case_deltas: dict[str, Sequence[float]],
    *,
    repetitions: int,
    rng: random.Random,
) -> tuple[float, str]:
    case_means = tuple(
        sum(values) / len(values)
        for _, values in sorted(case_deltas.items())
    )
    observed = abs(sum(case_means) / len(case_means))
    if len(case_means) <= 20:
        exceed = 0
        total = 1 << len(case_means)
        for mask in range(total):
            randomized = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(case_means)
            ) / len(case_means)
            exceed += int(abs(randomized) >= observed - 1e-15)
        return exceed / total, "exact_cluster_sign_flip"
    exceed = 0
    for _ in range(repetitions):
        randomized = sum(
            value if rng.random() < 0.5 else -value
            for value in case_means
        ) / len(case_means)
        exceed += int(abs(randomized) >= observed - 1e-15)
    return (exceed + 1) / (repetitions + 1), "monte_carlo_cluster_sign_flip"


def _advantage(anchor: float, comparator: float, direction: str) -> float:
    if direction == "higher":
        return anchor - comparator
    return comparator - anchor


def analyze(
    aggregate_csv: Path,
    *,
    anchor: str,
    expected_cases: int,
    expected_seeds: int,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
    random_seed: int,
    igd_noninferiority_margin: float,
) -> dict[str, object]:
    if expected_cases <= 0:
        raise ValueError("expected_cases must be positive.")
    if expected_seeds <= 0:
        raise ValueError("expected_seeds must be positive.")
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive.")
    if randomization_repetitions <= 0:
        raise ValueError("randomization_repetitions must be positive.")
    with aggregate_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("aggregate_runs.csv is empty.")
    algorithms = sorted({row["algorithm"] for row in rows})
    if anchor not in algorithms:
        raise ValueError(f"Anchor algorithm {anchor!r} is absent.")
    comparators = [algorithm for algorithm in algorithms if algorithm != anchor]
    if not comparators:
        raise ValueError("At least one comparator is required.")

    index: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["case"], row["algorithm"], int(row["seed"]))
        if key in index:
            raise ValueError(f"Duplicate matched row: {key}")
        index[key] = row
    cases = sorted({row["case"] for row in rows})
    if len(cases) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} cases; found {len(cases)}."
        )
    seed_ids = sorted({int(row["seed"]) for row in rows})
    if len(seed_ids) != expected_seeds:
        raise ValueError(
            f"Expected {expected_seeds} distinct seed IDs; found "
            f"{len(seed_ids)}: {seed_ids}."
        )
    expected_seed_set = set(seed_ids)
    matrix_errors = []
    for case in cases:
        for algorithm in algorithms:
            actual_seed_set = {
                seed
                for indexed_case, indexed_algorithm, seed in index
                if indexed_case == case and indexed_algorithm == algorithm
            }
            if actual_seed_set != expected_seed_set:
                matrix_errors.append(
                    f"case={case}, algorithm={algorithm}, "
                    f"missing={sorted(expected_seed_set - actual_seed_set)}, "
                    f"extra={sorted(actual_seed_set - expected_seed_set)}"
                )
    if matrix_errors:
        raise ValueError(
            "Matched case x algorithm x seed matrix is incomplete: "
            + "; ".join(matrix_errors)
        )
    expected_pairs = expected_cases * len(seed_ids)
    results: list[dict[str, object]] = []
    base_rng = random.Random(random_seed)
    for comparator in comparators:
        for metric_name, (column, direction) in METRICS.items():
            deltas = []
            by_case: dict[str, list[float]] = defaultdict(list)
            for case in cases:
                for seed in seed_ids:
                    anchor_row = index.get((case, anchor, seed))
                    comparator_row = index.get((case, comparator, seed))
                    if anchor_row is None or comparator_row is None:
                        raise ValueError(
                            "Missing matched row for "
                            f"case={case}, seed={seed}, comparator={comparator}."
                        )
                    value = _advantage(
                        float(anchor_row[column]),
                        float(comparator_row[column]),
                        direction,
                    )
                    deltas.append(value)
                    by_case[case].append(value)
            if len(deltas) != expected_pairs:
                raise AssertionError("Matched-pair count drifted unexpectedly.")
            tolerance = 1e-12
            wins = sum(value > tolerance for value in deltas)
            losses = sum(value < -tolerance for value in deltas)
            ties = len(deltas) - wins - losses
            rng_bootstrap = random.Random(base_rng.randrange(1 << 63))
            ci_low, ci_high = _cluster_bootstrap_ci(
                by_case,
                repetitions=bootstrap_repetitions,
                rng=rng_bootstrap,
            )
            rng_randomization = random.Random(base_rng.randrange(1 << 63))
            p_value, p_method = _cluster_randomization_p(
                by_case,
                repetitions=randomization_repetitions,
                rng=rng_randomization,
            )
            results.append(
                {
                    "comparator": comparator,
                    "metric": metric_name,
                    "advantage_direction": (
                        "positive_values_favor_anchor"
                    ),
                    "pairs": len(deltas),
                    "case_clusters": len(by_case),
                    "mean_advantage": sum(deltas) / len(deltas),
                    "median_advantage": median(deltas),
                    "cluster_bootstrap_ci95": [ci_low, ci_high],
                    "trimmed_mean_10pct": _trimmed_mean(deltas),
                    "winsorized_mean_10pct": _winsorized_mean(deltas),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "cluster_randomization_p_two_sided": p_value,
                    "cluster_randomization_method": p_method,
                }
            )

    by_key = {
        (str(item["comparator"]), str(item["metric"])): item
        for item in results
    }
    comparator_gates = []
    for comparator in comparators:
        hv = by_key[(comparator, "relative_final_hv")]
        auc = by_key[(comparator, "relative_eval_auc")]
        igd = by_key[(comparator, "igd_plus")]
        conditions = {
            "final_hv_ci_strictly_positive": (
                hv["cluster_bootstrap_ci95"][0] > 0.0  # type: ignore[index]
            ),
            "eval_auc_ci_strictly_positive": (
                auc["cluster_bootstrap_ci95"][0] > 0.0  # type: ignore[index]
            ),
            "final_hv_randomization_p_at_most_0_05": (
                hv["cluster_randomization_p_two_sided"] <= 0.05
            ),
            "eval_auc_randomization_p_at_most_0_05": (
                auc["cluster_randomization_p_two_sided"] <= 0.05
            ),
            "final_hv_wins_exceed_losses": hv["wins"] > hv["losses"],
            "eval_auc_wins_exceed_losses": auc["wins"] > auc["losses"],
            "final_hv_robust_means_positive": (
                hv["trimmed_mean_10pct"] > 0.0
                and hv["winsorized_mean_10pct"] > 0.0
            ),
            "eval_auc_robust_means_positive": (
                auc["trimmed_mean_10pct"] > 0.0
                and auc["winsorized_mean_10pct"] > 0.0
            ),
            "igd_plus_noninferior": (
                igd["cluster_bootstrap_ci95"][0]  # type: ignore[index]
                >= -igd_noninferiority_margin
            ),
        }
        comparator_gates.append(
            {
                "comparator": comparator,
                "conditions": conditions,
                "verdict": (
                    "PASS" if all(conditions.values()) else "FAIL"
                ),
            }
        )
    return {
        "schema": "pareto_smc_strict_paired_analysis_v1",
        "aggregate_csv": str(aggregate_csv.resolve()),
        "anchor": anchor,
        "algorithms": algorithms,
        "cases": len(cases),
        "seeds_per_case": len(seed_ids),
        "seed_ids": seed_ids,
        "matched_pairs_per_comparator": expected_pairs,
        "bootstrap": {
            "unit": "case_cluster",
            "repetitions": bootstrap_repetitions,
            "seed": random_seed,
        },
        "randomization": {
            "unit": "case_cluster_mean",
            "repetitions_if_monte_carlo": randomization_repetitions,
        },
        "igd_plus_noninferiority_margin": igd_noninferiority_margin,
        "metric_results": results,
        "comparator_gates": comparator_gates,
        "overall_adoption_verdict": (
            "ADOPT"
            if all(item["verdict"] == "PASS" for item in comparator_gates)
            else "REJECT"
        ),
        "claim_limit": (
            "This evaluates paired empirical effectiveness only; it does not "
            "instantiate the finite-particle coverage theorem."
        ),
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Pareto-SMC strict paired analysis",
        "",
        f"Anchor: `{report['anchor']}`. Positive advantages favor the anchor.",
        "",
        "| comparator | metric | pairs | mean | CI95 | median | trim10 | winsor10 | W/T/L | cluster p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["metric_results"]:  # type: ignore[union-attr]
        low, high = item["cluster_bootstrap_ci95"]
        lines.append(
            "| {comparator} | {metric} | {pairs} | {mean_advantage:.6g} | "
            "[{low:.6g}, {high:.6g}] | {median_advantage:.6g} | "
            "{trimmed_mean_10pct:.6g} | {winsorized_mean_10pct:.6g} | "
            "{wins}/{ties}/{losses} | {p:.6g} |".format(
                low=low,
                high=high,
                p=item["cluster_randomization_p_two_sided"],
                **item,
            )
        )
    lines.extend(
        [
            "",
            "## Adoption gate",
            "",
            f"Overall: **{report['overall_adoption_verdict']}**",
            "",
        ]
    )
    for gate in report["comparator_gates"]:  # type: ignore[union-attr]
        lines.append(f"- `{gate['comparator']}`: **{gate['verdict']}**")
        for name, passed in gate["conditions"].items():
            lines.append(f"  - {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", str(report["claim_limit"]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--anchor", default="annealed-pareto-smc")
    parser.add_argument("--expected-cases", type=int, default=35)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--randomization-repetitions", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=20260726)
    parser.add_argument("--igd-noninferiority-margin", type=float, default=0.01)
    args = parser.parse_args()
    report = analyze(
        args.aggregate,
        anchor=args.anchor,
        expected_cases=args.expected_cases,
        expected_seeds=args.expected_seeds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        randomization_repetitions=args.randomization_repetitions,
        random_seed=args.random_seed,
        igd_noninferiority_margin=args.igd_noninferiority_margin,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(f"ADOPTION {report['overall_adoption_verdict']}")
    print(f"JSON {args.output_json.resolve()}")
    print(f"MARKDOWN {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
