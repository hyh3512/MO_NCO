from __future__ import annotations

"""Frozen development-only parity metrics and case-cluster analysis.

This module deliberately does not authorize selection, calibration,
confirmation, or formal evidence.  It supplies the metric and statistical
objects needed by the prospective V21e3r1 C0 parity matrix.
"""

import hashlib
import math
import random
import statistics
from typing import Mapping, Sequence


_ARMS = ("V21E3_C0", "NSGAII", "MOEAD")
_COMPARATORS = ("NSGAII", "MOEAD")
_FAMILIES = ("MOTSP", "MOKP")
_SIZES = (100, 200, 500)


def normalized_hypervolume_2d(
    front: Sequence[Sequence[float]],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    if len(lower) != 2 or len(upper) != 2:
        raise ValueError("V21e3r1 normalized HV requires two objectives.")
    spans = tuple(float(hi) - float(lo) for lo, hi in zip(lower, upper))
    if any(not math.isfinite(span) or span <= 0.0 for span in spans):
        raise ValueError("Every analytic objective span must be finite and positive.")
    points: list[tuple[float, float]] = []
    for objective in front:
        if len(objective) != 2:
            raise ValueError("Every V21e3r1 HV point must be two-dimensional.")
        point = tuple(
            (float(value) - float(lo)) / span
            for value, lo, span in zip(objective, lower, spans)
        )
        if any(not math.isfinite(value) for value in point):
            raise ValueError("A normalized objective is not finite.")
        if any(value < -1e-12 or value > 1.0 + 1e-12 for value in point):
            raise ValueError("An objective lies outside the frozen analytic box.")
        points.append((float(point[0]), float(point[1])))
    unique = sorted(set(points))
    nondominated = [
        point
        for point in unique
        if not any(
            other != point
            and other[0] <= point[0]
            and other[1] <= point[1]
            for other in unique
        )
    ]
    hypervolume = 0.0
    best_y = 1.0
    for x_value, y_value in sorted(nondominated):
        if y_value < best_y:
            hypervolume += (1.0 - x_value) * (best_y - y_value)
            best_y = y_value
    if not -1e-12 <= hypervolume <= 1.0 + 1e-12:
        raise RuntimeError("Normalized hypervolume escaped [0, 1].")
    return min(1.0, max(0.0, hypervolume))


def normalized_left_continuous_auc(
    diagnostics: Sequence[object],
    *,
    budget: int,
    checkpoint_period: int,
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[float, float, tuple[dict[str, object], ...]]:
    """Return normalized left-continuous HV-AUC and checkpoint witnesses."""

    if budget <= 0 or checkpoint_period <= 0 or budget % checkpoint_period != 0:
        raise ValueError("The budget must be a positive multiple of checkpoint_period.")
    expected_grid = tuple(range(checkpoint_period, budget + 1, checkpoint_period))
    if len(diagnostics) != len(expected_grid):
        raise RuntimeError("Diagnostics omit a frozen common checkpoint.")
    previous_evaluation = 0
    previous_hv = 0.0
    area = 0.0
    checkpoints: list[dict[str, object]] = []
    for expected, diagnostic in zip(expected_grid, diagnostics):
        evaluation = int(getattr(diagnostic, "iteration"))
        if evaluation != expected:
            raise RuntimeError("Diagnostics do not match the frozen common grid.")
        area += previous_hv * (evaluation - previous_evaluation)
        current_hv = normalized_hypervolume_2d(
            getattr(diagnostic, "front"), lower=lower, upper=upper
        )
        checkpoints.append(
            {
                "evaluation": evaluation,
                "normalized_hv": current_hv,
                "archive_size": int(getattr(diagnostic, "archive_size")),
            }
        )
        previous_evaluation = evaluation
        previous_hv = current_hv
    return area / float(budget), previous_hv, tuple(checkpoints)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("A percentile requires at least one value.")
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _trimmed_mean(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    trim = int(math.floor(fraction * len(ordered)))
    retained = ordered[trim : len(ordered) - trim if trim else len(ordered)]
    if not retained:
        raise ValueError("The trim fraction removes every case cluster.")
    return statistics.fmean(retained)


def _bootstrap_ci(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    rng = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _exact_sign_flip(differences: Sequence[float]) -> dict[str, object]:
    count = len(differences)
    if count > 20:
        raise ValueError("Exact sign-flip enumeration is capped at 20 clusters.")
    observed = abs(statistics.fmean(differences))
    tolerance = 1e-15
    extreme = 0
    total = 1 << count
    for mask in range(total):
        estimate = statistics.fmean(
            value if mask & (1 << index) else -value
            for index, value in enumerate(differences)
        )
        if abs(estimate) + tolerance >= observed:
            extreme += 1
    return {
        "method": "exact_cluster_sign_flip",
        "alternative": "two_sided",
        "randomizations": total,
        "two_sided_p": extreme / total,
    }


def _domain_seed(base_seed: int, family: str, comparator: str) -> int:
    raw = f"v21e3r1-parity|{base_seed}|{family}|{comparator}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def analyze_development_parity(
    rows: Sequence[Mapping[str, object]],
    *,
    case_records: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
    margin: float = 0.005,
    size_stratum_margin: float = 0.010,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 31_061,
    tie_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Analyze one complete prospective development matched matrix."""

    if margin <= 0.0 or size_stratum_margin <= 0.0:
        raise ValueError("Noninferiority margins must be positive.")
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if not frozen_seeds or len(set(frozen_seeds)) != len(frozen_seeds):
        raise ValueError("Seeds must be a nonempty unique sequence.")
    case_by_id: dict[str, Mapping[str, object]] = {}
    for case in case_records:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in case_by_id:
            raise ValueError("Case records contain a duplicate or empty case_id.")
        family = str(case.get("family"))
        size = int(case.get("size", -1))
        if family not in _FAMILIES or size not in _SIZES:
            raise ValueError("Case records leave the frozen family/size design.")
        case_by_id[case_id] = case
    expected_case_keys = {
        (family, size): 2 for family in _FAMILIES for size in _SIZES
    }
    observed_case_keys = {key: 0 for key in expected_case_keys}
    for case in case_by_id.values():
        observed_case_keys[(str(case["family"]), int(case["size"]))] += 1
    if observed_case_keys != expected_case_keys:
        raise ValueError("Case records do not match the frozen 12-case design.")

    expected_keys = {
        (case_id, seed, arm_id)
        for case_id in case_by_id
        for seed in frozen_seeds
        for arm_id in _ARMS
    }
    by_key: dict[tuple[str, int, str], float] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        seed_value = row.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise ValueError("Every matrix seed must be an integer.")
        seed = int(seed_value)
        arm_id = str(row.get("arm_id", ""))
        key = (case_id, seed, arm_id)
        if key in by_key:
            raise ValueError("The matrix contains a duplicate case-seed-arm row.")
        if key not in expected_keys:
            raise ValueError("A matrix row lies outside the frozen matched product.")
        if str(row.get("family")) != str(case_by_id[case_id]["family"]):
            raise ValueError("A matrix row carries the wrong family.")
        if int(row.get("size", -1)) != int(case_by_id[case_id]["size"]):
            raise ValueError("A matrix row carries the wrong size.")
        raw_value = row.get("normalized_left_continuous_hv_auc")
        if type(raw_value) not in {int, float}:
            raise ValueError("Every matrix metric must be a JSON number.")
        value = float(raw_value)
        if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
            raise ValueError("A matrix metric lies outside [0, 1].")
        by_key[key] = min(1.0, max(0.0, value))
    if set(by_key) != expected_keys:
        raise ValueError("The matrix is not the complete matched case-seed-arm product.")

    comparisons: dict[str, dict[str, object]] = {}
    all_pass = True
    for family in _FAMILIES:
        family_cases = sorted(
            case_id
            for case_id, case in case_by_id.items()
            if str(case["family"]) == family
        )
        family_results: dict[str, object] = {}
        for comparator in _COMPARATORS:
            clusters: list[dict[str, object]] = []
            differences: list[float] = []
            for case_id in family_cases:
                c0_mean = statistics.fmean(
                    by_key[(case_id, seed, "V21E3_C0")] for seed in frozen_seeds
                )
                comparator_mean = statistics.fmean(
                    by_key[(case_id, seed, comparator)] for seed in frozen_seeds
                )
                difference = c0_mean - comparator_mean
                differences.append(difference)
                clusters.append(
                    {
                        "case_id": case_id,
                        "size": int(case_by_id[case_id]["size"]),
                        "replicates_per_arm": len(frozen_seeds),
                        "c0_mean": c0_mean,
                        "comparator_mean": comparator_mean,
                        "difference": difference,
                    }
                )
            lower, upper = _bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                seed=_domain_seed(bootstrap_seed, family, comparator),
            )
            median = statistics.median(differences)
            trimmed = _trimmed_mean(differences, 0.10)
            size_means = {
                str(size): statistics.fmean(
                    cluster["difference"]
                    for cluster in clusters
                    if cluster["size"] == size
                )
                for size in _SIZES
            }
            wins = sum(value > tie_tolerance for value in differences)
            losses = sum(value < -tie_tolerance for value in differences)
            ties = len(differences) - wins - losses
            checks = {
                "ci95_lower_at_least_negative_margin": lower >= -margin,
                "median_strictly_above_negative_margin": median > -margin,
                "trimmed_mean_strictly_above_negative_margin": trimmed > -margin,
                "all_size_strata_at_least_negative_margin": all(
                    value >= -size_stratum_margin for value in size_means.values()
                ),
            }
            passed = all(checks.values())
            all_pass = all_pass and passed
            family_results[comparator] = {
                "schema": "pareto_v21e3r1_case_cluster_comparison_v1",
                "treatment_arm": "V21E3_C0",
                "comparator_arm": comparator,
                "effect_direction": "C0_minus_comparator",
                "cluster_count": len(differences),
                "replicate_aggregation": "arithmetic_mean_within_case_and_arm",
                "mean_difference": statistics.fmean(differences),
                "median_difference": median,
                "trimmed_mean_difference": trimmed,
                "trim_fraction_each_tail": 0.10,
                "size_stratum_mean_differences": size_means,
                "cluster_bootstrap_ci95": {
                    "method": "paired_case_cluster_percentile_bootstrap",
                    "samples": bootstrap_samples,
                    "base_randomization_seed": bootstrap_seed,
                    "domain_randomization_seed": _domain_seed(
                        bootstrap_seed, family, comparator
                    ),
                    "lower": lower,
                    "upper": upper,
                },
                "wins_ties_losses": {
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                },
                "tie_tolerance": tie_tolerance,
                "sign_flip_test": _exact_sign_flip(differences),
                "noninferiority_margin": margin,
                "size_stratum_margin": size_stratum_margin,
                "checks": checks,
                "paired_clusters": clusters,
                "gate": "PASS" if passed else "FAIL",
            }
        comparisons[family] = family_results

    return {
        "schema": "pareto_v21e3r1_development_parity_analysis_v1",
        "status": "COMPLETE_DEVELOPMENT_EVIDENCE",
        "scientific_scope": "authors_generated_development_only_not_formal_evidence",
        "completeness_gate": "PASS",
        "expected_rows": len(expected_keys),
        "observed_rows": len(by_key),
        "case_count": len(case_by_id),
        "seeds": list(frozen_seeds),
        "arms": list(_ARMS),
        "comparisons": comparisons,
        "overall_gate": (
            "PASS_DEVELOPMENT_NONINFERIORITY"
            if all_pass
            else "FAIL_STOP_BEFORE_SELECTION_PARTITION_MATERIALIZATION"
        ),
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }


__all__ = [
    "analyze_development_parity",
    "normalized_hypervolume_2d",
    "normalized_left_continuous_auc",
]
