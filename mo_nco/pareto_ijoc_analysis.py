from __future__ import annotations

"""Independent metric and paired-inference audit for the IJOC formal matrix.

The module deliberately ignores metrics emitted by algorithm adapters.  It
recomputes every quality metric from the reported, replay-bound
archive/checkpoint witnesses and the frozen per-case reference manifest.  The
resulting estimand is reported-archive-relative; the V20 artifacts do not carry
an independently replayable event trace for every charged evaluation.
"""

import math
import hashlib
import csv
import io
import json
import os
import random
import shutil
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import additive_epsilon, igd_plus


QUALITY_METRIC_ORIENTATIONS = {
    "normalized_left_continuous_hypervolume_auc": "higher",
    "normalized_final_hypervolume": "higher",
    "igd_plus_to_frozen_supplied_reference": "lower",
    "additive_epsilon_to_frozen_supplied_reference": "lower",
}

POST_RUN_AUDIT_SCHEMA = "ijoc_post_run_audit_v2"
REPORTED_ARCHIVE_EVIDENCE_STATUS = (
    "REPORTED_ARCHIVE_MATRIX_INTEGRITY_ESTABLISHED"
)
QUALITY_POSTRUN_GATES = (
    "frozen_preflight_gate",
    "full_invocation_gate",
    "complete_row_set_gate",
    "terminal_success_gate",
    "budget_checkpoint_gate",
    "hash_binding_gate",
    "reported_archive_witness_self_consistency_gate",
    "attempt_history_enumeration_gate",
    "retry_quality_selection_gate",
    "frozen_command_gate",
)

METRIC_ORIENTATIONS = {
    **QUALITY_METRIC_ORIENTATIONS,
    "wall_time_seconds": "lower",
    "sampled_peak_process_tree_rss_bytes": "lower",
}


@dataclass(frozen=True)
class IJOCFormalAnalysisResult:
    output_directory: Path
    audit_path: Path
    row_count: int
    formal_metric_statistical_gate: str
    primary_superiority_gate: str
    efficiency_claim_gate: str


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _read_strict_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"Required {label} is missing: {path}.")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object.")
    return value, hashlib.sha256(raw).hexdigest()


def analyze_ijoc_formal_results(
    study_path: str | Path,
    execution_plan_path: str | Path,
    results_directory: str | Path,
    post_run_audit_path: str | Path,
    output_directory: str | Path,
) -> IJOCFormalAnalysisResult:
    """Audit and analyze a completed frozen matrix.

    The post-run integrity gate is deliberately checked before any result or
    study artifact is consumed.
    """

    postrun, _ = _read_strict_json(
        Path(post_run_audit_path).expanduser().resolve(),
        label="post-run audit",
    )
    if (
        postrun.get("schema") != POST_RUN_AUDIT_SCHEMA
        or postrun.get("formal_matched_matrix_gate") != "PASS"
        or postrun.get("evidence_status")
        != REPORTED_ARCHIVE_EVIDENCE_STATUS
        or postrun.get("quality_estimand_scope")
        != "reported_archive_relative"
        or postrun.get("all_evaluated_archive_claim_status")
        != "NOT_ESTABLISHED"
    ):
        raise ValueError(
            "Formal metric analysis requires a post-run quality gate PASS "
            "with reported-archive matrix-integrity evidence."
        )
    return _analyze_passing_matrix(
        study_path=Path(study_path).expanduser().resolve(),
        execution_plan_path=Path(execution_plan_path).expanduser().resolve(),
        results_directory=Path(results_directory).expanduser().resolve(),
        post_run_audit_path=Path(post_run_audit_path).expanduser().resolve(),
        postrun=postrun,
        output_directory=Path(output_directory).expanduser().resolve(),
    )


def _finite_vector(
    value: object,
    *,
    dimension: int | None,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty numeric vector.")
    vector: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(
            coordinate, (int, float)
        ):
            raise ValueError(f"{label} contains a nonnumeric coordinate.")
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError(f"{label} contains a nonfinite coordinate.")
        vector.append(number)
    if dimension is not None and len(vector) != dimension:
        raise ValueError(f"{label} has the wrong objective dimension.")
    return tuple(vector)


def _entry_objectives(
    entries: object,
    *,
    dimension: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError(f"{label} must contain at least one archive entry.")
    points: list[tuple[float, ...]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{label}[{index}] must be an object.")
        points.append(
            _finite_vector(
                entry.get("objectives"),
                dimension=dimension,
                label=f"{label}[{index}].objectives",
            )
        )
    return tuple(points)


def _nondominated_2d(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    unique = sorted(set(points), key=lambda point: (point[0], point[1]))
    frontier: list[tuple[float, float]] = []
    best_second = math.inf
    for first, second in unique:
        if second < best_second:
            frontier.append((first, second))
            best_second = second
    return tuple(frontier)


def _hypervolume_2d(
    points: Sequence[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    frontier = _nondominated_2d(points)
    previous_second = reference[1]
    hypervolume = 0.0
    for first, second in frontier:
        hypervolume += max(0.0, reference[0] - first) * max(
            0.0, previous_second - second
        )
        previous_second = min(previous_second, second)
    return hypervolume


def recompute_quality_metrics(
    *,
    final_entries: object,
    checkpoints: object,
    budget: int,
    checkpoint_period: int,
    reference: Mapping[str, Any],
) -> dict[str, float]:
    """Recompute the four frozen-reference biobjective quality metrics."""

    if (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
        or isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
        or budget % checkpoint_period
    ):
        raise ValueError("Budget/checkpoint period is not a positive exact grid.")
    ideal = _finite_vector(
        reference.get("ideal"), dimension=None, label="reference.ideal"
    )
    if len(ideal) != 2:
        raise ValueError("The IJOC metric audit currently requires two objectives.")
    nadir = _finite_vector(
        reference.get("nadir"), dimension=2, label="reference.nadir"
    )
    hv_reference = _finite_vector(
        reference.get("hv_reference"),
        dimension=2,
        label="reference.hv_reference",
    )
    if any(not lower < upper for lower, upper in zip(ideal, nadir)):
        raise ValueError("Frozen ideal/nadir ranges must be strictly positive.")
    if any(bound < upper for bound, upper in zip(hv_reference, nadir)):
        raise ValueError("Frozen HV reference must weakly dominate the nadir.")
    reference_points = tuple(
        _finite_vector(point, dimension=2, label="reference.reference_points")
        for point in reference.get("reference_points", [])
    )
    if not reference_points:
        raise ValueError("Frozen supplied reference front is empty.")

    scale = tuple(upper - lower for lower, upper in zip(ideal, nadir))

    def normalize(
        points: Sequence[tuple[float, ...]], *, label: str
    ) -> tuple[tuple[float, float], ...]:
        normalized: list[tuple[float, float]] = []
        for point in points:
            if any(
                value < lower - 1e-12 or value > upper + 1e-12
                for value, lower, upper in zip(point, ideal, nadir)
            ):
                raise ValueError(f"{label} leaves the frozen ideal/nadir box.")
            normalized.append(
                (
                    (point[0] - ideal[0]) / scale[0],
                    (point[1] - ideal[1]) / scale[1],
                )
            )
        return tuple(normalized)

    normalized_reference = normalize(
        reference_points, label="Frozen supplied reference point"
    )
    normalized_hv_reference = (
        (hv_reference[0] - ideal[0]) / scale[0],
        (hv_reference[1] - ideal[1]) / scale[1],
    )
    final_points = _entry_objectives(
        final_entries, dimension=2, label="final_entries"
    )
    normalized_final = normalize(final_points, label="Final archive point")

    if not isinstance(checkpoints, (list, tuple)):
        raise ValueError("Checkpoint witnesses must be an array.")
    expected_grid = list(range(checkpoint_period, budget + 1, checkpoint_period))
    observed_grid: list[int] = []
    checkpoint_hypervolumes: list[float] = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoints[{index}] must be an object.")
        evaluation = checkpoint.get("evaluation")
        if isinstance(evaluation, bool) or not isinstance(evaluation, int):
            raise ValueError("Checkpoint evaluation must be an integer.")
        observed_grid.append(evaluation)
        raw_points = _entry_objectives(
            checkpoint.get("entries"),
            dimension=2,
            label=f"checkpoints[{index}].entries",
        )
        normalized_points = normalize(
            raw_points, label=f"Checkpoint {evaluation} point"
        )
        checkpoint_hypervolumes.append(
            _hypervolume_2d(normalized_points, normalized_hv_reference)
        )
    if observed_grid != expected_grid:
        raise ValueError("Checkpoint witnesses do not cover the exact frozen grid.")

    area = 0.0
    previous_evaluation = 0
    previous_hypervolume = 0.0
    for evaluation, hypervolume in zip(
        observed_grid, checkpoint_hypervolumes
    ):
        area += previous_hypervolume * (evaluation - previous_evaluation)
        previous_evaluation = evaluation
        previous_hypervolume = hypervolume

    return {
        "normalized_left_continuous_hypervolume_auc": area / budget,
        "normalized_final_hypervolume": _hypervolume_2d(
            normalized_final, normalized_hv_reference
        ),
        "igd_plus_to_frozen_supplied_reference": igd_plus(
            normalized_final, normalized_reference
        ),
        "additive_epsilon_to_frozen_supplied_reference": additive_epsilon(
            normalized_final, normalized_reference
        ),
    }


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must be a nonempty unique string array.")
    return tuple(value)


def _integer_list(value: object, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must be a nonempty unique integer array.")
    return tuple(value)


def _numeric_metric(row: Mapping[str, Any], metric: str) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Every row must contain a metrics object.")
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric {metric!r} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Metric {metric!r} must be finite.")
    if metric in (
        "wall_time_seconds",
        "sampled_peak_process_tree_rss_bytes",
    ) and number <= 0.0:
        raise ValueError(f"Resource metric {metric!r} must be positive.")
    return number


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("A quantile requires at least one value.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Quantile probability must be in [0, 1].")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _derived_seed(base_seed: int, *labels: object) -> int:
    payload = json.dumps(
        [base_seed, *labels],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _case_cluster_bootstrap_ci(
    case_values: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive.")
    cases = sorted(case_values)
    if not cases or any(not case_values[case] for case in cases):
        raise ValueError("Every bootstrap case cluster must be nonempty.")
    means = {
        case: statistics.fmean(case_values[case]) for case in cases
    }
    rng = random.Random(seed)
    samples = [
        statistics.fmean(means[cases[rng.randrange(len(cases))]]
                         for _ in cases)
        for _ in range(replicates)
    ]
    alpha = 1.0 - confidence_level
    return (
        _quantile(samples, alpha / 2.0),
        _quantile(samples, 1.0 - alpha / 2.0),
    )


def _exact_case_cluster_sign_flip(
    case_values: Mapping[str, Sequence[float]],
) -> float:
    case_means = tuple(
        statistics.fmean(case_values[case]) for case in sorted(case_values)
    )
    if not case_means:
        raise ValueError("The exact sign-flip test requires case clusters.")
    # The precommitted V20 formal matrix has 15 cases per family (2^15
    # assignments).  Refuse an accidentally exponential, non-precommitted
    # workload instead of silently changing to Monte Carlo.
    if len(case_means) > 24:
        raise ValueError(
            "Exact case-cluster sign-flip enumeration is limited to 24 "
            "clusters; changing the test to Monte Carlo is forbidden."
        )
    observed = abs(statistics.fmean(case_means))
    total = 1 << len(case_means)
    exceed = 0
    for mask in range(total):
        randomized = math.fsum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(case_means)
        ) / len(case_means)
        if abs(randomized) >= observed - 1e-15:
            exceed += 1
    return exceed / total


def _holm_adjust(
    records: Sequence[dict[str, Any]],
    *,
    alpha: float,
) -> None:
    ordered = sorted(
        records,
        key=lambda item: (
            float(item["exact_case_cluster_sign_flip_p_value"]),
            str(item["family"]),
            str(item["baseline"]),
        ),
    )
    running = 0.0
    count = len(ordered)
    for rank, item in enumerate(ordered, start=1):
        raw = float(item["exact_case_cluster_sign_flip_p_value"])
        running = max(running, (count - rank + 1) * raw)
        adjusted = min(1.0, running)
        item["holm_rank"] = rank
        item["holm_adjusted_p_value"] = adjusted
        item["holm_reject_at_familywise_alpha"] = adjusted <= alpha


def build_paired_inference(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build all family/budget/baseline paired comparisons without pooling."""

    if not rows:
        raise ValueError("Row metrics are empty.")
    families = _string_list(plan.get("families"), label="plan.families")
    treatment = plan.get("treatment")
    if not isinstance(treatment, str) or not treatment:
        raise ValueError("plan.treatment must be nonempty.")
    baselines_raw = plan.get("required_baselines")
    if not isinstance(baselines_raw, dict) or set(baselines_raw) != set(
        families
    ):
        raise ValueError("Plan baseline families do not match plan.families.")
    baselines = {
        family: _string_list(
            baselines_raw[family],
            label=f"plan.required_baselines.{family}",
        )
        for family in families
    }
    seeds = _integer_list(
        plan.get("formal_seeds"), label="plan.formal_seeds"
    )
    budgets = _integer_list(
        plan.get("evaluation_budgets"), label="plan.evaluation_budgets"
    )
    primary_budget = plan.get("primary_budget")
    if (
        isinstance(primary_budget, bool)
        or not isinstance(primary_budget, int)
        or primary_budget not in budgets
    ):
        raise ValueError("plan.primary_budget must be a frozen budget.")
    uncertainty = plan.get("uncertainty")
    if not isinstance(uncertainty, dict):
        raise ValueError("plan.uncertainty must be an object.")
    confidence_level = float(uncertainty.get("confidence_level", math.nan))
    bootstrap_replicates = uncertainty.get(
        "case_cluster_bootstrap_replicates"
    )
    bootstrap_seed = uncertainty.get("bootstrap_seed")
    familywise_alpha = float(
        uncertainty.get("familywise_alpha", math.nan)
    )
    if (
        not 0.0 < confidence_level < 1.0
        or isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or not 0.0 < familywise_alpha < 1.0
        or uncertainty.get("randomization_test")
        != "exact_two_sided_case_cluster_sign_flip"
        or uncertainty.get("multiplicity")
        != "holm_across_six_primary_family_by_baseline_comparisons"
    ):
        raise ValueError("The frozen uncertainty contract is unsupported.")
    tie_contract = plan.get("wins_ties_losses")
    if not isinstance(tie_contract, dict):
        raise ValueError("plan.wins_ties_losses must be an object.")
    tie_tolerance = tie_contract.get(
        "normalized_metric_absolute_tie_tolerance"
    )
    if (
        isinstance(tie_tolerance, bool)
        or not isinstance(tie_tolerance, (int, float))
        or not math.isfinite(float(tie_tolerance))
        or float(tie_tolerance) < 0.0
    ):
        raise ValueError("The W/T/L tie tolerance is invalid.")
    tolerance = float(tie_tolerance)

    index: dict[tuple[str, str, str, int, int], Mapping[str, Any]] = {}
    cases_by_family: dict[str, set[str]] = defaultdict(set)
    allowed_algorithms = {
        family: {treatment, *baselines[family]} for family in families
    }
    for row in rows:
        family = row.get("family")
        case_id = row.get("case_id")
        algorithm = row.get("algorithm")
        seed = row.get("seed")
        budget = row.get("budget")
        if (
            not isinstance(family, str)
            or family not in families
            or not isinstance(case_id, str)
            or not case_id
            or not isinstance(algorithm, str)
            or algorithm not in allowed_algorithms[family]
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in seeds
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget not in budgets
        ):
            raise ValueError("A row lies outside the frozen matched matrix.")
        key = (family, case_id, algorithm, seed, budget)
        if key in index:
            raise ValueError(f"Duplicate row metric key: {key!r}.")
        for metric in METRIC_ORIENTATIONS:
            _numeric_metric(row, metric)
        index[key] = row
        cases_by_family[family].add(case_id)
    if set(cases_by_family) != set(families):
        raise ValueError("Row metrics do not cover every frozen family.")
    expected = {
        (family, case, algorithm, seed, budget)
        for family in families
        for case in cases_by_family[family]
        for algorithm in allowed_algorithms[family]
        for seed in seeds
        for budget in budgets
    }
    if set(index) != expected:
        missing = sorted(expected - set(index))
        extra = sorted(set(index) - expected)
        raise ValueError(
            "Paired row matrix is incomplete or unexpected; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}."
        )

    comparisons: list[dict[str, Any]] = []
    for family in families:
        for budget in budgets:
            for baseline in baselines[family]:
                for metric, orientation in METRIC_ORIENTATIONS.items():
                    deltas: list[float] = []
                    ratios: list[float] = []
                    by_case: dict[str, list[float]] = defaultdict(list)
                    ratio_by_case: dict[str, list[float]] = defaultdict(list)
                    for case_id in sorted(cases_by_family[family]):
                        for seed in seeds:
                            treatment_value = _numeric_metric(
                                index[
                                    (
                                        family,
                                        case_id,
                                        treatment,
                                        seed,
                                        budget,
                                    )
                                ],
                                metric,
                            )
                            baseline_value = _numeric_metric(
                                index[
                                    (
                                        family,
                                        case_id,
                                        baseline,
                                        seed,
                                        budget,
                                    )
                                ],
                                metric,
                            )
                            delta = (
                                treatment_value - baseline_value
                                if orientation == "higher"
                                else baseline_value - treatment_value
                            )
                            deltas.append(delta)
                            by_case[case_id].append(delta)
                            if metric in (
                                "wall_time_seconds",
                                "sampled_peak_process_tree_rss_bytes",
                            ):
                                ratio = treatment_value / baseline_value
                                ratios.append(ratio)
                                ratio_by_case[case_id].append(ratio)
                    effective_seed = _derived_seed(
                        bootstrap_seed,
                        family,
                        budget,
                        baseline,
                        metric,
                        "advantage",
                    )
                    ci = _case_cluster_bootstrap_ci(
                        by_case,
                        replicates=bootstrap_replicates,
                        seed=effective_seed,
                        confidence_level=confidence_level,
                    )
                    wins = sum(value > tolerance for value in deltas)
                    losses = sum(value < -tolerance for value in deltas)
                    item: dict[str, Any] = {
                        "family": family,
                        "budget": budget,
                        "baseline": baseline,
                        "metric": metric,
                        "orientation": orientation,
                        "paired_contrast": (
                            "positive_values_always_favor_treatment"
                        ),
                        "paired_case_seed_count": len(deltas),
                        "case_cluster_count": len(by_case),
                        "case_cluster_mean_advantage": statistics.fmean(
                            statistics.fmean(by_case[case])
                            for case in sorted(by_case)
                        ),
                        "case_cluster_bootstrap_ci95": list(ci),
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_effective_seed": effective_seed,
                        "paired_wins_ties_losses": {
                            "wins": wins,
                            "ties": len(deltas) - wins - losses,
                            "losses": losses,
                        },
                        "tie_tolerance": tolerance,
                        "exact_case_cluster_sign_flip_p_value": (
                            _exact_case_cluster_sign_flip(by_case)
                        ),
                        "randomization_method": (
                            "exact_two_sided_case_cluster_sign_flip"
                        ),
                    }
                    if ratios:
                        ratio_seed = _derived_seed(
                            bootstrap_seed,
                            family,
                            budget,
                            baseline,
                            metric,
                            "ratio",
                        )
                        ratio_ci = _case_cluster_bootstrap_ci(
                            ratio_by_case,
                            replicates=bootstrap_replicates,
                            seed=ratio_seed,
                            confidence_level=confidence_level,
                        )
                        item["treatment_over_baseline_ratio"] = {
                            "case_cluster_mean": statistics.fmean(
                                statistics.fmean(ratio_by_case[case])
                                for case in sorted(ratio_by_case)
                            ),
                            "case_cluster_bootstrap_ci95": list(ratio_ci),
                            "bootstrap_effective_seed": ratio_seed,
                        }
                    comparisons.append(item)

    primary_metric = "normalized_left_continuous_hypervolume_auc"
    primary: list[dict[str, Any]] = []
    for comparison in comparisons:
        if (
            comparison["budget"] == primary_budget
            and comparison["metric"] == primary_metric
        ):
            primary.append(
                {
                    "family": comparison["family"],
                    "baseline": comparison["baseline"],
                    "budget": primary_budget,
                    "metric": primary_metric,
                    "case_cluster_mean_advantage": comparison[
                        "case_cluster_mean_advantage"
                    ],
                    "case_cluster_bootstrap_ci95": comparison[
                        "case_cluster_bootstrap_ci95"
                    ],
                    "paired_wins_ties_losses": comparison[
                        "paired_wins_ties_losses"
                    ],
                    "exact_case_cluster_sign_flip_p_value": comparison[
                        "exact_case_cluster_sign_flip_p_value"
                    ],
                }
            )
    if len(primary) != 6:
        raise ValueError(
            "The frozen Holm family must contain exactly six primary "
            "family-by-baseline comparisons."
        )
    _holm_adjust(primary, alpha=familywise_alpha)

    efficiency = plan.get("efficiency_claim_gate")
    if not isinstance(efficiency, dict):
        raise ValueError("plan.efficiency_claim_gate must be an object.")
    runtime_limit = efficiency.get(
        "maximum_case_cluster_mean_runtime_ratio_ci95_upper"
    )
    if (
        efficiency.get("quality_gate_must_pass") is not True
        or isinstance(runtime_limit, bool)
        or not isinstance(runtime_limit, (int, float))
        or float(runtime_limit) <= 0.0
        or efficiency.get("memory_claim")
        != "reported_without_a_superiority_threshold"
    ):
        raise ValueError("The frozen efficiency-claim contract is unsupported.")
    runtime_index = {
        (
            item["family"],
            item["budget"],
            item["baseline"],
        ): item
        for item in comparisons
        if item["metric"] == "wall_time_seconds"
    }
    for item in primary:
        wtl = item["paired_wins_ties_losses"]
        quality_checks = {
            "auc_delta_ci95_lower_strictly_positive": (
                float(item["case_cluster_bootstrap_ci95"][0]) > 0.0
            ),
            "holm_adjusted_exact_p_at_most_familywise_alpha": (
                float(item["holm_adjusted_p_value"]) <= familywise_alpha
            ),
            "paired_wins_exceed_losses": (
                int(wtl["wins"]) > int(wtl["losses"])
            ),
        }
        item["quality_gate_checks"] = quality_checks
        item["quality_comparison_gate"] = (
            "PASS" if all(quality_checks.values()) else "FAIL"
        )
        runtime = runtime_index[
            (item["family"], primary_budget, item["baseline"])
        ]["treatment_over_baseline_ratio"]
        item["runtime_ratio_treatment_over_baseline"] = runtime
        efficiency_checks = {
            "quality_comparison_gate_pass": (
                item["quality_comparison_gate"] == "PASS"
            ),
            "runtime_ratio_ci95_upper_at_most_limit": (
                float(runtime["case_cluster_bootstrap_ci95"][1])
                <= float(runtime_limit)
            ),
        }
        item["efficiency_gate_checks"] = efficiency_checks
        item["efficiency_comparison_gate"] = (
            "PASS" if all(efficiency_checks.values()) else "FAIL"
        )

    family_gates = []
    for family in families:
        family_primary = [
            item for item in primary if item["family"] == family
        ]
        family_gates.append(
            {
                "family": family,
                "primary_comparison_count": len(family_primary),
                "primary_superiority_gate": (
                    "PASS"
                    if all(
                        item["quality_comparison_gate"] == "PASS"
                        for item in family_primary
                    )
                    else "FAIL"
                ),
                "efficiency_claim_gate": (
                    "PASS"
                    if all(
                        item["efficiency_comparison_gate"] == "PASS"
                        for item in family_primary
                    )
                    else "FAIL"
                ),
            }
        )
    return {
        "schema": "ijoc_formal_paired_inference_v2",
        "treatment": treatment,
        "families": list(families),
        "budgets": list(budgets),
        "primary_budget": primary_budget,
        "metric_orientations": METRIC_ORIENTATIONS,
        "comparison_unit": "same_family_case_seed_budget",
        "cluster_unit": "case_id",
        "family_pooling": "forbidden",
        "budget_pooling": "forbidden",
        "bootstrap": {
            "method": "case_cluster_percentile",
            "confidence_level": confidence_level,
            "replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
        },
        "randomization": {
            "method": "exact_two_sided_case_cluster_sign_flip",
            "monte_carlo_used": False,
        },
        "multiplicity": {
            "method": "Holm",
            "family": (
                "six_primary_family_by_required_baseline_comparisons"
            ),
            "familywise_alpha": familywise_alpha,
        },
        "comparisons": comparisons,
        "primary_comparisons": primary,
        "family_gates": family_gates,
        "primary_superiority_gate": (
            "PASS"
            if all(
                item["primary_superiority_gate"] == "PASS"
                for item in family_gates
            )
            else "FAIL"
        ),
        "efficiency_claim_gate": (
            "PASS"
            if all(
                item["efficiency_claim_gate"] == "PASS"
                for item in family_gates
            )
            else "FAIL"
        ),
        "memory_claim_gate": "NOT_APPLICABLE_REPORTED_WITHOUT_THRESHOLD",
        "reference_scope": (
            "supplied-reference-relative_only_not_true_pareto_front_completeness"
        ),
    }


def _apply_postrun_evidence_scope(
    inference: dict[str, Any],
    *,
    postrun: Mapping[str, Any],
) -> None:
    gates = postrun["gates"]
    computed_efficiency_gate = str(inference["efficiency_claim_gate"])
    computed_memory_gate = str(inference["memory_claim_gate"])
    inference.update(
        {
            "quality_estimand_scope": "reported_archive_relative",
            "reported_archive_witness_self_consistency": "PASS",
            "all_evaluated_trace_completeness": "NOT_ESTABLISHED",
            "resource_estimand_scope": (
                "descriptive_terminal_attempt_only"
            ),
            "resource_efficiency_evidence_gate": (
                gates["resource_efficiency_gate"]
            ),
            "computed_efficiency_gate_before_evidence_scope": (
                computed_efficiency_gate
            ),
            "computed_memory_gate_before_evidence_scope": (
                computed_memory_gate
            ),
        }
    )
    if gates["resource_efficiency_gate"] == "PASS":
        return
    for item in inference["primary_comparisons"]:
        item["computed_efficiency_comparison_gate"] = item[
            "efficiency_comparison_gate"
        ]
        item["efficiency_comparison_gate"] = "NOT_ESTABLISHED"
        item["efficiency_evidence_limitation"] = (
            "terminal-attempt resource measurements are descriptive only"
        )
    for item in inference["family_gates"]:
        item["computed_efficiency_claim_gate"] = item[
            "efficiency_claim_gate"
        ]
        item["efficiency_claim_gate"] = "NOT_ESTABLISHED"
    inference["efficiency_claim_gate"] = "NOT_ESTABLISHED"
    inference["memory_claim_gate"] = "NOT_ESTABLISHED"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> dict[str, object]:
    raw = _canonical_bytes(value)
    return _write_bytes(path, raw)


def _write_bytes(path: Path, raw: bytes) -> dict[str, object]:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _row_metrics_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    metric_names = tuple(METRIC_ORIENTATIONS)
    fields = [
        "family",
        "case_id",
        "algorithm",
        "seed",
        "budget",
        *metric_names,
        "run_key_sha256",
        "terminal_receipt_sha256",
        "algorithm_result_sha256",
        "archive_sha256",
        "checkpoint_witnesses_sha256",
        "replay_receipt_sha256",
        "reference_sha256",
        "reference_source_artifact_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        metrics = row["metrics"]
        artifacts = row["consumed_artifacts"]
        reference = row["reference_binding"]
        writer.writerow(
            {
                "family": row["family"],
                "case_id": row["case_id"],
                "algorithm": row["algorithm"],
                "seed": row["seed"],
                "budget": row["budget"],
                **{name: metrics[name] for name in metric_names},
                "run_key_sha256": row["run_key_sha256"],
                "terminal_receipt_sha256": artifacts[
                    "terminal_receipt"
                ]["sha256"],
                "algorithm_result_sha256": artifacts[
                    "algorithm_result"
                ]["sha256"],
                "archive_sha256": artifacts["all_evaluated_archive"][
                    "sha256"
                ],
                "checkpoint_witnesses_sha256": artifacts[
                    "checkpoint_witnesses"
                ]["sha256"],
                "replay_receipt_sha256": artifacts["replay_receipt"][
                    "sha256"
                ],
                "reference_sha256": reference["reference_sha256"],
                "reference_source_artifact_sha256": reference[
                    "source_artifact_sha256"
                ],
            }
        )
    return buffer.getvalue().encode("utf-8")


def _paired_inference_csv(inference: Mapping[str, Any]) -> bytes:
    primary_index = {
        (item["family"], item["budget"], item["baseline"], item["metric"]): item
        for item in inference["primary_comparisons"]
    }
    fields = [
        "family",
        "budget",
        "baseline",
        "metric",
        "orientation",
        "paired_case_seed_count",
        "case_cluster_count",
        "case_cluster_mean_advantage",
        "ci95_lower",
        "ci95_upper",
        "wins",
        "ties",
        "losses",
        "exact_case_cluster_sign_flip_p_value",
        "holm_adjusted_p_value_primary_only",
        "quality_comparison_gate_primary_only",
        "efficiency_comparison_gate_primary_only",
        "treatment_over_baseline_ratio_mean",
        "treatment_over_baseline_ratio_ci95_lower",
        "treatment_over_baseline_ratio_ci95_upper",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in inference["comparisons"]:
        wtl = item["paired_wins_ties_losses"]
        ci = item["case_cluster_bootstrap_ci95"]
        ratio = item.get("treatment_over_baseline_ratio", {})
        ratio_ci = ratio.get("case_cluster_bootstrap_ci95", ["", ""])
        primary = primary_index.get(
            (
                item["family"],
                item["budget"],
                item["baseline"],
                item["metric"],
            ),
            {},
        )
        writer.writerow(
            {
                "family": item["family"],
                "budget": item["budget"],
                "baseline": item["baseline"],
                "metric": item["metric"],
                "orientation": item["orientation"],
                "paired_case_seed_count": item["paired_case_seed_count"],
                "case_cluster_count": item["case_cluster_count"],
                "case_cluster_mean_advantage": item[
                    "case_cluster_mean_advantage"
                ],
                "ci95_lower": ci[0],
                "ci95_upper": ci[1],
                "wins": wtl["wins"],
                "ties": wtl["ties"],
                "losses": wtl["losses"],
                "exact_case_cluster_sign_flip_p_value": item[
                    "exact_case_cluster_sign_flip_p_value"
                ],
                "holm_adjusted_p_value_primary_only": primary.get(
                    "holm_adjusted_p_value", ""
                ),
                "quality_comparison_gate_primary_only": primary.get(
                    "quality_comparison_gate", ""
                ),
                "efficiency_comparison_gate_primary_only": primary.get(
                    "efficiency_comparison_gate", ""
                ),
                "treatment_over_baseline_ratio_mean": ratio.get(
                    "case_cluster_mean", ""
                ),
                "treatment_over_baseline_ratio_ci95_lower": ratio_ci[0],
                "treatment_over_baseline_ratio_ci95_upper": ratio_ci[1],
            }
        )
    return buffer.getvalue().encode("utf-8")


def _analysis_markdown(inference: Mapping[str, Any]) -> bytes:
    lines = [
        "# IJOC formal metric and statistical audit",
        "",
        "Formal metric/statistical integrity gate: **PASS**.",
        "",
        (
            "Primary superiority gate: "
            f"**{inference['primary_superiority_gate']}**."
        ),
        "",
        (
            "Efficiency claim gate: "
            f"**{inference['efficiency_claim_gate']}**."
        ),
        "",
        (
            "Quality estimand scope: **reported-archive-relative**. "
            "The replay establishes objective-witness self-consistency, not "
            "completeness of an all-evaluated event trace."
        ),
        "",
        (
            "Wall-time and sampled working-set outcomes are descriptive "
            "terminal-attempt summaries; the resource-efficiency evidence "
            "gate is **NOT_ESTABLISHED**."
        ),
        "",
        "## Six precommitted primary comparisons",
        "",
        (
            "| Family | Baseline | Mean AUC advantage | CI95 | W/T/L | "
            "Exact p | Holm p | Gate |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in inference["primary_comparisons"]:
        ci = item["case_cluster_bootstrap_ci95"]
        wtl = item["paired_wins_ties_losses"]
        lines.append(
            "| {family} | {baseline} | {mean:.9g} | "
            "[{lower:.9g}, {upper:.9g}] | {wins}/{ties}/{losses} | "
            "{p:.9g} | {holm:.9g} | {gate} |".format(
                family=item["family"],
                baseline=item["baseline"],
                mean=float(item["case_cluster_mean_advantage"]),
                lower=float(ci[0]),
                upper=float(ci[1]),
                wins=wtl["wins"],
                ties=wtl["ties"],
                losses=wtl["losses"],
                p=float(item["exact_case_cluster_sign_flip_p_value"]),
                holm=float(item["holm_adjusted_p_value"]),
                gate=item["quality_comparison_gate"],
            )
        )
    lines.extend(
        [
            "",
            "All contrasts are paired within family, case, seed, and budget; "
            "positive values favor the treatment. Families and budgets are "
            "never pooled.",
            "",
            "Quality metrics were independently recomputed from the reported, "
            "replay-bound archive and checkpoint witnesses under the frozen "
            "per-case ideal, nadir, and hypervolume reference. Adapter-emitted "
            "metrics were not trusted.",
            "",
            "Claims remain supplied-reference-relative and reported-archive-"
            "relative. This audit does not establish all-evaluated trace "
            "completeness, true Pareto-front completeness, or true-randomness "
            "certification.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _binding_for(path: Path, *, root: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    base = root.resolve(strict=True)
    try:
        relative = resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Artifact leaves its declared root: {path}.") from error
    return {
        "path": relative.as_posix(),
        "sha256": _file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _resolve_binding(
    root: Path,
    binding: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA-256 binding.")
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError(f"{label} binding is invalid.")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe.")
    base = root.resolve(strict=True)
    path = (base / relative).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its root.") from error
    if not path.is_file() or _file_sha256(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch.")
    return path, digest


def _receipt_binding(
    run_directory: Path,
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    label: str,
) -> tuple[Path, str]:
    return _resolve_binding(
        run_directory,
        {"path": record.get(path_key), "sha256": record.get(sha_key)},
        label=label,
    )


def _entry_identity(entry: object, *, label: str) -> tuple[str, tuple[float, ...]]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} must be an object.")
    if "solution" not in entry or "objectives" not in entry:
        raise ValueError(f"{label} lacks a solution/objective witness.")
    try:
        solution = json.dumps(
            entry["solution"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.solution is not strict JSON.") from error
    objectives = _finite_vector(
        entry["objectives"], dimension=2, label=f"{label}.objectives"
    )
    return solution, objectives


def _validate_front(entries: object, *, label: str) -> tuple[tuple[float, float], ...]:
    points_raw = _entry_objectives(entries, dimension=2, label=label)
    points = tuple((point[0], point[1]) for point in points_raw)
    if len(set(points)) != len(points):
        raise ValueError(f"{label} repeats an objective vector.")
    if len(_nondominated_2d(points)) != len(points):
        raise ValueError(f"{label} is not a zero-tolerance nondominated front.")
    return points


def _load_and_validate_references(
    *,
    frozen_root: Path,
    manifest: Mapping[str, Any],
    case_ids: set[str],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, object]]]:
    if manifest.get("schema") != "ijoc_metric_reference_manifest_v2":
        raise ValueError("Frozen metric-reference manifest schema mismatch.")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, dict) or set(raw_cases) != case_ids:
        raise ValueError(
            "Frozen metric-reference cases do not exactly match the study."
        )
    references: dict[str, Mapping[str, Any]] = {}
    source_bindings: list[dict[str, object]] = []
    for case_id in sorted(case_ids):
        value = raw_cases[case_id]
        if not isinstance(value, dict):
            raise ValueError(f"Metric reference {case_id!r} must be an object.")
        expected_keys = {
            "source_artifact",
            "source_role",
            "reference_sha256",
            "reference_points",
            "ideal",
            "nadir",
            "hv_reference",
        }
        if set(value) != expected_keys:
            raise ValueError(
                f"Metric reference {case_id!r} has an unexpected shape."
            )
        source_path, source_sha = _resolve_binding(
            frozen_root,
            value["source_artifact"],
            label=f"metric reference source {case_id}",
        )
        source, _ = _read_strict_json(
            source_path, label=f"metric reference source {case_id}"
        )
        if (
            source.get("schema") != "ijoc_calibration_reference_case_v1"
            or source.get("case_id") != case_id
            or source.get("source_role") != value.get("source_role")
        ):
            raise ValueError(
                f"Metric reference source identity mismatch for {case_id!r}."
            )
        metric_contract = source.get("metric_contract")
        if (
            not isinstance(metric_contract, dict)
            or metric_contract.get("objective_sense")
            != ["minimize", "minimize"]
            or metric_contract.get("dominance_tolerance") != 0.0
            or metric_contract.get("normalization")
            != "frozen_ideal_nadir_affine"
            or metric_contract.get("archive_semantics")
            != "calibration_all_evaluated_nondominated"
            or not isinstance(
                metric_contract.get("evaluation_code_sha256"), str
            )
            or len(metric_contract["evaluation_code_sha256"]) != 64
        ):
            raise ValueError(
                f"Metric contract drifted for reference case {case_id!r}."
            )
        for key in ("reference_points", "ideal", "nadir", "hv_reference"):
            if source.get(key) != value.get(key):
                raise ValueError(
                    f"Metric reference source/manifest {key} mismatch for "
                    f"{case_id!r}."
                )
        if value.get("reference_sha256") != _canonical_sha256(
            value.get("reference_points")
        ):
            raise ValueError(
                f"Metric reference-point digest mismatch for {case_id!r}."
            )
        # Exercise the complete box/front validation before any result row is
        # accepted.  A one-checkpoint dummy uses only the reference itself.
        _ = recompute_quality_metrics(
            final_entries=[
                {"solution": [index], "objectives": point}
                for index, point in enumerate(value["reference_points"])
            ],
            checkpoints=[
                {
                    "evaluation": 1,
                    "entries": [
                        {"solution": [index], "objectives": point}
                        for index, point in enumerate(
                            value["reference_points"]
                        )
                    ],
                }
            ],
            budget=1,
            checkpoint_period=1,
            reference=value,
        )
        references[case_id] = value
        binding = _binding_for(source_path, root=frozen_root)
        binding["case_id"] = case_id
        if binding["sha256"] != source_sha:
            raise AssertionError("Reference source binding drifted.")
        source_bindings.append(binding)
    return references, source_bindings


def _validate_postrun_bindings(
    *,
    postrun: Mapping[str, Any],
    postrun_path: Path,
    study_path: Path,
    execution_plan_path: Path,
    results_directory: Path,
    study_sha: str,
    matrix_sha: str,
    execution_plan_sha: str,
    expected_run_count: int,
) -> tuple[Path, Path]:
    implementation = postrun.get("audit_implementation")
    postrun_source = Path(__file__).with_name("pareto_ijoc_postrun.py")
    if (
        not isinstance(implementation, dict)
        or implementation.get("scope")
        != "posthoc_fail_closed_amendment_not_frozen_algorithm_runtime"
        or implementation.get("frozen_algorithm_modified") is not False
        or implementation.get("formal_results_modified") is not False
        or implementation.get("postrun_source_sha256")
        != _file_sha256(postrun_source)
    ):
        raise ValueError("Post-run audit implementation binding is invalid.")
    required_zero = (
        "missing_run_count",
        "duplicate_run_count",
        "unexpected_run_count",
        "invalid_run_count",
    )
    if (
        postrun.get("study_sha256") != study_sha
        or postrun.get("configuration_matrix_sha256") != matrix_sha
        or postrun.get("execution_plan_sha256") != execution_plan_sha
        or postrun.get("expected_run_count") != expected_run_count
        or postrun.get("observed_unique_run_count") != expected_run_count
        or postrun.get("valid_run_count") != expected_run_count
        or any(postrun.get(key) != 0 for key in required_zero)
    ):
        raise ValueError("Post-run audit does not bind a complete current matrix.")
    gates = postrun.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(gates.get(name) != "PASS" for name in QUALITY_POSTRUN_GATES)
        or gates.get("all_evaluated_trace_completeness_gate")
        != "NOT_ESTABLISHED"
        or gates.get("terminal_process_resource_measurement_gate") != "PASS"
        or gates.get("single_attempt_resource_cleanliness_gate")
        not in {"PASS", "NOT_ESTABLISHED"}
        or gates.get("resource_design_balance_gate") != "NOT_ESTABLISHED"
        or gates.get("resource_efficiency_gate") != "NOT_ESTABLISHED"
    ):
        raise ValueError(
            "Post-run quality, archive-scope, or resource-scope gates are invalid."
        )
    if (
        postrun.get("resource_estimand_scope")
        != "descriptive_terminal_attempt_only"
        or postrun.get("resource_efficiency_claim_status")
        != "NOT_ESTABLISHED"
    ):
        raise ValueError("Post-run resource claim scope is invalid.")
    attempt_audit = postrun.get("attempt_audit")
    if not isinstance(attempt_audit, dict):
        raise ValueError("Post-run attempt audit is missing.")
    retry_run_count = attempt_audit.get("retry_run_count")
    prior_attempt_count = attempt_audit.get("prior_attempt_count")
    retry_keys = attempt_audit.get("retry_run_key_sha256")
    retry_failures = attempt_audit.get("quality_retry_failures")
    histories = attempt_audit.get("histories")
    if (
        isinstance(retry_run_count, bool)
        or not isinstance(retry_run_count, int)
        or retry_run_count < 0
        or isinstance(prior_attempt_count, bool)
        or not isinstance(prior_attempt_count, int)
        or prior_attempt_count < retry_run_count
        or not isinstance(retry_keys, list)
        or len(retry_keys) != retry_run_count
        or len(set(retry_keys)) != retry_run_count
        or retry_failures != []
        or not isinstance(histories, list)
        or len(histories) != retry_run_count
    ):
        raise ValueError("Post-run attempt audit is internally inconsistent.")
    result_root = results_directory.resolve(strict=True)
    try:
        postrun_path.resolve(strict=True).relative_to(result_root)
    except ValueError as error:
        raise ValueError("Post-run audit is outside the selected results.") from error
    invocation_path = (result_root / "matrix_invocation.json").resolve(
        strict=True
    )
    if (
        not invocation_path.is_file()
        or _file_sha256(invocation_path)
        != postrun.get("matrix_invocation_sha256")
    ):
        raise ValueError("Post-run audit matrix-invocation binding mismatch.")
    invocation, _ = _read_strict_json(
        invocation_path, label="matrix invocation"
    )
    if (
        invocation.get("schema")
        != "ijoc_cold_process_matrix_invocation_v1"
        or invocation.get("execution_scope") != "formal_candidate"
        or invocation.get("formal_evidence_status") != "NOT_RUN"
        or invocation.get("selection") != {"kind": "all"}
        or invocation.get("selected_run_count") != expected_run_count
        or invocation.get("expected_run_count") != expected_run_count
    ):
        raise ValueError("Matrix invocation is not the exact full formal matrix.")
    freeze_receipt_path = (
        study_path.parent / "freeze_receipt.json"
    ).resolve(strict=True)
    if (
        not freeze_receipt_path.is_file()
        or _file_sha256(freeze_receipt_path)
        != postrun.get("freeze_receipt_sha256")
    ):
        raise ValueError("Post-run audit freeze-receipt binding mismatch.")
    if study_path.parent.resolve() != execution_plan_path.parent.resolve():
        raise ValueError("Study and execution plan are not one frozen packet.")
    _validate_current_attempt_inventory(
        postrun=postrun,
        result_root=result_root,
    )
    return invocation_path, freeze_receipt_path


def _validate_current_attempt_inventory(
    *,
    postrun: Mapping[str, Any],
    result_root: Path,
) -> None:
    attempt_audit = postrun["attempt_audit"]
    expected_retry_keys = set(attempt_audit["retry_run_key_sha256"])
    histories = {
        str(item.get("run_key_sha256")): item
        for item in attempt_audit["histories"]
        if isinstance(item, dict)
    }
    if set(histories) != expected_retry_keys:
        raise ValueError("Post-run attempt audit histories are inconsistent.")

    observed_retry_keys: set[str] = set()
    observed_prior_count = 0
    terminal_paths = sorted(result_root.rglob("terminal_receipt.json"))
    for terminal_path in terminal_paths:
        terminal, _ = _read_strict_json(
            terminal_path, label="terminal receipt during attempt audit"
        )
        run_key_sha = terminal.get("run_key_sha256")
        attempt_number = terminal.get("attempt_number")
        if (
            not isinstance(run_key_sha, str)
            or terminal_path.parent.name != run_key_sha
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number <= 0
        ):
            raise ValueError("Terminal attempt audit identity is invalid.")
        attempts_root = terminal_path.parent / "attempts"
        if not attempts_root.is_dir():
            raise ValueError("Terminal attempt audit directory is missing.")
        observed_numbers: list[int] = []
        for child in sorted(attempts_root.iterdir(), key=lambda path: path.name):
            if (
                not child.is_dir()
                or len(child.name) != 6
                or not child.name.isascii()
                or not child.name.isdigit()
                or int(child.name) <= 0
            ):
                raise ValueError("Attempt audit found a malformed entry.")
            observed_numbers.append(int(child.name))
        if observed_numbers != list(range(1, attempt_number + 1)):
            raise ValueError(
                "Attempt audit changed after the post-run receipt was written."
            )
        if attempt_number == 1:
            if run_key_sha in expected_retry_keys:
                raise ValueError("Attempt audit retry classification changed.")
            continue

        observed_retry_keys.add(run_key_sha)
        observed_prior_count += attempt_number - 1
        history = histories.get(run_key_sha)
        if (
            not isinstance(history, dict)
            or history.get("terminal_attempt_number") != attempt_number
            or history.get("observed_attempt_numbers") != observed_numbers
            or history.get("prior_attempt_count") != attempt_number - 1
            or history.get("retry_quality_eligible") is not True
        ):
            raise ValueError("Attempt audit history binding changed.")
        prior_records = history.get("prior_attempts")
        if (
            not isinstance(prior_records, list)
            or len(prior_records) != attempt_number - 1
        ):
            raise ValueError("Attempt audit prior-attempt list is invalid.")
        for prior in prior_records:
            if not isinstance(prior, dict):
                raise ValueError("Attempt audit prior-attempt record is invalid.")
            number = prior.get("attempt_number")
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number <= 0
                or number >= attempt_number
            ):
                raise ValueError("Attempt audit prior-attempt number is invalid.")
            attempt = attempts_root / f"{number:06d}"
            actual_artifacts = [
                {
                    "path": artifact.relative_to(result_root).as_posix(),
                    "sha256": _file_sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
                for artifact in sorted(attempt.rglob("*"))
                if artifact.is_file()
            ]
            actual_subdirectories = [
                artifact.relative_to(result_root).as_posix()
                for artifact in sorted(attempt.rglob("*"))
                if artifact.is_dir()
            ]
            if (
                prior.get("artifacts") != actual_artifacts
                or prior.get("unexpected_subdirectories")
                != actual_subdirectories
                or prior.get("input_matches_terminal") is not True
                or prior.get("exact_incomplete_artifact_set") is not True
                or prior.get("empty_algorithm_logs") is not True
                or prior.get("result_artifact_status")
                != "NO_RESULT_ARTIFACT"
                or prior.get("termination_reason") != "UNKNOWN_UNRECORDED"
            ):
                raise ValueError("Attempt audit prior-attempt bytes changed.")

    if (
        observed_retry_keys != expected_retry_keys
        or observed_prior_count != attempt_audit["prior_attempt_count"]
        or len(observed_retry_keys) != attempt_audit["retry_run_count"]
    ):
        raise ValueError("Attempt audit inventory changed after post-run.")


def _extract_row_metrics(
    *,
    results_directory: Path,
    rows: Sequence[Mapping[str, Any]],
    case_family: Mapping[str, str],
    references: Mapping[str, Mapping[str, Any]],
    checkpoint_period: int,
    global_hashes: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_receipts: set[Path] = set()
    row_metrics: list[dict[str, Any]] = []
    consumed_rows: list[dict[str, Any]] = []
    result_root = results_directory.resolve(strict=True)
    for raw in rows:
        run_key = {
            "case_id": raw["case_id"],
            "algorithm": raw["algorithm"],
            "seed": raw["seed"],
            "budget": raw["budget"],
        }
        run_sha = _canonical_sha256(run_key)
        run_directory = result_root / "runs" / run_sha
        terminal_path = run_directory / "terminal_receipt.json"
        expected_receipts.add(terminal_path.resolve())
        terminal, terminal_sha = _read_strict_json(
            terminal_path, label=f"terminal receipt {run_sha}"
        )
        if (
            terminal.get("schema") != "ijoc_cold_process_run_receipt_v1"
            or terminal.get("run_key") != run_key
            or terminal.get("run_key_sha256") != run_sha
            or terminal.get("study_sha256") != global_hashes["study"]
            or terminal.get("configuration_matrix_sha256")
            != global_hashes["configuration_matrix"]
            or terminal.get("execution_plan_sha256")
            != global_hashes["execution_plan"]
            or terminal.get("freeze_receipt_sha256")
            != global_hashes["freeze_receipt"]
            or terminal.get("execution_scope") != "formal_candidate"
            or terminal.get("formal_evidence_status")
            != "PENDING_POST_RUN_AUDIT"
            or terminal.get("status") != "SUCCESS"
        ):
            raise ValueError(f"Terminal receipt binding failed for {run_sha}.")
        process = terminal.get("algorithm_process")
        if not isinstance(process, dict):
            raise ValueError(f"Algorithm process record is absent for {run_sha}.")
        wall = process.get("wall_time_seconds")
        peak = process.get("sampled_peak_process_tree_rss_bytes")
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) <= 0.0
            or isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak <= 0
            or process.get("resource_measurement_status") != "PASS"
        ):
            raise ValueError(f"Resource measurement failed for {run_sha}.")

        result_record = terminal.get("algorithm_result")
        if not isinstance(result_record, dict):
            raise ValueError(f"Algorithm result binding is absent for {run_sha}.")
        expected_result_keys = {
            "path",
            "sha256",
            "archive_path",
            "archive_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
        }
        if set(result_record) != expected_result_keys:
            raise ValueError(f"Algorithm result binding shape failed for {run_sha}.")
        result_path, result_sha = _receipt_binding(
            run_directory,
            result_record,
            path_key="path",
            sha_key="sha256",
            label=f"algorithm result {run_sha}",
        )
        archive_path, archive_sha = _receipt_binding(
            run_directory,
            result_record,
            path_key="archive_path",
            sha_key="archive_sha256",
            label=f"all-evaluated archive {run_sha}",
        )
        checkpoint_path, checkpoint_sha = _receipt_binding(
            run_directory,
            result_record,
            path_key="checkpoint_path",
            sha_key="checkpoint_sha256",
            label=f"checkpoint witnesses {run_sha}",
        )
        result, _ = _read_strict_json(
            result_path, label=f"algorithm result {run_sha}"
        )
        budget = int(run_key["budget"])
        expected_grid = list(
            range(checkpoint_period, budget + 1, checkpoint_period)
        )
        if (
            result.get("schema") != "ijoc_algorithm_result_v1"
            or result.get("run_key") != run_key
            or result.get("status") != "SUCCESS"
            or result.get("evaluations_used") != budget
            or result.get("observed_checkpoints") != expected_grid
        ):
            raise ValueError(f"Algorithm result contract failed for {run_sha}.")
        bound_archive, bound_archive_sha = _resolve_binding(
            result_path.parent,
            result.get("archive_artifact"),
            label=f"result archive {run_sha}",
        )
        bound_checkpoints, bound_checkpoint_sha = _resolve_binding(
            result_path.parent,
            result.get("checkpoint_artifact"),
            label=f"result checkpoint witnesses {run_sha}",
        )
        if (
            bound_archive != archive_path
            or bound_archive_sha != archive_sha
            or bound_checkpoints != checkpoint_path
            or bound_checkpoint_sha != checkpoint_sha
        ):
            raise ValueError(
                f"Terminal/result artifact bindings disagree for {run_sha}."
            )

        archive, _ = _read_strict_json(
            archive_path, label=f"all-evaluated archive {run_sha}"
        )
        checkpoints, _ = _read_strict_json(
            checkpoint_path, label=f"checkpoint witnesses {run_sha}"
        )
        if (
            archive.get("schema") != "ijoc_all_evaluated_archive_v1"
            or archive.get("run_key") != run_key
            or float(archive.get("dominance_tolerance", math.nan)) != 0.0
            or checkpoints.get("schema")
            != "ijoc_checkpoint_solution_witnesses_v1"
            or checkpoints.get("run_key") != run_key
            or checkpoints.get("checkpoint_period") != checkpoint_period
        ):
            raise ValueError(f"Witness artifact identity failed for {run_sha}.")
        final_entries = archive.get("entries")
        _validate_front(final_entries, label=f"final archive {run_sha}")
        checkpoint_values = checkpoints.get("checkpoints")
        if not isinstance(checkpoint_values, list):
            raise ValueError(f"Checkpoint array is absent for {run_sha}.")
        for index, checkpoint in enumerate(checkpoint_values):
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Checkpoint {index} is invalid for {run_sha}.")
            _validate_front(
                checkpoint.get("entries"),
                label=f"checkpoint {index} archive {run_sha}",
            )
        if not checkpoint_values:
            raise ValueError(f"Checkpoint witnesses are empty for {run_sha}.")
        final_identity = {
            _entry_identity(entry, label=f"final archive entry {run_sha}")
            for entry in final_entries
        }
        checkpoint_identity = {
            _entry_identity(entry, label=f"final checkpoint entry {run_sha}")
            for entry in checkpoint_values[-1].get("entries", [])
        }
        if final_identity != checkpoint_identity:
            raise ValueError(
                f"Final checkpoint/archive witnesses differ for {run_sha}."
            )

        replay_record = terminal.get("replay_result")
        if not isinstance(replay_record, dict):
            raise ValueError(f"Replay binding is absent for {run_sha}.")
        replay_path, replay_sha = _resolve_binding(
            run_directory,
            replay_record,
            label=f"replay receipt {run_sha}",
        )
        replay, _ = _read_strict_json(
            replay_path, label=f"replay receipt {run_sha}"
        )
        if (
            replay.get("schema") != "ijoc_replay_receipt_v1"
            or replay.get("run_key") != run_key
            or replay.get("status") != "PASS"
            or replay.get("algorithm_result_sha256") != result_sha
            or replay.get("archive_sha256") != archive_sha
            or replay.get("checkpoint_artifact_sha256") != checkpoint_sha
            or replay.get("evaluations_used") != budget
            or replay.get("observed_checkpoints") != expected_grid
        ):
            raise ValueError(f"Replay receipt binding failed for {run_sha}.")

        quality = recompute_quality_metrics(
            final_entries=final_entries,
            checkpoints=checkpoint_values,
            budget=budget,
            checkpoint_period=checkpoint_period,
            reference=references[str(run_key["case_id"])],
        )
        metrics: dict[str, int | float] = {
            **quality,
            "wall_time_seconds": float(wall),
            "sampled_peak_process_tree_rss_bytes": int(peak),
        }
        artifact_bindings = {
            "terminal_receipt": _binding_for(
                terminal_path, root=result_root
            ),
            "algorithm_result": _binding_for(result_path, root=result_root),
            "all_evaluated_archive": _binding_for(
                archive_path, root=result_root
            ),
            "checkpoint_witnesses": _binding_for(
                checkpoint_path, root=result_root
            ),
            "replay_receipt": _binding_for(replay_path, root=result_root),
        }
        if artifact_bindings["terminal_receipt"]["sha256"] != terminal_sha:
            raise AssertionError("Terminal receipt digest drifted.")
        if artifact_bindings["replay_receipt"]["sha256"] != replay_sha:
            raise AssertionError("Replay receipt digest drifted.")
        family = case_family[str(run_key["case_id"])]
        row_metrics.append(
            {
                "family": family,
                **run_key,
                "run_key_sha256": run_sha,
                "metrics": metrics,
                "reference_binding": {
                    "reference_sha256": references[str(run_key["case_id"])][
                        "reference_sha256"
                    ],
                    "source_artifact_sha256": references[
                        str(run_key["case_id"])
                    ]["source_artifact"]["sha256"],
                },
                "consumed_artifacts": artifact_bindings,
            }
        )
        consumed_rows.append(
            {
                "run_key": run_key,
                "run_key_sha256": run_sha,
                "artifacts": artifact_bindings,
            }
        )

    observed_receipts = {
        path.resolve()
        for path in result_root.rglob("terminal_receipt.json")
        if path.is_file()
    }
    if observed_receipts != expected_receipts:
        raise ValueError(
            "Terminal receipt set differs from the frozen configuration matrix."
        )
    row_metrics.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["case_id"]),
            str(row["algorithm"]),
            int(row["seed"]),
            int(row["budget"]),
        )
    )
    consumed_rows.sort(key=lambda row: str(row["run_key_sha256"]))
    return row_metrics, consumed_rows


def _analyze_passing_matrix(
    *,
    study_path: Path,
    execution_plan_path: Path,
    results_directory: Path,
    post_run_audit_path: Path,
    postrun: Mapping[str, Any],
    output_directory: Path,
) -> IJOCFormalAnalysisResult:
    if output_directory.exists():
        raise ValueError("Analysis output directory already exists.")
    study, study_sha = _read_strict_json(study_path, label="frozen study")
    execution_plan, execution_plan_sha = _read_strict_json(
        execution_plan_path, label="frozen execution plan"
    )
    if study.get("schema") != "ijoc_competitive_study_v3":
        raise ValueError("Frozen study schema mismatch.")
    if (
        execution_plan.get("schema")
        != "ijoc_cold_process_execution_plan_v1"
        or execution_plan.get("study_sha256") != study_sha
        or execution_plan.get("execution_scope") != "formal_candidate"
        or execution_plan.get("formal_evidence_status") != "NOT_RUN"
    ):
        raise ValueError("Execution plan is not bound to the frozen study.")
    matrix_path, matrix_sha = _resolve_binding(
        study_path.parent,
        study.get("algorithm_configuration_matrix"),
        label="algorithm configuration matrix",
    )
    metric_path, metric_sha = _resolve_binding(
        study_path.parent,
        study.get("metric_reference_manifest"),
        label="metric reference manifest",
    )
    analysis_plan_path, analysis_plan_sha = _resolve_binding(
        study_path.parent,
        study.get("formal_analysis_plan"),
        label="formal analysis plan",
    )
    if execution_plan.get("configuration_matrix_sha256") != matrix_sha:
        raise ValueError("Execution plan configuration-matrix hash mismatch.")
    execution_plan_analysis_path, execution_plan_analysis_sha = _resolve_binding(
        execution_plan_path.parent,
        execution_plan.get("formal_analysis_plan"),
        label="execution-plan formal analysis plan",
    )
    if (
        execution_plan_analysis_path != analysis_plan_path
        or execution_plan_analysis_sha != analysis_plan_sha
    ):
        raise ValueError("Study/execution-plan analysis bindings disagree.")

    matrix, _ = _read_strict_json(
        matrix_path, label="algorithm configuration matrix"
    )
    metric_manifest, _ = _read_strict_json(
        metric_path, label="metric reference manifest"
    )
    analysis_plan, _ = _read_strict_json(
        analysis_plan_path, label="formal analysis plan"
    )
    if (
        matrix.get("schema") != "ijoc_algorithm_configuration_matrix_v1"
        or analysis_plan.get("schema") != "ijoc_formal_analysis_plan_v1"
        or analysis_plan.get("status")
        != "PRECOMMITTED_BEFORE_FORMAL_EXECUTION"
        or analysis_plan.get("formal_evidence_status") != "NOT_RUN"
    ):
        raise ValueError("Frozen matrix or formal analysis-plan schema mismatch.")
    rows_raw = matrix.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("Frozen configuration matrix is empty.")
    rows: list[Mapping[str, Any]] = []
    seen_run_keys: set[str] = set()
    for raw in rows_raw:
        if not isinstance(raw, dict):
            raise ValueError("Configuration matrix row must be an object.")
        expected_keys = {
            "case_id",
            "algorithm",
            "seed",
            "budget",
            "configuration",
            "configuration_sha256",
        }
        if set(raw) != expected_keys or not isinstance(
            raw.get("configuration"), dict
        ):
            raise ValueError("Configuration matrix row shape is invalid.")
        if raw.get("configuration_sha256") != _canonical_sha256(
            raw["configuration"]
        ):
            raise ValueError("Readable configuration hash mismatch.")
        run_key = {
            "case_id": raw.get("case_id"),
            "algorithm": raw.get("algorithm"),
            "seed": raw.get("seed"),
            "budget": raw.get("budget"),
        }
        run_sha = _canonical_sha256(run_key)
        if run_sha in seen_run_keys:
            raise ValueError("Configuration matrix repeats a run key.")
        seen_run_keys.add(run_sha)
        rows.append(raw)

    plan_families = _string_list(
        analysis_plan.get("families"), label="formal analysis plan families"
    )
    plan_seeds = _integer_list(
        analysis_plan.get("formal_seeds"),
        label="formal analysis plan seeds",
    )
    plan_budgets = _integer_list(
        analysis_plan.get("evaluation_budgets"),
        label="formal analysis plan budgets",
    )
    checkpoint_period = analysis_plan.get("anytime_checkpoint_period")
    if (
        study.get("seeds") != list(plan_seeds)
        or study.get("budgets") != list(plan_budgets)
        or study.get("anytime_checkpoint_period") != checkpoint_period
        or isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
        or any(budget % checkpoint_period for budget in plan_budgets)
    ):
        raise ValueError("Study and formal analysis grid contracts disagree.")
    primary_metric = analysis_plan.get("primary_metric")
    if (
        not isinstance(primary_metric, dict)
        or primary_metric.get("name")
        != "normalized_left_continuous_hypervolume_auc"
        or primary_metric.get("orientation") != "larger_is_better"
        or primary_metric.get("normalization")
        != "per_case_frozen_ideal_nadir_affine"
        or primary_metric.get("hypervolume_reference")
        != "per_case_frozen_reference_calibration_artifact"
        or primary_metric.get("checkpoint_semantics")
        != (
            "all_evaluated_nondominated_archive_at_each_common_checkpoint"
        )
        or primary_metric.get("initial_value") != 0.0
        or analysis_plan.get("comparison_unit")
        != "same_family_case_seed_budget"
        or analysis_plan.get("cluster_unit") != "case_id"
        or analysis_plan.get("family_pooling") != "forbidden"
        or analysis_plan.get("budget_pooling") != "forbidden"
        or analysis_plan.get("paired_contrast_orientation")
        != "positive_always_favors_treatment"
    ):
        raise ValueError("Primary metric/paired estimand contract drifted.")
    secondary_metrics = analysis_plan.get("secondary_metrics")
    expected_secondary = {
        "normalized_final_hypervolume": "larger_is_better",
        "igd_plus_to_frozen_supplied_reference": "smaller_is_better",
        "additive_epsilon_to_frozen_supplied_reference": "smaller_is_better",
        "wall_time_seconds": "smaller_is_better",
        "sampled_peak_process_tree_rss_bytes": "smaller_is_better",
    }
    if (
        not isinstance(secondary_metrics, list)
        or {
            str(item.get("name")): str(item.get("orientation"))
            for item in secondary_metrics
            if isinstance(item, dict)
        }
        != expected_secondary
        or len(secondary_metrics) != len(expected_secondary)
    ):
        raise ValueError("Secondary metric contract drifted.")
    tie_contract = analysis_plan.get("wins_ties_losses")
    missing_contract = analysis_plan.get("missing_or_failed_rows")
    primary_gate_contract = analysis_plan.get("primary_gate")
    if (
        not isinstance(tie_contract, dict)
        or tie_contract.get("unit") != "paired_case_seed"
        or not isinstance(missing_contract, dict)
        or missing_contract.get("imputation") != "forbidden"
        or missing_contract.get("formal_matrix_completeness") != "FAIL"
        or missing_contract.get("submission_status") != "HOLD"
        or not isinstance(primary_gate_contract, dict)
        or primary_gate_contract.get("scope")
        != "each_family_separately_at_primary_budget"
        or analysis_plan.get("reference_scope")
        != (
            "supplied-reference-relative_only_not_true_pareto_front_"
            "completeness"
        )
        or analysis_plan.get("randomness_scope")
        != (
            "pseudo_random_seeded_computational_experiment_not_a_true_"
            "randomness_certificate"
        )
    ):
        raise ValueError("Formal gate or claim-boundary contract drifted.")

    treatment = analysis_plan.get("treatment")
    baseline_contract = analysis_plan.get("required_baselines")
    raw_families = study.get("problem_families")
    if (
        not isinstance(treatment, str)
        or not isinstance(baseline_contract, dict)
        or not isinstance(raw_families, list)
    ):
        raise ValueError("Study family/algorithm contract is invalid.")
    case_family: dict[str, str] = {}
    expected_matrix_keys: set[tuple[str, str, int, int]] = set()
    study_families: set[str] = set()
    for raw_family in raw_families:
        if not isinstance(raw_family, dict):
            raise ValueError("Study family must be an object.")
        family = str(raw_family.get("id", "")).upper()
        if family not in plan_families or family in study_families:
            raise ValueError("Study family identity differs from the plan.")
        study_families.add(family)
        family_cases = _string_list(
            raw_family.get("cases"), label=f"study {family} cases"
        )
        family_algorithms = _string_list(
            raw_family.get("algorithms"),
            label=f"study {family} algorithms",
        )
        expected_baselines = _string_list(
            baseline_contract.get(family),
            label=f"analysis {family} baselines",
        )
        if (
            set(family_algorithms) != {treatment, *expected_baselines}
            or raw_family.get("required_baselines")
            != list(expected_baselines)
        ):
            raise ValueError(f"Study {family} algorithm contract drifted.")
        for case_id in family_cases:
            if case_id in case_family:
                raise ValueError("A formal case belongs to multiple families.")
            case_family[case_id] = family
            for algorithm in family_algorithms:
                for seed in plan_seeds:
                    for budget in plan_budgets:
                        expected_matrix_keys.add(
                            (case_id, algorithm, seed, budget)
                        )
    if study_families != set(plan_families):
        raise ValueError("Study does not cover every planned family.")
    observed_matrix_keys = {
        (
            str(row["case_id"]),
            str(row["algorithm"]),
            int(row["seed"]),
            int(row["budget"]),
        )
        for row in rows
    }
    if observed_matrix_keys != expected_matrix_keys:
        raise ValueError("Configuration matrix is not the exact matched product.")

    references, reference_sources = _load_and_validate_references(
        frozen_root=study_path.parent,
        manifest=metric_manifest,
        case_ids=set(case_family),
    )
    invocation_path, freeze_receipt_path = _validate_postrun_bindings(
        postrun=postrun,
        postrun_path=post_run_audit_path,
        study_path=study_path,
        execution_plan_path=execution_plan_path,
        results_directory=results_directory,
        study_sha=study_sha,
        matrix_sha=matrix_sha,
        execution_plan_sha=execution_plan_sha,
        expected_run_count=len(rows),
    )
    global_hashes = {
        "study": study_sha,
        "configuration_matrix": matrix_sha,
        "execution_plan": execution_plan_sha,
        "freeze_receipt": _file_sha256(freeze_receipt_path),
    }
    row_metrics, consumed_rows = _extract_row_metrics(
        results_directory=results_directory,
        rows=rows,
        case_family=case_family,
        references=references,
        checkpoint_period=checkpoint_period,
        global_hashes=global_hashes,
    )
    inference = build_paired_inference(row_metrics, plan=analysis_plan)
    _apply_postrun_evidence_scope(inference, postrun=postrun)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(
        output_directory.name + f".staging-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        top_inputs = {
            "study": _binding_for(study_path, root=study_path.parent),
            "execution_plan": _binding_for(
                execution_plan_path, root=study_path.parent
            ),
            "formal_analysis_plan": _binding_for(
                analysis_plan_path, root=study_path.parent
            ),
            "metric_reference_manifest": _binding_for(
                metric_path, root=study_path.parent
            ),
            "algorithm_configuration_matrix": _binding_for(
                matrix_path, root=study_path.parent
            ),
            "freeze_receipt": _binding_for(
                freeze_receipt_path, root=study_path.parent
            ),
            "matrix_invocation": _binding_for(
                invocation_path, root=results_directory
            ),
            "post_run_audit": _binding_for(
                post_run_audit_path, root=results_directory
            ),
        }
        consumed_manifest = {
            "schema": "ijoc_formal_analysis_consumed_artifacts_v1",
            "top_level_inputs": top_inputs,
            "metric_reference_sources": reference_sources,
            "row_artifacts": consumed_rows,
            "row_artifact_count": sum(
                len(row["artifacts"]) for row in consumed_rows
            ),
            "terminal_receipt_set_sha256": _canonical_sha256(
                [
                    row["artifacts"]["terminal_receipt"]
                    for row in consumed_rows
                ]
            ),
            "consumed_row_artifact_set_sha256": _canonical_sha256(
                consumed_rows
            ),
        }
        consumed_binding = _write_json(
            staging / "consumed_artifacts_manifest.json",
            consumed_manifest,
        )
        row_payload = {
            "schema": "ijoc_formal_row_metrics_v2",
            "study_sha256": study_sha,
            "execution_plan_sha256": execution_plan_sha,
            "formal_analysis_plan_sha256": analysis_plan_sha,
            "metric_reference_manifest_sha256": metric_sha,
            "post_run_audit_sha256": _file_sha256(post_run_audit_path),
            "metric_source": (
                "independently_recomputed_from_reported_replay_bound_"
                "checkpoint_and_archive_witnesses"
            ),
            "adapter_emitted_metrics_trusted": False,
            "quality_estimand_scope": "reported_archive_relative",
            "all_evaluated_trace_completeness": "NOT_ESTABLISHED",
            "resource_estimand_scope": (
                "descriptive_terminal_attempt_only"
            ),
            "row_count": len(row_metrics),
            "rows": row_metrics,
        }
        row_binding = _write_json(staging / "row_metrics.json", row_payload)
        row_csv_binding = _write_bytes(
            staging / "row_metrics.csv",
            _row_metrics_csv(row_metrics),
        )
        inference.update(
            {
                "study_sha256": study_sha,
                "formal_analysis_plan_sha256": analysis_plan_sha,
                "row_metrics_sha256": row_binding["sha256"],
            }
        )
        inference_binding = _write_json(
            staging / "paired_inference.json", inference
        )
        inference_csv_binding = _write_bytes(
            staging / "paired_inference.csv",
            _paired_inference_csv(inference),
        )
        markdown_binding = _write_bytes(
            staging / "FORMAL_ANALYSIS_REPORT.md",
            _analysis_markdown(inference),
        )
        formal_gate = "PASS"
        primary_gate = str(inference["primary_superiority_gate"])
        efficiency_gate = str(inference["efficiency_claim_gate"])
        audit_payload = {
            "schema": "ijoc_formal_metric_statistical_audit_v2",
            "audit_implementation": {
                "scope": (
                    "posthoc_fail_closed_amendment_not_frozen_algorithm_"
                    "runtime"
                ),
                "analysis_source_sha256": _file_sha256(
                    Path(__file__).resolve(strict=True)
                ),
                "frozen_algorithm_modified": False,
                "formal_results_modified": False,
            },
            "status": "COMPLETE",
            "formal_evidence_scope": (
                "reported_archive_relative_matched_matrix_metric_and_"
                "precommitted_paired_inference"
            ),
            "inputs": {
                **top_inputs,
                "consumed_artifact_manifest_sha256": consumed_binding[
                    "sha256"
                ],
            },
            "expected_row_count": len(rows),
            "recomputed_row_count": len(row_metrics),
            "paired_comparison_count": len(inference["comparisons"]),
            "primary_comparison_count": len(
                inference["primary_comparisons"]
            ),
            "outputs": {
                "consumed_artifacts_manifest": consumed_binding,
                "row_metrics": row_binding,
                "row_metrics_csv": row_csv_binding,
                "paired_inference": inference_binding,
                "paired_inference_csv": inference_csv_binding,
                "formal_analysis_report": markdown_binding,
            },
            "gates": {
                "post_run_formal_gate": "PASS",
                "frozen_input_hash_binding_gate": "PASS",
                "complete_row_recomputation_gate": "PASS",
                "paired_matrix_gate": "PASS",
                "case_cluster_bootstrap_gate": "PASS",
                "exact_sign_flip_gate": "PASS",
                "six_comparison_holm_gate": "PASS",
                "formal_metric_statistical_gate": formal_gate,
            },
            "primary_superiority_gate": primary_gate,
            "efficiency_claim_gate": efficiency_gate,
            "memory_claim_gate": inference["memory_claim_gate"],
            "scientific_result_action": (
                "REPORTED_ARCHIVE_RELATIVE_SUPERIORITY_CLAIM_PERMITTED_"
                "WITHIN_FROZEN_SCOPE"
                if primary_gate == "PASS"
                else "REPORT_NON_SUPERIORITY_OR_INCONCLUSIVE_RESULT"
            ),
            "submission_verdict": (
                "HOLD_PENDING_MANUSCRIPT_CONSISTENCY_AND_RELEASE_AUDIT"
            ),
            "claim_boundaries": {
                "reference": analysis_plan.get("reference_scope"),
                "randomness": analysis_plan.get("randomness_scope"),
                "archive": "reported_archive_relative",
                "reported_archive_witness_self_consistency": "PASS",
                "all_evaluated_trace_completeness": "NOT_ESTABLISHED",
                "resource": "descriptive_terminal_attempt_only",
                "resource_efficiency": "NOT_ESTABLISHED",
                "family_pooling": "forbidden",
                "budget_pooling": "forbidden",
                "failed_row_imputation": "forbidden",
                "true_pareto_front_completeness": "NOT_CLAIMED",
            },
        }
        _write_json(
            staging / "formal_metric_statistical_audit.json",
            audit_payload,
        )
        os.replace(staging, output_directory)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return IJOCFormalAnalysisResult(
        output_directory=output_directory,
        audit_path=output_directory / "formal_metric_statistical_audit.json",
        row_count=len(row_metrics),
        formal_metric_statistical_gate="PASS",
        primary_superiority_gate=str(inference["primary_superiority_gate"]),
        efficiency_claim_gate=str(inference["efficiency_claim_gate"]),
    )
