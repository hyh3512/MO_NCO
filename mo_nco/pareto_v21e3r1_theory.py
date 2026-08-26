from __future__ import annotations

"""Executable finite-budget theory helpers for the V21e3r1 IJOC mainline.

The objects here deliberately avoid importing the historical Sinkhorn/SMC/IPS
claims.  They encode only the finite deterministic identities and prospective
selection rules used by the corrected typed-hybrid algorithm.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AttemptBudgetCertificate:
    charged_evaluations: int
    retry_cap: int
    fallback_cap: int
    observed_attempts: int | None
    lower_bound: int
    upper_bound: int
    passed: bool | None


def successful_attempt_bound(
    charged_evaluations: int,
    *,
    retry_cap: int,
    fallback_cap: int,
    observed_attempts: int | None = None,
) -> AttemptBudgetCertificate:
    """Certify B <= A <= (1+R+F)B for a successful frozen retry service."""

    if charged_evaluations < 0 or retry_cap < 0 or fallback_cap < 0:
        raise ValueError("Counts and caps must be nonnegative.")
    lower = charged_evaluations
    upper = (1 + retry_cap + fallback_cap) * charged_evaluations
    passed = None
    if observed_attempts is not None:
        if observed_attempts < 0:
            raise ValueError("observed_attempts must be nonnegative.")
        passed = lower <= observed_attempts <= upper
    return AttemptBudgetCertificate(
        charged_evaluations=charged_evaluations,
        retry_cap=retry_cap,
        fallback_cap=fallback_cap,
        observed_attempts=observed_attempts,
        lower_bound=lower,
        upper_bound=upper,
        passed=passed,
    )


@dataclass(frozen=True)
class PrefixAccountingCertificate:
    attempts: int
    physical_starts: int
    charges: int
    cache_hits: int
    unresolved_decisions: int
    prefix_order_pass: bool
    terminal_success_pass: bool


@dataclass(frozen=True)
class DuplicateLivenessCertificate:
    conditional_new_state_probability_lower_bound: Fraction
    attempts_per_service: int
    requested_charges: int
    per_service_failure_upper_bound: Fraction
    run_failure_upper_bound: Fraction
    expected_attempts_per_service_upper_bound: Fraction


def duplicate_liveness_certificate(
    *,
    conditional_new_state_probability_lower_bound: Fraction | int,
    retry_cap: int,
    fallback_cap: int,
    requested_charges: int,
) -> DuplicateLivenessCertificate:
    """Bound duplicate exhaustion under a predictable unseen-state probability floor.

    The premise is conditional: before every attempt in a proposal service, given
    the complete past and previous duplicate failures in that service, the next
    attempt is a new valid canonical state with probability at least ``q``.
    Independence is not required.
    """

    q = Fraction(conditional_new_state_probability_lower_bound)
    if not 0 < q <= 1:
        raise ValueError("The conditional new-state probability floor must lie in (0,1].")
    if retry_cap < 0 or fallback_cap < 0 or requested_charges < 0:
        raise ValueError("Caps and requested charges must be nonnegative.")
    attempts = 1 + retry_cap + fallback_cap
    failure = (1 - q) ** attempts
    run_failure = min(Fraction(1), requested_charges * failure)
    expected_attempts = (1 - (1 - q) ** attempts) / q
    return DuplicateLivenessCertificate(
        conditional_new_state_probability_lower_bound=q,
        attempts_per_service=attempts,
        requested_charges=requested_charges,
        per_service_failure_upper_bound=failure,
        run_failure_upper_bound=run_failure,
        expected_attempts_per_service_upper_bound=expected_attempts,
    )


def first_visit_sequence(states: Iterable[object]) -> tuple[object, ...]:
    """Collapse exact duplicate attempts to their ordered first-visit sequence."""

    seen: set[object] = set()
    output: list[object] = []
    for state in states:
        try:
            unseen = state not in seen
        except TypeError as error:
            raise TypeError("States must be hashable exact canonical identities.") from error
        if unseen:
            seen.add(state)
            output.append(state)
    return tuple(output)


def information_time_equivalent(
    left_attempt_states: Iterable[object],
    right_attempt_states: Iterable[object],
) -> bool:
    """Check equality of ordered first-visit histories under information time."""

    return first_visit_sequence(left_attempt_states) == first_visit_sequence(
        right_attempt_states
    )


def validate_prefix_accounting(
    *,
    attempts: int,
    physical_starts: int,
    charges: int,
    cache_hits: int,
    unresolved_decisions: int,
    terminal_success: bool,
) -> PrefixAccountingCertificate:
    values = (attempts, physical_starts, charges, cache_hits, unresolved_decisions)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise TypeError("Accounting fields must be nonnegative exact integers.")
    prefix = attempts >= physical_starts >= charges
    success = (
        prefix
        and physical_starts == charges
        and unresolved_decisions == 0
        and attempts == charges + cache_hits
    )
    return PrefixAccountingCertificate(
        attempts=attempts,
        physical_starts=physical_starts,
        charges=charges,
        cache_hits=cache_hits,
        unresolved_decisions=unresolved_decisions,
        prefix_order_pass=prefix,
        terminal_success_pass=(success if terminal_success else False),
    )


@dataclass(frozen=True)
class InterventionSummary:
    candidate: str
    primary_lower_bound_by_family: Mapping[str, float]
    adjacent_lower_bound_by_family: Mapping[str, float] | None
    median_by_family: Mapping[str, float]
    trimmed_mean_by_family: Mapping[str, float]
    wins_by_family: Mapping[str, int]
    losses_by_family: Mapping[str, int]


@dataclass(frozen=True)
class HierarchicalSelectionDecision:
    selected_candidate: str
    reached_candidates: tuple[str, ...]
    not_reached_candidates: tuple[str, ...]
    reasons: tuple[str, ...]


def complexity_first_selection(
    summaries: Sequence[InterventionSummary],
    *,
    family_ids: Sequence[str],
    primary_threshold: float,
    adjacent_threshold: float,
    simultaneous_coverage_certified: bool = False,
) -> HierarchicalSelectionDecision:
    """Apply a contiguous practical-effect gate to simultaneous one-sided bounds.

    The Boolean is intentionally fail-closed: marginal or unadjusted intervals
    cannot be passed off as a familywise mechanism-admission certificate.
    """

    if any(
        type(value) not in {int, float}
        for value in (primary_threshold, adjacent_threshold)
    ):
        raise TypeError("Practical thresholds must be finite real numbers.")
    if any(
        not math.isfinite(float(value))
        for value in (primary_threshold, adjacent_threshold)
    ):
        raise ValueError("Practical thresholds must be finite.")
    if primary_threshold < 0 or adjacent_threshold < 0:
        raise ValueError("Practical thresholds must be nonnegative.")
    if type(simultaneous_coverage_certified) is not bool:
        raise TypeError("simultaneous_coverage_certified must be an exact Boolean.")
    if not simultaneous_coverage_certified:
        raise ValueError(
            "The candidate bounds must have certified simultaneous one-sided coverage."
        )
    if not family_ids:
        raise ValueError("family_ids must be nonempty.")
    if any(type(family) is not str or not family.strip() for family in family_ids):
        raise TypeError("family_ids must contain nonempty exact strings.")
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("family_ids must be unique.")
    expected = ["C1", "C2", "C3"]
    by_id = {summary.candidate: summary for summary in summaries}
    if len(summaries) != len(expected) or set(by_id) != set(expected):
        raise ValueError("Exactly one summary for C1, C2, and C3 is required.")
    validated: dict[
        str,
        dict[str, tuple[float, float, float, int, int, float | None]],
    ] = {}
    for candidate in expected:
        summary = by_id[candidate]
        validated[candidate] = {}
        for family in family_ids:
            primary = summary.primary_lower_bound_by_family[family]
            median = summary.median_by_family[family]
            trimmed = summary.trimmed_mean_by_family[family]
            adjacent = None
            if summary.adjacent_lower_bound_by_family is not None:
                adjacent = summary.adjacent_lower_bound_by_family[family]
            effect_values = (primary, median, trimmed)
            if adjacent is not None:
                effect_values += (adjacent,)
            if any(type(value) not in {int, float} for value in effect_values):
                raise TypeError(
                    "Effect summaries must contain finite real numbers."
                )
            if any(not math.isfinite(float(value)) for value in effect_values):
                raise ValueError(
                    "Effect summaries must contain finite real numbers."
                )
            wins = summary.wins_by_family[family]
            losses = summary.losses_by_family[family]
            if type(wins) is not int or type(losses) is not int:
                raise TypeError("Win/loss counts must be exact nonnegative integers.")
            if wins < 0 or losses < 0:
                raise ValueError("Win/loss counts must be nonnegative.")
            validated[candidate][family] = (
                float(primary),
                float(median),
                float(trimmed),
                wins,
                losses,
                None if adjacent is None else float(adjacent),
            )
    selected = "C0"
    reached: list[str] = []
    reasons: list[str] = []
    blocked = False
    for candidate in expected:
        if blocked:
            continue
        reached.append(candidate)
        for family in family_ids:
            primary, median, trimmed, wins, losses, adjacent = validated[candidate][
                family
            ]
            primary_pass = (
                primary > primary_threshold
                and median > 0.0
                and trimmed > 0.0
                and wins > losses
            )
            if not primary_pass:
                reasons.append(f"{candidate}/{family}: primary gate failed")
                blocked = True
                break
            if candidate != "C1":
                if adjacent is None:
                    reasons.append(f"{candidate}/{family}: adjacent bound missing")
                    blocked = True
                    break
                if adjacent <= adjacent_threshold:
                    reasons.append(f"{candidate}/{family}: adjacent gate failed")
                    blocked = True
                    break
        if not blocked:
            selected = candidate
    not_reached = tuple(c for c in expected if c not in reached)
    return HierarchicalSelectionDecision(
        selected_candidate=selected,
        reached_candidates=tuple(reached),
        not_reached_candidates=not_reached,
        reasons=tuple(reasons),
    )


def finite_suite_estimand(case_seed_means: Sequence[float]) -> float:
    if not case_seed_means:
        raise ValueError("At least one case contrast is required.")
    if any(not math.isfinite(float(value)) for value in case_seed_means):
        raise ValueError("Case contrasts must be finite.")
    return sum(float(value) for value in case_seed_means) / len(case_seed_means)


def dominated_region_monotone(
    previous_points: Sequence[Sequence[float]],
    next_points: Sequence[Sequence[float]],
) -> bool:
    """Check the sufficient append-only premise used by the archive theorem."""

    previous = {tuple(float(x) for x in point) for point in previous_points}
    following = {tuple(float(x) for x in point) for point in next_points}
    return previous <= following


def replacement_potential_delta(
    parent_scalars: Sequence[float],
    candidate_scalars: Sequence[float],
) -> float:
    if len(parent_scalars) != len(candidate_scalars):
        raise ValueError("Parent/candidate scalar vectors must share a length.")
    delta = sum(float(y) - float(x) for x, y in zip(parent_scalars, candidate_scalars))
    if not math.isfinite(delta):
        raise ValueError("Potential delta must be finite.")
    return delta


def resource_rank_reversal_example() -> dict[str, object]:
    """Return an exact counterexample: first-true and attempt budgets can rank methods oppositely."""

    # Larger quality is better.  A proposes q=.90, duplicates it, then proposes 1.00.
    # C proposes .95 and .96.  Two first-evaluation charges reach A's third attempt,
    # whereas two attempt slots do not.
    return {
        "algorithm_A_attempt_qualities": [0.90, 0.90, 1.00],
        "algorithm_C_attempt_qualities": [0.95, 0.96],
        "two_first_true_evaluations": {"A": 1.00, "C": 0.96, "winner": "A"},
        "two_attempts": {"A": 0.90, "C": 0.96, "winner": "C"},
        "conclusion": "Resource definitions can reverse algorithm rankings; report A, P, B, and wall time.",
    }


__all__ = [
    "DuplicateLivenessCertificate",
    "duplicate_liveness_certificate",
    "first_visit_sequence",
    "information_time_equivalent",
    "AttemptBudgetCertificate",
    "HierarchicalSelectionDecision",
    "InterventionSummary",
    "PrefixAccountingCertificate",
    "complexity_first_selection",
    "dominated_region_monotone",
    "finite_suite_estimand",
    "replacement_potential_delta",
    "resource_rank_reversal_example",
    "successful_attempt_bound",
    "validate_prefix_accounting",
]
