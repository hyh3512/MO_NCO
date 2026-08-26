from __future__ import annotations

"""Development-only information-time and archive-Lyapunov theory helpers.

These objects are intentionally separate from the immutable V4/V6/V7/V8
scientific artifacts.  They define a possible successor mechanism; importing or
executing this module does not authorize selection, confirmation, a formal
study, or an empirical claim.
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence, TypeVar


T = TypeVar("T")


_MAX_NUMERICAL_TOLERANCE = 1e-6


def _strict_real(value: object, *, label: str) -> float:
    """Return one finite built-in real without permissive coercion."""

    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an exact int or float, excluding bool.")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{label} must be finite.")
    return output


def _strict_tolerance(value: object) -> float:
    tolerance = _strict_real(value, label="tolerance")
    if tolerance < 0.0 or tolerance > _MAX_NUMERICAL_TOLERANCE:
        raise ValueError(
            "tolerance must lie in [0, 1e-6]; larger values do not define a "
            "numerical-boundary tolerance."
        )
    return tolerance


def _normalized_unit_interval(
    value: object,
    *,
    label: str,
    tolerance: float,
) -> float:
    """Validate a normalized quantity and clamp boundary-scale roundoff only."""

    output = _strict_real(value, label=label)
    if output < -tolerance or output > 1.0 + tolerance:
        raise ValueError(
            f"{label} must lie in [0, 1] up to the declared numerical tolerance."
        )
    return min(1.0, max(0.0, output))


@dataclass(frozen=True)
class InformationTimePath:
    first_visit_attempt_indices: tuple[int, ...]
    first_visit_states: tuple[object, ...]
    total_attempts: int


def information_time_path(states: Iterable[T]) -> InformationTimePath:
    """Return the ordered first-visit subsequence of canonical hashable states.

    Attempt indices are one-based.  Duplicate insertions are discarded but their
    cost remains visible through ``total_attempts``.  Inputs must already be
    exact canonical identities: this helper does not infer domain equivalences.
    """

    seen: set[T] = set()
    indices: list[int] = []
    first: list[T] = []
    total = 0
    for total, state in enumerate(states, start=1):
        try:
            unseen = state not in seen
        except TypeError as error:  # pragma: no cover - defensive boundary
            raise TypeError("Information-time states must be hashable.") from error
        if unseen:
            seen.add(state)
            indices.append(total)
            first.append(state)
    return InformationTimePath(tuple(indices), tuple(first), total)


def information_time_equivalent(left: Iterable[T], right: Iterable[T]) -> bool:
    """Whether two attempt histories have the same ordered first visits."""

    return information_time_path(left).first_visit_states == information_time_path(
        right
    ).first_visit_states


@dataclass(frozen=True)
class OperatorProductivity:
    attempts: int
    new_states: int
    total_quality_gain: float
    elapsed_seconds: float | None
    unseen_rate: float
    conditional_gain_per_new_state: float
    gain_per_attempt: float
    gain_per_second: float | None
    factorization_residual: float


def operator_productivity(
    *,
    attempts: int,
    new_states: int,
    total_quality_gain: float,
    elapsed_seconds: float | None = None,
    tolerance: float = 1e-12,
) -> OperatorProductivity:
    """Compute the exact empirical decomposition gain/attempt = q_hat * m_hat.

    ``q_hat`` is the first-visit rate and ``m_hat`` is the mean quality gain
    conditional on a first visit.  The identity shows why duplicate reduction
    alone is not a quality theorem: a larger q can be offset by a smaller m.
    ``total_quality_gain`` is the cumulative normalized archive gain and must
    lie in ``[0, 1]`` up to ``tolerance``.  When ``new_states == 0``, the
    conditional mean is not statistically identified; the returned zero is an
    explicit computational convention that preserves the empirical identity.
    """

    if type(attempts) is not int or attempts < 0:
        raise TypeError("attempts must be a nonnegative exact integer.")
    if type(new_states) is not int or new_states < 0:
        raise TypeError("new_states must be a nonnegative exact integer.")
    if new_states > attempts:
        raise ValueError("new_states cannot exceed attempts.")
    tol = _strict_tolerance(tolerance)
    gain = _normalized_unit_interval(
        total_quality_gain,
        label="total_quality_gain",
        tolerance=tol,
    )
    if new_states == 0:
        if gain > tol:
            raise ValueError(
                "total_quality_gain must be zero when no new state was evaluated."
            )
        gain = 0.0
    seconds = (
        None
        if elapsed_seconds is None
        else _strict_real(elapsed_seconds, label="elapsed_seconds")
    )
    if seconds is not None and seconds <= 0.0:
        raise ValueError("elapsed_seconds must be positive when supplied.")
    unseen_rate = new_states / attempts if attempts else 0.0
    conditional_gain = gain / new_states if new_states else 0.0
    per_attempt = gain / attempts if attempts else 0.0
    product = unseen_rate * conditional_gain
    return OperatorProductivity(
        attempts=attempts,
        new_states=new_states,
        total_quality_gain=gain,
        elapsed_seconds=seconds,
        unseen_rate=unseen_rate,
        conditional_gain_per_new_state=conditional_gain,
        gain_per_attempt=per_attempt,
        gain_per_second=(None if seconds is None else gain / seconds),
        factorization_residual=per_attempt - product,
    )


@dataclass(frozen=True)
class DualResourceBudget:
    first_evaluation_cap: int
    attempt_cap: int
    screening_cap: int
    wall_time_cap_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("first_evaluation_cap", self.first_evaluation_cap),
            ("attempt_cap", self.attempt_cap),
            ("screening_cap", self.screening_cap),
        ):
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be a nonnegative exact integer.")
        if self.wall_time_cap_seconds is not None:
            value = _strict_real(
                self.wall_time_cap_seconds,
                label="wall_time_cap_seconds",
            )
            if value <= 0.0:
                raise ValueError("wall_time_cap_seconds must be positive.")

    def permits(
        self,
        *,
        first_evaluations: int,
        attempts: int,
        screenings: int,
        elapsed_seconds: float | None = None,
    ) -> bool:
        values = (first_evaluations, attempts, screenings)
        if any(type(value) is not int or value < 0 for value in values):
            raise TypeError("Observed resource counts must be nonnegative exact integers.")
        elapsed = (
            None
            if elapsed_seconds is None
            else _strict_real(elapsed_seconds, label="elapsed_seconds")
        )
        if elapsed is not None and elapsed < 0.0:
            raise ValueError("elapsed_seconds must be nonnegative.")
        if first_evaluations > self.first_evaluation_cap:
            return False
        if attempts > self.attempt_cap or screenings > self.screening_cap:
            return False
        if self.wall_time_cap_seconds is not None:
            if elapsed is None:
                return False
            if elapsed > float(self.wall_time_cap_seconds):
                return False
        return True


@dataclass(frozen=True)
class CandidateScreenDecision:
    selected: object
    selected_rank: int
    candidates_examined: int
    cached_candidates_skipped: int
    exhausted: bool


def select_first_unseen(
    candidates: Iterable[T],
    *,
    is_seen: Callable[[T], bool],
    cap: int,
) -> CandidateScreenDecision:
    """Select the first exact unseen candidate within a bounded ordered screen.

    The routine performs no objective evaluation.  ``candidates_examined`` is a
    separate structural-work resource and must not be silently merged with true
    evaluations or proposal attempts.
    """

    if type(cap) is not int or cap <= 0:
        raise TypeError("cap must be a positive exact integer.")
    examined = 0
    cached = 0
    last: object = None
    have_last = False
    for rank, candidate in enumerate(candidates):
        if examined >= cap:
            break
        examined += 1
        last = candidate
        have_last = True
        seen = is_seen(candidate)
        if type(seen) is not bool:
            raise TypeError("is_seen must return an exact bool.")
        if seen:
            cached += 1
            continue
        return CandidateScreenDecision(
            selected=candidate,
            selected_rank=rank,
            candidates_examined=examined,
            cached_candidates_skipped=cached,
            exhausted=False,
        )
    if not have_last:
        raise ValueError("Candidate screen received no candidate.")
    return CandidateScreenDecision(
        selected=last,
        selected_rank=max(0, examined - 1),
        candidates_examined=examined,
        cached_candidates_skipped=cached,
        exhausted=True,
    )


@dataclass(frozen=True)
class ArchiveCompensatedReplacementDecision:
    selected_targets: tuple[int, ...]
    scalar_delta_by_target: tuple[tuple[int, float], ...]
    scalar_delta_sum: float
    positive_scalar_worsening_sum: float
    normalized_hv_gain: float
    tradeoff_lambda: float
    archive_credit: float
    composite_potential_change: float
    certified_nonincrease: bool


def archive_compensated_replacement(
    scalar_delta_by_target: Mapping[int, float],
    *,
    normalized_hv_gain: float,
    tradeoff_lambda: float,
    max_targets: int | None = None,
    tolerance: float = 1e-12,
) -> ArchiveCompensatedReplacementDecision:
    """Choose a deterministic target subset with a Lyapunov certificate.

    Let ``delta_s = U_s(y)-U_s(x_s)`` and let ``dH`` be the nonnegative
    normalized all-evaluated archive hypervolume gain.  In exact arithmetic
    (``tolerance == 0``), the returned subset S satisfies

        sum_{s in S} delta_s <= lambda * dH,

    hence the composite potential

        Psi(X,A) = sum_s U_s(x_s) - lambda * HV(A)

    is nonincreasing for the replacement event.  With nonzero ``tolerance``,
    the certified inequalities have an additive tolerance per event; callers
    must accumulate that slack over a run.  Nonpositive deltas are selected
    first; positive deltas are admitted in ascending order only while paid by
    archive credit.  This is a local invariant, not a global convergence or
    superiority theorem.
    """

    tol = _strict_tolerance(tolerance)
    gain = _normalized_unit_interval(
        normalized_hv_gain,
        label="normalized_hv_gain",
        tolerance=tol,
    )
    lam = _strict_real(tradeoff_lambda, label="tradeoff_lambda")
    if lam < 0.0:
        raise ValueError("tradeoff_lambda must be nonnegative.")
    if max_targets is None:
        cap = len(scalar_delta_by_target)
    else:
        if type(max_targets) is not int or max_targets < 0:
            raise TypeError("max_targets must be a nonnegative exact integer.")
        cap = max_targets
    items: list[tuple[int, float]] = []
    for target, raw_delta in scalar_delta_by_target.items():
        if type(target) is not int or target < 0:
            raise TypeError("Target identifiers must be nonnegative exact integers.")
        delta = _strict_real(raw_delta, label="Scalar deltas")
        items.append((target, delta))
    if len({target for target, _ in items}) != len(items):
        raise ValueError("Target identifiers must be unique.")
    nonpositive = sorted(
        (item for item in items if item[1] <= 0.0), key=lambda item: (item[1], item[0])
    )
    positive = sorted(
        (item for item in items if item[1] > 0.0), key=lambda item: (item[1], item[0])
    )
    selected: list[tuple[int, float]] = []
    total = 0.0
    positive_total = 0.0
    credit = lam * gain
    for item in nonpositive:
        if len(selected) >= cap:
            break
        selected.append(item)
        total += item[1]
    for item in positive:
        if len(selected) >= cap:
            break
        if positive_total + item[1] <= credit + tol:
            selected.append(item)
            total += item[1]
            positive_total += item[1]
    change = total - credit
    certified = change <= tol
    if not certified:  # pragma: no cover - internal algebra guard
        raise RuntimeError("Archive-compensated replacement violated its certificate.")
    return ArchiveCompensatedReplacementDecision(
        selected_targets=tuple(target for target, _ in selected),
        scalar_delta_by_target=tuple(sorted(items)),
        scalar_delta_sum=total,
        positive_scalar_worsening_sum=positive_total,
        normalized_hv_gain=gain,
        tradeoff_lambda=lam,
        archive_credit=credit,
        composite_potential_change=change,
        certified_nonincrease=certified,
    )


def composite_potential(
    population_scalar_values: Sequence[float],
    *,
    normalized_hypervolume: float,
    tradeoff_lambda: float,
    tolerance: float = 1e-12,
) -> float:
    tol = _strict_tolerance(tolerance)
    values = tuple(
        _strict_real(value, label="Population scalar values")
        for value in population_scalar_values
    )
    hv = _normalized_unit_interval(
        normalized_hypervolume,
        label="normalized_hypervolume",
        tolerance=tol,
    )
    lam = _strict_real(tradeoff_lambda, label="tradeoff_lambda")
    if lam < 0.0:
        raise ValueError("tradeoff_lambda must be nonnegative.")
    return sum(values) - lam * hv
