"""Finite-deadline certificate for shared-categorical Track-and-Stop v19.1.

The asymptotic Track-and-Stop theorem requires regularity and a quantitative
information-rate tail argument to control expected stopping.  This module
provides a separate nonasymptotic, branch-scoped fallback.  It assumes a
certified positive winner gap and deterministic lower bounds on the number of
samples allocated to every type by a declared deadline.

For R types and J cells, choose the smallest integer q_n with

    2**q_n >= 2 R J n(n+1) / alpha,

and define c_n^2 = q_n/(2n).  Hoeffding and a union bound over all
(type, cell, n) imply the simultaneous event

    |p_hat_rj(n)-p_rj| <= c_n  for every r,j,n

with probability at least 1-alpha.  If, at a deadline T, every certified best
arm a_j and challenger s satisfy

    2(c_{N_a(T)} + c_{N_s(T)}) < Delta_{j,s},

then the corresponding confidence intervals are strictly separated and the
fallback answer is correct on that event.  The comparison of sums of square
roots is performed exactly with rational arithmetic.

A hybrid procedure that stops early with a delta_glr-correct GLR rule and,
otherwise, returns the deadline-separated answer is (delta_glr+alpha)-correct
and has deterministic pull and evaluation-cost caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


class TrackStopDeadlineError(ValueError):
    pass


def as_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TrackStopDeadlineError("boolean is not a rational scalar")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction.from_float(value)
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise TrackStopDeadlineError(f"invalid rational scalar: {value!r}") from exc


def ceil_log2_fraction(value: Fraction | int | str) -> int:
    x = as_fraction(value)
    if x <= 0:
        raise TrackStopDeadlineError("ceil_log2 input must be positive")
    if x <= 1:
        return 0
    q = max(0, x.numerator.bit_length() - x.denominator.bit_length())
    while Fraction(1 << q, 1) < x:
        q += 1
    while q > 0 and Fraction(1 << (q - 1), 1) >= x:
        q -= 1
    return q


def confidence_radius_squared(
    sample_count: int,
    *,
    num_types: int,
    num_cells: int,
    alpha: Fraction | int | float | str,
) -> Fraction:
    if not isinstance(sample_count, int) or sample_count <= 0:
        raise TrackStopDeadlineError("sample_count must be positive")
    if not isinstance(num_types, int) or num_types <= 0:
        raise TrackStopDeadlineError("num_types must be positive")
    if not isinstance(num_cells, int) or num_cells <= 0:
        raise TrackStopDeadlineError("num_cells must be positive")
    alpha_q = as_fraction(alpha)
    if not (Fraction(0, 1) < alpha_q < Fraction(1, 1)):
        raise TrackStopDeadlineError("alpha must lie in (0,1)")
    target = Fraction(
        2 * num_types * num_cells * sample_count * (sample_count + 1),
        1,
    ) / alpha_q
    q = ceil_log2_fraction(target)
    return Fraction(q, 2 * sample_count)


def _sqrt_sum_strict_less(a: Fraction, b: Fraction, bound: Fraction) -> bool:
    """Decide sqrt(a)+sqrt(b) < bound exactly over nonnegative rationals."""

    if a < 0 or b < 0 or bound <= 0:
        return False
    residual = bound * bound - a - b
    if residual <= 0:
        return False
    return residual * residual > 4 * a * b


def strict_pair_separation(
    sample_count_a: int,
    sample_count_b: int,
    gap_lower: Fraction | int | float | str,
    *,
    num_types: int,
    num_cells: int,
    alpha: Fraction | int | float | str,
) -> bool:
    gap = as_fraction(gap_lower)
    if gap <= 0:
        raise TrackStopDeadlineError("gap lower bound must be positive")
    a = confidence_radius_squared(
        sample_count_a,
        num_types=num_types,
        num_cells=num_cells,
        alpha=alpha,
    )
    b = confidence_radius_squared(
        sample_count_b,
        num_types=num_types,
        num_cells=num_cells,
        alpha=alpha,
    )
    # 2(c_a+c_b) < gap  iff  c_a+c_b < gap/2.
    return _sqrt_sum_strict_less(a, b, gap / 2)


def allocation_count_lower(
    deadline: int,
    share_lower: Fraction | int | float | str,
    tracking_deficit: int,
) -> int:
    if not isinstance(deadline, int) or deadline < 0:
        raise TrackStopDeadlineError("deadline must be a nonnegative integer")
    share = as_fraction(share_lower)
    if share < 0 or share > 1:
        raise TrackStopDeadlineError("allocation share lower must lie in [0,1]")
    if not isinstance(tracking_deficit, int) or tracking_deficit < 0:
        raise TrackStopDeadlineError("tracking deficit must be nonnegative")
    floor_share = (share.numerator * deadline) // share.denominator
    return max(0, floor_share - tracking_deficit)


@dataclass(frozen=True)
class TrackStopDeadlineCertificate:
    num_types: int
    num_cells: int
    best_types: tuple[int, ...]
    gap_lower: tuple[tuple[Fraction, ...], ...]
    allocation_share_lower: tuple[Fraction, ...]
    tracking_deficit: tuple[int, ...]
    alpha_fallback: Fraction
    delta_glr: Fraction
    deadline: int
    sample_count_lower_at_deadline: tuple[int, ...]
    arm_costs: tuple[Fraction, ...]
    evaluation_cost_upper: Fraction
    total_error_upper: Fraction
    all_pairs_strictly_separated: bool
    pass_gate: bool
    semantics: str = "time_uniform_hoeffding_deadline_fallback_v19_1"

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "num_types": self.num_types,
            "num_cells": self.num_cells,
            "best_types": list(self.best_types),
            "gap_lower": [[str(x) for x in row] for row in self.gap_lower],
            "allocation_share_lower": [str(x) for x in self.allocation_share_lower],
            "tracking_deficit": list(self.tracking_deficit),
            "alpha_fallback": str(self.alpha_fallback),
            "delta_glr": str(self.delta_glr),
            "deadline": self.deadline,
            "sample_count_lower_at_deadline": list(self.sample_count_lower_at_deadline),
            "arm_costs": [str(x) for x in self.arm_costs],
            "evaluation_cost_upper": str(self.evaluation_cost_upper),
            "total_error_upper": str(self.total_error_upper),
            "all_pairs_strictly_separated": self.all_pairs_strictly_separated,
            "pass_gate": self.pass_gate,
            "claim": (
                "hybrid GLR/deadline procedure is delta_glr+alpha_fallback correct "
                "under the declared i.i.d. categorical law, gap, and tracking-share contracts"
            ),
            "not_claimed": [
                "asymptotic instance optimality",
                "automatic validity of the supplied gap lower bounds",
                "automatic validity of the tracking-share lower bounds",
            ],
        }


def build_track_stop_deadline_certificate(
    *,
    best_types: Sequence[int],
    gap_lower: Sequence[Sequence[Fraction | int | float | str]],
    allocation_share_lower: Sequence[Fraction | int | float | str],
    tracking_deficit: Sequence[int],
    alpha_fallback: Fraction | int | float | str,
    delta_glr: Fraction | int | float | str,
    arm_costs: Sequence[Fraction | int | float | str] | None = None,
    max_deadline: int = 10_000_000,
) -> TrackStopDeadlineCertificate:
    best = tuple(int(x) for x in best_types)
    gaps = tuple(tuple(as_fraction(x) for x in row) for row in gap_lower)
    shares = tuple(as_fraction(x) for x in allocation_share_lower)
    deficits = tuple(int(x) for x in tracking_deficit)
    if not best or not shares:
        raise TrackStopDeadlineError("best_types and allocation shares must be nonempty")
    r_count = len(shares)
    j_count = len(best)
    if len(deficits) != r_count or len(gaps) != j_count:
        raise TrackStopDeadlineError("deadline inputs are dimensionally inconsistent")
    if any(len(row) != r_count for row in gaps):
        raise TrackStopDeadlineError("one gap value is required per cell/type pair")
    if any(a < 0 or a > 1 for a in shares) or sum(shares, Fraction(0, 1)) > 1:
        raise TrackStopDeadlineError("allocation share lower bounds must be nonnegative and sum to at most one")
    if any(x < 0 for x in deficits):
        raise TrackStopDeadlineError("tracking deficits must be nonnegative")
    for j, winner in enumerate(best):
        if winner < 0 or winner >= r_count:
            raise TrackStopDeadlineError("best type index escaped the type set")
        for r in range(r_count):
            if r == winner:
                if gaps[j][r] != 0:
                    raise TrackStopDeadlineError("winner self-gap must equal zero")
            elif gaps[j][r] <= 0:
                raise TrackStopDeadlineError("every challenger gap lower bound must be positive")
    alpha = as_fraction(alpha_fallback)
    delta = as_fraction(delta_glr)
    if not (0 < alpha < 1) or not (0 <= delta < 1) or alpha + delta >= 1:
        raise TrackStopDeadlineError("invalid fallback/GLR error allocation")
    if not isinstance(max_deadline, int) or max_deadline <= 0:
        raise TrackStopDeadlineError("max_deadline must be positive")
    costs = (
        (Fraction(1, 1),) * r_count
        if arm_costs is None
        else tuple(as_fraction(x) for x in arm_costs)
    )
    if len(costs) != r_count or any(x <= 0 for x in costs):
        raise TrackStopDeadlineError("one positive cost is required per type")

    def separated_at(deadline: int) -> tuple[bool, tuple[int, ...]]:
        counts = tuple(
            allocation_count_lower(deadline, shares[r], deficits[r])
            for r in range(r_count)
        )
        if any(counts[r] <= 0 for r in range(r_count)):
            return False, counts
        for j, winner in enumerate(best):
            for challenger in range(r_count):
                if challenger == winner:
                    continue
                if not strict_pair_separation(
                    counts[winner],
                    counts[challenger],
                    gaps[j][challenger],
                    num_types=r_count,
                    num_cells=j_count,
                    alpha=alpha,
                ):
                    return False, counts
        return True, counts

    # Separation is monotone in the deadline because every certified sample
    # lower bound is nondecreasing and every confidence radius is nonincreasing.
    lo = 0
    hi = 1
    ok, hi_counts = separated_at(hi)
    while not ok and hi < max_deadline:
        lo = hi
        hi = min(max_deadline, 2 * hi)
        ok, hi_counts = separated_at(hi)
    if not ok:
        raise TrackStopDeadlineError("no certified deadline exists under max_deadline")
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        mid_ok, mid_counts = separated_at(mid)
        if mid_ok:
            hi = mid
            hi_counts = mid_counts
        else:
            lo = mid
    chosen_deadline = hi
    chosen_counts = hi_counts

    # The exact type sequence may cost less, but without the sequence the
    # deterministic safe upper bound is max_r c_r times the number of pulls.
    evaluation_cost_upper = max(costs) * chosen_deadline
    return TrackStopDeadlineCertificate(
        num_types=r_count,
        num_cells=j_count,
        best_types=best,
        gap_lower=gaps,
        allocation_share_lower=shares,
        tracking_deficit=deficits,
        alpha_fallback=alpha,
        delta_glr=delta,
        deadline=chosen_deadline,
        sample_count_lower_at_deadline=chosen_counts,
        arm_costs=costs,
        evaluation_cost_upper=evaluation_cost_upper,
        total_error_upper=alpha + delta,
        all_pairs_strictly_separated=True,
        pass_gate=True,
    )


__all__ = [
    "TrackStopDeadlineCertificate",
    "TrackStopDeadlineError",
    "allocation_count_lower",
    "as_fraction",
    "build_track_stop_deadline_certificate",
    "ceil_log2_fraction",
    "confidence_radius_squared",
    "strict_pair_separation",
]
