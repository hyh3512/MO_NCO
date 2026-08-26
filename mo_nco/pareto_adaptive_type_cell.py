from __future__ import annotations

"""Fixed-confidence type--cell identification and exact confirm allocation.

The module implements a cell-separated Bernoulli branch.  It is intentionally
separate from a shared categorical endpoint stream so the adaptive best-type
proof and confirm-cost ledger have explicit product-sample semantics.
"""

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from typing import Mapping

from .pareto_independent_replica_certificate import canonical_rational_string, parse_canonical_probability

ADAPTIVE_TYPE_CELL_SCHEMA_V16 = "pareto_adaptive_type_cell_theorem_v16"
CONFIRM_ALLOCATION_SCHEMA_V16 = "pareto_exact_confirm_risk_allocation_v16"
TRANSPORT_LOWER_BOUND_SCHEMA_V16 = "pareto_type_identification_transport_lower_bound_v16"

class AdaptiveTypeCellError(ValueError):
    pass

def _probability(value: Fraction | str | int, *, label: str) -> Fraction:
    return parse_canonical_probability(value, label=label)

def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdaptiveTypeCellError(f"{label} must be a positive integer.")
    return value

def _ceil_log2_fraction(value: Fraction) -> int:
    if value <= 0:
        raise AdaptiveTypeCellError("ceil_log2 requires a positive value.")
    if value <= 1:
        return 0
    n, d = value.numerator, value.denominator
    exponent = max(0, n.bit_length() - d.bit_length())
    if d << exponent < n:
        exponent += 1
    while exponent > 0 and d << (exponent - 1) >= n:
        exponent -= 1
    return exponent

def _sqrt_upper(value: Fraction, *, bits: int = 256) -> Fraction:
    if value < 0:
        raise AdaptiveTypeCellError("Cannot upper-bound a negative square root.")
    if value == 0:
        return Fraction(0)
    scaled = -(-(value.numerator << (2 * bits)) // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << bits)

def anytime_hoeffding_radius_upper(samples_per_arm: int, *, type_count: int,
                                    cell_count: int,
                                    familywise_error: Fraction | str) -> Fraction:
    """Dyadic upper radius for the all-type/all-cell/all-time event.

    The failure allocation at time n is alpha/[R J n(n+1)].  Hoeffding and
    exp(-x)<=2**(-x) yield a fully rational pre-run radius.
    """
    n = _positive_int(samples_per_arm, label="samples_per_arm")
    r = _positive_int(type_count, label="type_count")
    j = _positive_int(cell_count, label="cell_count")
    alpha = _probability(familywise_error, label="familywise_error")
    if alpha <= 0 or alpha >= 1:
        raise AdaptiveTypeCellError("familywise_error must lie in (0,1).")
    exponent = _ceil_log2_fraction(Fraction(2 * r * j * n * (n + 1), 1) / alpha)
    return _sqrt_upper(Fraction(exponent, 2 * n))

@dataclass(frozen=True)
class CellIdentificationBound:
    cell_id: str
    best_type: str
    best_probability: str
    second_probability: str
    gap: str
    stopping_round_upper: int
    per_type_samples_upper: int

@dataclass(frozen=True)
class AdaptiveIdentificationCertificate:
    schema: str
    type_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    familywise_error: str
    confidence_schedule: str
    cell_bounds: tuple[CellIdentificationBound, ...]
    total_pilot_replicas_upper: int
    unique_best_required: bool
    theorem_scope: str
    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["cell_bounds"] = [asdict(item) for item in self.cell_bounds]
        return payload

def certify_balanced_successive_elimination_upper_bound(
    probabilities: Mapping[str, Mapping[str, Fraction | str | int]], *,
    familywise_error: Fraction | str, max_rounds: int = 1_000_000,
) -> AdaptiveIdentificationCertificate:
    """Gap-dependent upper bound for balanced successive elimination.

    For each cell, sample every active type once per round and eliminate r if
    U_r < max_s L_s.  On the simultaneous confidence event the best type is
    never removed, and all suboptimal types are removed once 4*c_n < gap.
    """
    if not probabilities:
        raise AdaptiveTypeCellError("At least one type is required.")
    type_ids = tuple(sorted(probabilities))
    first_cells: tuple[str, ...] | None = None
    matrix: dict[str, dict[str, Fraction]] = {}
    for type_id in type_ids:
        row = probabilities[type_id]
        if not row:
            raise AdaptiveTypeCellError("Every type must declare cell probabilities.")
        cells = tuple(sorted(row))
        if first_cells is None:
            first_cells = cells
        elif cells != first_cells:
            raise AdaptiveTypeCellError("All types must cover the same cell IDs.")
        matrix[type_id] = {cell: _probability(row[cell], label=f"p[{type_id},{cell}]")
                           for cell in cells}
    assert first_cells is not None
    if len(type_ids) < 2:
        raise AdaptiveTypeCellError("Best-type identification requires at least two types.")
    alpha = _probability(familywise_error, label="familywise_error")
    if alpha <= 0 or alpha >= 1:
        raise AdaptiveTypeCellError("familywise_error must lie in (0,1).")
    maximum = _positive_int(max_rounds, label="max_rounds")
    bounds: list[CellIdentificationBound] = []
    total = 0
    for cell in first_cells:
        ranking = sorted(((matrix[t][cell], t) for t in type_ids), key=lambda item: (-item[0], item[1]))
        best_probability, best_type = ranking[0]
        second_probability = ranking[1][0]
        gap = best_probability - second_probability
        if gap <= 0:
            raise AdaptiveTypeCellError(f"Cell {cell!r} does not have a unique best type.")
        stop: int | None = None
        for n in range(1, maximum + 1):
            radius = anytime_hoeffding_radius_upper(
                n, type_count=len(type_ids), cell_count=len(first_cells), familywise_error=alpha
            )
            if 4 * radius < gap:
                stop = n
                break
        if stop is None:
            raise AdaptiveTypeCellError(f"No certified stopping round through {maximum} for {cell!r}.")
        total += len(type_ids) * stop
        bounds.append(CellIdentificationBound(
            cell_id=cell, best_type=best_type,
            best_probability=canonical_rational_string(best_probability),
            second_probability=canonical_rational_string(second_probability),
            gap=canonical_rational_string(gap), stopping_round_upper=stop,
            per_type_samples_upper=stop,
        ))
    return AdaptiveIdentificationCertificate(
        schema=ADAPTIVE_TYPE_CELL_SCHEMA_V16, type_ids=type_ids, cell_ids=first_cells,
        familywise_error=canonical_rational_string(alpha),
        confidence_schedule="Hoeffding_alpha_over_RJ_n_nplus1_exact_dyadic_upper",
        cell_bounds=tuple(bounds), total_pilot_replicas_upper=total,
        unique_best_required=True,
        theorem_scope="cell_separated_independent_Bernoulli_pilot",
    )

@dataclass(frozen=True)
class ConfirmAllocationCertificate:
    schema: str
    cell_ids: tuple[str, ...]
    hit_probability_lower_bounds: tuple[str, ...]
    replicas_by_cell: tuple[int, ...]
    miss_probability_by_cell: tuple[str, ...]
    total_union_miss_upper: str
    target_union_miss_budget: str
    total_replicas: int
    exact_integer_optimum: bool
    allocation_rule: str
    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)

def exact_confirm_risk_allocation(
    hit_probability_lower_bounds: Mapping[str, Fraction | str | int], *,
    union_miss_budget: Fraction | str, max_total_replicas: int = 10_000_000,
) -> ConfirmAllocationCertificate:
    """Exact minimum integer allocation under the union miss bound.

    Cell j contributes a_j**m_j with a_j=1-q_j.  Marginal reduction
    q_j*a_j**m is nonincreasing.  Selecting the largest available marginal at
    every step is globally optimal for each total integer budget by exchange.
    """
    if not hit_probability_lower_bounds:
        raise AdaptiveTypeCellError("At least one cell is required.")
    cell_ids = tuple(sorted(hit_probability_lower_bounds))
    q = tuple(_probability(hit_probability_lower_bounds[cell], label=f"q[{cell}]")
              for cell in cell_ids)
    if any(value <= 0 for value in q):
        raise AdaptiveTypeCellError("Every cell needs a strictly positive hit lower bound.")
    delta = _probability(union_miss_budget, label="union_miss_budget")
    if delta <= 0 or delta >= 1:
        raise AdaptiveTypeCellError("union_miss_budget must lie in (0,1).")
    maximum = _positive_int(max_total_replicas, label="max_total_replicas")
    bases = tuple(1 - value for value in q)
    replicas = [0] * len(cell_ids)
    misses = [Fraction(1) for _ in cell_ids]
    risk = Fraction(len(cell_ids))
    steps = 0
    risk_before_last: Fraction | None = None
    while risk > delta:
        if steps >= maximum:
            raise AdaptiveTypeCellError("Exact allocation exceeds max_total_replicas.")
        marginal = [q[index] * misses[index] for index in range(len(cell_ids))]
        chosen = max(range(len(cell_ids)), key=lambda index: (marginal[index], -index))
        risk_before_last = risk
        old = misses[chosen]
        misses[chosen] *= bases[chosen]
        risk -= old - misses[chosen]
        replicas[chosen] += 1
        steps += 1
    if steps > 0 and (risk_before_last is None or risk_before_last <= delta):
        raise AssertionError("Allocation was not a first-crossing optimum.")
    return ConfirmAllocationCertificate(
        schema=CONFIRM_ALLOCATION_SCHEMA_V16, cell_ids=cell_ids,
        hit_probability_lower_bounds=tuple(canonical_rational_string(v) for v in q),
        replicas_by_cell=tuple(replicas),
        miss_probability_by_cell=tuple(canonical_rational_string(v) for v in misses),
        total_union_miss_upper=canonical_rational_string(risk),
        target_union_miss_budget=canonical_rational_string(delta),
        total_replicas=sum(replicas), exact_integer_optimum=True,
        allocation_rule="largest_exact_marginal_miss_reduction_lexicographic_ties",
    )

def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)

def _bernoulli_kl(p: Fraction, q: Fraction) -> Decimal:
    with localcontext() as context:
        context.prec = 110
        pd, qd, one = _decimal_fraction(p), _decimal_fraction(q), Decimal(1)
        if p == 0:
            return -(one - pd) * (one - qd).ln()
        if p == 1:
            return -pd * qd.ln()
        return pd * (pd / qd).ln() + (one - pd) * ((one - pd) / (one - qd)).ln()

@dataclass(frozen=True)
class TransportLowerBound:
    schema: str
    cell_id: str
    best_type: str
    challenger_type: str
    best_probability: str
    challenger_probability: str
    error_probability: str
    binary_decision_kl: str
    midpoint_max_arm_kl: str
    expected_total_pilot_samples_lower: str
    theorem_scope: str
    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)

def pairwise_transportation_lower_bound(*, cell_id: str, best_type: str,
                                         challenger_type: str,
                                         best_probability: Fraction | str | int,
                                         challenger_probability: Fraction | str | int,
                                         error_probability: Fraction | str) -> TransportLowerBound:
    """Change-of-measure lower bound for any alpha-correct identifier."""
    p_star = _probability(best_probability, label="best_probability")
    p_other = _probability(challenger_probability, label="challenger_probability")
    alpha = _probability(error_probability, label="error_probability")
    if not (0 < alpha < Fraction(1, 2)):
        raise AdaptiveTypeCellError("error_probability must lie in (0,1/2).")
    if not p_star > p_other:
        raise AdaptiveTypeCellError("best_probability must exceed challenger_probability.")
    midpoint = (p_star + p_other) / 2
    decision_kl = _bernoulli_kl(1 - alpha, alpha)
    max_arm_kl = max(_bernoulli_kl(p_star, midpoint), _bernoulli_kl(p_other, midpoint))
    with localcontext() as context:
        context.prec = 90
        lower = decision_kl / max_arm_kl
        lower_text = format(lower.next_minus(), "f")
        decision_text = format(decision_kl.next_minus(), "f")
        arm_text = format(max_arm_kl.next_plus(), "f")
    return TransportLowerBound(
        schema=TRANSPORT_LOWER_BOUND_SCHEMA_V16, cell_id=cell_id,
        best_type=best_type, challenger_type=challenger_type,
        best_probability=canonical_rational_string(p_star),
        challenger_probability=canonical_rational_string(p_other),
        error_probability=canonical_rational_string(alpha),
        binary_decision_kl=decision_text, midpoint_max_arm_kl=arm_text,
        expected_total_pilot_samples_lower=lower_text,
        theorem_scope="any_alpha_correct_adaptive_Bernoulli_best_type_identifier",
    )

__all__ = [
    "ADAPTIVE_TYPE_CELL_SCHEMA_V16", "CONFIRM_ALLOCATION_SCHEMA_V16",
    "TRANSPORT_LOWER_BOUND_SCHEMA_V16", "AdaptiveIdentificationCertificate",
    "AdaptiveTypeCellError", "CellIdentificationBound", "ConfirmAllocationCertificate",
    "TransportLowerBound", "anytime_hoeffding_radius_upper",
    "certify_balanced_successive_elimination_upper_bound",
    "exact_confirm_risk_allocation", "pairwise_transportation_lower_bound",
]
