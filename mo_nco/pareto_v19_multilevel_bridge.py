r"""Certified multilevel rare-cell bridges for Pareto-SMC v19.1.

A direct hit of a rare target set with probability ``q`` requires order
``log(1/delta)/q`` trials.  A structural alternative is a frozen nested bridge

    A_0 \supseteq A_1 \supseteq ... \supseteq A_L,

with a separately certified transition probability at each level.  Suppose a
valid witness in ``A_l`` is available and, for every attempted transition to
``A_{l+1}``, the conditional success probability given the complete preceding
history and all previous failures is at least ``a_l``.  Independence is not
needed: the probability that all ``m_l`` trials fail is at most
``(1-a_l)**m_l``.

Because levels are executed sequentially only after all previous levels have
succeeded, the end-to-end success probability is at least the product of the
per-level success lower bounds.  Hence the sharp compositional certificate is

    1 - prod_l (1 - (1-a_l)**m_l),

which is never larger than the earlier union bound.  The module retains the
union bound as a diagnostic and uses exact rational arithmetic throughout.

The bridge is a new structural assumption.  Hashes bind set and transition
artifacts, but do not prove nesting or transition probabilities; the explicit
provenance fields keep those external proof obligations visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence
import heapq
import re


class MultilevelBridgeError(ValueError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TRANSITION_PROVENANCE = {
    "locally_recomputed_finite_transition",
    "independently_verified_transition_bound",
    "theorem_parameter_conditional_transition",
}
_ALLOWED_NESTING_PROVENANCE = {
    "locally_recomputed_finite_set_inclusion",
    "independently_verified_nested_set_relation",
    "theorem_parameter_conditional_nested_sets",
}
_TRIAL_CONTRACTS = {
    # Preferred v2 contract.  It is weaker than independence and is exactly
    # what the product-of-failure proof needs.
    "predictable_conditional_success_lower_given_valid_parent_v2",
    # Legacy contract is accepted because it implies the v2 condition.
    "conditionally_independent_private_trials_given_valid_parent_v1",
}


def as_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise MultilevelBridgeError("boolean is not a rational scalar")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction.from_float(value)
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise MultilevelBridgeError(f"invalid rational scalar: {value!r}") from exc


def exact_geometric_failure(success_lower: Fraction, trials: int) -> Fraction:
    if success_lower <= 0 or success_lower > 1:
        raise MultilevelBridgeError("success lower bound must lie in (0,1]")
    if not isinstance(trials, int) or trials < 0:
        raise MultilevelBridgeError("trial count must be a nonnegative integer")
    return (Fraction(1, 1) - success_lower) ** trials


def sequential_bridge_failure_upper(
    conditional_success_lower: Sequence[Fraction | int | float | str],
    trials: Sequence[int],
) -> Fraction:
    """Exact rational sequential-composition upper bound.

    If level ``l`` succeeds with conditional probability at least
    ``1-(1-a_l)**m_l`` given all preceding level successes, the chain rule gives
    an all-level success lower bound equal to the product of these terms.
    """

    success = tuple(as_fraction(x) for x in conditional_success_lower)
    counts = tuple(trials)
    if not success or len(success) != len(counts):
        raise MultilevelBridgeError("one trial count is required per bridge level")
    all_success_lower = Fraction(1, 1)
    for a, m in zip(success, counts, strict=True):
        all_success_lower *= 1 - exact_geometric_failure(a, m)
    return 1 - all_success_lower


def union_bridge_failure_upper(
    conditional_success_lower: Sequence[Fraction | int | float | str],
    trials: Sequence[int],
) -> Fraction:
    success = tuple(as_fraction(x) for x in conditional_success_lower)
    counts = tuple(trials)
    if not success or len(success) != len(counts):
        raise MultilevelBridgeError("one trial count is required per bridge level")
    return min(
        Fraction(1, 1),
        sum(
            (exact_geometric_failure(a, m) for a, m in zip(success, counts, strict=True)),
            Fraction(0, 1),
        ),
    )


def minimum_trials_for_failure(
    success_lower: Fraction | int | float | str,
    failure_budget: Fraction | int | float | str,
) -> int:
    """Return the exact minimal ``m`` with ``(1-a)^m <= delta``."""

    a = as_fraction(success_lower)
    delta = as_fraction(failure_budget)
    if not (Fraction(0, 1) < a <= Fraction(1, 1)):
        raise MultilevelBridgeError("success lower bound must lie in (0,1]")
    if not (Fraction(0, 1) < delta < Fraction(1, 1)):
        raise MultilevelBridgeError("failure budget must lie in (0,1)")
    if a == 1:
        return 1
    base = 1 - a
    lo, hi = 0, 1
    while base**hi > delta:
        lo, hi = hi, 2 * hi
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if base**mid <= delta:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class BridgeLevel:
    level_id: str
    conditional_success_lower: Fraction
    trials: int
    evaluation_cost_per_trial: Fraction
    parent_set_sha256: str
    child_set_sha256: str
    transition_proof_sha256: str
    transition_proof_provenance: str = "theorem_parameter_conditional_transition"
    nesting_proof_provenance: str = "theorem_parameter_conditional_nested_sets"
    trial_contract: str = "predictable_conditional_success_lower_given_valid_parent_v2"

    def __post_init__(self) -> None:
        if not self.level_id:
            raise MultilevelBridgeError("level_id must be nonempty")
        if not (Fraction(0, 1) < self.conditional_success_lower <= 1):
            raise MultilevelBridgeError("conditional success lower must lie in (0,1]")
        if not isinstance(self.trials, int) or self.trials <= 0:
            raise MultilevelBridgeError("every bridge level needs a positive trial count")
        if self.evaluation_cost_per_trial <= 0:
            raise MultilevelBridgeError("evaluation cost must be positive")
        if _HEX64.fullmatch(self.parent_set_sha256) is None:
            raise MultilevelBridgeError("every bridge parent set needs a canonical SHA-256")
        if _HEX64.fullmatch(self.child_set_sha256) is None:
            raise MultilevelBridgeError("every bridge child set needs a canonical SHA-256")
        if self.parent_set_sha256 == self.child_set_sha256:
            raise MultilevelBridgeError("a bridge level must change the certified set artifact")
        if _HEX64.fullmatch(self.transition_proof_sha256) is None:
            raise MultilevelBridgeError("every bridge level needs a transition proof SHA-256")
        if self.transition_proof_provenance not in _ALLOWED_TRANSITION_PROVENANCE:
            raise MultilevelBridgeError("unsupported transition proof provenance")
        if self.nesting_proof_provenance not in _ALLOWED_NESTING_PROVENANCE:
            raise MultilevelBridgeError("unsupported nesting proof provenance")
        if self.trial_contract not in _TRIAL_CONTRACTS:
            raise MultilevelBridgeError("unsupported bridge trial contract")

    @property
    def failure_upper(self) -> Fraction:
        return exact_geometric_failure(self.conditional_success_lower, self.trials)

    @property
    def success_lower(self) -> Fraction:
        return 1 - self.failure_upper

    @property
    def evaluation_cost(self) -> Fraction:
        return self.evaluation_cost_per_trial * self.trials


@dataclass(frozen=True)
class MultilevelBridgeCertificate:
    levels: tuple[BridgeLevel, ...]
    total_failure_upper: Fraction
    union_failure_upper: Fraction
    all_levels_success_lower: Fraction
    total_evaluation_cost: Fraction
    target_failure_budget: Fraction
    pass_gate: bool
    semantics: str = "nested_predictable-trial_sequential-product_certificate_v19_1"

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "levels": [
                {
                    "level_id": level.level_id,
                    "conditional_success_lower": str(level.conditional_success_lower),
                    "trials": level.trials,
                    "failure_upper": str(level.failure_upper),
                    "success_lower": str(level.success_lower),
                    "evaluation_cost_per_trial": str(level.evaluation_cost_per_trial),
                    "evaluation_cost": str(level.evaluation_cost),
                    "parent_set_sha256": level.parent_set_sha256,
                    "child_set_sha256": level.child_set_sha256,
                    "transition_proof_sha256": level.transition_proof_sha256,
                    "transition_proof_provenance": level.transition_proof_provenance,
                    "nesting_proof_provenance": level.nesting_proof_provenance,
                    "trial_contract": level.trial_contract,
                }
                for level in self.levels
            ],
            "total_failure_upper": str(self.total_failure_upper),
            "union_failure_upper": str(self.union_failure_upper),
            "all_levels_success_lower": str(self.all_levels_success_lower),
            "total_evaluation_cost": str(self.total_evaluation_cost),
            "target_failure_budget": str(self.target_failure_budget),
            "pass_gate": self.pass_gate,
            "conditional_external_dependencies": sorted(
                {
                    *("bridge transition probability proof" for level in self.levels if level.transition_proof_provenance != "locally_recomputed_finite_transition"),
                    *("nested bridge-set relation proof" for level in self.levels if level.nesting_proof_provenance != "locally_recomputed_finite_set_inclusion"),
                }
            ),
        }


def _aligned_strings(
    values: Sequence[str] | None,
    length: int,
    default: str,
    name: str,
) -> tuple[str, ...]:
    if values is None:
        return (default,) * length
    out = tuple(str(x) for x in values)
    if len(out) != length:
        raise MultilevelBridgeError(f"one {name} is required per bridge level")
    return out


def build_multilevel_bridge_certificate(
    conditional_success_lower: Sequence[Fraction | int | float | str],
    trials: Sequence[int],
    *,
    target_failure_budget: Fraction | int | float | str,
    evaluation_cost_per_trial: Sequence[Fraction | int | float | str] | None = None,
    level_ids: Sequence[str] | None = None,
    set_sha256_chain: Sequence[str] | None = None,
    transition_proof_sha256: Sequence[str] | None = None,
    transition_proof_provenance: Sequence[str] | None = None,
    nesting_proof_provenance: Sequence[str] | None = None,
    trial_contracts: Sequence[str] | None = None,
) -> MultilevelBridgeCertificate:
    success = tuple(as_fraction(x) for x in conditional_success_lower)
    trial_counts = tuple(trials)
    if not success or len(success) != len(trial_counts):
        raise MultilevelBridgeError("one trial count is required per nonempty bridge level")
    if evaluation_cost_per_trial is None:
        costs = (Fraction(1, 1),) * len(success)
    else:
        costs = tuple(as_fraction(x) for x in evaluation_cost_per_trial)
    if len(costs) != len(success):
        raise MultilevelBridgeError("one cost is required per bridge level")
    if level_ids is None:
        ids = tuple(f"level_{i}" for i in range(len(success)))
    else:
        ids = tuple(str(x) for x in level_ids)
    if len(ids) != len(success) or len(set(ids)) != len(ids):
        raise MultilevelBridgeError("level IDs must be unique and dimensionally aligned")
    if set_sha256_chain is None:
        raise MultilevelBridgeError("a canonical parent/child set SHA-256 chain is required")
    set_chain = tuple(str(x) for x in set_sha256_chain)
    if len(set_chain) != len(success) + 1:
        raise MultilevelBridgeError("the set SHA-256 chain must have one more entry than bridge levels")
    if any(_HEX64.fullmatch(value) is None for value in set_chain):
        raise MultilevelBridgeError("bridge set SHA-256 values are malformed")
    if transition_proof_sha256 is None:
        raise MultilevelBridgeError("transition proof identities are required")
    proofs = tuple(str(x) for x in transition_proof_sha256)
    if len(proofs) != len(success):
        raise MultilevelBridgeError("one transition proof identity is required per level")
    transition_provenance = _aligned_strings(
        transition_proof_provenance,
        len(success),
        "theorem_parameter_conditional_transition",
        "transition proof provenance",
    )
    nesting_provenance = _aligned_strings(
        nesting_proof_provenance,
        len(success),
        "theorem_parameter_conditional_nested_sets",
        "nesting proof provenance",
    )
    contracts = _aligned_strings(
        trial_contracts,
        len(success),
        "predictable_conditional_success_lower_given_valid_parent_v2",
        "trial contract",
    )
    delta = as_fraction(target_failure_budget)
    if not (Fraction(0, 1) < delta < Fraction(1, 1)):
        raise MultilevelBridgeError("target failure budget must lie in (0,1)")
    levels = tuple(
        BridgeLevel(level_id, a, m, c, parent, child, proof, tp, np, contract)
        for level_id, a, m, c, parent, child, proof, tp, np, contract in zip(
            ids,
            success,
            trial_counts,
            costs,
            set_chain[:-1],
            set_chain[1:],
            proofs,
            transition_provenance,
            nesting_provenance,
            contracts,
            strict=True,
        )
    )
    total_failure = sequential_bridge_failure_upper(success, trial_counts)
    union_failure = union_bridge_failure_upper(success, trial_counts)
    all_success = 1 - total_failure
    total_cost = sum((level.evaluation_cost for level in levels), Fraction(0, 1))
    return MultilevelBridgeCertificate(
        levels=levels,
        total_failure_upper=total_failure,
        union_failure_upper=union_failure,
        all_levels_success_lower=all_success,
        total_evaluation_cost=total_cost,
        target_failure_budget=delta,
        pass_gate=total_failure <= delta,
    )


def equal_risk_bridge_plan(
    conditional_success_lower: Sequence[Fraction | int | float | str],
    *,
    target_failure_budget: Fraction | int | float | str,
    evaluation_cost_per_trial: Sequence[Fraction | int | float | str] | None = None,
    level_ids: Sequence[str] | None = None,
    set_sha256_chain: Sequence[str] | None = None,
    transition_proof_sha256: Sequence[str] | None = None,
    transition_proof_provenance: Sequence[str] | None = None,
    nesting_proof_provenance: Sequence[str] | None = None,
    trial_contracts: Sequence[str] | None = None,
) -> MultilevelBridgeCertificate:
    """A conservative equal-union-budget incumbent.

    Allocating ``delta/L`` to each level remains sufficient.  The returned
    certificate is evaluated with the tighter sequential-product formula.
    """

    success = tuple(as_fraction(x) for x in conditional_success_lower)
    if not success:
        raise MultilevelBridgeError("bridge must have at least one level")
    delta = as_fraction(target_failure_budget)
    per_level = delta / len(success)
    trials = tuple(minimum_trials_for_failure(a, per_level) for a in success)
    return build_multilevel_bridge_certificate(
        success,
        trials,
        target_failure_budget=delta,
        evaluation_cost_per_trial=evaluation_cost_per_trial,
        level_ids=level_ids,
        set_sha256_chain=set_sha256_chain,
        transition_proof_sha256=transition_proof_sha256,
        transition_proof_provenance=transition_proof_provenance,
        nesting_proof_provenance=nesting_proof_provenance,
        trial_contracts=trial_contracts,
    )


@dataclass(frozen=True)
class OptimalBridgePlan:
    certificate: MultilevelBridgeCertificate
    optimal: bool
    explored_nodes: int
    search_exhausted: bool

    def to_dict(self) -> dict[str, object]:
        return self.certificate.to_dict() | {
            "optimal": self.optimal,
            "explored_nodes": self.explored_nodes,
            "search_exhausted": self.search_exhausted,
            "optimized_risk": "sequential_product_failure_upper",
        }


def exact_minimum_cost_bridge_plan(
    conditional_success_lower: Sequence[Fraction | int | float | str],
    *,
    target_failure_budget: Fraction | int | float | str,
    evaluation_cost_per_trial: Sequence[Fraction | int | float | str] | None = None,
    level_ids: Sequence[str] | None = None,
    set_sha256_chain: Sequence[str],
    transition_proof_sha256: Sequence[str],
    transition_proof_provenance: Sequence[str] | None = None,
    nesting_proof_provenance: Sequence[str] | None = None,
    trial_contracts: Sequence[str] | None = None,
    max_nodes: int = 2_000_000,
) -> OptimalBridgePlan:
    """Globally minimize rational bridge cost when Dijkstra completes."""

    success = tuple(as_fraction(x) for x in conditional_success_lower)
    if not success:
        raise MultilevelBridgeError("bridge must have at least one level")
    if evaluation_cost_per_trial is None:
        costs = (Fraction(1, 1),) * len(success)
    else:
        costs = tuple(as_fraction(x) for x in evaluation_cost_per_trial)
    if len(costs) != len(success) or any(c <= 0 for c in costs):
        raise MultilevelBridgeError("one positive cost is required per bridge level")
    if max_nodes <= 0:
        raise MultilevelBridgeError("max_nodes must be positive")
    delta = as_fraction(target_failure_budget)
    incumbent_cert = equal_risk_bridge_plan(
        success,
        target_failure_budget=delta,
        evaluation_cost_per_trial=costs,
        level_ids=level_ids,
        set_sha256_chain=set_sha256_chain,
        transition_proof_sha256=transition_proof_sha256,
        transition_proof_provenance=transition_proof_provenance,
        nesting_proof_provenance=nesting_proof_provenance,
        trial_contracts=trial_contracts,
    )
    incumbent_cost = incumbent_cert.total_evaluation_cost
    zero = tuple(0 for _ in success)

    def risk(state: tuple[int, ...]) -> Fraction:
        return sequential_bridge_failure_upper(success, state)

    queue: list[tuple[Fraction, int, tuple[int, ...]]] = [(Fraction(0, 1), 0, zero)]
    seen = {zero}
    explored = 0
    while queue:
        cost, l1, state = heapq.heappop(queue)
        explored += 1
        if explored > max_nodes:
            return OptimalBridgePlan(incumbent_cert, False, explored, True)
        if cost > incumbent_cost:
            break
        if risk(state) <= delta:
            cert = build_multilevel_bridge_certificate(
                success,
                state,
                target_failure_budget=delta,
                evaluation_cost_per_trial=costs,
                level_ids=level_ids,
                set_sha256_chain=set_sha256_chain,
                transition_proof_sha256=transition_proof_sha256,
                transition_proof_provenance=transition_proof_provenance,
                nesting_proof_provenance=nesting_proof_provenance,
                trial_contracts=trial_contracts,
            )
            return OptimalBridgePlan(cert, True, explored, False)
        for i in range(len(success)):
            child = list(state)
            child[i] += 1
            child_t = tuple(child)
            if child_t in seen:
                continue
            child_cost = cost + costs[i]
            if child_cost > incumbent_cost:
                continue
            seen.add(child_t)
            heapq.heappush(queue, (child_cost, l1 + 1, child_t))
    return OptimalBridgePlan(incumbent_cert, True, explored, False)


__all__ = [
    "BridgeLevel",
    "MultilevelBridgeCertificate",
    "MultilevelBridgeError",
    "OptimalBridgePlan",
    "build_multilevel_bridge_certificate",
    "equal_risk_bridge_plan",
    "exact_minimum_cost_bridge_plan",
    "exact_geometric_failure",
    "minimum_trials_for_failure",
    "sequential_bridge_failure_upper",
    "union_bridge_failure_upper",
]
