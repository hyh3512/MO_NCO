"""Executable fixed-origin TSP reference-cover branch-and-bound certificate.

The certificate proves that a frozen feasible reference set additively covers
*every* fixed-origin tour objective.  A node is a tour prefix.  For each
objective, the lower bound is the exact prefix cost plus the minimum outgoing
edge of every vertex whose outgoing edge is not fixed yet.  This ignores
in-degree and subtour constraints and is therefore weak but valid.

A node may be pruned only when one frozen reference objective ``q`` satisfies
``q <= lower_bound(node) + eta`` coordinatewise.  Otherwise it is split into
all unused next cities.  Complete leaves are evaluated exactly.  Exhausting the
node cap fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Sequence

from .pareto_v17_regeneration import as_fraction
from .pareto_v18_reference_completeness import (
    Point,
    ReferenceCompletenessError,
    Tour,
    _validate_matrices,
    evaluate_tour_exact,
)


@dataclass(frozen=True)
class ReferenceWitness:
    tour: Tour
    objective: Point


@dataclass(frozen=True)
class BranchAndBoundReferenceCertificate:
    city_count: int
    objective_count: int
    additive_eta: Point
    frozen_reference: tuple[Point, ...]
    visited_nodes: int
    pruned_nodes: int
    complete_leaves: int
    max_stack_size: int
    reference_witness_sha256: str
    proof_trace_sha256: str
    additive_cover_verified: bool
    scope: str = "fixed_origin_tsp_min_outgoing_branch_and_bound_v18"

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "city_count": self.city_count,
            "objective_count": self.objective_count,
            "additive_eta": [str(x) for x in self.additive_eta],
            "frozen_reference": [[str(x) for x in p] for p in self.frozen_reference],
            "visited_nodes": self.visited_nodes,
            "pruned_nodes": self.pruned_nodes,
            "complete_leaves": self.complete_leaves,
            "max_stack_size": self.max_stack_size,
            "reference_witness_sha256": self.reference_witness_sha256,
            "proof_trace_sha256": self.proof_trace_sha256,
            "additive_cover_verified": self.additive_cover_verified,
        }


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _parse_witnesses(
    matrices,
    raw_witnesses: Sequence[dict[str, object]],
) -> tuple[ReferenceWitness, ...]:
    out: list[ReferenceWitness] = []
    n = len(matrices[0])
    for item in raw_witnesses:
        tour = tuple(int(x) for x in item["tour"])  # type: ignore[index]
        if len(tour) != n:
            raise ReferenceCompletenessError("reference witness tour has the wrong length")
        objective = tuple(as_fraction(x) for x in item["objective"])  # type: ignore[index]
        actual = evaluate_tour_exact(matrices, tour)
        if actual != objective:
            raise ReferenceCompletenessError("reference witness objective does not replay exactly")
        out.append(ReferenceWitness(tour=tour, objective=objective))
    if not out:
        raise ReferenceCompletenessError("at least one reference witness is required")
    if len({w.objective for w in out}) != len(out):
        raise ReferenceCompletenessError("reference objectives must be unique")
    return tuple(out)


def prefix_lower_bound(matrices, prefix: Tour) -> Point:
    n = len(matrices[0])
    if not prefix or prefix[0] != 0 or len(set(prefix)) != len(prefix):
        raise ReferenceCompletenessError("invalid fixed-origin prefix")
    unused = set(range(n)) - set(prefix)
    result: list[Fraction] = []
    for matrix in matrices:
        fixed = sum(
            (matrix[prefix[i]][prefix[i + 1]] for i in range(len(prefix) - 1)),
            Fraction(0, 1),
        )
        unfixed_sources = {prefix[-1], *unused}
        outgoing = Fraction(0, 1)
        for source in unfixed_sources:
            outgoing += min(matrix[source][dest] for dest in range(n) if dest != source)
        result.append(fixed + outgoing)
    return tuple(result)


def _covered_by_reference(lower: Point, reference: Sequence[Point], eta: Point) -> int | None:
    for index, q in enumerate(reference):
        if all(q_i <= lb_i + e_i for q_i, lb_i, e_i in zip(q, lower, eta, strict=True)):
            return index
    return None


def certify_reference_cover_branch_and_bound(
    objective_matrices,
    reference_witnesses: Sequence[dict[str, object]],
    additive_eta,
    *,
    max_nodes: int = 2_000_000,
) -> BranchAndBoundReferenceCertificate:
    matrices = _validate_matrices(objective_matrices)
    n = len(matrices[0])
    witnesses = _parse_witnesses(matrices, reference_witnesses)
    reference = tuple(w.objective for w in witnesses)
    eta = tuple(as_fraction(x) for x in additive_eta)
    if len(eta) != len(matrices) or any(x < 0 for x in eta):
        raise ReferenceCompletenessError("additive eta has the wrong dimension or a negative entry")
    if not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ReferenceCompletenessError("max_nodes must be positive")

    stack: list[Tour] = [(0,)]
    trace: list[dict[str, object]] = []
    visited = pruned = complete = 0
    max_stack = 1
    while stack:
        max_stack = max(max_stack, len(stack))
        prefix = stack.pop()
        visited += 1
        if visited > max_nodes:
            raise ReferenceCompletenessError("branch-and-bound node cap exhausted")
        lower = prefix_lower_bound(matrices, prefix)
        ref_index = _covered_by_reference(lower, reference, eta)
        if ref_index is not None:
            pruned += 1
            trace.append({
                "prefix": list(prefix),
                "status": "covered_lower_bound_prune",
                "lower": [str(x) for x in lower],
                "reference_index": ref_index,
            })
            continue
        if len(prefix) == n:
            complete += 1
            objective = evaluate_tour_exact(matrices, prefix)
            ref_index = _covered_by_reference(objective, reference, eta)
            if ref_index is None:
                raise ReferenceCompletenessError(
                    "complete tour is not covered by the frozen reference"
                )
            trace.append({
                "prefix": list(prefix),
                "status": "covered_complete_leaf",
                "objective": [str(x) for x in objective],
                "reference_index": ref_index,
            })
            continue
        unused = sorted(set(range(1, n)) - set(prefix))
        for city in reversed(unused):
            stack.append((*prefix, city))

    witness_payload = [
        {"tour": list(w.tour), "objective": [str(x) for x in w.objective]}
        for w in witnesses
    ]
    return BranchAndBoundReferenceCertificate(
        city_count=n,
        objective_count=len(matrices),
        additive_eta=eta,
        frozen_reference=reference,
        visited_nodes=visited,
        pruned_nodes=pruned,
        complete_leaves=complete,
        max_stack_size=max_stack,
        reference_witness_sha256=_canonical_hash(witness_payload),
        proof_trace_sha256=_canonical_hash(trace),
        additive_cover_verified=True,
    )


__all__ = [
    "BranchAndBoundReferenceCertificate",
    "ReferenceWitness",
    "certify_reference_cover_branch_and_bound",
    "prefix_lower_bound",
]
