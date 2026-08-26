"""Fixed-dimensional geometric reference oracle for Pareto-SMC v19.1.

A general many-objective Pareto front can require exponentially many
representatives.  For a *fixed* objective dimension and strictly positive
objective bounds, a geometric grid over all but one frozen pivot coordinate
reduces reference construction to a finite family of constrained scalar
subproblems.

For pivot ``k`` and threshold vector ``b`` over the remaining coordinates, the
oracle solves

    min f_k(x)  subject to f_i(x) <= b_i, i != k.

A record is accepted only when it contains a feasible witness, a positive lower
bound on the constrained optimum, and
``witness_k <= alpha * lower_bound``.  Every lower bound is tied to an external
proof hash, or is recomputed by exact enumeration.  The resulting witness set
is a multiplicative cover with factor ``alpha`` in the pivot coordinate and
``1+eta`` in every constrained coordinate.

The pivot may be selected before the run to minimize the exact grid
cardinality.  This improves constants but does not remove exponential growth in
the objective dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import re
from typing import Iterable, Sequence


class ReferenceGridOracleError(ValueError):
    pass


_ALLOWED_PROVENANCE = {
    "exact_enumeration",
    "independently_verified_constrained_oracle",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def as_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ReferenceGridOracleError("boolean is not a rational scalar")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction.from_float(value)
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ReferenceGridOracleError(f"invalid rational scalar: {value!r}") from exc


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_pivot(dimension: int, pivot_index: int | None) -> int:
    if dimension < 2:
        raise ReferenceGridOracleError("at least two objectives are required")
    pivot = dimension - 1 if pivot_index is None else int(pivot_index)
    if pivot < 0 or pivot >= dimension:
        raise ReferenceGridOracleError("pivot_index escaped the objective dimension")
    return pivot


def geometric_levels(
    lower: Fraction | int | float | str,
    upper: Fraction | int | float | str,
    eta: Fraction | int | float | str,
    *,
    max_levels: int = 100_000,
) -> tuple[Fraction, ...]:
    lo = as_fraction(lower)
    hi = as_fraction(upper)
    eta_q = as_fraction(eta)
    if not (Fraction(0, 1) < lo <= hi):
        raise ReferenceGridOracleError("geometric bounds require 0 < lower <= upper")
    if eta_q <= 0:
        raise ReferenceGridOracleError("eta must be positive")
    if max_levels <= 0:
        raise ReferenceGridOracleError("max_levels must be positive")
    ratio = 1 + eta_q
    values = [lo]
    while values[-1] < hi:
        if len(values) >= max_levels:
            raise ReferenceGridOracleError("geometric level cap exceeded")
        nxt = values[-1] * ratio
        values.append(hi if nxt >= hi else nxt)
    return tuple(values)


def geometric_grid_point_count(
    lower: Sequence[Fraction | int | float | str],
    upper: Sequence[Fraction | int | float | str],
    eta: Fraction | int | float | str,
    *,
    pivot_index: int | None = None,
    max_levels: int = 100_000,
) -> int:
    lo = tuple(as_fraction(x) for x in lower)
    hi = tuple(as_fraction(x) for x in upper)
    if len(lo) != len(hi):
        raise ReferenceGridOracleError("objective bounds are not aligned")
    pivot = _resolve_pivot(len(lo), pivot_index)
    total = 1
    for i in range(len(lo)):
        if i == pivot:
            continue
        total *= len(
            geometric_levels(lo[i], hi[i], eta, max_levels=max_levels)
        )
    return total


def select_minimum_grid_pivot(
    lower: Sequence[Fraction | int | float | str],
    upper: Sequence[Fraction | int | float | str],
    eta: Fraction | int | float | str,
) -> int:
    lo = tuple(as_fraction(x) for x in lower)
    hi = tuple(as_fraction(x) for x in upper)
    if len(lo) != len(hi):
        raise ReferenceGridOracleError("objective bounds are not aligned")
    counts = tuple(
        geometric_grid_point_count(lo, hi, eta, pivot_index=pivot)
        for pivot in range(len(lo))
    )
    return min(range(len(lo)), key=lambda pivot: (counts[pivot], pivot))


def threshold_grid(
    lower: Sequence[Fraction | int | float | str],
    upper: Sequence[Fraction | int | float | str],
    eta: Fraction | int | float | str,
    *,
    pivot_index: int | None = None,
    max_grid_points: int = 2_000_000,
) -> tuple[tuple[Fraction, ...], ...]:
    lo = tuple(as_fraction(x) for x in lower)
    hi = tuple(as_fraction(x) for x in upper)
    if len(lo) != len(hi):
        raise ReferenceGridOracleError("objective bounds are not aligned")
    pivot = _resolve_pivot(len(lo), pivot_index)
    levels = tuple(
        geometric_levels(lo[i], hi[i], eta)
        for i in range(len(lo))
        if i != pivot
    )
    total = 1
    for values in levels:
        total *= len(values)
        if total > max_grid_points:
            raise ReferenceGridOracleError("geometric threshold grid cap exceeded")
    return tuple(tuple(point) for point in itertools.product(*levels))


@dataclass(frozen=True)
class GridOracleRecord:
    thresholds: tuple[Fraction, ...]
    feasible: bool
    witness_objective: tuple[Fraction, ...] | None
    constrained_lower_bound: Fraction | None
    approximation_factor: Fraction
    proof_sha256: str
    provenance: str
    pivot_index: int = -1

    def __post_init__(self) -> None:
        if self.provenance not in _ALLOWED_PROVENANCE:
            raise ReferenceGridOracleError(
                "unsupported oracle provenance; use exact enumeration or an independently verified constrained oracle"
            )
        if self.approximation_factor < 1:
            raise ReferenceGridOracleError("approximation factor must be at least one")
        if _HEX64.fullmatch(self.proof_sha256) is None:
            raise ReferenceGridOracleError("every threshold status needs a SHA-256 proof identity")
        dimension = len(self.thresholds) + 1
        pivot = dimension - 1 if self.pivot_index == -1 else int(self.pivot_index)
        if pivot < 0 or pivot >= dimension:
            raise ReferenceGridOracleError("record pivot_index escaped the objective dimension")
        object.__setattr__(self, "pivot_index", pivot)
        if not self.feasible:
            if self.witness_objective is not None or self.constrained_lower_bound is not None:
                raise ReferenceGridOracleError(
                    "an infeasible record cannot carry a witness or optimum bound"
                )
            return
        if self.witness_objective is None or self.constrained_lower_bound is None:
            raise ReferenceGridOracleError("a feasible record needs a witness and lower bound")
        if len(self.witness_objective) != dimension:
            raise ReferenceGridOracleError("witness dimension must be threshold dimension plus one")
        if any(x <= 0 for x in self.witness_objective):
            raise ReferenceGridOracleError("multiplicative reference objectives must be positive")
        if self.constrained_lower_bound <= 0:
            raise ReferenceGridOracleError("constrained lower bound must be positive")
        constrained = tuple(i for i in range(dimension) if i != pivot)
        if any(
            self.witness_objective[i] > threshold
            for i, threshold in zip(constrained, self.thresholds, strict=True)
        ):
            raise ReferenceGridOracleError("witness violates a frozen threshold")
        if (
            self.witness_objective[pivot]
            > self.approximation_factor * self.constrained_lower_bound
        ):
            raise ReferenceGridOracleError(
                "witness does not satisfy the alpha lower-bound certificate"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "thresholds": [str(x) for x in self.thresholds],
            "feasible": self.feasible,
            "witness_objective": (
                None
                if self.witness_objective is None
                else [str(x) for x in self.witness_objective]
            ),
            "constrained_lower_bound": (
                None
                if self.constrained_lower_bound is None
                else str(self.constrained_lower_bound)
            ),
            "approximation_factor": str(self.approximation_factor),
            "proof_sha256": self.proof_sha256,
            "provenance": self.provenance,
            "pivot_index": self.pivot_index,
        }


@dataclass(frozen=True)
class GeometricReferenceCertificate:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]
    eta: Fraction
    approximation_factor: Fraction
    pivot_index: int
    constrained_indices: tuple[int, ...]
    records: tuple[GridOracleRecord, ...]
    reference_set: tuple[tuple[Fraction, ...], ...]
    grid_point_count: int
    multiplicative_factors: tuple[Fraction, ...]
    additive_error_upper: tuple[Fraction, ...]
    certificate_sha256: str
    proof_status: str
    external_proof_record_count: int
    scope: str = "fixed_dimension_positive_objective_geometric_oracle_v19_1"

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "lower": [str(x) for x in self.lower],
            "upper": [str(x) for x in self.upper],
            "eta": str(self.eta),
            "approximation_factor": str(self.approximation_factor),
            "pivot_index": self.pivot_index,
            "constrained_indices": list(self.constrained_indices),
            "grid_point_count": self.grid_point_count,
            "multiplicative_factors": [str(x) for x in self.multiplicative_factors],
            "additive_error_upper": [str(x) for x in self.additive_error_upper],
            "reference_set": [[str(x) for x in point] for point in self.reference_set],
            "records": [record.to_dict() for record in self.records],
            "certificate_sha256": self.certificate_sha256,
            "proof_status": self.proof_status,
            "external_proof_record_count": self.external_proof_record_count,
        }


def build_geometric_reference_certificate(
    *,
    lower: Sequence[Fraction | int | float | str],
    upper: Sequence[Fraction | int | float | str],
    eta: Fraction | int | float | str,
    approximation_factor: Fraction | int | float | str,
    records: Sequence[GridOracleRecord],
    pivot_index: int | None = None,
    max_grid_points: int = 2_000_000,
) -> GeometricReferenceCertificate:
    lo = tuple(as_fraction(x) for x in lower)
    hi = tuple(as_fraction(x) for x in upper)
    eta_q = as_fraction(eta)
    alpha = as_fraction(approximation_factor)
    if len(lo) < 2 or len(lo) != len(hi):
        raise ReferenceGridOracleError("invalid objective bounds")
    if any(not (Fraction(0, 1) < a <= b) for a, b in zip(lo, hi, strict=True)):
        raise ReferenceGridOracleError("all objective bounds must be positive and ordered")
    if eta_q <= 0 or alpha < 1:
        raise ReferenceGridOracleError("eta must be positive and alpha at least one")
    pivot = _resolve_pivot(len(lo), pivot_index)
    constrained = tuple(i for i in range(len(lo)) if i != pivot)
    grid = threshold_grid(
        lo,
        hi,
        eta_q,
        pivot_index=pivot,
        max_grid_points=max_grid_points,
    )
    by_threshold = {record.thresholds: record for record in records}
    if len(by_threshold) != len(records):
        raise ReferenceGridOracleError("duplicate threshold records")
    if set(by_threshold) != set(grid):
        missing = len(set(grid) - set(by_threshold))
        extra = len(set(by_threshold) - set(grid))
        raise ReferenceGridOracleError(
            f"oracle record grid mismatch: missing={missing}, extra={extra}"
        )
    ordered = tuple(by_threshold[threshold] for threshold in grid)
    if any(record.approximation_factor != alpha for record in ordered):
        raise ReferenceGridOracleError(
            "all records must use the frozen approximation factor"
        )
    if any(record.pivot_index != pivot for record in ordered):
        raise ReferenceGridOracleError(
            "all records must use the frozen pivot coordinate"
        )
    feasible_records = tuple(record for record in ordered if record.feasible)
    if not feasible_records:
        raise ReferenceGridOracleError(
            "the geometric grid has no certified feasible subproblem"
        )
    reference = tuple(
        dict.fromkeys(
            record.witness_objective
            for record in feasible_records
            if record.witness_objective is not None
        )
    )
    factors = tuple(
        alpha if i == pivot else 1 + eta_q for i in range(len(lo))
    )
    additive = tuple((factors[i] - 1) * hi[i] for i in range(len(lo)))
    payload = {
        "lower": [str(x) for x in lo],
        "upper": [str(x) for x in hi],
        "eta": str(eta_q),
        "alpha": str(alpha),
        "pivot_index": pivot,
        "records": [record.to_dict() for record in ordered],
    }
    external_count = sum(
        record.provenance == "independently_verified_constrained_oracle"
        for record in ordered
    )
    proof_status = (
        "LOCALLY_RECOMPUTED_EXACT_ENUMERATION"
        if external_count == 0
        else "CONDITIONAL_ON_EXTERNAL_CONSTRAINED_ORACLE_PROOFS"
    )
    return GeometricReferenceCertificate(
        lower=lo,
        upper=hi,
        eta=eta_q,
        approximation_factor=alpha,
        pivot_index=pivot,
        constrained_indices=constrained,
        records=ordered,
        reference_set=reference,
        grid_point_count=len(grid),
        multiplicative_factors=factors,
        additive_error_upper=additive,
        certificate_sha256=_canonical_hash(payload),
        proof_status=proof_status,
        external_proof_record_count=external_count,
    )


def exact_enumeration_grid_records(
    feasible_objectives: Iterable[Sequence[Fraction | int | float | str]],
    *,
    lower: Sequence[Fraction | int | float | str],
    upper: Sequence[Fraction | int | float | str],
    eta: Fraction | int | float | str,
    pivot_index: int | None = None,
    max_grid_points: int = 2_000_000,
) -> tuple[GridOracleRecord, ...]:
    points = tuple(
        tuple(as_fraction(x) for x in point) for point in feasible_objectives
    )
    lo = tuple(as_fraction(x) for x in lower)
    hi = tuple(as_fraction(x) for x in upper)
    if not points or any(len(point) != len(lo) for point in points):
        raise ReferenceGridOracleError(
            "enumerated objective points are empty or dimensionally invalid"
        )
    if any(
        point[i] < lo[i] or point[i] > hi[i]
        for point in points
        for i in range(len(lo))
    ):
        raise ReferenceGridOracleError(
            "enumerated point lies outside frozen positive bounds"
        )
    pivot = _resolve_pivot(len(lo), pivot_index)
    constrained = tuple(i for i in range(len(lo)) if i != pivot)
    grid = threshold_grid(
        lo,
        hi,
        eta,
        pivot_index=pivot,
        max_grid_points=max_grid_points,
    )
    source_hash = _canonical_hash(
        [[str(x) for x in point] for point in points]
    )
    out: list[GridOracleRecord] = []
    for threshold in grid:
        feasible = tuple(
            point
            for point in points
            if all(
                point[i] <= bound
                for i, bound in zip(constrained, threshold, strict=True)
            )
        )
        if not feasible:
            proof_hash = _canonical_hash(
                {
                    "source_hash": source_hash,
                    "threshold": [str(x) for x in threshold],
                    "pivot_index": pivot,
                    "status": "infeasible",
                }
            )
            out.append(
                GridOracleRecord(
                    thresholds=threshold,
                    feasible=False,
                    witness_objective=None,
                    constrained_lower_bound=None,
                    approximation_factor=Fraction(1, 1),
                    proof_sha256=proof_hash,
                    provenance="exact_enumeration",
                    pivot_index=pivot,
                )
            )
            continue
        witness = min(feasible, key=lambda point: (point[pivot], point))
        optimum = witness[pivot]
        proof_hash = _canonical_hash(
            {
                "source_hash": source_hash,
                "threshold": [str(x) for x in threshold],
                "pivot_index": pivot,
                "optimum": str(optimum),
            }
        )
        out.append(
            GridOracleRecord(
                thresholds=threshold,
                feasible=True,
                witness_objective=witness,
                constrained_lower_bound=optimum,
                approximation_factor=Fraction(1, 1),
                proof_sha256=proof_hash,
                provenance="exact_enumeration",
                pivot_index=pivot,
            )
        )
    return tuple(out)


def verify_multiplicative_cover(
    pareto_objectives: Iterable[Sequence[Fraction | int | float | str]],
    certificate: GeometricReferenceCertificate,
) -> bool:
    points = tuple(
        tuple(as_fraction(x) for x in point) for point in pareto_objectives
    )
    if any(len(point) != len(certificate.lower) for point in points):
        raise ReferenceGridOracleError("Pareto point dimension mismatch")
    for point in points:
        if any(
            point[i] < certificate.lower[i] or point[i] > certificate.upper[i]
            for i in range(len(point))
        ):
            raise ReferenceGridOracleError(
                "Pareto point lies outside the frozen positive box"
            )
        covered = any(
            all(
                witness[i] <= certificate.multiplicative_factors[i] * point[i]
                for i in range(len(point))
            )
            for witness in certificate.reference_set
        )
        if not covered:
            return False
    return True


__all__ = [
    "GeometricReferenceCertificate",
    "GridOracleRecord",
    "ReferenceGridOracleError",
    "build_geometric_reference_certificate",
    "exact_enumeration_grid_records",
    "geometric_grid_point_count",
    "geometric_levels",
    "select_minimum_grid_pivot",
    "threshold_grid",
    "verify_multiplicative_cover",
]
