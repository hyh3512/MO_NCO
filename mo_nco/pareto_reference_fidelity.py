from __future__ import annotations

"""Conditional composition relative to a supplied finite front.

No function in this module verifies that the supplied points are the complete
Pareto front.  True-front language remains unavailable until a separate raw
completeness-artifact verifier is implemented.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Sequence

from .pareto_frozen_cells import canonical_fraction_text


REFERENCE_FIDELITY_SCHEMA_V15 = "pareto_reference_fidelity_certificate_v15"

Point = tuple[Fraction, ...]


class ReferenceFidelityError(ValueError):
    """Raised when a true-front composition premise is not established."""


def _fraction(value: Fraction | int, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise ReferenceFidelityError(
            f"{label} must be an exact Fraction or integer."
        )
    return Fraction(value)


def _points(
    raw: Sequence[Sequence[Fraction | int]],
    *,
    label: str,
) -> tuple[Point, ...]:
    if not raw or not raw[0]:
        raise ReferenceFidelityError(f"{label} must be nonempty.")
    dimension = len(raw[0])
    points = []
    for point_index, point in enumerate(raw):
        if len(point) != dimension:
            raise ReferenceFidelityError(
                f"{label}[{point_index}] has the wrong dimension."
            )
        points.append(
            tuple(
                _fraction(
                    value,
                    label=f"{label}[{point_index}][{coordinate}]",
                )
                for coordinate, value in enumerate(point)
            )
        )
    return tuple(points)


def _sqrt_upper(value: Fraction, *, bits: int = 192) -> Fraction:
    if value == 0:
        return Fraction(0)
    numerator = value.numerator << (2 * bits)
    scaled = -(-numerator // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << bits)


def _distance_upper(left: Point, right: Point, p: str) -> Fraction:
    differences = tuple(abs(a - b) for a, b in zip(left, right))
    if p == "1":
        return sum(differences, Fraction(0))
    if p == "2":
        return _sqrt_upper(
            sum((value * value for value in differences), Fraction(0))
        )
    if p == "infinity":
        return max(differences)
    raise ReferenceFidelityError("p must be '1', '2', or 'infinity'.")


def _vector_norm_upper(vector: Point, p: str) -> Fraction:
    return _distance_upper(tuple(Fraction(0) for _ in vector), vector, p)


def _additively_covers(
    candidate: Point,
    target: Point,
    error: Point,
) -> bool:
    return all(
        candidate_value <= target_value + allowed
        for candidate_value, target_value, allowed in zip(
            candidate,
            target,
            error,
        )
    )


@dataclass(frozen=True)
class ReferenceFidelityCertificate:
    schema: str
    scope: str
    supplied_front_provenance_note: str
    supplied_front_sha256: str
    supplied_front_count: int
    frozen_reference_count: int
    approximation_count: int
    reference_fidelity_vector: tuple[str, ...]
    algorithm_reference_vector: tuple[str, ...]
    composed_additive_vector: tuple[str, ...]
    reference_covers_supplied_front: bool
    approximation_covers_reference: bool
    composed_cover_verified: bool
    igd_plus_supplied_front_upper: str
    supplied_to_reference_metric_radius: str
    reference_to_approximation_metric_radius: str
    ordinary_igd_supplied_front_upper: str
    shifted_front_hv_deficit_upper: str
    external_true_front_completeness_verified: bool
    true_front_coverage_claimed: bool

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def certify_reference_fidelity_composition(
    *,
    true_front: Sequence[Sequence[Fraction | int]],
    frozen_reference: Sequence[Sequence[Fraction | int]],
    approximation: Sequence[Sequence[Fraction | int]],
    reference_fidelity_vector: Sequence[Fraction | int],
    algorithm_reference_vector: Sequence[Fraction | int],
    p: str,
    hv_reference: Sequence[Fraction | int],
    supplied_front_provenance_note: str,
) -> ReferenceFidelityCertificate:
    """Compose covers relative to a supplied finite front.

    This function deliberately does not verify that ``true_front`` is the
    complete Pareto front.  The argument name is retained for the mathematical
    theorem input, while the returned executable scope is supplied-front
    relative until an external completeness verifier exists.
    """

    if not supplied_front_provenance_note.strip():
        raise ReferenceFidelityError(
            "A supplied-front composition requires a nonempty provenance note."
        )
    true_points = _points(true_front, label="true_front")
    references = _points(frozen_reference, label="frozen_reference")
    approximation_points = _points(approximation, label="approximation")
    dimension = len(true_points[0])
    if (
        len(references[0]) != dimension
        or len(approximation_points[0]) != dimension
    ):
        raise ReferenceFidelityError("Point-set dimensions differ.")
    eta = tuple(
        _fraction(value, label=f"reference_fidelity_vector[{index}]")
        for index, value in enumerate(reference_fidelity_vector)
    )
    epsilon = tuple(
        _fraction(value, label=f"algorithm_reference_vector[{index}]")
        for index, value in enumerate(algorithm_reference_vector)
    )
    if (
        len(eta) != dimension
        or len(epsilon) != dimension
        or any(value < 0 for value in (*eta, *epsilon))
    ):
        raise ReferenceFidelityError(
            "Additive vectors have the wrong dimension or a negative entry."
        )
    composed = tuple(a + b for a, b in zip(eta, epsilon))
    reference_cover = all(
        any(_additively_covers(reference, point, eta) for reference in references)
        for point in true_points
    )
    approximation_cover = all(
        any(
            _additively_covers(candidate, reference, epsilon)
            for candidate in approximation_points
        )
        for reference in references
    )
    if not reference_cover:
        raise ReferenceFidelityError(
            "Frozen reference does not satisfy the declared supplied-front cover."
        )
    if not approximation_cover:
        raise ReferenceFidelityError(
            "Approximation does not satisfy the declared reference cover."
        )
    composed_cover = all(
        any(
            _additively_covers(candidate, point, composed)
            for candidate in approximation_points
        )
        for point in true_points
    )
    if not composed_cover:
        raise AssertionError("Verified additive covers failed to compose.")
    true_to_reference = max(
        min(_distance_upper(point, reference, p) for reference in references)
        for point in true_points
    )
    reference_to_approximation = max(
        min(
            _distance_upper(reference, candidate, p)
            for candidate in approximation_points
        )
        for reference in references
    )
    ordinary_upper = true_to_reference + reference_to_approximation
    hv_ref = tuple(
        _fraction(value, label=f"hv_reference[{index}]")
        for index, value in enumerate(hv_reference)
    )
    if len(hv_ref) != dimension:
        raise ReferenceFidelityError("hv_reference has the wrong dimension.")
    side_lengths = []
    for coordinate in range(dimension):
        minimum = min(point[coordinate] for point in true_points)
        if hv_ref[coordinate] <= minimum:
            raise ReferenceFidelityError(
                "hv_reference must be strictly worse than the true front."
            )
        side_lengths.append(hv_ref[coordinate] - minimum)
    hv_upper = sum(
        (
            composed[coordinate]
            * math.prod(
                side_lengths[other]
                for other in range(dimension)
                if other != coordinate
            )
            for coordinate in range(dimension)
        ),
        Fraction(0),
    )
    supplied_payload = [
        [canonical_fraction_text(value) for value in point]
        for point in true_points
    ]
    supplied_sha256 = hashlib.sha256(
        json.dumps(
            supplied_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return ReferenceFidelityCertificate(
        schema=REFERENCE_FIDELITY_SCHEMA_V15,
        scope="supplied_front_relative_conditional_composition",
        supplied_front_provenance_note=supplied_front_provenance_note,
        supplied_front_sha256=supplied_sha256,
        supplied_front_count=len(true_points),
        frozen_reference_count=len(references),
        approximation_count=len(approximation_points),
        reference_fidelity_vector=tuple(
            canonical_fraction_text(value) for value in eta
        ),
        algorithm_reference_vector=tuple(
            canonical_fraction_text(value) for value in epsilon
        ),
        composed_additive_vector=tuple(
            canonical_fraction_text(value) for value in composed
        ),
        reference_covers_supplied_front=True,
        approximation_covers_reference=True,
        composed_cover_verified=True,
        igd_plus_supplied_front_upper=canonical_fraction_text(
            _vector_norm_upper(composed, p)
        ),
        supplied_to_reference_metric_radius=canonical_fraction_text(
            true_to_reference
        ),
        reference_to_approximation_metric_radius=canonical_fraction_text(
            reference_to_approximation
        ),
        ordinary_igd_supplied_front_upper=canonical_fraction_text(
            ordinary_upper
        ),
        shifted_front_hv_deficit_upper=canonical_fraction_text(hv_upper),
        external_true_front_completeness_verified=False,
        true_front_coverage_claimed=False,
    )


__all__ = [
    "REFERENCE_FIDELITY_SCHEMA_V15",
    "ReferenceFidelityCertificate",
    "ReferenceFidelityError",
    "certify_reference_fidelity_composition",
]
