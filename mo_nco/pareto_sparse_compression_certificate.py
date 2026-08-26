from __future__ import annotations

"""Conditional v13 finite-reference sparse-compression certificate.

This module deliberately certifies only a *frozen finite* reference set.  It
does not claim that a fixed-size archive can approximate an arbitrary Pareto
front independently of the reference-set cardinality.

Let ``R`` be the full frozen minimization reference set and let ``A`` be its
strictly validated canonical greedy cover.  Suppose that, for every anchor
``a`` in ``A``, a retained feasible terminal witness ``w(a)`` satisfies

    |w(a)_j - a_j| <= h_j

for every objective coordinate ``j``.  Being in the same externally frozen
Cartesian cell of width ``h_j`` is sufficient for this premise.  If every
reference point assigned to ``a`` differs from it by at most ``rho_j``, then

    |w(a)_j - r_j| <= h_j + rho_j.

Consequently, the one-witness-per-anchor support has an ordinary IGD_p bound,
and its nondominated view has the corresponding IGD+_p and shifted-front
fixed-reference hypervolume-deficit bounds.  Nondominated filtering is not
used for the ordinary IGD claim.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Sequence, Tuple

from .pareto_sparse_reference import (
    SparseReferenceCover,
    SparseReferenceError,
    sparse_reference_metric_bounds,
)


SPARSE_COMPRESSION_CERTIFICATE_SCHEMA_V13 = (
    "pareto_smc_sparse_finite_reference_compression_certificate_v13"
)


class SparseCompressionCertificateError(ValueError):
    """Raised when a v13 sparse-compression certificate cannot be proved."""


Point = Tuple[float, ...]


@dataclass(frozen=True)
class SparseFiniteReferenceCompressionCertificate:
    """Auditable conditional metric certificate for one frozen reference set."""

    schema: str
    objective_dimension: int
    full_reference_count: int
    source_reference_sha256: str
    canonical_cover_sha256: str
    canonical_cover_p_norm: float
    metric_p_norm: float
    anchor_indices: Tuple[int, ...]
    anchor_reference_set: Tuple[Point, ...]
    cell_width_vector: Point
    reference_to_anchor_coordinate_radii: Point
    coordinatewise_inflation_vector: Point
    ordinary_igd_bound: float
    igd_plus_bound: float
    hv_reference: Point
    shifted_front_hv_deficit_bound: float
    archive_cardinality_bound: int
    retained_witness_support_cardinality_bound: int
    nondominated_archive_cardinality_bound: int
    conditional_anchor_witness_premise: str
    ordinary_igd_scope: str
    igd_plus_and_hv_scope: str
    numerical_contract: str
    reference_count_independent_universal_compression_claimed: bool

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("canonical_cover_p_norm", "metric_p_norm"):
            value = float(payload[key])
            if math.isinf(value) and value > 0.0:
                payload[key] = "infinity"
        return payload


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_token(value: float) -> str:
    """Bind the exact binary64 value rather than a rounded decimal rendering."""

    return value.hex()


def _finite_points(
    points: Sequence[Sequence[float]],
) -> Tuple[Point, ...]:
    if isinstance(points, (str, bytes)):
        raise SparseCompressionCertificateError(
            "reference_points must be a nonempty numeric array."
        )
    resolved: list[Point] = []
    try:
        for point_index, point in enumerate(points):
            if isinstance(point, (str, bytes)):
                raise SparseCompressionCertificateError(
                    f"reference_points[{point_index}] is not a coordinate array."
                )
            converted: list[float] = []
            for coordinate, raw_value in enumerate(point):
                if isinstance(raw_value, bool):
                    raise SparseCompressionCertificateError(
                        "Boolean reference coordinates are forbidden."
                    )
                value = float(raw_value)
                if not math.isfinite(value):
                    raise SparseCompressionCertificateError(
                        f"reference_points[{point_index}][{coordinate}] "
                        "must be finite."
                    )
                converted.append(value)
            resolved.append(tuple(converted))
    except SparseCompressionCertificateError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise SparseCompressionCertificateError(
            "reference_points must be a nonempty finite numeric array."
        ) from error
    if not resolved:
        raise SparseCompressionCertificateError(
            "reference_points must be nonempty."
        )
    dimension = len(resolved[0])
    if dimension == 0 or any(len(point) != dimension for point in resolved):
        raise SparseCompressionCertificateError(
            "Reference points must have one common positive dimension."
        )
    return tuple(resolved)


def _finite_vector(
    values: Sequence[float],
    *,
    dimension: int,
    label: str,
    nonnegative: bool,
) -> Point:
    if isinstance(values, (str, bytes)):
        raise SparseCompressionCertificateError(
            f"{label} must be a coordinate vector."
        )
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise SparseCompressionCertificateError(
            f"{label} must be a coordinate vector."
        ) from error
    if len(raw_values) != dimension:
        raise SparseCompressionCertificateError(
            f"{label} must contain exactly {dimension} coordinates."
        )
    converted: list[float] = []
    for coordinate, raw_value in enumerate(raw_values):
        if isinstance(raw_value, bool):
            raise SparseCompressionCertificateError(
                f"{label}[{coordinate}] cannot be Boolean."
            )
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise SparseCompressionCertificateError(
                f"{label}[{coordinate}] must be numeric."
            ) from error
        if not math.isfinite(value) or (nonnegative and value < 0.0):
            qualifier = "finite and nonnegative" if nonnegative else "finite"
            raise SparseCompressionCertificateError(
                f"{label}[{coordinate}] must be {qualifier}."
            )
        converted.append(value)
    return tuple(converted)


def _metric_p_norm(value: object) -> float:
    if isinstance(value, bool):
        raise SparseCompressionCertificateError(
            "igd_p must be exactly 1, 2, or positive infinity."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SparseCompressionCertificateError(
            "igd_p must be exactly 1, 2, or positive infinity."
        ) from error
    if result not in (1.0, 2.0) and not (
        math.isinf(result) and result > 0.0
    ):
        raise SparseCompressionCertificateError(
            "igd_p must be exactly 1, 2, or positive infinity."
        )
    return result


def _strict_cover_guard(cover: SparseReferenceCover) -> None:
    if not isinstance(cover, SparseReferenceCover):
        raise SparseCompressionCertificateError(
            "cover must be a SparseReferenceCover."
        )
    scalar_fields = (
        ("p_norm", cover.p_norm),
        ("requested_cover_radius", cover.requested_cover_radius),
        ("realized_cover_radius", cover.realized_cover_radius),
    )
    if cover.pairwise_anchor_separation_lower_bound is not None:
        scalar_fields += (
            (
                "pairwise_anchor_separation_lower_bound",
                cover.pairwise_anchor_separation_lower_bound,
            ),
        )
    for label, value in scalar_fields:
        if isinstance(value, bool):
            raise SparseCompressionCertificateError(
                f"cover.{label} cannot be Boolean."
            )
    for point_index, point in enumerate(cover.anchor_objectives):
        for coordinate, value in enumerate(point):
            if isinstance(value, bool):
                raise SparseCompressionCertificateError(
                    "Boolean cover anchor coordinates are forbidden: "
                    f"anchor_objectives[{point_index}][{coordinate}]."
                )


def _source_hash(points: Tuple[Point, ...]) -> str:
    return _canonical_sha256(
        {
            "schema": "pareto_sparse_finite_reference_source_v1",
            "ordered_reference_objective_float_hex": [
                [_float_token(value) for value in point]
                for point in points
            ],
        }
    )


def _cover_hash(
    cover: SparseReferenceCover,
    *,
    source_reference_sha256: str,
) -> str:
    separation = cover.pairwise_anchor_separation_lower_bound
    return _canonical_sha256(
        {
            "schema": "pareto_sparse_canonical_greedy_cover_v1",
            "source_reference_sha256": source_reference_sha256,
            "objective_dimension": cover.objective_dimension,
            "p_norm": (
                "infinity"
                if math.isinf(float(cover.p_norm))
                else _float_token(float(cover.p_norm))
            ),
            "requested_cover_radius_float_hex": _float_token(
                float(cover.requested_cover_radius)
            ),
            "anchor_indices": list(cover.anchor_indices),
            "assignment_by_reference": list(
                cover.assignment_by_reference
            ),
            "anchor_objective_float_hex": [
                [_float_token(float(value)) for value in point]
                for point in cover.anchor_objectives
            ],
            "realized_cover_radius_float_hex": _float_token(
                float(cover.realized_cover_radius)
            ),
            "pairwise_anchor_separation_lower_bound_float_hex": (
                None
                if separation is None
                else _float_token(float(separation))
            ),
            "cluster_sizes": list(cover.cluster_sizes),
        }
    )


def _fraction_to_float_upper(
    value: Fraction,
    *,
    label: str,
) -> float:
    if value < 0:
        raise SparseCompressionCertificateError(
            f"{label} must be nonnegative."
        )
    try:
        candidate = float(value)
    except OverflowError as error:
        raise SparseCompressionCertificateError(
            f"{label} exceeds the finite binary64 certificate domain."
        ) from error
    if not math.isfinite(candidate):
        raise SparseCompressionCertificateError(
            f"{label} exceeds the finite binary64 certificate domain."
        )
    if Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    if not math.isfinite(candidate) or Fraction.from_float(candidate) < value:
        raise SparseCompressionCertificateError(
            f"{label} has no finite conservative binary64 upper bound."
        )
    return candidate


def _sqrt_fraction_to_float_upper(
    value: Fraction,
    *,
    coordinates: Sequence[float],
) -> float:
    if value < 0:
        raise SparseCompressionCertificateError(
            "The squared IGD bound cannot be negative."
        )
    candidate = math.hypot(*coordinates)
    if not math.isfinite(candidate):
        raise SparseCompressionCertificateError(
            "The IGD bound exceeds the finite binary64 certificate domain."
        )
    # ``math.hypot`` is an approximation.  Certify the direction by exact
    # rational comparison and move upward until candidate^2 covers the input.
    while Fraction.from_float(candidate) ** 2 < value:
        advanced = math.nextafter(candidate, math.inf)
        if advanced == candidate or not math.isfinite(advanced):
            raise SparseCompressionCertificateError(
                "The IGD bound has no finite conservative binary64 upper bound."
            )
        candidate = advanced
    return candidate


def _nondominated_fraction_points(
    points: Sequence[Tuple[Fraction, ...]],
) -> Tuple[Tuple[Fraction, ...], ...]:
    unique = tuple(sorted(set(points)))
    return tuple(
        point
        for index, point in enumerate(unique)
        if not any(
            other_index != index
            and all(left <= right for left, right in zip(other, point))
            and any(left < right for left, right in zip(other, point))
            for other_index, other in enumerate(unique)
        )
    )


def _hypervolume_fraction(
    points: Sequence[Tuple[Fraction, ...]],
    reference: Tuple[Fraction, ...],
) -> Fraction:
    relevant = _nondominated_fraction_points(
        tuple(
            point
            for point in points
            if all(value < limit for value, limit in zip(point, reference))
        )
    )

    def recurse(
        active: Tuple[Tuple[Fraction, ...], ...],
        local_reference: Tuple[Fraction, ...],
    ) -> Fraction:
        if not active:
            return Fraction(0)
        if len(local_reference) == 1:
            return max(
                Fraction(0),
                local_reference[0] - min(point[0] for point in active),
            )
        levels = sorted(
            {
                point[0]
                for point in active
                if point[0] < local_reference[0]
            }
        )
        levels.append(local_reference[0])
        volume = Fraction(0)
        for left, right in zip(levels, levels[1:]):
            if right <= left:
                continue
            projection = tuple(
                point[1:] for point in active if point[0] <= left
            )
            volume += (right - left) * recurse(
                _nondominated_fraction_points(projection),
                local_reference[1:],
            )
        return volume

    return recurse(relevant, reference)


def _shifted_hv_deficit_upper(
    points: Tuple[Point, ...],
    *,
    additive_inflation: Point,
    hv_reference: Point,
) -> float:
    exact_points = tuple(
        tuple(Fraction.from_float(value) for value in point)
        for point in points
    )
    exact_reference = tuple(
        Fraction.from_float(value) for value in hv_reference
    )
    exact_inflation = tuple(
        Fraction.from_float(value) for value in additive_inflation
    )
    shifted = tuple(
        tuple(
            min(limit, value + width)
            for value, width, limit in zip(
                point,
                exact_inflation,
                exact_reference,
            )
        )
        for point in exact_points
    )
    deficit = max(
        Fraction(0),
        _hypervolume_fraction(exact_points, exact_reference)
        - _hypervolume_fraction(shifted, exact_reference),
    )
    return _fraction_to_float_upper(
        deficit,
        label="shifted-front hypervolume-deficit bound",
    )


def certify_sparse_finite_reference_compression(
    reference_points: Sequence[Sequence[float]],
    cover: SparseReferenceCover,
    *,
    cell_width_vector: Sequence[float],
    igd_p: float,
    hv_reference: Sequence[float],
) -> SparseFiniteReferenceCompressionCertificate:
    """Build the conditional v13 finite-reference compression certificate.

    The canonical cover is recomputed by
    :func:`pareto_sparse_reference.sparse_reference_metric_bounds`; any
    changed anchor, assignment, cluster size, radius, separation, tie choice,
    or source reference value therefore fails closed.

    Only ``p`` in ``{1, 2, infinity}`` is supported.  Coordinate radii, vector
    additions, norm inequalities, and hypervolume volumes are checked with
    exact rational arithmetic over the supplied binary64 values.  Every
    reported scalar bound is rounded upward to a finite binary64 value.
    """

    points = _finite_points(reference_points)
    dimension = len(points[0])
    widths = _finite_vector(
        cell_width_vector,
        dimension=dimension,
        label="cell_width_vector",
        nonnegative=True,
    )
    p_value = _metric_p_norm(igd_p)
    hv_ref = _finite_vector(
        hv_reference,
        dimension=dimension,
        label="hv_reference",
        nonnegative=False,
    )
    if any(
        limit < max(point[coordinate] for point in points)
        for coordinate, limit in enumerate(hv_ref)
    ):
        raise SparseCompressionCertificateError(
            "hv_reference must be coordinatewise no better than every "
            "frozen reference point."
        )

    _strict_cover_guard(cover)
    try:
        # This performs the strict canonical-greedy integrity check.  The
        # infinity metric requested here affects only the legacy returned
        # bound; the cover itself is validated under its bound p-norm.
        sparse_reference_metric_bounds(
            points,
            cover,
            cell_width_vector=widths,
            p_norm=math.inf,
        )
    except (SparseReferenceError, TypeError, ValueError, OverflowError) as error:
        raise SparseCompressionCertificateError(
            "The supplied cover failed strict canonical integrity validation."
        ) from error

    exact_coordinate_radii = [Fraction(0) for _ in range(dimension)]
    for point, anchor_position in zip(
        points,
        cover.assignment_by_reference,
    ):
        anchor = cover.anchor_objectives[anchor_position]
        for coordinate, (value, anchor_value) in enumerate(
            zip(point, anchor)
        ):
            difference = abs(
                Fraction.from_float(value)
                - Fraction.from_float(float(anchor_value))
            )
            exact_coordinate_radii[coordinate] = max(
                exact_coordinate_radii[coordinate],
                difference,
            )

    radius_upper = tuple(
        _fraction_to_float_upper(
            value,
            label=f"coordinate cover radius {coordinate}",
        )
        for coordinate, value in enumerate(exact_coordinate_radii)
    )
    # Build the published inflation from the already published conservative
    # coordinate radii.  This makes the JSON-level relation auditable without
    # relying on hidden higher-precision radii.
    exact_total_inflation = tuple(
        Fraction.from_float(width) + Fraction.from_float(radius)
        for width, radius in zip(widths, radius_upper)
    )
    inflation_upper = tuple(
        _fraction_to_float_upper(
            value,
            label=f"coordinate inflation {coordinate}",
        )
        for coordinate, value in enumerate(exact_total_inflation)
    )
    reported_inflation_exact = tuple(
        Fraction.from_float(value) for value in inflation_upper
    )

    if p_value == 1.0:
        igd_bound = _fraction_to_float_upper(
            sum(reported_inflation_exact, Fraction(0)),
            label="IGD_1 bound",
        )
    elif p_value == 2.0:
        squared = sum(
            (value * value for value in reported_inflation_exact),
            Fraction(0),
        )
        igd_bound = _sqrt_fraction_to_float_upper(
            squared,
            coordinates=inflation_upper,
        )
    else:
        igd_bound = _fraction_to_float_upper(
            max(reported_inflation_exact),
            label="IGD_infinity bound",
        )

    shifted_hv_bound = _shifted_hv_deficit_upper(
        points,
        additive_inflation=inflation_upper,
        hv_reference=hv_ref,
    )
    archive_cap = len(cover.anchor_indices)
    source_reference_sha256 = _source_hash(points)
    return SparseFiniteReferenceCompressionCertificate(
        schema=SPARSE_COMPRESSION_CERTIFICATE_SCHEMA_V13,
        objective_dimension=dimension,
        full_reference_count=len(points),
        source_reference_sha256=source_reference_sha256,
        canonical_cover_sha256=_cover_hash(
            cover,
            source_reference_sha256=source_reference_sha256,
        ),
        canonical_cover_p_norm=float(cover.p_norm),
        metric_p_norm=p_value,
        anchor_indices=tuple(cover.anchor_indices),
        anchor_reference_set=tuple(
            tuple(float(value) for value in point)
            for point in cover.anchor_objectives
        ),
        cell_width_vector=widths,
        reference_to_anchor_coordinate_radii=radius_upper,
        coordinatewise_inflation_vector=inflation_upper,
        ordinary_igd_bound=igd_bound,
        igd_plus_bound=igd_bound,
        hv_reference=hv_ref,
        shifted_front_hv_deficit_bound=shifted_hv_bound,
        archive_cardinality_bound=archive_cap,
        retained_witness_support_cardinality_bound=archive_cap,
        nondominated_archive_cardinality_bound=archive_cap,
        conditional_anchor_witness_premise=(
            "For every canonical anchor a, retain one feasible terminal "
            "witness w with |w_j-a_j| <= cell_width_vector[j] in every "
            "coordinate; membership in the same externally frozen Cartesian "
            "cell is sufficient."
        ),
        ordinary_igd_scope=(
            "the retained one-feasible-witness-per-anchor support before "
            "nondominated filtering"
        ),
        igd_plus_and_hv_scope=(
            "the nondominated view of the retained witness support"
        ),
        numerical_contract=(
            "inputs are finite binary64 values; p is 1, 2, or infinity; "
            "coordinate, norm, and hypervolume inequalities are certified "
            "with exact rational comparisons and finite upward-rounded "
            "binary64 outputs"
        ),
        reference_count_independent_universal_compression_claimed=False,
    )


__all__ = [
    "SPARSE_COMPRESSION_CERTIFICATE_SCHEMA_V13",
    "SparseCompressionCertificateError",
    "SparseFiniteReferenceCompressionCertificate",
    "certify_sparse_finite_reference_compression",
]
