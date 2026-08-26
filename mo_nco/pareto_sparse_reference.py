from __future__ import annotations

"""Strict sparse frozen-reference covers and many-objective lower bounds."""

import math
from dataclasses import asdict, dataclass
from typing import Sequence, Tuple


class SparseReferenceError(ValueError):
    """Raised when a frozen sparse-reference contract is malformed."""


@dataclass(frozen=True)
class SparseReferenceCover:
    objective_dimension: int
    p_norm: float
    requested_cover_radius: float
    anchor_indices: Tuple[int, ...]
    assignment_by_reference: Tuple[int, ...]
    anchor_objectives: Tuple[Tuple[float, ...], ...]
    realized_cover_radius: float
    pairwise_anchor_separation_lower_bound: float | None
    cluster_sizes: Tuple[int, ...]

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        if math.isinf(self.p_norm) and self.p_norm > 0.0:
            payload["p_norm"] = "infinity"
        return payload


@dataclass(frozen=True)
class SparseMetricBounds:
    cell_width_vector: Tuple[float, ...]
    reference_to_anchor_coordinate_radii: Tuple[float, ...]
    additive_error_vector: Tuple[float, ...]
    ordinary_igd_bound: float
    igd_plus_bound: float
    archive_cardinality_bound: int


@dataclass(frozen=True)
class SpernerLowerBound:
    objective_dimension: int
    layer_weight: int
    pareto_point_count: int
    additive_linf_tolerance: float
    minimum_required_feasible_representatives: int


@dataclass(frozen=True)
class ManyObjectiveCapacityLowerBound:
    """Conditional direction/cell capacity consequence of the Sperner set."""

    objective_dimension: int
    pareto_point_count: int
    minimum_required_feasible_representatives: int
    minimum_required_subunit_linf_cells: int
    maximum_certified_representatives_per_type: int
    minimum_required_types: int

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def _finite_points(points: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    try:
        resolved = tuple(tuple(float(value) for value in point) for point in points)
    except (TypeError, ValueError) as error:
        raise SparseReferenceError("Reference points must be a finite numeric array.") from error
    if not resolved:
        raise SparseReferenceError("Reference points must be nonempty.")
    dimension = len(resolved[0])
    if dimension == 0 or any(len(point) != dimension for point in resolved):
        raise SparseReferenceError("Reference points must have one common positive dimension.")
    if any(not math.isfinite(value) for point in resolved for value in point):
        raise SparseReferenceError("Reference points must contain finite values only.")
    return resolved


def _norm(differences: Sequence[float], p: float) -> float:
    if math.isinf(p) and p > 0.0:
        return max(abs(value) for value in differences)
    if not math.isfinite(p) or p < 1.0:
        raise SparseReferenceError("p_norm must lie in [1, infinity].")
    return sum(abs(value) ** p for value in differences) ** (1.0 / p)


def _validated_p_norm(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SparseReferenceError(
            "p_norm must lie in [1, positive infinity]."
        ) from error
    if not (
        (math.isinf(result) and result > 0.0)
        or (math.isfinite(result) and result >= 1.0)
    ):
        raise SparseReferenceError(
            "p_norm must lie in [1, positive infinity]."
        )
    return result


def _distance(left: Sequence[float], right: Sequence[float], p: float) -> float:
    return _norm(tuple(a - b for a, b in zip(left, right)), p)


def greedy_maximal_reference_net(
    reference_points: Sequence[Sequence[float]],
    *,
    cover_radius: float,
    p_norm: float = math.inf,
) -> SparseReferenceCover:
    """Build a deterministic maximal separated set in canonical order.

    Points are processed in lexicographic objective order, with original index
    as a final tie breaker.  A point becomes an anchor iff its distance to every
    existing anchor is strictly larger than ``cover_radius``.  The result is
    both a cover of the finite reference set and a separated anchor set.
    """

    points = _finite_points(reference_points)
    try:
        radius = float(cover_radius)
    except (TypeError, ValueError) as error:
        raise SparseReferenceError("cover_radius must be numeric.") from error
    if not math.isfinite(radius) or radius < 0.0:
        raise SparseReferenceError("cover_radius must be finite and nonnegative.")
    resolved_p = _validated_p_norm(p_norm)

    order = sorted(range(len(points)), key=lambda index: (points[index], index))
    anchors: list[int] = []
    for index in order:
        if not anchors or all(
            _distance(points[index], points[anchor], resolved_p) > radius
            for anchor in anchors
        ):
            anchors.append(index)

    assignments = []
    realized_radius = 0.0
    cluster_sizes = [0 for _ in anchors]
    for point in points:
        anchor_position = min(
            range(len(anchors)),
            key=lambda position: (
                _distance(point, points[anchors[position]], resolved_p),
                position,
            ),
        )
        distance = _distance(
            point,
            points[anchors[anchor_position]],
            resolved_p,
        )
        realized_radius = max(realized_radius, distance)
        assignments.append(anchor_position)
        cluster_sizes[anchor_position] += 1

    if len(anchors) <= 1:
        separation = None
    else:
        separation = min(
            _distance(points[left], points[right], resolved_p)
            for i, left in enumerate(anchors)
            for right in anchors[i + 1 :]
        )

    return SparseReferenceCover(
        objective_dimension=len(points[0]),
        p_norm=resolved_p,
        requested_cover_radius=radius,
        anchor_indices=tuple(anchors),
        assignment_by_reference=tuple(assignments),
        anchor_objectives=tuple(points[index] for index in anchors),
        realized_cover_radius=realized_radius,
        pairwise_anchor_separation_lower_bound=separation,
        cluster_sizes=tuple(cluster_sizes),
    )


def _validate_cover_integrity(
    points: Tuple[Tuple[float, ...], ...],
    cover: SparseReferenceCover,
) -> None:
    if (
        isinstance(cover.objective_dimension, bool)
        or not isinstance(cover.objective_dimension, int)
        or cover.objective_dimension != len(points[0])
    ):
        raise SparseReferenceError(
            "The cover has the wrong objective dimension."
        )
    norm_p = _validated_p_norm(cover.p_norm)
    radius = float(cover.requested_cover_radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise SparseReferenceError(
            "The cover radius must be finite and nonnegative."
        )
    anchors = tuple(cover.anchor_indices)
    declared_cluster_sizes = tuple(cover.cluster_sizes)
    anchor_objectives = tuple(
        tuple(float(value) for value in point)
        for point in cover.anchor_objectives
    )
    if (
        not anchors
        or len(anchors) != len(anchor_objectives)
        or len(anchors) != len(declared_cluster_sizes)
        or len(set(anchors)) != len(anchors)
        or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            for size in declared_cluster_sizes
        )
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(points)
            for index in anchors
        )
    ):
        raise SparseReferenceError(
            "The cover anchor arrays are empty or internally inconsistent."
        )
    if any(
        anchor != points[index]
        for index, anchor in zip(anchors, anchor_objectives)
    ):
        raise SparseReferenceError(
            "Every anchor objective must match its bound reference index."
        )
    assignments = tuple(cover.assignment_by_reference)
    if len(assignments) != len(points) or any(
        isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
        or position >= len(anchors)
        for position in assignments
    ):
        raise SparseReferenceError(
            "The cover assignment contains an invalid anchor position."
        )
    observed_sizes = [0] * len(anchors)
    realized_radius = 0.0
    for point, position in zip(points, assignments):
        observed_sizes[position] += 1
        distance = _distance(
            point,
            anchor_objectives[position],
            norm_p,
        )
        if distance > radius:
            raise SparseReferenceError(
                "The declared assignment does not satisfy the cover radius."
            )
        realized_radius = max(realized_radius, distance)
    if tuple(observed_sizes) != declared_cluster_sizes:
        raise SparseReferenceError(
            "The cover cluster sizes do not match its assignments."
        )
    if not math.isclose(
        float(cover.realized_cover_radius),
        realized_radius,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise SparseReferenceError(
            "The declared realized radius does not match the bound cover."
        )
    if len(anchors) == 1:
        if cover.pairwise_anchor_separation_lower_bound is not None:
            raise SparseReferenceError(
                "A singleton cover must encode pairwise separation as null."
            )
    else:
        separation = min(
            _distance(
                anchor_objectives[left],
                anchor_objectives[right],
                norm_p,
            )
            for left in range(len(anchors))
            for right in range(left + 1, len(anchors))
        )
        declared = cover.pairwise_anchor_separation_lower_bound
        if (
            declared is None
            or not math.isfinite(float(declared))
            or separation <= radius
            or not math.isclose(
                float(declared),
                separation,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise SparseReferenceError(
                "The declared anchor separation does not match the cover."
            )
    canonical = greedy_maximal_reference_net(
        points,
        cover_radius=radius,
        p_norm=norm_p,
    )
    if cover != canonical:
        raise SparseReferenceError(
            "The cover is geometrically valid but does not match the "
            "canonical greedy construction for the bound references."
        )


def sparse_reference_metric_bounds(
    reference_points: Sequence[Sequence[float]],
    cover: SparseReferenceCover,
    *,
    cell_width_vector: Sequence[float],
    p_norm: float | None = None,
) -> SparseMetricBounds:
    """Inflate same-anchor-cell metric bounds to the full frozen reference set."""

    points = _finite_points(reference_points)
    _validate_cover_integrity(points, cover)
    widths = tuple(float(value) for value in cell_width_vector)
    if len(widths) != cover.objective_dimension or any(
        not math.isfinite(value) or value < 0.0 for value in widths
    ):
        raise SparseReferenceError(
            "cell_width_vector must be finite, nonnegative, and dimension matched."
        )
    norm_p = _validated_p_norm(
        cover.p_norm if p_norm is None else p_norm
    )

    coordinate_radii = [0.0 for _ in widths]
    for point, anchor_position in zip(points, cover.assignment_by_reference):
        anchor = cover.anchor_objectives[anchor_position]
        for coordinate, (value, anchor_value) in enumerate(zip(point, anchor)):
            coordinate_radii[coordinate] = max(
                coordinate_radii[coordinate],
                abs(value - anchor_value),
            )
    additive = tuple(
        width + radius for width, radius in zip(widths, coordinate_radii)
    )
    igd_bound = _norm(additive, norm_p)
    return SparseMetricBounds(
        cell_width_vector=widths,
        reference_to_anchor_coordinate_radii=tuple(coordinate_radii),
        additive_error_vector=additive,
        ordinary_igd_bound=igd_bound,
        igd_plus_bound=igd_bound,
        archive_cardinality_bound=len(cover.anchor_indices),
    )


def doubling_cover_cardinality_bound(
    *,
    doubling_constant: int,
    diameter: float,
    separation_radius: float,
) -> int:
    """Bound a separated finite set by an independently certified doubling constant."""

    if isinstance(doubling_constant, bool) or not isinstance(doubling_constant, int) or doubling_constant < 1:
        raise SparseReferenceError("doubling_constant must be a positive integer.")
    diameter_value = float(diameter)
    separation = float(separation_radius)
    if not math.isfinite(diameter_value) or diameter_value < 0.0:
        raise SparseReferenceError("diameter must be finite and nonnegative.")
    if not math.isfinite(separation) or separation <= 0.0:
        raise SparseReferenceError("separation_radius must be finite and positive.")
    if diameter_value == 0.0:
        return 1
    levels = max(0, math.ceil(math.log2(2.0 * diameter_value / separation)))
    return doubling_constant**levels


def sperner_many_objective_lower_bound(
    *,
    objective_dimension: int,
    additive_linf_tolerance: float,
) -> SpernerLowerBound:
    """Return an exponential worst-case feasible-cover lower bound.

    The feasible objective set is the middle Hamming layer of ``{0,1}^d``.
    Every point has the same coordinate sum, so all points are mutually
    Pareto-incomparable.  For any coordinatewise additive tolerance below one,
    no distinct feasible point covers another.  Hence every feasible additive
    cover must retain the entire middle layer.
    """

    if isinstance(objective_dimension, bool) or not isinstance(objective_dimension, int) or objective_dimension < 2:
        raise SparseReferenceError("objective_dimension must be an integer at least two.")
    tolerance = float(additive_linf_tolerance)
    if not math.isfinite(tolerance) or not (0.0 <= tolerance < 1.0):
        raise SparseReferenceError(
            "additive_linf_tolerance must lie in [0,1)."
        )
    layer = objective_dimension // 2
    count = math.comb(objective_dimension, layer)
    return SpernerLowerBound(
        objective_dimension=objective_dimension,
        layer_weight=layer,
        pareto_point_count=count,
        additive_linf_tolerance=tolerance,
        minimum_required_feasible_representatives=count,
    )


def sperner_capacity_lower_bound(
    *,
    objective_dimension: int,
    additive_linf_tolerance: float,
    maximum_certified_representatives_per_type: int,
) -> ManyObjectiveCapacityLowerBound:
    """Translate the Sperner representative bound into conditional capacities.

    A Cartesian or other cell partition whose cells all have ``L_infinity``
    diameter below one needs one occupied cell per middle-layer point.  If a
    frozen reference type is permitted to certify at most ``c`` distinct
    representatives, at least ``ceil(|P_d|/c)`` types are required.  No lower
    bound on the number of directions follows without this explicit per-type
    output-capacity premise.
    """

    lower = sperner_many_objective_lower_bound(
        objective_dimension=objective_dimension,
        additive_linf_tolerance=additive_linf_tolerance,
    )
    capacity = maximum_certified_representatives_per_type
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise SparseReferenceError(
            "maximum_certified_representatives_per_type must be a positive integer."
        )
    return ManyObjectiveCapacityLowerBound(
        objective_dimension=lower.objective_dimension,
        pareto_point_count=lower.pareto_point_count,
        minimum_required_feasible_representatives=(
            lower.minimum_required_feasible_representatives
        ),
        minimum_required_subunit_linf_cells=lower.pareto_point_count,
        maximum_certified_representatives_per_type=capacity,
        minimum_required_types=math.ceil(lower.pareto_point_count / capacity),
    )


__all__ = [
    "ManyObjectiveCapacityLowerBound",
    "SparseMetricBounds",
    "SparseReferenceCover",
    "SparseReferenceError",
    "SpernerLowerBound",
    "doubling_cover_cardinality_bound",
    "greedy_maximal_reference_net",
    "sparse_reference_metric_bounds",
    "sperner_capacity_lower_bound",
    "sperner_many_objective_lower_bound",
]
