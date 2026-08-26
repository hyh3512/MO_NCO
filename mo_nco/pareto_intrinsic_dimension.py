from __future__ import annotations

"""Positive finite-reference complexity theorem for ordered bi-Lipschitz fronts."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Sequence

from .pareto_frozen_cells import canonical_fraction_text

INTRINSIC_DIMENSION_SCHEMA_V16 = "pareto_ordered_bilipschitz_reference_complexity_v16"

class IntrinsicDimensionError(ValueError):
    pass

def _fraction(value: Fraction | int, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise IntrinsicDimensionError(f"{label} must be a Fraction or integer.")
    return Fraction(value)

def _points(values: Sequence[Sequence[Fraction | int]]) -> tuple[tuple[Fraction, ...], ...]:
    if not values:
        raise IntrinsicDimensionError("Reference sequence must be nonempty.")
    dimension = len(values[0])
    result = []
    for i, point in enumerate(values):
        if dimension == 0 or len(point) != dimension:
            raise IntrinsicDimensionError(f"reference[{i}] has the wrong dimension.")
        result.append(tuple(_fraction(v, label=f"reference[{i}]") for v in point))
    if len(set(result)) != len(result):
        raise IntrinsicDimensionError("Ordered reference points must be distinct.")
    return tuple(result)

def _linf(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    return max((abs(a - b) for a, b in zip(left, right)), default=Fraction(0))

def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)

def _ceil_log2_fraction(value: Fraction) -> int:
    if value <= 1:
        return 0
    n, d = value.numerator, value.denominator
    exponent = max(0, n.bit_length() - d.bit_length())
    if d << exponent < n:
        exponent += 1
    while exponent > 0 and d << (exponent - 1) >= n:
        exponent -= 1
    return exponent

def canonical_maximal_tau_net(reference_points: Sequence[Sequence[Fraction | int]], *,
                              tau: Fraction | int) -> tuple[int, ...]:
    points = _points(reference_points)
    separation = _fraction(tau, label="tau")
    if separation <= 0:
        raise IntrinsicDimensionError("tau must be positive.")
    anchors: list[int] = []
    for index, point in enumerate(points):
        if all(_linf(point, points[a]) > separation for a in anchors):
            anchors.append(index)
    if any(min(_linf(point, points[a]) for a in anchors) > separation for point in points):
        raise AssertionError("Greedy anchor set is not a tau-net.")
    return tuple(anchors)

@dataclass(frozen=True)
class IntrinsicDimensionCertificate:
    schema: str
    reference_count: int
    objective_dimension: int
    lower_bilipschitz_constant: str
    upper_bilipschitz_constant: str
    doubling_constant_upper: int
    doubling_dimension_upper: str
    diameter: str
    tau: str
    anchor_indices: tuple[int, ...]
    actual_anchor_count: int
    theorem_anchor_count_upper: int
    pairwise_bilipschitz_verified: bool
    tau_net_verified: bool
    metric: str
    theorem_scope: str
    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)

def certify_ordered_bilipschitz_reference_family(
    reference_points: Sequence[Sequence[Fraction | int]], *,
    lower_constant: Fraction | int, upper_constant: Fraction | int,
    tau: Fraction | int,
) -> IntrinsicDimensionCertificate:
    """Verify c|i-j|/(N-1) <= d_inf(q_i,q_j) <= C|i-j|/(N-1).

    The ordered interval argument yields doubling constant at most
    ceil(4C/c)+1.  The standard doubling cover then bounds a maximal tau-net.
    """
    points = _points(reference_points)
    if len(points) < 2:
        raise IntrinsicDimensionError("At least two ordered references are required.")
    c = _fraction(lower_constant, label="lower_constant")
    C = _fraction(upper_constant, label="upper_constant")
    separation = _fraction(tau, label="tau")
    if c <= 0 or C < c or separation <= 0:
        raise IntrinsicDimensionError("Require 0<c<=C and tau>0.")
    denominator = len(points) - 1
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            parameter = Fraction(j - i, denominator)
            distance = _linf(points[i], points[j])
            if distance < c * parameter or distance > C * parameter:
                raise IntrinsicDimensionError(f"Pair ({i},{j}) violates bi-Lipschitz bounds.")
    diameter = max(_linf(points[i], points[j])
                   for i in range(len(points)) for j in range(i, len(points)))
    doubling = _ceil_fraction(4 * C / c) + 1
    levels = _ceil_log2_fraction(Fraction(2) * diameter / separation)
    theorem_upper = doubling ** levels
    anchors = canonical_maximal_tau_net(points, tau=separation)
    if len(anchors) > theorem_upper:
        raise AssertionError("Verified net violates the proved covering bound.")
    return IntrinsicDimensionCertificate(
        schema=INTRINSIC_DIMENSION_SCHEMA_V16, reference_count=len(points),
        objective_dimension=len(points[0]),
        lower_bilipschitz_constant=canonical_fraction_text(c),
        upper_bilipschitz_constant=canonical_fraction_text(C),
        doubling_constant_upper=doubling,
        doubling_dimension_upper=format(math.log2(doubling), ".17g"),
        diameter=canonical_fraction_text(diameter), tau=canonical_fraction_text(separation),
        anchor_indices=anchors, actual_anchor_count=len(anchors),
        theorem_anchor_count_upper=theorem_upper,
        pairwise_bilipschitz_verified=True, tau_net_verified=True,
        metric="L_infinity_on_objective_space",
        theorem_scope="finite_ordered_reference_family_with_verified_bilipschitz_parameterization",
    )

__all__ = ["INTRINSIC_DIMENSION_SCHEMA_V16", "IntrinsicDimensionCertificate",
           "IntrinsicDimensionError", "canonical_maximal_tau_net",
           "certify_ordered_bilipschitz_reference_family"]
