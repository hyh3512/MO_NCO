from __future__ import annotations

"""Exact-rational hard-cap metric propagation for frozen finite references."""

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Sequence

from .pareto_frozen_cells import canonical_fraction_text


ARCHIVE_CAP_CERTIFICATE_SCHEMA_V15 = (
    "pareto_archive_cap_metric_certificate_v15"
)

RationalPoint = tuple[Fraction, ...]


class ArchiveCapCertificateError(ValueError):
    """Raised when a hard-cap metric bound cannot be verified fail-closed."""


def _as_fraction(value: Fraction | int, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise ArchiveCapCertificateError(
            f"{label} must be an exact Fraction or integer."
        )
    return Fraction(value)


def _points(
    values: Sequence[Sequence[Fraction | int]],
    *,
    label: str,
) -> tuple[RationalPoint, ...]:
    if not values:
        raise ArchiveCapCertificateError(f"{label} must be nonempty.")
    dimension = len(values[0])
    if dimension == 0:
        raise ArchiveCapCertificateError(f"{label} points must be nonempty.")
    result = []
    for point_index, point in enumerate(values):
        if len(point) != dimension:
            raise ArchiveCapCertificateError(
                f"{label}[{point_index}] has the wrong dimension."
            )
        result.append(
            tuple(
                _as_fraction(
                    value,
                    label=f"{label}[{point_index}][{coordinate}]",
                )
                for coordinate, value in enumerate(point)
            )
        )
    return tuple(result)


def _squared_l2(left: RationalPoint, right: RationalPoint) -> Fraction:
    return sum(
        ((a - b) * (a - b) for a, b in zip(left, right)),
        Fraction(0),
    )


def _distance_order_key(
    left: RationalPoint,
    right: RationalPoint,
    p: str,
) -> Fraction:
    differences = tuple(abs(a - b) for a, b in zip(left, right))
    if p == "1":
        return sum(differences, Fraction(0))
    if p == "2":
        return sum((value * value for value in differences), Fraction(0))
    if p == "infinity":
        return max(differences)
    raise ArchiveCapCertificateError("p must be '1', '2', or 'infinity'.")


def _sqrt_upper(value: Fraction, *, bits: int = 192) -> Fraction:
    if value < 0:
        raise ArchiveCapCertificateError("Cannot take sqrt of a negative value.")
    if value == 0:
        return Fraction(0)
    scaled_numerator = value.numerator << (2 * bits)
    scaled = -(-scaled_numerator // value.denominator)
    root = math.isqrt(scaled)
    if root * root < scaled:
        root += 1
    return Fraction(root, 1 << bits)


def _distance_upper(
    left: RationalPoint,
    right: RationalPoint,
    p: str,
) -> Fraction:
    key = _distance_order_key(left, right, p)
    return _sqrt_upper(key) if p == "2" else key


def canonical_gonzalez_cap(
    witnesses: Sequence[Sequence[Fraction | int]],
    *,
    cap: int,
    p: str,
) -> tuple[int, ...]:
    """Return deterministic farthest-first indices using exact comparisons."""

    points = _points(witnesses, label="witnesses")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= len(points):
        raise ArchiveCapCertificateError(
            "cap must be an integer in [1, len(witnesses)]."
        )
    _distance_order_key(points[0], points[0], p)
    retained = [0]
    while len(retained) < cap:
        candidates = []
        for index, point in enumerate(points):
            if index in retained:
                continue
            nearest = min(
                _distance_order_key(point, points[other], p)
                for other in retained
            )
            candidates.append((nearest, -index, index))
        retained.append(max(candidates)[2])
    return tuple(retained)


def _nearest_retained(
    point: RationalPoint,
    points: tuple[RationalPoint, ...],
    retained: tuple[int, ...],
    *,
    p: str,
) -> int:
    return min(
        retained,
        key=lambda index: (
            _distance_order_key(point, points[index], p),
            index,
        ),
    )


def _lp_upper(vector: RationalPoint, p: str) -> Fraction:
    if p == "1":
        return sum((abs(value) for value in vector), Fraction(0))
    if p == "2":
        return _sqrt_upper(
            sum((value * value for value in vector), Fraction(0))
        )
    if p == "infinity":
        return max((abs(value) for value in vector), default=Fraction(0))
    raise ArchiveCapCertificateError("p must be '1', '2', or 'infinity'.")


@dataclass(frozen=True)
class ArchiveCapMetricCertificate:
    schema: str
    cap_method: str
    p: str
    retained_indices: tuple[int, ...]
    reference_to_witness: tuple[int, ...]
    witness_to_retained: tuple[int, ...]
    average_cap_distortion: str
    worst_cap_radius: str
    ordinary_igd_base_upper: str
    ordinary_igd_after_cap_upper: str
    directed_coordinate_cap_radius: tuple[str, ...]
    additive_base_vector: tuple[str, ...]
    additive_after_cap_vector: tuple[str, ...]
    igd_plus_after_cap_upper: str
    shifted_front_hv_deficit_upper: str
    max_ordinary_igd: str
    max_igd_plus: str
    max_hv_deficit: str
    ordinary_igd_gate: bool
    igd_plus_gate: bool
    hv_gate: bool
    passed: bool
    numeric_contract: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def certify_archive_cap(
    *,
    reference_points: Sequence[Sequence[Fraction | int]],
    witnesses: Sequence[Sequence[Fraction | int]],
    reference_to_witness: Sequence[int],
    cap: int,
    p: str,
    ordinary_igd_base_upper: Fraction | int,
    additive_base_vector: Sequence[Fraction | int],
    hv_reference: Sequence[Fraction | int],
    max_ordinary_igd: Fraction | int,
    max_igd_plus: Fraction | int,
    max_hv_deficit: Fraction | int,
    retained_indices: Sequence[int] | None = None,
) -> ArchiveCapMetricCertificate:
    """Recompute realized cap distortion and apply exact tolerance gates."""

    references = _points(reference_points, label="reference_points")
    witness_points = _points(witnesses, label="witnesses")
    dimension = len(references[0])
    if len(witness_points[0]) != dimension:
        raise ArchiveCapCertificateError(
            "Reference and witness dimensions differ."
        )
    if len(reference_to_witness) != len(references):
        raise ArchiveCapCertificateError(
            "reference_to_witness must map every frozen reference."
        )
    mapping = tuple(reference_to_witness)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(witness_points)
        for index in mapping
    ):
        raise ArchiveCapCertificateError(
            "reference_to_witness contains an invalid index."
        )
    if retained_indices is None:
        retained = canonical_gonzalez_cap(
            witness_points,
            cap=cap,
            p=p,
        )
        method = "canonical_gonzalez_farthest_first_v1"
    else:
        retained = tuple(retained_indices)
        if (
            len(retained) != cap
            or len(set(retained)) != len(retained)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(witness_points)
                for index in retained
            )
        ):
            raise ArchiveCapCertificateError(
                "retained_indices do not define the declared nonempty cap."
            )
        method = "externally_supplied_replayed_cap_v1"
    witness_to_retained = tuple(
        _nearest_retained(
            witness,
            witness_points,
            retained,
            p=p,
        )
        for witness in witness_points
    )
    per_reference_cap = tuple(
        _distance_upper(
            witness_points[witness_index],
            witness_points[witness_to_retained[witness_index]],
            p,
        )
        for witness_index in mapping
    )
    average_cap = sum(per_reference_cap, Fraction(0)) / len(references)
    per_witness_cap = tuple(
        _distance_upper(
            witness,
            witness_points[witness_to_retained[index]],
            p,
        )
        for index, witness in enumerate(witness_points)
    )
    worst_cap = max(per_witness_cap, default=Fraction(0))
    directed = []
    for coordinate in range(dimension):
        directed.append(
            max(
                (
                    max(
                        witness_points[
                            witness_to_retained[witness_index]
                        ][coordinate]
                        - witness[coordinate],
                        Fraction(0),
                    )
                    for witness_index, witness in enumerate(witness_points)
                ),
                default=Fraction(0),
            )
        )
    base_ordinary = _as_fraction(
        ordinary_igd_base_upper,
        label="ordinary_igd_base_upper",
    )
    if base_ordinary < 0:
        raise ArchiveCapCertificateError(
            "ordinary_igd_base_upper must be nonnegative."
        )
    base_additive = tuple(
        _as_fraction(value, label=f"additive_base_vector[{index}]")
        for index, value in enumerate(additive_base_vector)
    )
    if len(base_additive) != dimension or any(
        value < 0 for value in base_additive
    ):
        raise ArchiveCapCertificateError(
            "additive_base_vector has the wrong dimension or a negative entry."
        )
    after_additive = tuple(
        base + radius for base, radius in zip(base_additive, directed)
    )
    ordinary_after = base_ordinary + average_cap
    igd_plus_after = _lp_upper(after_additive, p)
    hv_ref = tuple(
        _as_fraction(value, label=f"hv_reference[{index}]")
        for index, value in enumerate(hv_reference)
    )
    if len(hv_ref) != dimension:
        raise ArchiveCapCertificateError("hv_reference has the wrong dimension.")
    side_lengths = []
    for coordinate in range(dimension):
        minimum = min(point[coordinate] for point in references)
        if hv_ref[coordinate] <= minimum:
            raise ArchiveCapCertificateError(
                "hv_reference must be strictly worse than the frozen references."
            )
        side_lengths.append(hv_ref[coordinate] - minimum)
    hv_upper = sum(
        (
            after_additive[coordinate]
            * math.prod(
                side_lengths[other]
                for other in range(dimension)
                if other != coordinate
            )
            for coordinate in range(dimension)
        ),
        Fraction(0),
    )
    ordinary_tolerance = _as_fraction(
        max_ordinary_igd,
        label="max_ordinary_igd",
    )
    plus_tolerance = _as_fraction(max_igd_plus, label="max_igd_plus")
    hv_tolerance = _as_fraction(max_hv_deficit, label="max_hv_deficit")
    if min(ordinary_tolerance, plus_tolerance, hv_tolerance) < 0:
        raise ArchiveCapCertificateError("Metric tolerances must be nonnegative.")
    ordinary_gate = ordinary_after <= ordinary_tolerance
    plus_gate = igd_plus_after <= plus_tolerance
    hv_gate = hv_upper <= hv_tolerance
    return ArchiveCapMetricCertificate(
        schema=ARCHIVE_CAP_CERTIFICATE_SCHEMA_V15,
        cap_method=method,
        p=p,
        retained_indices=retained,
        reference_to_witness=mapping,
        witness_to_retained=witness_to_retained,
        average_cap_distortion=canonical_fraction_text(average_cap),
        worst_cap_radius=canonical_fraction_text(worst_cap),
        ordinary_igd_base_upper=canonical_fraction_text(base_ordinary),
        ordinary_igd_after_cap_upper=canonical_fraction_text(ordinary_after),
        directed_coordinate_cap_radius=tuple(
            canonical_fraction_text(value) for value in directed
        ),
        additive_base_vector=tuple(
            canonical_fraction_text(value) for value in base_additive
        ),
        additive_after_cap_vector=tuple(
            canonical_fraction_text(value) for value in after_additive
        ),
        igd_plus_after_cap_upper=canonical_fraction_text(igd_plus_after),
        shifted_front_hv_deficit_upper=canonical_fraction_text(hv_upper),
        max_ordinary_igd=canonical_fraction_text(ordinary_tolerance),
        max_igd_plus=canonical_fraction_text(plus_tolerance),
        max_hv_deficit=canonical_fraction_text(hv_tolerance),
        ordinary_igd_gate=ordinary_gate,
        igd_plus_gate=plus_gate,
        hv_gate=hv_gate,
        passed=ordinary_gate and plus_gate and hv_gate,
        numeric_contract=(
            "exact_rational_inputs_with_dyadic_upper_brackets_for_sqrt_v1"
        ),
    )


__all__ = [
    "ARCHIVE_CAP_CERTIFICATE_SCHEMA_V15",
    "ArchiveCapCertificateError",
    "ArchiveCapMetricCertificate",
    "canonical_gonzalez_cap",
    "certify_archive_cap",
]
