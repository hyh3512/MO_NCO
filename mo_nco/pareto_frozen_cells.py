from __future__ import annotations

"""Fail-closed frozen Cartesian-cell manifests for the v15 certificate path.

The legacy optimizer accepts floating objective boxes for its heuristic path.
This module is deliberately narrower: every numeric field is a canonical
rational string, the complete finite grid is hash-bound, and endpoint
membership is decided with exact :class:`fractions.Fraction` comparisons.
There is no tolerance, clipping, or caller-supplied disjointness assertion.
"""

import hashlib
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FROZEN_CELL_MANIFEST_SCHEMA_V15 = "pareto_frozen_cell_manifest_v15"
METRIC_SEMANTICS_V15 = (
    "arithmetic_mean_over_references_of_nearest_local_lp_distance_v1"
)
BOUNDARY_CONVENTION_V15 = "half_open_cells_global_upper_closed_v1"
PROBABILITY_SEMANTICS_V15 = (
    "ideal_product_random_streams_python_prng_is_replay_only_v1"
)
OBJECTIVE_ARITHMETIC_V15 = (
    "exact_binary64_dyadic_edge_sum_for_endpoint_classification_v1"
)

Cell = tuple[int, ...]
RationalPoint = tuple[Fraction, ...]


class FrozenCellManifestError(ValueError):
    """Raised whenever a formal cell manifest cannot be verified exactly."""


def canonical_fraction_text(value: Fraction | int) -> str:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise FrozenCellManifestError(
            "Canonical manifest rationals must be Fraction or integer values."
        )
    resolved = Fraction(value)
    if resolved.denominator == 1:
        return str(resolved.numerator)
    return f"{resolved.numerator}/{resolved.denominator}"


def parse_canonical_fraction(value: object, *, label: str) -> Fraction:
    if not isinstance(value, str) or not value:
        raise FrozenCellManifestError(
            f"{label} must be a canonical rational string."
        )
    try:
        resolved = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise FrozenCellManifestError(
            f"{label} is not a valid rational string."
        ) from error
    if canonical_fraction_text(resolved) != value:
        raise FrozenCellManifestError(
            f"{label} is not canonical; expected "
            f"{canonical_fraction_text(resolved)!r}."
        )
    return resolved


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenCellManifestError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise FrozenCellManifestError(
            f"{label} fields differ from the v15 contract; "
            f"missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}."
        )


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _coerce_cells(
    raw_cells: object,
    *,
    dimension: int,
    counts: tuple[int, ...],
    label: str,
) -> tuple[Cell, ...]:
    if not isinstance(raw_cells, list):
        raise FrozenCellManifestError(f"{label} must be a JSON array.")
    cells: list[Cell] = []
    for cell_index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, list) or len(raw_cell) != dimension:
            raise FrozenCellManifestError(
                f"{label}[{cell_index}] has the wrong dimension."
            )
        cell: list[int] = []
        for coordinate, (raw_index, count) in enumerate(
            zip(raw_cell, counts)
        ):
            if (
                isinstance(raw_index, bool)
                or not isinstance(raw_index, int)
                or raw_index < 0
                or raw_index >= count
            ):
                raise FrozenCellManifestError(
                    f"{label}[{cell_index}][{coordinate}] is outside the "
                    "frozen grid."
                )
            cell.append(raw_index)
        cells.append(tuple(cell))
    if len(set(cells)) != len(cells):
        raise FrozenCellManifestError(f"{label} contains duplicate cells.")
    return tuple(cells)


@dataclass(frozen=True)
class FrozenCellManifest:
    schema: str
    raw_sha256: str
    lower: RationalPoint
    upper: RationalPoint
    widths: RationalPoint
    counts: tuple[int, ...]
    partition_cells: tuple[Cell, ...]
    observable_cells: tuple[Cell, ...]
    local_norm_p: str
    metric_semantics: str
    boundary_convention: str
    probability_semantics: str
    objective_arithmetic: str

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def classify(self, objective: Sequence[Fraction | int]) -> Cell:
        if len(objective) != self.dimension:
            raise FrozenCellManifestError(
                "Endpoint objective has the wrong dimension."
            )
        cell: list[int] = []
        for coordinate, (raw_value, low, high, width, count) in enumerate(
            zip(objective, self.lower, self.upper, self.widths, self.counts)
        ):
            if isinstance(raw_value, bool) or not isinstance(
                raw_value,
                (Fraction, int),
            ):
                raise FrozenCellManifestError(
                    "Endpoint classification requires exact Fraction or "
                    f"integer input at coordinate {coordinate}."
                )
            value = Fraction(raw_value)
            if value < low or value > high:
                raise FrozenCellManifestError(
                    f"Endpoint objective coordinate {coordinate}={value} "
                    f"leaves the exact frozen box [{low}, {high}]."
                )
            if value == high:
                index = count - 1
            else:
                quotient = (value - low) / width
                index = quotient.numerator // quotient.denominator
            if index < 0 or index >= count:
                raise FrozenCellManifestError(
                    f"Endpoint mapped to invalid cell coordinate {index}."
                )
            cell.append(index)
        resolved = tuple(cell)
        if resolved not in self.partition_cells:
            raise FrozenCellManifestError(
                f"Endpoint cell {resolved!r} is not in the hash-frozen family."
            )
        return resolved

    def is_observable(self, cell: Cell) -> bool:
        if cell not in self.partition_cells:
            raise FrozenCellManifestError(
                "Cell is not in the frozen partition family."
            )
        return cell in self.observable_cells


def load_frozen_cell_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FrozenCellManifest:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise FrozenCellManifestError(
            "Frozen-cell manifest SHA-256 does not match the commitment."
        )
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenCellManifestError(
            "Frozen-cell manifest is not strict UTF-8 JSON."
        ) from error
    if not isinstance(payload, dict):
        raise FrozenCellManifestError("Frozen-cell manifest root must be an object.")
    _exact_keys(
        payload,
        {
            "schema",
            "box",
            "partition_cells",
            "observable_cells",
            "metric",
            "boundary_convention",
            "probability_semantics",
            "objective_arithmetic",
        },
        label="manifest",
    )
    if payload["schema"] != FROZEN_CELL_MANIFEST_SCHEMA_V15:
        raise FrozenCellManifestError("Unexpected frozen-cell manifest schema.")
    box = payload["box"]
    metric = payload["metric"]
    if not isinstance(box, dict) or not isinstance(metric, dict):
        raise FrozenCellManifestError("box and metric must be JSON objects.")
    _exact_keys(box, {"lower", "upper", "widths"}, label="box")
    _exact_keys(
        metric,
        {"local_norm_p", "reference_aggregation"},
        label="metric",
    )
    raw_lower = box["lower"]
    raw_upper = box["upper"]
    raw_widths = box["widths"]
    if not all(isinstance(item, list) for item in (raw_lower, raw_upper, raw_widths)):
        raise FrozenCellManifestError("box coordinates must be JSON arrays.")
    dimension = len(raw_lower)
    if (
        dimension == 0
        or len(raw_upper) != dimension
        or len(raw_widths) != dimension
    ):
        raise FrozenCellManifestError("Frozen box dimensions are inconsistent.")
    lower = tuple(
        parse_canonical_fraction(value, label=f"box.lower[{index}]")
        for index, value in enumerate(raw_lower)
    )
    upper = tuple(
        parse_canonical_fraction(value, label=f"box.upper[{index}]")
        for index, value in enumerate(raw_upper)
    )
    widths = tuple(
        parse_canonical_fraction(value, label=f"box.widths[{index}]")
        for index, value in enumerate(raw_widths)
    )
    if any(high <= low for low, high in zip(lower, upper)):
        raise FrozenCellManifestError("Every frozen upper bound must exceed lower.")
    if any(width <= 0 for width in widths):
        raise FrozenCellManifestError("Every frozen cell width must be positive.")
    counts = tuple(
        _ceil_fraction((high - low) / width)
        for low, high, width in zip(lower, upper, widths)
    )
    partition = _coerce_cells(
        payload["partition_cells"],
        dimension=dimension,
        counts=counts,
        label="partition_cells",
    )
    full_partition = tuple(itertools.product(*(range(count) for count in counts)))
    if set(partition) != set(full_partition):
        raise FrozenCellManifestError(
            "partition_cells must enumerate the complete Cartesian grid; "
            "omissions would make endpoint membership undefined."
        )
    observable = _coerce_cells(
        payload["observable_cells"],
        dimension=dimension,
        counts=counts,
        label="observable_cells",
    )
    if not set(observable).issubset(partition):
        raise FrozenCellManifestError(
            "observable_cells must be a subset of partition_cells."
        )
    if metric["reference_aggregation"] != METRIC_SEMANTICS_V15:
        raise FrozenCellManifestError(
            "The metric manifest must use v15 arithmetic-mean IGD semantics."
        )
    local_norm = metric["local_norm_p"]
    if local_norm not in {"1", "2", "infinity"}:
        raise FrozenCellManifestError(
            "metric.local_norm_p must be '1', '2', or 'infinity'."
        )
    if payload["boundary_convention"] != BOUNDARY_CONVENTION_V15:
        raise FrozenCellManifestError("Unexpected cell-boundary convention.")
    if payload["probability_semantics"] != PROBABILITY_SEMANTICS_V15:
        raise FrozenCellManifestError(
            "Probability semantics must separate the ideal theorem from PRNG replay."
        )
    if payload["objective_arithmetic"] != OBJECTIVE_ARITHMETIC_V15:
        raise FrozenCellManifestError(
            "Unexpected endpoint objective-arithmetic contract."
        )
    return FrozenCellManifest(
        schema=FROZEN_CELL_MANIFEST_SCHEMA_V15,
        raw_sha256=digest,
        lower=lower,
        upper=upper,
        widths=widths,
        counts=counts,
        partition_cells=tuple(sorted(partition)),
        observable_cells=tuple(sorted(observable)),
        local_norm_p=local_norm,
        metric_semantics=METRIC_SEMANTICS_V15,
        boundary_convention=BOUNDARY_CONVENTION_V15,
        probability_semantics=PROBABILITY_SEMANTICS_V15,
        objective_arithmetic=OBJECTIVE_ARITHMETIC_V15,
    )


def canonical_manifest_payload(
    *,
    lower: Sequence[Fraction | int],
    upper: Sequence[Fraction | int],
    widths: Sequence[Fraction | int],
    observable_cells: Iterable[Cell],
    local_norm_p: str = "2",
) -> dict[str, object]:
    def exact_tuple(
        values: Sequence[Fraction | int],
        *,
        label: str,
    ) -> tuple[Fraction, ...]:
        resolved: list[Fraction] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(
                value,
                (Fraction, int),
            ):
                raise FrozenCellManifestError(
                    f"{label}[{index}] must be a Fraction or integer."
                )
            resolved.append(Fraction(value))
        return tuple(resolved)

    lower_q = exact_tuple(lower, label="lower")
    upper_q = exact_tuple(upper, label="upper")
    widths_q = exact_tuple(widths, label="widths")
    if not (len(lower_q) == len(upper_q) == len(widths_q)) or not lower_q:
        raise FrozenCellManifestError("Programmatic manifest dimensions differ.")
    if any(high <= low for low, high in zip(lower_q, upper_q)):
        raise FrozenCellManifestError(
            "Every programmatic upper bound must exceed its lower bound."
        )
    if any(width <= 0 for width in widths_q):
        raise FrozenCellManifestError(
            "Every programmatic cell width must be positive."
        )
    if local_norm_p not in {"1", "2", "infinity"}:
        raise FrozenCellManifestError(
            "local_norm_p must be '1', '2', or 'infinity'."
        )
    counts = tuple(
        _ceil_fraction((high - low) / width)
        for low, high, width in zip(lower_q, upper_q, widths_q)
    )
    partition = tuple(itertools.product(*(range(count) for count in counts)))
    try:
        raw_observable = [list(cell) for cell in observable_cells]
    except TypeError as error:
        raise FrozenCellManifestError(
            "observable_cells must be an iterable of cell coordinates."
        ) from error
    observable = _coerce_cells(
        raw_observable,
        dimension=len(lower_q),
        counts=counts,
        label="observable_cells",
    )
    return {
        "schema": FROZEN_CELL_MANIFEST_SCHEMA_V15,
        "box": {
            "lower": [canonical_fraction_text(value) for value in lower_q],
            "upper": [canonical_fraction_text(value) for value in upper_q],
            "widths": [canonical_fraction_text(value) for value in widths_q],
        },
        "partition_cells": [list(cell) for cell in partition],
        "observable_cells": [list(cell) for cell in sorted(observable)],
        "metric": {
            "local_norm_p": local_norm_p,
            "reference_aggregation": METRIC_SEMANTICS_V15,
        },
        "boundary_convention": BOUNDARY_CONVENTION_V15,
        "probability_semantics": PROBABILITY_SEMANTICS_V15,
        "objective_arithmetic": OBJECTIVE_ARITHMETIC_V15,
    }


__all__ = [
    "BOUNDARY_CONVENTION_V15",
    "FROZEN_CELL_MANIFEST_SCHEMA_V15",
    "FrozenCellManifest",
    "FrozenCellManifestError",
    "METRIC_SEMANTICS_V15",
    "OBJECTIVE_ARITHMETIC_V15",
    "PROBABILITY_SEMANTICS_V15",
    "canonical_fraction_text",
    "canonical_manifest_payload",
    "load_frozen_cell_manifest",
    "parse_canonical_fraction",
]
