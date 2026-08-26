from __future__ import annotations

"""Exact edge-sum arithmetic for objective matrices stored as binary64.

Every finite binary64 edge is converted to its unique reduced dyadic
``Fraction`` and then lifted to an objective-specific common power-of-two
denominator.  Tour sums and 2-opt deltas are accumulated with Python integers,
so they are exact for the *binary64 edge values supplied to this module* and
independent of summation order.

The contract deliberately stops at the edge sum.  :meth:`decode` rounds the
exact dyadic sum back to binary64.  This module therefore provides no exactness
claim for downstream normalization, energy, cell membership, dominance, or
Metropolis--Hastings acceptance decisions.
"""

import math
import operator
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, TypeAlias


EXACT_EDGE_SUM_CONTRACT = (
    "exact_edge_sum_then_binary64_objective_v1"
)
"""The narrow, versioned contract implemented by this module."""

# Compatibility-friendly alias for callers that spell out the subject.
DYADIC_OBJECTIVE_CONTRACT = EXACT_EDGE_SUM_CONTRACT

ScaledMatrix: TypeAlias = tuple[tuple[int, ...], ...]
ScaledObjectiveMatrices: TypeAlias = tuple[ScaledMatrix, ...]
ScaledObjectiveVector: TypeAlias = tuple[int, ...]


def canonical_binary64_fraction(value: float) -> Fraction:
    """Return the exact reduced dyadic value of a finite binary64 input.

    Inputs are normalized through Python's ``float`` constructor.  Thus an
    integer input, if supplied, denotes its binary64 conversion rather than an
    arbitrary-precision integer edge.  Booleans are rejected because treating
    them as numeric edge data is almost certainly a manifest error.
    """

    if isinstance(value, bool):
        raise TypeError("A binary64 edge value cannot be Boolean.")
    try:
        binary64 = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "An edge value must be convertible to finite binary64."
        ) from exc
    if not math.isfinite(binary64):
        raise ValueError(
            "Objective matrices must contain only finite binary64 values."
        )
    numerator, denominator = binary64.as_integer_ratio()
    return Fraction(numerator, denominator)


def _denominator_exponent(value: Fraction) -> int:
    denominator = value.denominator
    if denominator <= 0 or denominator & (denominator - 1):
        raise AssertionError("A binary64 denominator must be a power of two.")
    return denominator.bit_length() - 1


def _objective_is_symmetric(matrix: ScaledMatrix) -> bool:
    """Check exact symmetry once, during encoding construction."""

    for row_index, row in enumerate(matrix):
        for column_index in range(row_index + 1, len(row)):
            if row[column_index] != matrix[column_index][row_index]:
                return False
    return True


def _index(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer.") from exc


@dataclass(frozen=True, slots=True)
class DyadicObjectiveEncoding:
    """Common-denominator integer encoding for binary64 objectives.

    ``symmetry_flags`` are computed exactly from ``scaled_matrices`` at
    construction and are the only symmetry information consulted by the 2-opt
    hot path.
    """

    scaled_matrices: ScaledObjectiveMatrices
    denominator_exponents: tuple[int, ...]
    symmetry_flags: tuple[bool, ...]
    contract_name: str = EXACT_EDGE_SUM_CONTRACT

    @classmethod
    def from_binary64_matrices(
        cls,
        objective_matrices: Sequence[
            Sequence[Sequence[float]]
        ],
    ) -> "DyadicObjectiveEncoding":
        """Build an exact dyadic encoding and cache objective symmetry."""

        raw_objectives = tuple(objective_matrices)
        if not raw_objectives:
            raise ValueError("At least one objective matrix is required.")

        first_rows = tuple(raw_objectives[0])
        city_count = len(first_rows)
        if city_count < 3:
            raise ValueError("A TSP objective requires at least three cities.")

        scaled_objectives: list[ScaledMatrix] = []
        denominator_exponents: list[int] = []
        symmetry_flags: list[bool] = []

        for objective_index, raw_matrix in enumerate(raw_objectives):
            rows = (
                first_rows
                if objective_index == 0
                else tuple(raw_matrix)
            )
            if len(rows) != city_count:
                raise ValueError(
                    "All objective matrices must have the same dimension."
                )

            fraction_rows: list[tuple[Fraction, ...]] = []
            maximum_exponent = 0
            for row in rows:
                raw_values = tuple(row)
                if len(raw_values) != city_count:
                    raise ValueError("Objective matrices must be square.")
                fraction_row = tuple(
                    canonical_binary64_fraction(value)
                    for value in raw_values
                )
                if fraction_row:
                    maximum_exponent = max(
                        maximum_exponent,
                        *(
                            _denominator_exponent(value)
                            for value in fraction_row
                        ),
                    )
                fraction_rows.append(fraction_row)

            scaled_rows: ScaledMatrix = tuple(
                tuple(
                    value.numerator
                    << (
                        maximum_exponent
                        - _denominator_exponent(value)
                    )
                    for value in row
                )
                for row in fraction_rows
            )
            scaled_objectives.append(scaled_rows)
            denominator_exponents.append(maximum_exponent)
            symmetry_flags.append(
                _objective_is_symmetric(scaled_rows)
            )

        return cls(
            scaled_matrices=tuple(scaled_objectives),
            denominator_exponents=tuple(denominator_exponents),
            symmetry_flags=tuple(symmetry_flags),
        )

    @property
    def num_objectives(self) -> int:
        return len(self.scaled_matrices)

    @property
    def objective_count(self) -> int:
        return self.num_objectives

    @property
    def num_cities(self) -> int:
        return len(self.scaled_matrices[0])

    @property
    def city_count(self) -> int:
        return self.num_cities

    @property
    def symmetric_objectives(self) -> tuple[bool, ...]:
        return self.symmetry_flags

    @property
    def contract(self) -> str:
        return self.contract_name

    def edge_fraction(
        self,
        objective_index: int,
        source: int,
        target: int,
    ) -> Fraction:
        """Return one encoded edge as an exact reduced fraction."""

        objective = _index(objective_index, "objective_index")
        source_index = _index(source, "source")
        target_index = _index(target, "target")
        if not 0 <= objective < self.num_objectives:
            raise IndexError("objective_index is out of range.")
        if not 0 <= source_index < self.num_cities:
            raise IndexError("source is out of range.")
        if not 0 <= target_index < self.num_cities:
            raise IndexError("target is out of range.")
        return Fraction(
            self.scaled_matrices[objective][source_index][target_index],
            1 << self.denominator_exponents[objective],
        )

    def scaled_as_fraction(
        self,
        scaled_values: Sequence[int],
    ) -> tuple[Fraction, ...]:
        """Decode scaled totals without losing their exact dyadic values."""

        values = self._validated_scaled_values(scaled_values)
        return tuple(
            Fraction(value, 1 << exponent)
            for value, exponent in zip(
                values,
                self.denominator_exponents,
            )
        )

    def decode(
        self,
        scaled_values: Sequence[int],
    ) -> tuple[float, ...]:
        """Round exact scaled totals to binary64 objective values.

        Distinct scaled sums may deliberately decode to the same binary64
        value.  Callers needing exact comparisons must retain the scaled
        integer or :meth:`scaled_as_fraction` representation.
        """

        decoded: list[float] = []
        for value in self.scaled_as_fraction(scaled_values):
            try:
                binary64 = float(value)
            except OverflowError as exc:
                raise OverflowError(
                    "The exact tour sum is outside finite binary64 range."
                ) from exc
            if not math.isfinite(binary64):
                raise OverflowError(
                    "The exact tour sum is outside finite binary64 range."
                )
            decoded.append(binary64)
        return tuple(decoded)

    def exact_tour_scaled_sums(
        self,
        tour: Sequence[int],
    ) -> ScaledObjectiveVector:
        """Compute exact, summation-order-independent tour edge sums."""

        normalized_tour = self._validated_tour(tour)
        totals: list[int] = []
        for matrix in self.scaled_matrices:
            total = 0
            for position, source in enumerate(normalized_tour):
                target = normalized_tour[
                    (position + 1) % self.num_cities
                ]
                total += matrix[source][target]
            totals.append(total)
        return tuple(totals)

    def update_two_opt_scaled(
        self,
        tour: Sequence[int],
        current_scaled: Sequence[int],
        i: int,
        j: int,
    ) -> ScaledObjectiveVector:
        """Update exact scaled sums after reversing ``tour[i:j+1]``.

        The caller must supply a previously validated tour and its matching
        exact scaled sums.  The method performs only constant-time tour checks
        on the symmetric hot path: rescanning a tour or an objective matrix
        here would invalidate its ``O(d)`` contract.

        Exact symmetric objectives use four boundary edges per objective.
        Exact asymmetric objectives additionally visit each internal edge of
        the reversed segment, yielding
        ``O(d_symmetric + d_asymmetric * segment_length)`` time.
        """

        if len(tour) != self.num_cities:
            raise ValueError(
                "The tour length does not match the encoded matrices."
            )
        left = _index(i, "i")
        right = _index(j, "j")
        if left > right:
            left, right = right, left
        if left <= 0 or right >= self.num_cities:
            raise ValueError(
                "2-opt indices must satisfy 1 <= i <= j < n."
            )
        current = self._validated_scaled_values(current_scaled)

        a = self._hot_path_city(tour[left - 1])
        b = self._hot_path_city(tour[left])
        c = self._hot_path_city(tour[right])
        d = self._hot_path_city(
            tour[(right + 1) % self.num_cities]
        )

        updated: list[int] = []
        for objective_index, matrix in enumerate(self.scaled_matrices):
            delta = (
                -matrix[a][b]
                - matrix[c][d]
                + matrix[a][c]
                + matrix[b][d]
            )
            if not self.symmetry_flags[objective_index]:
                for position in range(left, right):
                    source = self._hot_path_city(tour[position])
                    target = self._hot_path_city(tour[position + 1])
                    delta += (
                        matrix[target][source]
                        - matrix[source][target]
                    )
            updated.append(current[objective_index] + delta)
        return tuple(updated)

    def _validated_scaled_values(
        self,
        scaled_values: Sequence[int],
    ) -> ScaledObjectiveVector:
        if len(scaled_values) != self.num_objectives:
            raise ValueError(
                "The scaled objective vector has the wrong "
                "objective dimension."
            )
        return tuple(
            _index(value, f"scaled_values[{index}]")
            for index, value in enumerate(scaled_values)
        )

    def _validated_tour(
        self,
        tour: Sequence[int],
    ) -> tuple[int, ...]:
        if len(tour) != self.num_cities:
            raise ValueError(
                "The tour length does not match the encoded matrices."
            )
        normalized = tuple(
            _index(city, f"tour[{position}]")
            for position, city in enumerate(tour)
        )
        if set(normalized) != set(range(self.num_cities)):
            raise ValueError(
                "A tour must be a permutation of the encoded cities."
            )
        return normalized

    def _hot_path_city(self, value: object) -> int:
        city = _index(value, "tour city")
        if not 0 <= city < self.num_cities:
            raise ValueError("A tour city is outside the encoded range.")
        return city


def encode_objective_matrices(
    objective_matrices: Sequence[Sequence[Sequence[float]]],
) -> DyadicObjectiveEncoding:
    """Construct the canonical common-denominator encoding."""

    return DyadicObjectiveEncoding.from_binary64_matrices(
        objective_matrices
    )


def exact_tour_scaled_sums(
    encoding: DyadicObjectiveEncoding,
    tour: Sequence[int],
) -> ScaledObjectiveVector:
    """Functional wrapper for :meth:`exact_tour_scaled_sums`."""

    return encoding.exact_tour_scaled_sums(tour)


def update_two_opt_scaled(
    encoding: DyadicObjectiveEncoding,
    tour: Sequence[int],
    current_scaled: Sequence[int],
    i: int,
    j: int,
) -> ScaledObjectiveVector:
    """Functional wrapper for the cached-symmetry 2-opt update."""

    return encoding.update_two_opt_scaled(
        tour,
        current_scaled,
        i,
        j,
    )


# Readable construction alias for callers that prefer a verb phrase.
build_dyadic_objective_encoding = encode_objective_matrices


__all__ = [
    "DYADIC_OBJECTIVE_CONTRACT",
    "EXACT_EDGE_SUM_CONTRACT",
    "DyadicObjectiveEncoding",
    "ScaledObjectiveVector",
    "build_dyadic_objective_encoding",
    "canonical_binary64_fraction",
    "encode_objective_matrices",
    "exact_tour_scaled_sums",
    "update_two_opt_scaled",
]
