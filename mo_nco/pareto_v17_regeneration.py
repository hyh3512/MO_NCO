"""Exact finite-state regeneration certificates for the Pareto-SMC v17 packet.

This module implements the deterministic algebra used by the final-regeneration
transfer theorem.  All certificate-facing quantities use ``fractions.Fraction``
so that no floating-point rounding can turn a failed inequality into a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from operator import mul
from typing import Iterable, Mapping, Sequence


class RegenerationCertificateError(ValueError):
    """Raised when a regeneration contract is malformed."""


def as_fraction(value: Fraction | int | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"unsupported exact rational value: {type(value)!r}")


def product(values: Iterable[Fraction]) -> Fraction:
    return reduce(mul, values, Fraction(1, 1))


@dataclass(frozen=True)
class MinorizationBlock:
    """A block of repeated kernels with common minorization coefficient.

    ``epsilon`` certifies ``K(x, ·) >= epsilon * pi(·)`` and ``steps`` is the
    number of consecutive applications.  The block residual is
    ``(1-epsilon)**steps``.
    """

    epsilon: Fraction
    steps: int

    def __post_init__(self) -> None:
        eps = as_fraction(self.epsilon)
        object.__setattr__(self, "epsilon", eps)
        if eps < 0 or eps > 1:
            raise RegenerationCertificateError("minorization epsilon must lie in [0,1]")
        if not isinstance(self.steps, int) or self.steps < 0:
            raise RegenerationCertificateError("steps must be a nonnegative integer")

    @property
    def residual(self) -> Fraction:
        return (Fraction(1, 1) - self.epsilon) ** self.steps


@dataclass(frozen=True)
class TypeRegenerationCertificate:
    type_id: str
    blocks: tuple[MinorizationBlock, ...]

    def __post_init__(self) -> None:
        if not self.type_id:
            raise RegenerationCertificateError("type_id must be nonempty")
        if not self.blocks:
            raise RegenerationCertificateError("at least one final-regeneration block is required")

    @property
    def residual(self) -> Fraction:
        """The exact residual ``b = prod_t (1-epsilon_t)**s_t``."""

        return product(block.residual for block in self.blocks)

    @property
    def target_component(self) -> Fraction:
        return Fraction(1, 1) - self.residual


@dataclass(frozen=True)
class RegenerationTransfer:
    """A canonical collection of type-wise regeneration certificates."""

    types: tuple[TypeRegenerationCertificate, ...]

    def __post_init__(self) -> None:
        ids = [item.type_id for item in self.types]
        if not ids:
            raise RegenerationCertificateError("at least one type is required")
        if len(ids) != len(set(ids)):
            raise RegenerationCertificateError("duplicate type_id in regeneration transfer")

    @property
    def residuals(self) -> Mapping[str, Fraction]:
        return {item.type_id: item.residual for item in self.types}

    @property
    def target_components(self) -> Mapping[str, Fraction]:
        return {item.type_id: item.target_component for item in self.types}


def endpoint_probability_bounds(
    target_probability: Fraction | int | str,
    residual: Fraction | int | str,
) -> tuple[Fraction, Fraction]:
    """Return the exact regeneration sandwich for one observable.

    If ``K^s = (1-b) Pi + b R`` and ``pi(C)=p``, then every initial law has

    ``(1-b)p <= nu K^s(C) <= (1-b)p + b``.
    """

    p = as_fraction(target_probability)
    b = as_fraction(residual)
    if p < 0 or p > 1 or b < 0 or b > 1:
        raise RegenerationCertificateError("probability and residual must lie in [0,1]")
    lower = (Fraction(1, 1) - b) * p
    upper = lower + b
    return lower, upper


def target_probability_lower_from_endpoint(
    endpoint_lower: Fraction | int | str,
    residual: Fraction | int | str,
) -> Fraction:
    """Invert the upper side of the sandwich conservatively.

    From ``q <= (1-b)p+b`` and a valid lower bound ``q >= q_lower`` one gets
    ``p >= max(0, (q_lower-b)/(1-b))``.  This inversion is meaningful only for
    ``b < 1``.
    """

    q = as_fraction(endpoint_lower)
    b = as_fraction(residual)
    if q < 0 or q > 1 or b < 0 or b > 1:
        raise RegenerationCertificateError("probability and residual must lie in [0,1]")
    if b == 1:
        return Fraction(0, 1)
    return max(Fraction(0, 1), (q - b) / (Fraction(1, 1) - b))


def endpoint_probability_lower_from_target(
    target_lower: Fraction | int | str,
    residual: Fraction | int | str,
) -> Fraction:
    p = as_fraction(target_lower)
    b = as_fraction(residual)
    if p < 0 or p > 1 or b < 0 or b > 1:
        raise RegenerationCertificateError("probability and residual must lie in [0,1]")
    return (Fraction(1, 1) - b) * p


def conditional_cell_miss_bound(
    endpoint_lower_by_type: Sequence[Fraction | int | str],
    particles_by_type: Sequence[int],
) -> Fraction:
    """Exact conditional miss bound under independent final random tapes."""

    if len(endpoint_lower_by_type) != len(particles_by_type):
        raise RegenerationCertificateError("probability/count dimensions differ")
    factors: list[Fraction] = []
    for p_raw, count in zip(endpoint_lower_by_type, particles_by_type, strict=True):
        p = as_fraction(p_raw)
        if p < 0 or p > 1:
            raise RegenerationCertificateError("endpoint lower probability must lie in [0,1]")
        if not isinstance(count, int) or count < 0:
            raise RegenerationCertificateError("particle count must be a nonnegative integer")
        factors.append((Fraction(1, 1) - p) ** count)
    return product(factors)


def simultaneous_union_miss_bound(
    endpoint_lower_matrix: Sequence[Sequence[Fraction | int | str]],
    particles_by_type: Sequence[int],
) -> Fraction:
    """Union bound over cells for a type-by-cell endpoint lower matrix."""

    if not endpoint_lower_matrix:
        raise RegenerationCertificateError("at least one type is required")
    r_count = len(endpoint_lower_matrix)
    if r_count != len(particles_by_type):
        raise RegenerationCertificateError("type/count dimensions differ")
    j_count = len(endpoint_lower_matrix[0])
    if j_count == 0:
        raise RegenerationCertificateError("at least one cell is required")
    if any(len(row) != j_count for row in endpoint_lower_matrix):
        raise RegenerationCertificateError("ragged endpoint lower matrix")

    total = Fraction(0, 1)
    for j in range(j_count):
        total += conditional_cell_miss_bound(
            [endpoint_lower_matrix[r][j] for r in range(r_count)],
            particles_by_type,
        )
    return min(Fraction(1, 1), total)


def verify_finite_kernel_regeneration(
    kernel: Sequence[Sequence[Fraction | int | str]],
    stationary: Sequence[Fraction | int | str],
    epsilon: Fraction | int | str,
) -> dict[str, object]:
    """Exact finite-matrix audit of invariance and minorization.

    This helper is used only for tiny-state adversarial checks.  It verifies
    row-stochasticity, ``pi K = pi``, and ``K(x,y) >= epsilon*pi(y)`` exactly.
    """

    eps = as_fraction(epsilon)
    pi = tuple(as_fraction(x) for x in stationary)
    k = tuple(tuple(as_fraction(x) for x in row) for row in kernel)
    n = len(pi)
    if n == 0 or len(k) != n or any(len(row) != n for row in k):
        raise RegenerationCertificateError("kernel must be a nonempty square matrix")
    if sum(pi, Fraction(0, 1)) != 1 or any(x < 0 for x in pi):
        raise RegenerationCertificateError("stationary law is not a probability vector")
    row_ok = all(sum(row, Fraction(0, 1)) == 1 and all(x >= 0 for x in row) for row in k)
    invariant = tuple(
        sum(pi[x] * k[x][y] for x in range(n)) for y in range(n)
    )
    invariant_ok = invariant == pi
    minorization_ok = all(k[x][y] >= eps * pi[y] for x in range(n) for y in range(n))
    return {
        "row_stochastic": row_ok,
        "stationary_invariant": invariant_ok,
        "minorization": minorization_ok,
        "pass": row_ok and invariant_ok and minorization_ok,
        "residual": str(Fraction(1, 1) - eps),
    }
