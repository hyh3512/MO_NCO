"""Exact-random-bit Metropolis acceptance for Pareto-SMC v19.

The certified branch in v18 separated an ideal exact-real MH kernel from the
binary64 implementation by an externally supplied TV perturbation bound.  This
module adds an executable exact branch under an *ideal independent random-bit*
model.

All energies, temperatures and objective-box data are rational.  For an uphill
move with rational exponent ``x = beta * (U(y)-U(x)) > 0`` the acceptance
probability is ``exp(-x)``.  A lazy binary expansion of a uniform random
variable is compared against nested rational Taylor intervals for ``exp(-x)``.
The decision is almost surely finite and its acceptance probability is exactly
``exp(-x)`` under the ideal random-bit model.

A deterministic PRNG can provide replayable bits to the implementation, but
that operational fact is not a proof that the concrete PRNG stream is an ideal
product random-bit process.  The certificate records this distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Protocol, Sequence


class ExactMHError(ValueError):
    pass


class RandomBitSource(Protocol):
    def getrandbits(self, k: int) -> int: ...


def as_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ExactMHError("boolean is not a rational scalar")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExactMHError("non-finite binary64 scalar")
        return Fraction.from_float(value)
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ExactMHError(f"invalid rational scalar: {value!r}") from exc


def _ceil_fraction(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _exp_positive_taylor_interval(x: Fraction, order: int) -> tuple[Fraction, Fraction]:
    """Return rational ``[L,U]`` containing ``exp(x)`` for ``x>=0``.

    ``L`` is the degree-``order`` Taylor partial sum.  Once ``order+2>x``, the
    remaining positive tail is dominated by a geometric series whose first
    term is ``x**(order+1)/(order+1)!`` and ratio ``x/(order+2)``.
    """

    if x < 0:
        raise ExactMHError("Taylor interval requires x>=0")
    if order < 0:
        raise ExactMHError("Taylor order must be nonnegative")
    if x == 0:
        return Fraction(1, 1), Fraction(1, 1)
    if Fraction(order + 2, 1) <= x:
        raise ExactMHError("Taylor order is too small for a geometric tail bound")

    term = Fraction(1, 1)
    lower = Fraction(1, 1)
    for k in range(1, order + 1):
        term *= x
        term /= k
        lower += term
    next_term = term * x / (order + 1)
    ratio = x / (order + 2)
    upper = lower + next_term / (1 - ratio)
    return lower, upper


def exp_neg_rational_interval(
    x: Fraction | int | float | str,
    *,
    order: int = 16,
    subdivisions: int | None = None,
) -> tuple[Fraction, Fraction]:
    """Return rational ``[L,U]`` containing ``exp(-x)`` for ``x>=0``.

    Range reduction writes ``x = m*y`` with ``0<=y<=1`` and raises a rational
    interval for ``exp(-y)`` to the integer power ``m``.  This keeps the Taylor
    order practical even when ``x`` is moderately large.
    """

    q = as_fraction(x)
    if q < 0:
        raise ExactMHError("exp(-x) interval requires x>=0")
    if q == 0:
        return Fraction(1, 1), Fraction(1, 1)
    m = subdivisions if subdivisions is not None else max(1, _ceil_fraction(q))
    if not isinstance(m, int) or m <= 0:
        raise ExactMHError("subdivisions must be a positive integer")
    y = q / m
    min_order = max(order, _ceil_fraction(y) - 1)
    exp_lo, exp_hi = _exp_positive_taylor_interval(y, min_order)
    one_step_lo = Fraction(1, 1) / exp_hi
    one_step_hi = Fraction(1, 1) / exp_lo
    return one_step_lo**m, one_step_hi**m




@dataclass(frozen=True)
class ExactRandBelowDecision:
    value: int
    modulus: int
    bits_per_draw: int
    draws: int
    draw_cap: int | None
    semantics: str = "ideal_rejection_uniform_integer_from_random_bits_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "modulus": self.modulus,
            "bits_per_draw": self.bits_per_draw,
            "draws": self.draws,
            "draw_cap": self.draw_cap,
            "semantics": self.semantics,
        }


def exact_randbelow(
    modulus: int,
    bit_source: RandomBitSource,
    *,
    max_draws: int | None = None,
) -> ExactRandBelowDecision:
    """Draw exactly uniformly from ``{0,...,modulus-1}`` by rejection.

    Under ideal independent bits, each accepted integer has the same number of
    binary preimages and the stopping time is almost surely finite.  The
    executable path fails closed rather than introducing modulo bias.
    """

    if not isinstance(modulus, int) or modulus <= 0:
        raise ExactMHError("modulus must be a positive integer")
    if max_draws is not None and (
        not isinstance(max_draws, int) or max_draws <= 0
    ):
        raise ExactMHError("max_draws must be None or a positive integer")
    if modulus == 1:
        return ExactRandBelowDecision(0, 1, 0, 0, max_draws)
    bits = (modulus - 1).bit_length()
    draw = 0
    while max_draws is None or draw < max_draws:
        draw += 1
        candidate = int(bit_source.getrandbits(bits))
        if candidate < 0 or candidate >= 1 << bits:
            raise ExactMHError("bit source returned an invalid integer chunk")
        if candidate < modulus:
            return ExactRandBelowDecision(candidate, modulus, bits, draw, max_draws)
    raise ExactMHError("exact randbelow rejection cap reached")


def exact_uniform_two_opt_indices(
    num_cities: int,
    bit_source: RandomBitSource,
    *,
    max_draws: int | None = None,
) -> tuple[int, int, ExactRandBelowDecision]:
    """Uniformly sample an unordered pair from positions ``1,...,n-1``."""

    if not isinstance(num_cities, int) or num_cities < 4:
        raise ExactMHError("2-opt requires at least four cities")
    tail = num_cities - 1
    pair_count = tail * (tail - 1) // 2
    decision = exact_randbelow(pair_count, bit_source, max_draws=max_draws)
    rank = decision.value
    # Lexicographic unranking of 0 <= a < b < tail.
    for a in range(tail - 1):
        width = tail - a - 1
        if rank < width:
            b = a + 1 + rank
            return a + 1, b + 1, decision
        rank -= width
    raise AssertionError("pair unranking failed")


def exact_uniform_fixed_origin_tour(
    num_cities: int,
    bit_source: RandomBitSource,
    *,
    max_draws_per_step: int | None = None,
) -> tuple[tuple[int, ...], tuple[ExactRandBelowDecision, ...]]:
    """Uniform fixed-origin tour using exact Fisher--Yates integer draws."""

    if not isinstance(num_cities, int) or num_cities < 3:
        raise ExactMHError("a tour requires at least three cities")
    tail = list(range(1, num_cities))
    decisions: list[ExactRandBelowDecision] = []
    for i in range(len(tail) - 1, 0, -1):
        decision = exact_randbelow(
            i + 1,
            bit_source,
            max_draws=max_draws_per_step,
        )
        decisions.append(decision)
        j = decision.value
        tail[i], tail[j] = tail[j], tail[i]
    return (0, *tail), tuple(decisions)


@dataclass(frozen=True)
class ExactBernoulliDecision:
    accepted: bool
    exponent: Fraction
    random_prefix_numerator: int
    random_prefix_bits: int
    probability_lower: Fraction
    probability_upper: Fraction
    taylor_order: int
    rounds: int
    refinement_cap: int | None
    semantics: str = "ideal_lazy_uniform_bits_vs_rational_exp_interval_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "exponent": str(self.exponent),
            "random_prefix_numerator": self.random_prefix_numerator,
            "random_prefix_bits": self.random_prefix_bits,
            "probability_lower": str(self.probability_lower),
            "probability_upper": str(self.probability_upper),
            "taylor_order": self.taylor_order,
            "rounds": self.rounds,
            "refinement_cap": self.refinement_cap,
            "semantics": self.semantics,
        }


def exact_bernoulli_exp_neg(
    x: Fraction | int | float | str,
    bit_source: RandomBitSource,
    *,
    bits_per_round: int = 16,
    initial_order: int = 8,
    max_rounds: int | None = None,
) -> ExactBernoulliDecision:
    """Draw an exact Bernoulli with probability ``exp(-x)``.

    The theorem is with respect to an ideal infinite i.i.d. bit stream.  The
    executable function is fail-closed if the numerical refinement cap is
    reached; it never guesses a decision from overlapping intervals.
    """

    q = as_fraction(x)
    if q < 0:
        raise ExactMHError("Bernoulli exponent must be nonnegative")
    if bits_per_round <= 0 or initial_order <= 0:
        raise ExactMHError("bits_per_round and initial_order must be positive")
    if max_rounds is not None and (
        not isinstance(max_rounds, int) or max_rounds <= 0
    ):
        raise ExactMHError("max_rounds must be None or a positive integer")
    if q == 0:
        return ExactBernoulliDecision(
            accepted=True,
            exponent=q,
            random_prefix_numerator=0,
            random_prefix_bits=0,
            probability_lower=Fraction(1, 1),
            probability_upper=Fraction(1, 1),
            taylor_order=0,
            rounds=0,
            refinement_cap=max_rounds,
        )

    prefix = 0
    prefix_bits = 0
    order = max(initial_order, _ceil_fraction(q / max(1, _ceil_fraction(q))) - 1)
    probability_lower, probability_upper = exp_neg_rational_interval(q, order=order)

    round_index = 0
    while max_rounds is None or round_index < max_rounds:
        round_index += 1
        chunk = int(bit_source.getrandbits(bits_per_round))
        if chunk < 0 or chunk >= 1 << bits_per_round:
            raise ExactMHError("bit source returned an invalid chunk")
        prefix = (prefix << bits_per_round) | chunk
        prefix_bits += bits_per_round
        denominator = 1 << prefix_bits
        uniform_lower = Fraction(prefix, denominator)
        uniform_upper = Fraction(prefix + 1, denominator)

        if uniform_upper <= probability_lower:
            return ExactBernoulliDecision(
                accepted=True,
                exponent=q,
                random_prefix_numerator=prefix,
                random_prefix_bits=prefix_bits,
                probability_lower=probability_lower,
                probability_upper=probability_upper,
                taylor_order=order,
                rounds=round_index,
                refinement_cap=max_rounds,
            )
        if uniform_lower >= probability_upper:
            return ExactBernoulliDecision(
                accepted=False,
                exponent=q,
                random_prefix_numerator=prefix,
                random_prefix_bits=prefix_bits,
                probability_lower=probability_lower,
                probability_upper=probability_upper,
                taylor_order=order,
                rounds=round_index,
                refinement_cap=max_rounds,
            )

        order *= 2
        probability_lower, probability_upper = exp_neg_rational_interval(q, order=order)

    raise ExactMHError(
        "exact Bernoulli refinement cap reached while intervals still overlap"
    )


def exact_augmented_tchebycheff_energy(
    objective: Sequence[Fraction | int | float | str],
    lower: Sequence[Fraction | int | float | str],
    upper: Sequence[Fraction | int | float | str],
    weights: Sequence[Fraction | int | float | str],
    rho: Fraction | int | float | str,
) -> Fraction:
    """Exact rational energy for the stored finite-precision contract."""

    f = tuple(as_fraction(x) for x in objective)
    lo = tuple(as_fraction(x) for x in lower)
    hi = tuple(as_fraction(x) for x in upper)
    w = tuple(as_fraction(x) for x in weights)
    rho_q = as_fraction(rho)
    if not f or not (len(f) == len(lo) == len(hi) == len(w)):
        raise ExactMHError("objective, box and weight dimensions must agree")
    if any(b <= a for a, b in zip(lo, hi, strict=True)):
        raise ExactMHError("every objective-box width must be positive")
    if any(weight <= 0 for weight in w) or rho_q < 0:
        raise ExactMHError("weights must be positive and rho nonnegative")
    if sum(w, Fraction(0, 1)) != 1:
        raise ExactMHError("exact certified weights must sum to one")
    z: list[Fraction] = []
    for value, a, b in zip(f, lo, hi, strict=True):
        if value < a or value > b:
            raise ExactMHError("objective left the frozen exact box")
        z.append((value - a) / (b - a))
    weighted = tuple(weight * value for weight, value in zip(w, z, strict=True))
    return max(weighted) + rho_q * sum(weighted, Fraction(0, 1))


@dataclass(frozen=True)
class ExactMetropolisDecision:
    accepted: bool
    current_energy: Fraction
    proposed_energy: Fraction
    beta: Fraction
    exponent: Fraction
    bernoulli: ExactBernoulliDecision | None
    semantics: str = "exact_rational_energy_lazy_random_bit_metropolis_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "current_energy": str(self.current_energy),
            "proposed_energy": str(self.proposed_energy),
            "beta": str(self.beta),
            "exponent": str(self.exponent),
            "bernoulli": None if self.bernoulli is None else self.bernoulli.to_dict(),
            "semantics": self.semantics,
        }


def exact_metropolis_accept(
    current_energy: Fraction | int | float | str,
    proposed_energy: Fraction | int | float | str,
    beta: Fraction | int | float | str,
    bit_source: RandomBitSource,
    *,
    bits_per_round: int = 16,
    initial_order: int = 8,
    max_rounds: int | None = None,
) -> ExactMetropolisDecision:
    current = as_fraction(current_energy)
    proposed = as_fraction(proposed_energy)
    beta_q = as_fraction(beta)
    if beta_q < 0:
        raise ExactMHError("beta must be nonnegative")
    exponent = beta_q * (proposed - current)
    if exponent <= 0:
        return ExactMetropolisDecision(
            accepted=True,
            current_energy=current,
            proposed_energy=proposed,
            beta=beta_q,
            exponent=exponent,
            bernoulli=None,
        )
    decision = exact_bernoulli_exp_neg(
        exponent,
        bit_source,
        bits_per_round=bits_per_round,
        initial_order=initial_order,
        max_rounds=max_rounds,
    )
    return ExactMetropolisDecision(
        accepted=decision.accepted,
        current_energy=current,
        proposed_energy=proposed,
        beta=beta_q,
        exponent=exponent,
        bernoulli=decision,
    )


__all__ = [
    "ExactBernoulliDecision",
    "ExactMHError",
    "ExactMetropolisDecision",
    "ExactRandBelowDecision",
    "RandomBitSource",
    "as_fraction",
    "exact_augmented_tchebycheff_energy",
    "exact_bernoulli_exp_neg",
    "exact_metropolis_accept",
    "exact_randbelow",
    "exact_uniform_fixed_origin_tour",
    "exact_uniform_two_opt_indices",
    "exp_neg_rational_interval",
]
