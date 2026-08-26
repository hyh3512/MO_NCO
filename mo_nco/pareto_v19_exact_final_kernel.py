"""Executable exact-rational final-target MH kernel for Pareto-SMC v19.

This module closes the gap between isolated exact proposal/acceptance helpers
and a complete *final-target* transition.  It does not implement the full SMC
weighting/resampling pipeline.  The mathematical claim is restricted to a
frozen finite state space, an exact rational objective evaluator, and an ideal
independent random-bit source.

Both proposal components are symmetric on fixed-origin tours:

* uniform segment-reversal 2-opt, and
* uniform independence refresh over all fixed-origin tours.

A state-independent exact rational mixture of those proposals followed by the
exact lazy-random-bit Metropolis decision is therefore reversible for the
frozen target proportional to ``exp(-beta * U(tour))`` with respect to the
uniform fixed-origin base measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Sequence

from .pareto_v19_exact_mh import (
    ExactBernoulliDecision,
    ExactMHError,
    ExactMetropolisDecision,
    ExactRandBelowDecision,
    RandomBitSource,
    as_fraction,
    exact_augmented_tchebycheff_energy,
    exact_metropolis_accept,
    exact_uniform_fixed_origin_tour,
    exact_uniform_two_opt_indices,
)
from .moves import two_opt_at
from .types import Tour


class ExactFinalKernelError(ExactMHError):
    pass


@dataclass(frozen=True)
class ExactRationalBernoulliDecision:
    accepted: bool
    probability: Fraction
    random_prefix_numerator: int
    random_prefix_bits: int
    rounds: int
    refinement_cap: int | None
    semantics: str = "ideal_lazy_uniform_bits_vs_exact_rational_probability_v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "probability": str(self.probability),
            "random_prefix_numerator": self.random_prefix_numerator,
            "random_prefix_bits": self.random_prefix_bits,
            "rounds": self.rounds,
            "refinement_cap": self.refinement_cap,
            "semantics": self.semantics,
        }


def exact_bernoulli_rational(
    probability: Fraction | int | float | str,
    bit_source: RandomBitSource,
    *,
    bits_per_round: int = 16,
    max_rounds: int | None = None,
) -> ExactRationalBernoulliDecision:
    """Exact Bernoulli for a rational probability under ideal random bits."""

    p = as_fraction(probability)
    if p < 0 or p > 1:
        raise ExactFinalKernelError("probability must lie in [0,1]")
    if bits_per_round <= 0:
        raise ExactFinalKernelError("bits_per_round must be positive")
    if max_rounds is not None and (
        not isinstance(max_rounds, int) or max_rounds <= 0
    ):
        raise ExactFinalKernelError("max_rounds must be None or positive")
    if p == 0:
        return ExactRationalBernoulliDecision(False, p, 0, 0, 0, max_rounds)
    if p == 1:
        return ExactRationalBernoulliDecision(True, p, 0, 0, 0, max_rounds)

    prefix = 0
    prefix_bits = 0
    rounds = 0
    while max_rounds is None or rounds < max_rounds:
        rounds += 1
        chunk = int(bit_source.getrandbits(bits_per_round))
        if chunk < 0 or chunk >= 1 << bits_per_round:
            raise ExactFinalKernelError("bit source returned an invalid chunk")
        prefix = (prefix << bits_per_round) | chunk
        prefix_bits += bits_per_round
        denominator = 1 << prefix_bits
        lower = Fraction(prefix, denominator)
        upper = Fraction(prefix + 1, denominator)
        if upper <= p:
            return ExactRationalBernoulliDecision(
                True, p, prefix, prefix_bits, rounds, max_rounds
            )
        if lower >= p:
            return ExactRationalBernoulliDecision(
                False, p, prefix, prefix_bits, rounds, max_rounds
            )
    raise ExactFinalKernelError(
        "exact rational Bernoulli refinement cap reached"
    )


@dataclass(frozen=True)
class ExactFinalKernelConfig:
    num_cities: int
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]
    weights: tuple[Fraction, ...]
    rho: Fraction
    beta: Fraction
    global_refresh_probability: Fraction
    semantics: str = "exact_rational_frozen_final_target_mh_kernel_v19"

    def __post_init__(self) -> None:
        if self.num_cities < 4:
            raise ExactFinalKernelError("exact final kernel requires at least four cities")
        if not self.lower or not (
            len(self.lower) == len(self.upper) == len(self.weights)
        ):
            raise ExactFinalKernelError("box and weight dimensions must agree")
        if any(b <= a for a, b in zip(self.lower, self.upper, strict=True)):
            raise ExactFinalKernelError("objective-box widths must be positive")
        if any(w <= 0 for w in self.weights) or sum(
            self.weights, Fraction(0, 1)
        ) != 1:
            raise ExactFinalKernelError("weights must be positive and sum exactly to one")
        if self.rho < 0 or self.beta < 0:
            raise ExactFinalKernelError("rho and beta must be nonnegative")
        if not (Fraction(0, 1) <= self.global_refresh_probability <= 1):
            raise ExactFinalKernelError("global refresh probability must lie in [0,1]")

    @classmethod
    def from_scalars(
        cls,
        *,
        num_cities: int,
        lower: Sequence[Fraction | int | float | str],
        upper: Sequence[Fraction | int | float | str],
        weights: Sequence[Fraction | int | float | str],
        rho: Fraction | int | float | str,
        beta: Fraction | int | float | str,
        global_refresh_probability: Fraction | int | float | str,
    ) -> "ExactFinalKernelConfig":
        return cls(
            num_cities=int(num_cities),
            lower=tuple(as_fraction(x) for x in lower),
            upper=tuple(as_fraction(x) for x in upper),
            weights=tuple(as_fraction(x) for x in weights),
            rho=as_fraction(rho),
            beta=as_fraction(beta),
            global_refresh_probability=as_fraction(global_refresh_probability),
        )


@dataclass(frozen=True)
class ExactFinalKernelStep:
    current_tour: Tour
    proposed_tour: Tour
    next_tour: Tour
    current_objective: tuple[Fraction, ...]
    proposed_objective: tuple[Fraction, ...]
    current_energy: Fraction
    proposed_energy: Fraction
    proposal_kind: str
    mixture_decision: ExactRationalBernoulliDecision
    proposal_draws: tuple[ExactRandBelowDecision, ...]
    metropolis_decision: ExactMetropolisDecision
    accepted: bool
    semantics: str = "replayed_exact_rational_final_target_step_v19"

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "current_tour": list(self.current_tour),
            "proposed_tour": list(self.proposed_tour),
            "next_tour": list(self.next_tour),
            "current_objective": [str(x) for x in self.current_objective],
            "proposed_objective": [str(x) for x in self.proposed_objective],
            "current_energy": str(self.current_energy),
            "proposed_energy": str(self.proposed_energy),
            "proposal_kind": self.proposal_kind,
            "mixture_decision": self.mixture_decision.to_dict(),
            "proposal_draws": [draw.to_dict() for draw in self.proposal_draws],
            "metropolis_decision": self.metropolis_decision.to_dict(),
            "accepted": self.accepted,
        }


def _validate_tour(tour: Tour, num_cities: int) -> Tour:
    parsed = tuple(int(x) for x in tour)
    if len(parsed) != num_cities or parsed[0] != 0 or set(parsed) != set(
        range(num_cities)
    ):
        raise ExactFinalKernelError("tour is not a fixed-origin permutation")
    return parsed


def _evaluate_exact(
    evaluator: Callable[[Tour], Sequence[Fraction | int | float | str]],
    tour: Tour,
    dimension: int,
) -> tuple[Fraction, ...]:
    objective = tuple(as_fraction(x) for x in evaluator(tour))
    if len(objective) != dimension:
        raise ExactFinalKernelError("exact objective evaluator dimension mismatch")
    return objective


def exact_final_kernel_step(
    config: ExactFinalKernelConfig,
    current_tour: Tour,
    exact_objective_evaluator: Callable[
        [Tour], Sequence[Fraction | int | float | str]
    ],
    bit_source: RandomBitSource,
    *,
    current_objective: Sequence[Fraction | int | float | str] | None = None,
    bits_per_round: int = 16,
    refinement_cap: int | None = None,
    proposal_draw_cap: int | None = None,
) -> ExactFinalKernelStep:
    """Execute one exact frozen-target MH transition.

    A finite ``refinement_cap`` or ``proposal_draw_cap`` makes the executable
    procedure partial and fail-closed.  The exact Markov-kernel theorem refers
    to the uncapped, almost-surely terminating ideal random-bit procedure.
    """

    current = _validate_tour(current_tour, config.num_cities)
    if current_objective is None:
        current_obj = _evaluate_exact(
            exact_objective_evaluator,
            current,
            len(config.lower),
        )
    else:
        current_obj = tuple(as_fraction(x) for x in current_objective)
        if len(current_obj) != len(config.lower):
            raise ExactFinalKernelError("current objective dimension mismatch")
        replay = _evaluate_exact(
            exact_objective_evaluator,
            current,
            len(config.lower),
        )
        if replay != current_obj:
            raise ExactFinalKernelError("cached current objective failed exact replay")

    mixture = exact_bernoulli_rational(
        config.global_refresh_probability,
        bit_source,
        bits_per_round=bits_per_round,
        max_rounds=refinement_cap,
    )
    proposal_draws: tuple[ExactRandBelowDecision, ...]
    if mixture.accepted:
        proposed, proposal_draws = exact_uniform_fixed_origin_tour(
            config.num_cities,
            bit_source,
            max_draws_per_step=proposal_draw_cap,
        )
        proposal_kind = "uniform_fixed_origin_tour_independence"
    else:
        i, j, draw = exact_uniform_two_opt_indices(
            config.num_cities,
            bit_source,
            max_draws=proposal_draw_cap,
        )
        proposed = two_opt_at(current, i, j)
        proposal_draws = (draw,)
        proposal_kind = "uniform_fixed_origin_two_opt_involution"

    proposed_obj = _evaluate_exact(
        exact_objective_evaluator,
        proposed,
        len(config.lower),
    )
    current_energy = exact_augmented_tchebycheff_energy(
        current_obj,
        config.lower,
        config.upper,
        config.weights,
        config.rho,
    )
    proposed_energy = exact_augmented_tchebycheff_energy(
        proposed_obj,
        config.lower,
        config.upper,
        config.weights,
        config.rho,
    )
    metropolis = exact_metropolis_accept(
        current_energy,
        proposed_energy,
        config.beta,
        bit_source,
        bits_per_round=bits_per_round,
        max_rounds=refinement_cap,
    )
    return ExactFinalKernelStep(
        current_tour=current,
        proposed_tour=proposed,
        next_tour=proposed if metropolis.accepted else current,
        current_objective=current_obj,
        proposed_objective=proposed_obj,
        current_energy=current_energy,
        proposed_energy=proposed_energy,
        proposal_kind=proposal_kind,
        mixture_decision=mixture,
        proposal_draws=proposal_draws,
        metropolis_decision=metropolis,
        accepted=metropolis.accepted,
    )


__all__ = [
    "ExactFinalKernelConfig",
    "ExactFinalKernelError",
    "ExactFinalKernelStep",
    "ExactRationalBernoulliDecision",
    "exact_bernoulli_rational",
    "exact_final_kernel_step",
]
