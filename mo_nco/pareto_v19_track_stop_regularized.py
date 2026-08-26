"""Regularized and rate-certified shared-categorical Track-and-Stop (v19).

The v17 asymptotic upper theorem assumed a unique characteristic allocation and
an unspecified summable tail argument.  This module makes both interfaces
explicit.

* Entropy regularization makes the allocation optimizer unique.  If ``g`` is
  the characteristic information game and ``w_lambda`` maximizes
  ``g(w)+lambda*H(w)``, then

      Gamma* - g(w_lambda) <= lambda * log(R).

  A tangent supergradient bracket adds a certified numerical optimization gap.

* Expected stopping-time claims require a quantitative information-rate
  certificate.  The code accepts an explicit deterministic information deficit
  and a rational geometric tail envelope.  It computes a finite expected-time
  upper bound without silently converting almost-sure convergence into an
  expectation statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Sequence

from .pareto_v17_track_and_stop import (
    TrackAndStopError,
    _validate_probability_matrix,
    characteristic_value,
)


class RegularizedTrackStopError(ValueError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def smooth_categorical_matrix(
    probabilities: Sequence[Sequence[float]],
    smoothing: float,
) -> tuple[tuple[float, ...], ...]:
    p = _validate_probability_matrix(probabilities)
    zeta = float(smoothing)
    if not math.isfinite(zeta) or zeta < 0.0 or zeta >= 1.0:
        raise RegularizedTrackStopError("smoothing must lie in [0,1)")
    k = len(p[0])
    return tuple(
        tuple((1.0 - zeta) * x + zeta / k for x in row)
        for row in p
    )


def binary_entropy(value: float) -> float:
    x = float(value)
    if x < 0.0 or x > 1.0 or not math.isfinite(x):
        raise RegularizedTrackStopError("binary entropy input must lie in [0,1]")
    if x == 0.0 or x == 1.0:
        return 0.0
    return -x * math.log(x) - (1.0 - x) * math.log(1.0 - x)


def binary_entropy_continuity_modulus(total_variation_upper: float) -> float:
    """Conservative binary-entropy continuity modulus.

    For Bernoulli laws at total-variation distance at most ``delta``, the
    standard sharp modulus is ``h_2(delta)`` only on ``delta<=1/2``.  Beyond
    that range the safe dimension-two bound is ``log 2``.  Using
    ``h_2(delta)`` directly for ``delta>1/2`` would become *smaller* as delta
    approaches one and is therefore anti-conservative.
    """

    delta = float(total_variation_upper)
    if not math.isfinite(delta) or delta < 0.0 or delta > 1.0:
        raise RegularizedTrackStopError(
            "total-variation upper bound must lie in [0,1]"
        )
    return binary_entropy(min(delta, 0.5))


def entropy(weights: Sequence[float]) -> float:
    total = 0.0
    for value in weights:
        if value < 0.0:
            raise RegularizedTrackStopError("weights must be nonnegative")
        if value > 0.0:
            total -= value * math.log(value)
    return total


@dataclass(frozen=True)
class EntropicCharacteristicCertificate:
    weights: tuple[float, ...]
    characteristic_lower: float
    smoothed_characteristic_lower: float
    regularized_objective_lower: float
    regularized_objective_upper: float
    numerical_regularized_gap: float
    regularization_bias_upper: float
    smoothing_bias_upper: float
    total_characteristic_gap_upper: float
    regularization: float
    smoothing: float
    iterations: int
    semantics: str = "entropy_regularized_characteristic_tangent_bracket_v19"

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "machine_verified_optimization_bound": False,
            "weights": list(self.weights),
            "characteristic_lower": self.characteristic_lower,
            "smoothed_characteristic_lower": self.smoothed_characteristic_lower,
            "regularized_objective_lower": self.regularized_objective_lower,
            "regularized_objective_upper": self.regularized_objective_upper,
            "numerical_regularized_gap": self.numerical_regularized_gap,
            "regularization_bias_upper": self.regularization_bias_upper,
            "smoothing_bias_upper": self.smoothing_bias_upper,
            "total_characteristic_gap_upper": self.total_characteristic_gap_upper,
            "regularization": self.regularization,
            "smoothing": self.smoothing,
            "iterations": self.iterations,
        }


def solve_entropic_characteristic_game(
    probabilities: Sequence[Sequence[float]],
    *,
    regularization: float,
    smoothing: float = 0.0,
    iterations: int = 30_000,
    step_scale: float = 0.25,
    minimum_weight: float = 1e-15,
) -> EntropicCharacteristicCertificate:
    raw_p = _validate_probability_matrix(probabilities)
    p = smooth_categorical_matrix(raw_p, smoothing)
    lam = float(regularization)
    if not math.isfinite(lam) or lam <= 0.0:
        raise RegularizedTrackStopError("regularization must be positive")
    if iterations <= 0 or step_scale <= 0.0:
        raise RegularizedTrackStopError("iterations and step_scale must be positive")
    r_count = len(p)
    if smoothing == 0.0 and any(x <= 0.0 for row in p for x in row):
        raise RegularizedTrackStopError(
            "zero-support models require positive smoothing or the epsilon-PAC branch"
        )
    if not (0.0 < minimum_weight < 1.0 / r_count):
        raise RegularizedTrackStopError("minimum_weight must be in (0,1/R)")

    weights = [1.0 / r_count] * r_count
    best_weights = tuple(weights)
    best_f = -math.inf
    best_g = -math.inf
    best_upper = math.inf

    for t in range(1, iterations + 1):
        g_value, _active, g_super = characteristic_value(p, weights)
        h_value = entropy(weights)
        f_value = g_value + lam * h_value
        combined = [
            g_super[i] + lam * (-(math.log(weights[i]) + 1.0))
            for i in range(r_count)
        ]
        tangent_upper = f_value + max(combined) - sum(
            combined[i] * weights[i] for i in range(r_count)
        )
        if f_value > best_f:
            best_f = f_value
            best_g = g_value
            best_weights = tuple(weights)
        best_upper = min(best_upper, tangent_upper)

        step = step_scale / math.sqrt(t)
        shift = max(step * value for value in combined)
        raw = [
            max(minimum_weight, weights[i] * math.exp(step * combined[i] - shift))
            for i in range(r_count)
        ]
        total = sum(raw)
        weights = [value / total for value in raw]

    if not (math.isfinite(best_f) and math.isfinite(best_upper) and best_upper >= best_f - 1e-10):
        raise RegularizedTrackStopError("failed to produce a finite regularized bracket")
    numerical_gap = max(0.0, best_upper - best_f)
    bias = lam * math.log(r_count)
    smoothing_bias = 4.0 * binary_entropy_continuity_modulus(float(smoothing))
    raw_g, _raw_active, _raw_super = characteristic_value(raw_p, best_weights)
    total_gap = max(0.0, best_upper - best_g)
    # The explicit theorem also gives bias+optimization gap.  Floating noise can
    # make the direct tangent expression smaller by a few ulps; use the larger
    # conservative value.
    total_gap = max(total_gap, bias + numerical_gap + smoothing_bias)
    return EntropicCharacteristicCertificate(
        weights=best_weights,
        characteristic_lower=raw_g,
        smoothed_characteristic_lower=best_g,
        regularized_objective_lower=best_f,
        regularized_objective_upper=best_upper,
        numerical_regularized_gap=numerical_gap,
        regularization_bias_upper=bias,
        smoothing_bias_upper=smoothing_bias,
        total_characteristic_gap_upper=total_gap,
        regularization=lam,
        smoothing=float(smoothing),
        iterations=iterations,
    )


def _as_fraction(value: Fraction | int | float | str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise RegularizedTrackStopError("boolean is not a rational scalar")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegularizedTrackStopError("non-finite scalar")
        return Fraction.from_float(value)
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise RegularizedTrackStopError(f"invalid rational scalar: {value!r}") from exc


def ceil_log2(value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise RegularizedTrackStopError("ceil_log2 requires a positive integer")
    return 0 if value == 1 else (value - 1).bit_length()


@dataclass(frozen=True)
class InformationRateCertificate:
    gamma_lower: Fraction
    deficit_scale: Fraction
    deficit_power: int
    threshold_log_inv_delta_upper: Fraction
    threshold_log2_time_coefficient: Fraction
    threshold_constant: Fraction
    tail_prefactor: Fraction
    tail_ratio: Fraction
    proof_sha256: str

    def __post_init__(self) -> None:
        if self.gamma_lower <= 0:
            raise RegularizedTrackStopError("gamma_lower must be positive")
        if self.deficit_scale < 0 or self.deficit_power <= 0:
            raise RegularizedTrackStopError("invalid information deficit envelope")
        if any(
            x < 0
            for x in (
                self.threshold_log_inv_delta_upper,
                self.threshold_log2_time_coefficient,
                self.threshold_constant,
                self.tail_prefactor,
            )
        ):
            raise RegularizedTrackStopError("threshold and tail constants must be nonnegative")
        if not (Fraction(0, 1) < self.tail_ratio < Fraction(1, 1)):
            raise RegularizedTrackStopError("tail_ratio must lie in (0,1)")
        if _HEX64.fullmatch(self.proof_sha256) is None:
            raise RegularizedTrackStopError("rate certificate needs a proof SHA-256")

    def deficit_upper(self, time: int) -> Fraction:
        if time <= 0:
            raise RegularizedTrackStopError("time must be positive")
        return self.deficit_scale / Fraction((time + 1) ** self.deficit_power, 1)

    def threshold_upper(self, time: int) -> Fraction:
        if time <= 0:
            raise RegularizedTrackStopError("time must be positive")
        return (
            self.threshold_log_inv_delta_upper
            + self.threshold_log2_time_coefficient * ceil_log2(time + 1)
            + self.threshold_constant
        )

    def bad_tail_sum_upper(self, start_time: int) -> Fraction:
        if start_time <= 0:
            raise RegularizedTrackStopError("start_time must be positive")
        return self.tail_prefactor * self.tail_ratio**start_time / (1 - self.tail_ratio)


@dataclass(frozen=True)
class ExpectedStoppingCertificate:
    deterministic_crossing_time: int
    persistent_crossing_from_time: int
    expected_stopping_time_upper: Fraction
    normalized_expected_upper: Fraction
    gamma_lower: Fraction
    threshold_at_crossing: Fraction
    information_lower_at_crossing: Fraction
    bad_tail_sum_upper: Fraction
    proof_sha256: str
    semantics: str = "quantitative_information_rate_to_expected_stopping_v19"

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "deterministic_crossing_time": self.deterministic_crossing_time,
            "persistent_crossing_from_time": self.persistent_crossing_from_time,
            "persistent_crossing_verified_by_linear_block_envelope": True,
            "expected_stopping_time_upper": str(self.expected_stopping_time_upper),
            "normalized_expected_upper": str(self.normalized_expected_upper),
            "gamma_lower": str(self.gamma_lower),
            "threshold_at_crossing": str(self.threshold_at_crossing),
            "information_lower_at_crossing": str(self.information_lower_at_crossing),
            "bad_tail_sum_upper": str(self.bad_tail_sum_upper),
            "proof_sha256": self.proof_sha256,
        }


def build_expected_stopping_certificate(
    rate: InformationRateCertificate,
    *,
    max_time: int = 100_000_000,
) -> ExpectedStoppingCertificate:
    """Convert a quantitative information-rate tail into an expectation bound.

    The rate artifact asserts

    ``P(exists s>=t: Z_s/s < gamma-r_s) <= C*rho**t``

    and the stopping threshold is bounded by

    ``B_delta + a*ceil(log2(s+1)) + b``.

    A one-time crossing is insufficient because the logarithmic threshold can
    jump later.  The implementation therefore finds a power-of-two block start
    ``n`` such that the *persistent* lower envelope

    ``s*gamma - deficit_scale``

    dominates the threshold for every ``s>=n``.  At block starts the margin
    changes by ``gamma*n-a``; after ``gamma*n>=a`` it cannot decrease, and
    inside a block the threshold is constant while the linear envelope grows.
    This closes the local-crossing gap in the earlier v19 draft.
    """

    if max_time <= 0:
        raise RegularizedTrackStopError("max_time must be positive")
    gamma = rate.gamma_lower
    scale = rate.deficit_scale
    log_coeff = rate.threshold_log2_time_coefficient

    # Power-of-two block start.  First ensure future block-start margins are
    # nondecreasing: gamma*n >= log_coeff.
    n = 1
    while gamma * n < log_coeff:
        n *= 2
        if n > max_time:
            raise RegularizedTrackStopError(
                "no persistent information-threshold regime within max_time"
            )

    while n <= max_time:
        linear_information = gamma * n - scale
        threshold = rate.threshold_upper(n)
        if linear_information >= threshold:
            actual_information = n * (gamma - rate.deficit_upper(n))
            if actual_information < threshold:
                raise AssertionError("linear persistent envelope exceeded the actual lower bound")
            tail = rate.bad_tail_sum_upper(n)
            expected = Fraction(n, 1) + tail
            normalizer = max(rate.threshold_log_inv_delta_upper, Fraction(1, 1))
            return ExpectedStoppingCertificate(
                deterministic_crossing_time=n,
                persistent_crossing_from_time=n,
                expected_stopping_time_upper=expected,
                normalized_expected_upper=expected / normalizer,
                gamma_lower=gamma,
                threshold_at_crossing=threshold,
                information_lower_at_crossing=actual_information,
                bad_tail_sum_upper=tail,
                proof_sha256=rate.proof_sha256,
            )
        n *= 2
    raise RegularizedTrackStopError(
        "no persistent certified information-threshold crossing within max_time"
    )



@dataclass(frozen=True)
class EntropicCostCharacteristicCertificate:
    cost_shares: tuple[float, ...]
    pull_proportions: tuple[float, ...]
    expected_cost_per_pull: float
    cost_information_lower: float
    smoothed_cost_information_lower: float
    regularized_objective_lower: float
    regularized_objective_upper: float
    total_cost_characteristic_gap_upper: float
    regularization_bias_upper: float
    smoothing_bias_upper: float
    costs: tuple[float, ...]
    regularization: float
    smoothing: float
    iterations: int
    semantics: str = "cost_share_entropy_regularized_characteristic_v19"

    @property
    def cost_characteristic_time_upper(self) -> float:
        return math.inf if self.cost_information_lower <= 0 else 1.0 / self.cost_information_lower

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": self.semantics,
            "machine_verified_optimization_bound": False,
            "cost_shares": list(self.cost_shares),
            "pull_proportions": list(self.pull_proportions),
            "expected_cost_per_pull": self.expected_cost_per_pull,
            "cost_information_lower": self.cost_information_lower,
            "smoothed_cost_information_lower": self.smoothed_cost_information_lower,
            "regularized_objective_lower": self.regularized_objective_lower,
            "regularized_objective_upper": self.regularized_objective_upper,
            "total_cost_characteristic_gap_upper": self.total_cost_characteristic_gap_upper,
            "regularization_bias_upper": self.regularization_bias_upper,
            "smoothing_bias_upper": self.smoothing_bias_upper,
            "costs": list(self.costs),
            "regularization": self.regularization,
            "smoothing": self.smoothing,
            "iterations": self.iterations,
            "cost_characteristic_time_upper": self.cost_characteristic_time_upper,
        }


def _cost_pair_information(
    probabilities: tuple[tuple[float, ...], ...],
    cost_shares: Sequence[float],
    costs: Sequence[float],
    cell_index: int,
    challenger: int,
) -> tuple[float, tuple[float, ...]]:
    # The information accumulated per unit expected cost has effective arm
    # weights lambda_r/c_r.
    from .pareto_v17_track_and_stop import answer_map, bernoulli_kl

    answers = answer_map(probabilities)
    best = answers[cell_index]
    if challenger == best:
        raise RegularizedTrackStopError("challenger equals the best type")
    category = cell_index + 1
    va = cost_shares[best] / costs[best]
    vs = cost_shares[challenger] / costs[challenger]
    total = va + vs
    pa = probabilities[best][category]
    ps = probabilities[challenger][category]
    if total <= 0.0:
        m = 0.5 * (pa + ps)
    else:
        m = (va * pa + vs * ps) / total
    da = bernoulli_kl(pa, m)
    ds = bernoulli_kl(ps, m)
    value = va * da + vs * ds
    gradient = [0.0] * len(probabilities)
    gradient[best] = da / costs[best]
    gradient[challenger] = ds / costs[challenger]
    return value, tuple(gradient)


def cost_characteristic_value(
    probabilities: Sequence[Sequence[float]],
    cost_shares: Sequence[float],
    costs: Sequence[float],
) -> tuple[float, tuple[int, int], tuple[float, ...]]:
    from .pareto_v17_track_and_stop import answer_map

    p = _validate_probability_matrix(probabilities)
    lam = tuple(float(x) for x in cost_shares)
    c = tuple(float(x) for x in costs)
    if len(lam) != len(p) or len(c) != len(p):
        raise RegularizedTrackStopError("cost-share and cost dimensions must match the types")
    if any(x < 0 or not math.isfinite(x) for x in lam) or abs(sum(lam) - 1.0) > 1e-8:
        raise RegularizedTrackStopError("cost shares must lie in the simplex")
    if any(x <= 0 or not math.isfinite(x) for x in c):
        raise RegularizedTrackStopError("arm costs must be finite and positive")
    answers = answer_map(p)
    active_value = math.inf
    active_pair = (-1, -1)
    active_gradient: tuple[float, ...] | None = None
    for j, best in enumerate(answers):
        for challenger in range(len(p)):
            if challenger == best:
                continue
            value, gradient = _cost_pair_information(p, lam, c, j, challenger)
            key = (j, challenger)
            if value < active_value - 1e-15 or (
                abs(value - active_value) <= 1e-15 and key < active_pair
            ):
                active_value = value
                active_pair = key
                active_gradient = gradient
    if active_gradient is None:
        raise RegularizedTrackStopError("no cost-aware answer-changing alternative exists")
    return active_value, active_pair, active_gradient


def solve_cost_aware_entropic_characteristic_game(
    probabilities: Sequence[Sequence[float]],
    costs: Sequence[float],
    *,
    regularization: float,
    smoothing: float = 0.0,
    iterations: int = 30_000,
    step_scale: float = 0.25,
    minimum_share: float = 1e-15,
) -> EntropicCostCharacteristicCertificate:
    raw_p = _validate_probability_matrix(probabilities)
    p = smooth_categorical_matrix(raw_p, smoothing)
    c = tuple(float(x) for x in costs)
    if len(c) != len(p) or any(x <= 0 or not math.isfinite(x) for x in c):
        raise RegularizedTrackStopError("one finite positive cost is required per type")
    lam_reg = float(regularization)
    if lam_reg <= 0 or not math.isfinite(lam_reg):
        raise RegularizedTrackStopError("regularization must be positive")
    if iterations <= 0 or step_scale <= 0.0:
        raise RegularizedTrackStopError("iterations and step_scale must be positive")
    r_count = len(p)
    if smoothing == 0.0 and any(x <= 0.0 for row in p for x in row):
        raise RegularizedTrackStopError("zero-support cost-aware models require smoothing")
    if not (0.0 < minimum_share < 1.0 / r_count):
        raise RegularizedTrackStopError("minimum_share must be in (0,1/R)")
    shares = [1.0 / r_count] * r_count
    best_shares = tuple(shares)
    best_f = -math.inf
    best_g = -math.inf
    best_upper = math.inf
    for t in range(1, iterations + 1):
        g_value, _active, g_super = cost_characteristic_value(p, shares, c)
        f_value = g_value + lam_reg * entropy(shares)
        combined = [
            g_super[i] + lam_reg * (-(math.log(shares[i]) + 1.0))
            for i in range(r_count)
        ]
        tangent_upper = f_value + max(combined) - sum(
            combined[i] * shares[i] for i in range(r_count)
        )
        if f_value > best_f:
            best_f = f_value
            best_g = g_value
            best_shares = tuple(shares)
        best_upper = min(best_upper, tangent_upper)
        step = step_scale / math.sqrt(t)
        shift = max(step * x for x in combined)
        raw = [
            max(minimum_share, shares[i] * math.exp(step * combined[i] - shift))
            for i in range(r_count)
        ]
        total = sum(raw)
        shares = [x / total for x in raw]
    if not (
        math.isfinite(best_f)
        and math.isfinite(best_upper)
        and best_upper >= best_f - 1e-10
    ):
        raise RegularizedTrackStopError(
            "failed to produce a finite cost-aware regularized bracket"
        )
    numerical_gap = max(0.0, best_upper - best_f)
    entropy_bias = lam_reg * math.log(r_count)
    smoothing_bias = (
        4.0 * binary_entropy_continuity_modulus(float(smoothing)) / min(c)
    )
    total_gap = max(0.0, best_upper - best_g, numerical_gap + entropy_bias + smoothing_bias)
    # Convert cost shares lambda to pull proportions w proportional to lambda/c.
    pull_raw = tuple(best_shares[i] / c[i] for i in range(r_count))
    pull_total = sum(pull_raw)
    pull = tuple(x / pull_total for x in pull_raw)
    expected_cost = sum(pull[i] * c[i] for i in range(r_count))
    raw_g, _raw_active, _raw_super = cost_characteristic_value(
        raw_p,
        best_shares,
        c,
    )
    return EntropicCostCharacteristicCertificate(
        cost_shares=best_shares,
        pull_proportions=pull,
        expected_cost_per_pull=expected_cost,
        cost_information_lower=raw_g,
        smoothed_cost_information_lower=best_g,
        regularized_objective_lower=best_f,
        regularized_objective_upper=best_upper,
        total_cost_characteristic_gap_upper=total_gap,
        regularization_bias_upper=entropy_bias,
        smoothing_bias_upper=smoothing_bias,
        costs=c,
        regularization=lam_reg,
        smoothing=float(smoothing),
        iterations=iterations,
    )


__all__ = [
    "EntropicCharacteristicCertificate",
    "EntropicCostCharacteristicCertificate",
    "ExpectedStoppingCertificate",
    "InformationRateCertificate",
    "RegularizedTrackStopError",
    "binary_entropy",
    "binary_entropy_continuity_modulus",
    "build_expected_stopping_certificate",
    "ceil_log2",
    "cost_characteristic_value",
    "entropy",
    "smooth_categorical_matrix",
    "solve_cost_aware_entropic_characteristic_game",
    "solve_entropic_characteristic_game",
]
