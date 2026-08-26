"""Automatic finite-horizon MH kernel perturbation bounds for v19."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .pareto_v19_exact_mh import as_fraction


class V19KernelPerturbationError(ValueError):
    pass


def gamma_n(operations: int, *, unit_roundoff: Fraction = Fraction(1, 2**53)) -> Fraction:
    if not isinstance(operations, int) or operations < 0:
        raise V19KernelPerturbationError("operation count must be nonnegative")
    product = operations * unit_roundoff
    if product >= 1:
        raise V19KernelPerturbationError("standard gamma_n bound is invalid when n*u>=1")
    return product / (1 - product)


def binary64_normalized_tchebycheff_energy_error_upper(
    dimension: int,
    rho: Fraction | int | float | str,
    *,
    unit_roundoff: Fraction = Fraction(1, 2**53),
) -> Fraction:
    """Uniform absolute error bound for the frozen-box energy evaluation.

    Assumptions: all exact inputs are finite normal binary64 values; no
    underflow/overflow occurs; exact weights are positive and sum to one;
    objectives lie in the exact frozen box.  The code path performs two
    subtractions and one division for normalization, one multiplication per
    weighted coordinate, sequential nonnegative summation, one rho
    multiplication and one final addition.
    """

    if not isinstance(dimension, int) or dimension <= 0:
        raise V19KernelPerturbationError("dimension must be positive")
    rho_q = as_fraction(rho)
    if rho_q < 0:
        raise V19KernelPerturbationError("rho must be nonnegative")
    u = unit_roundoff
    # Relative factor for (fl(f-l))/fl(u-l), followed by division rounding.
    normalization_error = ((1 + u) ** 2 / (1 - u)) - 1
    product_error = normalization_error + u * (1 + normalization_error)
    sum_error = dimension * product_error + gamma_n(
        max(0, dimension - 1), unit_roundoff=u
    ) * (1 + dimension * product_error)
    max_error = product_error
    rho_term_error = rho_q * sum_error + u * rho_q * (1 + sum_error)
    exact_energy_upper = Fraction(1, 1) + rho_q
    pre_final_error = max_error + rho_term_error
    final_add_error = u * (exact_energy_upper + pre_final_error)
    return pre_final_error + final_add_error


def mh_acceptance_probability_error_upper(
    beta: Fraction | int | float | str,
    energy_absolute_error_upper: Fraction | int | float | str,
    *,
    energy_span_upper: Fraction | int | float | str = 1,
    unit_roundoff: Fraction = Fraction(1, 2**53),
    transcendental_comparison_error_upper: Fraction | int | float | str = 0,
) -> tuple[Fraction, Fraction, Fraction]:
    """Bound the implemented acceptance-probability error.

    The exact energy at each endpoint lies in an interval of width at most
    ``energy_span_upper`` and the computed endpoint energy has absolute error
    at most ``energy_absolute_error_upper``.  Besides those endpoint errors,
    the actual binary64 path rounds the energy subtraction and the subsequent
    multiplication by ``beta``.  These two arithmetic terms must be included
    before applying the 1-Lipschitz bound for

    ``x -> min(1, exp(-x))``.

    The final argument is reserved for the independently certified difference
    between the ideal real comparison and the concrete random/log/libm
    comparison.  The function returns

    ``(difference_error, exponent_error, acceptance_error)``.
    """

    beta_q = as_fraction(beta)
    energy_error = as_fraction(energy_absolute_error_upper)
    span = as_fraction(energy_span_upper)
    function_error = as_fraction(transcendental_comparison_error_upper)
    if beta_q < 0 or energy_error < 0 or span < 0 or function_error < 0:
        raise V19KernelPerturbationError("error inputs must be nonnegative")
    u = unit_roundoff
    if u <= 0 or u >= 1:
        raise V19KernelPerturbationError("unit_roundoff must lie in (0,1)")

    # fl(Uhat_y-Uhat_x) versus the exact energy difference.
    difference_error = 2 * energy_error + u * (span + 2 * energy_error)
    # Magnitude of the rounded difference, used in the beta multiplication.
    rounded_difference_magnitude = (1 + u) * (span + 2 * energy_error)
    exponent_error = (
        beta_q * difference_error
        + u * beta_q * rounded_difference_magnitude
    )
    acceptance_error = min(
        Fraction(1, 1),
        exponent_error + function_error,
    )
    return difference_error, exponent_error, acceptance_error


def mh_kernel_row_tv_upper(
    proposal_row_tv_upper: Fraction | int | float | str,
    acceptance_error_upper: Fraction | int | float | str,
) -> Fraction:
    """Coupling bound for proposal plus acceptance perturbation."""

    proposal = as_fraction(proposal_row_tv_upper)
    acceptance = as_fraction(acceptance_error_upper)
    if proposal < 0 or proposal > 1 or acceptance < 0 or acceptance > 1:
        raise V19KernelPerturbationError("TV and acceptance errors must lie in [0,1]")
    return min(Fraction(1, 1), proposal + acceptance)


def contracted_product_tv_upper(
    per_step_tv_upper: Sequence[Fraction | int | float | str],
    ideal_tail_dobrushin_upper: Sequence[Fraction | int | float | str] | None = None,
) -> Fraction:
    """Duhamel bound, optionally contracted by the ideal kernel tail.

    If ``beta_t`` bounds the Dobrushin coefficient of ideal step ``t``, the
    perturbation at step ``t`` is multiplied by ``prod_{u>t} beta_u``.
    """

    kappas = tuple(as_fraction(x) for x in per_step_tv_upper)
    if any(x < 0 or x > 1 for x in kappas):
        raise V19KernelPerturbationError("per-step TV bounds must lie in [0,1]")
    if ideal_tail_dobrushin_upper is None:
        return min(Fraction(1, 1), sum(kappas, Fraction(0, 1)))
    betas = tuple(as_fraction(x) for x in ideal_tail_dobrushin_upper)
    if len(betas) != len(kappas) or any(x < 0 or x > 1 for x in betas):
        raise V19KernelPerturbationError("one valid Dobrushin bound is required per step")
    total = Fraction(0, 1)
    tail = Fraction(1, 1)
    for kappa, beta in zip(reversed(kappas), reversed(betas), strict=True):
        total += kappa * tail
        tail *= beta
    return min(Fraction(1, 1), total)


@dataclass(frozen=True)
class AutomaticKernelTVCertificate:
    dimension: int
    rho: Fraction
    beta: Fraction
    energy_span_upper: Fraction
    energy_error_upper: Fraction
    energy_difference_error_upper: Fraction
    exponent_arithmetic_error_upper: Fraction
    acceptance_comparison_error_upper: Fraction
    acceptance_error_upper: Fraction
    proposal_row_tv_upper: Fraction
    per_step_kernel_tv_upper: Fraction
    horizon_tv_upper: Fraction
    ideal_tail_dobrushin_upper: tuple[Fraction, ...]
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "rho": str(self.rho),
            "beta": str(self.beta),
            "energy_span_upper": str(self.energy_span_upper),
            "energy_error_upper": str(self.energy_error_upper),
            "energy_difference_error_upper": str(
                self.energy_difference_error_upper
            ),
            "exponent_arithmetic_error_upper": str(
                self.exponent_arithmetic_error_upper
            ),
            "acceptance_comparison_error_upper": str(
                self.acceptance_comparison_error_upper
            ),
            "acceptance_error_upper": str(self.acceptance_error_upper),
            "proposal_row_tv_upper": str(self.proposal_row_tv_upper),
            "per_step_kernel_tv_upper": str(self.per_step_kernel_tv_upper),
            "horizon_tv_upper": str(self.horizon_tv_upper),
            "ideal_tail_dobrushin_upper": [str(x) for x in self.ideal_tail_dobrushin_upper],
            "assumptions": list(self.assumptions),
        }


def build_automatic_kernel_tv_certificate(
    *,
    dimension: int,
    rho: Fraction | int | float | str,
    beta: Fraction | int | float | str,
    steps: int,
    proposal_row_tv_upper: Fraction | int | float | str = 0,
    transcendental_comparison_error_upper: Fraction | int | float | str = 0,
    ideal_step_dobrushin_upper: Fraction | int | float | str = 1,
) -> AutomaticKernelTVCertificate:
    if not isinstance(steps, int) or steps < 0:
        raise V19KernelPerturbationError("steps must be a nonnegative integer")
    rho_q = as_fraction(rho)
    beta_q = as_fraction(beta)
    proposal_q = as_fraction(proposal_row_tv_upper)
    comparison_error = as_fraction(transcendental_comparison_error_upper)
    energy_span = Fraction(1, 1) + rho_q
    energy_error = binary64_normalized_tchebycheff_energy_error_upper(dimension, rho_q)
    difference_error, exponent_error, acceptance_error = (
        mh_acceptance_probability_error_upper(
        beta_q,
        energy_error,
        energy_span_upper=energy_span,
        transcendental_comparison_error_upper=transcendental_comparison_error_upper,
        )
    )
    per_step = mh_kernel_row_tv_upper(proposal_q, acceptance_error)
    beta_step = as_fraction(ideal_step_dobrushin_upper)
    horizon = contracted_product_tv_upper(
        (per_step,) * steps,
        (beta_step,) * steps,
    )
    return AutomaticKernelTVCertificate(
        dimension=dimension,
        rho=rho_q,
        beta=beta_q,
        energy_span_upper=energy_span,
        energy_error_upper=energy_error,
        energy_difference_error_upper=difference_error,
        exponent_arithmetic_error_upper=exponent_error,
        acceptance_comparison_error_upper=comparison_error,
        acceptance_error_upper=acceptance_error,
        proposal_row_tv_upper=proposal_q,
        per_step_kernel_tv_upper=per_step,
        horizon_tv_upper=horizon,
        ideal_tail_dobrushin_upper=(beta_step,) * steps,
        assumptions=(
            "finite normal binary64 inputs",
            "every rounded intermediate used by the error model is normal or exact; no overflow or underflow",
            "exact positive weights summing to one",
            "objectives remain in the exact frozen box",
            "declared acceptance-comparison error covers random-grid, log/libm and comparison effects not included in the arithmetic bound",
        ),
    )


__all__ = [
    "AutomaticKernelTVCertificate",
    "V19KernelPerturbationError",
    "binary64_normalized_tchebycheff_energy_error_upper",
    "build_automatic_kernel_tv_certificate",
    "contracted_product_tv_upper",
    "gamma_n",
    "mh_acceptance_probability_error_upper",
    "mh_kernel_row_tv_upper",
]
