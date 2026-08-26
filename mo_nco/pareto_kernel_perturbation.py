from __future__ import annotations

"""Conditional implementation-kernel perturbation and interval decisions."""

from dataclasses import asdict, dataclass
from fractions import Fraction

from .pareto_frozen_cells import canonical_fraction_text


KERNEL_PERTURBATION_SCHEMA_V15 = "pareto_mh_kernel_perturbation_bound_v15"


class IndeterminateIntervalDecision(RuntimeError):
    """Raised when a certified interval straddles a decision boundary."""


def _fraction(value: Fraction | int, *, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise ValueError(f"{label} must be an exact Fraction or integer.")
    return Fraction(value)


@dataclass(frozen=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("Interval lower bound exceeds its upper bound.")

    @classmethod
    def make(
        cls,
        lower: Fraction | int,
        upper: Fraction | int,
    ) -> "RationalInterval":
        return cls(
            _fraction(lower, label="lower"),
            _fraction(upper, label="upper"),
        )


def decide_strict_less(
    left: RationalInterval,
    right: RationalInterval,
) -> bool:
    """Decide ``left < right`` only when the intervals prove one side."""

    if left.upper < right.lower:
        return True
    if left.lower >= right.upper:
        return False
    raise IndeterminateIntervalDecision(
        "Intervals cross the strict-comparison boundary; refine or FAIL."
    )


@dataclass(frozen=True)
class KernelPerturbationBound:
    schema: str
    beta: str
    uniform_energy_error: str
    steps: int
    energy_difference_error_upper: str
    acceptance_probability_error_upper: str
    kernel_row_l1_error_upper: str
    finite_step_tv_error_upper: str
    requires_verified_uniform_energy_interval: bool
    uniform_energy_interval_verified_by_this_module: bool
    implementation_kernel_equality_claimed: bool

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def certify_kernel_perturbation_bound(
    *,
    beta: Fraction | int,
    uniform_energy_error: Fraction | int,
    steps: int,
) -> KernelPerturbationBound:
    """Instantiate the review's conditional Lipschitz/TV perturbation bound."""

    resolved_beta = _fraction(beta, label="beta")
    epsilon = _fraction(
        uniform_energy_error,
        label="uniform_energy_error",
    )
    if resolved_beta < 0 or epsilon < 0:
        raise ValueError("beta and uniform_energy_error must be nonnegative.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    difference_error = 2 * epsilon
    acceptance_error = min(Fraction(1), 2 * resolved_beta * epsilon)
    row_l1_error = min(Fraction(2), 4 * resolved_beta * epsilon)
    tv_error = min(Fraction(1), 2 * steps * resolved_beta * epsilon)
    return KernelPerturbationBound(
        schema=KERNEL_PERTURBATION_SCHEMA_V15,
        beta=canonical_fraction_text(resolved_beta),
        uniform_energy_error=canonical_fraction_text(epsilon),
        steps=steps,
        energy_difference_error_upper=canonical_fraction_text(
            difference_error
        ),
        acceptance_probability_error_upper=canonical_fraction_text(
            acceptance_error
        ),
        kernel_row_l1_error_upper=canonical_fraction_text(row_l1_error),
        finite_step_tv_error_upper=canonical_fraction_text(tv_error),
        requires_verified_uniform_energy_interval=True,
        uniform_energy_interval_verified_by_this_module=False,
        implementation_kernel_equality_claimed=False,
    )


__all__ = [
    "KERNEL_PERTURBATION_SCHEMA_V15",
    "IndeterminateIntervalDecision",
    "KernelPerturbationBound",
    "RationalInterval",
    "certify_kernel_perturbation_bound",
    "decide_strict_less",
]
