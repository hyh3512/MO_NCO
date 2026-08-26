from __future__ import annotations

"""Explicit finite-particle constants for a deterministic bootstrap FK flow.

The certificate applies to the branch that multinomial-resamples every type at
all positive stages and then mutates every particle the same number of times
with a target-invariant Markov kernel.  It does not apply to online ESS
resampling.  The bound is deliberately state-space-cardinality free; its price
is an explicit potential-oscillation factor.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


class FeynmanKacCertificateError(ValueError):
    """Raised when a deterministic bootstrap certificate is ill formed."""


@dataclass(frozen=True)
class BootstrapFeynmanKacPlan:
    beta_schedule: Tuple[float, ...]
    potential_upper_bound: float
    particle_count: int
    failure_budget: float
    target_cell_mass_lower_bound: float | None
    backward_oscillation_ratios: Tuple[float, ...]
    stability_sum: float
    finite_particle_mse_constant: float
    cellwise_error_radius: float
    chebyshev_cell_miss_bound: float | None
    radius_gate_cell_miss_bound: float | None
    selected_cell_miss_bound: float | None
    required_particles_chebyshev: int | None
    required_particles_radius: int | None
    required_particles_selected: int | None
    coverage_gate_pass: bool | None


@dataclass(frozen=True)
class ContractionAwareFeynmanKacPlan:
    """Published finite-family concentration constants for fixed schedules.

    The constants instantiate Theorem 12 of Giraud and Del Moral (Bernoulli,
    2017).  Stagewise products are retained as diagnostics, but the certified
    radius is computed from that theorem's ``(M, a)`` condition rather than
    from an unsupported product-of-local-Lipschitz shortcut.
    """

    published_theorem: str
    beta_schedule: Tuple[float, ...]
    potential_upper_bound: float
    global_refresh_probability: float
    mutation_steps_by_stage: Tuple[int, ...]
    particle_count: int
    observable_count: int
    failure_budget: float
    incremental_potential_ratios: Tuple[float, ...]
    stage_minorization_constants: Tuple[float, ...]
    mutation_contraction_bounds: Tuple[float, ...]
    stage_lipschitz_bounds: Tuple[float, ...]
    maximum_incremental_potential_ratio: float
    maximum_mutation_dobrushin_bound: float
    regularity_product: float
    regularity_threshold: float
    published_concentration_gate: bool
    theorem_a: float | None
    concentration_r1: float | None
    concentration_r2: float | None
    concentration_y: float | None
    local_event_count: int
    simultaneous_error_radius_raw: float | None
    simultaneous_error_radius: float


def _validate_schedule(beta_schedule: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in beta_schedule)
    if not values:
        raise FeynmanKacCertificateError("beta_schedule cannot be empty.")
    if values[0] != 0.0:
        raise FeynmanKacCertificateError("beta_schedule must start at zero.")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise FeynmanKacCertificateError(
            "beta_schedule must contain finite nonnegative values."
        )
    if any(right <= left for left, right in zip(values, values[1:])):
        raise FeynmanKacCertificateError(
            "beta_schedule must be strictly increasing after beta_0."
        )
    return values


def _validate_stage_steps(
    mutation_steps_by_stage: Sequence[int],
    *,
    positive_stage_count: int,
) -> Tuple[int, ...]:
    steps = tuple(mutation_steps_by_stage)
    if len(steps) != positive_stage_count:
        raise FeynmanKacCertificateError(
            "mutation_steps_by_stage must contain one entry per positive stage."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in steps
    ):
        raise FeynmanKacCertificateError(
            "mutation_steps_by_stage must contain nonnegative integers."
        )
    return steps


def recommend_mutation_steps_for_stage_contraction(
    beta_schedule: Sequence[float],
    *,
    potential_upper_bound: float,
    global_refresh_probability: float,
    maximum_mixing_product: float,
) -> Tuple[int, ...]:
    """Plan fixed counts with ``max(g_l) max(rho_l) <= z_max < 1/2``.

    The returned counts are a design aid, not an adaptive stopping rule.  They
    must be frozen in the run contract before sampling.
    """

    beta = _validate_schedule(beta_schedule)
    v_max = float(potential_upper_bound)
    gamma = float(global_refresh_probability)
    target = float(maximum_mixing_product)
    if not math.isfinite(v_max) or v_max < 0.0:
        raise FeynmanKacCertificateError(
            "potential_upper_bound must be finite and nonnegative."
        )
    if not math.isfinite(gamma) or not (0.0 < gamma <= 1.0):
        raise FeynmanKacCertificateError(
            "global_refresh_probability must lie in (0, 1]."
        )
    if not math.isfinite(target) or not (0.0 < target < 0.5):
        raise FeynmanKacCertificateError(
            "maximum_mixing_product must lie in (0, 0.5)."
        )

    maximum_potential_ratio = max(
        math.exp((right - left) * v_max)
        for left, right in zip(beta, beta[1:])
    )
    maximum_contraction = target / maximum_potential_ratio
    planned = []
    for beta_stage in beta[1:]:
        minorization = gamma * math.exp(-beta_stage * v_max)
        if minorization >= 1.0:
            steps = 1
        else:
            log_contraction = math.log1p(-minorization)
            steps = max(
                0,
                int(
                    math.ceil(
                        math.log(maximum_contraction)
                        / log_contraction
                    )
                ),
            )
            while (
                math.exp(steps * log_contraction)
                > maximum_contraction
            ):
                steps += 1
        planned.append(steps)
    return tuple(planned)


def make_contraction_aware_fk_plan(
    beta_schedule: Sequence[float],
    *,
    potential_upper_bound: float,
    global_refresh_probability: float,
    mutation_steps_by_stage: Sequence[int],
    particle_count: int,
    observable_count: int,
    failure_budget: float,
) -> ContractionAwareFeynmanKacPlan:
    """Instantiate a published fixed-schedule concentration radius.

    At stage ``l``, the incremental potential has ratio ``g_l``.  The global
    refresh gives a one-step minorization ``epsilon_l`` and hence an
    ``s_l``-step Dobrushin bound ``b_l``.  Let ``M=max_l g_l`` and
    ``b=max_l b_l``.  The cited theorem applies when there is an ``a in (0,1)``
    such that ``b <= a/[M(1+a)]``; equivalently, ``M b < 1/2``.  We use the
    smallest admissible ``a=Mb/(1-Mb)`` (or a vanishing positive surrogate
    when ``b=0``) and the theorem's explicit one-sided constants.

    The returned radius is simultaneous for the declared finite family of
    final-time observables by a union bound.  It does not apply to observables
    selected from the same particle stream after inspection.
    """

    beta = _validate_schedule(beta_schedule)
    v_max = float(potential_upper_bound)
    gamma = float(global_refresh_probability)
    if not math.isfinite(v_max) or v_max < 0.0:
        raise FeynmanKacCertificateError(
            "potential_upper_bound must be finite and nonnegative."
        )
    if not math.isfinite(gamma) or not (0.0 <= gamma <= 1.0):
        raise FeynmanKacCertificateError(
            "global_refresh_probability must lie in [0, 1]."
        )
    steps = _validate_stage_steps(
        mutation_steps_by_stage,
        positive_stage_count=len(beta) - 1,
    )
    if (
        isinstance(particle_count, bool)
        or not isinstance(particle_count, int)
        or particle_count <= 0
    ):
        raise FeynmanKacCertificateError(
            "particle_count must be a positive integer."
        )
    if (
        isinstance(observable_count, bool)
        or not isinstance(observable_count, int)
        or observable_count <= 0
    ):
        raise FeynmanKacCertificateError(
            "observable_count must be a positive integer."
        )
    delta = float(failure_budget)
    if not math.isfinite(delta) or not (0.0 < delta < 1.0):
        raise FeynmanKacCertificateError(
            "failure_budget must lie in (0, 1)."
        )

    potential_ratios = tuple(
        math.exp((right - left) * v_max)
        for left, right in zip(beta, beta[1:])
    )
    minorizations = tuple(
        gamma * math.exp(-beta_stage * v_max)
        for beta_stage in beta[1:]
    )
    contractions = tuple(
        (1.0 - epsilon) ** stage_steps
        for epsilon, stage_steps in zip(minorizations, steps)
    )
    stage_lipschitz = tuple(
        ratio * contraction
        for ratio, contraction in zip(
            potential_ratios,
            contractions,
        )
    )
    maximum_ratio = max(potential_ratios, default=1.0)
    maximum_contraction = max(contractions, default=0.0)
    regularity_product = maximum_ratio * maximum_contraction
    regularity_gate = regularity_product < 0.5
    local_event_count = observable_count
    if regularity_gate:
        theorem_a = (
            math.nextafter(
                regularity_product / (1.0 - regularity_product),
                1.0,
            )
            if regularity_product > 0.0
            else 1e-15
        )
        r1 = (
            4.0
            * maximum_ratio
            * maximum_ratio
            * (1.0 + theorem_a) ** 2
            / (1.0 - theorem_a)
        )
        r2 = (
            2.0
            * math.sqrt(2.0)
            / math.sqrt(1.0 - theorem_a * theorem_a)
        )
        y = math.log(local_event_count / delta)
        h0 = 2.0 * (y + math.sqrt(y))
        raw_radius = (
            r1 * (1.0 + h0) / particle_count
            + r2 * math.sqrt(y / particle_count)
        )
        radius = min(1.0, raw_radius)
    else:
        theorem_a = None
        r1 = None
        r2 = None
        y = None
        raw_radius = None
        radius = 1.0
    return ContractionAwareFeynmanKacPlan(
        published_theorem=(
            "Giraud_Del_Moral_2017_Theorem_12_one_sided"
        ),
        beta_schedule=beta,
        potential_upper_bound=v_max,
        global_refresh_probability=gamma,
        mutation_steps_by_stage=steps,
        particle_count=particle_count,
        observable_count=observable_count,
        failure_budget=delta,
        incremental_potential_ratios=potential_ratios,
        stage_minorization_constants=minorizations,
        mutation_contraction_bounds=contractions,
        stage_lipschitz_bounds=stage_lipschitz,
        maximum_incremental_potential_ratio=maximum_ratio,
        maximum_mutation_dobrushin_bound=maximum_contraction,
        regularity_product=regularity_product,
        regularity_threshold=0.5,
        published_concentration_gate=regularity_gate,
        theorem_a=theorem_a,
        concentration_r1=r1,
        concentration_r2=r2,
        concentration_y=y,
        local_event_count=local_event_count,
        simultaneous_error_radius_raw=raw_radius,
        simultaneous_error_radius=radius,
    )


def bootstrap_fk_stability_constants(
    beta_schedule: Sequence[float],
    *,
    potential_upper_bound: float,
) -> tuple[Tuple[float, ...], float, float]:
    """Return ``q_{p,L}``, ``S_L`` and ``B_L^(2)``.

    For ``0 <= V <= Vmax`` and incremental potentials
    ``G_l=exp(-(beta_l-beta_{l-1})V)``, the unnormalized backward semigroup
    satisfies

    ``sup_x Q_{p,L}(1)(x) / inf_x Q_{p,L}(1)(x)
       <= exp((beta_L-beta_p)Vmax)``.

    The elementary telescoping/Minkowski proof then gives

    ``E[(eta_L^m(f)-eta_L(f))^2] <= B_L^(2)/m``

    for every ``0 <= f <= 1``, with
    ``B_L^(2)=0.25*(sum_p q_{p,L})^2``.
    """

    beta = _validate_schedule(beta_schedule)
    v_max = float(potential_upper_bound)
    if not math.isfinite(v_max) or v_max < 0.0:
        raise FeynmanKacCertificateError(
            "potential_upper_bound must be finite and nonnegative."
        )
    beta_final = beta[-1]
    ratios = tuple(
        math.exp((beta_final - beta_p) * v_max) for beta_p in beta
    )
    stability_sum = sum(ratios)
    mse_constant = 0.25 * stability_sum * stability_sum
    return ratios, stability_sum, mse_constant


def required_particles_for_bootstrap_radius(
    *,
    stability_sum: float,
    stage_count_including_zero: int,
    target_mass_lower_bound: float,
    failure_budget: float,
) -> int:
    """Smallest integer making the explicit radius strictly below the mass."""

    s = float(stability_sum)
    p = float(target_mass_lower_bound)
    delta = float(failure_budget)
    if not math.isfinite(s) or s <= 0.0:
        raise FeynmanKacCertificateError("stability_sum must be positive.")
    if stage_count_including_zero <= 0:
        raise FeynmanKacCertificateError(
            "stage_count_including_zero must be positive."
        )
    if not math.isfinite(p) or not (0.0 < p <= 1.0):
        raise FeynmanKacCertificateError(
            "target_mass_lower_bound must lie in (0, 1]."
        )
    if not math.isfinite(delta) or not (0.0 < delta < 1.0):
        raise FeynmanKacCertificateError("failure_budget must lie in (0, 1).")
    threshold = (
        s
        * s
        * math.log(2.0 * stage_count_including_zero / delta)
        / (2.0 * p * p)
    )
    # Strict radius < p is equivalent to m > threshold.
    return max(1, math.floor(threshold) + 1)


def make_bootstrap_fk_plan(
    beta_schedule: Sequence[float],
    *,
    potential_upper_bound: float,
    particle_count: int,
    failure_budget: float,
    target_cell_mass_lower_bound: float | None = None,
) -> BootstrapFeynmanKacPlan:
    """Instantiate the MSE and high-probability cellwise radius constants."""

    beta = _validate_schedule(beta_schedule)
    if (
        isinstance(particle_count, bool)
        or not isinstance(particle_count, int)
        or particle_count <= 0
    ):
        raise FeynmanKacCertificateError(
            "particle_count must be a positive integer."
        )
    m = particle_count
    delta = float(failure_budget)
    if not math.isfinite(delta) or not (0.0 < delta < 1.0):
        raise FeynmanKacCertificateError("failure_budget must lie in (0, 1).")
    ratios, stability_sum, mse_constant = bootstrap_fk_stability_constants(
        beta,
        potential_upper_bound=potential_upper_bound,
    )
    radius = stability_sum * math.sqrt(
        math.log(2.0 * len(beta) / delta) / (2.0 * m)
    )

    p_lower: float | None
    if target_cell_mass_lower_bound is None:
        p_lower = None
        chebyshev = None
        radius_failure = None
        selected = None
        required_cheb = None
        required_radius = None
        required_selected = None
        gate = None
    else:
        p_lower = float(target_cell_mass_lower_bound)
        if not math.isfinite(p_lower) or not (0.0 < p_lower <= 1.0):
            raise FeynmanKacCertificateError(
                "target_cell_mass_lower_bound must lie in (0, 1]."
            )
        chebyshev = min(1.0, mse_constant / (m * p_lower * p_lower))
        radius_failure = delta if radius < p_lower else 1.0
        selected = min(chebyshev, radius_failure)
        required_cheb = max(
            1,
            int(math.ceil(mse_constant / (delta * p_lower * p_lower))),
        )
        required_radius = required_particles_for_bootstrap_radius(
            stability_sum=stability_sum,
            stage_count_including_zero=len(beta),
            target_mass_lower_bound=p_lower,
            failure_budget=delta,
        )
        required_selected = min(required_cheb, required_radius)
        gate = selected <= delta

    return BootstrapFeynmanKacPlan(
        beta_schedule=beta,
        potential_upper_bound=float(potential_upper_bound),
        particle_count=m,
        failure_budget=delta,
        target_cell_mass_lower_bound=p_lower,
        backward_oscillation_ratios=ratios,
        stability_sum=stability_sum,
        finite_particle_mse_constant=mse_constant,
        cellwise_error_radius=radius,
        chebyshev_cell_miss_bound=chebyshev,
        radius_gate_cell_miss_bound=radius_failure,
        selected_cell_miss_bound=selected,
        required_particles_chebyshev=required_cheb,
        required_particles_radius=required_radius,
        required_particles_selected=required_selected,
        coverage_gate_pass=gate,
    )
