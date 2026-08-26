from __future__ import annotations

"""Strict terminal-regeneration certificates for fixed-reference Pareto-SMC.

The certificate in this module is deliberately narrower than a general
Feynman--Kac empirical-measure concentration theorem.  It exploits the exact
uniform-independence Metropolis refresh already present in the fixed-schedule
sampler.  A target-minorization decomposition gives a finite-step lower bound
on each terminal cell-hit probability.  Conditional Hoeffding bounds on an
independent pilot stream and an independent confirm no-hit calculation then
produce a direct metric certificate.

All functions are deterministic and fail closed on malformed inputs.  They do
not infer mathematical constants from observed ESS, acceptance rates, or
archive sizes.
"""

import math
import struct
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence, Tuple


class RegenerationCertificateError(ValueError):
    """Raised when a regeneration certificate input is malformed."""


_DECIMAL_PRECISION = 180
_SMALLEST_POSITIVE_FLOAT = math.nextafter(0.0, math.inf)


@dataclass(frozen=True)
class EqualDualStreamSchedule:
    """One exact equal-pilot/equal-confirm schedule under a fixed budget."""

    total_evaluations: int
    stream_count: int
    type_count: int
    particles_per_type: int
    total_mutations_per_particle: int
    particles_per_stream: int
    evaluations_per_stream: int
    evaluations_per_particle_per_stream: int
    checkpoint_period: int
    checkpoint_aligned: bool
    checkpoint_alignment_scope: str
    checkpoint_full_type_sweep_boundary_verified: bool
    particles_per_stream_within_cap: bool
    exact_budget_identity: bool
    mutation_sweeps_before_first_checkpoint: int | None




@dataclass(frozen=True)
class HeterogeneousPilotConfirmBudget:
    """Exact budget ledger for unequal pilot/confirm typed schedules."""

    pilot_particles_by_type: Tuple[int, ...]
    pilot_mutations_per_particle_by_type: Tuple[int, ...]
    confirm_particles_by_type: Tuple[int, ...]
    confirm_mutations_per_particle_by_type: Tuple[int, ...]
    pilot_evaluation_cost: int
    confirm_evaluation_cost: int
    total_evaluation_cost: int
    requested_total_evaluations: int
    exact_budget_identity: bool

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HoeffdingSuiteRequirement:
    """Distribution-free unit requirement for simultaneous bounded claims."""

    range_width: float
    simultaneous_claims: int
    familywise_alpha: float
    requested_half_width: float
    minimum_independent_units: int

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PilotMassCertificate:
    """One-sided pilot lower bound for a target cell mass."""

    empirical_terminal_mass: float
    pilot_particles: int
    pilot_failure_budget: float
    pilot_hoeffding_radius: float
    pilot_residual_weight: float
    target_mass_lower_bound: float
    positive_gate: bool


@dataclass(frozen=True)
class PilotObservationRequirement:
    """Pre-run nonemptiness gate for a desired pilot mass certificate."""

    desired_target_mass_lower_bound: float
    pilot_particles: int
    pilot_failure_budget: float
    pilot_residual_weight: float
    pilot_hoeffding_radius: float
    minimum_empirical_terminal_mass: float
    feasible: bool
    gate: str


@dataclass(frozen=True)
class AssignmentPilotNonemptinessPreflight:
    """Assignment-level pilot feasibility ledger for disjoint cells."""

    desired_target_mass_lower_bounds_by_cell: Tuple[float, ...]
    assigned_type_by_cell: Tuple[int, ...]
    pilot_particles_by_type: Tuple[int, ...]
    pilot_failure_budgets_by_cell: Tuple[float, ...]
    pilot_residual_weights_by_type: Tuple[float, ...]
    cell_requirements: Tuple[PilotObservationRequirement, ...]
    required_empirical_mass_by_cell: Tuple[float, ...]
    cell_feasible_by_cell: Tuple[bool, ...]
    required_empirical_mass_sum_by_type: Tuple[float, ...]
    simplex_feasible_by_type: Tuple[bool, ...]
    mutually_exclusive_cells_declared: bool
    all_cell_requirements_feasible: bool
    all_type_simplex_constraints_feasible: bool
    feasible: bool
    gate: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConfirmCellCertificate:
    """Finite-step confirm hit certificate for one assigned cell."""

    target_mass_lower_bound: float
    confirm_particles: int
    confirm_residual_weight: float
    per_particle_hit_lower_bound: float
    cell_miss_probability_upper_bound: float


@dataclass(frozen=True)
class JointCertificateDesign:
    """A complete confirm design evaluated against pilot mass lower bounds."""

    assigned_type_by_cell: Tuple[int, ...]
    particles_per_type: Tuple[int, ...]
    total_mutation_steps_by_type: Tuple[int, ...]
    terminal_regeneration_steps_by_type: Tuple[int, ...]
    global_refresh_probability_by_type: Tuple[float, ...]
    normalizer_lower_bound_by_type: Tuple[float, ...]
    residual_weight_by_type: Tuple[float, ...]
    target_mass_lower_bound_by_type_cell: Tuple[Tuple[float, ...], ...]
    per_cell_hit_lower_bound: Tuple[float, ...]
    per_cell_miss_upper_bound: Tuple[float, ...]
    simultaneous_miss_upper_bound: float
    mass_bound_failure_probability: float
    total_metric_failure_probability_upper_bound: float
    confirm_evaluation_cost: int
    expected_global_refresh_proposals: float
    refresh_application_scope: str
    success_igd_bound: float
    failure_igd_cap: float
    expected_igd_upper_bound: float
    success_hv_deficit_bound: float
    failure_hv_deficit_cap: float
    expected_hv_deficit_upper_bound: float
    requested_confirm_failure_budget: float
    confirm_failure_gate: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RefreshRequirement:
    """Closed-form minimum refresh probability for assigned cell risks."""

    target_mass_lower_bounds: Tuple[float, ...]
    cell_failure_budgets: Tuple[float, ...]
    particles: int
    terminal_steps: int
    normalizer_lower_bound: float
    maximum_residual_weight: float | None
    minimum_global_refresh_probability: float | None
    feasible: bool
    gate: str


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RegenerationCertificateError(f"{label} must be a positive integer.")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RegenerationCertificateError(
            f"{label} must be a nonnegative integer."
        )
    return value


def _probability(value: object, label: str, *, allow_zero: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RegenerationCertificateError(f"{label} must be numeric.") from error
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        interval = "[0,1]" if allow_zero else "(0,1]"
        raise RegenerationCertificateError(f"{label} must lie in {interval}.")
    return result


def _strict_failure_probability(value: object, label: str) -> float:
    result = _probability(value, label, allow_zero=False)
    if result >= 1.0:
        raise RegenerationCertificateError(f"{label} must lie in (0,1).")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RegenerationCertificateError(f"{label} must be numeric.") from error
    if not math.isfinite(result) or result < 0.0:
        raise RegenerationCertificateError(
            f"{label} must be finite and nonnegative."
        )
    return result


def _decimal_float(value: float) -> Decimal:
    """Return the exact decimal representation of one binary64 input."""

    return Decimal.from_float(value)


def _probability_float_lower(value: Decimal) -> float:
    """Map a nonnegative Decimal bound downward to binary64."""

    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    result = float(value)
    if result <= 0.0:
        return 0.0
    if _decimal_float(result) > value:
        result = math.nextafter(result, 0.0)
    return max(0.0, result)


def _probability_float_upper(
    value: Decimal,
    *,
    mathematically_positive: bool = False,
) -> float:
    """Map a Decimal probability bound upward to binary64.

    A positive probability that lies below binary64's normal or subnormal
    range must never become zero: the smallest positive float is then a valid
    outward upper bound.
    """

    if value <= 0:
        return _SMALLEST_POSITIVE_FLOAT if mathematically_positive else 0.0
    if value >= 1:
        return 1.0
    result = float(value)
    if result == 0.0:
        return _SMALLEST_POSITIVE_FLOAT
    if _decimal_float(result) < value:
        result = math.nextafter(result, math.inf)
    return min(1.0, result)


def _nonnegative_float_upper(
    value: Decimal,
    *,
    mathematically_positive: bool = False,
) -> float:
    """Map a nonnegative Decimal bound upward to binary64."""

    if value <= 0:
        return _SMALLEST_POSITIVE_FLOAT if mathematically_positive else 0.0
    result = float(value)
    if result == 0.0:
        return _SMALLEST_POSITIVE_FLOAT
    if math.isinf(result):
        return result
    if _decimal_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _hoeffding_half_width_upper(
    *,
    independent_units: int,
    failure_probability: float,
    range_width: float,
    union_factor: int,
) -> float:
    """Directed upper bound for a Hoeffding square-root expression.

    The returned value bounds
    ``range_width * sqrt(log(union_factor/alpha)/(2*N))`` while treating all
    validated binary64 inputs as their exact real values.
    """

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        ratio_upper = context.divide(
            Decimal(union_factor),
            _decimal_float(failure_probability),
        )
        log_upper = context.ln(ratio_upper)
        log_upper = context.next_plus(log_upper)
        scaled_upper = context.divide(
            log_upper,
            Decimal(2 * independent_units),
        )
        root_upper = context.sqrt(scaled_upper)
        root_upper = context.next_plus(root_upper)
        width_upper = context.multiply(
            _decimal_float(range_width),
            root_upper,
        )
        width_upper = context.next_plus(width_upper)
    return _nonnegative_float_upper(
        width_upper,
        mathematically_positive=True,
    )


def _positive_power_probability_upper(
    base_upper: Decimal,
    exponent: int,
) -> float:
    """Return an outward binary64 upper bound on ``base_upper**exponent``."""

    if exponent == 0 or base_upper >= 1:
        return 1.0
    if base_upper <= 0:
        return 0.0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        powered = context.power(base_upper, exponent)
        if powered > 0:
            powered = context.next_plus(powered)
    return _probability_float_upper(
        powered,
        mathematically_positive=True,
    )


def _hit_probability_lower(
    *,
    residual_weight_upper: float,
    target_mass_lower: float,
) -> float:
    """Return a directed lower bound on ``(1-residual)*mass``."""

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        regenerated_mass_lower = context.subtract(
            Decimal(1),
            _decimal_float(residual_weight_upper),
        )
        if regenerated_mass_lower <= 0:
            return 0.0
        hit_lower = context.multiply(
            regenerated_mass_lower,
            _decimal_float(target_mass_lower),
        )
    return _probability_float_lower(hit_lower)


def _sum_probability_upper(values: Iterable[float]) -> float:
    """Sum nonnegative binary64 bounds with directed outward rounding."""

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for value in values:
            total = context.add(total, _decimal_float(value))
    return _probability_float_upper(total)


def _sum_nonnegative_float_upper(values: Iterable[float]) -> float:
    """Sum nonnegative finite bounds without clipping the result to one."""

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for value in values:
            if math.isinf(value):
                return math.inf
            total = context.add(total, _decimal_float(value))
    return _nonnegative_float_upper(total)


def _negative_product_exponential_lower(
    left: float,
    right: float,
) -> Decimal:
    """Lower-bound ``exp(-left*right)`` including product rounding."""

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        exponent_upper = context.multiply(
            _decimal_float(left),
            _decimal_float(right),
        )
        if exponent_upper == 0:
            return Decimal(1)
        # Values below exp(-1000) cannot have a positive binary64 lower
        # representation.  Avoid an unnecessary extreme Decimal exp call.
        if exponent_upper > Decimal(1000):
            return Decimal(0)
        exponential = context.exp(-exponent_upper)
        if exponential > 0:
            exponential = context.next_minus(exponential)
        return exponential


def _residual_from_minorization(epsilon: float, steps: int) -> float:
    """Return an outward upper bound on ``(1-epsilon)**steps``."""

    if steps == 0:
        return 1.0
    if epsilon <= 0.0:
        return 1.0
    if epsilon >= 1.0:
        return 0.0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        base_upper = context.subtract(
            Decimal(1),
            _decimal_float(epsilon),
        )
    return _positive_power_probability_upper(base_upper, steps)


def _zero_hit_probability(hit_probability: float, particles: int) -> float:
    """Return an outward upper bound on ``(1-hit_probability)**particles``."""

    if hit_probability <= 0.0:
        return 1.0
    if hit_probability >= 1.0:
        return 0.0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        base_upper = context.subtract(
            Decimal(1),
            _decimal_float(hit_probability),
        )
    return _positive_power_probability_upper(base_upper, particles)


def target_normalizer_lower_bound(beta: float, potential_upper_bound: float) -> float:
    """Return ``exp(-beta * V) <= Z`` for ``0 <= U <= V`` and uniform base law."""

    beta_value = _finite_nonnegative(beta, "beta")
    potential_bound = _finite_nonnegative(
        potential_upper_bound,
        "potential_upper_bound",
    )
    decimal_lower = _negative_product_exponential_lower(
        beta_value,
        potential_bound,
    )
    lower = _probability_float_lower(decimal_lower)
    if lower <= 0.0:
        raise RegenerationCertificateError(
            "exp(-beta * potential_upper_bound) has no positive outward-"
            "rounded binary64 lower bound; supply a stronger representable "
            "certified normalizer lower bound."
        )
    return lower



def subset_normalizer_lower_bound(
    *,
    beta: float,
    subset_base_mass_lower_bound: float,
    potential_upper_bound_on_subset: float,
) -> float:
    """Return ``kappa * exp(-beta*v) <= Z`` from a verified base subset.

    The subset and its mass proof must be frozen independently of the random
    streams.  Competitive use additionally requires the subset to be authorized
    by the information contract; hidden metric-reference witness tours must not
    be injected into the algorithm through this artifact.
    """

    beta_value = _finite_nonnegative(beta, "beta")
    kappa = _probability(
        subset_base_mass_lower_bound,
        "subset_base_mass_lower_bound",
        allow_zero=False,
    )
    upper = _finite_nonnegative(
        potential_upper_bound_on_subset,
        "potential_upper_bound_on_subset",
    )
    exponential_lower = _negative_product_exponential_lower(
        beta_value,
        upper,
    )
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        decimal_lower = context.multiply(
            _decimal_float(kappa),
            exponential_lower,
        )
    lower = _probability_float_lower(decimal_lower)
    if lower <= 0.0:
        raise RegenerationCertificateError(
            "The outward-rounded subset normalizer lower bound is not "
            "positive in binary64."
        )
    return lower


def deterministic_target_mass_lower_bound(
    *,
    beta: float,
    subset_base_mass_lower_bound: float,
    potential_upper_bound_on_subset: float,
) -> float:
    """Lower-bound target mass of a cell containing a verified base subset.

    For ``pi(dx) proportional to exp(-beta U(x)) mu(dx)`` with ``U >= 0``, the
    partition function is at most one.  A subset ``B`` contained in the target
    cell with ``mu(B) >= kappa`` and ``U <= v`` therefore gives
    ``pi(cell) >= kappa*exp(-beta*v)``.
    """

    return subset_normalizer_lower_bound(
        beta=beta,
        subset_base_mass_lower_bound=subset_base_mass_lower_bound,
        potential_upper_bound_on_subset=potential_upper_bound_on_subset,
    )

def terminal_residual_weight(
    *,
    global_refresh_probability: float,
    normalizer_lower_bound: float,
    mutation_steps: int,
) -> float:
    """Return the residual weight ``(1-gamma*z_lower)^s``.

    For a uniform independence-MH refresh, one mutation step satisfies
    ``K(x,.) >= gamma * Z * pi(.)``.  Replacing ``Z`` by a certified lower bound
    ``z_lower`` gives the target-regeneration residual above.
    """

    gamma = _probability(
        global_refresh_probability,
        "global_refresh_probability",
    )
    z_lower = _probability(
        normalizer_lower_bound,
        "normalizer_lower_bound",
        allow_zero=False,
    )
    steps = _nonnegative_int(mutation_steps, "mutation_steps")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        epsilon_lower = context.multiply(
            _decimal_float(gamma),
            _decimal_float(z_lower),
        )
    if epsilon_lower <= 0:
        return 1.0
    if epsilon_lower >= 1:
        return 0.0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        base_upper = context.subtract(Decimal(1), epsilon_lower)
    return _positive_power_probability_upper(base_upper, steps)


def minimum_pilot_empirical_mass_for_target_bound(
    *,
    desired_target_mass_lower_bound: float,
    pilot_particles: int,
    pilot_failure_budget: float,
    pilot_residual_weight: float,
) -> PilotObservationRequirement:
    """Return the pilot frequency required to certify a desired target mass.

    The pilot lower bound ``(qhat-e-b)/(1-b)`` reaches ``p0`` exactly when
    ``qhat >= e+b+(1-b)p0``.  A threshold above one proves that the requested
    certificate is empty before the pilot is run.
    """

    desired = _probability(
        desired_target_mass_lower_bound,
        "desired_target_mass_lower_bound",
        allow_zero=False,
    )
    particles = _positive_int(pilot_particles, "pilot_particles")
    delta = _strict_failure_probability(
        pilot_failure_budget,
        "pilot_failure_budget",
    )
    residual = _probability(
        pilot_residual_weight,
        "pilot_residual_weight",
    )
    radius = _hoeffding_half_width_upper(
        independent_units=particles,
        failure_probability=delta,
        range_width=1.0,
        union_factor=1,
    )
    if residual >= 1.0:
        return PilotObservationRequirement(
            desired_target_mass_lower_bound=desired,
            pilot_particles=particles,
            pilot_failure_budget=delta,
            pilot_residual_weight=residual,
            pilot_hoeffding_radius=radius,
            minimum_empirical_terminal_mass=math.inf,
            feasible=False,
            gate="FAIL_PILOT_RESIDUAL_ONE",
        )
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        residual_decimal = _decimal_float(residual)
        desired_decimal = _decimal_float(desired)
        threshold_decimal = context.add(
            _decimal_float(radius),
            context.add(
                desired_decimal,
                context.multiply(
                    residual_decimal,
                    context.subtract(Decimal(1), desired_decimal),
                ),
            ),
        )
    threshold = _nonnegative_float_upper(threshold_decimal)
    feasible = threshold <= 1.0
    return PilotObservationRequirement(
        desired_target_mass_lower_bound=desired,
        pilot_particles=particles,
        pilot_failure_budget=delta,
        pilot_residual_weight=residual,
        pilot_hoeffding_radius=radius,
        minimum_empirical_terminal_mass=threshold,
        feasible=feasible,
        gate="PASS" if feasible else "FAIL_PILOT_CERTIFICATE_EMPTY",
    )


def assignment_pilot_nonemptiness_preflight(
    *,
    desired_target_mass_lower_bounds_by_cell: Sequence[float],
    assigned_type_by_cell: Sequence[int],
    pilot_particles_by_type: Sequence[int],
    pilot_failure_budgets_by_cell: Sequence[float],
    pilot_residual_weights_by_type: Sequence[float],
    mutually_exclusive_cells: bool,
) -> AssignmentPilotNonemptinessPreflight:
    """Preflight cellwise pilot thresholds and per-type simplex feasibility.

    The simplex check is valid only for cells declared mutually exclusive.
    Each cell uses the particle count and residual of its assigned type and its
    own one-sided pilot failure budget.  Passing every cell separately is not
    sufficient: for each type, the required empirical frequencies of its
    assigned disjoint cells must also sum to at most one.
    """

    if mutually_exclusive_cells is not True:
        raise RegenerationCertificateError(
            "mutually_exclusive_cells must be explicitly True for the "
            "assignment simplex preflight."
        )
    try:
        raw_desired = tuple(desired_target_mass_lower_bounds_by_cell)
        raw_assignment = tuple(assigned_type_by_cell)
        raw_particles = tuple(pilot_particles_by_type)
        raw_budgets = tuple(pilot_failure_budgets_by_cell)
        raw_residuals = tuple(pilot_residual_weights_by_type)
    except TypeError as error:
        raise RegenerationCertificateError(
            "Assignment pilot preflight inputs must be finite sequences."
        ) from error

    if not raw_desired:
        raise RegenerationCertificateError(
            "At least one assigned cell is required."
        )
    cell_count = len(raw_desired)
    if len(raw_assignment) != cell_count or len(raw_budgets) != cell_count:
        raise RegenerationCertificateError(
            "Desired masses, assignments, and per-cell failure budgets must "
            "have equal nonzero length."
        )
    if not raw_particles or len(raw_residuals) != len(raw_particles):
        raise RegenerationCertificateError(
            "Pilot particle and residual vectors must have equal positive "
            "type count."
        )

    desired = tuple(
        _probability(
            value,
            f"desired_target_mass_lower_bounds_by_cell[{index}]",
            allow_zero=False,
        )
        for index, value in enumerate(raw_desired)
    )
    particles = tuple(
        _positive_int(value, f"pilot_particles_by_type[{index}]")
        for index, value in enumerate(raw_particles)
    )
    budgets = tuple(
        _strict_failure_probability(
            value,
            f"pilot_failure_budgets_by_cell[{index}]",
        )
        for index, value in enumerate(raw_budgets)
    )
    residuals = tuple(
        _probability(
            value,
            f"pilot_residual_weights_by_type[{index}]",
        )
        for index, value in enumerate(raw_residuals)
    )
    type_count = len(particles)
    assignment = tuple(raw_assignment)
    if any(
        isinstance(type_index, bool)
        or not isinstance(type_index, int)
        or type_index < 0
        or type_index >= type_count
        for type_index in assignment
    ):
        raise RegenerationCertificateError(
            "assigned_type_by_cell must contain one valid integer type "
            "index per cell."
        )

    requirements = tuple(
        minimum_pilot_empirical_mass_for_target_bound(
            desired_target_mass_lower_bound=desired[cell_index],
            pilot_particles=particles[type_index],
            pilot_failure_budget=budgets[cell_index],
            pilot_residual_weight=residuals[type_index],
        )
        for cell_index, type_index in enumerate(assignment)
    )
    required_by_cell = tuple(
        requirement.minimum_empirical_terminal_mass
        for requirement in requirements
    )
    cell_feasible = tuple(
        requirement.feasible for requirement in requirements
    )
    required_sum_by_type = tuple(
        _sum_nonnegative_float_upper(
            required_by_cell[cell_index]
            for cell_index, assigned_type in enumerate(assignment)
            if assigned_type == type_index
        )
        for type_index in range(type_count)
    )
    simplex_feasible = tuple(
        required_sum <= 1.0 for required_sum in required_sum_by_type
    )
    all_cells_feasible = all(cell_feasible)
    all_simplex_feasible = all(simplex_feasible)
    feasible = all_cells_feasible and all_simplex_feasible
    if feasible:
        gate = "PASS"
    elif not all_cells_feasible and not all_simplex_feasible:
        gate = "FAIL_CELL_NONEMPTY_AND_TYPE_SIMPLEX"
    elif not all_cells_feasible:
        gate = "FAIL_CELL_NONEMPTY"
    else:
        gate = "FAIL_TYPE_SIMPLEX"
    return AssignmentPilotNonemptinessPreflight(
        desired_target_mass_lower_bounds_by_cell=desired,
        assigned_type_by_cell=assignment,
        pilot_particles_by_type=particles,
        pilot_failure_budgets_by_cell=budgets,
        pilot_residual_weights_by_type=residuals,
        cell_requirements=requirements,
        required_empirical_mass_by_cell=required_by_cell,
        cell_feasible_by_cell=cell_feasible,
        required_empirical_mass_sum_by_type=required_sum_by_type,
        simplex_feasible_by_type=simplex_feasible,
        mutually_exclusive_cells_declared=True,
        all_cell_requirements_feasible=all_cells_feasible,
        all_type_simplex_constraints_feasible=all_simplex_feasible,
        feasible=feasible,
        gate=gate,
    )


def pilot_target_mass_lower_bound(
    *,
    empirical_terminal_mass: float,
    pilot_particles: int,
    pilot_failure_budget: float,
    pilot_residual_weight: float,
) -> PilotMassCertificate:
    """Convert a pilot terminal hit rate into a target-mass lower bound.

    Conditional on the pre-final pilot history, terminal cell indicators are
    independent, but their different starting states mean they need not be
    identically distributed.  Let ``qbar(C)`` be the average of their
    conditional means.  Hoeffding gives ``qbar(C) >= qhat-e``.  Applying
    ``q_i(C) <= (1-b)pi(C)+b`` particle by particle and averaging then yields

    ``pi(C) >= [qhat-e-b]_+/(1-b)``.
    """

    empirical = _probability(empirical_terminal_mass, "empirical_terminal_mass")
    particles = _positive_int(pilot_particles, "pilot_particles")
    delta = _strict_failure_probability(
        pilot_failure_budget,
        "pilot_failure_budget",
    )
    residual = _probability(
        pilot_residual_weight,
        "pilot_residual_weight",
    )
    if residual >= 1.0:
        raise RegenerationCertificateError(
            "pilot_residual_weight must be strictly below one."
        )
    radius = _hoeffding_half_width_upper(
        independent_units=particles,
        failure_probability=delta,
        range_width=1.0,
        union_factor=1,
    )
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        numerator_lower = context.subtract(
            context.subtract(
                _decimal_float(empirical),
                _decimal_float(radius),
            ),
            _decimal_float(residual),
        )
    if numerator_lower <= 0:
        lower = 0.0
    else:
        with localcontext() as context:
            context.prec = _DECIMAL_PRECISION
            context.rounding = ROUND_CEILING
            denominator_upper = context.subtract(
                Decimal(1),
                _decimal_float(residual),
            )
        with localcontext() as context:
            context.prec = _DECIMAL_PRECISION
            context.rounding = ROUND_FLOOR
            lower_decimal = context.divide(
                numerator_lower,
                denominator_upper,
            )
        lower = _probability_float_lower(lower_decimal)
    return PilotMassCertificate(
        empirical_terminal_mass=empirical,
        pilot_particles=particles,
        pilot_failure_budget=delta,
        pilot_hoeffding_radius=radius,
        pilot_residual_weight=residual,
        target_mass_lower_bound=min(1.0, lower),
        positive_gate=lower > 0.0,
    )


def confirm_cell_certificate(
    *,
    target_mass_lower_bound: float,
    confirm_particles: int,
    confirm_residual_weight: float,
) -> ConfirmCellCertificate:
    """Return the exact union-bound ingredient for one confirm cell."""

    mass = _probability(
        target_mass_lower_bound,
        "target_mass_lower_bound",
    )
    particles = _positive_int(confirm_particles, "confirm_particles")
    residual = _probability(
        confirm_residual_weight,
        "confirm_residual_weight",
    )
    hit = _hit_probability_lower(
        residual_weight_upper=residual,
        target_mass_lower=mass,
    )
    miss = _zero_hit_probability(hit, particles)
    return ConfirmCellCertificate(
        target_mass_lower_bound=mass,
        confirm_particles=particles,
        confirm_residual_weight=residual,
        per_particle_hit_lower_bound=hit,
        cell_miss_probability_upper_bound=miss,
    )


def minimum_refresh_for_assigned_cells(
    *,
    target_mass_lower_bounds: Sequence[float],
    cell_failure_budgets: Sequence[float],
    particles: int,
    terminal_steps: int,
    normalizer_lower_bound: float,
) -> RefreshRequirement:
    """Closed-form minimum refresh probability for separate cell risk caps.

    For each assigned cell, the sufficient condition

    ``[1-(1-b)p]^m <= delta``

    is equivalent to ``b <= 1-(1-delta**(1/m))/p``.  The strongest assigned
    cell gives the maximum admissible residual.  Since
    ``b=(1-gamma*z_lower)^s``, monotonicity yields the minimum ``gamma``.
    """

    masses = tuple(
        _probability(value, f"target_mass_lower_bounds[{index}]")
        for index, value in enumerate(target_mass_lower_bounds)
    )
    budgets = tuple(
        _strict_failure_probability(value, f"cell_failure_budgets[{index}]")
        for index, value in enumerate(cell_failure_budgets)
    )
    if not masses or len(masses) != len(budgets):
        raise RegenerationCertificateError(
            "target_mass_lower_bounds and cell_failure_budgets must be "
            "nonempty and have equal length."
        )
    m = _positive_int(particles, "particles")
    steps = _positive_int(terminal_steps, "terminal_steps")
    z_lower = _probability(
        normalizer_lower_bound,
        "normalizer_lower_bound",
        allow_zero=False,
    )

    residual_caps = []
    for mass, delta in zip(masses, budgets):
        # Stable form of 1 - delta**(1/m).  The direct expression loses
        # significant relative precision when delta is close to one or m is
        # large, exactly the regime in which the feasibility boundary is
        # tight.
        required_hit = -math.expm1(math.log(delta) / m)
        if mass <= 0.0 or required_hit > mass:
            return RefreshRequirement(
                target_mass_lower_bounds=masses,
                cell_failure_budgets=budgets,
                particles=m,
                terminal_steps=steps,
                normalizer_lower_bound=z_lower,
                maximum_residual_weight=None,
                minimum_global_refresh_probability=None,
                feasible=False,
                gate="FAIL_PARTICLE_MASS_INSUFFICIENT",
            )
        residual_caps.append(1.0 - required_hit / mass)

    residual_cap = min(residual_caps)
    if residual_cap <= 0.0:
        required_minorization = 1.0
    else:
        # Stable form of 1 - residual_cap**(1/steps).
        required_minorization = -math.expm1(math.log(residual_cap) / steps)
    gamma = required_minorization / z_lower

    def risks_satisfied(candidate: float) -> bool:
        residual = terminal_residual_weight(
            global_refresh_probability=candidate,
            normalizer_lower_bound=z_lower,
            mutation_steps=steps,
        )
        for mass, delta in zip(masses, budgets):
            # Use the same directed lower hit bound as the public confirm
            # certificate.  A multiply followed by a single ``nextafter`` is
            # not a valid general replacement for multi-operation directed
            # rounding and can make the inverse design one ulp too optimistic.
            hit = _hit_probability_lower(
                residual_weight_upper=residual,
                target_mass_lower=mass,
            )
            if _zero_hit_probability(hit, m) > delta:
                return False
        return True

    feasible = risks_satisfied(1.0)
    if feasible:
        # Positive finite IEEE-754 binary64 bit patterns have the same order as
        # their numerical values.  Search that discrete ordered set directly,
        # rather than stopping at the rounded algebraic inversion.  This
        # proves that the returned value satisfies the public certificate
        # predicate while its immediate predecessor does not.
        def float_to_bits(value: float) -> int:
            return int.from_bytes(
                struct.pack(">d", value),
                "big",
                signed=False,
            )

        def bits_to_float(value: int) -> float:
            return struct.unpack(
                ">d",
                value.to_bytes(8, "big", signed=False),
            )[0]

        lower_bits = float_to_bits(0.0)
        upper_bits = float_to_bits(1.0)
        while lower_bits + 1 < upper_bits:
            midpoint_bits = (lower_bits + upper_bits) // 2
            candidate = bits_to_float(midpoint_bits)
            if risks_satisfied(candidate):
                upper_bits = midpoint_bits
            else:
                lower_bits = midpoint_bits
        gamma = bits_to_float(upper_bits)
        if (
            not risks_satisfied(gamma)
            or risks_satisfied(math.nextafter(gamma, -math.inf))
        ):
            raise RegenerationCertificateError(
                "Binary64 refresh-probability minimality search failed."
            )
    return RefreshRequirement(
        target_mass_lower_bounds=masses,
        cell_failure_budgets=budgets,
        particles=m,
        terminal_steps=steps,
        normalizer_lower_bound=z_lower,
        maximum_residual_weight=residual_cap,
        minimum_global_refresh_probability=gamma if feasible else None,
        feasible=feasible,
        gate="PASS" if feasible else "FAIL_REFRESH_PROBABILITY_EXCEEDS_ONE",
    )


def evaluate_joint_certificate_design(
    *,
    target_mass_lower_bound_by_type_cell: Sequence[Sequence[float]],
    particles_per_type: Sequence[int],
    total_mutation_steps_by_type: Sequence[int],
    terminal_regeneration_steps_by_type: Sequence[int],
    global_refresh_probability_by_type: Sequence[float],
    normalizer_lower_bound_by_type: Sequence[float],
    requested_confirm_failure_budget: float,
    success_igd_bound: float,
    failure_igd_cap: float,
    success_hv_deficit_bound: float,
    failure_hv_deficit_cap: float,
    mass_bound_failure_probability: float,
    assigned_type_by_cell: Sequence[int] | None = None,
) -> JointCertificateDesign:
    """Evaluate a complete confirm design and its metric-risk bounds.

    If no assignment is supplied, each cell is assigned to the type maximizing
    its certified finite-step per-particle hit lower bound.  The assignment is
    deterministic and therefore may be applied to pilot-certified mass bounds
    before an independent confirm stream.
    """

    mass_matrix = tuple(
        tuple(
            _probability(value, f"mass[{r}][{j}]")
            for j, value in enumerate(row)
        )
        for r, row in enumerate(target_mass_lower_bound_by_type_cell)
    )
    if not mass_matrix or not mass_matrix[0]:
        raise RegenerationCertificateError("The target-mass matrix must be nonempty.")
    cell_count = len(mass_matrix[0])
    if any(len(row) != cell_count for row in mass_matrix):
        raise RegenerationCertificateError("All target-mass rows must have equal length.")
    type_count = len(mass_matrix)

    particles = tuple(
        _positive_int(value, f"particles_per_type[{index}]")
        for index, value in enumerate(particles_per_type)
    )
    total_steps = tuple(
        _nonnegative_int(value, f"total_mutation_steps_by_type[{index}]")
        for index, value in enumerate(total_mutation_steps_by_type)
    )
    terminal_steps = tuple(
        _nonnegative_int(value, f"terminal_regeneration_steps_by_type[{index}]")
        for index, value in enumerate(terminal_regeneration_steps_by_type)
    )
    gammas = tuple(
        _probability(value, f"global_refresh_probability_by_type[{index}]")
        for index, value in enumerate(global_refresh_probability_by_type)
    )
    z_lowers = tuple(
        _probability(
            value,
            f"normalizer_lower_bound_by_type[{index}]",
            allow_zero=False,
        )
        for index, value in enumerate(normalizer_lower_bound_by_type)
    )
    lengths = {len(particles), len(total_steps), len(terminal_steps), len(gammas), len(z_lowers)}
    if lengths != {type_count}:
        raise RegenerationCertificateError(
            "Every per-type design vector must match the target-mass row count."
        )
    if any(t > total for t, total in zip(terminal_steps, total_steps)):
        raise RegenerationCertificateError(
            "terminal regeneration steps cannot exceed total mutation steps."
        )

    residuals = tuple(
        terminal_residual_weight(
            global_refresh_probability=gamma,
            normalizer_lower_bound=z_lower,
            mutation_steps=steps,
        )
        for gamma, z_lower, steps in zip(gammas, z_lowers, terminal_steps)
    )

    certified_hit_matrix = tuple(
        tuple(
            _hit_probability_lower(
                residual_weight_upper=residuals[type_index],
                target_mass_lower=mass_matrix[type_index][cell_index],
            )
            for cell_index in range(cell_count)
        )
        for type_index in range(type_count)
    )

    if assigned_type_by_cell is None:
        assignment = tuple(
            max(
                range(type_count),
                key=lambda r: (certified_hit_matrix[r][j], -r),
            )
            for j in range(cell_count)
        )
    else:
        assignment = tuple(assigned_type_by_cell)
        if len(assignment) != cell_count or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= type_count
            for value in assignment
        ):
            raise RegenerationCertificateError(
                "assigned_type_by_cell must contain one valid type index per cell."
            )

    hits = []
    misses = []
    for cell_index, type_index in enumerate(assignment):
        hit = certified_hit_matrix[type_index][cell_index]
        miss = _zero_hit_probability(hit, particles[type_index])
        hits.append(hit)
        misses.append(miss)
    total_miss = _sum_probability_upper(misses)
    mass_failure = _probability(
        mass_bound_failure_probability,
        "mass_bound_failure_probability",
    )
    total_metric_failure = _sum_probability_upper(
        (mass_failure, total_miss)
    )
    failure_budget = _strict_failure_probability(
        requested_confirm_failure_budget,
        "requested_confirm_failure_budget",
    )
    igd_success = _finite_nonnegative(success_igd_bound, "success_igd_bound")
    igd_cap = _finite_nonnegative(failure_igd_cap, "failure_igd_cap")
    hv_success = _finite_nonnegative(
        success_hv_deficit_bound,
        "success_hv_deficit_bound",
    )
    hv_cap = _finite_nonnegative(
        failure_hv_deficit_cap,
        "failure_hv_deficit_cap",
    )
    if igd_cap < igd_success or hv_cap < hv_success:
        raise RegenerationCertificateError(
            "Failure-event metric caps must not be smaller than success bounds."
        )

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        metric_failure_decimal = _decimal_float(total_metric_failure)
        igd_success_decimal = _decimal_float(igd_success)
        igd_cap_decimal = _decimal_float(igd_cap)
        hv_success_decimal = _decimal_float(hv_success)
        hv_cap_decimal = _decimal_float(hv_cap)
        expected_igd_decimal = context.add(
            igd_success_decimal,
            context.multiply(
                metric_failure_decimal,
                context.subtract(igd_cap_decimal, igd_success_decimal),
            ),
        )
        expected_hv_decimal = context.add(
            hv_success_decimal,
            context.multiply(
                metric_failure_decimal,
                context.subtract(hv_cap_decimal, hv_success_decimal),
            ),
        )
    expected_igd = min(
        igd_cap,
        _probability_float_upper(expected_igd_decimal)
        if igd_cap <= 1.0
        else (
            math.nextafter(float(expected_igd_decimal), math.inf)
            if _decimal_float(float(expected_igd_decimal))
            < expected_igd_decimal
            else float(expected_igd_decimal)
        ),
    )
    expected_hv = min(
        hv_cap,
        _probability_float_upper(expected_hv_decimal)
        if hv_cap <= 1.0
        else (
            math.nextafter(float(expected_hv_decimal), math.inf)
            if _decimal_float(float(expected_hv_decimal))
            < expected_hv_decimal
            else float(expected_hv_decimal)
        ),
    )
    confirm_cost = sum(
        m * (1 + steps) for m, steps in zip(particles, total_steps)
    )
    refresh_burden = sum(
        m * steps * gamma
        for m, steps, gamma in zip(particles, total_steps, gammas)
    )

    return JointCertificateDesign(
        assigned_type_by_cell=assignment,
        particles_per_type=particles,
        total_mutation_steps_by_type=total_steps,
        terminal_regeneration_steps_by_type=terminal_steps,
        global_refresh_probability_by_type=gammas,
        normalizer_lower_bound_by_type=z_lowers,
        residual_weight_by_type=residuals,
        target_mass_lower_bound_by_type_cell=mass_matrix,
        per_cell_hit_lower_bound=tuple(hits),
        per_cell_miss_upper_bound=tuple(misses),
        simultaneous_miss_upper_bound=total_miss,
        mass_bound_failure_probability=mass_failure,
        total_metric_failure_probability_upper_bound=(
            total_metric_failure
        ),
        confirm_evaluation_cost=confirm_cost,
        expected_global_refresh_proposals=refresh_burden,
        refresh_application_scope="all_declared_mutation_steps",
        success_igd_bound=igd_success,
        failure_igd_cap=igd_cap,
        expected_igd_upper_bound=expected_igd,
        success_hv_deficit_bound=hv_success,
        failure_hv_deficit_cap=hv_cap,
        expected_hv_deficit_upper_bound=expected_hv,
        requested_confirm_failure_budget=failure_budget,
        confirm_failure_gate="PASS" if total_miss <= failure_budget else "FAIL",
    )


def pareto_minimal_designs(
    designs: Iterable[JointCertificateDesign],
) -> Tuple[JointCertificateDesign, ...]:
    """Return the exact nondominated finite design set.

    Minimized coordinates are evaluation cost, expected refresh proposals,
    simultaneous miss bound, expected IGD upper bound, and expected HV-deficit
    upper bound.  Because the input menu is finite, this is a global result for
    the declared certificate objective vector.
    """

    resolved = tuple(designs)
    vectors = tuple(
        (
            design.confirm_evaluation_cost,
            design.expected_global_refresh_proposals,
            design.simultaneous_miss_upper_bound,
            design.expected_igd_upper_bound,
            design.expected_hv_deficit_upper_bound,
        )
        for design in resolved
    )
    keep = []
    for index, vector in enumerate(vectors):
        dominated = False
        for other_index, other in enumerate(vectors):
            if index == other_index:
                continue
            if all(a <= b for a, b in zip(other, vector)) and any(
                a < b for a, b in zip(other, vector)
            ):
                dominated = True
                break
        if not dominated:
            keep.append(resolved[index])
    return tuple(keep)


def enumerate_equal_dual_stream_schedules(
    *,
    total_evaluations: int,
    type_count: int,
    max_particles_per_stream: int,
    checkpoint_period: int,
) -> Tuple[EqualDualStreamSchedule, ...]:
    """Enumerate every equal pilot-confirm integer solution exactly.

    The identity is ``2 * R * m * (1+L) = B``.  A schedule is checkpoint
    aligned when the per-stream particle sweep ``R*m`` divides the common
    checkpoint period, so each required checkpoint occurs after an integral
    number of charged initialization/mutation sweeps.
    """

    budget = _positive_int(total_evaluations, "total_evaluations")
    types = _positive_int(type_count, "type_count")
    particle_cap = _positive_int(
        max_particles_per_stream,
        "max_particles_per_stream",
    )
    checkpoint = _positive_int(checkpoint_period, "checkpoint_period")
    stream_count = 2
    denominator = stream_count * types
    if budget % denominator != 0:
        return ()
    product = budget // denominator
    schedules = []
    for particles_per_type in range(1, product + 1):
        if product % particles_per_type:
            continue
        per_particle = product // particles_per_type
        mutations = per_particle - 1
        particles_per_stream = types * particles_per_type
        checkpoint_aligned = checkpoint % particles_per_stream == 0
        sweeps_before_first = (
            checkpoint // particles_per_stream - 1 if checkpoint_aligned else None
        )
        exact_stream = particles_per_stream * per_particle
        exact_total = stream_count * exact_stream
        schedules.append(
            EqualDualStreamSchedule(
                total_evaluations=budget,
                stream_count=stream_count,
                type_count=types,
                particles_per_type=particles_per_type,
                total_mutations_per_particle=mutations,
                particles_per_stream=particles_per_stream,
                evaluations_per_stream=exact_stream,
                evaluations_per_particle_per_stream=per_particle,
                checkpoint_period=checkpoint,
                checkpoint_aligned=checkpoint_aligned,
                checkpoint_alignment_scope=(
                    "evaluation_budget_grid_only"
                ),
                checkpoint_full_type_sweep_boundary_verified=False,
                particles_per_stream_within_cap=(
                    particles_per_stream <= particle_cap
                ),
                exact_budget_identity=(exact_total == budget),
                mutation_sweeps_before_first_checkpoint=(
                    None
                ),
            )
        )
    return tuple(schedules)



def heterogeneous_pilot_confirm_budget(
    *,
    pilot_particles_by_type: Sequence[int],
    pilot_mutations_per_particle_by_type: Sequence[int],
    confirm_particles_by_type: Sequence[int],
    confirm_mutations_per_particle_by_type: Sequence[int],
    requested_total_evaluations: int,
) -> HeterogeneousPilotConfirmBudget:
    """Compute the exact unequal typed pilot--confirm evaluation ledger.

    The identity is

    ``sum_r nP[r] * (1+sP[r]) + sum_r nC[r] * (1+sC[r])``.

    This removes the artificial equal-stream budget quantum without changing
    the requirement that the finite menu and the pilot-to-confirm selection
    rule be frozen before the pilot starts.
    """

    pilot_particles = tuple(
        _positive_int(value, f"pilot_particles_by_type[{index}]")
        for index, value in enumerate(pilot_particles_by_type)
    )
    pilot_steps = tuple(
        _nonnegative_int(
            value,
            f"pilot_mutations_per_particle_by_type[{index}]",
        )
        for index, value in enumerate(pilot_mutations_per_particle_by_type)
    )
    confirm_particles = tuple(
        _positive_int(value, f"confirm_particles_by_type[{index}]")
        for index, value in enumerate(confirm_particles_by_type)
    )
    confirm_steps = tuple(
        _nonnegative_int(
            value,
            f"confirm_mutations_per_particle_by_type[{index}]",
        )
        for index, value in enumerate(confirm_mutations_per_particle_by_type)
    )
    if not pilot_particles or not confirm_particles:
        raise RegenerationCertificateError(
            "Pilot and confirm typed schedules must both be nonempty."
        )
    if len(pilot_particles) != len(pilot_steps):
        raise RegenerationCertificateError(
            "Pilot particle and mutation vectors must have equal length."
        )
    if len(confirm_particles) != len(confirm_steps):
        raise RegenerationCertificateError(
            "Confirm particle and mutation vectors must have equal length."
        )
    if len(pilot_particles) != len(confirm_particles):
        raise RegenerationCertificateError(
            "Pilot and confirm schedules must use the same frozen type count."
        )
    requested = _positive_int(
        requested_total_evaluations,
        "requested_total_evaluations",
    )
    pilot_cost = sum(
        particles * (1 + steps)
        for particles, steps in zip(pilot_particles, pilot_steps)
    )
    confirm_cost = sum(
        particles * (1 + steps)
        for particles, steps in zip(confirm_particles, confirm_steps)
    )
    total = pilot_cost + confirm_cost
    return HeterogeneousPilotConfirmBudget(
        pilot_particles_by_type=pilot_particles,
        pilot_mutations_per_particle_by_type=pilot_steps,
        confirm_particles_by_type=confirm_particles,
        confirm_mutations_per_particle_by_type=confirm_steps,
        pilot_evaluation_cost=pilot_cost,
        confirm_evaluation_cost=confirm_cost,
        total_evaluation_cost=total,
        requested_total_evaluations=requested,
        exact_budget_identity=(total == requested),
    )


def finite_suite_hoeffding_half_width(
    *,
    independent_units: int,
    simultaneous_claims: int,
    familywise_alpha: float,
    range_width: float = 2.0,
) -> float:
    """Return the simultaneous two-sided Hoeffding half-width.

    For independent variables whose declared intervals all have width
    ``range_width``, a union bound over ``H`` claims gives

    ``range_width * sqrt(log(2H/alpha)/(2N))``.
    """

    units = _positive_int(independent_units, "independent_units")
    claims = _positive_int(simultaneous_claims, "simultaneous_claims")
    alpha = _strict_failure_probability(
        familywise_alpha,
        "familywise_alpha",
    )
    width = _finite_nonnegative(range_width, "range_width")
    if width <= 0.0:
        raise RegenerationCertificateError("range_width must be strictly positive.")
    return _hoeffding_half_width_upper(
        independent_units=units,
        failure_probability=alpha,
        range_width=width,
        union_factor=2 * claims,
    )


def minimum_independent_units_for_hoeffding_half_width(
    *,
    requested_half_width: float,
    simultaneous_claims: int,
    familywise_alpha: float,
    range_width: float = 2.0,
) -> HoeffdingSuiteRequirement:
    """Return the smallest integer unit count meeting a Hoeffding width cap."""

    target = _finite_nonnegative(
        requested_half_width,
        "requested_half_width",
    )
    if target <= 0.0:
        raise RegenerationCertificateError(
            "requested_half_width must be strictly positive."
        )
    claims = _positive_int(simultaneous_claims, "simultaneous_claims")
    alpha = _strict_failure_probability(
        familywise_alpha,
        "familywise_alpha",
    )
    width = _finite_nonnegative(range_width, "range_width")
    if width <= 0.0:
        raise RegenerationCertificateError("range_width must be strictly positive.")
    def satisfies(candidate_units: int) -> bool:
        return _hoeffding_half_width_upper(
            independent_units=candidate_units,
            failure_probability=alpha,
            range_width=width,
            union_factor=2 * claims,
        ) <= target

    upper_units = 1
    while not satisfies(upper_units):
        upper_units *= 2
    lower_units = 0
    while upper_units - lower_units > 1:
        midpoint = (lower_units + upper_units) // 2
        if satisfies(midpoint):
            upper_units = midpoint
        else:
            lower_units = midpoint
    units = upper_units
    return HoeffdingSuiteRequirement(
        range_width=width,
        simultaneous_claims=claims,
        familywise_alpha=alpha,
        requested_half_width=target,
        minimum_independent_units=units,
    )

def regeneration_exposure(
    particles_per_type: int,
    mutation_steps: int,
    one_step_target_minorization: float,
) -> float:
    """Return ``m * [1-(1-epsilon)^s]`` for schedule comparisons."""

    particles = _positive_int(particles_per_type, "particles_per_type")
    steps = _nonnegative_int(mutation_steps, "mutation_steps")
    epsilon = _probability(
        one_step_target_minorization,
        "one_step_target_minorization",
    )
    residual = _residual_from_minorization(epsilon, steps)
    return particles * (1.0 - residual)


__all__ = [
    "AssignmentPilotNonemptinessPreflight",
    "ConfirmCellCertificate",
    "EqualDualStreamSchedule",
    "HeterogeneousPilotConfirmBudget",
    "HoeffdingSuiteRequirement",
    "JointCertificateDesign",
    "PilotMassCertificate",
    "PilotObservationRequirement",
    "RefreshRequirement",
    "RegenerationCertificateError",
    "assignment_pilot_nonemptiness_preflight",
    "confirm_cell_certificate",
    "enumerate_equal_dual_stream_schedules",
    "evaluate_joint_certificate_design",
    "finite_suite_hoeffding_half_width",
    "heterogeneous_pilot_confirm_budget",
    "minimum_independent_units_for_hoeffding_half_width",
    "minimum_pilot_empirical_mass_for_target_bound",
    "minimum_refresh_for_assigned_cells",
    "pareto_minimal_designs",
    "pilot_target_mass_lower_bound",
    "regeneration_exposure",
    "subset_normalizer_lower_bound",
    "deterministic_target_mass_lower_bound",
    "target_normalizer_lower_bound",
    "terminal_residual_weight",
]
