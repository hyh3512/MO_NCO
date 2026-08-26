from __future__ import annotations

"""Fail-closed design ledgers for a prospective Pareto-SMC v11 layer.

This module does not alter the sampler.  It audits three pre-run design
questions:

* the smallest stagewise global-refresh probabilities that meet fixed
  worst-case contraction caps;
* exact single-stream and two-stream particle budgets; and
* the finite complexity induced by frozen directions, a Cartesian grid, and
  frozen reference cells.

The refresh objective is only the expected number of global-refresh proposals
per particle, ``sum_l s_l * gamma_l``.  No result in this module claims to
optimize acceptance, runtime, terminal quality, or the unknown Pareto front.
"""

import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from dataclasses import dataclass
from typing import Sequence, Tuple


class ParetoSMCDesignError(ValueError):
    """Raised when a v11 design input is malformed or unauditable."""


_DECIMAL_PRECISION = 180
_SMALLEST_POSITIVE_FLOAT = math.nextafter(0.0, math.inf)


@dataclass(frozen=True)
class StageRefreshDesign:
    """One stage of the separable global-refresh proxy design."""

    stage_index: int
    beta: float
    mutation_steps: int
    target_contraction_cap: float
    refresh_minorization_scale: float
    required_minorization: float | None
    feasible: bool
    minimum_global_refresh_probability: float | None
    contraction_at_minimum: float | None
    expected_global_refresh_proposals_at_minimum: float | None
    infeasibility_reason: str | None


@dataclass(frozen=True)
class RefreshScheduleDesign:
    """Stagewise minimum of the declared refresh-proposal proxy."""

    beta_stages: Tuple[float, ...]
    potential_upper_bound: float
    mutation_steps_by_stage: Tuple[int, ...]
    target_stage_contraction_cap: float
    stages: Tuple[StageRefreshDesign, ...]
    all_stages_feasible: bool
    feasibility_gate: str
    minimum_expected_global_refresh_proposals_per_particle: float | None
    proxy_objective: str
    optimality_scope: str
    numerical_rounding_contract: str
    terminal_quality_optimality_claimed: bool


@dataclass(frozen=True)
class FixedScheduleBudgetPlan:
    """Exact budget arithmetic for one declared number of streams."""

    requested_total_evaluations: int
    stream_count: int
    type_count: int
    mutation_steps_by_stage: Tuple[int, ...]
    evaluations_per_particle_per_stream: int
    exact_budget_quantum: int
    maximum_particles_per_type: int
    particles_per_stream: int
    exact_evaluations_per_stream: int
    exact_total_evaluations: int
    leftover_evaluations: int
    divisible: bool
    particle_feasible: bool
    exact_budget_feasible: bool
    divisibility_gate: str
    budget_gate: str


@dataclass(frozen=True)
class SingleDualBudgetDesign:
    """Single- and two-stream plans under the same total budget envelope."""

    requested_total_evaluations: int
    single_stream: FixedScheduleBudgetPlan
    two_stream: FixedScheduleBudgetPlan
    budget_scope: str


@dataclass(frozen=True)
class FrozenComplexityLedger:
    """Finite design-size ledger for externally frozen structures."""

    objective_dimension: int
    reference_directions: Tuple[Tuple[float, ...], ...]
    type_count: int
    unique_direction_count: int
    grid_cell_counts: Tuple[int, ...]
    grid_cell_capacity: int
    supplied_reference_cell_count: int
    unique_reference_cells: Tuple[Tuple[int, ...], ...]
    unique_reference_cell_count: int
    duplicate_reference_cell_count: int
    pilot_type_cell_observable_count: int
    confirm_cell_observable_count: int
    minimum_archive_cap_for_one_per_reference_cell: int
    declared_archive_cap: int | None
    archive_cap_cardinality_gate: str
    max_type_count: int
    max_grid_cell_count: int
    max_type_gate: str
    max_cell_gate: str
    max_type_max_cell_gate: str
    frozen_structure_sha256: str
    unknown_pareto_front_coverage_claimed: bool
    fixed_size_archive_metric_preservation_claimed: bool
    archive_cap_floor_scope: str


def _validate_nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParetoSMCDesignError(
            f"{label} must be a nonnegative integer."
        )
    return value


def _validate_positive_integer(value: object, label: str) -> int:
    result = _validate_nonnegative_integer(value, label)
    if result == 0:
        raise ParetoSMCDesignError(f"{label} must be positive.")
    return result


def _validate_steps(
    mutation_steps_by_stage: Sequence[int],
    *,
    allow_empty: bool = False,
) -> Tuple[int, ...]:
    try:
        steps = tuple(mutation_steps_by_stage)
    except TypeError as error:
        raise ParetoSMCDesignError(
            "mutation_steps_by_stage must be a finite sequence."
        ) from error
    if not steps and not allow_empty:
        raise ParetoSMCDesignError(
            "mutation_steps_by_stage must contain at least one stage."
        )
    return tuple(
        _validate_nonnegative_integer(
            value,
            f"mutation_steps_by_stage[{index}]",
        )
        for index, value in enumerate(steps)
    )


def _validate_probability_cap(value: float) -> float:
    try:
        cap = float(value)
    except (TypeError, ValueError) as error:
        raise ParetoSMCDesignError(
            "target_stage_contraction_cap must be numeric."
        ) from error
    if not math.isfinite(cap) or not (0.0 <= cap <= 1.0):
        raise ParetoSMCDesignError(
            "target_stage_contraction_cap must lie in [0, 1]."
        )
    return cap


def _required_minorization(cap: float, steps: int) -> float | None:
    if steps == 0:
        return 0.0 if cap == 1.0 else None
    if cap == 0.0:
        return 1.0
    if cap == 1.0:
        return 0.0
    return -math.expm1(math.log(cap) / steps)


def _decimal_float(value: float) -> Decimal:
    return Decimal.from_float(value)


def _probability_float_lower(value: Decimal) -> float:
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


def _refresh_scale_lower(beta: float, potential_upper_bound: float) -> float:
    """Outward lower binary64 bound on ``exp(-beta*V)``."""

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        exponent_upper = context.multiply(
            _decimal_float(beta),
            _decimal_float(potential_upper_bound),
        )
        if exponent_upper == 0:
            return 1.0
        if exponent_upper > Decimal(1000):
            return 0.0
        scale_lower = context.exp(-exponent_upper)
        if scale_lower > 0:
            scale_lower = context.next_minus(scale_lower)
    return _probability_float_lower(scale_lower)


def _stage_contraction_upper(
    gamma: float,
    scale_lower: float,
    steps: int,
) -> float:
    """Outward upper bound on ``(1-gamma*exp(-beta*V))**steps``."""

    if steps == 0:
        return 1.0
    if gamma <= 0.0 or scale_lower <= 0.0:
        return 1.0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        epsilon_lower = context.multiply(
            _decimal_float(gamma),
            _decimal_float(scale_lower),
        )
    if epsilon_lower >= 1:
        return 0.0
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        base_upper = context.subtract(Decimal(1), epsilon_lower)
        contraction_upper = context.power(base_upper, steps)
        if contraction_upper > 0:
            contraction_upper = context.next_plus(contraction_upper)
    return _probability_float_upper(
        contraction_upper,
        mathematically_positive=True,
    )


def design_stagewise_global_refresh(
    beta_stages: Sequence[float],
    *,
    potential_upper_bound: float,
    mutation_steps_by_stage: Sequence[int],
    target_stage_contraction_cap: float,
) -> RefreshScheduleDesign:
    """Minimize ``sum_l s_l gamma_l`` under separate stage caps.

    For stage ``l``, put ``a_l = exp(-beta_l V)``.  If ``s_l > 0``,
    monotonicity gives the analytic lower bound

    ``gamma_l >= (1 - b_bar ** (1 / s_l)) / a_l``.

    The stage is infeasible when that lower bound exceeds one.  A zero-step
    stage is feasible only for the vacuous cap ``b_bar = 1``.  Mathematical
    infeasibility is returned in the ledger; malformed inputs raise
    :class:`ParetoSMCDesignError`.
    """

    try:
        beta = tuple(float(value) for value in beta_stages)
    except (TypeError, ValueError) as error:
        raise ParetoSMCDesignError(
            "beta_stages must be a numeric sequence."
        ) from error
    if not beta:
        raise ParetoSMCDesignError(
            "beta_stages must contain at least one positive-stage beta."
        )
    if any(not math.isfinite(value) or value < 0.0 for value in beta):
        raise ParetoSMCDesignError(
            "beta_stages must contain finite nonnegative values."
        )
    if any(right <= left for left, right in zip(beta, beta[1:])):
        raise ParetoSMCDesignError(
            "beta_stages must be strictly increasing."
        )

    try:
        v_max = float(potential_upper_bound)
    except (TypeError, ValueError) as error:
        raise ParetoSMCDesignError(
            "potential_upper_bound must be numeric."
        ) from error
    if not math.isfinite(v_max) or v_max < 0.0:
        raise ParetoSMCDesignError(
            "potential_upper_bound must be finite and nonnegative."
        )
    steps = _validate_steps(mutation_steps_by_stage)
    if len(steps) != len(beta):
        raise ParetoSMCDesignError(
            "beta_stages and mutation_steps_by_stage must have equal length."
        )
    cap = _validate_probability_cap(target_stage_contraction_cap)

    stage_designs = []
    for stage_index, (beta_stage, stage_steps) in enumerate(
        zip(beta, steps),
        start=1,
    ):
        exponent = beta_stage * v_max
        scale = _refresh_scale_lower(beta_stage, v_max)
        needed_minorization = _required_minorization(cap, stage_steps)

        if needed_minorization is None:
            feasible = False
            minimum_gamma = None
            achieved_contraction = None
            expected_refreshes = None
            reason = (
                "A zero-step stage has contraction one and cannot meet a "
                "cap below one."
            )
        elif needed_minorization == 0.0:
            feasible = True
            minimum_gamma = 0.0
            achieved_contraction = 1.0
            expected_refreshes = 0.0
            reason = None
        elif scale == 0.0:
            feasible = False
            minimum_gamma = None
            achieved_contraction = None
            expected_refreshes = None
            reason = (
                "Even gamma=1 has zero representable minorization scale; "
                "the nontrivial contraction cap is infeasible."
            )
        else:
            def contraction(gamma: float) -> float:
                return _stage_contraction_upper(
                    gamma,
                    scale,
                    stage_steps,
                )

            if contraction(1.0) > cap:
                feasible = False
                minimum_gamma = None
                achieved_contraction = None
                expected_refreshes = None
                reason = (
                    "The analytic minimum global-refresh probability "
                    "exceeds one."
                )
            else:
                feasible = True
                log_required_gamma = (
                    math.log(needed_minorization) + exponent
                )
                analytic_gamma = min(
                    1.0,
                    math.exp(log_required_gamma),
                )
                upper_gamma = (
                    analytic_gamma
                    if contraction(analytic_gamma) <= cap
                    else 1.0
                )
                lower_gamma = 0.0
                # The monotone predicate is evaluated with a Decimal-derived
                # scale lower bound and a directed contraction upper bound.
                # Binary64 bisection then identifies adjacent bracketing
                # probabilities, independent of the analytic seed's rounding.
                for _ in range(1100):
                    next_lower = math.nextafter(lower_gamma, math.inf)
                    if next_lower >= upper_gamma:
                        break
                    midpoint = lower_gamma + (
                        upper_gamma - lower_gamma
                    ) / 2.0
                    if midpoint <= lower_gamma or midpoint >= upper_gamma:
                        break
                    if contraction(midpoint) <= cap:
                        upper_gamma = midpoint
                    else:
                        lower_gamma = midpoint
                minimum_gamma = upper_gamma
                achieved_contraction = contraction(minimum_gamma)
                if achieved_contraction > cap:
                    feasible = False
                    minimum_gamma = None
                    achieved_contraction = None
                    expected_refreshes = None
                    reason = (
                        "Conservative Decimal outward-rounding verification "
                        "failed."
                    )
                else:
                    expected_refreshes = stage_steps * minimum_gamma
                    reason = None

        stage_designs.append(
            StageRefreshDesign(
                stage_index=stage_index,
                beta=beta_stage,
                mutation_steps=stage_steps,
                target_contraction_cap=cap,
                refresh_minorization_scale=scale,
                required_minorization=needed_minorization,
                feasible=feasible,
                minimum_global_refresh_probability=minimum_gamma,
                contraction_at_minimum=achieved_contraction,
                expected_global_refresh_proposals_at_minimum=(
                    expected_refreshes
                ),
                infeasibility_reason=reason,
            )
        )

    stages = tuple(stage_designs)
    all_feasible = all(stage.feasible for stage in stages)
    minimum_expected = (
        sum(
            float(
                stage.expected_global_refresh_proposals_at_minimum
            )
            for stage in stages
        )
        if all_feasible
        else None
    )
    return RefreshScheduleDesign(
        beta_stages=beta,
        potential_upper_bound=v_max,
        mutation_steps_by_stage=steps,
        target_stage_contraction_cap=cap,
        stages=stages,
        all_stages_feasible=all_feasible,
        feasibility_gate="PASS" if all_feasible else "FAIL",
        minimum_expected_global_refresh_proposals_per_particle=(
            minimum_expected
        ),
        proxy_objective="sum_l mutation_steps_l * gamma_l",
        optimality_scope=(
            "smallest conservatively bracketed binary64 probability for "
            "separable stagewise contraction constraints with fixed beta, "
            "potential bound, steps, and cap"
        ),
        numerical_rounding_contract=(
            "decimal_outward_scale_and_contraction_then_binary64_"
            "bisection_v2"
        ),
        terminal_quality_optimality_claimed=False,
    )


def plan_fixed_schedule_budget(
    total_evaluation_budget: int,
    *,
    type_count: int,
    mutation_steps_by_stage: Sequence[int],
    stream_count: int,
) -> FixedScheduleBudgetPlan:
    """Audit exact particle feasibility for ``stream_count`` equal streams.

    The deterministic branch charges one initialization evaluation plus every
    declared mutation proposal.  Thus a design with ``m`` particles per type
    costs exactly

    ``stream_count * type_count * m * (1 + sum_l s_l)``.

    The returned particle count is the largest whole ``m`` that fits.  A
    nonzero leftover is reported as a failed exact-budget gate rather than
    silently discarded.
    """

    budget = _validate_nonnegative_integer(
        total_evaluation_budget,
        "total_evaluation_budget",
    )
    types = _validate_positive_integer(type_count, "type_count")
    streams = _validate_positive_integer(stream_count, "stream_count")
    steps = _validate_steps(mutation_steps_by_stage)
    per_particle = 1 + sum(steps)
    quantum = streams * types * per_particle
    particles_per_type, leftover = divmod(budget, quantum)
    exact_total = budget - leftover
    exact_per_stream = types * particles_per_type * per_particle
    particle_feasible = particles_per_type >= 1
    divisible = leftover == 0
    exact_feasible = particle_feasible and divisible
    if exact_feasible:
        budget_gate = "PASS"
    elif not particle_feasible:
        budget_gate = "FAIL_INSUFFICIENT_FOR_ONE_PARTICLE_PER_TYPE"
    else:
        budget_gate = "FAIL_NON_DIVISIBLE"
    return FixedScheduleBudgetPlan(
        requested_total_evaluations=budget,
        stream_count=streams,
        type_count=types,
        mutation_steps_by_stage=steps,
        evaluations_per_particle_per_stream=per_particle,
        exact_budget_quantum=quantum,
        maximum_particles_per_type=particles_per_type,
        particles_per_stream=types * particles_per_type,
        exact_evaluations_per_stream=exact_per_stream,
        exact_total_evaluations=exact_total,
        leftover_evaluations=leftover,
        divisible=divisible,
        particle_feasible=particle_feasible,
        exact_budget_feasible=exact_feasible,
        divisibility_gate="PASS" if divisible else "FAIL",
        budget_gate=budget_gate,
    )


def plan_single_and_dual_stream_budgets(
    total_evaluation_budget: int,
    *,
    type_count: int,
    mutation_steps_by_stage: Sequence[int],
) -> SingleDualBudgetDesign:
    """Compare one-stream and pilot-confirm plans under one total envelope."""

    single = plan_fixed_schedule_budget(
        total_evaluation_budget,
        type_count=type_count,
        mutation_steps_by_stage=mutation_steps_by_stage,
        stream_count=1,
    )
    dual = plan_fixed_schedule_budget(
        total_evaluation_budget,
        type_count=type_count,
        mutation_steps_by_stage=mutation_steps_by_stage,
        stream_count=2,
    )
    return SingleDualBudgetDesign(
        requested_total_evaluations=single.requested_total_evaluations,
        single_stream=single,
        two_stream=dual,
        budget_scope=(
            "the same total evaluation envelope is compared for one stream "
            "and two equal-size pilot-confirm streams"
        ),
    )


def _validate_grid_cell_counts(
    grid_cell_counts: Sequence[int],
) -> Tuple[int, ...]:
    try:
        raw_counts = tuple(grid_cell_counts)
    except TypeError as error:
        raise ParetoSMCDesignError(
            "grid_cell_counts must be a finite sequence."
        ) from error
    if not raw_counts:
        raise ParetoSMCDesignError(
            "grid_cell_counts must contain at least one objective axis."
        )
    return tuple(
        _validate_positive_integer(
            value,
            f"grid_cell_counts[{index}]",
        )
        for index, value in enumerate(raw_counts)
    )


def _validate_directions(
    reference_directions: Sequence[Sequence[float]],
    *,
    dimension: int,
) -> Tuple[Tuple[float, ...], ...]:
    try:
        raw_directions = tuple(reference_directions)
    except TypeError as error:
        raise ParetoSMCDesignError(
            "reference_directions must be a finite sequence."
        ) from error
    if not raw_directions:
        raise ParetoSMCDesignError(
            "reference_directions must contain at least one direction."
        )
    directions = []
    for index, raw_direction in enumerate(raw_directions):
        try:
            direction = tuple(float(value) for value in raw_direction)
        except (TypeError, ValueError) as error:
            raise ParetoSMCDesignError(
                f"reference_directions[{index}] must be numeric."
            ) from error
        if len(direction) != dimension:
            raise ParetoSMCDesignError(
                f"reference_directions[{index}] has the wrong dimension."
            )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in direction
        ):
            raise ParetoSMCDesignError(
                "Reference-direction components must be finite and "
                "strictly positive."
            )
        if not math.isclose(
            sum(direction),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ParetoSMCDesignError(
                "Every reference direction must sum to one."
            )
        directions.append(direction)
    resolved = tuple(directions)
    if len(set(resolved)) != len(resolved):
        raise ParetoSMCDesignError(
            "reference_directions must be unique."
        )
    return resolved


def _validate_reference_cells(
    reference_cells: Sequence[Sequence[int]],
    *,
    grid_cell_counts: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], ...]:
    try:
        raw_cells = tuple(reference_cells)
    except TypeError as error:
        raise ParetoSMCDesignError(
            "reference_cells must be a finite sequence."
        ) from error
    if not raw_cells:
        raise ParetoSMCDesignError(
            "reference_cells must contain at least one frozen cell."
        )
    cells = []
    for cell_index, raw_cell in enumerate(raw_cells):
        try:
            cell = tuple(raw_cell)
        except TypeError as error:
            raise ParetoSMCDesignError(
                f"reference_cells[{cell_index}] must be a sequence."
            ) from error
        if len(cell) != len(grid_cell_counts):
            raise ParetoSMCDesignError(
                f"reference_cells[{cell_index}] has the wrong dimension."
            )
        validated = []
        for axis, (coordinate, axis_count) in enumerate(
            zip(cell, grid_cell_counts)
        ):
            value = _validate_nonnegative_integer(
                coordinate,
                f"reference_cells[{cell_index}][{axis}]",
            )
            if value >= axis_count:
                raise ParetoSMCDesignError(
                    f"reference_cells[{cell_index}][{axis}] leaves the "
                    "frozen grid."
                )
            validated.append(value)
        cells.append(tuple(validated))
    return tuple(cells)


def build_frozen_complexity_ledger(
    *,
    reference_directions: Sequence[Sequence[float]],
    grid_cell_counts: Sequence[int],
    reference_cells: Sequence[Sequence[int]],
    max_type_count: int,
    max_grid_cell_count: int,
    archive_cap: int | None = None,
) -> FrozenComplexityLedger:
    """Ledger the finite cost of a pre-frozen typed cell design.

    ``max_grid_cell_count`` gates the full Cartesian grid capacity, not merely
    the reference cells that happen to be supplied.  The archive-cap floor is
    a cardinality necessity for retaining one representative per frozen
    reference cell; it is not a compression theorem and is not sufficient for
    metric preservation by an arbitrary archive policy.
    """

    grid_counts = _validate_grid_cell_counts(grid_cell_counts)
    dimension = len(grid_counts)
    directions = _validate_directions(
        reference_directions,
        dimension=dimension,
    )
    cells = _validate_reference_cells(
        reference_cells,
        grid_cell_counts=grid_counts,
    )
    type_limit = _validate_positive_integer(
        max_type_count,
        "max_type_count",
    )
    cell_limit = _validate_positive_integer(
        max_grid_cell_count,
        "max_grid_cell_count",
    )
    if archive_cap is None:
        resolved_archive_cap = None
    else:
        resolved_archive_cap = _validate_nonnegative_integer(
            archive_cap,
            "archive_cap",
        )

    unique_cells = tuple(sorted(set(cells)))
    type_count = len(directions)
    grid_capacity = math.prod(grid_counts)
    unique_cell_count = len(unique_cells)
    archive_floor = unique_cell_count
    type_gate_pass = type_count <= type_limit
    cell_gate_pass = grid_capacity <= cell_limit
    if resolved_archive_cap is None:
        archive_gate = "NOT_EVALUATED"
    else:
        archive_gate = (
            "PASS"
            if resolved_archive_cap >= archive_floor
            else "FAIL"
        )

    frozen_payload = {
        "reference_directions": directions,
        "grid_cell_counts": grid_counts,
        "unique_reference_cells": unique_cells,
    }
    encoded = json.dumps(
        frozen_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    frozen_hash = hashlib.sha256(encoded).hexdigest()

    return FrozenComplexityLedger(
        objective_dimension=dimension,
        reference_directions=directions,
        type_count=type_count,
        unique_direction_count=len(set(directions)),
        grid_cell_counts=grid_counts,
        grid_cell_capacity=grid_capacity,
        supplied_reference_cell_count=len(cells),
        unique_reference_cells=unique_cells,
        unique_reference_cell_count=unique_cell_count,
        duplicate_reference_cell_count=len(cells) - unique_cell_count,
        pilot_type_cell_observable_count=type_count * unique_cell_count,
        confirm_cell_observable_count=unique_cell_count,
        minimum_archive_cap_for_one_per_reference_cell=archive_floor,
        declared_archive_cap=resolved_archive_cap,
        archive_cap_cardinality_gate=archive_gate,
        max_type_count=type_limit,
        max_grid_cell_count=cell_limit,
        max_type_gate="PASS" if type_gate_pass else "FAIL",
        max_cell_gate="PASS" if cell_gate_pass else "FAIL",
        max_type_max_cell_gate=(
            "PASS" if type_gate_pass and cell_gate_pass else "FAIL"
        ),
        frozen_structure_sha256=frozen_hash,
        unknown_pareto_front_coverage_claimed=False,
        fixed_size_archive_metric_preservation_claimed=False,
        archive_cap_floor_scope=(
            "necessary cardinality for one representative per frozen "
            "reference cell; not sufficient for arbitrary archive "
            "compression or metric preservation"
        ),
    )


__all__ = [
    "FixedScheduleBudgetPlan",
    "FrozenComplexityLedger",
    "ParetoSMCDesignError",
    "RefreshScheduleDesign",
    "SingleDualBudgetDesign",
    "StageRefreshDesign",
    "build_frozen_complexity_ledger",
    "design_stagewise_global_refresh",
    "plan_fixed_schedule_budget",
    "plan_single_and_dual_stream_budgets",
]
