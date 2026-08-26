from __future__ import annotations

"""Source-bound finite-step Pareto-cell coverage certificates.

The adaptive Pareto-SMC core is not used as a finite-particle certificate.
Instead, each predeclared objective cell receives independent terminal probe
chains.  The target is a frozen cell-penalized augmented-Tchebycheff law with
uniform fixed-zero-tour base measure.  A source-bound lower certificate for
the base cell mass, a uniform-refresh Metropolis component, and independent
random streams yield explicit target-mass, Doeblin, finite-step TV,
cell-miss, and empirical-radius bounds.

This module deliberately does not infer any theorem constant from ESS,
acceptance rate, archive size, or an observed successful run.
"""

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .contracts import ClaimLevel
from .evaluation import evaluation_count, remaining_evaluations
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .moves import random_tour, sample_two_opt_indices, two_opt_at
from .sampler import Diagnostic, OptimizationResult
from .types import ObjectiveVector, Tour

Cell = Tuple[int, ...]


class CellCertificationContractError(ValueError):
    """Raised when a source-bound cell contract is not well formed."""


@dataclass(frozen=True)
class CertifiedCellType:
    """A predeclared cell target and its source-bound constants.

    ``base_cell_mass_lower_bound`` is a theorem/certificate input satisfying
    ``m0(f^{-1}(cell)) >= kappa`` for the uniform fixed-zero-tour law ``m0``.
    It is not estimated from the current run.  ``base_mass_proof_sha256`` binds
    that input to an external or exact-enumeration proof artifact.
    """

    cell: Cell
    reference_direction: ObjectiveVector
    base_cell_mass_lower_bound: float
    base_mass_proof_sha256: str
    outside_cell_penalty: float
    global_refresh_probability: float
    mutation_steps: int
    particle_count: int
    failure_budget: float


@dataclass(frozen=True)
class CellTypePlan:
    """Closed-form constants in the finite-step coverage proof."""

    cell: Cell
    base_cell_mass_lower_bound: float
    target_cell_mass_lower_bound: float
    doeblin_minorization: float
    mutation_tv_radius: float
    endpoint_cell_hit_lower_bound: float
    cell_miss_probability_bound: float
    cellwise_empirical_radius: float
    finite_particle_mse_constant: float
    required_particle_count: int
    required_mutation_steps: int
    plan_pass: bool


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CellCertificationContractError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _validate_probability(
    value: float,
    label: str,
    *,
    allow_zero: bool = False,
    open_upper: bool = False,
) -> float:
    value = float(value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    upper_ok = value < 1.0 if open_upper else value <= 1.0
    if not math.isfinite(value) or not lower_ok or not upper_ok:
        left = "[" if allow_zero else "("
        right = ")" if open_upper else "]"
        raise CellCertificationContractError(
            f"{label} must lie in {left}0, 1{right}."
        )
    return value


def augmented_tchebycheff_energy(
    objective: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    direction: Sequence[float],
    rho: float,
) -> float:
    """Frozen-box augmented Tchebycheff energy in ``[0, 1 + rho]``."""

    dimension = len(objective)
    if dimension == 0 or not (
        len(lower) == len(upper) == len(direction) == dimension
    ):
        raise CellCertificationContractError(
            "objective, lower, upper, and direction must have one common positive dimension."
        )
    rho = float(rho)
    if not math.isfinite(rho) or rho <= 0.0:
        raise CellCertificationContractError("rho must be finite and positive.")
    direction_values = tuple(float(value) for value in direction)
    if any(not math.isfinite(value) or value <= 0.0 for value in direction_values):
        raise CellCertificationContractError(
            "reference directions must be finite and strictly positive."
        )
    if not math.isclose(sum(direction_values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CellCertificationContractError("reference directions must sum to one.")
    normalized: List[float] = []
    for coordinate, (raw, low_raw, high_raw) in enumerate(
        zip(objective, lower, upper)
    ):
        value = float(raw)
        low = float(low_raw)
        high = float(high_raw)
        if not all(math.isfinite(item) for item in (value, low, high)):
            raise CellCertificationContractError("objective-box values must be finite.")
        if high <= low:
            raise CellCertificationContractError("each objective-box span must be positive.")
        tolerance = 1e-12 * max(1.0, abs(low), abs(high))
        if value < low - tolerance or value > high + tolerance:
            raise CellCertificationContractError(
                f"objective coordinate {coordinate} lies outside the frozen box."
            )
        z = min(1.0, max(0.0, (value - low) / (high - low)))
        normalized.append(z)
    weighted = tuple(
        weight * value for weight, value in zip(direction_values, normalized)
    )
    result = max(weighted) + rho * sum(weighted)
    if result < -1e-12 or result > 1.0 + rho + 1e-12:
        raise RuntimeError("augmented Tchebycheff range invariant failed.")
    return min(1.0 + rho, max(0.0, result))


def original_cell_index(
    objective: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    widths: Sequence[float],
) -> Cell:
    """Index an original-unit point using half-open cells and a closed upper edge."""

    if not (len(objective) == len(lower) == len(upper) == len(widths)):
        raise CellCertificationContractError("Cell coordinates have inconsistent dimensions.")
    if not objective:
        raise CellCertificationContractError("Cell coordinates must have positive dimension.")
    result: List[int] = []
    for coordinate, (raw, low_raw, high_raw, width_raw) in enumerate(
        zip(objective, lower, upper, widths)
    ):
        value = float(raw)
        low = float(low_raw)
        high = float(high_raw)
        width = float(width_raw)
        if not all(math.isfinite(v) for v in (value, low, high, width)):
            raise CellCertificationContractError("Cell coordinates must be finite.")
        if high <= low or width <= 0.0 or width > high - low:
            raise CellCertificationContractError(
                "Box spans must be positive and each width must lie in (0, span]."
            )
        tolerance = 1e-12 * max(1.0, abs(low), abs(high))
        if value < low - tolerance or value > high + tolerance:
            raise CellCertificationContractError(
                f"objective coordinate {coordinate}={value!r} is outside "
                f"the frozen metric box [{low!r}, {high!r}]."
            )
        count = max(1, int(math.ceil((high - low) / width)))
        if value >= high - tolerance:
            result.append(count - 1)
        else:
            result.append(
                min(count - 1, max(0, int(math.floor((value - low) / width))))
            )
    return tuple(result)


def original_cell_index_or_none(
    objective: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    widths: Sequence[float],
) -> Optional[Cell]:
    """Return the metric-cell index, or ``None`` outside the metric box.

    The target safety box and the metric box are distinct objects.  A feasible
    state may be valid for target normalization while lying outside the tighter
    metric box used for an IGD/HV certificate.  Such a state belongs to no
    certified metric cell and must not be clipped into a boundary cell.
    """

    if not (len(objective) == len(lower) == len(upper) == len(widths)):
        raise CellCertificationContractError(
            "Cell coordinates have inconsistent dimensions."
        )
    for raw, low_raw, high_raw in zip(objective, lower, upper):
        value = float(raw)
        low = float(low_raw)
        high = float(high_raw)
        if not all(math.isfinite(item) for item in (value, low, high)):
            raise CellCertificationContractError(
                "Metric-box coordinates must be finite."
            )
        tolerance = 1e-12 * max(1.0, abs(low), abs(high))
        if value < low - tolerance or value > high + tolerance:
            return None
    return original_cell_index(
        objective,
        lower=lower,
        upper=upper,
        widths=widths,
    )


def target_cell_mass_lower_bound(
    *,
    base_cell_mass_lower_bound: float,
    beta: float,
    base_energy_upper: float,
    outside_cell_penalty: float,
) -> float:
    """Lower-bound the cell mass of a cell-penalized Gibbs target.

    Let ``m0(C) >= kappa``, ``0 <= U <= Umax``, and
    ``V = U + lambda * 1_{C^c}``.  Then

    ``pi(C) >= kappa exp(-beta Umax) /
      [kappa exp(-beta Umax) + (1-kappa) exp(-beta lambda)]``.
    """

    kappa = _validate_probability(
        base_cell_mass_lower_bound,
        "base_cell_mass_lower_bound",
    )
    beta = float(beta)
    u_max = float(base_energy_upper)
    penalty = float(outside_cell_penalty)
    if not math.isfinite(beta) or beta < 0.0:
        raise CellCertificationContractError("beta must be finite and nonnegative.")
    if not math.isfinite(u_max) or u_max < 0.0:
        raise CellCertificationContractError(
            "base_energy_upper must be finite and nonnegative."
        )
    if not math.isfinite(penalty) or penalty < 0.0:
        raise CellCertificationContractError(
            "outside_cell_penalty must be finite and nonnegative."
        )
    # Evaluate the ratio in log space.  The direct numerator/denominator
    # formula may underflow to 0/0 for a mathematically valid large-beta
    # contract, which must not be silently converted into a false certificate.
    log_inside = math.log(kappa) - beta * u_max
    if kappa == 1.0:
        return 1.0
    log_outside = math.log1p(-kappa) - beta * penalty
    difference = log_outside - log_inside
    if difference >= 0.0:
        exp_negative = math.exp(-difference)
        return exp_negative / (1.0 + exp_negative)
    exp_positive = math.exp(difference)
    return 1.0 / (1.0 + exp_positive)


def doeblin_minorization_constant(
    *,
    global_refresh_probability: float,
    beta: float,
    potential_range_upper: float,
) -> float:
    """Minorization for local MH mixed with uniform independence MH."""

    gamma = _validate_probability(
        global_refresh_probability,
        "global_refresh_probability",
    )
    beta = float(beta)
    v_max = float(potential_range_upper)
    if not math.isfinite(beta) or beta < 0.0:
        raise CellCertificationContractError("beta must be finite and nonnegative.")
    if not math.isfinite(v_max) or v_max < 0.0:
        raise CellCertificationContractError(
            "potential_range_upper must be finite and nonnegative."
        )
    return gamma * math.exp(-beta * v_max)


def mixing_tv_radius(minorization: float, mutation_steps: int) -> float:
    """Worst-start total-variation radius after a finite number of transitions."""

    epsilon = float(minorization)
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise CellCertificationContractError("minorization must lie in (0, 1].")
    if (
        isinstance(mutation_steps, bool)
        or not isinstance(mutation_steps, int)
        or mutation_steps < 0
    ):
        raise CellCertificationContractError(
            "mutation_steps must be a nonnegative integer."
        )
    return (1.0 - epsilon) ** mutation_steps


def required_mutation_steps(minorization: float, target_radius: float) -> int:
    """Smallest integer ``t`` with ``(1-epsilon)^t <= target_radius``."""

    epsilon = float(minorization)
    radius = float(target_radius)
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise CellCertificationContractError("minorization must lie in (0, 1].")
    if not math.isfinite(radius) or radius <= 0.0 or radius >= 1.0:
        raise CellCertificationContractError("target_radius must lie in (0, 1).")
    if epsilon == 1.0:
        return 1
    return max(1, int(math.ceil(math.log(radius) / math.log1p(-epsilon))))


def required_particles_for_miss_bound(
    hit_lower_bound: float,
    failure_budget: float,
) -> int:
    """Smallest ``m`` with ``(1-hit_lower_bound)^m <= failure_budget``."""

    hit = float(hit_lower_bound)
    delta = _validate_probability(failure_budget, "failure_budget")
    if not math.isfinite(hit) or hit <= 0.0 or hit > 1.0:
        raise CellCertificationContractError("hit_lower_bound must lie in (0, 1].")
    if hit == 1.0:
        return 1
    return max(1, int(math.ceil(math.log(delta) / math.log1p(-hit))))


def plan_cell_type(
    contract: CertifiedCellType,
    *,
    beta: float,
    chebyshev_rho: float,
) -> CellTypePlan:
    """Compute every constant in the finite-step cell certificate."""

    if (
        isinstance(contract.particle_count, bool)
        or not isinstance(contract.particle_count, int)
        or contract.particle_count <= 0
    ):
        raise CellCertificationContractError(
            "particle_count must be a positive integer."
        )
    u_max = 1.0 + float(chebyshev_rho)
    p_lower = target_cell_mass_lower_bound(
        base_cell_mass_lower_bound=contract.base_cell_mass_lower_bound,
        beta=beta,
        base_energy_upper=u_max,
        outside_cell_penalty=contract.outside_cell_penalty,
    )
    epsilon = doeblin_minorization_constant(
        global_refresh_probability=contract.global_refresh_probability,
        beta=beta,
        potential_range_upper=u_max + contract.outside_cell_penalty,
    )
    radius = mixing_tv_radius(epsilon, contract.mutation_steps)
    hit_lower = max(0.0, p_lower - radius)
    miss_bound = (
        1.0
        if hit_lower <= 0.0
        else (1.0 - hit_lower) ** contract.particle_count
    )
    empirical_radius = radius + math.sqrt(
        math.log(2.0 / contract.failure_budget)
        / (2.0 * contract.particle_count)
    )
    mse_constant = 0.25 + contract.particle_count * radius * radius
    if p_lower <= 0.0:
        raise CellCertificationContractError(
            "The floating-point target-cell lower bound vanished; the "
            "certificate cannot be represented at this precision."
        )
    required_steps = required_mutation_steps(epsilon, 0.5 * p_lower)
    required_particles = (
        math.inf
        if hit_lower <= 0.0
        else required_particles_for_miss_bound(
            hit_lower,
            contract.failure_budget,
        )
    )
    # The direct miss theorem only requires tau < p_lower and a sufficiently
    # large particle count.  ``required_steps`` is the canonical stronger
    # recommendation tau <= p_lower/2; it is not a necessary gate condition.
    plan_pass = bool(
        hit_lower > 0.0
        and contract.particle_count >= required_particles
        and miss_bound <= contract.failure_budget
    )
    return CellTypePlan(
        cell=contract.cell,
        base_cell_mass_lower_bound=contract.base_cell_mass_lower_bound,
        target_cell_mass_lower_bound=p_lower,
        doeblin_minorization=epsilon,
        mutation_tv_radius=radius,
        endpoint_cell_hit_lower_bound=hit_lower,
        cell_miss_probability_bound=miss_bound,
        cellwise_empirical_radius=empirical_radius,
        finite_particle_mse_constant=mse_constant,
        required_particle_count=(
            int(required_particles)
            if math.isfinite(required_particles)
            else 2**63 - 1
        ),
        required_mutation_steps=required_steps,
        plan_pass=plan_pass,
    )


class CellCertifiedParetoSampler:
    """Independent finite-step probes for source-bound objective cells.

    The branch is separate from adaptive SMC.  Probe chains do not interact or
    resample.  Conditional on independent ideal random streams, the direct
    miss bound and the Hoeffding radius therefore apply literally.
    """

    contract_name = "source_bound_cell_probe_v2"
    implementation_version = "0.2.0"

    def __init__(
        self,
        instance: MultiObjectiveTSPInstance,
        *,
        cell_types: Sequence[CertifiedCellType],
        objective_lower_bounds: Sequence[float],
        objective_upper_bounds: Sequence[float],
        cell_widths: Sequence[float],
        metric_lower_bounds: Optional[Sequence[float]] = None,
        metric_upper_bounds: Optional[Sequence[float]] = None,
        beta: float,
        chebyshev_rho: float,
        seed: int = 0,
        confidence_delta: float = 0.05,
        cell_completeness_proof_sha256: Optional[str] = None,
        objective_box_proof_sha256: Optional[str] = None,
        metric_box_proof_sha256: Optional[str] = None,
        metric_igd_p: float = 2.0,
        max_igd_bound: Optional[float] = None,
        hv_reference: Optional[Sequence[float]] = None,
        max_hv_deficit_bound: Optional[float] = None,
        archive_max_size: Optional[int] = None,
    ) -> None:
        if instance.num_cities < 4:
            raise CellCertificationContractError("At least four cities are required.")
        self.instance = instance
        self.seed = int(seed)
        self.beta = float(beta)
        self.chebyshev_rho = float(chebyshev_rho)
        self.confidence_delta = _validate_probability(
            confidence_delta,
            "confidence_delta",
            open_upper=True,
        )
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise CellCertificationContractError("beta must be finite and nonnegative.")
        if not math.isfinite(self.chebyshev_rho) or self.chebyshev_rho <= 0.0:
            raise CellCertificationContractError(
                "chebyshev_rho must be finite and positive."
            )
        # ``target_*`` contains every feasible objective vector used by the
        # frozen target.  ``metric_*`` may be tighter and only needs the
        # source-bound reference/Pareto set used by the metric theorem.
        self.target_lower = tuple(
            float(value) for value in objective_lower_bounds
        )
        self.target_upper = tuple(
            float(value) for value in objective_upper_bounds
        )
        self.metric_lower = tuple(
            float(value)
            for value in (
                objective_lower_bounds
                if metric_lower_bounds is None
                else metric_lower_bounds
            )
        )
        self.metric_upper = tuple(
            float(value)
            for value in (
                objective_upper_bounds
                if metric_upper_bounds is None
                else metric_upper_bounds
            )
        )
        self.widths = tuple(float(value) for value in cell_widths)
        self.metric_igd_p = float(metric_igd_p)
        if not (
            math.isinf(self.metric_igd_p)
            or (math.isfinite(self.metric_igd_p) and self.metric_igd_p >= 1.0)
        ):
            raise CellCertificationContractError(
                "metric_igd_p must lie in [1, infinity]."
            )
        self.max_igd_bound = (
            None if max_igd_bound is None else float(max_igd_bound)
        )
        self.max_hv_deficit_bound = (
            None
            if max_hv_deficit_bound is None
            else float(max_hv_deficit_bound)
        )
        dimension = instance.num_objectives
        if not (
            len(self.target_lower)
            == len(self.target_upper)
            == len(self.metric_lower)
            == len(self.metric_upper)
            == len(self.widths)
            == dimension
        ):
            raise CellCertificationContractError(
                "Objective-box and cell-width dimensions disagree."
            )
        for low, high in zip(self.target_lower, self.target_upper):
            if not all(math.isfinite(value) for value in (low, high)):
                raise CellCertificationContractError(
                    "Target safety-box values must be finite."
                )
            if high <= low:
                raise CellCertificationContractError(
                    "Each target safety-box span must be positive."
                )
        for low, high, width in zip(
            self.metric_lower,
            self.metric_upper,
            self.widths,
        ):
            if not all(math.isfinite(value) for value in (low, high, width)):
                raise CellCertificationContractError(
                    "Metric-box values must be finite."
                )
            if high <= low or width <= 0.0 or width > high - low:
                raise CellCertificationContractError(
                    "Each metric-box span must be positive and each width must lie in (0, span]."
                )
        if any(
            metric_low < target_low or metric_high > target_high
            for target_low, target_high, metric_low, metric_high in zip(
                self.target_lower,
                self.target_upper,
                self.metric_lower,
                self.metric_upper,
            )
        ):
            raise CellCertificationContractError(
                "The metric box must be contained in the target safety box."
            )
        self.metric_cell_counts = tuple(
            max(1, int(math.ceil((high - low) / width)))
            for low, high, width in zip(
                self.metric_lower,
                self.metric_upper,
                self.widths,
            )
        )
        self.hv_reference = tuple(
            float(value)
            for value in (
                self.metric_upper if hv_reference is None else hv_reference
            )
        )
        if len(self.hv_reference) != dimension:
            raise CellCertificationContractError(
                "hv_reference has the wrong dimension."
            )
        if any(
            not math.isfinite(reference) or reference < upper
            for reference, upper in zip(self.hv_reference, self.metric_upper)
        ):
            raise CellCertificationContractError(
                "hv_reference must be finite and coordinatewise no better than the metric-box upper endpoint."
            )
        for value, label in (
            (self.max_igd_bound, "max_igd_bound"),
            (self.max_hv_deficit_bound, "max_hv_deficit_bound"),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise CellCertificationContractError(
                    f"{label} must be finite and nonnegative when supplied."
                )
        if math.isinf(self.metric_igd_p):
            self.metric_igd_bound = max(self.widths)
        else:
            self.metric_igd_bound = sum(
                width**self.metric_igd_p for width in self.widths
            ) ** (1.0 / self.metric_igd_p)
        self.metric_hv_slab_bound = sum(
            width
            * math.prod(
                self.hv_reference[other] - self.metric_lower[other]
                for other in range(dimension)
                if other != coordinate
            )
            for coordinate, width in enumerate(self.widths)
        )
        self.metric_nonvacuity_configured = bool(
            self.max_igd_bound is not None
            and self.max_hv_deficit_bound is not None
        )
        self.metric_nonvacuity_pass = bool(
            self.metric_nonvacuity_configured
            and self.metric_igd_bound <= self.max_igd_bound + 1e-15
            and self.metric_hv_slab_bound
            <= self.max_hv_deficit_bound + 1e-15
        )
        if not cell_types:
            raise CellCertificationContractError(
                "At least one source-bound cell type is required."
            )
        self.cell_types = tuple(cell_types)
        if len({contract.cell for contract in self.cell_types}) != len(
            self.cell_types
        ):
            raise CellCertificationContractError(
                "Source-bound cell identifiers must be unique."
            )
        self._validate_contracts()
        total_failure = sum(contract.failure_budget for contract in self.cell_types)
        if total_failure > self.confidence_delta + 1e-15:
            raise CellCertificationContractError(
                "The sum of per-cell failure budgets exceeds confidence_delta."
            )
        self._counted_instance = hasattr(instance, "evaluations")
        self._counter_start = evaluation_count(instance)
        self._logical_evaluations = 0
        self.evaluation_budget = sum(
            contract.particle_count * (1 + contract.mutation_steps)
            for contract in self.cell_types
        )
        available = remaining_evaluations(instance)
        if available is not None and available < self.evaluation_budget:
            raise CellCertificationContractError(
                "The counting instance has fewer evaluations than the probe plan requires."
            )
        self.cell_completeness_proof_sha256 = self._validate_optional_hash(
            cell_completeness_proof_sha256,
            "cell_completeness_proof_sha256",
        )
        self.objective_box_proof_sha256 = self._validate_optional_hash(
            objective_box_proof_sha256,
            "objective_box_proof_sha256",
        )
        self.metric_box_proof_sha256 = self._validate_optional_hash(
            metric_box_proof_sha256,
            "metric_box_proof_sha256",
        )
        self.plans = tuple(
            plan_cell_type(
                contract,
                beta=self.beta,
                chebyshev_rho=self.chebyshev_rho,
            )
            for contract in self.cell_types
        )
        self.archive = ParetoArchive(max_size=archive_max_size)
        self.instance_sha256 = instance_sha256(instance)
        self.context_hash = _payload_sha256(self._context_payload())
        self._start = time.perf_counter()
        self._has_run = False

    @staticmethod
    def _validate_optional_hash(value: Optional[str], label: str) -> Optional[str]:
        if value is None:
            return None
        return _validate_sha256(value, label)

    def _validate_contracts(self) -> None:
        dimension = self.instance.num_objectives
        for index, contract in enumerate(self.cell_types):
            for value, label in (
                (
                    contract.base_cell_mass_lower_bound,
                    "base_cell_mass_lower_bound",
                ),
                (
                    contract.outside_cell_penalty,
                    "outside_cell_penalty",
                ),
                (
                    contract.global_refresh_probability,
                    "global_refresh_probability",
                ),
                (contract.failure_budget, "failure_budget"),
            ):
                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    raise CellCertificationContractError(
                        f"cell_types[{index}].{label} must be numeric."
                    )
            if len(contract.cell) != dimension or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in contract.cell
            ):
                raise CellCertificationContractError(
                    f"cell_types[{index}].cell is invalid."
                )
            if any(
                coordinate >= count
                for coordinate, count in zip(
                    contract.cell,
                    self.metric_cell_counts,
                )
            ):
                raise CellCertificationContractError(
                    f"cell_types[{index}].cell lies outside the metric grid."
                )
            if len(contract.reference_direction) != dimension:
                raise CellCertificationContractError(
                    f"cell_types[{index}].reference_direction has the wrong dimension."
                )
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in contract.reference_direction
            ) or not math.isclose(
                sum(contract.reference_direction),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise CellCertificationContractError(
                    f"cell_types[{index}].reference_direction must be strictly positive and sum to one."
                )
            _validate_probability(
                contract.base_cell_mass_lower_bound,
                f"cell_types[{index}].base_cell_mass_lower_bound",
            )
            _validate_sha256(
                contract.base_mass_proof_sha256,
                f"cell_types[{index}].base_mass_proof_sha256",
            )
            _validate_probability(
                contract.global_refresh_probability,
                f"cell_types[{index}].global_refresh_probability",
            )
            _validate_probability(
                contract.failure_budget,
                f"cell_types[{index}].failure_budget",
                open_upper=True,
            )
            if contract.outside_cell_penalty < 0.0 or not math.isfinite(
                contract.outside_cell_penalty
            ):
                raise CellCertificationContractError(
                    "outside_cell_penalty must be finite and nonnegative."
                )
            if (
                isinstance(contract.mutation_steps, bool)
                or not isinstance(contract.mutation_steps, int)
                or contract.mutation_steps < 0
                or isinstance(contract.particle_count, bool)
                or not isinstance(contract.particle_count, int)
                or contract.particle_count <= 0
            ):
                raise CellCertificationContractError(
                    "Mutation steps and particle count must be integers with "
                    "mutation_steps >= 0 and particle_count > 0."
                )

    def run(self) -> OptimizationResult:
        if self._has_run:
            raise RuntimeError("CellCertifiedParetoSampler instances are single-use.")
        self._has_run = True
        endpoints: List[Tour] = []
        endpoint_objectives: List[ObjectiveVector] = []
        terminal_cell_representatives: Dict[Cell, Dict[str, object]] = {}
        type_ledgers: List[Dict[str, object]] = []
        accepted_total = 0
        attempts_total = 0

        for type_index, (contract, plan) in enumerate(
            zip(self.cell_types, self.plans)
        ):
            hits = 0
            chain_records: List[Dict[str, object]] = []
            for chain_index in range(contract.particle_count):
                chain_seed = self._derived_chain_seed(type_index, chain_index)
                rng = random.Random(chain_seed)
                tour = random_tour(self.instance.num_cities, rng)
                objective = self._evaluate(tour)
                energy = self._penalized_energy(objective, contract)
                chain_accepts = 0
                mutations: List[Dict[str, object]] = []
                for step in range(contract.mutation_steps):
                    use_global = (
                        rng.random() < contract.global_refresh_probability
                    )
                    if use_global:
                        proposed_tour = random_tour(self.instance.num_cities, rng)
                        proposal_kind = "uniform_fixed_zero_independence"
                    else:
                        i, j = sample_two_opt_indices(
                            self.instance.num_cities,
                            rng,
                        )
                        proposed_tour = two_opt_at(tour, i, j)
                        proposal_kind = "uniform_symmetric_two_opt"
                    proposed_objective = self._evaluate(proposed_tour)
                    proposed_energy = self._penalized_energy(
                        proposed_objective,
                        contract,
                    )
                    # Both proposals are symmetric with respect to the uniform
                    # fixed-zero-tour base measure.
                    log_alpha = min(
                        0.0,
                        -self.beta * (proposed_energy - energy),
                    )
                    draw = rng.random()
                    log_uniform = -math.inf if draw == 0.0 else math.log(draw)
                    accepted = log_uniform < log_alpha
                    attempts_total += 1
                    if accepted:
                        tour = proposed_tour
                        objective = proposed_objective
                        energy = proposed_energy
                        chain_accepts += 1
                        accepted_total += 1
                    mutations.append(
                        {
                            "step": step,
                            "proposal_kind": proposal_kind,
                            "proposed_tour": proposed_tour,
                            "proposed_objectives": proposed_objective,
                            "log_alpha": log_alpha,
                            "accepted": accepted,
                        }
                    )
                cell = original_cell_index_or_none(
                    objective,
                    lower=self.metric_lower,
                    upper=self.metric_upper,
                    widths=self.widths,
                )
                hit = cell == contract.cell
                hits += int(hit)
                endpoints.append(tour)
                endpoint_objectives.append(objective)
                self.archive.update((ArchiveEntry(tour, objective),))
                if cell is not None:
                    terminal_cell_representatives.setdefault(
                        cell,
                        {
                            "cell": cell,
                            "tour": tour,
                            "objectives": objective,
                            "type_index": type_index,
                            "chain_index": chain_index,
                        },
                    )
                chain_records.append(
                    {
                        "chain_index": chain_index,
                        "chain_seed": chain_seed,
                        "endpoint_tour": tour,
                        "endpoint_objectives": objective,
                        "endpoint_cell": cell,
                        "declared_cell_hit": hit,
                        "accepted_mutations": chain_accepts,
                        "mutation_steps": contract.mutation_steps,
                        "mutation_trace_hash": _payload_sha256(mutations),
                    }
                )
            type_ledgers.append(
                {
                    "type_index": type_index,
                    "cell": contract.cell,
                    "reference_direction": contract.reference_direction,
                    "base_cell_mass_lower_bound": contract.base_cell_mass_lower_bound,
                    "base_mass_proof_sha256": contract.base_mass_proof_sha256,
                    "particle_count": contract.particle_count,
                    "mutation_steps": contract.mutation_steps,
                    "observed_hits": hits,
                    "observed_hit_fraction": hits / contract.particle_count,
                    "target_cell_mass_lower_bound": plan.target_cell_mass_lower_bound,
                    "doeblin_minorization": plan.doeblin_minorization,
                    "mutation_tv_radius": plan.mutation_tv_radius,
                    "endpoint_cell_hit_lower_bound": plan.endpoint_cell_hit_lower_bound,
                    "cell_miss_probability_bound": plan.cell_miss_probability_bound,
                    "cellwise_empirical_radius": plan.cellwise_empirical_radius,
                    "finite_particle_mse_constant": plan.finite_particle_mse_constant,
                    "finite_particle_mse_statement": "E[(p_hat-pi(C))^2] <= B_j^(2)/m_j",
                    "required_particle_count": plan.required_particle_count,
                    "required_mutation_steps": plan.required_mutation_steps,
                    "plan_pass": plan.plan_pass,
                    "chains": tuple(chain_records),
                }
            )

        if self._evaluations_used() != self.evaluation_budget:
            raise RuntimeError(
                "Probe runner did not consume its exact evaluation budget."
            )
        total_failure_bound = min(
            1.0,
            sum(plan.cell_miss_probability_bound for plan in self.plans),
        )
        design_pass = bool(
            all(plan.plan_pass for plan in self.plans)
            and total_failure_bound <= self.confidence_delta
            and self.cell_completeness_proof_sha256 is not None
            and self.objective_box_proof_sha256 is not None
            and self.metric_box_proof_sha256 is not None
            and self.metric_nonvacuity_pass
        )
        # Runtime verifies contract shape and hashes, not the mathematical truth
        # of external proof artifacts.  Exact audit may upgrade this level.
        claim_level = (
            ClaimLevel.PARETO_CELL_SOURCE_BOUND.value
            if design_pass
            else ClaimLevel.PARETO_SMC_MECHANICAL.value
        )
        observed_all_cells_hit = all(
            ledger["observed_hits"] > 0 for ledger in type_ledgers
        )
        elapsed = time.perf_counter() - self._start
        diagnostic = Diagnostic(
            iteration=self._evaluations_used(),
            temperature=(math.inf if self.beta == 0.0 else 1.0 / self.beta),
            acceptance_rate=accepted_total / max(1, attempts_total),
            archive_size=len(self.archive),
            hypervolume_2d=(
                self.archive.hypervolume_2d(reference=self.hv_reference)
                if self.instance.num_objectives == 2
                else 0.0
            ),
            empirical_energy=0.0,
            positive_archive_jump=0.0,
            front=tuple(entry.objectives for entry in self.archive.entries),
            elapsed_seconds=elapsed,
            replacement_attempts=attempts_total,
            accepted_replacements=accepted_total,
            rejected_replacements=attempts_total - accepted_total,
            rejection_rate=(attempts_total - accepted_total)
            / max(1, attempts_total),
        )
        metadata: Dict[str, object] = {
            "algorithm_contract": self.contract_name,
            "implementation_version": self.implementation_version,
            "claim_level": claim_level,
            "context_hash": self.context_hash,
            "instance_sha256": self.instance_sha256,
            "target_safety_lower_bounds": self.target_lower,
            "target_safety_upper_bounds": self.target_upper,
            "metric_lower_bounds": self.metric_lower,
            "metric_upper_bounds": self.metric_upper,
            "cell_widths_original_units": self.widths,
            "beta": self.beta,
            "chebyshev_rho": self.chebyshev_rho,
            "cell_completeness_proof_sha256": (
                self.cell_completeness_proof_sha256
            ),
            "objective_box_proof_sha256": self.objective_box_proof_sha256,
            "metric_box_proof_sha256": self.metric_box_proof_sha256,
            "metric_igd_p": self.metric_igd_p,
            "metric_igd_bound": self.metric_igd_bound,
            "metric_igd_tolerance": self.max_igd_bound,
            "metric_hv_reference": self.hv_reference,
            "metric_hv_slab_bound": self.metric_hv_slab_bound,
            "metric_hv_deficit_tolerance": self.max_hv_deficit_bound,
            "metric_nonvacuity_configured": self.metric_nonvacuity_configured,
            "metric_nonvacuity_gate": (
                "PASS" if self.metric_nonvacuity_pass else "FAIL"
            ),
            "independent_terminal_probe_chains": True,
            "ideal_random_stream_assumption": True,
            "resampling_used": False,
            "finite_particle_radius_source": (
                "doeblin_contraction_plus_independent_hoeffding"
            ),
            "global_refresh_kernel": "uniform_fixed_zero_independence_mh",
            "local_kernel": "uniform_symmetric_two_opt_mh",
            "total_failure_probability_bound": total_failure_bound,
            "requested_confidence_delta": self.confidence_delta,
            "coverage_probability_lower_bound": 1.0 - total_failure_bound,
            "scientific_design_gate": "PASS" if design_pass else "FAIL",
            "proof_truth_verified_by_runtime": False,
            "observed_all_declared_cells_hit": observed_all_cells_hit,
            "terminal_cell_ledger": tuple(
                terminal_cell_representatives[cell]
                for cell in sorted(terminal_cell_representatives)
            ),
            "terminal_cell_ledger_untruncated": True,
            "type_ledgers": tuple(type_ledgers),
            "type_ledger_hash": _payload_sha256(type_ledgers),
            "probe_evaluation_budget": self.evaluation_budget,
            "probe_evaluations_used": self._evaluations_used(),
            "evaluation_identity": "sum_j m_j*(1+t_j)",
        }
        return OptimizationResult(
            particles=tuple(endpoints),
            objectives=tuple(endpoint_objectives),
            archive=self.archive,
            diagnostics=(diagnostic,),
            metadata=metadata,
        )

    def _context_payload(self) -> Mapping[str, object]:
        return {
            "algorithm_contract": self.contract_name,
            "instance_sha256": instance_sha256(self.instance),
            "target_lower": self.target_lower,
            "target_upper": self.target_upper,
            "metric_lower": self.metric_lower,
            "metric_upper": self.metric_upper,
            "widths": self.widths,
            "beta": self.beta,
            "chebyshev_rho": self.chebyshev_rho,
            "cell_completeness_proof_sha256": (
                self.cell_completeness_proof_sha256
            ),
            "target_safety_box_proof_sha256": (
                self.objective_box_proof_sha256
            ),
            "metric_box_proof_sha256": self.metric_box_proof_sha256,
            "metric_igd_p": self.metric_igd_p,
            "max_igd_bound": self.max_igd_bound,
            "hv_reference": self.hv_reference,
            "max_hv_deficit_bound": self.max_hv_deficit_bound,
            "cell_types": [
                {
                    "cell": contract.cell,
                    "reference_direction": contract.reference_direction,
                    "base_cell_mass_lower_bound": (
                        contract.base_cell_mass_lower_bound
                    ),
                    "base_mass_proof_sha256": contract.base_mass_proof_sha256,
                    "outside_cell_penalty": contract.outside_cell_penalty,
                    "global_refresh_probability": (
                        contract.global_refresh_probability
                    ),
                    "mutation_steps": contract.mutation_steps,
                    "particle_count": contract.particle_count,
                    "failure_budget": contract.failure_budget,
                }
                for contract in self.cell_types
            ],
        }

    def _derived_chain_seed(self, type_index: int, chain_index: int) -> int:
        payload = (
            f"{self.seed}:{type_index}:{chain_index}:{self.context_hash}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _penalized_energy(
        self,
        objective: ObjectiveVector,
        contract: CertifiedCellType,
    ) -> float:
        base = augmented_tchebycheff_energy(
            objective,
            lower=self.target_lower,
            upper=self.target_upper,
            direction=contract.reference_direction,
            rho=self.chebyshev_rho,
        )
        cell = original_cell_index_or_none(
            objective,
            lower=self.metric_lower,
            upper=self.metric_upper,
            widths=self.widths,
        )
        return base + (
            0.0 if cell == contract.cell else contract.outside_cell_penalty
        )

    def _evaluate(self, tour: Tour) -> ObjectiveVector:
        objective = tuple(float(value) for value in self.instance.evaluate(tour))
        if not self._counted_instance:
            self._logical_evaluations += 1
        # Fail closed on the source-bound target safety box.  The metric box
        # may be tighter; points outside it are valid target states but belong
        # to no certified metric cell.
        augmented_tchebycheff_energy(
            objective,
            lower=self.target_lower,
            upper=self.target_upper,
            direction=self.cell_types[0].reference_direction,
            rho=self.chebyshev_rho,
        )
        return objective

    def _evaluations_used(self) -> int:
        if self._counted_instance:
            return evaluation_count(self.instance) - self._counter_start
        return self._logical_evaluations
