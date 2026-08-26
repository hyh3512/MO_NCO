from __future__ import annotations

"""Finite-state reference audit for the strict typed Metropolis kernel.

The production optimizer is deliberately not called here.  Instead, this
module enumerates every fixed-zero tour on a tiny instance and constructs the
*ideal real-arithmetic* transition matrix from the mathematical definition. It
therefore acts as an independent executable oracle for:

* row stochasticity and proposal-graph connectivity;
* Gibbs detailed balance and stationarity;
* ordinary and absolute spectral gaps;
* the random-scan product-chain gap formula; and
* a falsifiable, finite-grid temperature/distortion/mixing gate.

It is not a proof that Python's finite 53-bit pseudorandom grid implements the
same transition probabilities exactly.  Moreover, the current Hamiltonian is
a sum of coordinate-typed energies and hence factorizes: this audits a
random-scan product of non-interacting typed chains, not an MMD/mean-field IPS.

Two kernels are reported.  ``evaluation_clock`` includes the explicit lazy
identity mixture and is the production budget clock: both active proposals and
lazy identity transitions perform exactly one objective evaluation.
``active_proposal_kernel`` removes that identity mixture and is retained only
as a diagnostic contrast; it does not govern the fixed-evaluation output.
"""

import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .instance import MultiObjectiveTSPInstance
from .moves import random_tour, two_opt_at
from .potential import ScalarArchivePotential
from .types import ObjectiveVector, Tour


SCHEMA = "enumerable_typed_mh_audit_v1"
_MATRIX_TOLERANCE = 1e-12
_LOG_MIN_SUBNORMAL = math.log(float.fromhex("0x0.0000000000001p-1022"))


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError(
            "The enumerable spectral audit requires NumPy. "
            "Install the project with the 'accelerate' extra."
        ) from exc
    return np


def _enumerate_tours(num_cities: int, max_states: int) -> tuple[Tour, ...]:
    if num_cities < 4:
        raise ValueError("The strict 2-opt audit requires at least four cities.")
    if max_states <= 0:
        raise ValueError("max_states must be positive.")
    state_count = math.factorial(num_cities - 1)
    if state_count > max_states:
        raise ValueError(
            f"Refusing to enumerate {(num_cities - 1)}!={state_count} tours; "
            f"increase max_states (currently {max_states}) explicitly."
        )
    return tuple(
        (0,) + tuple(tail)
        for tail in itertools.permutations(range(1, num_cities))
    )


def _two_opt_pairs(num_cities: int) -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(1, num_cities), 2))


def _fixed_context(
    instance: MultiObjectiveTSPInstance,
    *,
    num_particles: int,
    context_seed: int,
    chebyshev_rho: float,
    minimum_scale_fraction: float,
    absolute_scale_floor: float,
) -> dict[str, Any]:
    if num_particles <= 0:
        raise ValueError("num_particles must be positive.")
    if not math.isfinite(chebyshev_rho) or chebyshev_rho < 0.0:
        raise ValueError("chebyshev_rho must be finite and nonnegative.")
    if not math.isfinite(minimum_scale_fraction) or minimum_scale_fraction < 0.0:
        raise ValueError("minimum_scale_fraction must be finite and nonnegative.")
    if not math.isfinite(absolute_scale_floor) or absolute_scale_floor <= 0.0:
        raise ValueError("absolute_scale_floor must be finite and positive.")

    rng = random.Random(context_seed)
    initial_population = tuple(
        random_tour(instance.num_cities, rng)
        for _ in range(num_particles)
    )
    initial_objectives = tuple(
        instance.evaluate(tour)
        for tour in initial_population
    )
    ideal = tuple(
        min(objective[d] for objective in initial_objectives)
        for d in range(instance.num_objectives)
    )
    nadir = tuple(
        max(objective[d] for objective in initial_objectives)
        for d in range(instance.num_objectives)
    )
    scale_estimates = tuple(float(value) for value in instance.objective_scale_estimates)
    scales = tuple(
        max(
            absolute_scale_floor,
            hi - lo,
            minimum_scale_fraction * estimate,
        )
        for lo, hi, estimate in zip(ideal, nadir, scale_estimates)
    )
    weights = ScalarArchivePotential.reference_directions(
        instance.num_objectives,
        num_particles,
    )
    if len(weights) != num_particles:
        raise RuntimeError("Expected exactly one reference direction per particle.")

    hash_payload = {
        "ideal": ideal,
        "nadir": nadir,
        "scales": scales,
        "weights": weights,
        "chebyshev_rho": float(chebyshev_rho),
        # Temperature and laziness are intentionally excluded here: the scan
        # holds the state-energy context fixed while varying those kernel knobs.
    }
    encoded = json.dumps(
        hash_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "context_hash": hashlib.sha256(encoded).hexdigest(),
        "context_hash_scope": (
            "frozen typed-energy context; scanned temperature and laziness excluded"
        ),
        "context_seed": context_seed,
        "num_particles": num_particles,
        "ideal": ideal,
        "nadir": nadir,
        "scales": scales,
        "weights": weights,
        "chebyshev_rho": float(chebyshev_rho),
        "minimum_scale_fraction": float(minimum_scale_fraction),
        "absolute_scale_floor": float(absolute_scale_floor),
        "initial_population": initial_population,
        "initial_objectives": initial_objectives,
    }


def _production_kernel_context_hash(
    context: Mapping[str, Any],
    *,
    temperature: float,
    lazy_probability: float,
) -> str:
    """Match ``CertifiedSingleSiteIPSOptimizer._make_context_hash``."""
    payload = {
        "ideal": context["ideal"],
        "nadir": context["nadir"],
        "scales": context["scales"],
        "weights": context["weights"],
        "chebyshev_rho": context["chebyshev_rho"],
        "temperature": temperature,
        "lazy_probability": lazy_probability,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _typed_energy(
    objective: ObjectiveVector,
    weight: Sequence[float],
    *,
    ideal: Sequence[float],
    scales: Sequence[float],
    chebyshev_rho: float,
) -> float:
    normalized = tuple(
        (float(value) - float(lo)) / float(scale)
        for value, lo, scale in zip(objective, ideal, scales)
    )
    weighted = tuple(
        max(1e-3, float(direction)) * value
        for direction, value in zip(weight, normalized)
    )
    return max(weighted) + chebyshev_rho * sum(weighted)


def _proposal_targets(
    tours: Sequence[Tour],
    pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    state_index = {tour: index for index, tour in enumerate(tours)}
    return tuple(
        tuple(state_index[two_opt_at(tour, i, j)] for i, j in pairs)
        for tour in tours
    )


def _connected_from_targets(targets: Sequence[Sequence[int]]) -> bool:
    if not targets:
        return False
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in targets[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == len(targets)


def _numerically_irreducible(matrix: Any) -> bool:
    np = _numpy()
    size = int(matrix.shape[0])
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        neighbors = np.flatnonzero(matrix[current] > 0.0)
        for raw_neighbor in neighbors:
            neighbor = int(raw_neighbor)
            if neighbor != current and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == size


def _stationary_distribution(energies: Any, temperature: float) -> tuple[Any, Any]:
    np = _numpy()
    shifted = -(energies - float(np.min(energies))) / temperature
    unnormalized = np.exp(shifted)
    normalizer = float(np.sum(unnormalized))
    probabilities = unnormalized / normalizer
    log_normalizer = float(np.logaddexp.reduce(shifted))
    log_probabilities = shifted - log_normalizer
    return probabilities, log_probabilities


def _embedded_metropolis_matrix(
    energies: Any,
    proposal_targets: Sequence[Sequence[int]],
    temperature: float,
) -> tuple[Any, int, float]:
    np = _numpy()
    state_count = len(proposal_targets)
    proposal_count = len(proposal_targets[0])
    kernel = np.eye(state_count, dtype=float)
    proposal_probability = 1.0 / proposal_count
    underflowed = 0
    minimum_log_acceptance = 0.0
    for source, neighbors in enumerate(proposal_targets):
        for target in neighbors:
            delta = float(energies[target] - energies[source])
            log_acceptance = min(0.0, -delta / temperature)
            minimum_log_acceptance = min(minimum_log_acceptance, log_acceptance)
            if log_acceptance < _LOG_MIN_SUBNORMAL:
                acceptance = 0.0
                underflowed += 1
            else:
                acceptance = math.exp(log_acceptance)
            flow = proposal_probability * acceptance
            kernel[source, target] += flow
            kernel[source, source] -= flow
    return kernel, underflowed, minimum_log_acceptance


def _spectral_metrics(matrix: Any, stationary: Any) -> dict[str, Any]:
    np = _numpy()
    if bool(np.all(stationary > 0.0)):
        sqrt_pi = np.sqrt(stationary)
        symmetric = (
            sqrt_pi[:, None] * matrix
        ) / sqrt_pi[None, :]
        symmetry_error = float(np.max(np.abs(symmetric - symmetric.T)))
        eigenvalues = np.linalg.eigvalsh((symmetric + symmetric.T) * 0.5)
        max_imaginary_part = 0.0
    else:
        eigenvalues_complex = np.linalg.eigvals(matrix)
        max_imaginary_part = float(np.max(np.abs(np.imag(eigenvalues_complex))))
        eigenvalues = np.real(eigenvalues_complex)
        # A JSON string is deliberate: no finite reversible similarity
        # transform exists after the floating-point Gibbs weights underflow.
        symmetry_error = "inf"

    ordered = np.sort(eigenvalues)[::-1]
    largest = float(ordered[0])
    second = float(ordered[1]) if len(ordered) > 1 else largest
    minimum = float(ordered[-1])
    slem = float(np.max(np.abs(ordered[1:]))) if len(ordered) > 1 else 0.0
    return {
        "largest_eigenvalue": largest,
        "second_largest_eigenvalue": second,
        "minimum_eigenvalue": minimum,
        "ordinary_gap": max(0.0, 1.0 - second),
        "slem": min(1.0, max(0.0, slem)),
        "absolute_gap": max(0.0, 1.0 - slem),
        "reversible_transform_symmetry_error": symmetry_error,
        "max_eigenvalue_imaginary_part": max_imaginary_part,
    }


def _kernel_metrics(matrix: Any, stationary: Any) -> dict[str, Any]:
    np = _numpy()
    row_error = float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0)))
    minimum_entry = float(np.min(matrix))
    stationarity = stationary @ matrix
    stationarity_l1 = float(np.sum(np.abs(stationarity - stationary)))
    forward_flow = stationary[:, None] * matrix
    reverse_flow = forward_flow.T
    absolute_db = np.abs(forward_flow - reverse_flow)
    denominator = np.maximum(np.maximum(np.abs(forward_flow), np.abs(reverse_flow)), 1e-300)
    relative_db = absolute_db / denominator
    metrics: dict[str, Any] = {
        "row_sum_max_abs_error": row_error,
        "minimum_transition_probability": minimum_entry,
        "stationarity_l1_residual": stationarity_l1,
        "db_max_abs_flow_residual": float(np.max(absolute_db)),
        "db_max_relative_flow_residual": float(np.max(relative_db)),
        "numerically_irreducible": _numerically_irreducible(matrix),
    }
    metrics.update(_spectral_metrics(matrix, stationary))
    return metrics


def _log_tv_prefactor(log_pi_min: float) -> float:
    # log[1/2 sqrt(1/pi_min - 1)], evaluated without overflowing 1/pi.
    if log_pi_min >= 0.0:
        return -math.inf
    magnitude = -log_pi_min
    if magnitude > 50.0:
        log_inverse_minus_one = magnitude + math.log1p(-math.exp(-magnitude))
    else:
        log_inverse_minus_one = math.log(math.expm1(magnitude))
    return math.log(0.5) + 0.5 * log_inverse_minus_one


def _tv_bound(log_pi_min: float, slem: float, steps: int) -> float:
    if steps < 0:
        raise ValueError("steps must be nonnegative.")
    log_prefactor = _log_tv_prefactor(log_pi_min)
    if slem <= 0.0:
        return min(1.0, math.exp(log_prefactor)) if steps == 0 else 0.0
    if slem >= 1.0:
        return min(1.0, math.exp(min(0.0, log_prefactor)))
    log_bound = log_prefactor + steps * math.log(slem)
    if log_bound >= 0.0:
        return 1.0
    return math.exp(log_bound) if log_bound > -745.0 else 0.0


def _required_steps_for_tv(
    log_pi_min: float,
    slem: float,
    tv_tolerance: float,
) -> int | str:
    log_prefactor = _log_tv_prefactor(log_pi_min)
    if log_prefactor <= math.log(tv_tolerance):
        return 0
    if slem >= 1.0 - 1e-14:
        return "inf"
    if slem <= 0.0:
        return 1
    steps = math.ceil(
        (math.log(tv_tolerance) - log_prefactor) / math.log(slem)
    )
    return max(0, int(steps))


def _coordinate_audit(
    energies: Any,
    proposal_targets: Sequence[Sequence[int]],
    *,
    temperature: float,
    lazy_probability: float,
) -> dict[str, Any]:
    np = _numpy()
    active, underflowed, minimum_log_acceptance = _embedded_metropolis_matrix(
        energies,
        proposal_targets,
        temperature,
    )
    evaluation_clock = (
        lazy_probability * np.eye(len(proposal_targets), dtype=float)
        + (1.0 - lazy_probability) * active
    )
    stationary, log_stationary = _stationary_distribution(energies, temperature)
    minimum_energy = float(np.min(energies))
    maximum_energy = float(np.max(energies))
    expected_energy = float(stationary @ energies)
    energy_range = maximum_energy - minimum_energy
    relative_excess = (
        (expected_energy - minimum_energy) / energy_range
        if energy_range > 0.0
        else 0.0
    )
    return {
        "energy_min": minimum_energy,
        "energy_max": maximum_energy,
        "energy_range": energy_range,
        "stationary_expected_energy": expected_energy,
        "stationary_expected_excess": expected_energy - minimum_energy,
        "relative_stationary_excess": relative_excess,
        "stationary_min_probability": float(np.min(stationary)),
        "stationary_max_probability": float(np.max(stationary)),
        "log_stationary_min_probability": float(np.min(log_stationary)),
        "minimum_log_acceptance": minimum_log_acceptance,
        "underflowed_positive_acceptances": underflowed,
        "active_proposal_kernel": _kernel_metrics(active, stationary),
        "evaluation_clock": _kernel_metrics(evaluation_clock, stationary),
    }


def _product_spectrum(
    coordinate_rows: Sequence[Mapping[str, Any]],
    clock: str,
) -> dict[str, float]:
    particle_count = len(coordinate_rows)
    coordinate_metrics = [row[clock] for row in coordinate_rows]
    coordinate_gaps = [float(row["ordinary_gap"]) for row in coordinate_metrics]
    coordinate_minima = [float(row["minimum_eigenvalue"]) for row in coordinate_metrics]
    ordinary_gap = min(coordinate_gaps) / particle_count
    second_largest = 1.0 - ordinary_gap
    minimum_eigenvalue = sum(coordinate_minima) / particle_count
    slem = max(abs(second_largest), abs(minimum_eigenvalue))
    return {
        "ordinary_gap": max(0.0, ordinary_gap),
        "second_largest_eigenvalue": second_largest,
        "minimum_eigenvalue": minimum_eigenvalue,
        "slem": min(1.0, max(0.0, slem)),
        "absolute_gap": max(0.0, 1.0 - slem),
    }


def _explicit_product_audit(
    typed_energies: Sequence[Any],
    proposal_targets: Sequence[Sequence[int]],
    *,
    temperature: float,
    lazy_probability: float,
    max_product_states: int,
    derived_evaluation_spectrum: Mapping[str, float],
    derived_active_spectrum: Mapping[str, float],
) -> dict[str, Any]:
    """Directly enumerate the random-scan product matrix when it is tiny."""
    np = _numpy()
    coordinate_state_count = len(proposal_targets)
    product_state_count = coordinate_state_count ** len(typed_energies)
    if product_state_count > max_product_states:
        return {
            "performed": False,
            "num_product_states": product_state_count,
            "max_product_states": max_product_states,
            "reason": "product_state_cap_exceeded",
        }

    coordinate_kernels = []
    coordinate_stationary = []
    for energies in typed_energies:
        kernel, _, _ = _embedded_metropolis_matrix(
            energies,
            proposal_targets,
            temperature,
        )
        stationary, _ = _stationary_distribution(energies, temperature)
        coordinate_kernels.append(kernel)
        coordinate_stationary.append(stationary)

    identity = np.eye(coordinate_state_count, dtype=float)
    active_product = np.zeros(
        (product_state_count, product_state_count),
        dtype=float,
    )
    for active_coordinate, active_kernel in enumerate(coordinate_kernels):
        term = np.asarray([[1.0]])
        for coordinate in range(len(coordinate_kernels)):
            factor = active_kernel if coordinate == active_coordinate else identity
            term = np.kron(term, factor)
        active_product += term / len(coordinate_kernels)
    evaluation_product = (
        lazy_probability * np.eye(product_state_count, dtype=float)
        + (1.0 - lazy_probability) * active_product
    )
    product_stationary = np.asarray([1.0])
    for stationary in coordinate_stationary:
        product_stationary = np.kron(product_stationary, stationary)

    active_metrics = _kernel_metrics(
        active_product,
        product_stationary,
    )
    evaluation_metrics = _kernel_metrics(
        evaluation_product,
        product_stationary,
    )
    residuals = [
        abs(
            float(evaluation_metrics["ordinary_gap"])
            - float(derived_evaluation_spectrum["ordinary_gap"])
        ),
        abs(
            float(evaluation_metrics["slem"])
            - float(derived_evaluation_spectrum["slem"])
        ),
        abs(
            float(active_metrics["ordinary_gap"])
            - float(derived_active_spectrum["ordinary_gap"])
        ),
        abs(
            float(active_metrics["slem"])
            - float(derived_active_spectrum["slem"])
        ),
    ]
    return {
        "performed": True,
        "num_product_states": product_state_count,
        "max_product_states": max_product_states,
        "evaluation_clock": evaluation_metrics,
        "active_proposal_kernel": active_metrics,
        "max_spectral_formula_residual": max(residuals),
    }


def _temperature_row(
    typed_energies: Sequence[Any],
    proposal_targets: Sequence[Sequence[int]],
    *,
    temperature: float,
    lazy_probability: float,
    evaluation_budget: int,
    max_relative_stationary_excess: float,
    tv_tolerance: float,
    max_product_states: int,
    production_kernel_context_hash: str,
) -> dict[str, Any]:
    coordinate_rows = tuple(
        _coordinate_audit(
            energies,
            proposal_targets,
            temperature=temperature,
            lazy_probability=lazy_probability,
        )
        for energies in typed_energies
    )
    active_spectrum = _product_spectrum(
        coordinate_rows,
        "active_proposal_kernel",
    )
    evaluation_spectrum = _product_spectrum(coordinate_rows, "evaluation_clock")
    explicit_product = _explicit_product_audit(
        typed_energies,
        proposal_targets,
        temperature=temperature,
        lazy_probability=lazy_probability,
        max_product_states=max_product_states,
        derived_evaluation_spectrum=evaluation_spectrum,
        derived_active_spectrum=active_spectrum,
    )
    available_evaluations = max(0, evaluation_budget - len(coordinate_rows))
    log_product_pi_min = sum(
        float(row["log_stationary_min_probability"])
        for row in coordinate_rows
    )
    required_evaluations = _required_steps_for_tv(
        log_product_pi_min,
        evaluation_spectrum["slem"],
        tv_tolerance,
    )
    mixing_gate = (
        isinstance(required_evaluations, int)
        and required_evaluations <= available_evaluations
    )
    relative_excesses = [
        float(row["relative_stationary_excess"])
        for row in coordinate_rows
    ]
    distortion_gate = (
        max(relative_excesses, default=0.0)
        <= max_relative_stationary_excess
    )

    row_errors_evaluation = [
        float(row["evaluation_clock"]["row_sum_max_abs_error"])
        for row in coordinate_rows
    ]
    row_errors_active = [
        float(row["active_proposal_kernel"]["row_sum_max_abs_error"])
        for row in coordinate_rows
    ]
    db_errors_evaluation = [
        float(row["evaluation_clock"]["db_max_abs_flow_residual"])
        for row in coordinate_rows
    ]
    db_errors_active = [
        float(row["active_proposal_kernel"]["db_max_abs_flow_residual"])
        for row in coordinate_rows
    ]
    stationarity_errors = [
        float(row["evaluation_clock"]["stationarity_l1_residual"])
        for row in coordinate_rows
    ]
    all_irreducible = all(
        bool(row["evaluation_clock"]["numerically_irreducible"])
        for row in coordinate_rows
    )
    underflowed = sum(
        int(row["underflowed_positive_acceptances"])
        for row in coordinate_rows
    )
    invariant_gate = (
        max(row_errors_evaluation, default=0.0) <= _MATRIX_TOLERANCE
        and max(row_errors_active, default=0.0) <= _MATRIX_TOLERANCE
        and max(db_errors_evaluation, default=0.0) <= _MATRIX_TOLERANCE
        and max(db_errors_active, default=0.0) <= _MATRIX_TOLERANCE
        and max(stationarity_errors, default=0.0) <= _MATRIX_TOLERANCE
        and all_irreducible
        and underflowed == 0
    )
    if bool(explicit_product["performed"]):
        invariant_gate = (
            invariant_gate
            and float(explicit_product["max_spectral_formula_residual"]) <= 1e-10
            and float(
                explicit_product["evaluation_clock"]["row_sum_max_abs_error"]
            )
            <= _MATRIX_TOLERANCE
            and float(
                explicit_product["evaluation_clock"]["db_max_abs_flow_residual"]
            )
            <= _MATRIX_TOLERANCE
            and float(
                explicit_product["active_proposal_kernel"][
                    "row_sum_max_abs_error"
                ]
            )
            <= _MATRIX_TOLERANCE
            and float(
                explicit_product["active_proposal_kernel"][
                    "db_max_abs_flow_residual"
                ]
            )
            <= _MATRIX_TOLERANCE
        )
    feasible = invariant_gate and distortion_gate and mixing_gate

    return {
        "temperature": float(temperature),
        "production_kernel_context_hash": production_kernel_context_hash,
        "available_transition_evaluations_after_initialization": available_evaluations,
        "max_relative_stationary_excess": max(relative_excesses, default=0.0),
        "mean_relative_stationary_excess": (
            sum(relative_excesses) / len(relative_excesses)
            if relative_excesses
            else 0.0
        ),
        "distortion_gate": distortion_gate,
        "product_log_min_stationary_probability": log_product_pi_min,
        "product_ordinary_gap_evaluation_clock": evaluation_spectrum["ordinary_gap"],
        "product_absolute_gap_evaluation_clock": evaluation_spectrum["absolute_gap"],
        "product_slem_evaluation_clock": evaluation_spectrum["slem"],
        "product_minimum_eigenvalue_evaluation_clock": evaluation_spectrum[
            "minimum_eigenvalue"
        ],
        "product_ordinary_gap_active_proposal_kernel": active_spectrum[
            "ordinary_gap"
        ],
        "product_absolute_gap_active_proposal_kernel": active_spectrum[
            "absolute_gap"
        ],
        "product_slem_active_proposal_kernel": active_spectrum["slem"],
        "product_minimum_eigenvalue_active_proposal_kernel": active_spectrum[
            "minimum_eigenvalue"
        ],
        "tv_bound_at_evaluation_budget": _tv_bound(
            log_product_pi_min,
            evaluation_spectrum["slem"],
            available_evaluations,
        ),
        "required_evaluations_tv_bound": required_evaluations,
        "mixing_gate_evaluation_clock": mixing_gate,
        "max_row_sum_error_evaluation_clock": max(
            row_errors_evaluation,
            default=0.0,
        ),
        "max_row_sum_error_active_proposal_kernel": max(
            row_errors_active,
            default=0.0,
        ),
        "max_db_flow_residual_evaluation_clock": max(
            db_errors_evaluation,
            default=0.0,
        ),
        "max_db_flow_residual_active_proposal_kernel": max(
            db_errors_active,
            default=0.0,
        ),
        "max_stationarity_l1_residual_evaluation_clock": max(
            stationarity_errors,
            default=0.0,
        ),
        "all_coordinate_kernels_irreducible": all_irreducible,
        "underflowed_positive_acceptances": underflowed,
        "kernel_invariant_gate": invariant_gate,
        "h1_feasible_on_grid": feasible,
        "explicit_product_audit": explicit_product,
        "coordinate_audits": coordinate_rows,
    }


def _feasible_grid_runs(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group adjacent feasible *sampled points* without claiming continuity."""
    runs: list[dict[str, Any]] = []
    current: list[float] = []
    for row in rows:
        temperature = float(row["temperature"])
        if bool(row["h1_feasible_on_grid"]):
            current.append(temperature)
        elif current:
            runs.append(
                {
                    "sampled_temperatures": tuple(current),
                    "num_sampled_points": len(current),
                    "sampled_temperature_min": current[0],
                    "sampled_temperature_max": current[-1],
                }
            )
            current = []
    if current:
        runs.append(
            {
                "sampled_temperatures": tuple(current),
                "num_sampled_points": len(current),
                "sampled_temperature_min": current[0],
                "sampled_temperature_max": current[-1],
            }
        )
    return runs


def audit_typed_mh_temperature_grid(
    instance: MultiObjectiveTSPInstance,
    *,
    num_particles: int,
    context_seed: int,
    temperatures: Iterable[float],
    evaluation_budget: int = 512,
    lazy_probability: float = 0.05,
    chebyshev_rho: float = 0.03,
    minimum_scale_fraction: float = 1e-3,
    absolute_scale_floor: float = 1e-12,
    max_states: int = 720,
    max_product_states: int = 4096,
    max_relative_stationary_excess: float = 0.05,
    tv_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Enumerate and audit a fixed-context typed-MH temperature grid.

    H1 is the finite-grid existential statement that at least one scanned
    temperature simultaneously satisfies:

    1. every typed coordinate's stationary expected energy excess, divided by
       its full enumerated energy range, is at most
       ``max_relative_stationary_excess``; and
    2. the standard worst-start reversible-chain L2-to-TV bound is at most
       ``tv_tolerance`` within the proposal evaluations remaining after the
       initial particle evaluations.

    A ``FALSIFIED_ON_GRID`` verdict is intentionally limited to the supplied
    finite temperature grid and thresholds; it is not a continuous-temperature
    impossibility theorem.
    """
    if evaluation_budget < num_particles:
        raise ValueError("evaluation_budget must cover all initial particles.")
    if max_product_states <= 0:
        raise ValueError("max_product_states must be positive.")
    if not math.isfinite(lazy_probability) or not 0.0 < lazy_probability < 1.0:
        raise ValueError("lazy_probability must lie strictly between zero and one.")
    if (
        not math.isfinite(max_relative_stationary_excess)
        or max_relative_stationary_excess < 0.0
    ):
        raise ValueError("max_relative_stationary_excess must be finite and nonnegative.")
    if not math.isfinite(tv_tolerance) or not 0.0 < tv_tolerance < 1.0:
        raise ValueError("tv_tolerance must lie strictly between zero and one.")

    temperature_grid = tuple(sorted(set(float(value) for value in temperatures)))
    if not temperature_grid:
        raise ValueError("At least one temperature is required.")
    if any(not math.isfinite(value) or value <= 0.0 for value in temperature_grid):
        raise ValueError("All temperatures must be finite and strictly positive.")

    tours = _enumerate_tours(instance.num_cities, max_states)
    pairs = _two_opt_pairs(instance.num_cities)
    targets = _proposal_targets(tours, pairs)
    proposal_connected = _connected_from_targets(targets)
    objectives = tuple(instance.evaluate(tour) for tour in tours)
    context = _fixed_context(
        instance,
        num_particles=num_particles,
        context_seed=context_seed,
        chebyshev_rho=chebyshev_rho,
        minimum_scale_fraction=minimum_scale_fraction,
        absolute_scale_floor=absolute_scale_floor,
    )
    np = _numpy()
    typed_energies = tuple(
        np.asarray(
            [
                _typed_energy(
                    objective,
                    weight,
                    ideal=context["ideal"],
                    scales=context["scales"],
                    chebyshev_rho=chebyshev_rho,
                )
                for objective in objectives
            ],
            dtype=float,
        )
        for weight in context["weights"]
    )
    rows = tuple(
        _temperature_row(
            typed_energies,
            targets,
            temperature=temperature,
            lazy_probability=lazy_probability,
            evaluation_budget=evaluation_budget,
            max_relative_stationary_excess=max_relative_stationary_excess,
            tv_tolerance=tv_tolerance,
            max_product_states=max_product_states,
            production_kernel_context_hash=_production_kernel_context_hash(
                context,
                temperature=temperature,
                lazy_probability=lazy_probability,
            ),
        )
        for temperature in temperature_grid
    )
    feasible_grid_runs = _feasible_grid_runs(rows)
    if feasible_grid_runs:
        verdict = "NOT_FALSIFIED_ON_GRID"
    elif all(bool(row["kernel_invariant_gate"]) for row in rows):
        verdict = "FALSIFIED_ON_GRID"
    else:
        verdict = "INDETERMINATE_NUMERIC_OR_INVARIANT_FAILURE"

    context_json = dict(context)
    distance_matrices = instance.distance_matrices
    distance_matrix_bytes = json.dumps(
        distance_matrices,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": SCHEMA,
        "instance": {
            "name": instance.name,
            "num_cities": instance.num_cities,
            "num_objectives": instance.num_objectives,
            "symmetric_objectives": instance.symmetric_objectives,
            "distance_matrices_sha256": hashlib.sha256(
                distance_matrix_bytes
            ).hexdigest(),
            "distance_matrices": distance_matrices,
        },
        "state_space": {
            "representation": "fixed_zero_hamiltonian_tours",
            "num_tours": len(tours),
            "tours": tours,
            "objectives_by_tour_index": objectives,
            "typed_energies_by_coordinate": tuple(
                tuple(float(value) for value in energies)
                for energies in typed_energies
            ),
        },
        "proposal_graph": {
            "proposal": "uniform_symmetric_two_opt",
            "two_opt_pairs": pairs,
            "degree": len(pairs),
            "connected": proposal_connected,
            "target_tour_indices_by_state": targets,
        },
        "context": context_json,
        "evaluation_budget": evaluation_budget,
        "lazy_probability": lazy_probability,
        "budget_clock_note": (
            "Every post-initialization transition, including an explicit lazy "
            "identity, performs one objective evaluation in the production "
            "loop. Therefore the lazy mixture is the fixed-evaluation output "
            "kernel; the active-proposal kernel is diagnostic only."
        ),
        "h1_definition": {
            "scope": "existential over the supplied finite temperature grid",
            "max_relative_stationary_excess": max_relative_stationary_excess,
            "relative_distortion": (
                "(E_pi[H_r]-min H_r)/(max H_r-min H_r), with zero for a flat energy"
            ),
            "tv_tolerance": tv_tolerance,
            "mixing_proxy": (
                "0.5*sqrt(1/Pi_min-1)*SLEM^k worst-start L2-to-TV upper bound"
            ),
            "budget_steps": (
                "evaluation_budget-num_particles evaluated transitions, "
                "including lazy identity transitions"
            ),
            "gate_clock": "evaluation_clock",
        },
        "kernel_semantics": {
            "matrix_model": (
                "ideal-real Metropolis formula evaluated in float64 with "
                "underflow explicitly gated"
            ),
            "runtime_prng": (
                "Python random.Random draws on a finite approximately 53-bit grid; "
                "this audit does not claim machine-exact equality to the ideal kernel"
            ),
            "hamiltonian_structure": "sum_of_frozen_coordinate_typed_energies",
            "interaction_structure": "factorized_noninteracting_product_chain",
            "not_instantiated_here": (
                "MMD interaction, mean-field feedback, archive feedback, learned "
                "proposal, crossover, local descent, or time-varying context"
            ),
        },
        "temperature_rows": rows,
        "feasible_temperature_grid_runs": feasible_grid_runs,
        "h1_grid_verdict": verdict,
        "claim_limit": (
            "The verdict audits the exact finite-state reference definition on "
            "this instance/context/grid. Contiguous feasible sampled points are "
            "reported only as grid runs, not as continuous intervals. The audit "
            "does not prove continuous-temperature infeasibility, large-n "
            "polynomial mixing, finite-PRNG equivalence, or production-code "
            "equivalence."
        ),
    }


def _flat_temperature_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (dict, list, tuple))
    }
    explicit = row.get("explicit_product_audit", {})
    if isinstance(explicit, Mapping):
        flattened["explicit_product_performed"] = bool(
            explicit.get("performed", False)
        )
        flattened["explicit_product_num_states"] = explicit.get(
            "num_product_states",
            "",
        )
        flattened["explicit_product_formula_residual"] = explicit.get(
            "max_spectral_formula_residual",
            "",
        )
    return flattened


def write_enumerable_kernel_audit(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write the complete JSON report and a flat temperature-grid CSV."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "enumerable_kernel_audit.json"
    csv_path = destination / "temperature_scan.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    raw_rows = [
        _flat_temperature_csv_row(row)
        for row in report["temperature_rows"]
    ]
    fieldnames = sorted({key for row in raw_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return json_path, csv_path
