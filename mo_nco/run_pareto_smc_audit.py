from __future__ import annotations

"""Exact tiny-state audit for the Annealed Pareto-SMC reduction chain."""

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_bounds import (
    certify_pareto_bounds,
    nondominated_points,
    normalized_cell_index,
)
from .pareto_smc import AnnealedParetoSMCOptimizer
from .pareto_smc_spec import (
    analytic_objective_box,
    load_pareto_smc_specification,
    original_unit_cell_widths,
)


FINITE_PARTICLE_SCHEMA = "pareto_smc_cellwise_mse_certificate_v1"


def _energy(
    objective: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    direction: Sequence[float],
    rho: float,
) -> float:
    normalized = tuple(
        (value - low) / (high - low)
        for value, low, high in zip(objective, lower, upper)
    )
    weighted = tuple(
        weight * value for weight, value in zip(direction, normalized)
    )
    return max(weighted) + rho * sum(weighted)


def _exact_target_pareto_cell_masses(
    feasible_objectives: Sequence[Sequence[float]],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    normalized_widths: Sequence[float],
    reference_directions: Sequence[Sequence[float]],
    beta: float,
    rho: float,
) -> dict[tuple[int, ...], float]:
    normalized = tuple(
        tuple(
            (value - low) / (high - low)
            for value, low, high in zip(objective, lower, upper)
        )
        for objective in feasible_objectives
    )
    pareto = nondominated_points(normalized)
    pareto_cells = {
        normalized_cell_index(point, normalized_widths) for point in pareto
    }
    mixture = {cell: 0.0 for cell in pareto_cells}
    for direction in reference_directions:
        log_unnormalized = tuple(
            -beta
            * _energy(
                objective,
                lower=lower,
                upper=upper,
                direction=direction,
                rho=rho,
            )
            for objective in feasible_objectives
        )
        maximum = max(log_unnormalized)
        unnormalized = tuple(
            math.exp(value - maximum) for value in log_unnormalized
        )
        normalizer = sum(unnormalized)
        for objective, mass in zip(normalized, unnormalized):
            cell = normalized_cell_index(objective, normalized_widths)
            if cell in mixture:
                mixture[cell] += (
                    mass / normalizer / len(reference_directions)
                )
    return mixture


def _load_finite_particle_certificate(
    path: Path,
    *,
    expected_instance_sha256: str,
    expected_context_hash: str,
    expected_particle_count: int,
) -> tuple[float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Finite-particle certificate must be a JSON object.")
    if payload.get("schema") != FINITE_PARTICLE_SCHEMA:
        raise ValueError(
            f"Finite-particle certificate schema must be {FINITE_PARTICLE_SCHEMA!r}."
        )
    bindings = {
        "instance_sha256": expected_instance_sha256,
        "target_context_hash": expected_context_hash,
        "particle_count": expected_particle_count,
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Finite-particle certificate {key} does not match this run."
            )
    proof_hash = payload.get("proof_artifact_sha256")
    if (
        not isinstance(proof_hash, str)
        or len(proof_hash) != 64
        or any(character not in "0123456789abcdef" for character in proof_hash)
    ):
        raise ValueError(
            "Finite-particle certificate must bind a lowercase SHA-256 proof artifact."
        )
    B_L = float(payload.get("cellwise_mse_constant_B_L"))
    if not math.isfinite(B_L) or B_L < 0.0:
        raise ValueError(
            "Finite-particle cellwise MSE constant must be finite and nonnegative."
        )
    return B_L, payload


def run_audit(
    *,
    specification_path: Path,
    output_path: Path,
    cities: int,
    instance_seed: int,
    algorithm_seed: int,
    population: int,
    evaluations: int,
    confidence_delta: float,
    finite_particle_certificate_path: Path | None,
) -> dict[str, Any]:
    if cities < 4 or cities > 9:
        raise ValueError("Exact audit cities must lie in [4, 9].")
    instance = MultiObjectiveTSPInstance.random_biobjective(
        cities,
        seed=instance_seed,
    )
    specification = load_pareto_smc_specification(
        specification_path,
        objective_dimension=instance.num_objectives,
    )
    reference_count = len(specification.reference_directions)
    if population < reference_count or population % reference_count != 0:
        raise ValueError(
            "population must be a positive multiple of the reference count."
        )
    optimizer = AnnealedParetoSMCOptimizer(
        instance,
        particles_per_reference=population // reference_count,
        evaluations=evaluations,
        seed=algorithm_seed,
        beta_schedule=specification.beta_schedule,
        reference_directions=specification.reference_directions,
        num_reference_types=reference_count,
        epsilon=original_unit_cell_widths(instance, specification),
        ess_threshold=specification.ess_threshold_fraction,
        resampling_policy=(
            "always"
            if specification.mutation_steps_by_stage is not None
            else "ess"
        ),
        mutation_steps_by_stage=specification.mutation_steps_by_stage,
        chebyshev_rho=specification.chebyshev_rho,
        global_refresh_probability=(
            specification.global_refresh_probability
        ),
        archive_max_size=specification.archive_max_size,
    )
    result = optimizer.run()

    tours = tuple(
        (0,) + permutation
        for permutation in itertools.permutations(range(1, cities))
    )
    feasible_objectives = tuple(instance.evaluate(tour) for tour in tours)
    lower, upper = analytic_objective_box(instance)
    target_cell_masses = _exact_target_pareto_cell_masses(
        feasible_objectives,
        lower=lower,
        upper=upper,
        normalized_widths=specification.normalized_cell_widths,
        reference_directions=specification.reference_directions,
        beta=specification.beta_schedule[-1],
        rho=specification.chebyshev_rho,
    )
    p_min = min(target_cell_masses.values())

    terminal_weights = tuple(
        weight
        for reference_weights in result.metadata[
            "final_normalized_weights_by_reference"
        ]
        for weight in reference_weights
    )
    finite_particle_payload: Mapping[str, Any] | None = None
    if finite_particle_certificate_path is None:
        B_L = 1.0
        finite_particle_status = "UNRESOLVED_MISSING_EXTERNAL_CONSTANT"
    else:
        B_L, finite_particle_payload = _load_finite_particle_certificate(
            finite_particle_certificate_path.resolve(),
            expected_instance_sha256=instance_sha256(instance),
            expected_context_hash=str(result.metadata["context_hash"]),
            expected_particle_count=population,
        )
        finite_particle_status = "EXTERNAL_CERTIFICATE_BOUND"

    geometry_certificate = certify_pareto_bounds(
        feasible_objectives=feasible_objectives,
        particle_objectives=result.objectives,
        particle_weights=terminal_weights,
        objective_lower=lower,
        objective_upper=upper,
        normalized_cell_widths=specification.normalized_cell_widths,
        target_pareto_cell_probabilities=target_cell_masses,
        declared_p_min=p_min,
        cellwise_mse_constant_B_L=B_L,
        confidence_delta=confidence_delta,
    )
    stage_ledger = result.metadata["stage_ledger"]
    mechanics_checks = {
        "algorithm_contract": (
            result.metadata["algorithm_contract"]
            == "annealed_pareto_smc_feynman_kac_v4"
        ),
        "target_and_reporting_context_frozen": bool(
            result.metadata["context_frozen"]
            and result.metadata["stage_targets_frozen"]
            and result.metadata["frozen_contract_checked_at_stage_boundaries"]
        ),
        "archive_and_cell_observer_no_feedback": bool(
            not result.metadata["archive_feedback"]
            and not result.metadata["cell_observer_feedback"]
        ),
        "typed_resampling_only": (
            result.metadata["resampling_scope"]
            == "within_fixed_reference_type_only"
        ),
        "explicit_incremental_weights": all(
            all(
                len(reference["incremental_log_weights"]) == population
                // reference_count
                for reference in stage["references"]
            )
            for stage in stage_ledger
        ),
        "exact_run_local_evaluation_budget": (
            result.metadata["evaluations_used"] == evaluations
            and result.metadata["initial_population_evaluations"]
            + result.metadata["mutation_evaluations"]
            == evaluations
        ),
        "ideal_real_arithmetic_db_identity": (
            result.metadata[
                "db_max_abs_log_residual_real_arithmetic_identity"
            ]
            <= 1e-12
            and not result.metadata["machine_exact_detailed_balance_claimed"]
        ),
        "external_specification_predeclared": (
            len(specification.sha256) == 64
        ),
        "fixed_schedule_certificate_bound_when_requested": (
            specification.mutation_steps_by_stage is None
            or (
                result.metadata["resampling_policy"] == "always"
                and result.metadata[
                    "mutation_steps_by_stage_predeclared"
                ]
                == specification.mutation_steps_by_stage
                and result.metadata[
                    "contraction_aware_fixed_schedule_certificate"
                ]
                is not None
            )
        ),
    }
    mechanics_verdict = (
        "PASS" if all(mechanics_checks.values()) else "FAIL"
    )
    if finite_particle_certificate_path is None:
        scientific_verdict = "UNRESOLVED"
    elif geometry_certificate["verdict"] == "PASS":
        scientific_verdict = "PASS"
    else:
        scientific_verdict = "FAIL"
    report = {
        "schema": "pareto_smc_exact_tiny_audit_v1",
        "mechanics_verdict": mechanics_verdict,
        "scientific_verdict": scientific_verdict,
        "claim_boundary": (
            "mechanics and exact tiny-state target masses do not instantiate "
            "the finite-particle error constant"
        ),
        "specification": {
            "path": str(specification.path),
            "sha256": specification.sha256,
        },
        "instance": {
            "cities": cities,
            "instance_seed": instance_seed,
            "sha256": instance_sha256(instance),
            "enumerated_fixed_zero_tours": len(tours),
        },
        "run": {
            "algorithm_seed": algorithm_seed,
            "population": population,
            "evaluations": evaluations,
            "target_context_hash": result.metadata["context_hash"],
            "reporting_context_hash": result.metadata[
                "reporting_context_hash"
            ],
            "run_contract_hash": result.metadata["run_contract_hash"],
            "stage_ledger_hash": result.metadata["stage_ledger_hash"],
        },
        "mechanics_checks": mechanics_checks,
        "finite_particle_gate": {
            "status": finite_particle_status,
            "certificate_path": (
                None
                if finite_particle_certificate_path is None
                else str(finite_particle_certificate_path.resolve())
            ),
            "certificate_payload": finite_particle_payload,
            "p_min_exact_tiny_state": p_min,
            "B_L_used_for_conditional_report": B_L,
            "warning": (
                "B_L=1 is a placeholder used only to expose the unresolved "
                "gate; it is not a proved SMC constant."
                if finite_particle_certificate_path is None
                else None
            ),
        },
        "conditional_geometry_certificate": geometry_certificate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cities", type=int, default=5)
    parser.add_argument("--instance-seed", type=int, default=20260726)
    parser.add_argument("--algorithm-seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=512)
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    parser.add_argument("--finite-particle-certificate", type=Path)
    parser.add_argument("--require-scientific-pass", action="store_true")
    args = parser.parse_args()
    report = run_audit(
        specification_path=args.spec,
        output_path=args.output,
        cities=args.cities,
        instance_seed=args.instance_seed,
        algorithm_seed=args.algorithm_seed,
        population=args.population,
        evaluations=args.evaluations,
        confidence_delta=args.confidence_delta,
        finite_particle_certificate_path=args.finite_particle_certificate,
    )
    print(f"MECHANICS {report['mechanics_verdict']}")
    print(f"SCIENTIFIC_GATE {report['scientific_verdict']}")
    print(f"OUTPUT {args.output.resolve()}")
    if report["mechanics_verdict"] != "PASS":
        return 1
    if args.require_scientific_pass and report["scientific_verdict"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
