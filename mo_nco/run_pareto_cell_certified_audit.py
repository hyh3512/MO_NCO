from __future__ import annotations

"""Exact tiny-state audit for source-bound finite-step cell probes."""

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .contracts import ClaimLevel
from .evaluation import CountingTSPInstance
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_bounds import certify_independent_cell_probe_bounds, nondominated_points
from .pareto_cell_certification import (
    CertifiedCellType,
    CellCertifiedParetoSampler,
    augmented_tchebycheff_energy,
    original_cell_index,
    plan_cell_type,
)
from .types import ObjectiveVector, Tour


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _tight_box(
    objectives: Sequence[ObjectiveVector],
) -> Tuple[ObjectiveVector, ObjectiveVector]:
    dimension = len(objectives[0])
    lower = []
    upper = []
    for coordinate in range(dimension):
        values = [point[coordinate] for point in objectives]
        low = min(values)
        high = max(values)
        if high <= low:
            high = low + max(1e-12, 1e-12 * abs(low))
        lower.append(low)
        upper.append(high)
    return tuple(lower), tuple(upper)


def _separating_widths(
    pareto: Sequence[ObjectiveVector],
    lower: Sequence[float],
    upper: Sequence[float],
) -> ObjectiveVector:
    for divisions in (4, 8, 16, 32, 64, 128, 256, 512):
        widths = tuple(
            (high - low) / divisions for low, high in zip(lower, upper)
        )
        cells = {
            original_cell_index(
                point,
                lower=lower,
                upper=upper,
                widths=widths,
            )
            for point in pareto
        }
        if len(cells) == len(pareto):
            return widths
    raise RuntimeError("Could not separate the exact Pareto points into grid cells.")


def _optimize_plan(
    *,
    cell: tuple[int, ...],
    direction: ObjectiveVector,
    base_cell_mass_lower_bound: float,
    mass_proof_hash: str,
    beta: float,
    rho: float,
    penalty: float,
    gamma: float,
    failure_budget: float,
) -> CertifiedCellType:
    best: tuple[int, int, int] | None = None
    for steps in range(0, 301):
        provisional = CertifiedCellType(
            cell=cell,
            reference_direction=direction,
            base_cell_mass_lower_bound=base_cell_mass_lower_bound,
            base_mass_proof_sha256=mass_proof_hash,
            outside_cell_penalty=penalty,
            global_refresh_probability=gamma,
            mutation_steps=steps,
            particle_count=1,
            failure_budget=failure_budget,
        )
        plan = plan_cell_type(provisional, beta=beta, chebyshev_rho=rho)
        hit = plan.endpoint_cell_hit_lower_bound
        if hit <= 0.0:
            continue
        particles = plan.required_particle_count
        cost = particles * (1 + steps)
        if best is None or cost < best[0]:
            best = (cost, steps, particles)
    if best is None:
        raise RuntimeError("No finite probe plan was found in 300 mutation steps.")
    _, steps, particles = best
    return CertifiedCellType(
        cell=cell,
        reference_direction=direction,
        base_cell_mass_lower_bound=base_cell_mass_lower_bound,
        base_mass_proof_sha256=mass_proof_hash,
        outside_cell_penalty=penalty,
        global_refresh_probability=gamma,
        mutation_steps=steps,
        particle_count=particles,
        failure_budget=failure_budget,
    )


def _penalized_energy(
    objective: ObjectiveVector,
    *,
    contract: CertifiedCellType,
    lower: Sequence[float],
    upper: Sequence[float],
    widths: Sequence[float],
    rho: float,
) -> float:
    base = augmented_tchebycheff_energy(
        objective,
        lower=lower,
        upper=upper,
        direction=contract.reference_direction,
        rho=rho,
    )
    cell = original_cell_index(
        objective,
        lower=lower,
        upper=upper,
        widths=widths,
    )
    return base + (
        0.0 if cell == contract.cell else contract.outside_cell_penalty
    )


def _matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    n = len(left)
    return [
        [
            sum(left[i][k] * right[k][j] for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]


def _exact_kernel_checks(
    tours: Sequence[Tour],
    objectives: Sequence[ObjectiveVector],
    *,
    contract: CertifiedCellType,
    lower: Sequence[float],
    upper: Sequence[float],
    widths: Sequence[float],
    beta: float,
    rho: float,
) -> Dict[str, float]:
    n = len(tours)
    energy = [
        _penalized_energy(
            objective,
            contract=contract,
            lower=lower,
            upper=upper,
            widths=widths,
            rho=rho,
        )
        for objective in objectives
    ]
    unnormalized = [math.exp(-beta * value) / n for value in energy]
    normalizer = sum(unnormalized)
    target = [value / normalizer for value in unnormalized]
    target_cell_mass = sum(
        probability
        for probability, objective in zip(target, objectives)
        if original_cell_index(
            objective,
            lower=lower,
            upper=upper,
            widths=widths,
        )
        == contract.cell
    )

    # Complete global independence-MH ratio for the implemented binary64
    # energy. Tiny audit uses gamma=1, so this is
    # the exact runtime kernel rather than merely the defensive component.
    kernel = [[0.0 for _ in range(n)] for _ in range(n)]
    for x in range(n):
        accepted_sum = 0.0
        for y in range(n):
            alpha = min(1.0, math.exp(-beta * (energy[y] - energy[x])))
            transition = alpha / n
            kernel[x][y] += transition
            accepted_sum += transition
        kernel[x][x] += 1.0 - accepted_sum
    min_ratio = min(
        kernel[x][y] / target[y]
        for x in range(n)
        for y in range(n)
        if target[y] > 0.0
    )

    power = [[float(i == j) for j in range(n)] for i in range(n)]
    base = kernel
    exponent = contract.mutation_steps
    while exponent:
        if exponent & 1:
            power = _matrix_multiply(power, base)
        base = _matrix_multiply(base, base)
        exponent //= 2
    worst_tv = max(
        0.5 * sum(abs(power[x][y] - target[y]) for y in range(n))
        for x in range(n)
    )
    return {
        "exact_target_cell_mass": target_cell_mass,
        "exact_min_kernel_over_target": min_ratio,
        "exact_worst_start_tv_after_steps": worst_tv,
    }


def run_audit(
    *,
    output: Path,
    proof_output: Path | None = None,
    cities: int = 5,
    instance_seed: int = 20260726,
    algorithm_seed: int = 0,
    confidence_delta: float = 0.05,
) -> dict[str, Any]:
    if cities < 4 or cities > 8:
        raise ValueError("Exact cell audit cities must lie in [4, 8].")
    base = MultiObjectiveTSPInstance.random_biobjective(cities, seed=instance_seed)
    tours = tuple((0,) + tail for tail in itertools.permutations(range(1, cities)))
    objectives = tuple(base.evaluate(tour) for tour in tours)
    pareto = nondominated_points(objectives)
    lower, upper = _tight_box(objectives)
    widths = _separating_widths(pareto, lower, upper)
    all_cells = tuple(
        original_cell_index(
            objective,
            lower=lower,
            upper=upper,
            widths=widths,
        )
        for objective in objectives
    )
    pareto_cells = tuple(
        sorted(
            {
                original_cell_index(
                    point,
                    lower=lower,
                    upper=upper,
                    widths=widths,
                )
                for point in pareto
            }
        )
    )
    counts = Counter(all_cells)
    state_count = len(tours)

    proof_payload = {
        "schema": "exact_cell_mass_and_completeness_proof_v1",
        "instance_sha256": instance_sha256(base),
        "enumerated_tours": [list(tour) for tour in tours],
        "objectives": [list(point) for point in objectives],
        "pareto_points": [list(point) for point in pareto],
        "pareto_cells": [list(cell) for cell in pareto_cells],
        "lower": list(lower),
        "upper": list(upper),
        "widths": list(widths),
        "cell_counts": [
            {"cell": list(cell), "count": counts[cell]}
            for cell in pareto_cells
        ],
    }
    proof_hash = _sha256(proof_payload)
    if proof_output is not None:
        proof_output.parent.mkdir(parents=True, exist_ok=True)
        proof_output.write_bytes(_canonical_json_bytes(proof_payload))
        if hashlib.sha256(proof_output.read_bytes()).hexdigest() != proof_hash:
            raise RuntimeError("Serialized proof artifact hash mismatch.")

    # High-temperature exact audit emphasizes a nontrivial finite-step bound.
    beta = 0.1
    rho = 0.03
    penalty = 0.0
    gamma = 1.0
    direction = tuple(
        1.0 / base.num_objectives for _ in range(base.num_objectives)
    )
    per_cell_delta = confidence_delta / len(pareto_cells)
    contracts = tuple(
        _optimize_plan(
            cell=cell,
            direction=direction,
            base_cell_mass_lower_bound=counts[cell] / state_count,
            mass_proof_hash=proof_hash,
            beta=beta,
            rho=rho,
            penalty=penalty,
            gamma=gamma,
            failure_budget=per_cell_delta,
        )
        for cell in pareto_cells
    )
    total_budget = sum(
        contract.particle_count * (1 + contract.mutation_steps)
        for contract in contracts
    )
    counted = CountingTSPInstance(base=base, max_evaluations=total_budget)
    sampler = CellCertifiedParetoSampler(
        counted,
        cell_types=contracts,
        objective_lower_bounds=lower,
        objective_upper_bounds=upper,
        cell_widths=widths,
        beta=beta,
        chebyshev_rho=rho,
        seed=algorithm_seed,
        confidence_delta=confidence_delta,
        cell_completeness_proof_sha256=proof_hash,
        objective_box_proof_sha256=proof_hash,
        metric_box_proof_sha256=proof_hash,
        metric_igd_p=2.0,
        max_igd_bound=math.sqrt(sum(width * width for width in widths)),
        hv_reference=upper,
        max_hv_deficit_bound=sum(
            width
            * math.prod(
                upper[other] - lower[other]
                for other in range(base.num_objectives)
                if other != coordinate
            )
            for coordinate, width in enumerate(widths)
        ),
        archive_max_size=None,
    )
    result = sampler.run()

    exact_checks = []
    for contract, plan in zip(contracts, sampler.plans):
        exact = _exact_kernel_checks(
            tours,
            objectives,
            contract=contract,
            lower=lower,
            upper=upper,
            widths=widths,
            beta=beta,
            rho=rho,
        )
        actual_base_mass = counts[contract.cell] / state_count
        exact_checks.append(
            {
                "cell": contract.cell,
                "plan": plan.__dict__,
                "exact_uniform_base_cell_mass": actual_base_mass,
                **exact,
                "base_mass_certificate_verified": (
                    actual_base_mass + 1e-12
                    >= contract.base_cell_mass_lower_bound
                ),
                "mass_lower_bound_verified": (
                    exact["exact_target_cell_mass"] + 1e-12
                    >= plan.target_cell_mass_lower_bound
                ),
                "minorization_verified": (
                    exact["exact_min_kernel_over_target"] + 1e-12
                    >= plan.doeblin_minorization
                ),
                "tv_bound_verified": (
                    exact["exact_worst_start_tv_after_steps"]
                    <= plan.mutation_tv_radius + 1e-12
                ),
            }
        )

    geometry = certify_independent_cell_probe_bounds(
        feasible_objectives=objectives,
        probe_objectives=result.objectives,
        declared_cells=pareto_cells,
        objective_lower=lower,
        objective_upper=upper,
        cell_widths_original=widths,
        source_bound_failure_probability=float(
            result.metadata["total_failure_probability_bound"]
        ),
        requested_confidence_delta=confidence_delta,
        igd_p=2.0,
        hv_reference=upper,
    )
    checks = {
        "source_bound_runtime_gate": (
            result.metadata["scientific_design_gate"] == "PASS"
        ),
        "exact_budget": counted.evaluations == total_budget,
        "all_base_mass_certificates": all(
            item["base_mass_certificate_verified"] for item in exact_checks
        ),
        "all_mass_lower_bounds": all(
            item["mass_lower_bound_verified"] for item in exact_checks
        ),
        "all_minorization_bounds": all(
            item["minorization_verified"] for item in exact_checks
        ),
        "all_tv_bounds": all(
            item["tv_bound_verified"] for item in exact_checks
        ),
        "geometry_design_gate": geometry["design_verdict"] == "PASS",
        "realized_geometry": geometry["realized_geometry_verdict"] == "PASS",
    }
    verified_claim_level = (
        ClaimLevel.PARETO_CELL_CERTIFIED.value
        if all(checks.values())
        else result.metadata["claim_level"]
    )
    report = {
        "schema": "pareto_cell_certified_exact_audit_v2",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "verified_claim_level": verified_claim_level,
        "checks": checks,
        "instance": {
            "cities": cities,
            "instance_seed": instance_seed,
            "sha256": instance_sha256(base),
            "state_count": state_count,
            "pareto_point_count": len(pareto),
            "pareto_cell_count": len(pareto_cells),
        },
        "proof_artifact": proof_payload,
        "proof_artifact_sha256": proof_hash,
        "proof_artifact_hash_contract": (
            "sha256_of_canonical_utf8_json_no_trailing_newline"
        ),
        "proof_artifact_path": (
            None if proof_output is None else str(proof_output.resolve())
        ),
        "contracts": [contract.__dict__ for contract in contracts],
        "exact_kernel_checks": exact_checks,
        "runtime_metadata": result.metadata,
        "geometry_certificate": geometry,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path)
    parser.add_argument("--cities", type=int, default=5)
    parser.add_argument("--instance-seed", type=int, default=20260726)
    parser.add_argument("--algorithm-seed", type=int, default=0)
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    args = parser.parse_args()
    report = run_audit(
        output=args.output,
        proof_output=args.proof_output,
        cities=args.cities,
        instance_seed=args.instance_seed,
        algorithm_seed=args.algorithm_seed,
        confidence_delta=args.confidence_delta,
    )
    print(report["verdict"])
    print(args.output.resolve())
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
