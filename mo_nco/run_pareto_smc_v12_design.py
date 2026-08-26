from __future__ import annotations

"""Fail-closed arithmetic preflight for exact-budget regeneration candidates.

The output does not bind a full per-stage mutation vector and therefore is not
by itself a formally executable Pareto-SMC schedule.
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from .pareto_regeneration_certificate import (
    confirm_cell_certificate,
    enumerate_equal_dual_stream_schedules,
    regeneration_exposure,
    terminal_residual_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-evaluations", type=int, required=True)
    parser.add_argument("--types", type=int, required=True)
    parser.add_argument("--max-particles-per-stream", type=int, required=True)
    parser.add_argument("--checkpoint-period", type=int, required=True)
    parser.add_argument(
        "--final-stage-mutations",
        type=int,
        required=True,
        help=(
            "Certified terminal regeneration steps reserved inside each "
            "candidate's total mutation count."
        ),
    )
    parser.add_argument("--one-step-minorization", type=float, required=True)
    parser.add_argument("--target-cell-mass-lower-bound", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for label, value in (
        ("one-step-minorization", args.one_step_minorization),
        (
            "target-cell-mass-lower-bound",
            args.target_cell_mass_lower_bound,
        ),
    ):
        if not math.isfinite(value) or not (0.0 < value <= 1.0):
            raise SystemExit(f"{label} must be finite and lie in (0,1].")
    if (
        isinstance(args.final_stage_mutations, bool)
        or args.final_stage_mutations <= 0
    ):
        raise SystemExit(
            "final-stage-mutations must be a positive integer."
        )
    rows = []
    for schedule in enumerate_equal_dual_stream_schedules(
        total_evaluations=args.total_evaluations,
        type_count=args.types,
        max_particles_per_stream=args.max_particles_per_stream,
        checkpoint_period=args.checkpoint_period,
    ):
        payload = asdict(schedule)
        epsilon = args.one_step_minorization
        steps = args.final_stage_mutations
        terminal_steps_within_total = (
            steps <= schedule.total_mutations_per_particle
        )
        residual = terminal_residual_weight(
            global_refresh_probability=epsilon,
            normalizer_lower_bound=1.0,
            mutation_steps=steps,
        )
        cell = confirm_cell_certificate(
            target_mass_lower_bound=(
                args.target_cell_mass_lower_bound
            ),
            confirm_particles=schedule.particles_per_type,
            confirm_residual_weight=residual,
        )
        hit = cell.per_particle_hit_lower_bound
        miss = cell.cell_miss_probability_upper_bound
        payload.update(
            {
                "one_step_minorization": args.one_step_minorization,
                "target_cell_mass_lower_bound": args.target_cell_mass_lower_bound,
                "terminal_regeneration_steps": steps,
                "terminal_steps_within_total": (
                    terminal_steps_within_total
                ),
                "terminal_residual_weight": residual,
                "per_particle_hit_lower_bound": hit,
                "cell_miss_probability_upper_bound": miss,
                "regeneration_exposure": regeneration_exposure(
                    schedule.particles_per_type,
                    steps,
                    args.one_step_minorization,
                ),
                "regeneration_exposure_scope": (
                    "declared_terminal_regeneration_steps_only"
                ),
            }
        )
        rows.append(payload)
    admissible = [
        row
        for row in rows
        if row["exact_budget_identity"]
        and row["checkpoint_aligned"]
        and row["particles_per_stream_within_cap"]
        and row["terminal_steps_within_total"]
    ]
    best = min(
        admissible,
        key=lambda row: (
            row["cell_miss_probability_upper_bound"],
            row["particles_per_stream"],
        ),
    ) if admissible else None
    output = {
        "schema": "pareto_smc_v12_exact_budget_design_v1",
        "design_gate": "PASS" if best is not None else "FAIL",
        "claim_scope": (
            "budget_feasible_single_cell_arithmetic_not_formal_schedule"
        ),
        "formal_stage_vector_binding_gate": "NOT_PERFORMED",
        "failure_reasons": (
            []
            if best is not None
            else [
                "no exact schedule satisfies the particle cap, checkpoint "
                "grid, and positive final-stage mutation requirement"
            ]
        ),
        "input": {
            "total_evaluations": args.total_evaluations,
            "type_count": args.types,
            "max_particles_per_stream": args.max_particles_per_stream,
            "checkpoint_period": args.checkpoint_period,
            "final_stage_mutations": args.final_stage_mutations,
            "one_step_minorization": args.one_step_minorization,
            "target_cell_mass_lower_bound": args.target_cell_mass_lower_bound,
        },
        "all_integer_schedules": rows,
        "admissible_schedules": admissible,
        "best_admissible_for_declared_single_cell_miss_bound": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if best is None:
        print(
            "No admissible regeneration schedule exists for the declared "
            "constraints.",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
