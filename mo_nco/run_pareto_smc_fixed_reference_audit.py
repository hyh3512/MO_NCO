from __future__ import annotations

"""Executable random-biobjective audit for the v10 pilot-confirm contract."""

import argparse
import json
from pathlib import Path

from .instance import MultiObjectiveTSPInstance
from .pareto_fixed_reference_spec import (
    load_fixed_reference_certificate_specification,
)
from .pareto_fixed_schedule_experiment import (
    run_fixed_schedule_pilot_confirm,
)
from .pareto_smc_spec import load_pareto_smc_specification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smc-spec", type=Path, required=True)
    parser.add_argument("--certificate-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cities", type=int, required=True)
    parser.add_argument("--instance-seed", type=int, required=True)
    parser.add_argument(
        "--particles-per-reference",
        type=int,
        required=True,
    )
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--run-seed", type=int, default=0)
    args = parser.parse_args()

    instance = MultiObjectiveTSPInstance.random_biobjective(
        args.cities,
        seed=args.instance_seed,
    )
    smc = load_pareto_smc_specification(
        args.smc_spec,
        objective_dimension=instance.num_objectives,
    )
    certificate_spec = (
        load_fixed_reference_certificate_specification(
            args.certificate_spec,
            objective_dimension=instance.num_objectives,
            instance=instance,
        )
    )
    result = run_fixed_schedule_pilot_confirm(
        instance,
        pareto_smc_specification=smc,
        certificate_specification=certificate_spec,
        particles_per_reference=args.particles_per_reference,
        run_seed=args.run_seed,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result.certificate,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    design = result.certificate["scientific_design_gate"]
    realized = result.certificate["realized_metric_gate"]
    print(f"SCIENTIFIC_DESIGN_GATE {design}")
    print(f"REALIZED_METRIC_GATE {realized}")
    print(
        "FORMAL_PACKET_GATE "
        f"{result.certificate['formal_packet_gate']}"
    )
    print(
        "TOTAL_CERTIFICATE_EVALUATIONS "
        f"{result.certificate['total_certificate_evaluations']}"
    )
    print(f"OUTPUT {output}")
    if (
        args.require_pass
        and result.certificate["formal_packet_gate"] != "PASS"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
