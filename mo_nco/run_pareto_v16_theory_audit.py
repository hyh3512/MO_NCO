from __future__ import annotations

"""CLI for the canonical v16 P0/P1/P2 theorem-package audit."""

import argparse
import json
from pathlib import Path

from .pareto_v16_artifact_bundle import canonical_json_bytes
from .pareto_v16_theory_gate import evaluate_v16_theory_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composed-bundle", type=Path, required=True)
    parser.add_argument("--theory-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-p0-p1-p2", action="store_true")
    arguments = parser.parse_args()
    gate, composed, theory = evaluate_v16_theory_gate(
        composed_bundle_path=arguments.composed_bundle,
        theory_packet_path=arguments.theory_packet,
    )
    payload = {
        "gate": gate.to_jsonable(),
        "p0_certificate": composed.to_jsonable(),
        "p1_certificate": theory.to_jsonable(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "p0_correctness_gate": gate.p0_correctness_gate,
                "p1_main_theory_gate": gate.p1_main_theory_gate,
                "p2_mathematical_contribution_gate": (
                    gate.p2_mathematical_contribution_gate
                ),
                "submission_verdict": gate.submission_verdict,
                "output": str(arguments.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if arguments.require_p0_p1_p2 and not (
        gate.p0_correctness_gate
        and gate.p1_main_theory_gate
        and gate.p2_mathematical_contribution_gate
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
