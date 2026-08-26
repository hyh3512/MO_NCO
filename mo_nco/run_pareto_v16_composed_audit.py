from __future__ import annotations

"""CLI for the v16 canonical raw-artifact certificate."""

import argparse
import json
from pathlib import Path

from .pareto_v16_artifact_bundle import canonical_json_bytes, verify_v16_composed_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    certificate = verify_v16_composed_bundle(arguments.bundle)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(certificate.to_jsonable()))
    print(json.dumps({
        "schema": certificate.schema,
        "packet_sha256": certificate.packet_sha256,
        "p0_correctness_gate": certificate.p0_correctness_gate,
        "output": str(arguments.output),
    }, indent=2, sort_keys=True))
    if arguments.require_pass and not certificate.p0_correctness_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
