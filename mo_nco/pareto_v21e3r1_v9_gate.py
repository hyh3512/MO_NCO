from __future__ import annotations

"""Machine-enforced V9R2 pre-development readiness gate.

The gate deliberately has no path that converts the current packaged HOLD
contract into a development-promotion PASS.  It records why the full matrix,
simultaneous inference, selection, confirmation, formal study, and submission
must not run.  A future protocol must use a new schema and identity.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from .pareto_v21e3r1_v9_protocol import load_v9_predevelopment_protocol


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def evaluate_v9_predevelopment_readiness(
    protocol_path: str | Path | None = None,
    *,
    expected_protocol_file_sha256: str | None = None,
) -> dict[str, object]:
    """Return a canonical, non-authorizing readiness receipt."""

    certificate = load_v9_predevelopment_protocol(
        protocol_path,
        expected_file_sha256=expected_protocol_file_sha256,
    )
    payload = certificate["payload"]
    execution = certificate["execution_authorization"]
    later = certificate["later_phase_authorization"]
    unmet = list(certificate["unmet_required_artifacts"])
    resource = payload["resource_contract"]
    gates = {
        "protocol_canonical_and_self_bound": True,
        "exposed_development_manifest_frozen": True,
        "four_arm_candidate_menu_frozen": True,
        "single_case_engineering_smoke_authorized": (
            execution["single_case_smoke"] is True
        ),
        "full_matrix_resource_caps_frozen": (
            resource["caps_frozen_for_full_development_matrix"] is True
        ),
        "all_required_artifacts_bound": not unmet,
        "full_development_matrix_authorized": (
            execution["full_development_matrix"] is True
        ),
        "scientific_development_claims_authorized": (
            execution["scientific_development_claims"] is True
        ),
        "all_later_phases_prohibited": all(value is False for value in later.values()),
    }
    hold_reasons = [
        "full_matrix_resource_caps_not_frozen",
        *[f"required_artifact_missing:{name}" for name in unmet],
        "full_development_matrix_not_authorized",
        "scientific_development_claims_not_authorized",
    ]
    core: dict[str, object] = {
        "schema": "pareto_v21e3r1_v9r2_predevelopment_readiness_receipt_v1",
        "status": "PRE_DEVELOPMENT_HOLD",
        "protocol": {
            "canonical_sha256": certificate["canonical_sha256"],
            "source_file_sha256": certificate["source_file_sha256"],
            "protocol_payload_sha256": certificate["protocol_payload_sha256"],
            "resource_contract_sha256": certificate["resource_contract_sha256"],
        },
        "gates": gates,
        "hold_reasons": hold_reasons,
        "known_required_artifact_count": len(unmet),
        "required_artifact_inventory_exhaustive_for_later_scientific_phases": False,
        "additional_external_authority_and_submission_requirements_may_apply": True,
        "development_matrix_execution_started": False,
        "development_rows_materialized": 0,
        "development_promotion_evaluated": False,
        "development_promotion_gate_passed": False,
        "simultaneous_inference_performed": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
        "authorized_next_phase": (
            "NONE_NEW_PROTOCOL_AND_ALL_APPLICABLE_REQUIREMENTS_REQUIRED"
        ),
    }
    if gates["full_development_matrix_authorized"] or gates[
        "scientific_development_claims_authorized"
    ]:
        raise RuntimeError(
            "The V9R2 pre-development gate cannot accept an authorizing protocol."
        )
    return {
        **core,
        "receipt_payload_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def write_v9_predevelopment_readiness_receipt(
    output: str | Path,
    *,
    protocol_path: str | Path | None = None,
    expected_protocol_file_sha256: str | None = None,
) -> dict[str, object]:
    """Exclusively materialize one canonical HOLD receipt."""

    destination = Path(output).resolve()
    receipt = evaluate_v9_predevelopment_readiness(
        protocol_path,
        expected_protocol_file_sha256=expected_protocol_file_sha256,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(_canonical_bytes(receipt) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the fail-closed V9R2 pre-development readiness gate."
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-file-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = write_v9_predevelopment_readiness_receipt(
        args.output,
        protocol_path=args.protocol,
        expected_protocol_file_sha256=args.expected_protocol_file_sha256,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_v9_predevelopment_readiness",
    "main",
    "write_v9_predevelopment_readiness_receipt",
]
