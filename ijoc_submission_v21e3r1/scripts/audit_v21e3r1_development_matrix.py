from __future__ import annotations

"""Same-implementation post-process audit of a historical V21e3r1 matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

try:
    from ijoc_submission_v21e3r1.scripts.run_v21e3r1_development_parity import (
        _default_paths,
        _matrix_plan,
        load_frozen_contract,
        verify_finalized_matrix_output,
    )
except ModuleNotFoundError:  # Direct path execution from the extracted release.
    from run_v21e3r1_development_parity import (  # type: ignore[no-redef]
        _default_paths,
        _matrix_plan,
        load_frozen_contract,
        verify_finalized_matrix_output,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_OWNED_FILE_PATHS = (
    "ijoc_submission_v21e3r1/scripts/audit_v21e3r1_development_matrix.py",
    "ijoc_submission_v21e3r1/scripts/run_v21e3r1_development_parity.py",
    "mo_nco/pareto_v21e3_parity.py",
    "mo_nco/pareto_v21e3_trace_verify.py",
)


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_verifier_owned_files() -> tuple[list[dict[str, object]], str]:
    entries: list[dict[str, object]] = []
    for relative in _OWNED_FILE_PATHS:
        path = (_REPOSITORY_ROOT / relative).resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Live verifier owned file is not regular: {relative}")
        raw = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    entries.sort(key=lambda entry: str(entry["path"]))
    return entries, hashlib.sha256(_canonical_bytes(entries)).hexdigest()


def audit_development_matrix(
    *,
    matrix_directory: Path,
    output: Path,
    case_manifest_path: Path,
    reference_manifest_path: Path,
    config_manifest_path: Path,
    metric_manifest_path: Path,
    protocol_path: Path,
    authorization_path: Path,
    source_snapshot_root_sha256: str,
) -> dict[str, object]:
    """Recompute every row metric/statistic and write a versioned receipt."""

    destination = output.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace same-implementation receipt: {destination}"
        )
    contract = load_frozen_contract(
        case_manifest_path=case_manifest_path,
        reference_manifest_path=reference_manifest_path,
        config_manifest_path=config_manifest_path,
        metric_manifest_path=metric_manifest_path,
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        source_snapshot_root_sha256=source_snapshot_root_sha256,
        require_matrix_authorization=True,
    )
    matrix = matrix_directory.resolve()
    plan_path = matrix / "matrix.plan.json"
    observed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_plan = _matrix_plan(contract)
    if observed_plan != expected_plan:
        raise RuntimeError("Same-implementation audit rejected the matrix plan.")
    recomputed = verify_finalized_matrix_output(
        matrix,
        contract=contract,
        plan=expected_plan,
    )
    analysis = recomputed["analysis"]
    if not isinstance(analysis, Mapping):
        raise RuntimeError(
            "Same-implementation audit did not reproduce the analysis object."
        )
    gate = str(analysis["overall_gate"])
    owned_files, owned_files_root = _live_verifier_owned_files()
    receipt = {
        "schema": (
            "pareto_v21e3r1_same_implementation_development_matrix_"
            "post_run_audit_v1"
        ),
        "status": "PASS_SAME_IMPLEMENTATION_POST_PROCESS_RECOMPUTATION",
        "scientific_scope": "development_only_engineering_evidence_not_formal_evidence",
        "implementation_independence": False,
        "scientific_independence": False,
        "external_third_party_audit": False,
        "fixed_author_generated_cases_descriptive_only": True,
        "population_inference_authorized": False,
        "sign_flip_assumptions_verified": False,
        "trimmed_mean_distinct_from_mean": False,
        "verifier_relationship": (
            "SAME_PROJECT_VERIFIER_POST_HOC_SUCCESSOR_NOT_HISTORICAL_PRODUCER"
        ),
        "historical_matrix_producer": {
            "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
            "authorization_receipt_sha256": contract.authorization_sha256,
        },
        "live_verifier_owned_file_count": len(owned_files),
        "live_verifier_owned_files": owned_files,
        "live_verifier_owned_files_root_sha256": owned_files_root,
        "matrix_directory": ".",
        "matrix_directory_path_semantics": "self_describing_matrix_root_v1",
        "matrix_plan_sha256": _sha256(plan_path),
        "matrix_aggregate_sha256": recomputed["aggregate_sha256"],
        "runner_post_run_audit_sha256": recomputed[
            "runner_post_run_audit_sha256"
        ],
        "source_snapshot_root_sha256": contract.source_snapshot_root_sha256,
        "authorization_receipt_sha256": contract.authorization_sha256,
        "objective_archive_and_metric_replayed_rows": recomputed["row_count"],
        "analysis_sha256": hashlib.sha256(_canonical_bytes(analysis)).hexdigest(),
        "noninferiority_gate": gate,
        "phase_release_effect": (
            "STOP_BEFORE_SELECTION_PARTITION_MATERIALIZATION"
            if gate.startswith("FAIL_")
            else "DEVELOPMENT_GATE_ONLY_NO_LATER_PHASE_RELEASE"
        ),
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "runtime_efficiency_claim_authorized": False,
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
        "submission_status": "IJOC_HOLD",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(receipt)
    with destination.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def main(argv: list[str] | None = None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, default=defaults["case_manifest"])
    parser.add_argument(
        "--reference-manifest", type=Path, default=defaults["reference_manifest"]
    )
    parser.add_argument(
        "--config-manifest", type=Path, default=defaults["config_manifest"]
    )
    parser.add_argument(
        "--metric-manifest", type=Path, default=defaults["metric_manifest"]
    )
    parser.add_argument("--protocol", type=Path, default=defaults["protocol"])
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--source-snapshot-root-sha256", required=True)
    args = parser.parse_args(argv)
    receipt = audit_development_matrix(
        matrix_directory=args.matrix_directory,
        output=args.output,
        case_manifest_path=args.case_manifest,
        reference_manifest_path=args.reference_manifest,
        config_manifest_path=args.config_manifest,
        metric_manifest_path=args.metric_manifest,
        protocol_path=args.protocol,
        authorization_path=args.authorization_receipt,
        source_snapshot_root_sha256=args.source_snapshot_root_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
