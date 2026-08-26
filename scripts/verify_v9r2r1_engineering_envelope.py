"""Independently verify a V9R2R1 scoped engineering recovery envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET


ENVELOPE_SCHEMA = "v21e3r1_v9r2r1_engineering_recovery_envelope_v1"
VERIFICATION_SCHEMA = "v21e3r1_v9r2r1_engineering_envelope_verification_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class EngineeringEnvelopeVerificationError(ValueError):
    """Raised when the envelope or a bound artifact drifts."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EngineeringEnvelopeVerificationError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise EngineeringEnvelopeVerificationError(
        f"non-finite JSON value prohibited: {value}"
    )


def _strict_json(path: Path) -> tuple[dict[str, object], bytes]:
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EngineeringEnvelopeVerificationError(
            f"invalid strict JSON: {path}"
        ) from error
    if type(payload) is not dict or raw != _canonical_json(payload) + b"\n":
        raise EngineeringEnvelopeVerificationError(
            f"JSON must be a canonical object plus one newline: {path}"
        )
    return payload, raw


def _validate_self_hash(
    payload: Mapping[str, object], field: str, *, label: str
) -> None:
    declared = payload.get(field)
    if type(declared) is not str or not _SHA256_RE.fullmatch(declared):
        raise EngineeringEnvelopeVerificationError(f"invalid {label} self-hash")
    core = dict(payload)
    del core[field]
    if declared != _sha256(_canonical_json(core)):
        raise EngineeringEnvelopeVerificationError(f"{label} self-hash mismatch")


def _safe_artifact_path(root: Path, value: object) -> Path:
    if type(value) is not str or not value or "\\" in value:
        raise EngineeringEnvelopeVerificationError("unsafe artifact path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value or ".." in relative.parts:
        raise EngineeringEnvelopeVerificationError(f"unsafe artifact path: {value!r}")
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise EngineeringEnvelopeVerificationError(
            f"artifact escapes project root: {value!r}"
        ) from error
    if not candidate.is_file() or candidate.is_symlink():
        raise EngineeringEnvelopeVerificationError(
            f"artifact must be a regular file: {value!r}"
        )
    return candidate


def _junit_counts(raw: bytes) -> dict[str, int]:
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise EngineeringEnvelopeVerificationError(
            "DTD/entity declarations are prohibited"
        )
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise EngineeringEnvelopeVerificationError("invalid JUnit XML") from error
    testcases = list(root.iter("testcase"))
    failures = sum(bool(case.findall("failure")) for case in testcases)
    errors = sum(bool(case.findall("error")) for case in testcases)
    skipped = sum(bool(case.findall("skipped")) for case in testcases)
    if any(
        len(case.findall("failure"))
        + len(case.findall("error"))
        + len(case.findall("skipped"))
        > 1
        for case in testcases
    ):
        raise EngineeringEnvelopeVerificationError(
            "JUnit testcase has multiple outcomes"
        )
    return {
        "testcases": len(testcases),
        "passed": len(testcases) - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _require_false(payload: Mapping[str, object], fields: Sequence[str], label: str) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise EngineeringEnvelopeVerificationError(
                f"{label} boundary drifted: {field}"
            )


def verify_engineering_envelope(
    envelope_path: Path, *, root: Path
) -> dict[str, object]:
    root = root.resolve(strict=True)
    envelope_path = envelope_path.resolve(strict=True)
    envelope, envelope_raw = _strict_json(envelope_path)
    required_keys = {
        "schema",
        "status",
        "identity",
        "artifacts",
        "source",
        "environment",
        "junit",
        "expected_historical_v8_failure_contract",
        "full_repository_contract_status",
        "repository_wide_green",
        "environment_lock_satisfied",
        "scientific_stage_authorized",
        "full_development_matrix_authorized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "ijoc_submission_authorized",
        "envelope_payload_sha256",
    }
    if set(envelope) != required_keys:
        raise EngineeringEnvelopeVerificationError("envelope key set drifted")
    if (
        envelope.get("schema") != ENVELOPE_SCHEMA
        or envelope.get("status")
        != "PASS_SCOPED_ENGINEERING_RECOVERY_ENVELOPE_ONLY"
        or envelope.get("full_repository_contract_status")
        != "EXPECTED_8_FROZEN_V8_FAILURES"
    ):
        raise EngineeringEnvelopeVerificationError("envelope status/schema drifted")
    _validate_self_hash(envelope, "envelope_payload_sha256", label="envelope")
    _require_false(
        envelope,
        (
            "repository_wide_green",
            "environment_lock_satisfied",
            "scientific_stage_authorized",
            "full_development_matrix_authorized",
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
        ),
        "envelope",
    )

    artifacts = envelope.get("artifacts")
    expected_artifact_names = {
        "source_manifest",
        "environment_preflight",
        "pymoo_environment_recovery_junit",
        "targeted_junit",
        "full_repository_junit",
        "expected_failure_registry",
        "expected_failure_receipt",
    }
    if type(artifacts) is not dict or set(artifacts) != expected_artifact_names:
        raise EngineeringEnvelopeVerificationError("artifact set drifted")
    artifact_bytes: dict[str, bytes] = {}
    artifact_paths: dict[str, Path] = {}
    for name, entry in artifacts.items():
        if type(entry) is not dict or set(entry) != {"path", "bytes", "sha256"}:
            raise EngineeringEnvelopeVerificationError(
                f"artifact entry shape drifted: {name}"
            )
        path = _safe_artifact_path(root, entry["path"])
        raw = path.read_bytes()
        if (
            type(entry["bytes"]) is not int
            or len(raw) != entry["bytes"]
            or type(entry["sha256"]) is not str
            or not _SHA256_RE.fullmatch(entry["sha256"])
            or _sha256(raw) != entry["sha256"]
        ):
            raise EngineeringEnvelopeVerificationError(
                f"artifact bytes/hash drifted: {name}"
            )
        artifact_bytes[name] = raw
        artifact_paths[name] = path

    source, _ = _strict_json(artifact_paths["source_manifest"])
    preflight, _ = _strict_json(artifact_paths["environment_preflight"])
    registry, _ = _strict_json(artifact_paths["expected_failure_registry"])
    failure_receipt, _ = _strict_json(
        artifact_paths["expected_failure_receipt"]
    )
    _validate_self_hash(source, "manifest_payload_sha256", label="source manifest")
    _validate_self_hash(preflight, "receipt_payload_sha256", label="preflight")
    _validate_self_hash(registry, "manifest_payload_sha256", label="failure registry")
    _validate_self_hash(
        failure_receipt, "receipt_payload_sha256", label="failure receipt"
    )

    source_record = envelope.get("source")
    if type(source_record) is not dict or source_record != {
        "file_count": source.get("file_count"),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "manifest_payload_sha256": source.get("manifest_payload_sha256"),
        "full_source_freeze_requirement_satisfied": False,
    }:
        raise EngineeringEnvelopeVerificationError("source cross-binding drifted")
    if source_record["file_count"] != 203:
        raise EngineeringEnvelopeVerificationError("source count is not 203")

    environment = envelope.get("environment")
    interpreter = preflight.get("interpreter")
    if type(environment) is not dict or type(interpreter) is not dict:
        raise EngineeringEnvelopeVerificationError("environment record missing")
    if environment != {
        "python_executable": interpreter.get("observed_executable"),
        "python_version": interpreter.get("observed_version"),
        "preflight_receipt_payload_sha256": preflight.get(
            "receipt_payload_sha256"
        ),
        "environment_lock_satisfied": False,
    }:
        raise EngineeringEnvelopeVerificationError("environment cross-binding drifted")
    if (
        preflight.get("status") != "PASS_FULL_SUITE_ENVIRONMENT_PREFLIGHT"
        or preflight.get("full_suite_execution_preflight_passed") is not True
        or preflight.get("environment_lock_requirement_satisfied") is not False
    ):
        raise EngineeringEnvelopeVerificationError("preflight boundary drifted")

    junit = envelope.get("junit")
    if type(junit) is not dict or set(junit) != {
        "pymoo_environment_recovery",
        "targeted",
        "full_repository",
    }:
        raise EngineeringEnvelopeVerificationError("JUnit record set drifted")
    observed_junit = {
        "pymoo_environment_recovery": _junit_counts(
            artifact_bytes["pymoo_environment_recovery_junit"]
        ),
        "targeted": _junit_counts(artifact_bytes["targeted_junit"]),
        "full_repository": _junit_counts(
            artifact_bytes["full_repository_junit"]
        ),
    }
    if junit != observed_junit:
        raise EngineeringEnvelopeVerificationError("JUnit counts drifted")
    for label in ("pymoo_environment_recovery", "targeted"):
        if junit[label]["failures"] != 0 or junit[label]["errors"] != 0:
            raise EngineeringEnvelopeVerificationError(f"{label} is not green")

    contract = envelope.get("expected_historical_v8_failure_contract")
    if type(contract) is not dict or contract != {
        "historical_source_root_sha256": registry.get(
            "historical_source_root_sha256"
        ),
        "expected_failure_count": 8,
        "exact_node_ids": registry.get("node_ids"),
        "xfail_allowed": False,
        "verification_receipt_payload_sha256": failure_receipt.get(
            "receipt_payload_sha256"
        ),
    }:
        raise EngineeringEnvelopeVerificationError(
            "historical failure contract cross-binding drifted"
        )
    if source.get("source_tree_sha256") != registry.get(
        "current_source_root_sha256"
    ):
        raise EngineeringEnvelopeVerificationError("source-to-registry root drifted")
    if (
        failure_receipt.get("status")
        != "PASS_EXACT_EXPECTED_HISTORICAL_V8_FAILURE_SET"
        or failure_receipt.get("exact_failure_node_ids") != registry.get("node_ids")
        or failure_receipt.get("registry", {}).get("sha256")
        != artifacts["expected_failure_registry"]["sha256"]
        or failure_receipt.get("junit", {}).get("sha256")
        != artifacts["full_repository_junit"]["sha256"]
    ):
        raise EngineeringEnvelopeVerificationError("failure receipt binding drifted")
    if any(
        junit["full_repository"].get(field)
        != registry.get("expected_counts", {}).get(field)
        for field in ("testcases", "passed", "failures", "errors", "skipped")
    ):
        raise EngineeringEnvelopeVerificationError(
            "full repository count contract drifted"
        )

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "PASS_VERIFIED_SCOPED_ENGINEERING_RECOVERY_ENVELOPE_ONLY",
        "envelope_bytes": len(envelope_raw),
        "envelope_sha256": _sha256(envelope_raw),
        "envelope_payload_sha256": envelope["envelope_payload_sha256"],
        "artifact_count": len(artifacts),
        "exact_expected_failure_count": 8,
        "repository_wide_green": False,
        "environment_lock_satisfied": False,
        "scientific_stage_authorized": False,
    }


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    raw = _canonical_json(payload) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise EngineeringEnvelopeVerificationError(
            f"refusing to overwrite: {path}"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = verify_engineering_envelope(args.envelope, root=args.root)
        if args.output is not None:
            _write_exclusive(args.output, receipt)
    except (EngineeringEnvelopeVerificationError, FileNotFoundError, OSError) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENVELOPE_SCHEMA",
    "EngineeringEnvelopeVerificationError",
    "VERIFICATION_SCHEMA",
    "main",
    "verify_engineering_envelope",
]
