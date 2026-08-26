"""Build a fail-closed V9R2R1 scoped engineering recovery envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET


ENVELOPE_SCHEMA = "v21e3r1_v9r2r1_engineering_recovery_envelope_v1"
PASS_STATUS = "PASS_SCOPED_ENGINEERING_RECOVERY_ENVELOPE_ONLY"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class EngineeringEnvelopeError(ValueError):
    """Raised when an input or cross-binding is incomplete or inconsistent."""


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
            raise EngineeringEnvelopeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise EngineeringEnvelopeError(f"non-finite JSON value prohibited: {value}")


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
        raise EngineeringEnvelopeError(f"invalid strict JSON: {path}") from error
    if type(payload) is not dict:
        raise EngineeringEnvelopeError(f"JSON input must be an object: {path}")
    if raw != _canonical_json(payload) + b"\n":
        raise EngineeringEnvelopeError(
            f"JSON input is not canonical plus one newline: {path}"
        )
    return payload, raw


def _validate_self_hash(
    payload: Mapping[str, object], field: str, *, label: str
) -> None:
    declared = payload.get(field)
    if type(declared) is not str or not _SHA256_RE.fullmatch(declared):
        raise EngineeringEnvelopeError(f"invalid {label} self-hash")
    core = dict(payload)
    del core[field]
    if declared != _sha256(_canonical_json(core)):
        raise EngineeringEnvelopeError(f"{label} self-hash mismatch")


def _artifact(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise EngineeringEnvelopeError(f"artifact must be a regular file: {resolved}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise EngineeringEnvelopeError(
            f"artifact is outside the declared project root: {resolved}"
        ) from error
    raw = resolved.read_bytes()
    return {
        "path": relative.as_posix(),
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _junit_counts(path: Path) -> dict[str, int]:
    raw = path.resolve(strict=True).read_bytes()
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise EngineeringEnvelopeError("DTD/entity declarations are prohibited")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise EngineeringEnvelopeError(f"invalid JUnit XML: {path}") from error
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
        raise EngineeringEnvelopeError("JUnit testcase has multiple outcomes")
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
            raise EngineeringEnvelopeError(f"{label} boundary drifted: {field}")


def build_engineering_envelope(
    *,
    root: Path,
    source_manifest: Path,
    environment_preflight: Path,
    pymoo_junit: Path,
    targeted_junit: Path,
    full_repository_junit: Path,
    expected_failure_registry: Path,
    expected_failure_receipt: Path,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise EngineeringEnvelopeError(f"not a project root: {root}")

    source, _source_raw = _strict_json(source_manifest)
    preflight, _preflight_raw = _strict_json(environment_preflight)
    registry, _registry_raw = _strict_json(expected_failure_registry)
    failure_receipt, _failure_receipt_raw = _strict_json(expected_failure_receipt)
    _validate_self_hash(source, "manifest_payload_sha256", label="source manifest")
    _validate_self_hash(preflight, "receipt_payload_sha256", label="preflight")
    _validate_self_hash(registry, "manifest_payload_sha256", label="failure registry")
    _validate_self_hash(
        failure_receipt, "receipt_payload_sha256", label="failure receipt"
    )

    identity = {
        "distribution": "mo-nco",
        "revision": "V21E3R1_V9R2R1",
        "version": "0.21.3.14",
    }
    source_identity = source.get("identity")
    if type(source_identity) is not dict or any(
        source_identity.get(key) != value for key, value in identity.items()
    ):
        raise EngineeringEnvelopeError("source identity drifted")
    if preflight.get("identity") != identity or registry.get("identity") != identity:
        raise EngineeringEnvelopeError("preflight/registry identity drifted")
    if source.get("file_count") != 203:
        raise EngineeringEnvelopeError("source manifest must bind exactly 203 files")
    source_root = source.get("source_tree_sha256")
    if (
        type(source_root) is not str
        or not _SHA256_RE.fullmatch(source_root)
        or registry.get("current_source_root_sha256") != source_root
    ):
        raise EngineeringEnvelopeError("current source root cross-binding drifted")

    if (
        preflight.get("status") != "PASS_FULL_SUITE_ENVIRONMENT_PREFLIGHT"
        or preflight.get("full_suite_execution_preflight_passed") is not True
    ):
        raise EngineeringEnvelopeError("full-suite environment preflight is not PASS")
    _require_false(
        preflight,
        (
            "environment_lock_requirement_satisfied",
            "full_development_matrix_authorized",
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
        ),
        "preflight",
    )
    _require_false(
        registry,
        (
            "repository_wide_green",
            "scientific_stage_authorized",
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
        ),
        "failure registry",
    )
    if registry.get("xfail_allowed") is not False:
        raise EngineeringEnvelopeError("failure registry must prohibit xfail")
    if (
        failure_receipt.get("status")
        != "PASS_EXACT_EXPECTED_HISTORICAL_V8_FAILURE_SET"
    ):
        raise EngineeringEnvelopeError("historical failure receipt is not PASS")
    _require_false(
        failure_receipt,
        (
            "repository_wide_green",
            "scientific_stage_authorized",
            "selection_authorized",
            "confirmation_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
        ),
        "failure receipt",
    )

    artifacts = {
        "source_manifest": _artifact(source_manifest, root),
        "environment_preflight": _artifact(environment_preflight, root),
        "pymoo_environment_recovery_junit": _artifact(pymoo_junit, root),
        "targeted_junit": _artifact(targeted_junit, root),
        "full_repository_junit": _artifact(full_repository_junit, root),
        "expected_failure_registry": _artifact(expected_failure_registry, root),
        "expected_failure_receipt": _artifact(expected_failure_receipt, root),
    }
    if failure_receipt.get("registry", {}).get("sha256") != artifacts[
        "expected_failure_registry"
    ]["sha256"]:
        raise EngineeringEnvelopeError("failure receipt registry hash drifted")
    if failure_receipt.get("junit", {}).get("sha256") != artifacts[
        "full_repository_junit"
    ]["sha256"]:
        raise EngineeringEnvelopeError("failure receipt full JUnit hash drifted")
    if failure_receipt.get("exact_failure_node_ids") != registry.get("node_ids"):
        raise EngineeringEnvelopeError("failure receipt node-id set drifted")

    junit = {
        "pymoo_environment_recovery": _junit_counts(pymoo_junit),
        "targeted": _junit_counts(targeted_junit),
        "full_repository": _junit_counts(full_repository_junit),
    }
    for label in ("pymoo_environment_recovery", "targeted"):
        if junit[label]["failures"] != 0 or junit[label]["errors"] != 0:
            raise EngineeringEnvelopeError(f"{label} JUnit is not green")
    expected_counts = registry.get("expected_counts")
    if type(expected_counts) is not dict or any(
        junit["full_repository"].get(field) != expected_counts.get(field)
        for field in ("testcases", "passed", "failures", "errors", "skipped")
    ):
        raise EngineeringEnvelopeError("full repository JUnit count contract drifted")

    interpreter = preflight.get("interpreter")
    if type(interpreter) is not dict:
        raise EngineeringEnvelopeError("preflight interpreter record missing")
    observed_executable = interpreter.get("observed_executable")
    observed_version = interpreter.get("observed_version")
    if type(observed_executable) is not str or type(observed_version) is not str:
        raise EngineeringEnvelopeError("preflight interpreter identity invalid")

    core: dict[str, object] = {
        "schema": ENVELOPE_SCHEMA,
        "status": PASS_STATUS,
        "identity": identity,
        "artifacts": artifacts,
        "source": {
            "file_count": source["file_count"],
            "source_tree_sha256": source_root,
            "manifest_payload_sha256": source["manifest_payload_sha256"],
            "full_source_freeze_requirement_satisfied": False,
        },
        "environment": {
            "python_executable": observed_executable,
            "python_version": observed_version,
            "preflight_receipt_payload_sha256": preflight[
                "receipt_payload_sha256"
            ],
            "environment_lock_satisfied": False,
        },
        "junit": junit,
        "expected_historical_v8_failure_contract": {
            "historical_source_root_sha256": registry[
                "historical_source_root_sha256"
            ],
            "expected_failure_count": 8,
            "exact_node_ids": registry["node_ids"],
            "xfail_allowed": False,
            "verification_receipt_payload_sha256": failure_receipt[
                "receipt_payload_sha256"
            ],
        },
        "full_repository_contract_status": "EXPECTED_8_FROZEN_V8_FAILURES",
        "repository_wide_green": False,
        "environment_lock_satisfied": False,
        "scientific_stage_authorized": False,
        "full_development_matrix_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }
    return {**core, "envelope_payload_sha256": _sha256(_canonical_json(core))}


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    raw = _canonical_json(payload) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise EngineeringEnvelopeError(f"refusing to overwrite: {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--environment-preflight", type=Path, required=True)
    parser.add_argument("--pymoo-junit", type=Path, required=True)
    parser.add_argument("--targeted-junit", type=Path, required=True)
    parser.add_argument("--full-repository-junit", type=Path, required=True)
    parser.add_argument("--expected-failure-registry", type=Path, required=True)
    parser.add_argument("--expected-failure-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        envelope = build_engineering_envelope(
            root=args.root,
            source_manifest=args.source_manifest,
            environment_preflight=args.environment_preflight,
            pymoo_junit=args.pymoo_junit,
            targeted_junit=args.targeted_junit,
            full_repository_junit=args.full_repository_junit,
            expected_failure_registry=args.expected_failure_registry,
            expected_failure_receipt=args.expected_failure_receipt,
        )
        _write_exclusive(args.output, envelope)
    except (EngineeringEnvelopeError, FileNotFoundError, OSError) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENVELOPE_SCHEMA",
    "EngineeringEnvelopeError",
    "PASS_STATUS",
    "build_engineering_envelope",
    "main",
]
