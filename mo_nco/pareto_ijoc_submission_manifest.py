from __future__ import annotations

"""Fail-closed final-submission artifact manifest and receipt protocol.

Only files named by a canonical required-file specification are hashed.  The
large formal-run tree is deliberately outside this traversal: its row-level
bytes are bound by the post-run and formal-analysis artifacts listed in the
specification.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Mapping, Sequence


REQUIRED_FILES_SCHEMA = "ijoc_final_submission_required_files_v1"
MANIFEST_SCHEMA = "ijoc_final_submission_artifact_manifest_v1"
RECEIPT_SCHEMA = "ijoc_final_submission_verification_receipt_v1"

_AUTHOR_PLACEHOLDER_PATTERNS = (
    (
        "AUTHOR_NAMES_AND_AFFILIATIONS_TO_BE_INSERTED",
        re.compile(
            r"author names? and affiliations? to be inserted",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "CORRESPONDING_AUTHOR_NAME_PLACEHOLDER",
        re.compile(r"corresponding author name", flags=re.IGNORECASE),
    ),
    (
        "EMAIL_ORCID_TO_BE_INSERTED",
        re.compile(r"email and orcid to be inserted", flags=re.IGNORECASE),
    ),
    (
        "AUTHOR_POSTAL_ADDRESS_TO_BE_INSERTED",
        re.compile(r"author postal address to be inserted", flags=re.IGNORECASE),
    ),
    (
        "BRACKETED_AUTHOR_PLACEHOLDER",
        re.compile(
            r"(?:\[|<)(?:author|affiliation|corresponding[ _-]?author|orcid)"
            r"[^\]>]*(?:\]|>)",
            flags=re.IGNORECASE,
        ),
    ),
)

_HOLD_MARKER_PATTERNS = (
    ("HOLD", re.compile(r"(?<![A-Za-z])HOLD(?![A-Za-z])")),
    ("NOT_RUN", re.compile(r"(?<![A-Za-z])NOT(?:_| )RUN(?![A-Za-z])")),
    (
        "NOT_ESTABLISHED",
        re.compile(r"(?<![A-Za-z])NOT(?:_| )ESTABLISHED(?![A-Za-z])"),
    ),
    (
        "NOT_PERFORMED",
        re.compile(r"(?<![A-Za-z])NOT(?:_| )PERFORMED(?![A-Za-z])"),
    ),
    ("OPEN", re.compile(r"(?<![A-Za-z])OPEN(?![A-Za-z])")),
    ("UNKNOWN", re.compile(r"(?<![A-Za-z])UNKNOWN(?![A-Za-z])")),
    (
        "CONDITIONAL",
        re.compile(r"(?<![A-Za-z])CONDITIONAL(?![A-Za-z])"),
    ),
)

_AUTHOR_SOURCE_ROLES = {
    "cover_letter_source",
    "main_manuscript_source",
    "supplement_source",
}

_SOURCE_ARCHIVE_ROLES = {
    "formal_source_archive",
    "release_source_archive",
}


class SubmissionArtifactError(ValueError):
    """Raised when a submission artifact cannot be verified fail-closed."""


@dataclass(frozen=True)
class SubmissionManifestBuild:
    manifest_path: Path
    manifest_sha256: str
    required_file_count: int
    effective_submission_status: str


@dataclass(frozen=True)
class SubmissionReceiptVerification:
    receipt_path: Path
    receipt_sha256: str
    manifest_sha256: str
    required_file_count: int
    effective_submission_status: str


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SubmissionArtifactError(
            "Value is not canonical-JSON serializable."
        ) from error


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SubmissionArtifactError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _load_canonical_json(path: Path, *, label: str) -> tuple[object, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SubmissionArtifactError(f"Cannot read {label}: {path}.") from error
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SubmissionArtifactError(
            f"{label} is not strict UTF-8 JSON."
        ) from error
    if raw != canonical_json_bytes(value):
        raise SubmissionArtifactError(
            f"{label} is not canonical UTF-8 JSON without a trailing newline."
        )
    return value, raw


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SubmissionArtifactError(f"{label} must be a JSON object.")
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise SubmissionArtifactError(f"{label} must be a JSON array.")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise SubmissionArtifactError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}."
        )


def _safe_relative_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionArtifactError(f"{label} must be nonempty text.")
    if "\\" in value:
        raise SubmissionArtifactError(
            f"{label} must use canonical forward slashes."
        )
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise SubmissionArtifactError(f"{label} must be a relative path.")
    if value != posix.as_posix() or any(part in {"", ".", ".."} for part in posix.parts):
        raise SubmissionArtifactError(f"{label} is not a canonical relative path.")
    return value


def _resolve_required_file(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    try:
        candidate = (
            resolved_root / Path(*PurePosixPath(relative_path).parts)
        ).resolve(strict=True)
    except OSError as error:
        raise SubmissionArtifactError(
            f"Required file is missing or unreadable: {relative_path}."
        ) from error
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise SubmissionArtifactError(
            f"Required path escapes packet root: {relative_path}."
        ) from error
    if not candidate.is_file():
        raise SubmissionArtifactError(
            f"Required path is not a regular file: {relative_path}."
        )
    return candidate


def _safe_output_path(root: Path, output_path: str | Path, *, label: str) -> Path:
    resolved_root = root.resolve(strict=True)
    output = Path(output_path)
    if not output.is_absolute():
        output = resolved_root / output
    parent = output.parent.resolve(strict=True)
    try:
        parent.relative_to(resolved_root)
    except ValueError as error:
        raise SubmissionArtifactError(f"{label} escapes packet root.") from error
    return parent / output.name


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _write_atomic(path: Path, raw: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubmissionArtifactError(f"{label} must be nonempty text.")
    return value


def _parse_spec(
    spec_path: Path,
) -> tuple[dict[str, object], bytes, list[dict[str, str]]]:
    raw_value, raw = _load_canonical_json(spec_path, label="required-file spec")
    spec = _mapping(raw_value, label="required-file spec")
    _exact_keys(
        spec,
        {
            "schema",
            "journal_target",
            "declared_submission_status",
            "release_metadata",
            "required_files",
            "readiness_scan",
        },
        label="required-file spec",
    )
    if spec["schema"] != REQUIRED_FILES_SCHEMA:
        raise SubmissionArtifactError("Unexpected required-file spec schema.")
    _text(spec["journal_target"], label="journal_target")
    if spec["declared_submission_status"] not in {"HOLD", "READY"}:
        raise SubmissionArtifactError(
            "declared_submission_status must be HOLD or READY."
        )
    release = _mapping(spec["release_metadata"], label="release_metadata")
    _exact_keys(
        release,
        {"immutable_revision", "public_repository_url", "release_tag"},
        label="release_metadata",
    )
    for key, value in release.items():
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise SubmissionArtifactError(
                f"release_metadata.{key} must be null or nonempty text."
            )

    required: list[dict[str, str]] = []
    observed_paths: set[str] = set()
    observed_casefolded_paths: set[str] = set()
    for index, item_value in enumerate(
        _sequence(spec["required_files"], label="required_files")
    ):
        item = _mapping(item_value, label=f"required_files[{index}]")
        _exact_keys(
            item, {"artifact_role", "path"}, label=f"required_files[{index}]"
        )
        path = _safe_relative_text(
            item["path"], label=f"required_files[{index}].path"
        )
        role = _text(
            item["artifact_role"], label=f"required_files[{index}].artifact_role"
        )
        if re.fullmatch(r"[a-z][a-z0-9_]*", role) is None:
            raise SubmissionArtifactError(
                f"required_files[{index}].artifact_role is not lower_snake_case."
            )
        if path in observed_paths or path.casefold() in observed_casefolded_paths:
            raise SubmissionArtifactError(f"Duplicate required path: {path}.")
        observed_paths.add(path)
        observed_casefolded_paths.add(path.casefold())
        required.append({"artifact_role": role, "path": path})
    if not required:
        raise SubmissionArtifactError("required_files must not be empty.")

    readiness = _mapping(spec["readiness_scan"], label="readiness_scan")
    _exact_keys(
        readiness,
        {"author_placeholder_paths", "hold_marker_paths"},
        label="readiness_scan",
    )
    for field in ("author_placeholder_paths", "hold_marker_paths"):
        seen: set[str] = set()
        for index, value in enumerate(
            _sequence(readiness[field], label=f"readiness_scan.{field}")
        ):
            path = _safe_relative_text(
                value, label=f"readiness_scan.{field}[{index}]"
            )
            if path not in observed_paths:
                raise SubmissionArtifactError(
                    f"readiness_scan.{field} path is not required: {path}."
                )
            if path in seen:
                raise SubmissionArtifactError(
                    f"Duplicate readiness-scan path in {field}: {path}."
                )
            seen.add(path)
    return spec, raw, required


def _scan_text_markers(
    root: Path,
    paths: Sequence[object],
    patterns: Sequence[tuple[str, re.Pattern[str]]],
    *,
    label: str,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path_value in sorted(str(value) for value in paths):
        artifact = _resolve_required_file(root, path_value)
        try:
            text = artifact.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise SubmissionArtifactError(
                f"{label} target is not UTF-8 text: {path_value}."
            ) from error
        for line_number, line in enumerate(text.splitlines(), start=1):
            for marker, pattern in patterns:
                if pattern.search(line):
                    findings.append(
                        {
                            "line": line_number,
                            "marker": marker,
                            "path": path_value,
                        }
                    )
    return findings


def _readiness_status(
    root: Path, spec: Mapping[str, object]
) -> dict[str, object]:
    release = _mapping(spec["release_metadata"], label="release_metadata")
    declared = str(spec["declared_submission_status"])
    scan = _mapping(spec["readiness_scan"], label="readiness_scan")
    required = [
        _mapping(value, label="required_files item")
        for value in _sequence(spec["required_files"], label="required_files")
    ]
    mandatory_author_paths = {
        str(item["path"])
        for item in required
        if item.get("artifact_role") in _AUTHOR_SOURCE_ROLES
    }
    mandatory_hold_paths = {
        str(item["path"])
        for item in required
        if item.get("artifact_role") in _AUTHOR_SOURCE_ROLES
        or str(item.get("artifact_role", "")).endswith("_status")
    }
    author_paths = mandatory_author_paths | {
        str(value)
        for value in _sequence(
            scan["author_placeholder_paths"],
            label="readiness_scan.author_placeholder_paths",
        )
    }
    hold_paths = mandatory_hold_paths | {
        str(value)
        for value in _sequence(
            scan["hold_marker_paths"],
            label="readiness_scan.hold_marker_paths",
        )
    }
    author_findings = _scan_text_markers(
        root,
        sorted(author_paths),
        _AUTHOR_PLACEHOLDER_PATTERNS,
        label="author-placeholder scan",
    )
    hold_findings = _scan_text_markers(
        root,
        sorted(hold_paths),
        _HOLD_MARKER_PATTERNS,
        label="hold-marker scan",
    )
    public_repository_status = (
        "PRESENT" if release["public_repository_url"] else "MISSING"
    )
    immutable_revision_status = (
        "PRESENT" if release["immutable_revision"] else "MISSING"
    )
    release_tag_status = "PRESENT" if release["release_tag"] else "MISSING"
    blocking_reasons: list[str] = []
    if declared == "HOLD":
        blocking_reasons.append("DECLARED_SUBMISSION_STATUS_HOLD")
    if author_findings:
        blocking_reasons.append("AUTHOR_PLACEHOLDERS_PRESENT")
    if hold_findings:
        blocking_reasons.append("HOLD_MARKERS_PRESENT")
    if public_repository_status == "MISSING":
        blocking_reasons.append("PUBLIC_REPOSITORY_MISSING")
    if immutable_revision_status == "MISSING":
        blocking_reasons.append("IMMUTABLE_REVISION_MISSING")
    if release_tag_status == "MISSING":
        blocking_reasons.append("RELEASE_TAG_MISSING")
    effective = "HOLD" if blocking_reasons else "READY"
    return {
        "author_placeholders": {
            "finding_count": len(author_findings),
            "findings": author_findings,
            "status": "PRESENT" if author_findings else "ABSENT",
        },
        "blocking_reasons": blocking_reasons,
        "declared_submission_status": declared,
        "effective_submission_status": effective,
        "hold_markers": {
            "finding_count": len(hold_findings),
            "findings": hold_findings,
            "status": "PRESENT" if hold_findings else "ABSENT",
        },
        "immutable_revision_status": immutable_revision_status,
        "public_repository_status": public_repository_status,
        "release_tag_status": release_tag_status,
    }


def _build_manifest_payload(
    root: Path, spec_path: Path
) -> tuple[dict[str, object], bytes]:
    spec, spec_raw, required = _parse_spec(spec_path)
    artifacts: list[dict[str, object]] = []
    for item in required:
        path = _resolve_required_file(root, item["path"])
        size, sha256 = _sha256_file(path)
        artifacts.append(
            {
                "artifact_role": item["artifact_role"],
                "bytes": size,
                "path": item["path"],
                "sha256": sha256,
            }
        )
    artifacts.sort(key=lambda item: str(item["path"]))
    payload = {
        "artifacts": artifacts,
        "coverage": {
            "raw_formal_run_tree_hashed": False,
            "row_level_formal_artifacts": (
                "BOUND_BY_LISTED_POSTRUN_AND_ANALYSIS_MANIFESTS"
            ),
        },
        "journal_target": spec["journal_target"],
        "readiness_status": _readiness_status(root, spec),
        "release_metadata": spec["release_metadata"],
        "required_file_count": len(artifacts),
        "required_file_spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "schema": MANIFEST_SCHEMA,
    }
    return payload, canonical_json_bytes(payload)


def build_submission_manifest(
    packet_root: str | Path,
    required_file_spec: str | Path,
    output_path: str | Path,
) -> SubmissionManifestBuild:
    """Build one canonical manifest from an explicit required-file spec."""

    root = Path(packet_root)
    if not root.is_dir():
        raise SubmissionArtifactError(f"Packet root is not a directory: {root}.")
    output = _safe_output_path(root, output_path, label="manifest output")
    spec_path = Path(required_file_spec)
    try:
        resolved_spec = spec_path.resolve(strict=True)
    except OSError as error:
        raise SubmissionArtifactError(
            f"Cannot read required-file spec: {spec_path}."
        ) from error
    if output.resolve(strict=False) == resolved_spec:
        raise SubmissionArtifactError(
            "Manifest output cannot overwrite the required-file spec."
        )
    payload, raw = _build_manifest_payload(root, spec_path)
    listed_paths = {str(item["path"]) for item in payload["artifacts"]}
    output_relative = output.relative_to(root.resolve(strict=True)).as_posix()
    if output_relative in listed_paths:
        raise SubmissionArtifactError(
            "Manifest output cannot be one of its own required files."
        )
    _write_atomic(output, raw)
    readiness = _mapping(payload["readiness_status"], label="readiness_status")
    return SubmissionManifestBuild(
        manifest_path=output,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        required_file_count=int(payload["required_file_count"]),
        effective_submission_status=str(
            readiness["effective_submission_status"]
        ),
    )


def _relative_existing_file(root: Path, path: str | Path, *, label: str) -> tuple[Path, str]:
    resolved_root = root.resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise SubmissionArtifactError(f"Cannot read {label}: {candidate}.") from error
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise SubmissionArtifactError(f"{label} escapes packet root.") from error
    if not resolved.is_file():
        raise SubmissionArtifactError(f"{label} is not a regular file.")
    return resolved, relative


def _verified_at_text(value: str | None) -> str:
    if value is None:
        timestamp = datetime.now(timezone.utc)
    else:
        if not isinstance(value, str) or not value:
            raise SubmissionArtifactError("verified_at must be nonempty text.")
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise SubmissionArtifactError(
                "verified_at must be an ISO-8601 timestamp."
            ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise SubmissionArtifactError("verified_at must include a timezone.")
    return timestamp.isoformat(timespec="seconds")


def _receipt_payload(
    *,
    manifest_payload: Mapping[str, object],
    manifest_relative_path: str,
    manifest_raw: bytes,
    verifier_role: str,
    verified_at: str,
) -> dict[str, object]:
    artifacts = _sequence(manifest_payload["artifacts"], label="manifest.artifacts")
    final_pdf_roles = {
        "main_manuscript_pdf",
        "supplement_pdf",
        "cover_letter_pdf",
    }
    final_pdfs = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("artifact_role") in final_pdf_roles
    ]
    source_archives = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and item.get("artifact_role") in _SOURCE_ARCHIVE_ROLES
    ]
    module_path = Path(__file__).resolve(strict=True)
    module_bytes, module_sha256 = _sha256_file(module_path)
    readiness = _mapping(
        manifest_payload["readiness_status"], label="manifest.readiness_status"
    )
    return {
        "final_pdf_artifacts": final_pdfs,
        "independent_recomputation": {
            "all_required_files_rehashed": True,
            "canonical_manifest_rebuilt": True,
            "raw_formal_run_tree_rehashed": False,
            "row_level_binding": (
                "DELEGATED_TO_LISTED_POSTRUN_AND_ANALYSIS_MANIFESTS"
            ),
        },
        "integrity_status": "PASS",
        "journal_target": manifest_payload["journal_target"],
        "manifest": {
            "bytes": len(manifest_raw),
            "path": manifest_relative_path,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
        "readiness_status": readiness,
        "release_metadata": manifest_payload["release_metadata"],
        "required_file_count": manifest_payload["required_file_count"],
        "required_file_spec_sha256": manifest_payload[
            "required_file_spec_sha256"
        ],
        "schema": RECEIPT_SCHEMA,
        "source_archive_artifacts": source_archives,
        "submission_status": readiness["effective_submission_status"],
        "verified_at": verified_at,
        "verification_implementation": {
            "cryptographic_signature_status": "NOT_PERFORMED",
            "independence_attestation_scope": (
                "VERIFIER_ROLE_DECLARATION_ONLY"
            ),
            "module": "mo_nco.pareto_ijoc_submission_manifest",
            "module_bytes": module_bytes,
            "module_sha256": module_sha256,
        },
        "verifier_role": verifier_role,
    }


def _recompute_and_match_manifest(
    root: Path,
    spec_path: Path,
    manifest_path: str | Path,
) -> tuple[dict[str, object], bytes, str]:
    resolved_manifest, manifest_relative = _relative_existing_file(
        root, manifest_path, label="manifest"
    )
    parsed, manifest_raw = _load_canonical_json(
        resolved_manifest, label="submission manifest"
    )
    manifest_payload = _mapping(parsed, label="submission manifest")
    expected_payload, expected_raw = _build_manifest_payload(root, spec_path)
    if manifest_raw != expected_raw:
        raise SubmissionArtifactError(
            "Submission manifest does not match freshly recomputed required-file "
            "bytes (hash drift or manifest metadata drift)."
        )
    if manifest_payload != expected_payload:
        raise SubmissionArtifactError("Submission manifest semantic drift detected.")
    return manifest_payload, manifest_raw, manifest_relative


def verify_submission_manifest(
    packet_root: str | Path,
    required_file_spec: str | Path,
    manifest_path: str | Path,
    receipt_output_path: str | Path,
    *,
    verifier_role: str,
    verified_at: str | None = None,
) -> SubmissionReceiptVerification:
    """Independently recompute a manifest and write a canonical receipt."""

    root = Path(packet_root)
    if not root.is_dir():
        raise SubmissionArtifactError(f"Packet root is not a directory: {root}.")
    role = _text(verifier_role, label="verifier_role")
    timestamp = _verified_at_text(verified_at)
    output = _safe_output_path(root, receipt_output_path, label="receipt output")
    spec_path = Path(required_file_spec)
    try:
        resolved_spec = spec_path.resolve(strict=True)
    except OSError as error:
        raise SubmissionArtifactError(
            f"Cannot read required-file spec: {spec_path}."
        ) from error
    if output.resolve(strict=False) == resolved_spec:
        raise SubmissionArtifactError(
            "Receipt output cannot overwrite the required-file spec."
        )
    manifest_payload, manifest_raw, manifest_relative = (
        _recompute_and_match_manifest(root, spec_path, manifest_path)
    )
    receipt_payload = _receipt_payload(
        manifest_payload=manifest_payload,
        manifest_relative_path=manifest_relative,
        manifest_raw=manifest_raw,
        verifier_role=role,
        verified_at=timestamp,
    )
    receipt_raw = canonical_json_bytes(receipt_payload)
    required_paths = {
        str(item["path"])
        for item in _sequence(manifest_payload["artifacts"], label="artifacts")
        if isinstance(item, dict)
    }
    output_relative = output.relative_to(root.resolve(strict=True)).as_posix()
    if output_relative in required_paths or output_relative == manifest_relative:
        raise SubmissionArtifactError(
            "Receipt output cannot overwrite a required file or the manifest."
        )
    _write_atomic(output, receipt_raw)
    readiness = _mapping(
        manifest_payload["readiness_status"], label="readiness_status"
    )
    return SubmissionReceiptVerification(
        receipt_path=output,
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        required_file_count=int(manifest_payload["required_file_count"]),
        effective_submission_status=str(
            readiness["effective_submission_status"]
        ),
    )


def audit_submission_receipt(
    packet_root: str | Path,
    required_file_spec: str | Path,
    manifest_path: str | Path,
    receipt_path: str | Path,
) -> SubmissionReceiptVerification:
    """Recompute and audit an existing receipt without mutating the packet."""

    root = Path(packet_root)
    if not root.is_dir():
        raise SubmissionArtifactError(f"Packet root is not a directory: {root}.")
    resolved_receipt, _ = _relative_existing_file(
        root, receipt_path, label="receipt"
    )
    parsed_receipt, receipt_raw = _load_canonical_json(
        resolved_receipt, label="submission receipt"
    )
    receipt = _mapping(parsed_receipt, label="submission receipt")
    _exact_keys(
        receipt,
        {
            "final_pdf_artifacts",
            "independent_recomputation",
            "integrity_status",
            "journal_target",
            "manifest",
            "readiness_status",
            "release_metadata",
            "required_file_count",
            "required_file_spec_sha256",
            "schema",
            "source_archive_artifacts",
            "submission_status",
            "verified_at",
            "verification_implementation",
            "verifier_role",
        },
        label="submission receipt",
    )
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise SubmissionArtifactError("Unexpected submission receipt schema.")
    if receipt["integrity_status"] != "PASS":
        raise SubmissionArtifactError("Receipt integrity_status is not PASS.")
    verifier_role = _text(receipt["verifier_role"], label="receipt.verifier_role")
    verified_at = _verified_at_text(
        _text(receipt["verified_at"], label="receipt.verified_at")
    )
    manifest_payload, manifest_raw, manifest_relative = (
        _recompute_and_match_manifest(
            root, Path(required_file_spec), manifest_path
        )
    )
    expected = _receipt_payload(
        manifest_payload=manifest_payload,
        manifest_relative_path=manifest_relative,
        manifest_raw=manifest_raw,
        verifier_role=verifier_role,
        verified_at=verified_at,
    )
    expected_raw = canonical_json_bytes(expected)
    if receipt_raw != expected_raw:
        raise SubmissionArtifactError(
            "Submission receipt does not match independent recomputation "
            "(receipt drift or hash drift)."
        )
    readiness = _mapping(
        manifest_payload["readiness_status"], label="readiness_status"
    )
    return SubmissionReceiptVerification(
        receipt_path=resolved_receipt,
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        required_file_count=int(manifest_payload["required_file_count"]),
        effective_submission_status=str(
            readiness["effective_submission_status"]
        ),
    )
