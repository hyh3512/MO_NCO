"""Fail-closed offline verification for the V9R2R1 GitHub CI envelope.

The v1 envelope records locally downloaded artifact-archive digests and exact
safe member hashes.  It remains an engineering-only record: GitHub action
references and runner images are not artifact-locked, and none of the
scientific or publication gates are opened.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET
import zipfile


ENVELOPE_SCHEMA = "v21e3r1_v9r2r1_github_ci_envelope_v1"
VERIFICATION_SCHEMA = "v21e3r1_v9r2r1_github_ci_envelope_verification_v1"
ENVELOPE_STATUS = (
    "PASS_GITHUB_CI_ARTIFACT_ENVELOPE_ENGINEERING_ONLY__PRE_DEVELOPMENT_HOLD"
)
ARTIFACT_CONTENT_SCOPE = "DOWNLOADED_ZIP_SHA256_AND_SAFE_EXACT_MEMBER_HASHES"

_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_API_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_GREEN_PYTEST_SUMMARY_RE = re.compile(
    rb"(?m)^(?P<passed>\d+) passed"
    rb"(?:, (?P<subtests>\d+) subtests passed)? in [^\r\n]+\r?$"
)
_PLACEHOLDER_RE = re.compile(
    r"(?:TODO|TBD|PLACEHOLDER|UNKNOWN|TO[_ -]?BE[_ -]?FILLED)", re.IGNORECASE
)

_IDENTITY = {
    "distribution": "mo-nco",
    "revision": "V21E3R1_V9R2R1",
    "version": "0.21.3.14",
}
_SUBJECT_BLOB_PATHS = {
    "public_registry": (
        "provenance/V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json"
    ),
    "historical_registry": (
        "provenance/V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json"
    ),
    "historical_junit": (
        "evidence/v9r2r1_environment_recovery_20260825_002/"
        "full_repository.junit.xml"
    ),
    "public_reference_junit": "evidence/public_checkout/full_repository.sanitized.junit.xml",
    "public_reference_log": "evidence/public_checkout/full_repository.sanitized.log",
    "generic_sanitizer": "scripts/sanitize_public_ci_artifacts.py",
    "sanitization_engine": "scripts/sanitize_public_checkout_outputs.py",
    "public_failure_verifier": "scripts/verify_expected_public_checkout_failure_set.py",
    "historical_failure_verifier": "scripts/verify_expected_historical_failure_set.py",
    "installed_gate": "evidence/gate/installed_gate_receipt.json",
}
_WORKFLOWS = {
    "current_source": ".github/workflows/current-source.yml",
    "clean_room_package": ".github/workflows/clean-room-package.yml",
    "repository_contract": ".github/workflows/full-repository-contract.yml",
}
_RUNS = {
    "push_current_source": ("current_source", "push"),
    "push_clean_room": ("clean_room_package", "push"),
    "push_repository_contract": ("repository_contract", "push"),
    "manual_public_live": ("repository_contract", "workflow_dispatch"),
}
_ARTIFACT_NAMES = {
    "push_current_source": {
        "v9r2r1-targeted-sanitized-evidence",
        "v9r2r1-public-backend-sanitized-evidence",
    },
    "push_clean_room": {"v9r2r1-clean-room-package"},
    "push_repository_contract": {
        "v9r2r1-public-checkout-reference-contract",
        "v9r2r1-internal-complete-tree-exact-eight-reference",
    },
    "manual_public_live": {
        "v9r2r1-public-checkout-reference-contract",
        "v9r2r1-internal-complete-tree-exact-eight-reference",
        "v9r2r1-public-checkout-live-contract",
    },
}
_ARTIFACT_MEMBERS = {
    "v9r2r1-targeted-sanitized-evidence": {
        "targeted.sanitization.json": "sanitization_receipt",
        "targeted.sanitized.junit.xml": "targeted_junit",
        "targeted.sanitized.log": "targeted_log",
    },
    "v9r2r1-public-backend-sanitized-evidence": {
        "full_suite_environment_preflight.sanitized.json": "environment_preflight",
        "pymoo-recovery.sanitization.json": "sanitization_receipt",
        "pymoo-recovery.sanitized.junit.xml": "backend_junit",
        "pymoo-recovery.sanitized.log": "backend_log",
    },
    "v9r2r1-clean-room-package": {
        "installed-gate.json": "installed_gate",
        "sdist-normalized/mo_nco-0.21.3.14-a.tar.gz": "normalized_sdist",
        "wheel-a/mo_nco-0.21.3.14-py3-none-any.whl": "wheel",
    },
    "v9r2r1-public-checkout-reference-contract": {
        "public-checkout-reference-contract.json": "contract_receipt",
    },
    "v9r2r1-internal-complete-tree-exact-eight-reference": {
        "internal-complete-tree-exact-eight-reference.json": "contract_receipt",
    },
    "v9r2r1-public-checkout-live-contract": {
        "V9R2R1_RAW_OUTPUT_SANITIZATION_RECEIPT.json": (
            "output_sanitization_receipt"
        ),
        "full_repository.sanitized.junit.xml": "full_repository_junit",
        "full_repository.sanitized.log": "full_repository_log",
        "public-checkout-live-contract.sanitized.json": "contract_receipt",
        "public_checkout_live_environment_preflight.sanitization.json": (
            "preflight_sanitization_receipt"
        ),
        "public_checkout_live_environment_preflight.sanitized.json": (
            "environment_preflight"
        ),
        "public_checkout_live_environment_preflight.sanitized.log": (
            "preflight_log"
        ),
    },
}
_EVIDENCE = {
    "targeted_regression": (
        "push_current_source",
        "v9r2r1-targeted-sanitized-evidence",
        "PASS_SCOPED_TARGETED_REGRESSION_ONLY",
        {
            "junit_sha256",
            "log_sha256",
            "sanitization_receipt_sha256",
        },
    ),
    "public_backend": (
        "push_current_source",
        "v9r2r1-public-backend-sanitized-evidence",
        "PASS_PUBLIC_SYNTHETIC_BACKEND_SEAM_ONLY",
        {
            "environment_preflight_sha256",
            "junit_sha256",
            "log_sha256",
            "sanitization_receipt_sha256",
        },
    ),
    "clean_room_package": (
        "push_clean_room",
        "v9r2r1-clean-room-package",
        "PASS_DETERMINISTIC_PACKAGE_AND_CLEAN_INSTALL_ONLY",
        {"installed_gate_sha256", "normalized_sdist_sha256", "wheel_sha256"},
    ),
    "public_reference": (
        "push_repository_contract",
        "v9r2r1-public-checkout-reference-contract",
        "PASS_REFERENCE_PUBLIC_CHECKOUT_78_FAILURE_CONTRACT",
        {"contract_receipt_sha256", "failure_registry_sha256"},
    ),
    "preserved_internal_exact_eight_reference": (
        "push_repository_contract",
        "v9r2r1-internal-complete-tree-exact-eight-reference",
        "PASS_PRESERVED_INTERNAL_EXACT_EIGHT_REFERENCE_ONLY",
        {
            "contract_receipt_sha256",
            "historical_failure_registry_sha256",
            "reference_junit_sha256",
        },
    ),
    "public_live_failure_contract": (
        "manual_public_live",
        "v9r2r1-public-checkout-live-contract",
        "PASS_LIVE_PUBLIC_CHECKOUT_EXACT_78_FAILURE_CONTRACT",
        {
            "contract_receipt_sha256",
            "environment_preflight_sha256",
            "failure_registry_sha256",
            "junit_sha256",
            "log_sha256",
            "output_sanitization_receipt_sha256",
            "preflight_log_sha256",
            "preflight_sanitization_receipt_sha256",
        },
    ),
}
_INNER_MEMBER_ROLES = {
    "targeted_regression": {
        "junit_sha256": "targeted_junit",
        "log_sha256": "targeted_log",
        "sanitization_receipt_sha256": "sanitization_receipt",
    },
    "public_backend": {
        "environment_preflight_sha256": "environment_preflight",
        "junit_sha256": "backend_junit",
        "log_sha256": "backend_log",
        "sanitization_receipt_sha256": "sanitization_receipt",
    },
    "clean_room_package": {
        "installed_gate_sha256": "installed_gate",
        "normalized_sdist_sha256": "normalized_sdist",
        "wheel_sha256": "wheel",
    },
    "public_reference": {"contract_receipt_sha256": "contract_receipt"},
    "preserved_internal_exact_eight_reference": {
        "contract_receipt_sha256": "contract_receipt"
    },
    "public_live_failure_contract": {
        "contract_receipt_sha256": "contract_receipt",
        "environment_preflight_sha256": "environment_preflight",
        "junit_sha256": "full_repository_junit",
        "log_sha256": "full_repository_log",
        "output_sanitization_receipt_sha256": "output_sanitization_receipt",
        "preflight_log_sha256": "preflight_log",
        "preflight_sanitization_receipt_sha256": (
            "preflight_sanitization_receipt"
        ),
    },
}
_EVIDENCE_JOB_NAMES = {
    "targeted_regression": (
        "Scoped source identity and targeted regression (GitHub CPython 3.11.9)"
    ),
    "public_backend": "Version-pinned pymoo/moocore public backend seam",
    "clean_room_package": (
        "Deterministic package and clean install (GitHub CPython 3.11.9)"
    ),
    "public_reference": "Public checkout expected-failure reference (78 outcomes)",
    "preserved_internal_exact_eight_reference": (
        "Internal complete-tree historical reference (exact eight)"
    ),
    "public_live_failure_contract": "Public checkout live expected-failure contract",
}
_JOB_CONCLUSIONS_BY_RUN = {
    "push_current_source": {
        _EVIDENCE_JOB_NAMES["targeted_regression"]: "success",
        _EVIDENCE_JOB_NAMES["public_backend"]: "success",
    },
    "push_clean_room": {
        _EVIDENCE_JOB_NAMES["clean_room_package"]: "success",
    },
    "push_repository_contract": {
        _EVIDENCE_JOB_NAMES["public_reference"]: "success",
        _EVIDENCE_JOB_NAMES["preserved_internal_exact_eight_reference"]: "success",
        _EVIDENCE_JOB_NAMES["public_live_failure_contract"]: "skipped",
    },
    "manual_public_live": {
        _EVIDENCE_JOB_NAMES["public_reference"]: "success",
        _EVIDENCE_JOB_NAMES["preserved_internal_exact_eight_reference"]: "success",
        _EVIDENCE_JOB_NAMES["public_live_failure_contract"]: "success",
    },
}
_CLAIM_FIELDS = {
    "actions_immutable_commit_refs",
    "artifact_hashed_cross_platform_lock_complete",
    "confirmation_authorized",
    "environment_lock",
    "formal_authorized",
    "full_development_matrix_authorized",
    "github_hosted_runner_image_artifact_lock",
    "github_api_metadata_cryptographically_authenticated",
    "ijoc_submission_authorized",
    "release_assets_authorized",
    "repository_wide_green",
    "scientific_independence",
    "scientific_stage_authorized",
    "selection_authorized",
    "v8_redistribution_authorized",
}


class GitHubCIEnvelopeVerificationError(ValueError):
    """Raised when the GitHub CI envelope is incomplete or inconsistent."""


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
            raise GitHubCIEnvelopeVerificationError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise GitHubCIEnvelopeVerificationError(
        f"non-finite JSON value prohibited: {value}"
    )


def _strict_json(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GitHubCIEnvelopeVerificationError("invalid strict JSON") from error
    if type(payload) is not dict or raw != _canonical_json(payload) + b"\n":
        raise GitHubCIEnvelopeVerificationError(
            "envelope must be a canonical JSON object plus one newline"
        )
    return payload, raw


def _require_keys(
    value: object, expected: set[str], *, label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise GitHubCIEnvelopeVerificationError(f"{label} key set drifted")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GitHubCIEnvelopeVerificationError(f"{label} must be a positive int")
    return value


def _sha1(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA1_RE.fullmatch(value):
        raise GitHubCIEnvelopeVerificationError(f"{label} is not a lowercase SHA-1")
    return value


def _sha256_value(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise GitHubCIEnvelopeVerificationError(
            f"{label} is not a lowercase SHA-256"
        )
    return value


def _utc(value: object, *, label: str) -> str:
    if type(value) is not str or not _UTC_RE.fullmatch(value):
        raise GitHubCIEnvelopeVerificationError(
            f"{label} is not a second-resolution UTC timestamp"
        )
    return value


def _reject_placeholders(value: object, *, path: str = "envelope") -> None:
    if type(value) is dict:
        for key, child in value.items():
            _reject_placeholders(child, path=f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_placeholders(child, path=f"{path}[{index}]")
    elif type(value) is str:
        stripped = value.strip()
        if (
            not stripped
            or stripped != value
            or _PLACEHOLDER_RE.search(value)
            or (value.startswith("<") and value.endswith(">"))
            or value in {"0" * 40, "0" * 64, "sha256:" + "0" * 64}
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"placeholder or blank value prohibited at {path}"
            )


def _safe_external_file(root: Path, value: object) -> Path:
    if type(value) is not str or not value or "\\" in value:
        raise GitHubCIEnvelopeVerificationError("unsafe downloaded ZIP path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value or ".." in relative.parts:
        raise GitHubCIEnvelopeVerificationError(
            f"unsafe downloaded ZIP path: {value!r}"
        )
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise GitHubCIEnvelopeVerificationError(
            f"downloaded ZIP escapes artifact root: {value!r}"
        ) from error
    if not candidate.is_file() or candidate.is_symlink():
        raise GitHubCIEnvelopeVerificationError(
            f"downloaded ZIP must be a regular non-symlink file: {value!r}"
        )
    return candidate


def _verified_zip_members(path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            observed: dict[str, bytes] = {}
            for info in infos:
                name = info.filename
                pure = PurePosixPath(name)
                unix_mode = info.external_attr >> 16
                if (
                    not name
                    or "\\" in name
                    or pure.is_absolute()
                    or pure.as_posix() != name
                    or ".." in pure.parts
                    or info.is_dir()
                    or name in observed
                    or unix_mode & 0o170000 == 0o120000
                    or info.flag_bits & 0x1
                ):
                    raise GitHubCIEnvelopeVerificationError(
                        "unsafe, encrypted, directory, symlink, or duplicate ZIP member"
                    )
                raw = archive.read(info)
                if len(raw) != info.file_size:
                    raise GitHubCIEnvelopeVerificationError(
                        "ZIP member declared/read size drifted"
                    )
                observed[name] = raw
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise GitHubCIEnvelopeVerificationError(
            f"invalid downloaded artifact ZIP: {path.name}"
        ) from error
    return observed


def _git_text(root: Path, arguments: Sequence[str], *, label: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitHubCIEnvelopeVerificationError(
            f"could not resolve {label} in the local Git object database"
        ) from error
    return result.stdout.strip()


def _git_subject_tree(root: Path, commit: str) -> str:
    return _git_text(
        root, ["rev-parse", f"{commit}^{{tree}}"], label="subject commit tree"
    )


def _git_first_parent(root: Path, reference: str) -> str:
    return _git_text(
        root,
        ["rev-parse", f"{reference}^{{commit}}^1"],
        label="container first parent",
    )


def _git_file_bytes(root: Path, reference: str, path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "show", f"{reference}^{{commit}}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitHubCIEnvelopeVerificationError(
            f"required Git blob is absent: {reference}:{path}"
        ) from error
    return result.stdout


def _git_commit_parents(root: Path, reference: str) -> list[str]:
    line = _git_text(
        root,
        ["rev-list", "--parents", "-n", "1", f"{reference}^{{commit}}"],
        label="container parents",
    )
    fields = line.split()
    if not fields or any(not _SHA1_RE.fullmatch(field) for field in fields):
        raise GitHubCIEnvelopeVerificationError("invalid container parent record")
    return fields[1:]


def _git_changed_paths(root: Path, subject: str, container: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-status",
                "--no-renames",
                subject,
                f"{container}^{{commit}}",
                "--",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitHubCIEnvelopeVerificationError(
            "could not inspect additive container diff"
        ) from error
    changed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D", "T"}:
            raise GitHubCIEnvelopeVerificationError(
                "unsupported container diff record"
            )
        status, path = fields
        if path in changed:
            raise GitHubCIEnvelopeVerificationError("duplicate container diff path")
        changed[path] = status
    return changed


def _validate_workflows(
    root: Path, subject_commit_sha1: str, value: object
) -> dict[str, object]:
    workflows = _require_keys(value, set(_WORKFLOWS), label="workflow_definitions")
    for role, expected_path in _WORKFLOWS.items():
        entry = _require_keys(
            workflows[role], {"path", "sha256"}, label=f"workflow {role}"
        )
        if entry["path"] != expected_path:
            raise GitHubCIEnvelopeVerificationError(f"workflow path drifted: {role}")
        declared = _sha256_value(entry["sha256"], label=f"workflow {role} sha256")
        subject_bytes = _git_file_bytes(root, subject_commit_sha1, entry["path"])
        if _sha256(subject_bytes) != declared:
            raise GitHubCIEnvelopeVerificationError(f"workflow hash drifted: {role}")
    return workflows


def _validate_jobs(
    value: object, *, run_role: str, run_id: int, subject_commit_sha1: str
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    if type(value) is not list or not value:
        raise GitHubCIEnvelopeVerificationError(f"run {run_role} jobs missing")
    jobs: list[dict[str, object]] = []
    jobs_by_id: dict[int, dict[str, object]] = {}
    for index, raw_job in enumerate(value):
        job = _require_keys(
            raw_job,
            {
                "job_id",
                "name",
                "head_sha1",
                "status",
                "conclusion",
                "html_url",
                "steps",
            },
            label=f"run {run_role} job {index}",
        )
        job_id = _positive_int(job["job_id"], label="job_id")
        if job_id in jobs_by_id:
            raise GitHubCIEnvelopeVerificationError("duplicate job_id in run")
        if (
            type(job["name"]) is not str
            or job["head_sha1"] != subject_commit_sha1
            or job["status"] != "completed"
            or job["conclusion"] not in {"success", "skipped"}
            or job["html_url"]
            != f"https://github.com/hyh3512/MO_NCO/actions/runs/{run_id}/job/{job_id}"
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"run {run_role} job identity/status drifted"
            )
        jobs_by_id[job_id] = job
        steps = job["steps"]
        if type(steps) is not list or not steps:
            raise GitHubCIEnvelopeVerificationError("job steps missing")
        numbers: set[int] = set()
        for step_index, raw_step in enumerate(steps):
            step = _require_keys(
                raw_step,
                {"number", "name", "status", "conclusion"},
                label=f"job {job_id} step {step_index}",
            )
            number = _positive_int(step["number"], label="step number")
            if number in numbers:
                raise GitHubCIEnvelopeVerificationError("duplicate job step number")
            numbers.add(number)
            if (
                type(step["name"]) is not str
                or step["status"] != "completed"
                or step["conclusion"] not in {"success", "skipped"}
            ):
                raise GitHubCIEnvelopeVerificationError("job step status drifted")
        jobs.append(job)
    return jobs, jobs_by_id


def _validate_runs(
    value: object,
    workflows: Mapping[str, object],
    *,
    subject_commit_sha1: str,
    subject_tree_sha1: str,
) -> tuple[dict[str, object], dict[str, dict[int, dict[str, object]]]]:
    runs = _require_keys(value, set(_RUNS), label="runs")
    all_run_ids: set[int] = set()
    all_job_ids: set[int] = set()
    jobs_by_run: dict[str, dict[int, dict[str, object]]] = {}
    for run_role, (workflow_role, expected_event) in _RUNS.items():
        run = _require_keys(
            runs[run_role],
            {
                "workflow_role",
                "workflow_path",
                "event",
                "run_id",
                "run_attempt",
                "run_number",
                "workflow_id",
                "head_sha1",
                "head_tree_sha1",
                "head_branch",
                "status",
                "conclusion",
                "html_url",
                "created_at",
                "run_started_at",
                "updated_at",
                "jobs",
                "artifact_ids",
            },
            label=f"run {run_role}",
        )
        run_id = _positive_int(run["run_id"], label=f"run {run_role} run_id")
        if run_id in all_run_ids:
            raise GitHubCIEnvelopeVerificationError("duplicate run_id")
        all_run_ids.add(run_id)
        for field in ("run_attempt", "run_number", "workflow_id"):
            _positive_int(run[field], label=f"run {run_role} {field}")
        if run["run_attempt"] != 1:
            raise GitHubCIEnvelopeVerificationError("run attempt must be exactly one")
        expected_path = workflows[workflow_role]["path"]
        if (
            run["workflow_role"] != workflow_role
            or run["workflow_path"] != expected_path
            or run["event"] != expected_event
            or run["head_sha1"] != subject_commit_sha1
            or run["head_tree_sha1"] != subject_tree_sha1
            or run["head_branch"] != "main"
            or run["status"] != "completed"
            or run["conclusion"] != "success"
            or run["html_url"]
            != f"https://github.com/hyh3512/MO_NCO/actions/runs/{run_id}"
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"run {run_role} identity/conclusion drifted"
            )
        for field in ("created_at", "run_started_at", "updated_at"):
            _utc(run[field], label=f"run {run_role} {field}")
        _jobs, jobs_by_id = _validate_jobs(
            run["jobs"],
            run_role=run_role,
            run_id=run_id,
            subject_commit_sha1=subject_commit_sha1,
        )
        observed_job_conclusions = {
            job["name"]: job["conclusion"] for job in jobs_by_id.values()
        }
        if (
            len(observed_job_conclusions) != len(jobs_by_id)
            or observed_job_conclusions != _JOB_CONCLUSIONS_BY_RUN[run_role]
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"exact job name/conclusion set drifted: {run_role}"
            )
        job_ids = set(jobs_by_id)
        if all_job_ids.intersection(job_ids):
            raise GitHubCIEnvelopeVerificationError("duplicate job_id across runs")
        all_job_ids.update(job_ids)
        jobs_by_run[run_role] = jobs_by_id
        artifact_ids = run["artifact_ids"]
        if (
            type(artifact_ids) is not list
            or not artifact_ids
            or any(type(item) is not int or item <= 0 for item in artifact_ids)
            or artifact_ids != sorted(set(artifact_ids))
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"run {run_role} artifact_ids must be sorted unique positive ints"
            )
    if (
        runs["push_repository_contract"]["workflow_id"]
        != runs["manual_public_live"]["workflow_id"]
    ):
        raise GitHubCIEnvelopeVerificationError(
            "repository workflow_id drifted across push/manual runs"
        )
    return runs, jobs_by_run


def _validate_artifacts(
    value: object, runs: Mapping[str, object], *, artifact_root: Path
) -> tuple[
    dict[str, object],
    dict[int, dict[str, object]],
    dict[int, dict[str, bytes]],
]:
    if type(value) is not dict or not value:
        raise GitHubCIEnvelopeVerificationError("artifacts must be a non-empty object")
    artifacts_by_id: dict[int, dict[str, object]] = {}
    contents_by_id: dict[int, dict[str, bytes]] = {}
    names_by_run: dict[str, set[str]] = {role: set() for role in _RUNS}
    downloaded_paths: set[Path] = set()
    for key, raw_artifact in value.items():
        artifact = _require_keys(
            raw_artifact,
            {
                "run_role",
                "artifact_id",
                "name",
                "api_size_in_bytes",
                "api_digest",
                "expired",
                "created_at",
                "updated_at",
                "expires_at",
                "content_verification",
                "downloaded_zip",
                "files",
            },
            label=f"artifact {key}",
        )
        artifact_id = _positive_int(artifact["artifact_id"], label="artifact_id")
        if key != str(artifact_id) or artifact_id in artifacts_by_id:
            raise GitHubCIEnvelopeVerificationError(
                "artifact key/id mismatch or duplicate artifact_id"
            )
        run_role = artifact["run_role"]
        if run_role not in _RUNS or artifact_id not in runs[run_role]["artifact_ids"]:
            raise GitHubCIEnvelopeVerificationError("artifact-to-run binding drifted")
        name = artifact["name"]
        if type(name) is not str or name not in _ARTIFACT_NAMES[run_role]:
            raise GitHubCIEnvelopeVerificationError("artifact name/run drifted")
        if name in names_by_run[run_role]:
            raise GitHubCIEnvelopeVerificationError("duplicate artifact name in run")
        names_by_run[run_role].add(name)
        _positive_int(artifact["api_size_in_bytes"], label="api_size_in_bytes")
        if (
            type(artifact["api_digest"]) is not str
            or not _API_SHA256_RE.fullmatch(artifact["api_digest"])
            or artifact["api_digest"] == "sha256:" + "0" * 64
            or artifact["expired"] is not False
            or artifact["content_verification"] != ARTIFACT_CONTENT_SCOPE
        ):
            raise GitHubCIEnvelopeVerificationError(
                "artifact digest/lifecycle/verification scope drifted"
            )
        for field in ("created_at", "updated_at", "expires_at"):
            _utc(artifact[field], label=f"artifact {artifact_id} {field}")
        downloaded = _require_keys(
            artifact["downloaded_zip"],
            {"path", "bytes", "sha256", "api_digest_matches"},
            label=f"artifact {artifact_id} downloaded_zip",
        )
        zip_path = _safe_external_file(artifact_root, downloaded["path"])
        if zip_path in downloaded_paths:
            raise GitHubCIEnvelopeVerificationError(
                "downloaded ZIP reused by multiple artifact records"
            )
        downloaded_paths.add(zip_path)
        zip_raw = zip_path.read_bytes()
        _positive_int(downloaded["bytes"], label="downloaded ZIP bytes")
        zip_sha256 = _sha256_value(
            downloaded["sha256"], label="downloaded ZIP sha256"
        )
        if (
            downloaded["bytes"] != len(zip_raw)
            or downloaded["bytes"] != artifact["api_size_in_bytes"]
            or zip_sha256 != _sha256(zip_raw)
            or artifact["api_digest"] != f"sha256:{zip_sha256}"
            or downloaded["api_digest_matches"] is not True
        ):
            raise GitHubCIEnvelopeVerificationError(
                "downloaded ZIP/API digest binding drifted"
            )
        files = artifact["files"]
        if type(files) is not list or not files:
            raise GitHubCIEnvelopeVerificationError("artifact member list missing")
        observed_members: dict[str, str] = {}
        ordered_paths: list[str] = []
        for member_index, raw_member in enumerate(files):
            member = _require_keys(
                raw_member,
                {"path", "bytes", "sha256", "role"},
                label=f"artifact {artifact_id} member {member_index}",
            )
            member_path = member["path"]
            if type(member_path) is not str or "\\" in member_path:
                raise GitHubCIEnvelopeVerificationError("unsafe artifact member path")
            pure_path = PurePosixPath(member_path)
            if (
                pure_path.is_absolute()
                or pure_path.as_posix() != member_path
                or ".." in pure_path.parts
                or member_path in observed_members
            ):
                raise GitHubCIEnvelopeVerificationError(
                    "unsafe or duplicate artifact member path"
                )
            _positive_int(member["bytes"], label="artifact member bytes")
            _sha256_value(member["sha256"], label="artifact member sha256")
            if type(member["role"]) is not str:
                raise GitHubCIEnvelopeVerificationError("artifact member role missing")
            observed_members[member_path] = member["role"]
            ordered_paths.append(member_path)
        if ordered_paths != sorted(ordered_paths):
            raise GitHubCIEnvelopeVerificationError(
                "artifact members must be sorted by path"
            )
        if observed_members != _ARTIFACT_MEMBERS[name]:
            raise GitHubCIEnvelopeVerificationError(
                f"artifact exact member set drifted: {name}"
            )
        zip_contents = _verified_zip_members(zip_path)
        zip_members = {
            path: (len(raw), _sha256(raw)) for path, raw in zip_contents.items()
        }
        declared_members = {
            member["path"]: (member["bytes"], member["sha256"])
            for member in files
        }
        if zip_members != declared_members:
            raise GitHubCIEnvelopeVerificationError(
                f"artifact member bytes/hash drifted: {name}"
            )
        artifacts_by_id[artifact_id] = artifact
        contents_by_id[artifact_id] = zip_contents
    for run_role, expected_names in _ARTIFACT_NAMES.items():
        if names_by_run[run_role] != expected_names:
            raise GitHubCIEnvelopeVerificationError(
                f"artifact name set drifted for run {run_role}"
            )
        if set(runs[run_role]["artifact_ids"]) != {
            artifact_id
            for artifact_id, artifact in artifacts_by_id.items()
            if artifact["run_role"] == run_role
        }:
            raise GitHubCIEnvelopeVerificationError(
                f"artifact ID set drifted for run {run_role}"
            )
    artifacts_by_run_and_name = {
        (artifact["run_role"], artifact["name"]): artifact_id
        for artifact_id, artifact in artifacts_by_id.items()
    }
    for artifact_name in (
        "v9r2r1-public-checkout-reference-contract",
        "v9r2r1-internal-complete-tree-exact-eight-reference",
    ):
        push_id = artifacts_by_run_and_name[
            ("push_repository_contract", artifact_name)
        ]
        manual_id = artifacts_by_run_and_name[("manual_public_live", artifact_name)]
        if contents_by_id[manual_id] != contents_by_id[push_id]:
            raise GitHubCIEnvelopeVerificationError(
                f"manual duplicate artifact member bytes drifted: {artifact_name}"
            )
    return value, artifacts_by_id, contents_by_id


def _validate_test_counts(metrics: Mapping[str, object], *, minimum_passed: int) -> None:
    expected_keys = {
        "errors",
        "failures",
        "passed",
        "skipped",
        "subtests_passed",
        "testcases",
    }
    if set(metrics) != expected_keys:
        raise GitHubCIEnvelopeVerificationError("test metric key set drifted")
    for key in expected_keys:
        if type(metrics[key]) is not int or metrics[key] < 0:
            raise GitHubCIEnvelopeVerificationError("test metric must be a nonnegative int")
    if (
        metrics["errors"] != 0
        or metrics["failures"] != 0
        or metrics["skipped"] != 0
        or metrics["passed"] < minimum_passed
        or metrics["testcases"] != metrics["passed"]
        or metrics["subtests_passed"] < 0
    ):
        raise GitHubCIEnvelopeVerificationError("green test metric contract drifted")


def _validate_evidence_metrics(role: str, value: object) -> None:
    if type(value) is not dict:
        raise GitHubCIEnvelopeVerificationError(f"evidence metrics missing: {role}")
    if role == "targeted_regression":
        _validate_test_counts(value, minimum_passed=325)
        return
    if role == "public_backend":
        _validate_test_counts(value, minimum_passed=4)
        return
    if role == "clean_room_package":
        expected = {
            "artifact_member_hashes_verified": True,
            "installed_gate_exit_code": 2,
            "installed_gate_status": "PRE_DEVELOPMENT_HOLD",
            "rebuild_identity_reported_by_successful_workflow_not_reexecuted": True,
            "sdist_distribution": "mo-nco",
            "sdist_version": "0.21.3.14",
            "wheel_distribution": "mo-nco",
            "wheel_version": "0.21.3.14",
        }
        if value != expected:
            raise GitHubCIEnvelopeVerificationError("clean-room metrics drifted")
        return
    if role == "preserved_internal_exact_eight_reference":
        if value != {"expected_failure_count": 8, "xfail_allowed": False}:
            raise GitHubCIEnvelopeVerificationError("exact-eight metrics drifted")
        return
    common = {
        "frozen_v8_fail_closed_outcomes": 7,
        "held_or_rights_sensitive_dependency_outcomes": 70,
        "junit_failure_or_error_testcases": 77,
        "pytest_failed_or_subfailed_outcomes": 78,
        "sealed_output_outcomes": 1,
        "unclassified_outcomes": 0,
    }
    if role == "public_reference":
        if value != common:
            raise GitHubCIEnvelopeVerificationError("public reference metrics drifted")
        return
    expected_keys = set(common) | {
        "junit_passed_testcases",
        "junit_testcases",
        "pytest_passed",
        "pytest_skipped",
        "pytest_subtests_passed",
    }
    if set(value) != expected_keys or any(value[key] != item for key, item in common.items()):
        raise GitHubCIEnvelopeVerificationError("public live failure metrics drifted")
    for key in expected_keys - set(common):
        if type(value[key]) is not int or value[key] < 0:
            raise GitHubCIEnvelopeVerificationError("live metric must be nonnegative int")
    if (
        value["junit_passed_testcases"] < 1327
        or value["junit_testcases"] < 1408
        or value["pytest_passed"] < 1328
        or value["pytest_skipped"] != 4
        or value["pytest_subtests_passed"] != 267
    ):
        raise GitHubCIEnvelopeVerificationError("public live count boundary drifted")


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GitHubCIEnvelopeVerificationError(
            f"invalid strict JSON artifact: {label}"
        ) from error
    if type(payload) is not dict or raw != _canonical_json(payload) + b"\n":
        raise GitHubCIEnvelopeVerificationError(
            f"noncanonical strict JSON artifact: {label}"
        )
    return payload


def _validate_json_self_hash(
    payload: Mapping[str, object], *, field: str, label: str
) -> None:
    declared = _sha256_value(payload.get(field), label=f"{label} {field}")
    core = dict(payload)
    del core[field]
    if declared != _sha256(_canonical_json(core)):
        raise GitHubCIEnvelopeVerificationError(f"{label} self-hash mismatch")


def _junit_green_metrics(junit_raw: bytes, log_raw: bytes) -> dict[str, int]:
    if b"<!DOCTYPE" in junit_raw or b"<!ENTITY" in junit_raw:
        raise GitHubCIEnvelopeVerificationError("DTD/entity prohibited in JUnit")
    try:
        root = ET.fromstring(junit_raw)
    except ET.ParseError as error:
        raise GitHubCIEnvelopeVerificationError("invalid green JUnit XML") from error
    testcases = list(root.iter("testcase"))
    failures = sum(len(case.findall("failure")) for case in testcases)
    errors = sum(len(case.findall("error")) for case in testcases)
    skipped = sum(len(case.findall("skipped")) for case in testcases)
    if any(
        len(case.findall("failure"))
        + len(case.findall("error"))
        + len(case.findall("skipped"))
        > 1
        for case in testcases
    ):
        raise GitHubCIEnvelopeVerificationError("JUnit testcase has multiple outcomes")
    passed = len(testcases) - failures - errors - skipped
    matches = list(_GREEN_PYTEST_SUMMARY_RE.finditer(log_raw))
    if len(matches) != 1:
        raise GitHubCIEnvelopeVerificationError("green pytest summary is not unique")
    log_passed = int(matches[0].group("passed"))
    subtests_raw = matches[0].group("subtests")
    subtests = int(subtests_raw) if subtests_raw is not None else 0
    if failures or errors or skipped or log_passed != passed:
        raise GitHubCIEnvelopeVerificationError("green JUnit/log counts disagree")
    return {
        "errors": errors,
        "failures": failures,
        "passed": passed,
        "skipped": skipped,
        "subtests_passed": subtests,
        "testcases": len(testcases),
    }


def _load_exact_module(path: Path, *, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GitHubCIEnvelopeVerificationError(f"cannot load subject module: {name}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise GitHubCIEnvelopeVerificationError(
            f"cannot execute subject module: {name}"
        ) from error
    return module


def _write_temp(root: Path, relative: str, raw: bytes) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _verify_generic_sanitization_bundle(
    *,
    module: ModuleType,
    receipt_path: Path,
    outputs: Mapping[str, Path],
    kinds: Mapping[str, str],
    subject_commit_sha1: str,
    subject_tree_sha1: str,
    generic_source_path: Path,
    engine_source_path: Path,
) -> dict[str, object]:
    # The existing verifier's only mutable-checkout operation is replaced by
    # the already verified subject commit/tree pair.  All verifier source bytes
    # themselves were materialized from that subject below.
    module._git_checkout_identity = lambda _root: (
        subject_commit_sha1,
        subject_tree_sha1,
    )
    try:
        receipt = module.verify_ci_artifact_bundle(
            receipt_path=receipt_path,
            outputs=outputs,
            kinds=kinds,
            repository_root=r"C:\CI_ENVELOPE_REPOSITORY_SENTINEL",
            user_home=r"C:\Users\CI_ENVELOPE_VERIFIER_SENTINEL",
            host_name="CI_ENVELOPE_VERIFIER_SENTINEL_HOST",
            generic_source_path=generic_source_path,
            engine_source_path=engine_source_path,
        )
    except Exception as error:
        raise GitHubCIEnvelopeVerificationError(
            "generic sanitization bundle verification failed"
        ) from error
    if receipt.get("reference_checkout") != {
        "commit_sha1": subject_commit_sha1,
        "git_tree_sha1": subject_tree_sha1,
    }:
        raise GitHubCIEnvelopeVerificationError(
            "generic sanitization receipt subject binding drifted"
        )
    return receipt


def _validate_environment_preflight(raw: bytes, *, label: str) -> None:
    payload = _strict_json_bytes(raw, label=label)
    # Path sanitization intentionally preserves the raw receipt's embedded
    # self-hash while changing path-bearing fields.  The generic sanitization
    # receipt above binds the transformed bytes and proves strict-JSON
    # semantics; treating the preserved raw hash as a sanitized self-hash
    # would reject the authentic artifact.
    _sha256_value(
        payload.get("receipt_payload_sha256"),
        label=f"{label} preserved raw receipt hash",
    )
    if (
        payload.get("schema")
        != "v21e3r1_v9r2r1_full_suite_environment_preflight_v1"
        or payload.get("status") != "PASS_FULL_SUITE_ENVIRONMENT_PREFLIGHT"
        or payload.get("hold_reasons") != []
        or payload.get("full_suite_execution_preflight_passed") is not True
        or payload.get("full_suite_execution_recommended") is not True
        or payload.get("scoped_v9_tests_affected") is not False
        or payload.get("environment_lock_requirement_satisfied") is not False
    ):
        raise GitHubCIEnvelopeVerificationError(
            f"environment preflight semantics drifted: {label}"
        )
    if payload.get("identity") != _IDENTITY:
        raise GitHubCIEnvelopeVerificationError(
            f"environment preflight identity drifted: {label}"
        )
    interpreter = payload.get("interpreter")
    if (
        type(interpreter) is not dict
        or interpreter.get("expected_version_prefix") != "3.13.12"
        or interpreter.get("checks")
        != {"executable_exact_match": True, "version_prefix_match": True}
    ):
        raise GitHubCIEnvelopeVerificationError(
            f"environment interpreter contract drifted: {label}"
        )
    expected_distributions = {"moocore": "0.3.1", "pymoo": "0.6.1.6"}
    distributions = payload.get("distributions")
    if type(distributions) is not dict or set(distributions) != set(
        expected_distributions
    ):
        raise GitHubCIEnvelopeVerificationError(
            f"environment distribution set drifted: {label}"
        )
    for name, version in expected_distributions.items():
        if distributions[name] != {
            "exact_match": True,
            "expected_version": version,
            "metadata_error": None,
            "observed_version": version,
        }:
            raise GitHubCIEnvelopeVerificationError(
                f"environment distribution identity drifted: {name}"
            )
    imports = payload.get("backend_imports")
    expected_imports = {
        "moocore",
        "pymoo.algorithms.moo.moead",
        "pymoo.algorithms.moo.nsga2",
    }
    if type(imports) is not dict or set(imports) != expected_imports:
        raise GitHubCIEnvelopeVerificationError("backend import set drifted")
    if any(
        record
        != {"exception_message": None, "exception_type": None, "status": "PASS"}
        for record in imports.values()
    ):
        raise GitHubCIEnvelopeVerificationError("backend import status drifted")
    native = payload.get("native_artifacts")
    if type(native) is not dict or set(native) != set(expected_distributions):
        raise GitHubCIEnvelopeVerificationError("native artifact set drifted")
    for name, record in native.items():
        if (
            type(record) is not dict
            or set(record) != {"error", "files", "status"}
            or record.get("error") is not None
            or record.get("status") != "PASS"
            or type(record.get("files")) is not list
            or not record["files"]
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"native artifact status drifted: {name}"
            )
        for artifact in record["files"]:
            if (
                type(artifact) is not dict
                or set(artifact) != {"bytes", "path", "sha256"}
                or type(artifact["bytes"]) is not int
                or artifact["bytes"] <= 0
                or type(artifact["path"]) is not str
            ):
                raise GitHubCIEnvelopeVerificationError(
                    f"native artifact record drifted: {name}"
                )
            _sha256_value(
                artifact["sha256"], label=f"native artifact {name} sha256"
            )
    for field in (
        "full_development_matrix_authorized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "ijoc_submission_authorized",
    ):
        if payload.get(field) is not False:
            raise GitHubCIEnvelopeVerificationError(
                f"environment preflight authority drifted: {field}"
            )


def _validate_installed_gate(raw: bytes) -> None:
    gate = _strict_json_bytes(raw, label="installed gate")
    _validate_json_self_hash(
        gate, field="receipt_payload_sha256", label="installed gate"
    )
    if (
        gate.get("schema")
        != "pareto_v21e3r1_v9r2_predevelopment_readiness_receipt_v1"
        or gate.get("status") != "PRE_DEVELOPMENT_HOLD"
        or gate.get("authorized_next_phase") != (
            "NONE_NEW_PROTOCOL_AND_ALL_APPLICABLE_REQUIREMENTS_REQUIRED"
        )
        or gate.get("development_rows_materialized") != 0
        or gate.get("selection_authorized") is not False
        or gate.get("confirmation_authorized") is not False
        or gate.get("formal_authorized") is not False
        or gate.get("ijoc_submission_authorized") is not False
    ):
        raise GitHubCIEnvelopeVerificationError("installed gate semantics drifted")
    gates = gate.get("gates")
    if (
        type(gates) is not dict
        or gates.get("all_later_phases_prohibited") is not True
        or gates.get("full_development_matrix_authorized") is not False
        or gates.get("scientific_development_claims_authorized") is not False
    ):
        raise GitHubCIEnvelopeVerificationError("installed gate boundary drifted")


def _validate_wheel(raw: bytes) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise GitHubCIEnvelopeVerificationError("wheel has duplicate members")
            if any(
                not name
                or "\\" in name
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                for name in names
            ):
                raise GitHubCIEnvelopeVerificationError("wheel has unsafe members")
            metadata_name = "mo_nco-0.21.3.14.dist-info/METADATA"
            record_name = "mo_nco-0.21.3.14.dist-info/RECORD"
            wheel_name = "mo_nco-0.21.3.14.dist-info/WHEEL"
            required = {
                metadata_name,
                record_name,
                wheel_name,
                "mo_nco/__init__.py",
            }
            if not required.issubset(names):
                raise GitHubCIEnvelopeVerificationError("wheel identity files missing")
            metadata = archive.read(metadata_name).decode("utf-8")
            wheel_metadata = archive.read(wheel_name).decode("utf-8")
            record_text = archive.read(record_name).decode("utf-8")
            rows = list(csv.reader(io.StringIO(record_text, newline="")))
            if len(rows) != len(names) or any(len(row) != 3 for row in rows):
                raise GitHubCIEnvelopeVerificationError("wheel RECORD shape drifted")
            records = {row[0]: (row[1], row[2]) for row in rows}
            if len(records) != len(rows) or set(records) != set(names):
                raise GitHubCIEnvelopeVerificationError("wheel RECORD member set drifted")
            for name in names:
                digest, size = records[name]
                member_raw = archive.read(name)
                if name == record_name:
                    if digest or size:
                        raise GitHubCIEnvelopeVerificationError(
                            "wheel RECORD self-entry drifted"
                        )
                    continue
                expected_digest = base64.urlsafe_b64encode(
                    hashlib.sha256(member_raw).digest()
                ).rstrip(b"=").decode("ascii")
                if digest != f"sha256={expected_digest}" or size != str(
                    len(member_raw)
                ):
                    raise GitHubCIEnvelopeVerificationError(
                        f"wheel RECORD hash/size drifted: {name}"
                    )
    except (zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise GitHubCIEnvelopeVerificationError("invalid clean-room wheel") from error
    if not re.search(r"(?m)^Name: mo-nco\r?$", metadata) or not re.search(
        r"(?m)^Version: 0\.21\.3\.14\r?$", metadata
    ):
        raise GitHubCIEnvelopeVerificationError("wheel metadata identity drifted")
    if not re.search(r"(?m)^Tag: py3-none-any\r?$", wheel_metadata):
        raise GitHubCIEnvelopeVerificationError("wheel compatibility tag drifted")
    return "mo-nco", "0.21.3.14"


def _validate_sdist(raw: bytes) -> tuple[str, str]:
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            names: set[str] = set()
            for member in members:
                name = member.name
                pure = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or name in names
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise GitHubCIEnvelopeVerificationError(
                        "sdist has unsafe or duplicate members"
                    )
                if (
                    member.mtime != 1700000000
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                ):
                    raise GitHubCIEnvelopeVerificationError(
                        "normalized sdist ownership/timestamp drifted"
                    )
                names.add(name)
            prefix = "mo_nco-0.21.3.14/"
            required = {prefix + "PKG-INFO", prefix + "pyproject.toml"}
            if not required.issubset(names):
                raise GitHubCIEnvelopeVerificationError("sdist identity files missing")
            extracted = archive.extractfile(prefix + "PKG-INFO")
            if extracted is None:
                raise GitHubCIEnvelopeVerificationError("sdist PKG-INFO missing")
            metadata = extracted.read().decode("utf-8")
    except (tarfile.TarError, UnicodeDecodeError) as error:
        raise GitHubCIEnvelopeVerificationError("invalid normalized sdist") from error
    if not re.search(r"(?m)^Name: mo-nco\r?$", metadata) or not re.search(
        r"(?m)^Version: 0\.21\.3\.14\r?$", metadata
    ):
        raise GitHubCIEnvelopeVerificationError("sdist metadata identity drifted")
    return "mo-nco", "0.21.3.14"


def _public_contract_metrics(receipt: Mapping[str, object]) -> dict[str, int]:
    classifications = receipt["classification_counts"]
    return {
        "frozen_v8_fail_closed_outcomes": classifications["frozen_v8_fail_closed"],
        "held_or_rights_sensitive_dependency_outcomes": classifications[
            "held_or_rights_sensitive_dependency"
        ],
        "junit_failure_or_error_testcases": receipt["counts"]["junit"][
            "failure_or_error_testcases"
        ],
        "pytest_failed_or_subfailed_outcomes": receipt["counts"]["pytest"][
            "failed_or_subfailed_outcomes"
        ],
        "sealed_output_outcomes": classifications["sealed_output"],
        "unclassified_outcomes": classifications["unclassified"],
    }


def _derive_and_validate_inner_evidence(
    *,
    root: Path,
    subject_commit_sha1: str,
    subject_tree_sha1: str,
    artifacts_by_id: Mapping[int, Mapping[str, object]],
    contents_by_id: Mapping[int, Mapping[str, bytes]],
    evidence: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    subject_blobs = {
        name: _git_file_bytes(root, subject_commit_sha1, path)
        for name, path in _SUBJECT_BLOB_PATHS.items()
    }
    with tempfile.TemporaryDirectory(prefix="mo-nco-ci-envelope-") as temp_name:
        temp_root = Path(temp_name)
        blob_paths = {
            name: _write_temp(temp_root / "subject", path, subject_blobs[name])
            for name, path in _SUBJECT_BLOB_PATHS.items()
        }
        modules = {
            "public": _load_exact_module(
                blob_paths["public_failure_verifier"], name="subject_public_failure"
            ),
            "historical": _load_exact_module(
                blob_paths["historical_failure_verifier"],
                name="subject_historical_failure",
            ),
            "generic": _load_exact_module(
                blob_paths["generic_sanitizer"], name="subject_generic_sanitizer"
            ),
            "engine": _load_exact_module(
                blob_paths["sanitization_engine"], name="subject_sanitization_engine"
            ),
        }
        member_paths: dict[int, dict[str, Path]] = {}
        for artifact_id, contents in contents_by_id.items():
            member_paths[artifact_id] = {
                path: _write_temp(
                    temp_root / "artifacts" / str(artifact_id), path, raw
                )
                for path, raw in contents.items()
            }

        def artifact_for(role: str) -> tuple[Mapping[str, object], Mapping[str, bytes], Mapping[str, Path]]:
            artifact_id = evidence[role]["artifact_id"]
            return (
                artifacts_by_id[artifact_id],
                contents_by_id[artifact_id],
                member_paths[artifact_id],
            )

        _target_artifact, target_raw, target_paths = artifact_for(
            "targeted_regression"
        )
        _verify_generic_sanitization_bundle(
            module=modules["generic"],
            receipt_path=target_paths["targeted.sanitization.json"],
            outputs={
                "targeted.junit.xml": target_paths["targeted.sanitized.junit.xml"],
                "targeted.log": target_paths["targeted.sanitized.log"],
            },
            kinds={
                "targeted.junit.xml": "PYTEST_JUNIT_XML",
                "targeted.log": "PYTEST_LOG",
            },
            subject_commit_sha1=subject_commit_sha1,
            subject_tree_sha1=subject_tree_sha1,
            generic_source_path=blob_paths["generic_sanitizer"],
            engine_source_path=blob_paths["sanitization_engine"],
        )
        targeted_metrics = _junit_green_metrics(
            target_raw["targeted.sanitized.junit.xml"],
            target_raw["targeted.sanitized.log"],
        )

        _backend_artifact, backend_raw, backend_paths = artifact_for("public_backend")
        _verify_generic_sanitization_bundle(
            module=modules["generic"],
            receipt_path=backend_paths["pymoo-recovery.sanitization.json"],
            outputs={
                "backend.environment-preflight.json": backend_paths[
                    "full_suite_environment_preflight.sanitized.json"
                ],
                "backend.junit.xml": backend_paths[
                    "pymoo-recovery.sanitized.junit.xml"
                ],
                "backend.log": backend_paths["pymoo-recovery.sanitized.log"],
            },
            kinds={
                "backend.environment-preflight.json": "STRICT_JSON",
                "backend.junit.xml": "PYTEST_JUNIT_XML",
                "backend.log": "PYTEST_LOG",
            },
            subject_commit_sha1=subject_commit_sha1,
            subject_tree_sha1=subject_tree_sha1,
            generic_source_path=blob_paths["generic_sanitizer"],
            engine_source_path=blob_paths["sanitization_engine"],
        )
        _validate_environment_preflight(
            backend_raw["full_suite_environment_preflight.sanitized.json"],
            label="backend environment preflight",
        )
        backend_metrics = _junit_green_metrics(
            backend_raw["pymoo-recovery.sanitized.junit.xml"],
            backend_raw["pymoo-recovery.sanitized.log"],
        )

        _package_artifact, package_raw, _package_paths = artifact_for(
            "clean_room_package"
        )
        if package_raw["installed-gate.json"] != subject_blobs["installed_gate"]:
            raise GitHubCIEnvelopeVerificationError(
                "installed gate differs from the exact subject Git blob"
            )
        _validate_installed_gate(package_raw["installed-gate.json"])
        wheel_name, wheel_version = _validate_wheel(
            package_raw["wheel-a/mo_nco-0.21.3.14-py3-none-any.whl"]
        )
        sdist_name, sdist_version = _validate_sdist(
            package_raw["sdist-normalized/mo_nco-0.21.3.14-a.tar.gz"]
        )
        package_metrics = {
            "artifact_member_hashes_verified": True,
            "installed_gate_exit_code": 2,
            "installed_gate_status": "PRE_DEVELOPMENT_HOLD",
            "rebuild_identity_reported_by_successful_workflow_not_reexecuted": True,
            "sdist_distribution": sdist_name,
            "sdist_version": sdist_version,
            "wheel_distribution": wheel_name,
            "wheel_version": wheel_version,
        }

        _public_artifact, public_raw, _public_paths = artifact_for("public_reference")
        try:
            public_receipt = modules["public"].verify_expected_failure_set(
                blob_paths["public_registry"],
                blob_paths["public_reference_junit"],
                blob_paths["public_reference_log"],
                require_reference_sha256=True,
            )
        except Exception as error:
            raise GitHubCIEnvelopeVerificationError(
                "subject public reference contract verification failed"
            ) from error
        if public_raw["public-checkout-reference-contract.json"] != (
            _canonical_json(public_receipt) + b"\n"
        ):
            raise GitHubCIEnvelopeVerificationError(
                "public reference receipt semantics/bytes drifted"
            )
        public_metrics = _public_contract_metrics(public_receipt)

        _historical_artifact, historical_raw, _historical_paths = artifact_for(
            "preserved_internal_exact_eight_reference"
        )
        try:
            historical_receipt = modules["historical"].verify_expected_failure_set(
                blob_paths["historical_registry"],
                blob_paths["historical_junit"],
                require_reference_sha256=True,
            )
        except Exception as error:
            raise GitHubCIEnvelopeVerificationError(
                "subject exact-eight reference verification failed"
            ) from error
        if historical_raw[
            "internal-complete-tree-exact-eight-reference.json"
        ] != (_canonical_json(historical_receipt) + b"\n"):
            raise GitHubCIEnvelopeVerificationError(
                "exact-eight receipt semantics/bytes drifted"
            )
        historical_metrics = {
            "expected_failure_count": historical_receipt["counts"]["failures"],
            "xfail_allowed": False,
        }

        _live_artifact, live_raw, live_paths = artifact_for(
            "public_live_failure_contract"
        )
        _verify_generic_sanitization_bundle(
            module=modules["generic"],
            receipt_path=live_paths[
                "public_checkout_live_environment_preflight.sanitization.json"
            ],
            outputs={
                "environment-preflight.json": live_paths[
                    "public_checkout_live_environment_preflight.sanitized.json"
                ],
                "environment-preflight.log": live_paths[
                    "public_checkout_live_environment_preflight.sanitized.log"
                ],
            },
            kinds={
                "environment-preflight.json": "STRICT_JSON",
                "environment-preflight.log": "STRICT_JSON",
            },
            subject_commit_sha1=subject_commit_sha1,
            subject_tree_sha1=subject_tree_sha1,
            generic_source_path=blob_paths["generic_sanitizer"],
            engine_source_path=blob_paths["sanitization_engine"],
        )
        _validate_environment_preflight(
            live_raw[
                "public_checkout_live_environment_preflight.sanitized.json"
            ],
            label="live environment preflight",
        )
        if live_raw[
            "public_checkout_live_environment_preflight.sanitized.log"
        ] != live_raw[
            "public_checkout_live_environment_preflight.sanitized.json"
        ]:
            raise GitHubCIEnvelopeVerificationError(
                "live preflight stdout JSON and receipt bytes disagree"
            )
        try:
            live_receipt = modules["public"].verify_expected_failure_set(
                blob_paths["public_registry"],
                live_paths["full_repository.sanitized.junit.xml"],
                live_paths["full_repository.sanitized.log"],
                require_reference_sha256=False,
            )
        except Exception as error:
            raise GitHubCIEnvelopeVerificationError(
                "live public failure contract verification failed"
            ) from error
        if live_raw["public-checkout-live-contract.sanitized.json"] != (
            _canonical_json(live_receipt) + b"\n"
        ):
            raise GitHubCIEnvelopeVerificationError(
                "live public contract receipt semantics/bytes drifted"
            )
        try:
            output_sanitization = modules["engine"].verify_sanitized_bundle(
                receipt_path=live_paths[
                    "V9R2R1_RAW_OUTPUT_SANITIZATION_RECEIPT.json"
                ],
                junit_path=live_paths["full_repository.sanitized.junit.xml"],
                log_path=live_paths["full_repository.sanitized.log"],
                sanitizer_source_path=blob_paths["sanitization_engine"],
            )
        except Exception as error:
            raise GitHubCIEnvelopeVerificationError(
                "live output sanitization receipt verification failed"
            ) from error
        if output_sanitization.get("reference_checkout") != {
            "commit_sha1": subject_commit_sha1,
            "git_tree_sha1": subject_tree_sha1,
        }:
            raise GitHubCIEnvelopeVerificationError(
                "live output sanitization subject binding drifted"
            )
        live_metrics = {
            **_public_contract_metrics(live_receipt),
            "junit_passed_testcases": live_receipt["counts"]["junit"][
                "passed_testcases"
            ],
            "junit_testcases": live_receipt["counts"]["junit"]["testcases"],
            "pytest_passed": live_receipt["counts"]["pytest"]["passed"],
            "pytest_skipped": live_receipt["counts"]["pytest"]["skipped"],
            "pytest_subtests_passed": live_receipt["counts"]["pytest"][
                "subtests_passed"
            ],
        }
        return {
            "targeted_regression": targeted_metrics,
            "public_backend": backend_metrics,
            "clean_room_package": package_metrics,
            "public_reference": public_metrics,
            "preserved_internal_exact_eight_reference": historical_metrics,
            "public_live_failure_contract": live_metrics,
        }


def _validate_evidence(
    root: Path,
    value: object,
    runs: Mapping[str, object],
    jobs_by_run: Mapping[str, Mapping[int, Mapping[str, object]]],
    artifacts_by_id: Mapping[int, Mapping[str, object]],
    contents_by_id: Mapping[int, Mapping[str, bytes]],
    *,
    subject_commit_sha1: str,
    subject_tree_sha1: str,
) -> dict[str, object]:
    evidence = _require_keys(value, set(_EVIDENCE), label="evidence_contracts")
    for role, (run_role, artifact_name, status, hash_keys) in _EVIDENCE.items():
        record = _require_keys(
            evidence[role],
            {
                "run_role",
                "job_id",
                "artifact_id",
                "artifact_name",
                "artifact_content_verification",
                "reported_status",
                "reported_inner_sha256",
                "metrics",
            },
            label=f"evidence {role}",
        )
        job_id = _positive_int(record["job_id"], label=f"evidence {role} job_id")
        artifact_id = _positive_int(
            record["artifact_id"], label=f"evidence {role} artifact_id"
        )
        artifact = artifacts_by_id.get(artifact_id)
        evidence_job = jobs_by_run[run_role].get(job_id)
        if (
            record["run_role"] != run_role
            or evidence_job is None
            or evidence_job["conclusion"] != "success"
            or evidence_job["name"] != _EVIDENCE_JOB_NAMES[role]
            or artifact is None
            or artifact["run_role"] != run_role
            or record["artifact_name"] != artifact_name
            or artifact["name"] != artifact_name
            or record["artifact_content_verification"] != ARTIFACT_CONTENT_SCOPE
            or record["reported_status"] != status
        ):
            raise GitHubCIEnvelopeVerificationError(
                f"evidence cross-binding drifted: {role}"
            )
        hashes = _require_keys(
            record["reported_inner_sha256"], hash_keys, label=f"evidence {role} hashes"
        )
        for name, digest in hashes.items():
            _sha256_value(digest, label=f"evidence {role} {name}")
        members_by_role = {
            member["role"]: member["sha256"] for member in artifact["files"]
        }
        for hash_field, member_role in _INNER_MEMBER_ROLES[role].items():
            if hashes[hash_field] != members_by_role[member_role]:
                raise GitHubCIEnvelopeVerificationError(
                    f"evidence/member hash binding drifted: {role}.{hash_field}"
                )
    public_registry = _sha256(
        _git_file_bytes(
            root, subject_commit_sha1, _SUBJECT_BLOB_PATHS["public_registry"]
        )
    )
    historical_registry = _sha256(
        _git_file_bytes(
            root, subject_commit_sha1, _SUBJECT_BLOB_PATHS["historical_registry"]
        )
    )
    public_reference_hash = evidence["public_reference"]["reported_inner_sha256"][
        "failure_registry_sha256"
    ]
    public_live_hash = evidence["public_live_failure_contract"][
        "reported_inner_sha256"
    ]["failure_registry_sha256"]
    if public_reference_hash != public_registry or public_live_hash != public_registry:
        raise GitHubCIEnvelopeVerificationError("public failure registry binding drifted")
    if evidence["preserved_internal_exact_eight_reference"][
        "reported_inner_sha256"
    ]["historical_failure_registry_sha256"] != historical_registry:
        raise GitHubCIEnvelopeVerificationError("historical registry binding drifted")
    historical_junit = _sha256(
        _git_file_bytes(
            root, subject_commit_sha1, _SUBJECT_BLOB_PATHS["historical_junit"]
        )
    )
    if evidence["preserved_internal_exact_eight_reference"][
        "reported_inner_sha256"
    ]["reference_junit_sha256"] != historical_junit:
        raise GitHubCIEnvelopeVerificationError("historical JUnit binding drifted")
    derived_metrics = _derive_and_validate_inner_evidence(
        root=root,
        subject_commit_sha1=subject_commit_sha1,
        subject_tree_sha1=subject_tree_sha1,
        artifacts_by_id=artifacts_by_id,
        contents_by_id=contents_by_id,
        evidence=evidence,
    )
    for role, metrics in derived_metrics.items():
        _validate_evidence_metrics(role, metrics)
        if evidence[role]["metrics"] != metrics:
            raise GitHubCIEnvelopeVerificationError(
                f"self-reported evidence metrics differ from derived bytes: {role}"
            )
    return evidence


def verify_github_ci_envelope(
    envelope_path: Path,
    *,
    root: Path,
    artifact_root: Path,
    expected_commit_sha1: str,
    expected_tree_sha1: str,
    container_ref: str | None = None,
) -> dict[str, object]:
    """Verify the canonical v1 metadata envelope without network access."""

    root = root.resolve(strict=True)
    artifact_root = artifact_root.resolve(strict=True)
    if not artifact_root.is_dir():
        raise GitHubCIEnvelopeVerificationError("artifact root must be a directory")
    envelope, raw = _strict_json(envelope_path)
    _reject_placeholders(envelope)
    required_keys = {
        "schema",
        "status",
        "identity",
        "subject",
        "container_contract",
        "workflow_definitions",
        "runs",
        "artifacts",
        "evidence_contracts",
        "claim_boundary",
        "envelope_payload_sha256",
    }
    if set(envelope) != required_keys:
        raise GitHubCIEnvelopeVerificationError("envelope key set drifted")
    expected_commit_sha1 = _sha1(expected_commit_sha1, label="expected commit")
    expected_tree_sha1 = _sha1(expected_tree_sha1, label="expected tree")
    if envelope["schema"] != ENVELOPE_SCHEMA or envelope["status"] != ENVELOPE_STATUS:
        raise GitHubCIEnvelopeVerificationError("envelope schema/status drifted")
    if envelope["identity"] != _IDENTITY:
        raise GitHubCIEnvelopeVerificationError("identity drifted")

    subject = _require_keys(
        envelope["subject"],
        {"repository", "repository_id", "commit_sha1", "git_tree_sha1", "branch"},
        label="subject",
    )
    _positive_int(subject["repository_id"], label="repository_id")
    if subject != {
        "repository": "hyh3512/MO_NCO",
        "repository_id": 1347294242,
        "commit_sha1": expected_commit_sha1,
        "git_tree_sha1": expected_tree_sha1,
        "branch": "main",
    }:
        raise GitHubCIEnvelopeVerificationError("subject identity drifted")
    if _git_subject_tree(root, expected_commit_sha1) != expected_tree_sha1:
        raise GitHubCIEnvelopeVerificationError("local Git subject/tree binding drifted")

    container = _require_keys(
        envelope["container_contract"],
        {
            "mode",
            "envelope_path",
            "expected_container_first_parent_sha1",
            "containing_commit_sha1_embedded",
            "subject_commit_claimed_as_container_commit",
            "artifact_content_verification",
            "artifact_bytes_downloaded_and_verified",
            "allowed_changed_paths",
        },
        label="container_contract",
    )
    if container != {
        "mode": "POST_CI_ADDITIVE_EVIDENCE_COMMIT",
        "envelope_path": "provenance/V9R2R1_GITHUB_CI_ENVELOPE.json",
        "expected_container_first_parent_sha1": expected_commit_sha1,
        "containing_commit_sha1_embedded": False,
        "subject_commit_claimed_as_container_commit": False,
        "artifact_content_verification": ARTIFACT_CONTENT_SCOPE,
        "artifact_bytes_downloaded_and_verified": True,
        "allowed_changed_paths": [
            "GITHUB_EXPORT_CONTENTS.json",
            "provenance/V9R2R1_GITHUB_CI_ENVELOPE.json",
        ],
    }:
        raise GitHubCIEnvelopeVerificationError("container contract drifted")

    workflows = _validate_workflows(
        root, expected_commit_sha1, envelope["workflow_definitions"]
    )
    runs, jobs_by_run = _validate_runs(
        envelope["runs"],
        workflows,
        subject_commit_sha1=expected_commit_sha1,
        subject_tree_sha1=expected_tree_sha1,
    )
    _artifacts, artifacts_by_id, contents_by_id = _validate_artifacts(
        envelope["artifacts"], runs, artifact_root=artifact_root
    )
    _validate_evidence(
        root,
        envelope["evidence_contracts"],
        runs,
        jobs_by_run,
        artifacts_by_id,
        contents_by_id,
        subject_commit_sha1=expected_commit_sha1,
        subject_tree_sha1=expected_tree_sha1,
    )

    claim = _require_keys(
        envelope["claim_boundary"],
        _CLAIM_FIELDS | {"development_study_readiness", "ijoc_status"},
        label="claim_boundary",
    )
    for field in _CLAIM_FIELDS:
        if claim[field] is not False:
            raise GitHubCIEnvelopeVerificationError(
                f"scientific/authorization boundary drifted: {field}"
            )
    if (
        claim["development_study_readiness"] != "PRE_DEVELOPMENT_HOLD"
        or claim["ijoc_status"] != "HOLD_NO_SUBMIT"
    ):
        raise GitHubCIEnvelopeVerificationError("HOLD status boundary drifted")

    declared_hash = _sha256_value(
        envelope["envelope_payload_sha256"], label="envelope payload sha256"
    )
    core = dict(envelope)
    del core["envelope_payload_sha256"]
    if declared_hash != _sha256(_canonical_json(core)):
        raise GitHubCIEnvelopeVerificationError("envelope payload hash mismatch")

    container_binding_verified = container_ref is not None
    if container_ref is not None:
        if not container_ref or container_ref.strip() != container_ref:
            raise GitHubCIEnvelopeVerificationError("invalid container ref")
        parents = _git_commit_parents(root, container_ref)
        if parents != [expected_commit_sha1]:
            raise GitHubCIEnvelopeVerificationError(
                "container must have exactly one parent equal to the envelope subject"
            )
        if _git_file_bytes(
            root, container_ref, container["envelope_path"]
        ) != raw:
            raise GitHubCIEnvelopeVerificationError(
                "container commit envelope bytes drifted"
            )
        changed = _git_changed_paths(root, expected_commit_sha1, container_ref)
        if changed != {
            "GITHUB_EXPORT_CONTENTS.json": "M",
            "provenance/V9R2R1_GITHUB_CI_ENVELOPE.json": "A",
        }:
            raise GitHubCIEnvelopeVerificationError(
                "container diff exceeds the exact additive evidence allowlist"
            )

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": (
            "PASS_VERIFIED_GITHUB_CI_ARTIFACT_ENVELOPE_ENGINEERING_ONLY__"
            "PRE_DEVELOPMENT_HOLD"
        ),
        "subject_commit_sha1": expected_commit_sha1,
        "subject_git_tree_sha1": expected_tree_sha1,
        "container_commit_binding_verified": container_binding_verified,
        "workflow_count": len(workflows),
        "run_count": len(runs),
        "artifact_count": len(artifacts_by_id),
        "evidence_contract_count": len(_EVIDENCE),
        "envelope_bytes": len(raw),
        "envelope_sha256": _sha256(raw),
        "artifact_archive_bytes_and_members_verified": True,
        "repository_wide_green": False,
        "environment_lock": False,
        "github_api_metadata_cryptographically_authenticated": False,
        "scientific_independence": False,
        "scientific_stage_authorized": False,
        "development_study_readiness": "PRE_DEVELOPMENT_HOLD",
        "ijoc_status": "HOLD_NO_SUBMIT",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--container-ref", default="HEAD")
    args = parser.parse_args(argv)
    try:
        result = verify_github_ci_envelope(
            args.envelope,
            root=args.root,
            artifact_root=args.artifact_root,
            expected_commit_sha1=args.expected_commit,
            expected_tree_sha1=args.expected_tree,
            container_ref=args.container_ref,
        )
    except (
        GitHubCIEnvelopeVerificationError,
        FileNotFoundError,
        OSError,
    ) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENVELOPE_SCHEMA",
    "ENVELOPE_STATUS",
    "GitHubCIEnvelopeVerificationError",
    "ARTIFACT_CONTENT_SCOPE",
    "VERIFICATION_SCHEMA",
    "main",
    "verify_github_ci_envelope",
]
