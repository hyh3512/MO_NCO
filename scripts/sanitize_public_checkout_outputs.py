"""Create and verify deterministic, path-sanitized public pytest evidence.

The transformation is deliberately narrow: it replaces only the explicitly
supplied checkout root, temporary root, user home, Python environment prefix,
user name, and machine host name.  It preserves all other bytes, including
line endings and six versioned public-source fixture literals, and fails closed
if any other Windows absolute path remains.  A canonical receipt
binds raw/output hashes, the sanitizer source, replacement counts, the public
fixture allowlist, and pre/post semantic observations.
"""

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


RECEIPT_SCHEMA = "v21e3r1_v9r2r1_raw_output_sanitization_receipt_v1"
PASS_STATUS = "PASS_DETERMINISTIC_SANITIZATION_SEMANTICALLY_EQUIVALENT"
IDENTITY = {
    "distribution": "mo-nco",
    "revision": "V21E3R1_V9R2R1",
    "version": "0.21.3.14",
}
SANITIZER_SOURCE_PATH = "scripts/sanitize_public_checkout_outputs.py"
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    rb"(?i)(?<![a-z0-9])[a-z]:(?:\\+|/)"
)
_PUBLIC_WINDOWS_PATH_FIXTURES = (
    "C:/drive-absolute",
    "C:/escape.pdf",
    "C:/outside/metric.json",
    "C:/trace.sqlite3",
    r"C:\\miniconda3\\python.exe artifacts\\v21e3r1_v8_work_20260822\\run_frozen_diagnostic_metric_timeout_recovery_continuation.py --output-directory outputs\\relative",
    r"C:\\miniconda3\\python.exe -m ijoc_submission_v21e3r1.scripts.run_v21e3r1_development_diagnostics --output-directory outputs\\relative",
)
_SUMMARY_RE = re.compile(
    r"^(?P<failed>\d+) failed, (?P<passed>\d+) passed, "
    r"(?P<skipped>\d+) skipped, (?P<subtests_passed>\d+) subtests passed "
    r"in (?P<wall_seconds>\d+(?:\.\d+)?)s(?: \((?P<wall_clock>[^\r\n]+)\))?\r?$",
    re.MULTILINE,
)
_FAILURE_SUMMARY_RE = re.compile(
    r"^(?P<kind>FAILED|SUBFAILED(?:\([^\r\n]*\)))\s+"
    r"(?P<node>tests[/\\][^\r\n]+?)\s*$",
    re.MULTILINE,
)


class SanitizationError(ValueError):
    """Raised when sanitization or verification cannot prove its contract."""


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


def _strict_utf8(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SanitizationError(f"{label} is not strict UTF-8") from error


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise SanitizationError("receipt path must be a non-empty POSIX string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise SanitizationError(f"unsafe receipt path: {value!r}")
    return value


def _strict_json(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.resolve(strict=True).read_bytes()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SanitizationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise SanitizationError(f"non-finite JSON value prohibited: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SanitizationError(f"invalid strict JSON: {path}") from error
    if type(payload) is not dict:
        raise SanitizationError("receipt must be a JSON object")
    if raw != _canonical_json(payload) + b"\n":
        raise SanitizationError(
            "receipt must be canonical JSON followed by exactly one newline"
        )
    return payload, raw


def _pytest_node_id(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname")
    name = testcase.attrib.get("name")
    if not classname or not name:
        raise SanitizationError("JUnit testcase lacks classname or name")
    parts = classname.split(".")
    module_index = -1
    for index, part in enumerate(parts):
        if part.startswith("test_"):
            module_index = index
    if module_index < 0:
        raise SanitizationError(
            f"cannot derive pytest node id from classname: {classname!r}"
        )
    module = "/".join(parts[: module_index + 1]) + ".py"
    suffix = [*parts[module_index + 1 :], name]
    return module + "::" + "::".join(suffix)


def junit_observation(raw: bytes) -> dict[str, object]:
    """Return path-insensitive counts and terminal-outcome identities."""

    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise SanitizationError("DTD/entity declarations are prohibited")
    _strict_utf8(raw, "JUnit evidence")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise SanitizationError("invalid JUnit XML") from error
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if len(suites) != 1:
        raise SanitizationError("exactly one JUnit testsuite is required")
    suite = suites[0]
    declared: dict[str, int] = {}
    for field in ("tests", "failures", "errors", "skipped"):
        value = suite.attrib.get(field)
        if value is None or not value.isdecimal():
            raise SanitizationError(f"JUnit testsuite lacks integer {field!r}")
        declared[field] = int(value)

    testcases = list(suite.iter("testcase"))
    testcase_node_ids = [_pytest_node_id(testcase) for testcase in testcases]
    if len(testcase_node_ids) != len(set(testcase_node_ids)):
        raise SanitizationError("JUnit testcase node identities are not unique")
    testcase_node_ids.sort()
    failure_elements = 0
    error_elements = 0
    skipped_elements = 0
    failing_testcases = 0
    terminal_ids: list[str] = []
    failure_or_error_ids: list[str] = []
    for testcase in testcases:
        base = _pytest_node_id(testcase)
        outcomes: list[tuple[str, ET.Element]] = []
        for kind in ("failure", "error", "skipped"):
            children = testcase.findall(kind)
            outcomes.extend((kind, child) for child in children)
            if kind == "failure":
                failure_elements += len(children)
            elif kind == "error":
                error_elements += len(children)
            else:
                skipped_elements += len(children)
        if any(kind in {"failure", "error"} for kind, _ in outcomes):
            failing_testcases += 1
        ordinals: dict[str, int] = {"failure": 0, "error": 0, "skipped": 0}
        for kind, _child in outcomes:
            ordinal = ordinals[kind]
            ordinals[kind] += 1
            outcome_id = f"{base}::{kind}[{ordinal}]"
            terminal_ids.append(outcome_id)
            if kind in {"failure", "error"}:
                failure_or_error_ids.append(outcome_id)

    if declared["failures"] != failure_elements:
        raise SanitizationError("declared/observed JUnit failure count drifted")
    if declared["errors"] != error_elements:
        raise SanitizationError("declared/observed JUnit error count drifted")
    if declared["skipped"] != skipped_elements:
        raise SanitizationError("declared/observed JUnit skipped count drifted")
    terminal_ids.sort()
    failure_or_error_ids.sort()
    counts = {
        "declared_tests": declared["tests"],
        "declared_failures": declared["failures"],
        "declared_errors": declared["errors"],
        "declared_skipped": declared["skipped"],
        "testcase_elements": len(testcases),
        "passing_testcase_elements": len(testcases)
        - failing_testcases
        - skipped_elements,
        "failing_testcase_elements": failing_testcases,
        "failure_elements": failure_elements,
        "error_elements": error_elements,
        "skipped_elements": skipped_elements,
        "terminal_outcome_elements": len(terminal_ids),
    }
    return {
        "counts": counts,
        "testcase_node_ids_sha256": _sha256(_canonical_json(testcase_node_ids)),
        "terminal_outcome_ids_sha256": _sha256(_canonical_json(terminal_ids)),
        "failure_or_error_outcome_ids_sha256": _sha256(
            _canonical_json(failure_or_error_ids)
        ),
    }


def log_observation(raw: bytes) -> dict[str, object]:
    """Return the final pytest summary and exact short-summary entry hash."""

    text = _strict_utf8(raw, "pytest log")
    summaries = list(_SUMMARY_RE.finditer(text))
    if len(summaries) != 1:
        raise SanitizationError("pytest log must contain exactly one terminal summary")
    match = summaries[0]
    summary = {
        "failed": int(match.group("failed")),
        "passed": int(match.group("passed")),
        "skipped": int(match.group("skipped")),
        "subtests_passed": int(match.group("subtests_passed")),
        "wall_seconds": match.group("wall_seconds"),
        "wall_clock": match.group("wall_clock"),
    }
    failure_entries: list[str] = []
    for failure in _FAILURE_SUMMARY_RE.finditer(text):
        node = failure.group("node")
        if " - " in node:
            node = node.split(" - ", 1)[0]
        failure_entries.append(f"{failure.group('kind')} {node}")
    if len(failure_entries) != summary["failed"]:
        raise SanitizationError(
            "pytest log failed count does not match short-summary entries"
        )
    if len(failure_entries) != len(set(failure_entries)):
        raise SanitizationError("pytest short-summary failure entries are not unique")
    failure_entries.sort()
    return {
        "summary": summary,
        "failure_summary_entry_count": len(failure_entries),
        "failure_summary_entries_sha256": _sha256(
            _canonical_json(failure_entries)
        ),
    }


def _windows_path_variants(value: str) -> list[bytes]:
    if not re.fullmatch(r"[A-Za-z]:\\[^\r\n]+", value):
        raise SanitizationError(f"expected an absolute Windows path: {value!r}")
    variants = [value.replace("\\", "\\" * depth) for depth in (4, 2, 1)]
    variants.append(value.replace("\\", "/"))
    return [variant.encode("utf-8") for variant in dict.fromkeys(variants)]


def _replace_case_insensitive(
    raw: bytes, needle: bytes, replacement: bytes
) -> tuple[bytes, int]:
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    return pattern.subn(lambda _match: replacement, raw)


def _mask_public_source_fixtures(
    raw: bytes,
) -> tuple[bytes, dict[str, int], list[tuple[bytes, bytes]]]:
    """Mask exact public-source fixtures without treating them as machine data."""

    masked = raw
    counts: dict[str, int] = {}
    restorations: list[tuple[bytes, bytes]] = []
    for index, literal in enumerate(_PUBLIC_WINDOWS_PATH_FIXTURES):
        literal_raw = literal.encode("ascii")
        pattern = re.compile(
            re.escape(literal_raw) + rb"(?=$|[\]\s<>\"'(),])"
        )
        token = f"__PUBLIC_SOURCE_FIXTURE_{index}__".encode("ascii")
        if token in masked:
            raise SanitizationError("public fixture mask token collides with evidence")
        masked, count = pattern.subn(lambda _match: token, masked)
        counts[literal] = count
        restorations.append((token, literal_raw))
    return masked, counts, restorations


def _restore_public_source_fixtures(
    raw: bytes,
    restorations: Sequence[tuple[bytes, bytes]],
) -> bytes:
    result = raw
    for token, literal in restorations:
        result = result.replace(token, literal)
    return result


def _public_fixture_counts_and_residuals(
    raw: bytes,
) -> tuple[dict[str, int], int]:
    """Count exact public fixtures and detect all other drive-absolute paths."""

    masked, counts, _restorations = _mask_public_source_fixtures(raw)
    residual_count = len(_WINDOWS_ABSOLUTE_PATH_RE.findall(masked))
    return counts, residual_count


def _sanitize_one(
    raw: bytes,
    *,
    repository_root: str,
    temp_root: str,
    user_home: str,
    environment_prefix: str,
    host_name: str,
) -> tuple[bytes, dict[str, int], dict[str, int]]:
    _strict_utf8(raw, "raw evidence")
    result, fixture_counts, fixture_restorations = _mask_public_source_fixtures(raw)
    counts: dict[str, int] = {}
    logical_rules = (
        ("repository_root", repository_root, b"__REPO_ROOT__"),
        ("temp_root", temp_root, b"__TEMP_ROOT__"),
        ("user_home", user_home, b"__USER_HOME__"),
        ("environment_prefix", environment_prefix, b"__PYTHON_PREFIX__"),
    )
    for rule_id, value, replacement in logical_rules:
        total = 0
        for variant in _windows_path_variants(value):
            result, count = _replace_case_insensitive(result, variant, replacement)
            total += count
        counts[rule_id] = total

    username = Path(user_home).name
    if not username or username in {".", ".."}:
        raise SanitizationError("cannot derive a user name from user_home")
    result, username_count = _replace_case_insensitive(
        result, username.encode("utf-8"), b"__USERNAME__"
    )
    counts["username"] = username_count

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host_name):
        raise SanitizationError("host_name must be a non-empty machine name token")
    result, host_name_count = _replace_case_insensitive(
        result, host_name.encode("utf-8"), b"__HOSTNAME__"
    )
    counts["host_name"] = host_name_count

    for value in (
        repository_root,
        temp_root,
        user_home,
        environment_prefix,
        username,
        host_name,
    ):
        for variant in (
            _windows_path_variants(value)
            if ":\\" in value
            else [value.encode("utf-8")]
        ):
            if re.search(re.escape(variant), result, re.IGNORECASE):
                raise SanitizationError("a declared sensitive value remains")
    result = _restore_public_source_fixtures(result, fixture_restorations)
    observed_fixture_counts, residual_count = _public_fixture_counts_and_residuals(
        result
    )
    if observed_fixture_counts != fixture_counts:
        raise SanitizationError("public-source fixture occurrence count drifted")
    if residual_count:
        raise SanitizationError(
            "unregistered Windows absolute path remains "
            f"(count={residual_count})"
        )
    return result, counts, fixture_counts


def _artifact(raw: bytes, *, path: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"bytes": len(raw), "sha256": _sha256(raw)}
    if path is not None:
        result["path"] = _safe_relative_path(path)
    return result


def _assert_distinct_paths(paths: Sequence[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise SanitizationError("input and output paths must all be distinct")


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise SanitizationError(f"refusing to overwrite: {path}") from error


def create_sanitized_bundle(
    *,
    raw_junit_path: Path,
    raw_log_path: Path,
    output_junit_path: Path,
    output_log_path: Path,
    receipt_path: Path,
    repository_root: str,
    temp_root: str,
    user_home: str,
    environment_prefix: str,
    host_name: str,
    reference_commit: str,
    reference_tree: str,
    sanitizer_source_path: Path,
) -> dict[str, object]:
    """Sanitize two raw files, prove equivalence, and write a bound bundle."""

    if not _SHA1_RE.fullmatch(reference_commit):
        raise SanitizationError("reference_commit must be a lowercase Git SHA-1")
    if not _SHA1_RE.fullmatch(reference_tree):
        raise SanitizationError("reference_tree must be a lowercase Git SHA-1")
    paths = (
        raw_junit_path,
        raw_log_path,
        output_junit_path,
        output_log_path,
        receipt_path,
        sanitizer_source_path,
    )
    _assert_distinct_paths(paths)
    for target in (output_junit_path, output_log_path, receipt_path):
        if target.exists():
            raise SanitizationError(f"refusing to overwrite: {target}")

    raw_junit = raw_junit_path.resolve(strict=True).read_bytes()
    raw_log = raw_log_path.resolve(strict=True).read_bytes()
    source_raw = sanitizer_source_path.resolve(strict=True).read_bytes()
    junit_before = junit_observation(raw_junit)
    log_before = log_observation(raw_log)
    sanitized_junit, junit_replacements, junit_fixture_counts = _sanitize_one(
        raw_junit,
        repository_root=repository_root,
        temp_root=temp_root,
        user_home=user_home,
        environment_prefix=environment_prefix,
        host_name=host_name,
    )
    sanitized_log, log_replacements, log_fixture_counts = _sanitize_one(
        raw_log,
        repository_root=repository_root,
        temp_root=temp_root,
        user_home=user_home,
        environment_prefix=environment_prefix,
        host_name=host_name,
    )
    junit_after = junit_observation(sanitized_junit)
    log_after = log_observation(sanitized_log)
    if junit_before != junit_after:
        raise SanitizationError("JUnit semantics changed during sanitization")
    if log_before != log_after:
        raise SanitizationError("pytest log semantics changed during sanitization")

    rules = []
    replacements = {
        "repository_root": "__REPO_ROOT__",
        "temp_root": "__TEMP_ROOT__",
        "user_home": "__USER_HOME__",
        "environment_prefix": "__PYTHON_PREFIX__",
        "username": "__USERNAME__",
        "host_name": "__HOSTNAME__",
    }
    for rule_id, replacement in replacements.items():
        rules.append(
            {
                "id": rule_id,
                "replacement": replacement,
                "match_counts": {
                    "junit": junit_replacements[rule_id],
                    "log": log_replacements[rule_id],
                },
            }
        )

    public_fixtures = [
        {
            "literal": literal,
            "classification": "PUBLIC_SOURCE_SYNTHETIC_FIXTURE_NOT_MACHINE_OBSERVATION",
            "occurrence_counts": {
                "junit": junit_fixture_counts[literal],
                "log": log_fixture_counts[literal],
            },
        }
        for literal in _PUBLIC_WINDOWS_PATH_FIXTURES
    ]

    core: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": PASS_STATUS,
        "identity": IDENTITY,
        "reference_checkout": {
            "commit_sha1": reference_commit,
            "git_tree_sha1": reference_tree,
        },
        "sanitizer_source": _artifact(
            source_raw, path=SANITIZER_SOURCE_PATH
        ),
        "sanitization_contract": {
            "encoding": "UTF-8_STRICT",
            "newline_policy": "PRESERVE_INPUT_BYTES",
            "replacement_order": [
                "repository_root",
                "temp_root",
                "user_home",
                "environment_prefix",
                "username",
                "host_name",
            ],
            "rules": rules,
            "residual_declared_sensitive_value_count": 0,
            "residual_unallowlisted_windows_absolute_path_count": 0,
            "preserved_public_source_windows_path_fixtures": public_fixtures,
            "sensitive_values_recorded_in_receipt": False,
        },
        "raw_inputs": {
            "junit": _artifact(raw_junit),
            "log": _artifact(raw_log),
        },
        "sanitized_outputs": {
            "junit": _artifact(
                sanitized_junit,
                path="evidence/public_checkout/full_repository.sanitized.junit.xml",
            ),
            "log": _artifact(
                sanitized_log,
                path="evidence/public_checkout/full_repository.sanitized.log",
            ),
        },
        "semantic_equivalence": {
            "exact_statistics_preserved": True,
            "exact_failure_node_set_preserved": True,
            "exact_testcase_identity_set_preserved": True,
            "junit_before": junit_before,
            "junit_after": junit_after,
            "log_before": log_before,
            "log_after": log_after,
        },
        "claim_boundary": {
            "repository_wide_green": False,
            "scientific_stage_authorized": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_authorized": False,
            "ijoc_submission_authorized": False,
        },
    }
    receipt = {
        **core,
        "receipt_payload_sha256": _sha256(_canonical_json(core)),
    }
    _write_exclusive(output_junit_path, sanitized_junit)
    _write_exclusive(output_log_path, sanitized_log)
    _write_exclusive(receipt_path, _canonical_json(receipt) + b"\n")
    return receipt


def _validate_artifact_record(
    value: object, *, expected_path: str | None
) -> dict[str, object]:
    required = {"bytes", "sha256"}
    if expected_path is not None:
        required.add("path")
    if type(value) is not dict or set(value) != required:
        raise SanitizationError("artifact record shape drifted")
    if type(value.get("bytes")) is not int or value["bytes"] < 0:
        raise SanitizationError("artifact byte count is invalid")
    digest = value.get("sha256")
    if type(digest) is not str or not _SHA256_RE.fullmatch(digest):
        raise SanitizationError("artifact SHA-256 is invalid")
    if expected_path is not None and value.get("path") != expected_path:
        raise SanitizationError("artifact path drifted")
    return value


def _validate_receipt_contract(receipt: Mapping[str, object]) -> None:
    required = {
        "schema",
        "status",
        "identity",
        "reference_checkout",
        "sanitizer_source",
        "sanitization_contract",
        "raw_inputs",
        "sanitized_outputs",
        "semantic_equivalence",
        "claim_boundary",
        "receipt_payload_sha256",
    }
    if set(receipt) != required:
        raise SanitizationError("receipt key set drifted")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != PASS_STATUS:
        raise SanitizationError("receipt identity/status drifted")
    if receipt.get("identity") != IDENTITY:
        raise SanitizationError("receipt distribution identity drifted")

    checkout = receipt.get("reference_checkout")
    if type(checkout) is not dict or set(checkout) != {
        "commit_sha1",
        "git_tree_sha1",
    }:
        raise SanitizationError("reference checkout shape drifted")
    for field in ("commit_sha1", "git_tree_sha1"):
        value = checkout.get(field)
        if type(value) is not str or not _SHA1_RE.fullmatch(value):
            raise SanitizationError(f"reference checkout {field} is invalid")

    _validate_artifact_record(
        receipt.get("sanitizer_source"), expected_path=SANITIZER_SOURCE_PATH
    )
    raw_inputs = receipt.get("raw_inputs")
    if type(raw_inputs) is not dict or set(raw_inputs) != {"junit", "log"}:
        raise SanitizationError("raw_inputs shape drifted")
    for value in raw_inputs.values():
        _validate_artifact_record(value, expected_path=None)
    outputs = receipt.get("sanitized_outputs")
    if type(outputs) is not dict or set(outputs) != {"junit", "log"}:
        raise SanitizationError("sanitized_outputs shape drifted")
    _validate_artifact_record(
        outputs["junit"],
        expected_path="evidence/public_checkout/full_repository.sanitized.junit.xml",
    )
    _validate_artifact_record(
        outputs["log"],
        expected_path="evidence/public_checkout/full_repository.sanitized.log",
    )

    contract = receipt.get("sanitization_contract")
    if type(contract) is not dict or set(contract) != {
        "encoding",
        "newline_policy",
        "replacement_order",
        "rules",
        "residual_declared_sensitive_value_count",
        "residual_unallowlisted_windows_absolute_path_count",
        "preserved_public_source_windows_path_fixtures",
        "sensitive_values_recorded_in_receipt",
    }:
        raise SanitizationError("sanitization contract shape drifted")
    expected_ids = [
        "repository_root",
        "temp_root",
        "user_home",
        "environment_prefix",
        "username",
        "host_name",
    ]
    if contract.get("encoding") != "UTF-8_STRICT":
        raise SanitizationError("sanitization encoding drifted")
    if contract.get("newline_policy") != "PRESERVE_INPUT_BYTES":
        raise SanitizationError("newline policy drifted")
    if contract.get("replacement_order") != expected_ids:
        raise SanitizationError("replacement order drifted")
    expected_replacements = {
        "repository_root": "__REPO_ROOT__",
        "temp_root": "__TEMP_ROOT__",
        "user_home": "__USER_HOME__",
        "environment_prefix": "__PYTHON_PREFIX__",
        "username": "__USERNAME__",
        "host_name": "__HOSTNAME__",
    }
    rules = contract.get("rules")
    if type(rules) is not list or len(rules) != len(expected_ids):
        raise SanitizationError("sanitization rule list drifted")
    for rule_id, rule in zip(expected_ids, rules, strict=True):
        if type(rule) is not dict or set(rule) != {
            "id",
            "replacement",
            "match_counts",
        }:
            raise SanitizationError("sanitization rule shape drifted")
        if rule.get("id") != rule_id:
            raise SanitizationError("sanitization rule order/id drifted")
        if rule.get("replacement") != expected_replacements[rule_id]:
            raise SanitizationError("sanitization replacement drifted")
        counts = rule.get("match_counts")
        if type(counts) is not dict or set(counts) != {"junit", "log"}:
            raise SanitizationError("sanitization match-count shape drifted")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise SanitizationError("sanitization match count is invalid")
    if contract.get("residual_declared_sensitive_value_count") != 0:
        raise SanitizationError("receipt admits residual sensitive values")
    if contract.get("residual_unallowlisted_windows_absolute_path_count") != 0:
        raise SanitizationError("receipt admits unallowlisted Windows paths")
    fixtures = contract.get("preserved_public_source_windows_path_fixtures")
    if type(fixtures) is not list or len(fixtures) != len(
        _PUBLIC_WINDOWS_PATH_FIXTURES
    ):
        raise SanitizationError("public Windows path fixture allowlist drifted")
    for literal, fixture in zip(_PUBLIC_WINDOWS_PATH_FIXTURES, fixtures, strict=True):
        if type(fixture) is not dict or set(fixture) != {
            "literal",
            "classification",
            "occurrence_counts",
        }:
            raise SanitizationError("public path fixture record shape drifted")
        if fixture.get("literal") != literal:
            raise SanitizationError("public path fixture literal/order drifted")
        if fixture.get("classification") != (
            "PUBLIC_SOURCE_SYNTHETIC_FIXTURE_NOT_MACHINE_OBSERVATION"
        ):
            raise SanitizationError("public path fixture classification drifted")
        counts = fixture.get("occurrence_counts")
        if type(counts) is not dict or set(counts) != {"junit", "log"}:
            raise SanitizationError("public path fixture count shape drifted")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise SanitizationError("public path fixture count is invalid")
    if contract.get("sensitive_values_recorded_in_receipt") is not False:
        raise SanitizationError("receipt admits recorded sensitive values")

    boundary = receipt.get("claim_boundary")
    expected_boundary = {
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
    }
    if boundary != expected_boundary:
        raise SanitizationError("claim boundary drifted")


def verify_sanitized_bundle(
    *,
    receipt_path: Path,
    junit_path: Path,
    log_path: Path,
    sanitizer_source_path: Path,
) -> dict[str, object]:
    """Verify checked-in sanitized evidence without requiring private raw files."""

    receipt, _receipt_raw = _strict_json(receipt_path)
    _validate_receipt_contract(receipt)
    declared_hash = receipt.get("receipt_payload_sha256")
    if type(declared_hash) is not str or not _SHA256_RE.fullmatch(declared_hash):
        raise SanitizationError("invalid receipt payload SHA-256")
    core = dict(receipt)
    del core["receipt_payload_sha256"]
    if declared_hash != _sha256(_canonical_json(core)):
        raise SanitizationError("receipt payload self-hash mismatch")
    source_raw = sanitizer_source_path.resolve(strict=True).read_bytes()
    if receipt.get("sanitizer_source") != _artifact(
        source_raw, path=SANITIZER_SOURCE_PATH
    ):
        raise SanitizationError("sanitizer source hash/size drifted")
    junit_raw = junit_path.resolve(strict=True).read_bytes()
    log_raw = log_path.resolve(strict=True).read_bytes()
    outputs = receipt.get("sanitized_outputs")
    if type(outputs) is not dict:
        raise SanitizationError("sanitized_outputs shape drifted")
    expected_junit = _artifact(
        junit_raw,
        path="evidence/public_checkout/full_repository.sanitized.junit.xml",
    )
    expected_log = _artifact(
        log_raw,
        path="evidence/public_checkout/full_repository.sanitized.log",
    )
    if outputs.get("junit") != expected_junit:
        raise SanitizationError("sanitized JUnit hash/size drifted")
    if outputs.get("log") != expected_log:
        raise SanitizationError("sanitized log hash/size drifted")
    observed_fixture_counts: dict[str, dict[str, int]] = {}
    for label, raw in (("junit", junit_raw), ("log", log_raw)):
        fixture_counts, residual_count = _public_fixture_counts_and_residuals(raw)
        if residual_count:
            raise SanitizationError(
                f"sanitized {label} contains an unallowlisted Windows absolute path"
            )
        observed_fixture_counts[label] = fixture_counts
    contract = receipt["sanitization_contract"]
    for fixture in contract["preserved_public_source_windows_path_fixtures"]:
        literal = fixture["literal"]
        if fixture["occurrence_counts"] != {
            "junit": observed_fixture_counts["junit"][literal],
            "log": observed_fixture_counts["log"][literal],
        }:
            raise SanitizationError("public path fixture occurrence count drifted")

    equivalence = receipt.get("semantic_equivalence")
    if type(equivalence) is not dict:
        raise SanitizationError("semantic_equivalence shape drifted")
    if equivalence.get("exact_statistics_preserved") is not True:
        raise SanitizationError("statistics-preserved verdict drifted")
    if equivalence.get("exact_failure_node_set_preserved") is not True:
        raise SanitizationError("failure-node-set verdict drifted")
    if equivalence.get("exact_testcase_identity_set_preserved") is not True:
        raise SanitizationError("testcase-identity-set verdict drifted")
    if equivalence.get("junit_before") != equivalence.get("junit_after"):
        raise SanitizationError("receipt does not bind equal JUnit observations")
    if equivalence.get("log_before") != equivalence.get("log_after"):
        raise SanitizationError("receipt does not bind equal log observations")
    if junit_observation(junit_raw) != equivalence.get("junit_after"):
        raise SanitizationError("sanitized JUnit semantics drifted")
    if log_observation(log_raw) != equivalence.get("log_after"):
        raise SanitizationError("sanitized log semantics drifted")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create an exclusive bundle")
    create.add_argument("--raw-junit", type=Path, required=True)
    create.add_argument("--raw-log", type=Path, required=True)
    create.add_argument("--output-junit", type=Path, required=True)
    create.add_argument("--output-log", type=Path, required=True)
    create.add_argument("--receipt", type=Path, required=True)
    create.add_argument("--repository-root", required=True)
    create.add_argument("--temp-root", required=True)
    create.add_argument("--user-home", required=True)
    create.add_argument("--environment-prefix", required=True)
    create.add_argument("--host-name", required=True)
    create.add_argument("--reference-commit", required=True)
    create.add_argument("--reference-tree", required=True)
    create.add_argument(
        "--sanitizer-source",
        type=Path,
        default=Path(__file__),
    )
    verify = subparsers.add_parser("verify", help="verify a checked-in bundle")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--junit", type=Path, required=True)
    verify.add_argument("--log", type=Path, required=True)
    verify.add_argument(
        "--sanitizer-source",
        type=Path,
        default=Path(__file__),
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            receipt = create_sanitized_bundle(
                raw_junit_path=args.raw_junit,
                raw_log_path=args.raw_log,
                output_junit_path=args.output_junit,
                output_log_path=args.output_log,
                receipt_path=args.receipt,
                repository_root=args.repository_root,
                temp_root=args.temp_root,
                user_home=args.user_home,
                environment_prefix=args.environment_prefix,
                host_name=args.host_name,
                reference_commit=args.reference_commit,
                reference_tree=args.reference_tree,
                sanitizer_source_path=args.sanitizer_source,
            )
        else:
            receipt = verify_sanitized_bundle(
                receipt_path=args.receipt,
                junit_path=args.junit,
                log_path=args.log,
                sanitizer_source_path=args.sanitizer_source,
            )
    except (SanitizationError, FileNotFoundError, OSError) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PASS_STATUS",
    "RECEIPT_SCHEMA",
    "SanitizationError",
    "create_sanitized_bundle",
    "junit_observation",
    "log_observation",
    "main",
    "verify_sanitized_bundle",
]
