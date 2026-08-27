"""Verify the expected non-green failure contract for the public checkout.

The public engineering checkout intentionally omits rights-sensitive, sealed,
and formal-study material.  Consequently, its full repository test run is not
green.  This verifier makes that state explicit and fail-closed: both pytest's
78 failed/subfailed summary outcomes and JUnit's 77 failing testcase identities
must match the versioned registry exactly.

A successful verification means only that the observed non-green run matches
the declared public-checkout contract.  It never authorizes selection,
confirmation, formal study, scientific claims, or submission.
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


REGISTRY_SCHEMA = "v21e3r1_v9r2r1_expected_public_checkout_failure_set_v2"
RECEIPT_SCHEMA = "v21e3r1_v9r2r1_public_checkout_failure_set_verification_v2"
PASS_STATUS = "PASS_EXACT_EXPECTED_PUBLIC_CHECKOUT_NON_GREEN_FAILURE_SET"
REFERENCE_COMMIT = "f6ad6a73ea9e2c46eeadded3f4446775097fdc48"
REFERENCE_TREE = "368bf76b61938521d073130a50ebe8cb876af41a"
REFERENCE_SOURCE_ROOT = (
    "50ad30da8670eb488848e6db084084185fea7725e86c7fea480639caa193d9eb"
)
REFERENCE_TEST_SOURCE_ROOT = (
    "55850b6bc1e75c50ce8aef7efd54dc7841e7d2b8f8ab467bc911fc5235fa8e86"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SUMMARY_RE = re.compile(
    r"(?P<failed>\d+) failed, (?P<passed>\d+) passed, "
    r"(?P<skipped>\d+) skipped, (?P<subtests>\d+) subtests passed in "
    r"(?P<duration>[^\r\n]+)\Z"
)
_SUBFAILED_RE = re.compile(r"SUBFAILED\([^\r\n]+\) (?P<node>.+)\Z")

HELD_CATEGORY = "HELD_OR_RIGHTS_SENSITIVE_DEPENDENCY"
FROZEN_V8_CATEGORY = "FROZEN_V8_FAIL_CLOSED"
SEALED_CATEGORY = "SEALED_OUTPUT"
ALLOWED_CATEGORIES = {HELD_CATEGORY, FROZEN_V8_CATEGORY, SEALED_CATEGORY}

FROZEN_V8_NODE_IDS = frozenset(
    {
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_external_scheduling_missing_bound_file_is_fail_closed",
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_external_scheduling_raw_receipt_tamper_is_fail_closed",
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_external_scheduling_schema_authority_and_cross_hash_drift_fail_closed"
        "[external-scheduling s01 claim-handoff_receipt_sha256-"
        "0000000000000000000000000000000000000000000000000000000000000000-"
        "claim semantic/cross-hash drifted]",
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_external_scheduling_schema_authority_and_cross_hash_drift_fail_closed"
        "[external-scheduling s01 success receipt-runtime_authority-True-"
        "receipt semantic/cross-hash drifted]",
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_external_scheduling_schema_authority_and_cross_hash_drift_fail_closed"
        "[external-scheduling s01 success receipt-schema-drifted_schema-"
        "receipt semantic/cross-hash drifted]",
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_external_scheduling_schema_authority_and_cross_hash_drift_fail_closed"
        "[external-scheduling s01 success seal-receipt_sha256-"
        "0000000000000000000000000000000000000000000000000000000000000000-"
        "seal semantic/cross-hash drifted]",
        "tests/test_v21e3r1_frozen_diagnostic_metric_timeout_recovery_continuation.py::"
        "test_real_incident_preflight_is_read_only_and_requires_fresh_exact17",
    }
)
FROZEN_MANIFEST_MARKER = "Frozen diagnostic source manifest drifted"
HISTORICAL_INTERPRETER_MARKER = (
    "Helper must use the exact historical main-job interpreter"
)
EXPECTED_FROZEN_V8_FAILURE_MARKER_CONTRACT = {
    "default_allowed_markers": [
        FROZEN_MANIFEST_MARKER,
        HISTORICAL_INTERPRETER_MARKER,
    ],
    "node_overrides": {},
}
SEALED_NODE_IDS = frozenset(
    {
        "tests/test_v21e3r1_successor_metric.py::"
        "test_real_n500_trace_matches_sealed_golden_and_exposes_distinct_identity"
    }
)


class PublicCheckoutFailureSetError(ValueError):
    """Raised when the declared or observed public failure contract drifts."""


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
            raise PublicCheckoutFailureSetError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise PublicCheckoutFailureSetError(f"non-finite JSON value prohibited: {value}")


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
        raise PublicCheckoutFailureSetError(f"invalid strict JSON: {path}") from error
    if type(payload) is not dict:
        raise PublicCheckoutFailureSetError("registry must be a JSON object")
    if raw != _canonical_json(payload) + b"\n":
        raise PublicCheckoutFailureSetError(
            "registry must be canonical JSON followed by exactly one newline"
        )
    return payload, raw


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise PublicCheckoutFailureSetError("evidence path must be a POSIX string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise PublicCheckoutFailureSetError(f"unsafe evidence path: {value!r}")
    return value


def _validate_digest(value: object, field: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise PublicCheckoutFailureSetError(f"invalid {field}")
    return value


def _expected_category(node_id: str) -> str:
    if node_id in FROZEN_V8_NODE_IDS:
        return FROZEN_V8_CATEGORY
    if node_id in SEALED_NODE_IDS:
        return SEALED_CATEGORY
    return HELD_CATEGORY


def _parse_registry_outcome(item: object) -> dict[str, str]:
    if type(item) is not dict or set(item) != {
        "category",
        "kind",
        "pytest_node_id",
        "summary_line",
    }:
        raise PublicCheckoutFailureSetError("pytest outcome entry shape drifted")
    if any(type(item[field]) is not str or not item[field] for field in item):
        raise PublicCheckoutFailureSetError("pytest outcome fields must be strings")
    kind = item["kind"]
    line = item["summary_line"]
    node_id = item["pytest_node_id"]
    if kind == "FAILED":
        if line != f"FAILED {node_id}":
            raise PublicCheckoutFailureSetError("FAILED summary line/node mismatch")
    elif kind == "SUBFAILED":
        match = _SUBFAILED_RE.fullmatch(line)
        if match is None or match.group("node") != node_id:
            raise PublicCheckoutFailureSetError("SUBFAILED summary line/node mismatch")
    else:
        raise PublicCheckoutFailureSetError(f"unexpected pytest outcome kind: {kind!r}")
    expected_category = _expected_category(node_id)
    if item["category"] != expected_category:
        raise PublicCheckoutFailureSetError(
            f"failure classification drifted for {line!r}: "
            f"expected {expected_category!r}"
        )
    return dict(item)


def _validate_registry(payload: Mapping[str, object]) -> dict[str, object]:
    required = {
        "authorization_boundaries",
        "classification_counts",
        "expected_counts",
        "frozen_v8_failure_marker_contract",
        "expected_junit_failure_signatures",
        "expected_junit_failure_or_error_node_ids",
        "expected_pytest_outcomes",
        "identity",
        "junit_failure_or_error_node_ids_sha256",
        "junit_failure_signatures_sha256",
        "manifest_payload_sha256",
        "pytest_outcome_summary_lines_sha256",
        "reference_evidence",
        "reference_environment",
        "reference_git",
        "reference_source_closure",
        "schema",
        "xfail_allowed",
    }
    if set(payload) != required:
        raise PublicCheckoutFailureSetError("registry key set drifted")
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise PublicCheckoutFailureSetError("unexpected registry schema")
    if payload.get("identity") != {
        "distribution": "mo-nco",
        "revision": "V21E3R1_V9R2R1",
        "version": "0.21.3.14",
    }:
        raise PublicCheckoutFailureSetError("registry identity drifted")
    if payload.get("reference_git") != {
        "commit": REFERENCE_COMMIT,
        "tree": REFERENCE_TREE,
    }:
        raise PublicCheckoutFailureSetError("reference Git identity drifted")
    if payload.get("reference_source_closure") != {
        "source_file_count": 203,
        "source_root_sha256": REFERENCE_SOURCE_ROOT,
        "test_module_count": 136,
        "test_source_root_sha256": REFERENCE_TEST_SOURCE_ROOT,
    }:
        raise PublicCheckoutFailureSetError("reference source closure drifted")
    if payload.get("reference_environment") != {
        "interpreter": "CPython 3.13.12",
        "platform": "Windows",
    }:
        raise PublicCheckoutFailureSetError("reference environment drifted")

    expected_boundaries = {
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
        "selection_authorized": False,
    }
    if payload.get("authorization_boundaries") != expected_boundaries:
        raise PublicCheckoutFailureSetError("authorization boundary drifted")
    if payload.get("xfail_allowed") is not False:
        raise PublicCheckoutFailureSetError("pytest xfail must remain prohibited")

    expected_counts = {
        "junit_error_testcases": 0,
        "junit_failure_children": 78,
        "junit_failure_or_error_testcases": 77,
        "junit_passed_testcases": 1327,
        "junit_skipped_testcases": 4,
        "junit_testcases": 1408,
        "pytest_failed_or_subfailed_outcomes": 78,
        "pytest_failed_outcomes": 76,
        "pytest_passed": 1328,
        "pytest_skipped": 4,
        "pytest_subfailed_outcomes": 2,
        "pytest_subtests_passed": 267,
    }
    if payload.get("expected_counts") != expected_counts:
        raise PublicCheckoutFailureSetError("reference count contract drifted")
    expected_classification = {
        "frozen_v8_fail_closed": 7,
        "held_or_rights_sensitive_dependency": 70,
        "sealed_output": 1,
        "unclassified": 0,
    }
    if payload.get("classification_counts") != expected_classification:
        raise PublicCheckoutFailureSetError("classification count contract drifted")
    if (
        payload.get("frozen_v8_failure_marker_contract")
        != EXPECTED_FROZEN_V8_FAILURE_MARKER_CONTRACT
    ):
        raise PublicCheckoutFailureSetError(
            "frozen-V8 failure marker contract drifted"
        )

    evidence = payload.get("reference_evidence")
    if type(evidence) is not dict or set(evidence) != {"junit", "log"}:
        raise PublicCheckoutFailureSetError("reference evidence shape drifted")
    expected_paths = {
        "junit": "evidence/public_checkout/full_repository.sanitized.junit.xml",
        "log": "evidence/public_checkout/full_repository.sanitized.log",
    }
    for name, item in evidence.items():
        if type(item) is not dict or set(item) != {"bytes", "path", "sha256"}:
            raise PublicCheckoutFailureSetError(
                f"reference evidence entry shape drifted: {name}"
            )
        if _safe_relative_path(item["path"]) != expected_paths[name]:
            raise PublicCheckoutFailureSetError(
                f"reference evidence path drifted: {name}"
            )
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise PublicCheckoutFailureSetError(
                f"invalid reference evidence byte count: {name}"
            )
        _validate_digest(item["sha256"], f"reference {name} SHA-256")

    node_ids = payload.get("expected_junit_failure_or_error_node_ids")
    if (
        type(node_ids) is not list
        or len(node_ids) != expected_counts["junit_failure_or_error_testcases"]
        or any(type(item) is not str or not item for item in node_ids)
        or node_ids != sorted(node_ids)
        or len(node_ids) != len(set(node_ids))
    ):
        raise PublicCheckoutFailureSetError(
            "JUnit node IDs must be 77 unique sorted strings"
        )
    node_digest = _validate_digest(
        payload.get("junit_failure_or_error_node_ids_sha256"),
        "JUnit node-id set SHA-256",
    )
    if node_digest != _sha256(_canonical_json(node_ids)):
        raise PublicCheckoutFailureSetError("JUnit node-id set hash mismatch")

    raw_signatures = payload.get("expected_junit_failure_signatures")
    if type(raw_signatures) is not list:
        raise PublicCheckoutFailureSetError(
            "expected_junit_failure_signatures must be a list"
        )
    signatures: list[dict[str, object]] = []
    for item in raw_signatures:
        if type(item) is not dict or set(item) != {
            "category",
            "exception_types",
            "failure_child_count",
            "node_id",
        }:
            raise PublicCheckoutFailureSetError(
                "JUnit failure signature entry shape drifted"
            )
        if (
            type(item["node_id"]) is not str
            or not item["node_id"]
            or item["category"] != _expected_category(item["node_id"])
            or type(item["failure_child_count"]) is not int
            or item["failure_child_count"] <= 0
            or type(item["exception_types"]) is not list
            or len(item["exception_types"]) != item["failure_child_count"]
            or any(
                type(value) is not str or not value
                for value in item["exception_types"]
            )
            or item["exception_types"] != sorted(item["exception_types"])
        ):
            raise PublicCheckoutFailureSetError(
                "JUnit failure signature value drifted"
            )
        signatures.append(dict(item))
    if (
        [item["node_id"] for item in signatures] != node_ids
        or sum(item["failure_child_count"] for item in signatures) != 78
    ):
        raise PublicCheckoutFailureSetError(
            "JUnit failure signatures and node/count contract disagree"
        )
    signature_digest = _validate_digest(
        payload.get("junit_failure_signatures_sha256"),
        "JUnit failure signature set SHA-256",
    )
    if signature_digest != _sha256(_canonical_json(signatures)):
        raise PublicCheckoutFailureSetError("JUnit failure signature hash mismatch")

    raw_outcomes = payload.get("expected_pytest_outcomes")
    if type(raw_outcomes) is not list:
        raise PublicCheckoutFailureSetError("expected_pytest_outcomes must be a list")
    outcomes = [_parse_registry_outcome(item) for item in raw_outcomes]
    summary_lines = [item["summary_line"] for item in outcomes]
    if (
        len(outcomes) != expected_counts["pytest_failed_or_subfailed_outcomes"]
        or summary_lines != sorted(summary_lines)
        or len(summary_lines) != len(set(summary_lines))
    ):
        raise PublicCheckoutFailureSetError(
            "pytest outcomes must be 78 unique entries sorted by summary line"
        )
    if sum(item["kind"] == "FAILED" for item in outcomes) != 76:
        raise PublicCheckoutFailureSetError("FAILED outcome count drifted")
    if sum(item["kind"] == "SUBFAILED" for item in outcomes) != 2:
        raise PublicCheckoutFailureSetError("SUBFAILED outcome count drifted")
    category_counts = {
        HELD_CATEGORY: sum(item["category"] == HELD_CATEGORY for item in outcomes),
        FROZEN_V8_CATEGORY: sum(
            item["category"] == FROZEN_V8_CATEGORY for item in outcomes
        ),
        SEALED_CATEGORY: sum(item["category"] == SEALED_CATEGORY for item in outcomes),
    }
    if category_counts != {
        HELD_CATEGORY: 70,
        FROZEN_V8_CATEGORY: 7,
        SEALED_CATEGORY: 1,
    }:
        raise PublicCheckoutFailureSetError("per-outcome classification count drifted")
    if set(item["pytest_node_id"] for item in outcomes) != set(node_ids):
        raise PublicCheckoutFailureSetError(
            "pytest and JUnit failure identity sets are not equivalent"
        )
    outcome_digest = _validate_digest(
        payload.get("pytest_outcome_summary_lines_sha256"),
        "pytest outcome set SHA-256",
    )
    if outcome_digest != _sha256(_canonical_json(summary_lines)):
        raise PublicCheckoutFailureSetError("pytest outcome set hash mismatch")

    declared = _validate_digest(
        payload.get("manifest_payload_sha256"), "manifest payload SHA-256"
    )
    core = dict(payload)
    del core["manifest_payload_sha256"]
    if declared != _sha256(_canonical_json(core)):
        raise PublicCheckoutFailureSetError("registry payload self-hash mismatch")
    return dict(payload)


def _pytest_node_id(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname")
    name = testcase.attrib.get("name")
    if not classname or not name:
        raise PublicCheckoutFailureSetError("JUnit testcase lacks classname or name")
    parts = classname.split(".")
    module_index = -1
    for index, part in enumerate(parts):
        if part.startswith("test_"):
            module_index = index
    if module_index < 0:
        raise PublicCheckoutFailureSetError(
            f"cannot derive pytest node id from classname: {classname!r}"
        )
    module = "/".join(parts[: module_index + 1]) + ".py"
    suffix = [*parts[module_index + 1 :], name]
    return module + "::" + "::".join(suffix)


def _matches_exact_marker_prefix(message: str, allowed_markers: Sequence[str]) -> bool:
    matches = [
        marker
        for marker in allowed_markers
        if message == marker or message.startswith(marker + " ")
    ]
    return len(matches) == 1


def _junit_observation(
    raw: bytes,
    frozen_marker_contract: Mapping[str, object],
) -> dict[str, object]:
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise PublicCheckoutFailureSetError("DTD/entity declarations are prohibited")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise PublicCheckoutFailureSetError("invalid JUnit XML") from error
    if root.tag != "testsuites":
        raise PublicCheckoutFailureSetError("JUnit root must be testsuites")
    suites = list(root.iter("testsuite"))
    if len(suites) != 1:
        raise PublicCheckoutFailureSetError("JUnit must contain exactly one testsuite")
    suite = suites[0]
    declared: dict[str, int] = {}
    for field in ("errors", "failures", "skipped", "tests"):
        value = suite.attrib.get(field)
        if value is None or re.fullmatch(r"[0-9]+", value) is None:
            raise PublicCheckoutFailureSetError(
                f"JUnit testsuite lacks strict integer {field!r} attribute"
            )
        declared[field] = int(value)
    testcases = list(root.iter("testcase"))
    failing_nodes: list[str] = []
    signatures: list[dict[str, object]] = []
    failure_children = 0
    error_children = 0
    skipped = 0
    for testcase in testcases:
        failures = testcase.findall("failure")
        errors = testcase.findall("error")
        skips = testcase.findall("skipped")
        terminal_kinds = bool(failures) + bool(errors) + bool(skips)
        if terminal_kinds > 1:
            raise PublicCheckoutFailureSetError(
                "JUnit testcase carries conflicting terminal outcome kinds"
            )
        if failures or errors:
            node_id = _pytest_node_id(testcase)
            failing_nodes.append(node_id)
            failure_children += len(failures)
            error_children += len(errors)
            terminal_children = [*failures, *errors]
            category = _expected_category(node_id)
            if category == FROZEN_V8_CATEGORY:
                overrides = frozen_marker_contract["node_overrides"]
                allowed_markers = overrides.get(
                    node_id,
                    frozen_marker_contract["default_allowed_markers"],
                )
                messages = [
                    child.attrib.get("message", "").partition(":")[2].strip()
                    for child in terminal_children
                ]
                if any(
                    not _matches_exact_marker_prefix(message, allowed_markers)
                    for message in messages
                ):
                    raise PublicCheckoutFailureSetError(
                        f"frozen-V8 failure marker drifted: {node_id!r}"
                    )
            texts = [
                "\n".join(
                    part
                    for part in (child.attrib.get("message", ""), child.text or "")
                    if part
                )
                for child in terminal_children
            ]
            if category == SEALED_CATEGORY and any(
                "trace.sqlite3" not in text for text in texts
            ):
                raise PublicCheckoutFailureSetError(
                    f"sealed-output failure marker drifted: {node_id!r}"
                )
            exception_types = sorted(
                child.attrib.get("message", "").partition(":")[0].strip()
                for child in terminal_children
            )
            if any(not value for value in exception_types):
                raise PublicCheckoutFailureSetError(
                    f"JUnit failure/error lacks exception type: {node_id!r}"
                )
            signatures.append(
                {
                    "category": category,
                    "exception_types": exception_types,
                    "failure_child_count": len(terminal_children),
                    "node_id": node_id,
                }
            )
        elif skips:
            if len(skips) != 1:
                raise PublicCheckoutFailureSetError(
                    "JUnit testcase carries multiple skipped outcomes"
                )
            skipped += 1
            text = "\n".join(
                part
                for part in (
                    skips[0].attrib.get("type", ""),
                    skips[0].attrib.get("message", ""),
                    skips[0].text or "",
                )
                if part
            ).casefold()
            if "xfail" in text:
                raise PublicCheckoutFailureSetError("pytest xfail outcome is prohibited")
    if len(failing_nodes) != len(set(failing_nodes)):
        raise PublicCheckoutFailureSetError("duplicate JUnit failing testcase node ID")
    counts = {
        "error_testcases": sum(
            bool(testcase.findall("error")) for testcase in testcases
        ),
        "failure_children": failure_children,
        "failure_or_error_testcases": len(failing_nodes),
        "passed_testcases": len(testcases) - len(failing_nodes) - skipped,
        "skipped_testcases": skipped,
        "testcases": len(testcases),
    }
    if error_children != counts["error_testcases"]:
        raise PublicCheckoutFailureSetError(
            "JUnit testcase carries multiple error children"
        )
    if (
        declared["errors"] != error_children
        or declared["failures"] != failure_children
        or declared["skipped"] != skipped
        or declared["tests"] < len(testcases)
    ):
        raise PublicCheckoutFailureSetError(
            "JUnit declared counts disagree with terminal outcome elements"
        )
    return {
        "counts": counts,
        "declared_counts": declared,
        "node_ids": sorted(failing_nodes),
        "signatures": sorted(signatures, key=lambda item: item["node_id"]),
    }


def _log_observation(raw: bytes) -> dict[str, object]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise PublicCheckoutFailureSetError("pytest log is not UTF-8") from error
    lines = text.splitlines()
    prohibited = [
        line
        for line in lines
        if line.startswith(("XFAIL ", "XPASS ", "XPASS(strict) "))
    ]
    if prohibited:
        raise PublicCheckoutFailureSetError("pytest xfail/xpass outcome is prohibited")
    summary_lines = sorted(
        line
        for line in lines
        if line.startswith("FAILED ") or _SUBFAILED_RE.fullmatch(line) is not None
    )
    if len(summary_lines) != len(set(summary_lines)):
        raise PublicCheckoutFailureSetError("duplicate pytest failed summary outcome")
    terminal_matches = [
        match
        for line in lines
        if (match := _SUMMARY_RE.fullmatch(line)) is not None
    ]
    if len(terminal_matches) != 1:
        raise PublicCheckoutFailureSetError(
            "pytest log must contain exactly one strict terminal summary"
        )
    match = terminal_matches[0]
    counts = {
        "failed_or_subfailed_outcomes": int(match.group("failed")),
        "failed_outcomes": sum(line.startswith("FAILED ") for line in summary_lines),
        "passed": int(match.group("passed")),
        "skipped": int(match.group("skipped")),
        "subfailed_outcomes": sum(
            line.startswith("SUBFAILED(") for line in summary_lines
        ),
        "subtests_passed": int(match.group("subtests")),
    }
    if counts["failed_or_subfailed_outcomes"] != len(summary_lines):
        raise PublicCheckoutFailureSetError(
            "pytest terminal failed count and summary outcomes disagree"
        )
    return {
        "counts": counts,
        "duration_text": match.group("duration"),
        "summary_lines": summary_lines,
    }


def verify_expected_failure_set(
    registry_path: Path,
    junit_path: Path,
    log_path: Path,
    *,
    require_reference_sha256: bool = False,
) -> dict[str, object]:
    """Verify an observed JUnit/log pair against the public failure registry."""

    registry_path = registry_path.resolve(strict=True)
    junit_path = junit_path.resolve(strict=True)
    log_path = log_path.resolve(strict=True)
    registry, registry_raw = _strict_json(registry_path)
    registry = _validate_registry(registry)
    junit_raw = junit_path.read_bytes()
    log_raw = log_path.read_bytes()
    if require_reference_sha256:
        for name, raw in (("junit", junit_raw), ("log", log_raw)):
            expected = registry["reference_evidence"][name]
            if len(raw) != expected["bytes"] or _sha256(raw) != expected["sha256"]:
                raise PublicCheckoutFailureSetError(
                    f"reference {name} bytes/hash drifted"
                )

    junit = _junit_observation(
        junit_raw,
        registry["frozen_v8_failure_marker_contract"],
    )
    log = _log_observation(log_raw)
    expected_counts = registry["expected_counts"]
    observed_junit_counts = junit["counts"]
    observed_junit_declared = junit["declared_counts"]
    observed_log_counts = log["counts"]

    exact_junit_fields = {
        "error_testcases": "junit_error_testcases",
        "failure_children": "junit_failure_children",
        "failure_or_error_testcases": "junit_failure_or_error_testcases",
        "skipped_testcases": "junit_skipped_testcases",
    }
    for observed_field, expected_field in exact_junit_fields.items():
        if observed_junit_counts[observed_field] != expected_counts[expected_field]:
            raise PublicCheckoutFailureSetError(
                f"JUnit count drifted: {observed_field}="
                f"{observed_junit_counts[observed_field]!r}"
            )
    exact_log_fields = {
        "failed_or_subfailed_outcomes": "pytest_failed_or_subfailed_outcomes",
        "failed_outcomes": "pytest_failed_outcomes",
        "skipped": "pytest_skipped",
        "subfailed_outcomes": "pytest_subfailed_outcomes",
        "subtests_passed": "pytest_subtests_passed",
    }
    for observed_field, expected_field in exact_log_fields.items():
        if observed_log_counts[observed_field] != expected_counts[expected_field]:
            raise PublicCheckoutFailureSetError(
                f"pytest log count drifted: {observed_field}="
                f"{observed_log_counts[observed_field]!r}"
            )
    if (
        observed_junit_declared["errors"] != expected_counts["junit_error_testcases"]
        or observed_junit_declared["failures"]
        != expected_counts["pytest_failed_or_subfailed_outcomes"]
        or observed_junit_declared["skipped"] != expected_counts["junit_skipped_testcases"]
        or observed_junit_declared["tests"]
        != observed_junit_counts["testcases"]
        + observed_log_counts["subtests_passed"]
        + observed_log_counts["subfailed_outcomes"]
        or observed_log_counts["passed"]
        != observed_junit_counts["passed_testcases"] + 1
    ):
        raise PublicCheckoutFailureSetError(
            "pytest log and JUnit declared/testcase counts disagree"
        )
    if require_reference_sha256:
        if (
            observed_junit_counts["testcases"] != expected_counts["junit_testcases"]
            or observed_junit_counts["passed_testcases"]
            != expected_counts["junit_passed_testcases"]
            or observed_log_counts["passed"] != expected_counts["pytest_passed"]
        ):
            raise PublicCheckoutFailureSetError("reference passed/testcase count drifted")
    else:
        if (
            observed_junit_counts["testcases"] < expected_counts["junit_testcases"]
            or observed_junit_counts["passed_testcases"]
            < expected_counts["junit_passed_testcases"]
            or observed_log_counts["passed"] < expected_counts["pytest_passed"]
        ):
            raise PublicCheckoutFailureSetError(
                "live passed/testcase count regressed below reference"
            )

    if junit["node_ids"] != registry["expected_junit_failure_or_error_node_ids"]:
        missing = sorted(
            set(registry["expected_junit_failure_or_error_node_ids"])
            - set(junit["node_ids"])
        )
        unexpected = sorted(
            set(junit["node_ids"])
            - set(registry["expected_junit_failure_or_error_node_ids"])
        )
        raise PublicCheckoutFailureSetError(
            f"exact JUnit failure node-id set drifted; missing={missing!r}; "
            f"unexpected={unexpected!r}"
        )
    if junit["signatures"] != registry["expected_junit_failure_signatures"]:
        raise PublicCheckoutFailureSetError(
            "exact JUnit failure exception/category signatures drifted"
        )
    expected_summary_lines = [
        item["summary_line"] for item in registry["expected_pytest_outcomes"]
    ]
    if log["summary_lines"] != expected_summary_lines:
        missing = sorted(set(expected_summary_lines) - set(log["summary_lines"]))
        unexpected = sorted(set(log["summary_lines"]) - set(expected_summary_lines))
        raise PublicCheckoutFailureSetError(
            f"exact pytest failure outcome set drifted; missing={missing!r}; "
            f"unexpected={unexpected!r}"
        )

    core: dict[str, object] = {
        "authorization_boundaries": registry["authorization_boundaries"],
        "classification_counts": registry["classification_counts"],
        "counts": {
            "junit": {**observed_junit_counts, "declared": observed_junit_declared},
            "pytest": observed_log_counts,
        },
        "evidence": {
            "junit": {"bytes": len(junit_raw), "sha256": _sha256(junit_raw)},
            "log": {"bytes": len(log_raw), "sha256": _sha256(log_raw)},
            "reference_bytes_required": require_reference_sha256,
        },
        "exact_junit_failure_or_error_node_ids": junit["node_ids"],
        "exact_junit_failure_signatures": junit["signatures"],
        "exact_pytest_failure_summary_lines": log["summary_lines"],
        "reference_git": registry["reference_git"],
        "registry": {
            "bytes": len(registry_raw),
            "manifest_payload_sha256": registry["manifest_payload_sha256"],
            "sha256": _sha256(registry_raw),
        },
        "schema": RECEIPT_SCHEMA,
        "status": PASS_STATUS,
    }
    return {**core, "receipt_payload_sha256": _sha256(_canonical_json(core))}


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    raw = _canonical_json(payload) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise PublicCheckoutFailureSetError(f"refusing to overwrite: {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-reference-evidence-sha256", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = verify_expected_failure_set(
            args.registry,
            args.junit,
            args.log,
            require_reference_sha256=args.require_reference_evidence_sha256,
        )
        if args.output is not None:
            _write_exclusive(args.output, receipt)
    except (PublicCheckoutFailureSetError, FileNotFoundError, OSError) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PASS_STATUS",
    "RECEIPT_SCHEMA",
    "REGISTRY_SCHEMA",
    "PublicCheckoutFailureSetError",
    "main",
    "verify_expected_failure_set",
]
