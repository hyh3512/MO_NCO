"""Verify the exact frozen-V8 failure contract in a pytest JUnit file.

This verifier accepts only the versioned node-id allowlist.  Matching a failure
message is necessary but never sufficient: an unregistered failure with the
same text is rejected, as are missing registered failures, errors, count drift,
or pytest xfail outcomes.
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


REGISTRY_SCHEMA = "v21e3r1_v9r2r1_expected_historical_v8_failure_set_v1"
RECEIPT_SCHEMA = "v21e3r1_v9r2r1_historical_failure_set_verification_v1"
PASS_STATUS = "PASS_EXACT_EXPECTED_HISTORICAL_V8_FAILURE_SET"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class HistoricalFailureSetError(ValueError):
    """Raised when the registry or observed JUnit contract is invalid."""


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
            raise HistoricalFailureSetError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise HistoricalFailureSetError(f"non-finite JSON value prohibited: {value}")


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
        raise HistoricalFailureSetError(f"invalid strict JSON: {path}") from error
    if type(payload) is not dict:
        raise HistoricalFailureSetError("registry must be a JSON object")
    if raw != _canonical_json(payload) + b"\n":
        raise HistoricalFailureSetError(
            "registry must be canonical JSON followed by exactly one newline"
        )
    return payload, raw


def _safe_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise HistoricalFailureSetError("evidence path must be a POSIX string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise HistoricalFailureSetError(f"unsafe evidence path: {value!r}")
    return value


def _validate_registry(payload: Mapping[str, object]) -> dict[str, object]:
    required = {
        "schema",
        "identity",
        "historical_source_root_sha256",
        "current_source_root_sha256",
        "reference_evidence",
        "expected_counts",
        "required_failure_text_exact",
        "node_ids",
        "xfail_allowed",
        "repository_wide_green",
        "scientific_stage_authorized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "ijoc_submission_authorized",
        "manifest_payload_sha256",
    }
    if set(payload) != required:
        raise HistoricalFailureSetError("registry key set drifted")
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise HistoricalFailureSetError("unexpected registry schema")
    identity = payload.get("identity")
    if identity != {
        "distribution": "mo-nco",
        "revision": "V21E3R1_V9R2R1",
        "version": "0.21.3.14",
    }:
        raise HistoricalFailureSetError("registry identity drifted")
    for field in ("historical_source_root_sha256", "current_source_root_sha256"):
        value = payload.get(field)
        if type(value) is not str or not _SHA256_RE.fullmatch(value):
            raise HistoricalFailureSetError(f"invalid {field}")
    declared = payload.get("manifest_payload_sha256")
    if type(declared) is not str or not _SHA256_RE.fullmatch(declared):
        raise HistoricalFailureSetError("invalid manifest_payload_sha256")
    core = dict(payload)
    del core["manifest_payload_sha256"]
    if declared != _sha256(_canonical_json(core)):
        raise HistoricalFailureSetError("registry payload self-hash mismatch")

    evidence = payload.get("reference_evidence")
    if type(evidence) is not dict or set(evidence) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise HistoricalFailureSetError("reference_evidence shape drifted")
    _safe_relative_path(evidence["path"])
    if type(evidence["bytes"]) is not int or evidence["bytes"] < 0:
        raise HistoricalFailureSetError("invalid reference evidence byte count")
    if type(evidence["sha256"]) is not str or not _SHA256_RE.fullmatch(
        evidence["sha256"]
    ):
        raise HistoricalFailureSetError("invalid reference evidence SHA-256")

    counts = payload.get("expected_counts")
    if type(counts) is not dict or set(counts) != {
        "testcases",
        "passed",
        "failures",
        "errors",
        "skipped",
        "reported_subtests",
    }:
        raise HistoricalFailureSetError("expected_counts shape drifted")
    expected_counts = {
        "testcases": 1356,
        "passed": 1344,
        "failures": 8,
        "errors": 0,
        "skipped": 4,
        "reported_subtests": 269,
    }
    if counts != expected_counts:
        raise HistoricalFailureSetError("expected count contract drifted")

    marker = payload.get("required_failure_text_exact")
    if marker != "Frozen diagnostic source manifest drifted":
        raise HistoricalFailureSetError("failure marker drifted")
    node_ids = payload.get("node_ids")
    if (
        type(node_ids) is not list
        or len(node_ids) != counts["failures"]
        or any(type(item) is not str or not item for item in node_ids)
        or node_ids != sorted(node_ids)
        or len(node_ids) != len(set(node_ids))
    ):
        raise HistoricalFailureSetError("node_ids must be eight unique sorted strings")
    if payload.get("xfail_allowed") is not False:
        raise HistoricalFailureSetError("xfail must remain prohibited")
    for field in (
        "repository_wide_green",
        "scientific_stage_authorized",
        "selection_authorized",
        "confirmation_authorized",
        "formal_authorized",
        "ijoc_submission_authorized",
    ):
        if payload.get(field) is not False:
            raise HistoricalFailureSetError(f"authorization boundary drifted: {field}")
    return dict(payload)


def _pytest_node_id(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname")
    name = testcase.attrib.get("name")
    if not classname or not name:
        raise HistoricalFailureSetError("JUnit testcase lacks classname or name")
    parts = classname.split(".")
    module_index = -1
    for index, part in enumerate(parts):
        if part.startswith("test_"):
            module_index = index
    if module_index < 0:
        raise HistoricalFailureSetError(
            f"cannot derive pytest node id from classname: {classname!r}"
        )
    module = "/".join(parts[: module_index + 1]) + ".py"
    suffix = [*parts[module_index + 1 :], name]
    return module + "::" + "::".join(suffix)


def _junit_observation(raw: bytes) -> dict[str, object]:
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise HistoricalFailureSetError("DTD/entity declarations are prohibited")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise HistoricalFailureSetError("invalid JUnit XML") from error
    testcases = list(root.iter("testcase"))
    failures: list[dict[str, str]] = []
    errors = 0
    skipped = 0
    for testcase in testcases:
        failure_children = testcase.findall("failure")
        error_children = testcase.findall("error")
        skipped_children = testcase.findall("skipped")
        outcomes = len(failure_children) + len(error_children) + len(skipped_children)
        if outcomes > 1:
            raise HistoricalFailureSetError("testcase carries multiple terminal outcomes")
        if failure_children:
            failure = failure_children[0]
            text = "\n".join(
                part
                for part in (failure.attrib.get("message", ""), failure.text or "")
                if part
            )
            failures.append({"node_id": _pytest_node_id(testcase), "text": text})
        elif error_children:
            errors += 1
        elif skipped_children:
            skipped += 1
            skipped_node = skipped_children[0]
            skipped_text = "\n".join(
                part
                for part in (
                    skipped_node.attrib.get("type", ""),
                    skipped_node.attrib.get("message", ""),
                    skipped_node.text or "",
                )
                if part
            ).casefold()
            if "xfail" in skipped_text:
                raise HistoricalFailureSetError("pytest xfail outcome is prohibited")
    counts = {
        "testcases": len(testcases),
        "passed": len(testcases) - len(failures) - errors - skipped,
        "failures": len(failures),
        "errors": errors,
        "skipped": skipped,
    }
    return {"counts": counts, "failures": failures}


def verify_expected_failure_set(
    registry_path: Path,
    junit_path: Path,
    *,
    require_reference_sha256: bool = False,
) -> dict[str, object]:
    registry_path = registry_path.resolve(strict=True)
    junit_path = junit_path.resolve(strict=True)
    registry, registry_raw = _strict_json(registry_path)
    registry = _validate_registry(registry)
    junit_raw = junit_path.read_bytes()
    reference = registry["reference_evidence"]
    if require_reference_sha256 and (
        len(junit_raw) != reference["bytes"]
        or _sha256(junit_raw) != reference["sha256"]
    ):
        raise HistoricalFailureSetError("reference JUnit bytes/hash drifted")

    observation = _junit_observation(junit_raw)
    observed_counts = observation["counts"]
    expected_counts = registry["expected_counts"]
    for field in ("testcases", "passed", "failures", "errors", "skipped"):
        if observed_counts[field] != expected_counts[field]:
            raise HistoricalFailureSetError(
                f"JUnit count drifted: {field}={observed_counts[field]!r}"
            )
    observed_failures = observation["failures"]
    observed_nodes = sorted(item["node_id"] for item in observed_failures)
    if observed_nodes != registry["node_ids"]:
        missing = sorted(set(registry["node_ids"]) - set(observed_nodes))
        unexpected = sorted(set(observed_nodes) - set(registry["node_ids"]))
        raise HistoricalFailureSetError(
            f"exact failure node-id set drifted; missing={missing!r}; "
            f"unexpected={unexpected!r}"
        )
    marker = registry["required_failure_text_exact"]
    wrong_text = sorted(
        item["node_id"] for item in observed_failures if marker not in item["text"]
    )
    if wrong_text:
        raise HistoricalFailureSetError(
            f"registered failures lack exact frozen-manifest marker: {wrong_text!r}"
        )

    core: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": PASS_STATUS,
        "registry": {
            "bytes": len(registry_raw),
            "sha256": _sha256(registry_raw),
            "manifest_payload_sha256": registry["manifest_payload_sha256"],
        },
        "junit": {
            "bytes": len(junit_raw),
            "sha256": _sha256(junit_raw),
            "reference_bytes_required": require_reference_sha256,
        },
        "counts": observed_counts,
        "exact_failure_node_ids": observed_nodes,
        "required_failure_text_exact": marker,
        "repository_wide_green": False,
        "scientific_stage_authorized": False,
        "selection_authorized": False,
        "confirmation_authorized": False,
        "formal_authorized": False,
        "ijoc_submission_authorized": False,
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
        raise HistoricalFailureSetError(f"refusing to overwrite: {path}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-reference-sha256", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = verify_expected_failure_set(
            args.registry,
            args.junit,
            require_reference_sha256=args.require_reference_sha256,
        )
        if args.output is not None:
            _write_exclusive(args.output, receipt)
    except (HistoricalFailureSetError, FileNotFoundError, OSError) as error:
        print(f"FAIL_CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HistoricalFailureSetError",
    "PASS_STATUS",
    "RECEIPT_SCHEMA",
    "REGISTRY_SCHEMA",
    "main",
    "verify_expected_failure_set",
]
