from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_expected_public_checkout_failure_set.py"
REGISTRY = ROOT / "provenance" / "V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json"
JUNIT = ROOT / "evidence" / "public_checkout" / "full_repository.sanitized.junit.xml"
LOG = ROOT / "evidence" / "public_checkout" / "full_repository.sanitized.log"
INTERPRETER_MARKER = "Helper must use the exact historical main-job interpreter"
MANIFEST_MARKER = "Frozen diagnostic source manifest drifted"


def _load_module():
    spec = importlib.util.spec_from_file_location("public_failure_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_module()
FROZEN_V8_NODES = tuple(sorted(VERIFIER.FROZEN_V8_NODE_IDS))


def _replace_failure_marker(
    tmp_path: Path,
    *,
    node_id: str,
    marker: str,
    exception_type: str = "RuntimeError",
) -> Path:
    tree = ET.parse(JUNIT)
    testcase = next(
        case
        for case in tree.getroot().iter("testcase")
        if VERIFIER._pytest_node_id(case) == node_id
    )
    terminal = testcase.find("failure")
    if terminal is None:
        terminal = testcase.find("error")
    assert terminal is not None
    terminal.attrib["message"] = f"{exception_type}: {marker}"
    terminal.text = marker
    output = tmp_path / "variant.junit.xml"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _append_unexpected_junit_failure(tmp_path: Path) -> tuple[Path, str]:
    tree = ET.parse(JUNIT)
    suite = next(tree.getroot().iter("testsuite"))
    node_id = "tests/test_public_contract_probe.py::test_unexpected_failure"
    testcase = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "tests.test_public_contract_probe",
            "name": "test_unexpected_failure",
            "time": "0.001",
        },
    )
    failure = ET.SubElement(
        testcase,
        "failure",
        {"message": "AssertionError: unexpected public failure"},
    )
    failure.text = "AssertionError: unexpected public failure"
    suite.attrib["tests"] = str(int(suite.attrib["tests"]) + 1)
    suite.attrib["failures"] = str(int(suite.attrib["failures"]) + 1)
    output = tmp_path / "unexpected-failure.junit.xml"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output, node_id


def test_unexpected_junit_node_is_reported_before_count_drift(
    tmp_path: Path,
) -> None:
    junit, node_id = _append_unexpected_junit_failure(tmp_path)

    with pytest.raises(
        VERIFIER.PublicCheckoutFailureSetError,
        match=(
            "exact JUnit failure node-id set drifted; missing=\\[\\]; "
            f"unexpected=\\['{node_id}'\\]"
        ),
    ):
        VERIFIER.verify_expected_failure_set(REGISTRY, junit, LOG)


@pytest.mark.parametrize(
    "terminal_failed_count",
    [78, 79],
    ids=["terminal-count-stale", "terminal-count-drifted"],
)
def test_duplicate_pytest_outcome_reports_unexpected_multiplicity_before_counts(
    tmp_path: Path,
    terminal_failed_count: int,
) -> None:
    lines = LOG.read_text(encoding="utf-8-sig").splitlines()
    duplicate = next(line for line in lines if line.startswith("FAILED "))
    terminal_index = next(
        index
        for index, line in enumerate(lines)
        if VERIFIER._SUMMARY_RE.fullmatch(line) is not None
    )
    lines.insert(terminal_index, duplicate)
    terminal_index += 1
    terminal = lines[terminal_index]
    assert terminal.startswith("78 failed, ")
    if terminal_failed_count == 79:
        lines[terminal_index] = "79 failed, " + terminal.removeprefix("78 failed, ")
    log = tmp_path / "duplicate-outcome.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(
        VERIFIER.PublicCheckoutFailureSetError,
        match="exact pytest failure outcome set drifted",
    ) as captured:
        VERIFIER.verify_expected_failure_set(REGISTRY, JUNIT, log)
    assert "missing=[]" in str(captured.value)
    assert f"unexpected={[duplicate]!r}" in str(captured.value)


@pytest.mark.parametrize("node_id", FROZEN_V8_NODES)
def test_interpreter_marker_is_accepted_for_every_exact_frozen_node(
    tmp_path: Path,
    node_id: str,
) -> None:
    junit = _replace_failure_marker(
        tmp_path,
        node_id=node_id,
        marker=(
            f"{INTERPRETER_MARKER} C:\\hostedtoolcache\\Python\\3.13.12\\python.exe; "
            "observed D:\\a\\MO_NCO\\python.exe"
        ),
    )

    receipt = VERIFIER.verify_expected_failure_set(REGISTRY, junit, LOG)

    assert receipt["status"] == VERIFIER.PASS_STATUS
    signature = next(
        item
        for item in receipt["exact_junit_failure_signatures"]
        if item["node_id"] == node_id
    )
    assert signature == {
        "category": "FROZEN_V8_FAIL_CLOSED",
        "exception_types": ["RuntimeError"],
        "failure_child_count": 1,
        "node_id": node_id,
    }


def test_checked_in_manifest_marker_reference_remains_exact() -> None:
    receipt = VERIFIER.verify_expected_failure_set(
        REGISTRY,
        JUNIT,
        LOG,
        require_reference_sha256=True,
    )

    assert receipt["status"] == VERIFIER.PASS_STATUS
    assert receipt["classification_counts"] == {
        "frozen_v8_fail_closed": 7,
        "held_or_rights_sensitive_dependency": 70,
        "sealed_output": 1,
        "unclassified": 0,
    }
    assert receipt["counts"]["junit"]["failure_children"] == 78
    assert len(receipt["exact_junit_failure_or_error_node_ids"]) == 77
    assert len(receipt["exact_pytest_failure_summary_lines"]) == 78


@pytest.mark.parametrize(
    "marker",
    [
        "Helper may use the exact historical main-job interpreter",
        f"{INTERPRETER_MARKER}-suffix-without-marker-boundary",
        f"prefix {INTERPRETER_MARKER}",
        f"{MANIFEST_MARKER}-suffix-without-marker-boundary",
        f"prefix {MANIFEST_MARKER}",
        "unregistered fail-closed marker",
    ],
)
@pytest.mark.parametrize("node_id", FROZEN_V8_NODES)
def test_arbitrary_or_substring_marker_is_rejected_for_every_frozen_node(
    tmp_path: Path,
    node_id: str,
    marker: str,
) -> None:
    junit = _replace_failure_marker(
        tmp_path,
        node_id=node_id,
        marker=marker,
    )

    with pytest.raises(
        VERIFIER.PublicCheckoutFailureSetError,
        match="frozen-V8 failure marker drifted",
    ):
        VERIFIER.verify_expected_failure_set(REGISTRY, junit, LOG)


def test_allowed_marker_does_not_relax_runtime_error_signature(tmp_path: Path) -> None:
    node_id = FROZEN_V8_NODES[0]
    junit = _replace_failure_marker(
        tmp_path,
        node_id=node_id,
        marker=INTERPRETER_MARKER,
        exception_type="ValueError",
    )
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = next(
        item
        for item in payload["expected_junit_failure_signatures"]
        if item["node_id"] == node_id
    )
    unexpected = {**expected, "exception_types": ["ValueError"]}

    with pytest.raises(
        VERIFIER.PublicCheckoutFailureSetError,
        match="exact JUnit failure exception/category signatures drifted",
    ) as captured:
        VERIFIER.verify_expected_failure_set(REGISTRY, junit, LOG)
    assert f"missing={[expected]!r}" in str(captured.value)
    assert f"unexpected={[unexpected]!r}" in str(captured.value)


def test_junit_signature_multiplicity_is_reported_before_count_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = next(
        item
        for item in payload["expected_junit_failure_signatures"]
        if item["category"] == "HELD_OR_RIGHTS_SENSITIVE_DEPENDENCY"
        and item["failure_child_count"] == 1
    )
    tree = ET.parse(JUNIT)
    suite = next(tree.getroot().iter("testsuite"))
    testcase = next(
        case
        for case in suite.iter("testcase")
        if VERIFIER._pytest_node_id(case) == expected["node_id"]
    )
    extra = ET.SubElement(
        testcase,
        "failure",
        {"message": "ValueError: second failure child"},
    )
    extra.text = "ValueError: second failure child"
    suite.attrib["failures"] = str(int(suite.attrib["failures"]) + 1)
    junit = tmp_path / "signature-multiplicity.junit.xml"
    tree.write(junit, encoding="utf-8", xml_declaration=True)
    unexpected = {
        **expected,
        "exception_types": sorted([*expected["exception_types"], "ValueError"]),
        "failure_child_count": 2,
    }

    with pytest.raises(
        VERIFIER.PublicCheckoutFailureSetError,
        match="exact JUnit failure exception/category signatures drifted",
    ) as captured:
        VERIFIER.verify_expected_failure_set(REGISTRY, junit, LOG)
    assert f"missing={[expected]!r}" in str(captured.value)
    assert f"unexpected={[unexpected]!r}" in str(captured.value)


def test_frozen_contract_keeps_exact_seven_runtime_error_signatures() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    frozen_signatures = {
        item["node_id"]: item
        for item in payload["expected_junit_failure_signatures"]
        if item["category"] == "FROZEN_V8_FAIL_CLOSED"
    }

    assert len(FROZEN_V8_NODES) == 7
    assert set(frozen_signatures) == set(FROZEN_V8_NODES)
    assert {
        tuple(item["exception_types"]) for item in frozen_signatures.values()
    } == {("RuntimeError",)}
    assert {
        item["failure_child_count"] for item in frozen_signatures.values()
    } == {1}


def test_resealed_registry_cannot_expand_the_marker_allowlist(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["frozen_v8_failure_marker_contract"]["default_allowed_markers"].append(
        "unregistered fail-closed marker"
    )
    core = dict(payload)
    del core["manifest_payload_sha256"]
    payload["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    registry = tmp_path / "expanded-registry.json"
    registry.write_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    with pytest.raises(
        VERIFIER.PublicCheckoutFailureSetError,
        match="frozen-V8 failure marker contract drifted",
    ):
        VERIFIER.verify_expected_failure_set(registry, JUNIT, LOG)
