from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_expected_historical_failure_set.py"
REGISTRY = ROOT / "provenance" / "V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json"
JUNIT = (
    ROOT
    / "evidence"
    / "v9r2r1_environment_recovery_20260825_002"
    / "full_repository.junit.xml"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("historical_failure_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_module()


def _write_tree(tree: ET.ElementTree, path: Path) -> None:
    tree.write(path, encoding="utf-8", xml_declaration=True)


def test_reference_junit_matches_exact_eight_node_allowlist() -> None:
    receipt = VERIFIER.verify_expected_failure_set(
        REGISTRY, JUNIT, require_reference_sha256=True
    )
    assert receipt["status"] == "PASS_EXACT_EXPECTED_HISTORICAL_V8_FAILURE_SET"
    assert receipt["counts"] == {
        "testcases": 1356,
        "passed": 1344,
        "failures": 8,
        "errors": 0,
        "skipped": 4,
    }
    assert len(receipt["exact_failure_node_ids"]) == 8
    assert receipt["repository_wide_green"] is False
    assert receipt["scientific_stage_authorized"] is False


def test_same_message_on_unregistered_node_does_not_pass(
    tmp_path: Path,
) -> None:
    tree = ET.parse(JUNIT)
    cases = list(tree.getroot().iter("testcase"))
    registered_case = next(case for case in cases if case.find("failure") is not None)
    registered_case.remove(registered_case.find("failure"))
    unregistered_case = next(
        case
        for case in cases
        if case.find("failure") is None
        and case.find("error") is None
        and case.find("skipped") is None
    )
    failure = ET.SubElement(
        unregistered_case,
        "failure",
        {"message": "RuntimeError: Frozen diagnostic source manifest drifted"},
    )
    failure.text = "Frozen diagnostic source manifest drifted"
    drifted = tmp_path / "swapped_failure.junit.xml"
    _write_tree(tree, drifted)
    with pytest.raises(
        VERIFIER.HistoricalFailureSetError,
        match="exact failure node-id set drifted",
    ):
        VERIFIER.verify_expected_failure_set(REGISTRY, drifted)


def test_registered_node_without_exact_marker_fails_closed(tmp_path: Path) -> None:
    tree = ET.parse(JUNIT)
    failure = next(tree.getroot().iter("failure"))
    failure.attrib["message"] = "RuntimeError: unrelated failure"
    failure.text = "unrelated failure"
    drifted = tmp_path / "wrong_marker.junit.xml"
    _write_tree(tree, drifted)
    with pytest.raises(
        VERIFIER.HistoricalFailureSetError,
        match="lack exact frozen-manifest marker",
    ):
        VERIFIER.verify_expected_failure_set(REGISTRY, drifted)


def test_xfail_outcome_is_never_accepted_as_registered_history(
    tmp_path: Path,
) -> None:
    tree = ET.parse(JUNIT)
    skipped = next(tree.getroot().iter("skipped"))
    skipped.attrib["message"] = "pytest.xfail was requested"
    drifted = tmp_path / "xfail.junit.xml"
    _write_tree(tree, drifted)
    with pytest.raises(
        VERIFIER.HistoricalFailureSetError,
        match="xfail outcome is prohibited",
    ):
        VERIFIER.verify_expected_failure_set(REGISTRY, drifted)
