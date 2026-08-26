from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "ijoc_submission_v21e3" / "development_manifests_v1"


def _load(name: str) -> tuple[dict[str, object], bytes]:
    raw = (MANIFEST_ROOT / name).read_bytes()
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload, raw


def test_v21e3_real_development_metric_reference_and_config_manifests() -> None:
    metric, metric_raw = _load("metric_manifest.json")
    reference, reference_raw = _load("reference_manifest_development.json")
    config, config_raw = _load("config_manifest_development.json")
    receipt, _ = _load("build_receipt.json")

    assert metric["formal_use"] == "NOT_AUTHORIZED"
    assert metric["out_of_box_action"] == "FAIL_BEFORE_SCALARIZATION_ARCHIVE_OR_METRIC"
    assert reference["split"] == "development"
    assert reference["formal_use"] == "NOT_AUTHORIZED"
    cases = reference["cases"]
    assert isinstance(cases, list) and len(cases) == 12
    for case in cases:
        assert isinstance(case, dict)
        lower = case["objective_lower_bounds"]
        upper = case["objective_upper_bounds"]
        assert isinstance(lower, list) and isinstance(upper, list)
        assert len(lower) == len(upper) == 2
        assert all(float(left) < float(right) for left, right in zip(lower, upper))

    assert config["candidate_ids"] == ["C0", "C1", "C2", "C3"]
    assert config["selection_partition"] == "NOT_GENERATED"
    assert config["calibration_confirmation_partition"] == "NOT_GENERATED"
    assert config["formal_cases"] == "NOT_MATERIALIZED"
    assert config["calibration_execution_authorized"] is False
    directions = config["reference_directions"]
    assert isinstance(directions, list) and len(directions) == 21
    assert all(abs(sum(direction) - 1.0) <= 1e-12 for direction in directions)

    expected = {
        "metric_manifest.json": metric_raw,
        "reference_manifest_development.json": reference_raw,
        "config_manifest_development.json": config_raw,
    }
    entries = receipt["entries"]
    assert isinstance(entries, list)
    assert {entry["path"] for entry in entries} == set(expected)
    for entry in entries:
        raw = expected[str(entry["path"])]
        assert entry["bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["status"] == "PASS"
    assert receipt["formal_artifacts_created"] is False
    assert receipt["selection_or_confirmation_artifacts_created"] is False


