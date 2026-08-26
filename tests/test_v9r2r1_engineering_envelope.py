from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAILURE_VERIFIER = _load(
    "historical_failure_verifier_for_envelope",
    "scripts/verify_expected_historical_failure_set.py",
)
BUILDER = _load(
    "v9r2r1_engineering_envelope_builder",
    "scripts/build_v9r2r1_engineering_envelope.py",
)
ENVELOPE_VERIFIER = _load(
    "v9r2r1_engineering_envelope_verifier",
    "scripts/verify_v9r2r1_engineering_envelope.py",
)


SOURCE_FILES = (
    "pyproject.toml",
    "V21E3R1_V9R2R1_SOURCE_MANIFEST.json",
    "provenance/V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json",
    (
        "evidence/v9r2r1_environment_recovery_20260825_002/"
        "full_suite_environment_preflight.json"
    ),
    (
        "evidence/v9r2r1_environment_recovery_20260825_002/"
        "pymoo_environment_recovery.junit.xml"
    ),
    (
        "evidence/v9r2r1_environment_recovery_20260825_002/"
        "targeted_final.junit.xml"
    ),
    (
        "evidence/v9r2r1_environment_recovery_20260825_002/"
        "full_repository.junit.xml"
    ),
)


def _copy_fixture_root(tmp_path: Path) -> Path:
    copied = tmp_path / "project"
    copied.mkdir()
    for relative in SOURCE_FILES:
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return copied


def _paths(root: Path) -> dict[str, Path]:
    evidence = root / "evidence" / "v9r2r1_environment_recovery_20260825_002"
    return {
        "source_manifest": root / "V21E3R1_V9R2R1_SOURCE_MANIFEST.json",
        "environment_preflight": evidence / "full_suite_environment_preflight.json",
        "pymoo_junit": evidence / "pymoo_environment_recovery.junit.xml",
        "targeted_junit": evidence / "targeted_final.junit.xml",
        "full_repository_junit": evidence / "full_repository.junit.xml",
        "expected_failure_registry": (
            root / "provenance" / "V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json"
        ),
    }


def _build(root: Path) -> tuple[dict[str, object], Path, Path]:
    paths = _paths(root)
    receipt = FAILURE_VERIFIER.verify_expected_failure_set(
        paths["expected_failure_registry"],
        paths["full_repository_junit"],
        require_reference_sha256=True,
    )
    receipt_path = root / "generated" / "expected_failure.receipt.json"
    receipt_path.parent.mkdir()
    FAILURE_VERIFIER._write_exclusive(receipt_path, receipt)
    envelope = BUILDER.build_engineering_envelope(
        root=root,
        expected_failure_receipt=receipt_path,
        **paths,
    )
    envelope_path = root / "generated" / "engineering_envelope.json"
    BUILDER._write_exclusive(envelope_path, envelope)
    return envelope, envelope_path, receipt_path


def test_envelope_cross_binds_all_inputs_and_preserves_hold_boundaries(
    tmp_path: Path,
) -> None:
    root = _copy_fixture_root(tmp_path)
    envelope, envelope_path, _receipt_path = _build(root)
    verification = ENVELOPE_VERIFIER.verify_engineering_envelope(
        envelope_path, root=root
    )
    assert envelope["status"] == "PASS_SCOPED_ENGINEERING_RECOVERY_ENVELOPE_ONLY"
    assert envelope["repository_wide_green"] is False
    assert envelope["environment_lock_satisfied"] is False
    assert envelope["scientific_stage_authorized"] is False
    assert envelope["expected_historical_v8_failure_contract"][
        "exact_node_ids"
    ]
    assert verification["status"] == (
        "PASS_VERIFIED_SCOPED_ENGINEERING_RECOVERY_ENVELOPE_ONLY"
    )
    assert verification["artifact_count"] == 7


def test_verifier_rejects_any_bound_artifact_byte_drift(tmp_path: Path) -> None:
    root = _copy_fixture_root(tmp_path)
    _envelope, envelope_path, _receipt_path = _build(root)
    targeted = _paths(root)["targeted_junit"]
    targeted.write_bytes(targeted.read_bytes() + b"\n")
    with pytest.raises(
        ENVELOPE_VERIFIER.EngineeringEnvelopeVerificationError,
        match="artifact bytes/hash drifted: targeted_junit",
    ):
        ENVELOPE_VERIFIER.verify_engineering_envelope(envelope_path, root=root)


def test_builder_rejects_full_junit_not_bound_by_failure_receipt(
    tmp_path: Path,
) -> None:
    root = _copy_fixture_root(tmp_path)
    paths = _paths(root)
    receipt = FAILURE_VERIFIER.verify_expected_failure_set(
        paths["expected_failure_registry"],
        paths["full_repository_junit"],
        require_reference_sha256=True,
    )
    receipt_path = root / "generated" / "expected_failure.receipt.json"
    receipt_path.parent.mkdir()
    FAILURE_VERIFIER._write_exclusive(receipt_path, receipt)
    alternate = root / "generated" / "alternate_full.junit.xml"
    alternate.write_bytes(paths["full_repository_junit"].read_bytes() + b"\n")
    paths["full_repository_junit"] = alternate
    with pytest.raises(
        BUILDER.EngineeringEnvelopeError,
        match="failure receipt full JUnit hash drifted",
    ):
        BUILDER.build_engineering_envelope(
            root=root,
            expected_failure_receipt=receipt_path,
            **paths,
        )
