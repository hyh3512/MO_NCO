from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "issue_v21e3r1_v3_invalidation.py"
)
DYNAMIC_PATHS = (
    "ijoc_submission_v21e3/scripts/build_v21e3_code_release.py",
    "ijoc_submission_v21e3/scripts/verify_v21e3_clean_room.py",
)
V3_PREFIX = "ijoc_v21e3r1_experiment_code"
PARENT_PREFIX = "ijoc_v21e3_experiment_code"
TRIGGERING_TEST_PATH = "tests/test_pareto_v21e3_release.py"


def _issuer():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_v3_invalidation_issuer", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _synthetic_v3_fixture(
    tmp_path: Path, issuer
) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    repo = tmp_path / "synthetic-repo"
    repo.mkdir()
    live_dynamic = {
        DYNAMIC_PATHS[0]: b"# synthetic parent-identical builder\n",
        DYNAMIC_PATHS[1]: b"# synthetic hardened live verifier\n",
    }
    parent_dynamic = {
        DYNAMIC_PATHS[0]: live_dynamic[DYNAMIC_PATHS[0]],
        DYNAMIC_PATHS[1]: b"# synthetic older parent verifier\n",
    }
    for relative, raw in live_dynamic.items():
        _write(repo / relative, raw)

    triggering_test = (
        b"from importlib.util import spec_from_file_location\n"
        b"BUILDER = 'build_v21e3_code_release.py'\n"
        b"VERIFIER = 'verify_v21e3_clean_room.py'\n"
    )
    bound_files = [
        {
            "bytes": len(triggering_test),
            "path": TRIGGERING_TEST_PATH,
            "sha256": _sha256(triggering_test),
        }
    ]
    source_root = _sha256(_canonical(bound_files))
    snapshot = {
        "bound_file_count": len(bound_files),
        "bound_files": bound_files,
        "bound_files_root_sha256": source_root,
        "formal_authorized": False,
        "schema": "pareto_v21e3r1_development_source_snapshot_freeze_v1",
        "status": "PASS_ENGINEERING_SNAPSHOT_ONLY",
        "submission_status": "IJOC_HOLD",
    }
    snapshot_raw = _canonical(snapshot)
    authorization = {
        "formal_authorized": False,
        "schema": "pareto_v21e3r1_development_parity_authorization_v1",
        "source_snapshot_receipt_sha256": _sha256(snapshot_raw),
        "source_snapshot_root_sha256": source_root,
        "status": "AUTHORIZED_DEVELOPMENT_PARITY_ONLY",
        "submission_status": "IJOC_HOLD",
    }
    authorization_raw = _canonical(authorization)
    authorization_sha256 = _sha256(authorization_raw)

    aggregate = {
        "authorization_receipt_sha256": authorization_sha256,
        "expected_rows": 1,
        "formal_authorized": False,
        "formal_execution": "PROHIBITED",
        "observed_rows": 1,
        "rows": [{"row_id": "synthetic-row"}],
        "schema": "pareto_v21e3r1_development_matched_matrix_aggregate_v1",
        "source_snapshot_root_sha256": source_root,
        "status": "COMPLETE_DEVELOPMENT_MATRIX_ENGINEERING_EVIDENCE",
    }
    aggregate_raw = _canonical(aggregate)
    runner = {
        "authorization_receipt_sha256": authorization_sha256,
        "expected_rows": 1,
        "matrix_aggregate_sha256": _sha256(aggregate_raw),
        "observed_rows": 1,
        "schema": "pareto_v21e3r1_development_matrix_post_run_audit_v1",
        "source_snapshot_root_sha256": source_root,
        "status": "PASS_COMPLETE_DEVELOPMENT_MATRIX_AUDITED",
    }
    runner_raw = _canonical(runner)
    independent = {
        "authorization_receipt_sha256": authorization_sha256,
        "matrix_aggregate_sha256": _sha256(aggregate_raw),
        "objective_archive_and_metric_replayed_rows": 1,
        "runner_post_run_audit_sha256": _sha256(runner_raw),
        "schema": (
            "pareto_v21e3r1_independent_development_matrix_post_run_audit_v1"
        ),
        "source_snapshot_root_sha256": source_root,
        "status": "PASS_INDEPENDENT_POST_PROCESS_RECOMPUTATION",
        "submission_status": "IJOC_HOLD",
    }
    independent_raw = _canonical(independent)

    release_archive_raw_file = repo / "inputs" / "v3-release.zip"
    release_archive_raw_file.parent.mkdir(parents=True)
    with zipfile.ZipFile(release_archive_raw_file, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{V3_PREFIX}/{TRIGGERING_TEST_PATH}", triggering_test)
    release_archive_raw = release_archive_raw_file.read_bytes()
    release_manifest = {
        "archive": {
            "bytes": len(release_archive_raw),
            "filename": "ijoc_v21e3r1_experiment_code.zip",
            "sha256": _sha256(release_archive_raw),
        },
        "archive_prefix": V3_PREFIX,
        "file_count": 1,
        "files": [
            {
                "archive_path": TRIGGERING_TEST_PATH,
                "bytes": len(triggering_test),
                "sha256": _sha256(triggering_test),
                "source_path": TRIGGERING_TEST_PATH,
            }
        ],
        "frozen_source_provenance": {
            "authorization_receipt_sha256": authorization_sha256,
            "source_snapshot_receipt_sha256": _sha256(snapshot_raw),
            "source_snapshot_root_sha256": source_root,
        },
        "schema": "ijoc_v21e3r1_standalone_release_manifest_v1",
    }
    release_manifest_raw = _canonical(release_manifest)
    release_checksum_raw = (
        f"{_sha256(release_archive_raw)}  ijoc_v21e3r1_experiment_code.zip\n"
    ).encode("ascii")

    step05_lines: list[str] = []
    for relative in DYNAMIC_PATHS:
        windows_relative = relative.replace("/", "\\")
        traceback_relative = windows_relative.replace("\\", "\\\\")
        step05_lines.extend(
            [
                f"path = 'C:\\\\fixture\\\\{traceback_relative}'",
                f"E   FileNotFoundError: C:\\\\fixture\\\\{traceback_relative}",
            ]
        )
    step05_lines.append("7 failed, 197 passed in 0.01s")
    step05_raw = ("\n".join(step05_lines) + "\n").encode("utf-8")
    clean_stderr_raw = (
        "RuntimeError: Clean-room step '05_run_extracted_tree_v21_tests' failed\n"
        "FileNotFoundError: verify_v21e3_clean_room.py\n"
    ).encode("utf-8")
    clean_receipt = {
        "archive_verification": {
            "archive_sha256": _sha256(release_archive_raw),
            "manifest_sha256": _sha256(release_manifest_raw),
        },
        "error": (
            "Clean-room step '05_run_extracted_tree_v21_tests' failed: "
            "FileNotFoundError verify_v21e3_clean_room.py"
        ),
        "formal_authorized": False,
        "pinned_inputs": {
            "archive_sha256": _sha256(release_archive_raw),
            "manifest_sha256": _sha256(release_manifest_raw),
        },
        "schema": "ijoc_v21e3r1_clean_room_gate_receipt_v2",
        "status": "FAIL",
    }
    clean_receipt_raw = _canonical(clean_receipt)

    parent_archive_file = repo / "inputs" / "parent.zip"
    with zipfile.ZipFile(parent_archive_file, "w", zipfile.ZIP_STORED) as archive:
        for relative, raw in parent_dynamic.items():
            archive.writestr(f"{PARENT_PREFIX}/{relative}", raw)
    parent_archive_raw = parent_archive_file.read_bytes()

    artifact_raw = {
        "v3_source_snapshot": snapshot_raw,
        "v3_development_authorization": authorization_raw,
        "v3_matrix_aggregate": aggregate_raw,
        "v3_matrix_raw_aggregate": aggregate_raw,
        "v3_runner_post_run_audit": runner_raw,
        "v3_matrix_raw_runner_post_run_audit": runner_raw,
        "v3_independent_post_run_audit": independent_raw,
        "v3_matrix_stdout_log": runner_raw,
        "v3_matrix_stderr_log": b"",
        "v3_independent_audit_stdout_log": independent_raw,
        "v3_independent_audit_stderr_log": b"",
        "v3_clean_room_stdout_log": b"",
        "v3_clean_room_stderr_log": clean_stderr_raw,
        "v3_release_archive": release_archive_raw,
        "v3_release_manifest": release_manifest_raw,
        "v3_release_checksum": release_checksum_raw,
        "v3_clean_room_fail_receipt": clean_receipt_raw,
        "v3_clean_room_step05_log": step05_raw,
        "v21e3_immutable_parent_archive": parent_archive_raw,
    }
    artifact_paths: dict[str, str] = {}
    for index, (name, raw) in enumerate(artifact_raw.items()):
        relative = f"fixture-evidence/{index:02d}-{name}.bin"
        artifact_paths[name] = relative
        _write(repo / relative, raw)
    contract: dict[str, object] = {
        "artifact_paths": artifact_paths,
        "artifact_sha256": {
            name: _sha256(raw) for name, raw in artifact_raw.items()
        },
        "authorization_sha256": authorization_sha256,
        "dynamic_paths": list(DYNAMIC_PATHS),
        "dynamic_sha256": {
            path: {
                "immutable_parent": _sha256(parent_dynamic[path]),
                "live": _sha256(live_dynamic[path]),
            }
            for path in DYNAMIC_PATHS
        },
        "matrix_row_count": 1,
        "parent_archive_prefix": PARENT_PREFIX,
        "release_file_count": 1,
        "snapshot_bound_file_count": 1,
        "source_root_sha256": source_root,
        "triggering_test_bytes": len(triggering_test),
        "triggering_test_path": TRIGGERING_TEST_PATH,
        "triggering_test_sha256": _sha256(triggering_test),
        "v3_archive_prefix": V3_PREFIX,
    }
    return repo, contract, artifact_raw


def test_synthetic_v3_invalidation_is_independently_bound_and_exclusive(
    tmp_path: Path,
) -> None:
    issuer = _issuer()
    repo, contract, artifact_raw = _synthetic_v3_fixture(tmp_path, issuer)
    output = tmp_path / "V21E3R1_V3_INVALIDATION.json"

    receipt = issuer.issue_v3_invalidation(
        repo_root=repo,
        output=output,
        verification_contract=contract,
    )

    raw = output.read_bytes()
    assert raw == _canonical(receipt)
    assert json.loads(raw) == receipt
    assert receipt["schema"] == "pareto_v21e3r1_v3_invalidation_receipt_v1"
    assert receipt["status"] == (
        "INVALIDATED_POST_EXECUTION_UNBOUND_DYNAMIC_TEST_DEPENDENCY_"
        "AND_FAILED_CLEAN_ROOM_P0"
    )
    assert receipt["v3_value_status"] == (
        "NON_AUTHORITATIVE_DEVELOPMENT_DIAGNOSTIC"
    )
    assert receipt["v3_reuse_for_v4"] == "PROHIBITED"
    assert receipt["formal_authorized"] is False
    assert receipt["submission_status"] == "IJOC_HOLD"
    assert receipt["v4_supersession"] == {"status": "PENDING_NOT_BOUND"}

    evidence = receipt["evidence_artifacts"]
    for name, expected_raw in artifact_raw.items():
        assert evidence[name]["sha256"] == _sha256(expected_raw)
        assert evidence[name]["bytes"] == len(expected_raw)
    dynamic = {
        item["path"]: item
        for item in receipt["dynamic_test_dependency_audit"][
            "required_dynamic_paths"
        ]
    }
    assert set(dynamic) == set(DYNAMIC_PATHS)
    assert all(item["absent_from_v3_snapshot"] for item in dynamic.values())
    assert all(
        item["absent_from_v3_release_archive"] for item in dynamic.values()
    )
    assert dynamic[DYNAMIC_PATHS[0]]["live_vs_immutable_parent"] == "IDENTICAL"
    assert dynamic[DYNAMIC_PATHS[1]]["live_vs_immutable_parent"] == "DIFFERENT"
    assert receipt["dynamic_test_dependency_audit"][
        "parent_verifier_differs_from_live"
    ] is True

    clean_room = receipt["clean_room_failure_audit"]
    assert clean_room["receipt_status"] == "FAIL"
    assert clean_room["failed_step"] == "05_run_extracted_tree_v21_tests"
    assert clean_room["pytest_summary"] == "7 failed, 197 passed"
    assert set(clean_room["missing_dynamic_paths_observed"]) == set(DYNAMIC_PATHS)
    assert receipt["matrix_chain_audit"]["observed_rows"] == 1
    assert receipt["matrix_chain_audit"]["status"] == (
        "BOUND_BUT_INVALIDATED_WITH_V3"
    )

    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        issuer.issue_v3_invalidation(
            repo_root=repo,
            output=output,
            verification_contract=contract,
        )
    assert output.read_bytes() == original

    defaults = issuer.default_verification_contract()
    assert defaults["source_root_sha256"] == (
        "f8aa5cd1f57b51654d303fa4e9a2996c717249799869087508b92cd1aae10114"
    )


def test_optional_v4_successor_is_hash_bound(tmp_path: Path) -> None:
    issuer = _issuer()
    repo, contract, _ = _synthetic_v3_fixture(tmp_path, issuer)
    v4_snapshot = tmp_path / "V4_snapshot.json"
    v4_snapshot_payload = {
        "bound_files_root_sha256": "a" * 64,
        "formal_authorized": False,
        "schema": "pareto_v21e3r1_development_source_snapshot_freeze_v1",
        "status": "PASS_ENGINEERING_SNAPSHOT_ONLY",
        "submission_status": "IJOC_HOLD",
    }
    v4_snapshot.write_bytes(_canonical(v4_snapshot_payload))
    v4_authorization = tmp_path / "V4_authorization.json"
    v4_authorization_payload = {
        "formal_authorized": False,
        "schema": "pareto_v21e3r1_development_parity_authorization_v1",
        "source_snapshot_receipt_sha256": _sha256(v4_snapshot.read_bytes()),
        "source_snapshot_root_sha256": "a" * 64,
        "status": "AUTHORIZED_DEVELOPMENT_PARITY_ONLY",
        "submission_status": "IJOC_HOLD",
    }
    v4_authorization.write_bytes(_canonical(v4_authorization_payload))

    output = tmp_path / "bound-invalidation.json"
    receipt = issuer.issue_v3_invalidation(
        repo_root=repo,
        output=output,
        v4_snapshot_path=v4_snapshot,
        v4_authorization_path=v4_authorization,
        verification_contract=contract,
    )

    supersession = receipt["v4_supersession"]
    assert supersession["status"] == "BOUND_AUTHORIZED_SUCCESSOR"
    assert supersession["source_snapshot"]["sha256"] == _sha256(
        v4_snapshot.read_bytes()
    )
    assert supersession["development_authorization"]["sha256"] == _sha256(
        v4_authorization.read_bytes()
    )
    assert supersession["source_snapshot_root_sha256"] == "a" * 64


def test_optional_v4_successor_requires_snapshot_and_authorization_pair(
    tmp_path: Path,
) -> None:
    issuer = _issuer()
    repo = tmp_path / "synthetic-repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="provided together"):
        issuer.issue_v3_invalidation(
            repo_root=repo,
            output=tmp_path / "must-not-exist.json",
            v4_snapshot_path=tmp_path / "snapshot-only.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()

