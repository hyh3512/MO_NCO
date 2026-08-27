from __future__ import annotations

import copy
import base64
import csv
import functools
import hashlib
import importlib.util
import json
import io
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_v9r2r1_github_ci_envelope.py"
SPEC = importlib.util.spec_from_file_location("github_ci_envelope_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
SUBJECT_COMMIT_SHA1 = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
SUBJECT_TREE_SHA1 = subprocess.run(
    ["git", "rev-parse", "HEAD^{tree}"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@functools.lru_cache(maxsize=None)
def _file_sha(relative: str) -> str:
    raw = subprocess.run(
        ["git", "show", f"{SUBJECT_COMMIT_SHA1}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return _sha(raw)


def _job(run_id: int, job_id: int, *, conclusion: str = "success") -> dict:
    names = {
        2001: VERIFIER._EVIDENCE_JOB_NAMES["targeted_regression"],
        2002: VERIFIER._EVIDENCE_JOB_NAMES["public_backend"],
        2003: VERIFIER._EVIDENCE_JOB_NAMES["clean_room_package"],
        2004: VERIFIER._EVIDENCE_JOB_NAMES["public_reference"],
        2005: VERIFIER._EVIDENCE_JOB_NAMES[
            "preserved_internal_exact_eight_reference"
        ],
        2006: VERIFIER._EVIDENCE_JOB_NAMES["public_reference"],
        2007: VERIFIER._EVIDENCE_JOB_NAMES[
            "preserved_internal_exact_eight_reference"
        ],
        2008: VERIFIER._EVIDENCE_JOB_NAMES["public_live_failure_contract"],
        2099: VERIFIER._EVIDENCE_JOB_NAMES["public_live_failure_contract"],
    }
    return {
        "job_id": job_id,
        "name": names[job_id],
        "head_sha1": SUBJECT_COMMIT_SHA1,
        "status": "completed",
        "conclusion": conclusion,
        "html_url": (
            f"https://github.com/hyh3512/MO_NCO/actions/runs/{run_id}/job/"
            f"{job_id}"
        ),
        "steps": [
            {
                "number": 1,
                "name": "complete",
                "status": "completed",
                "conclusion": conclusion,
            }
        ],
    }


def _run(
    role: str,
    run_id: int,
    workflow_id: int,
    artifact_ids: list[int],
    jobs: list[dict],
) -> dict:
    workflow_role, event = VERIFIER._RUNS[role]
    return {
        "workflow_role": workflow_role,
        "workflow_path": VERIFIER._WORKFLOWS[workflow_role],
        "event": event,
        "run_id": run_id,
        "run_attempt": 1,
        "run_number": 4,
        "workflow_id": workflow_id,
        "head_sha1": SUBJECT_COMMIT_SHA1,
        "head_tree_sha1": SUBJECT_TREE_SHA1,
        "head_branch": "main",
        "status": "completed",
        "conclusion": "success",
        "html_url": f"https://github.com/hyh3512/MO_NCO/actions/runs/{run_id}",
        "created_at": "2026-08-27T03:40:00Z",
        "run_started_at": "2026-08-27T03:40:01Z",
        "updated_at": "2026-08-27T03:50:00Z",
        "jobs": jobs,
        "artifact_ids": artifact_ids,
    }


def _artifact(run_role: str, artifact_id: int, name: str) -> dict:
    zip_sha = _sha(f"zip:{artifact_id}".encode())
    members = []
    for path, role in sorted(VERIFIER._ARTIFACT_MEMBERS[name].items()):
        content = f"{name}:{path}".encode()
        members.append(
            {
                "path": path,
                "bytes": len(content),
                "sha256": _sha(content),
                "role": role,
            }
        )
    return {
        "run_role": run_role,
        "artifact_id": artifact_id,
        "name": name,
        "api_size_in_bytes": artifact_id + 100,
        "api_digest": f"sha256:{zip_sha}",
        "expired": False,
        "created_at": "2026-08-27T03:50:01Z",
        "updated_at": "2026-08-27T03:50:01Z",
        "expires_at": "2026-11-25T03:50:01Z",
        "content_verification": VERIFIER.ARTIFACT_CONTENT_SCOPE,
        "downloaded_zip": {
            "path": f"downloads/{run_role}-{artifact_id}.zip",
            "bytes": artifact_id + 100,
            "sha256": zip_sha,
            "api_digest_matches": True,
        },
        "files": members,
    }


def _member_hash(artifact: dict, role: str) -> str:
    matches = [item["sha256"] for item in artifact["files"] if item["role"] == role]
    assert len(matches) == 1
    return matches[0]


def _evidence(
    role: str,
    job_id: int,
    artifact: dict,
    hashes: dict[str, str],
    metrics: dict,
) -> dict:
    return {
        "run_role": VERIFIER._EVIDENCE[role][0],
        "job_id": job_id,
        "artifact_id": artifact["artifact_id"],
        "artifact_name": artifact["name"],
        "artifact_content_verification": VERIFIER.ARTIFACT_CONTENT_SCOPE,
        "reported_status": VERIFIER._EVIDENCE[role][2],
        "reported_inner_sha256": hashes,
        "metrics": metrics,
    }


def _payload() -> dict:
    workflows = {
        role: {"path": path, "sha256": _file_sha(path)}
        for role, path in VERIFIER._WORKFLOWS.items()
    }
    runs = {
        "push_current_source": _run(
            "push_current_source",
            1001,
            501,
            [3001, 3002],
            [_job(1001, 2001), _job(1001, 2002)],
        ),
        "push_clean_room": _run(
            "push_clean_room", 1002, 502, [3003], [_job(1002, 2003)]
        ),
        "push_repository_contract": _run(
            "push_repository_contract",
            1003,
            503,
            [3004, 3005],
            [_job(1003, 2004), _job(1003, 2005), _job(1003, 2099, conclusion="skipped")],
        ),
        "manual_public_live": _run(
            "manual_public_live",
            1004,
            503,
            [3006, 3007, 3008],
            [_job(1004, 2006), _job(1004, 2007), _job(1004, 2008)],
        ),
    }
    artifact_specs = [
        ("push_current_source", 3001, "v9r2r1-targeted-sanitized-evidence"),
        ("push_current_source", 3002, "v9r2r1-public-backend-sanitized-evidence"),
        ("push_clean_room", 3003, "v9r2r1-clean-room-package"),
        (
            "push_repository_contract",
            3004,
            "v9r2r1-public-checkout-reference-contract",
        ),
        (
            "push_repository_contract",
            3005,
            "v9r2r1-internal-complete-tree-exact-eight-reference",
        ),
        ("manual_public_live", 3006, "v9r2r1-public-checkout-reference-contract"),
        (
            "manual_public_live",
            3007,
            "v9r2r1-internal-complete-tree-exact-eight-reference",
        ),
        ("manual_public_live", 3008, "v9r2r1-public-checkout-live-contract"),
    ]
    artifacts = {
        str(artifact_id): _artifact(run_role, artifact_id, name)
        for run_role, artifact_id, name in artifact_specs
    }
    targeted = artifacts["3001"]
    backend = artifacts["3002"]
    package = artifacts["3003"]
    public_reference = artifacts["3004"]
    internal_reference = artifacts["3005"]
    live = artifacts["3008"]
    public_registry = _file_sha(
        "provenance/V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json"
    )
    historical_registry = _file_sha(
        "provenance/V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json"
    )
    historical_junit = _file_sha(
        "evidence/v9r2r1_environment_recovery_20260825_002/full_repository.junit.xml"
    )
    public_metrics = {
        "frozen_v8_fail_closed_outcomes": 7,
        "held_or_rights_sensitive_dependency_outcomes": 70,
        "junit_failure_or_error_testcases": 77,
        "pytest_failed_or_subfailed_outcomes": 78,
        "sealed_output_outcomes": 1,
        "unclassified_outcomes": 0,
    }
    evidence = {
        "targeted_regression": _evidence(
            "targeted_regression",
            2001,
            targeted,
            {
                "junit_sha256": _member_hash(targeted, "targeted_junit"),
                "log_sha256": _member_hash(targeted, "targeted_log"),
                "sanitization_receipt_sha256": _member_hash(
                    targeted, "sanitization_receipt"
                ),
            },
            {
                "errors": 0,
                "failures": 0,
                "passed": 325,
                "skipped": 0,
                "subtests_passed": 0,
                "testcases": 325,
            },
        ),
        "public_backend": _evidence(
            "public_backend",
            2002,
            backend,
            {
                "environment_preflight_sha256": _member_hash(
                    backend, "environment_preflight"
                ),
                "junit_sha256": _member_hash(backend, "backend_junit"),
                "log_sha256": _member_hash(backend, "backend_log"),
                "sanitization_receipt_sha256": _member_hash(
                    backend, "sanitization_receipt"
                ),
            },
            {
                "errors": 0,
                "failures": 0,
                "passed": 4,
                "skipped": 0,
                "subtests_passed": 2,
                "testcases": 4,
            },
        ),
        "clean_room_package": _evidence(
            "clean_room_package",
            2003,
            package,
            {
                "installed_gate_sha256": _member_hash(package, "installed_gate"),
                "normalized_sdist_sha256": _member_hash(package, "normalized_sdist"),
                "wheel_sha256": _member_hash(package, "wheel"),
            },
            {
                "artifact_member_hashes_verified": True,
                "installed_gate_exit_code": 2,
                "installed_gate_status": "PRE_DEVELOPMENT_HOLD",
                "rebuild_identity_reported_by_successful_workflow_not_reexecuted": True,
                "sdist_distribution": "mo-nco",
                "sdist_version": "0.21.3.14",
                "wheel_distribution": "mo-nco",
                "wheel_version": "0.21.3.14",
            },
        ),
        "public_reference": _evidence(
            "public_reference",
            2004,
            public_reference,
            {
                "contract_receipt_sha256": _member_hash(
                    public_reference, "contract_receipt"
                ),
                "failure_registry_sha256": public_registry,
            },
            public_metrics,
        ),
        "preserved_internal_exact_eight_reference": _evidence(
            "preserved_internal_exact_eight_reference",
            2005,
            internal_reference,
            {
                "contract_receipt_sha256": _member_hash(
                    internal_reference, "contract_receipt"
                ),
                "historical_failure_registry_sha256": historical_registry,
                "reference_junit_sha256": historical_junit,
            },
            {"expected_failure_count": 8, "xfail_allowed": False},
        ),
        "public_live_failure_contract": _evidence(
            "public_live_failure_contract",
            2008,
            live,
            {
                "contract_receipt_sha256": _member_hash(live, "contract_receipt"),
                "environment_preflight_sha256": _member_hash(
                    live, "environment_preflight"
                ),
                "failure_registry_sha256": public_registry,
                "junit_sha256": _member_hash(live, "full_repository_junit"),
                "log_sha256": _member_hash(live, "full_repository_log"),
                "output_sanitization_receipt_sha256": _member_hash(
                    live, "output_sanitization_receipt"
                ),
                "preflight_log_sha256": _member_hash(live, "preflight_log"),
                "preflight_sanitization_receipt_sha256": _member_hash(
                    live, "preflight_sanitization_receipt"
                ),
            },
            {
                **public_metrics,
                "junit_passed_testcases": 1329,
                "junit_testcases": 1410,
                "pytest_passed": 1330,
                "pytest_skipped": 4,
                "pytest_subtests_passed": 267,
            },
        ),
    }
    claims = {field: False for field in VERIFIER._CLAIM_FIELDS}
    claims.update(
        {
            "development_study_readiness": "PRE_DEVELOPMENT_HOLD",
            "ijoc_status": "HOLD_NO_SUBMIT",
        }
    )
    payload = {
        "schema": VERIFIER.ENVELOPE_SCHEMA,
        "status": VERIFIER.ENVELOPE_STATUS,
        "identity": {
            "distribution": "mo-nco",
            "revision": "V21E3R1_V9R2R1",
            "version": "0.21.3.14",
        },
        "subject": {
            "repository": "hyh3512/MO_NCO",
            "repository_id": 1347294242,
            "commit_sha1": SUBJECT_COMMIT_SHA1,
            "git_tree_sha1": SUBJECT_TREE_SHA1,
            "branch": "main",
        },
        "container_contract": {
            "mode": "POST_CI_ADDITIVE_EVIDENCE_COMMIT",
            "envelope_path": "provenance/V9R2R1_GITHUB_CI_ENVELOPE.json",
            "expected_container_first_parent_sha1": SUBJECT_COMMIT_SHA1,
            "containing_commit_sha1_embedded": False,
            "subject_commit_claimed_as_container_commit": False,
            "artifact_content_verification": VERIFIER.ARTIFACT_CONTENT_SCOPE,
            "artifact_bytes_downloaded_and_verified": True,
            "allowed_changed_paths": [
                "GITHUB_EXPORT_CONTENTS.json",
                "provenance/V9R2R1_GITHUB_CI_ENVELOPE.json",
            ],
        },
        "workflow_definitions": workflows,
        "runs": runs,
        "artifacts": artifacts,
        "evidence_contracts": evidence,
        "claim_boundary": claims,
    }
    payload["envelope_payload_sha256"] = _sha(VERIFIER._canonical_json(payload))
    return payload


def _baseline_derived_metrics() -> dict[str, dict]:
    payload = _payload()
    return {
        role: copy.deepcopy(record["metrics"])
        for role, record in payload["evidence_contracts"].items()
    }


@pytest.fixture(autouse=True)
def _isolate_outer_contract_tests_from_large_inner_fixtures(monkeypatch):
    """Keep outer attack tests small; dedicated tests cover byte parsers below."""

    baseline = _baseline_derived_metrics()
    monkeypatch.setattr(
        VERIFIER,
        "_derive_and_validate_inner_evidence",
        lambda **_kwargs: copy.deepcopy(baseline),
    )


def _write(path: Path, payload: dict, *, refresh_hash: bool = True) -> None:
    if refresh_hash:
        core = dict(payload)
        core.pop("envelope_payload_sha256", None)
        payload["envelope_payload_sha256"] = _sha(VERIFIER._canonical_json(core))
    path.write_bytes(VERIFIER._canonical_json(payload) + b"\n")


def _materialize_zips(tmp_path: Path, payload: dict) -> Path:
    artifact_root = tmp_path / "artifact-root"
    artifact_root.mkdir(exist_ok=True)
    for artifact in payload["artifacts"].values():
        relative = artifact["downloaded_zip"]["path"]
        destination = artifact_root / Path(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in artifact["files"]:
                info = zipfile.ZipInfo(member["path"], date_time=(2026, 8, 27, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(
                    info,
                    f"{artifact['name']}:{member['path']}".encode(),
                )
        raw = destination.read_bytes()
        digest = _sha(raw)
        artifact["api_size_in_bytes"] = len(raw)
        artifact["api_digest"] = f"sha256:{digest}"
        artifact["downloaded_zip"]["bytes"] = len(raw)
        artifact["downloaded_zip"]["sha256"] = digest
    return artifact_root


def _rebind_outer_zip(artifact_root: Path, artifact: dict) -> None:
    destination = artifact_root / Path(
        *PurePosixPath(artifact["downloaded_zip"]["path"]).parts
    )
    raw = destination.read_bytes()
    digest = _sha(raw)
    artifact["api_size_in_bytes"] = len(raw)
    artifact["api_digest"] = f"sha256:{digest}"
    artifact["downloaded_zip"]["bytes"] = len(raw)
    artifact["downloaded_zip"]["sha256"] = digest


def _verify(
    tmp_path: Path,
    payload: dict,
    *,
    artifact_root: Path | None = None,
) -> dict:
    if artifact_root is None:
        artifact_root = _materialize_zips(tmp_path, payload)
    path = tmp_path / "envelope.json"
    _write(path, payload)
    return VERIFIER.verify_github_ci_envelope(
        path,
        root=ROOT,
        artifact_root=artifact_root,
        expected_commit_sha1=SUBJECT_COMMIT_SHA1,
        expected_tree_sha1=SUBJECT_TREE_SHA1,
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_repo_file(root: Path, relative: str, raw: bytes) -> None:
    path = root.joinpath(*PurePosixPath(relative).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _container_fixture(
    tmp_path: Path, *, forbidden_path: str | None = None
) -> tuple[Path, Path, Path, dict, str, str]:
    repository = tmp_path / "container-repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Envelope Test")
    _git(repository, "config", "user.email", "envelope@example.invalid")
    subject_bytes = {
        ".github/workflows/current-source.yml": b"name: current-source\n",
        ".github/workflows/clean-room-package.yml": b"name: clean-room\n",
        ".github/workflows/full-repository-contract.yml": b"name: repository\n",
        "provenance/V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json": (
            b"subject-public-registry\n"
        ),
        "provenance/V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json": (
            b"subject-historical-registry\n"
        ),
        (
            "evidence/v9r2r1_environment_recovery_20260825_002/"
            "full_repository.junit.xml"
        ): b"subject-historical-junit\n",
        "GITHUB_EXPORT_CONTENTS.json": b"{}\n",
    }
    for relative, raw in subject_bytes.items():
        _write_repo_file(repository, relative, raw)
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "subject")
    subject_commit = _git(repository, "rev-parse", "HEAD")
    subject_tree = _git(repository, "rev-parse", "HEAD^{tree}")

    payload = _payload()
    payload["subject"]["commit_sha1"] = subject_commit
    payload["subject"]["git_tree_sha1"] = subject_tree
    payload["container_contract"]["expected_container_first_parent_sha1"] = (
        subject_commit
    )
    for run in payload["runs"].values():
        run["head_sha1"] = subject_commit
        run["head_tree_sha1"] = subject_tree
        for job in run["jobs"]:
            job["head_sha1"] = subject_commit
    for role, path in VERIFIER._WORKFLOWS.items():
        payload["workflow_definitions"][role]["sha256"] = _sha(subject_bytes[path])
    public_registry_hash = _sha(
        subject_bytes[
            "provenance/V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json"
        ]
    )
    payload["evidence_contracts"]["public_reference"][
        "reported_inner_sha256"
    ]["failure_registry_sha256"] = public_registry_hash
    payload["evidence_contracts"]["public_live_failure_contract"][
        "reported_inner_sha256"
    ]["failure_registry_sha256"] = public_registry_hash
    payload["evidence_contracts"]["preserved_internal_exact_eight_reference"][
        "reported_inner_sha256"
    ]["historical_failure_registry_sha256"] = _sha(
        subject_bytes[
            "provenance/V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json"
        ]
    )
    payload["evidence_contracts"]["preserved_internal_exact_eight_reference"][
        "reported_inner_sha256"
    ]["reference_junit_sha256"] = _sha(
        subject_bytes[
            (
                "evidence/v9r2r1_environment_recovery_20260825_002/"
                "full_repository.junit.xml"
            )
        ]
    )
    artifact_root = _materialize_zips(tmp_path, payload)
    envelope_path = (
        repository / "provenance" / "V9R2R1_GITHUB_CI_ENVELOPE.json"
    )
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    _write(envelope_path, payload)
    _write_repo_file(repository, "GITHUB_EXPORT_CONTENTS.json", b'{"updated":true}\n')
    if forbidden_path is not None:
        _write_repo_file(repository, forbidden_path, b"forbidden mutation\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "container")
    return (
        repository,
        artifact_root,
        envelope_path,
        payload,
        subject_commit,
        subject_tree,
    )


def test_complete_envelope_cross_binds_runs_artifacts_members_and_holds(
    tmp_path: Path,
) -> None:
    result = _verify(tmp_path, _payload())
    assert result["status"].startswith("PASS_VERIFIED_GITHUB_CI_ARTIFACT_ENVELOPE")
    assert result["run_count"] == 4
    assert result["artifact_count"] == 8
    assert result["evidence_contract_count"] == 6
    assert result["artifact_archive_bytes_and_members_verified"] is True
    assert result["repository_wide_green"] is False
    assert result["scientific_stage_authorized"] is False
    assert result["development_study_readiness"] == "PRE_DEVELOPMENT_HOLD"


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="duplicate JSON key"
    ):
        VERIFIER.verify_github_ci_envelope(
            path,
            root=ROOT,
            artifact_root=tmp_path,
            expected_commit_sha1=SUBJECT_COMMIT_SHA1,
            expected_tree_sha1=SUBJECT_TREE_SHA1,
        )


@pytest.mark.parametrize(
    "location",
    ["top", "run", "artifact", "member", "evidence", "claim"],
)
def test_unknown_keys_are_rejected(tmp_path: Path, location: str) -> None:
    payload = _payload()
    targets = {
        "top": payload,
        "run": payload["runs"]["push_current_source"],
        "artifact": payload["artifacts"]["3001"],
        "member": payload["artifacts"]["3001"]["files"][0],
        "evidence": payload["evidence_contracts"]["targeted_regression"],
        "claim": payload["claim_boundary"],
    }
    targets[location]["unexpected"] = False
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="key set"):
        _verify(tmp_path, payload)


def test_placeholder_is_rejected_before_it_can_be_attested(tmp_path: Path) -> None:
    payload = _payload()
    payload["runs"]["push_current_source"]["jobs"][0]["name"] = "TODO"
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="placeholder"):
        _verify(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit_sha1", "1" * 40),
        ("git_tree_sha1", "2" * 40),
        ("branch", "release"),
        ("repository", "someone/else"),
    ],
)
def test_subject_identity_is_exact(
    tmp_path: Path, field: str, value: str
) -> None:
    payload = _payload()
    payload["subject"][field] = value
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="subject"):
        _verify(tmp_path, payload)


def test_workflow_hash_must_match_local_bytes(tmp_path: Path) -> None:
    payload = _payload()
    payload["workflow_definitions"]["current_source"]["sha256"] = "1" * 64
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="workflow hash"):
        _verify(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event", "pull_request"),
        ("conclusion", "failure"),
        ("head_sha1", "1" * 40),
        ("head_tree_sha1", "2" * 40),
        ("run_attempt", 2),
        ("run_id", True),
    ],
)
def test_run_identity_and_success_are_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _payload()
    payload["runs"]["push_current_source"][field] = value
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError):
        _verify(tmp_path, payload)


@pytest.mark.parametrize(
    "mutation",
    ["api_digest", "zip_bytes", "api_match", "member_missing", "member_unsafe"],
)
def test_artifact_zip_and_exact_member_contract_is_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    payload = _payload()
    artifact_root = _materialize_zips(tmp_path, payload)
    artifact = payload["artifacts"]["3001"]
    if mutation == "api_digest":
        artifact["api_digest"] = "sha256:" + "1" * 64
    elif mutation == "zip_bytes":
        artifact["downloaded_zip"]["bytes"] += 1
    elif mutation == "api_match":
        artifact["downloaded_zip"]["api_digest_matches"] = False
    elif mutation == "member_missing":
        artifact["files"].pop()
    else:
        artifact["files"][0]["path"] = "../escape"
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError):
        _verify(tmp_path, payload, artifact_root=artifact_root)


def test_evidence_inner_hash_must_equal_downloaded_member_hash(tmp_path: Path) -> None:
    payload = _payload()
    payload["evidence_contracts"]["targeted_regression"][
        "reported_inner_sha256"
    ]["junit_sha256"] = "1" * 64
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="evidence/member hash"
    ):
        _verify(tmp_path, payload)


def test_downloaded_zip_member_bytes_are_recomputed(tmp_path: Path) -> None:
    payload = _payload()
    artifact_root = _materialize_zips(tmp_path, payload)
    artifact = payload["artifacts"]["3001"]
    destination = artifact_root / Path(
        *PurePosixPath(artifact["downloaded_zip"]["path"]).parts
    )
    with zipfile.ZipFile(destination, "w") as archive:
        for index, member in enumerate(artifact["files"]):
            raw = (
                b"tampered"
                if index == 0
                else f"{artifact['name']}:{member['path']}".encode()
            )
            archive.writestr(member["path"], raw)
    _rebind_outer_zip(artifact_root, artifact)
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="member bytes/hash"
    ):
        _verify(tmp_path, payload, artifact_root=artifact_root)


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "dir\\escape"])
def test_downloaded_zip_rejects_unsafe_member_names(
    tmp_path: Path, unsafe_name: str
) -> None:
    payload = _payload()
    artifact_root = _materialize_zips(tmp_path, payload)
    artifact = payload["artifacts"]["3001"]
    destination = artifact_root / Path(
        *PurePosixPath(artifact["downloaded_zip"]["path"]).parts
    )
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr(unsafe_name, b"unsafe")
    _rebind_outer_zip(artifact_root, artifact)
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError):
        _verify(tmp_path, payload, artifact_root=artifact_root)


def test_downloaded_zip_rejects_duplicate_member_names(tmp_path: Path) -> None:
    payload = _payload()
    artifact_root = _materialize_zips(tmp_path, payload)
    artifact = payload["artifacts"]["3001"]
    destination = artifact_root / Path(
        *PurePosixPath(artifact["downloaded_zip"]["path"]).parts
    )
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("duplicate.txt", b"one")
            archive.writestr("duplicate.txt", b"two")
    _rebind_outer_zip(artifact_root, artifact)
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="duplicate ZIP member"
    ):
        _verify(tmp_path, payload, artifact_root=artifact_root)


@pytest.mark.parametrize(
    "manual_artifact_id",
    ["3006", "3007"],
    ids=["public-reference-duplicate", "exact-eight-duplicate"],
)
def test_manual_duplicate_artifact_must_equal_push_member_bytes(
    tmp_path: Path, manual_artifact_id: str
) -> None:
    payload = _payload()
    artifact_root = _materialize_zips(tmp_path, payload)
    artifact = payload["artifacts"][manual_artifact_id]
    target_member = artifact["files"][0]
    tampered_raw = b"fabricated manual duplicate with internally consistent hashes"
    target_member["bytes"] = len(tampered_raw)
    target_member["sha256"] = _sha(tampered_raw)
    destination = artifact_root / Path(
        *PurePosixPath(artifact["downloaded_zip"]["path"]).parts
    )
    with zipfile.ZipFile(destination, "w") as archive:
        for member in artifact["files"]:
            raw = (
                tampered_raw
                if member["path"] == target_member["path"]
                else f"{artifact['name']}:{member['path']}".encode()
            )
            archive.writestr(member["path"], raw)
    _rebind_outer_zip(artifact_root, artifact)
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError,
        match="manual duplicate artifact member bytes drifted",
    ):
        _verify(tmp_path, payload, artifact_root=artifact_root)


def test_public_registry_hash_is_bound_to_checked_in_bytes(tmp_path: Path) -> None:
    payload = _payload()
    payload["evidence_contracts"]["public_reference"][
        "reported_inner_sha256"
    ]["failure_registry_sha256"] = "1" * 64
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="public failure registry"
    ):
        _verify(tmp_path, payload)


@pytest.mark.parametrize("field", sorted(VERIFIER._CLAIM_FIELDS))
def test_no_scientific_or_publication_boundary_can_be_opened(
    tmp_path: Path, field: str
) -> None:
    payload = _payload()
    payload["claim_boundary"][field] = True
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="boundary drifted"
    ):
        _verify(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("development_study_readiness", "READY"),
        ("ijoc_status", "SUBMIT"),
    ],
)
def test_hold_labels_are_exact(tmp_path: Path, field: str, value: str) -> None:
    payload = _payload()
    payload["claim_boundary"][field] = value
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="HOLD"):
        _verify(tmp_path, payload)


def test_payload_self_hash_is_mandatory(tmp_path: Path) -> None:
    payload = _payload()
    artifact_root = _materialize_zips(tmp_path, payload)
    path = tmp_path / "envelope.json"
    payload["envelope_payload_sha256"] = "1" * 64
    _write(path, payload, refresh_hash=False)
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="payload hash"):
        VERIFIER.verify_github_ci_envelope(
            path,
            root=ROOT,
            artifact_root=artifact_root,
            expected_commit_sha1=SUBJECT_COMMIT_SHA1,
            expected_tree_sha1=SUBJECT_TREE_SHA1,
        )


def test_noncanonical_json_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(VERIFIER.GitHubCIEnvelopeVerificationError, match="canonical"):
        VERIFIER.verify_github_ci_envelope(
            path,
            root=ROOT,
            artifact_root=tmp_path,
            expected_commit_sha1=SUBJECT_COMMIT_SHA1,
            expected_tree_sha1=SUBJECT_TREE_SHA1,
        )


def test_fabricated_self_reported_metrics_are_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["evidence_contracts"]["targeted_regression"]["metrics"][
        "passed"
    ] = 999
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="differ from derived bytes"
    ):
        _verify(tmp_path, payload)


def test_green_junit_and_log_counts_are_independently_derived() -> None:
    junit = (
        b'<testsuites tests="2" failures="0" errors="0" skipped="0">'
        b'<testsuite tests="2"><testcase classname="a" name="one"/>'
        b'<testcase classname="a" name="two"/></testsuite></testsuites>'
    )
    assert VERIFIER._junit_green_metrics(junit, b"2 passed in 0.01s\n") == {
        "errors": 0,
        "failures": 0,
        "passed": 2,
        "skipped": 0,
        "subtests_passed": 0,
        "testcases": 2,
    }
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="counts disagree"
    ):
        VERIFIER._junit_green_metrics(junit, b"3 passed in 0.01s\n")


def test_live_output_uses_generic_sanitizer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "V9R2R1_RAW_OUTPUT_SANITIZATION_RECEIPT.json"
    junit = tmp_path / "full_repository.sanitized.junit.xml"
    log = tmp_path / "full_repository.sanitized.log"
    generic_source = tmp_path / "sanitize_public_ci_artifacts.py"
    engine_source = tmp_path / "sanitize_public_checkout_outputs.py"
    observed: dict[str, object] = {}

    def fake_verify(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"verified": True}

    monkeypatch.setattr(VERIFIER, "_verify_generic_sanitization_bundle", fake_verify)
    module = object()
    result = VERIFIER._verify_live_output_sanitization_bundle(
        module=module,
        member_paths={
            receipt.name: receipt,
            junit.name: junit,
            log.name: log,
        },
        subject_commit_sha1="1" * 40,
        subject_tree_sha1="2" * 40,
        generic_source_path=generic_source,
        engine_source_path=engine_source,
    )
    assert result == {"verified": True}
    assert observed == {
        "module": module,
        "receipt_path": receipt,
        "outputs": {
            "full_repository.junit.xml": junit,
            "full_repository.log": log,
        },
        "kinds": {
            "full_repository.junit.xml": "PYTEST_JUNIT_XML",
            "full_repository.log": "PYTEST_LOG",
        },
        "subject_commit_sha1": "1" * 40,
        "subject_tree_sha1": "2" * 40,
        "generic_source_path": generic_source,
        "engine_source_path": engine_source,
    }


def test_historical_interpreter_rule_is_fixed() -> None:
    valid = type(
        "ValidGenericSanitizer",
        (),
        {
            "RULE_IDS": (
                "historical_interpreter",
                "repository_root",
                "temp_root",
                "user_home",
                "environment_prefix",
                "username",
                "host_name",
            ),
            "HISTORICAL_INTERPRETER": (
                r"C:\miniconda3\envs\ssm_env\python.exe"
            ),
            "HISTORICAL_INTERPRETER_REPLACEMENT": (
                "__HISTORICAL_INTERPRETER__"
            ),
            "REPLACEMENTS": {
                "historical_interpreter": "__HISTORICAL_INTERPRETER__",
                "repository_root": "__REPO_ROOT__",
                "temp_root": "__TEMP_ROOT__",
                "user_home": "__USER_HOME__",
                "environment_prefix": "__PYTHON_PREFIX__",
                "username": "__USERNAME__",
                "host_name": "__HOSTNAME__",
            },
        },
    )()
    VERIFIER._require_historical_interpreter_rule(valid)

    for rule_ids, replacements in (
        (
            ("repository_root", "historical_interpreter"),
            {
                "historical_interpreter": "__HISTORICAL_INTERPRETER__",
                "repository_root": "__REPO_ROOT__",
            },
        ),
        (
            (
                "historical_interpreter",
                "repository_root",
                "temp_root",
                "user_home",
                "environment_prefix",
                "username",
                "host_name",
            ),
            {
                "historical_interpreter": "__WRONG__",
                "repository_root": "__REPO_ROOT__",
                "temp_root": "__TEMP_ROOT__",
                "user_home": "__USER_HOME__",
                "environment_prefix": "__PYTHON_PREFIX__",
                "username": "__USERNAME__",
                "host_name": "__HOSTNAME__",
            },
        ),
    ):
        invalid = type(
            "InvalidGenericSanitizer",
            (),
            {
                "RULE_IDS": rule_ids,
                "HISTORICAL_INTERPRETER": (
                    r"C:\miniconda3\envs\ssm_env\python.exe"
                ),
                "HISTORICAL_INTERPRETER_REPLACEMENT": (
                    "__HISTORICAL_INTERPRETER__"
                ),
                "REPLACEMENTS": replacements,
            },
        )()
        with pytest.raises(
            VERIFIER.GitHubCIEnvelopeVerificationError,
            match="historical-interpreter rule drifted",
        ):
            VERIFIER._require_historical_interpreter_rule(invalid)


def test_historical_interpreter_receipt_counts_are_bound_to_output_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.log"
    output.write_bytes(
        b"runtime=__HISTORICAL_INTERPRETER__\n"
        b"runtime=__HISTORICAL_INTERPRETER__\n"
    )
    receipt = {
        "replacement_contract": {
            "replacement_order": ["historical_interpreter"],
            "rules": [
                {
                    "id": "historical_interpreter",
                    "replacement": "__HISTORICAL_INTERPRETER__",
                    "match_counts": {"run.log": 2},
                }
            ],
        }
    }
    VERIFIER._require_historical_interpreter_receipt(
        receipt, outputs={"run.log": output}
    )
    receipt["replacement_contract"]["rules"][0]["match_counts"]["run.log"] = 1
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError,
        match="output/count binding drifted",
    ):
        VERIFIER._require_historical_interpreter_receipt(
            receipt, outputs={"run.log": output}
        )


def test_installed_gate_semantics_cannot_be_fabricated() -> None:
    raw = subprocess.run(
        ["git", "show", f"{SUBJECT_COMMIT_SHA1}:evidence/gate/installed_gate_receipt.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    VERIFIER._validate_installed_gate(raw)
    payload = json.loads(raw)
    payload["status"] = "READY"
    core = dict(payload)
    del core["receipt_payload_sha256"]
    payload["receipt_payload_sha256"] = _sha(VERIFIER._canonical_json(core))
    tampered = VERIFIER._canonical_json(payload) + b"\n"
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="gate semantics"
    ):
        VERIFIER._validate_installed_gate(tampered)


def test_wheel_and_sdist_identity_is_parsed_from_content() -> None:
    wheel_members = {
        "mo_nco/__init__.py": b"__version__ = '0.21.3.14'\n",
        "mo_nco-0.21.3.14.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: mo-nco\nVersion: 0.21.3.14\n"
        ),
        "mo_nco-0.21.3.14.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nTag: py3-none-any\n"
        ),
    }
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name, raw in wheel_members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
        writer.writerow([name, f"sha256={digest.decode('ascii')}", str(len(raw))])
    record_name = "mo_nco-0.21.3.14.dist-info/RECORD"
    writer.writerow([record_name, "", ""])
    wheel_members[record_name] = record_buffer.getvalue().encode()
    wheel_buffer = io.BytesIO()
    with zipfile.ZipFile(wheel_buffer, "w") as archive:
        for name, raw in wheel_members.items():
            archive.writestr(name, raw)
    assert VERIFIER._validate_wheel(wheel_buffer.getvalue()) == (
        "mo-nco",
        "0.21.3.14",
    )

    sdist_buffer = io.BytesIO()
    with tarfile.open(fileobj=sdist_buffer, mode="w:gz") as archive:
        for name, raw in (
            (
                "mo_nco-0.21.3.14/PKG-INFO",
                b"Metadata-Version: 2.1\nName: mo-nco\nVersion: 0.21.3.14\n",
            ),
            ("mo_nco-0.21.3.14/pyproject.toml", b"[build-system]\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mtime = 1700000000
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(raw))
    assert VERIFIER._validate_sdist(sdist_buffer.getvalue()) == (
        "mo-nco",
        "0.21.3.14",
    )


def test_container_single_parent_and_exact_allowlist_can_pass(tmp_path: Path) -> None:
    repository, artifact_root, envelope_path, _payload_value, subject, tree = (
        _container_fixture(tmp_path)
    )
    result = VERIFIER.verify_github_ci_envelope(
        envelope_path,
        root=repository,
        artifact_root=artifact_root,
        expected_commit_sha1=subject,
        expected_tree_sha1=tree,
        container_ref="HEAD",
    )
    assert result["container_commit_binding_verified"] is True


def test_subject_bindings_ignore_mutable_container_worktree_bytes(
    tmp_path: Path,
) -> None:
    repository, artifact_root, envelope_path, _payload_value, subject, tree = (
        _container_fixture(tmp_path)
    )
    for relative in (
        ".github/workflows/current-source.yml",
        "provenance/V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json",
        (
            "evidence/v9r2r1_environment_recovery_20260825_002/"
            "full_repository.junit.xml"
        ),
    ):
        _write_repo_file(repository, relative, b"uncommitted mutable drift\n")
    result = VERIFIER.verify_github_ci_envelope(
        envelope_path,
        root=repository,
        artifact_root=artifact_root,
        expected_commit_sha1=subject,
        expected_tree_sha1=tree,
        container_ref="HEAD",
    )
    assert result["container_commit_binding_verified"] is True


@pytest.mark.parametrize(
    "forbidden_path",
    [
        ".github/workflows/current-source.yml",
        "scripts/verify_expected_public_checkout_failure_set.py",
        "provenance/V9R2R1_EXPECTED_PUBLIC_CHECKOUT_FAILURE_SET.json",
        "tests/test_forbidden_container_mutation.py",
    ],
)
def test_container_rejects_any_nonmechanical_mutation(
    tmp_path: Path, forbidden_path: str
) -> None:
    repository, artifact_root, envelope_path, _payload_value, subject, tree = (
        _container_fixture(tmp_path, forbidden_path=forbidden_path)
    )
    with pytest.raises(
        VERIFIER.GitHubCIEnvelopeVerificationError, match="exact additive evidence"
    ):
        VERIFIER.verify_github_ci_envelope(
            envelope_path,
            root=repository,
            artifact_root=artifact_root,
            expected_commit_sha1=subject,
            expected_tree_sha1=tree,
            container_ref="HEAD",
        )
