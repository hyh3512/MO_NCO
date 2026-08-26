from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ijoc_submission_v21"
    / "scripts"
    / "build_v21_code_release.py"
)


def _load_release_builder():
    spec = importlib.util.spec_from_file_location(
        "build_v21_code_release", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def test_release_zip_is_deterministic_and_manifest_binds_every_entry(
    tmp_path: Path,
) -> None:
    builder = _load_release_builder()
    repo = tmp_path / "repo"
    _write(repo / "mo_nco" / "pareto_v21_hybrid.py", "VALUE = 21\n")
    _write(
        repo / "ijoc_submission_v21" / "release" / "README.md",
        "V21 experiment-code release boundary.\n",
    )

    first = builder.build_release(
        repo,
        archive_path=tmp_path / "first.zip",
        manifest_path=tmp_path / "first.manifest.json",
        checksum_path=tmp_path / "first.zip.sha256",
        archive_prefix="ijoc_v21_experiment_code",
    )
    second = builder.build_release(
        repo,
        archive_path=tmp_path / "second.zip",
        manifest_path=tmp_path / "second.manifest.json",
        checksum_path=tmp_path / "second.zip.sha256",
        archive_prefix="ijoc_v21_experiment_code",
    )

    first_raw = (tmp_path / "first.zip").read_bytes()
    assert first_raw == (tmp_path / "second.zip").read_bytes()
    assert first["archive"]["sha256"] == hashlib.sha256(first_raw).hexdigest()
    assert second["archive"]["sha256"] == first["archive"]["sha256"]

    manifest = json.loads(
        (tmp_path / "first.manifest.json").read_text(encoding="utf-8")
    )
    expected_paths = [
        "ijoc_submission_v21/release/README.md",
        "mo_nco/pareto_v21_hybrid.py",
    ]
    assert [entry["path"] for entry in manifest["files"]] == expected_paths
    assert manifest["archive"]["sha256"] == first["archive"]["sha256"]
    for entry in manifest["files"]:
        raw = (repo / entry["path"]).read_bytes()
        assert entry["bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()

    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == [
            f"ijoc_v21_experiment_code/{path}" for path in expected_paths
        ]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
        assert all((info.external_attr >> 16) == 0o100644 for info in infos)

    assert (tmp_path / "first.zip.sha256").read_text(encoding="ascii") == (
        f"{first['archive']['sha256']}  first.zip\n"
    )


def test_explicit_allowlist_includes_v3_inputs_and_excludes_large_evidence(
    tmp_path: Path,
) -> None:
    builder = _load_release_builder()
    repo = tmp_path / "repo"
    allowed = {
        "README.md": "project\n",
        "pyproject.toml": "[project]\nname='fixture'\n",
        "requirements-optional.txt": "pytest\n",
        "mo_nco/pareto_v21_trace.py": "TRACE = True\n",
        "tests/test_pareto_v21_hybrid.py": "def test_fixture(): pass\n",
        "ijoc_submission_v21/scripts/run_calibration_matrix.py": "RUN = False\n",
        "ijoc_submission_v21/protocol/CALIBRATION_GATE.md": "NOT_RUN\n",
        "ijoc_submission_v21/protocol/generator_spec_v2.json": "{}\n",
        "ijoc_submission_v21/manuscript/v21_theory_protocol.tex": "draft\n",
        "ijoc_submission_v21/manuscript/v21_theory_protocol.pdf": "pdf fixture\n",
        "ijoc_submission_v21/provenance/partition_audit_v3.json": "{}\n",
        "ijoc_submission_v21/provenance/V20_IMMUTABLE_PARENT.md": "frozen\n",
        "ijoc_submission_v21/release/README.md": "boundary\n",
        "ijoc_submission_v21/development/trace_microbenchmark/receipt.json": "{}\n",
        "ijoc_submission_v21/calibration/epoch_v21e1/selection_gate_receipt.json": "{}\n",
        "ijoc_submission_v21/prospective_partitions_v3/development/case_manifest.json": "{}\n",
        "ijoc_submission_v21/prospective_partitions_v3/development/instances/case.json": "{}\n",
        "ijoc_submission_v21/prospective_partitions_v3/calibration/selection/case_manifest.json": "{}\n",
        "ijoc_submission_v21/prospective_partitions_v3/calibration/selection/instances/case.json": "{}\n",
    }
    excluded = {
        "mo_nco/unrelated.py": "NO\n",
        "tests/test_archive.py": "NO\n",
        "ijoc_submission_v21/manuscript/v21_theory_protocol.aux": "NO\n",
        "ijoc_submission_v21/__pycache__/cached.pyc": "NO\n",
        "ijoc_submission_v21/development/trace.sqlite3": "NO\n",
        "ijoc_submission_v21/development/trace.sqlite3-wal": "NO\n",
        "ijoc_submission_v21/development/trace.sqlite3-shm": "NO\n",
        "ijoc_submission_v21/calibration/epoch_v21e1/selection_runs/run_rows.jsonl": "NO\n",
        "ijoc_submission_v21/calibration/epoch_v21e1/selection_runs/traces/trace.json": "NO\n",
        "ijoc_submission_v21/calibration/epoch_v21e1/confirmation_runs/traces/trace.json": "NO\n",
        "ijoc_submission_v21/prospective_partitions_v1/development/instances/old.json": "NO\n",
        "ijoc_submission_v21/prospective_partitions_v2/development/instances/old.json": "NO\n",
        "ijoc_submission_v21/release/old.zip": "NO\n",
        "ijoc_submission_v21/release/old.manifest.json": "NO\n",
    }
    for relative, value in {**allowed, **excluded}.items():
        _write(repo / relative, value)

    observed = {
        path.relative_to(repo).as_posix()
        for path in builder.select_release_files(repo)
    }

    assert observed == set(allowed)
    assert observed.isdisjoint(excluded)


def test_release_boundary_is_fail_closed_and_outputs_are_immutable(
    tmp_path: Path,
) -> None:
    builder = _load_release_builder()
    repo = tmp_path / "repo"
    _write(repo / "mo_nco" / "pareto_v21_trace.py", "TRACE = True\n")
    _write(
        repo / "ijoc_submission_v21" / "release" / "README.md",
        "boundary\n",
    )
    outputs = {
        "archive_path": tmp_path / "release.zip",
        "manifest_path": tmp_path / "release.manifest.json",
        "checksum_path": tmp_path / "release.zip.sha256",
    }
    builder.build_release(repo, **outputs)
    before = {name: path.read_bytes() for name, path in outputs.items()}

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        builder.build_release(repo, **outputs)

    assert {name: path.read_bytes() for name, path in outputs.items()} == before
    manifest = json.loads(
        outputs["manifest_path"].read_text(encoding="utf-8")
    )
    boundary = manifest["evidence_boundary"]
    assert manifest["artifact_scope"] == (
        "V21_EXPERIMENT_CODE_AND_PROSPECTIVE_INPUTS_ONLY"
    )
    assert boundary["formal_3600_result_package"] == "NOT_INCLUDED"
    assert boundary["large_evaluation_traces"] == "NOT_INCLUDED"
    assert boundary["package_creation_changes_scientific_gates"] is False
    assert boundary["formal_status_without_calibration_pass"] == (
        "NOT_MATERIALIZED"
    )
    assert manifest["rights_status"] == "HOLD_UNLESS_SEPARATELY_CLOSED"


def test_release_readme_states_code_only_and_formal_gate_boundary() -> None:
    readme = (
        SCRIPT_PATH.parents[1] / "release" / "README.md"
    ).read_text(encoding="utf-8")
    lowered = readme.lower()

    assert "experiment-code package" in lowered
    assert "not the formal 3,600-row result package" in lowered
    assert "not_materialized" in lowered
    assert "calibration gate" in lowered
    assert "rights" in lowered and "hold" in lowered
    assert "sqlite3" in lowered
    assert "prospective_partitions_v3" in lowered

