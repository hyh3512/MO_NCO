from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TARGET_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "run_v21e3r1_target_structural.py"
)
PREFLIGHT_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "preflight_v21e3r1_development_parity.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_receipt_uses_repo_relative_and_row_relative_posix_paths(
    tmp_path: Path,
) -> None:
    target = _load(TARGET_SCRIPT, "v21e3r1_target_path_contract")
    preflight = _load(PREFLIGHT_SCRIPT, "v21e3r1_preflight_path_contract")
    repo = tmp_path / "repo"
    row = repo / "evidence" / "rows" / "row-1"
    row.mkdir(parents=True)
    trace = row / "trace.sqlite3"
    trace.write_bytes(b"trace")

    assert target._repo_relative(trace, repo_root=repo) == (
        "evidence/rows/row-1/trace.sqlite3"
    )
    assert preflight._repo_artifact_path(
        repo,
        "evidence/rows/row-1/trace.sqlite3",
        label="fixture",
    ) == trace.resolve()
    assert preflight._row_artifact_path(
        row, "trace.sqlite3", label="fixture"
    ) == trace.resolve()


@pytest.mark.parametrize(
    "bad",
    ("../trace.sqlite3", "rows\\trace.sqlite3", "/trace.sqlite3", "C:/trace.sqlite3"),
)
def test_target_preflight_rejects_absolute_or_escaping_paths(
    tmp_path: Path, bad: str,
) -> None:
    preflight = _load(PREFLIGHT_SCRIPT, "v21e3r1_preflight_bad_path_contract")
    with pytest.raises(ValueError, match="canonical relative POSIX"):
        preflight._repo_artifact_path(tmp_path, bad, label="fixture")

