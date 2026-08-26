from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_v9r2r1_current_source.py"
MANIFEST = ROOT / "V21E3R1_V9R2R1_SOURCE_MANIFEST.json"


def test_checked_out_manifest_is_complete_and_git_tracked() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(ROOT),
            "--manifest",
            str(MANIFEST),
            "--require-git-tracked",
            "--require-pass",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_CURRENT_SOURCE_MANIFEST"
    assert payload["declared_file_count"] == 203
    assert payload["missing"] == []
    assert payload["sha256_mismatches"] == []
    assert payload["untracked"] == []
    assert payload["scientific_stage_authorized"] is False


def test_missing_root_fails_closed_without_authorizing_science(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--manifest",
            str(MANIFEST),
            "--require-pass",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert payload["scientific_stage_authorized"] is False
