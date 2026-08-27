from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v9r2r1_full_suite_test_manifest.py"
MANIFEST = ROOT / "provenance" / "V9R2R1_FULL_SUITE_TEST_MANIFEST.json"


def test_recorded_full_suite_test_sources_are_hash_closed() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--root", str(ROOT), "--manifest", str(MANIFEST)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS_FULL_SUITE_TEST_SOURCE_CLOSURE"
    assert payload["test_module_count"] == 139
    assert payload["repository_wide_green"] is False
    assert payload["scientific_stage_authorized"] is False

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = [entry["path"] for entry in manifest["files"]]
    observed = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_*.py")
    )
    assert manifest["test_module_selection"] == "ALL_CURRENT_TESTS_TEST_STAR_PY"
    assert declared == observed
