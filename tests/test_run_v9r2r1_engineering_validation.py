from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v9r2r1_engineering_validation.ps1"


def test_powershell_orchestrator_parses_without_errors() -> None:
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPT}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { "
        "[Console]::Error.WriteLine($_.Message) }; exit 1 }"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_orchestrator_keeps_registered_failure_and_science_boundaries_explicit() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "V9R2R1_EXPECTED_HISTORICAL_V8_FAILURE_SET.json" in text
    assert "verify_expected_historical_failure_set.py" in text
    assert "build_v9r2r1_engineering_envelope.py" in text
    assert "verify_v9r2r1_engineering_envelope.py" in text
    assert "repository_wide_green = $false" in text
    assert "scientific_stage_authorized = $false" in text
    assert "full_development_matrix_authorized = $false" in text
    assert "xfail" not in text.casefold()
