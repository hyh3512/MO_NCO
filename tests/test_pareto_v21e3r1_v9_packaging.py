from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_fresh(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_package_and_project_versions_are_v9r2r1_identity() -> None:
    from mo_nco import __version__

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == "0.21.3.14"
    assert project["project"]["version"] == __version__


def test_plain_package_import_does_not_preload_executable_v9_modules() -> None:
    code = r"""
import json
import sys
import mo_nco
print(json.dumps({
    "gate": "mo_nco.pareto_v21e3r1_v9_gate" in sys.modules,
    "diagnostics": "mo_nco.pareto_v21e3r1_v9_diagnostics" in sys.modules,
    "protocol": "mo_nco.pareto_v21e3r1_v9_protocol" in sys.modules,
}, sort_keys=True))
"""
    result = _run_fresh("-c", code)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "diagnostics": False,
        "gate": False,
        "protocol": False,
    }


def test_lazy_root_exports_preserve_public_import_compatibility() -> None:
    code = r"""
import json
import sys
import mo_nco
before = sorted(name for name in sys.modules if name.startswith("mo_nco.pareto_v21e3r1_v9_"))
from mo_nco import (
    V9PredevelopmentProtocolError,
    analyze_v9_trace_database,
    evaluate_v9_predevelopment_readiness,
    load_v9_predevelopment_protocol,
    validate_v9_predevelopment_protocol,
    validate_v9_resource_caps,
    write_v9_predevelopment_readiness_receipt,
)
values = [
    V9PredevelopmentProtocolError,
    analyze_v9_trace_database,
    evaluate_v9_predevelopment_readiness,
    load_v9_predevelopment_protocol,
    validate_v9_predevelopment_protocol,
    validate_v9_resource_caps,
    write_v9_predevelopment_readiness_receipt,
]
after = sorted(name for name in sys.modules if name.startswith("mo_nco.pareto_v21e3r1_v9_"))
print(json.dumps({
    "before": before,
    "after": after,
    "all_resolved": all(isinstance(value, type) or callable(value) for value in values),
    "cached": mo_nco.analyze_v9_trace_database is analyze_v9_trace_database,
}, sort_keys=True))
"""
    result = _run_fresh("-c", code)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["before"] == ["mo_nco.pareto_v21e3r1_v9_theory"]
    assert observed["all_resolved"] is True
    assert observed["cached"] is True
    assert "mo_nco.pareto_v21e3r1_v9_diagnostics" in observed["after"]
    assert "mo_nco.pareto_v21e3r1_v9_gate" in observed["after"]
    assert "mo_nco.pareto_v21e3r1_v9_protocol" in observed["after"]


@pytest.mark.parametrize(
    "module",
    [
        "mo_nco.pareto_v21e3r1_v9_gate",
        "mo_nco.pareto_v21e3r1_v9_diagnostics",
    ],
)
def test_python_m_cli_help_is_runtimewarning_clean(module: str) -> None:
    result = _run_fresh("-W", "error::RuntimeWarning", "-m", module, "--help")
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr


def test_unknown_package_attribute_still_raises_attribute_error() -> None:
    import mo_nco

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(mo_nco, "definitely_not_a_public_symbol")

