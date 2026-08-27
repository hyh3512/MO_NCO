from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;]+)")


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(specifier: str) -> str:
    return _normalized(re.split(r"[<>=!~; ]", specifier, maxsplit=1)[0])


def _requirements(relative_path: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw_line in (ROOT / relative_path).read_text(encoding="utf-8").splitlines():
        match = REQUIREMENT.match(raw_line.strip())
        if match:
            rows[_normalized(match.group(1))] = match.group(2)
    return rows


def _top_level_external_test_imports() -> set[str]:
    local_roots = {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    external: set[str] = set()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        module = ast.parse(path.read_bytes(), filename=str(path))
        for node in module.body:
            if isinstance(node, ast.Import):
                names = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.partition(".")[0]]
            else:
                continue
            external.update(
                name
                for name in names
                if name not in sys.stdlib_module_names and name not in local_roots
            )
    return external


def test_frozen_package_extras_and_public_locks_cover_collection_imports() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    extras = project["optional-dependencies"]
    test_names = {_requirement_name(item) for item in extras["test"]}
    receipt_names = {_requirement_name(item) for item in extras["receipt"]}

    assert _top_level_external_test_imports() == {"cryptography", "pytest"}
    assert "pytest" in test_names
    assert "cryptography" in receipt_names
    assert {"cryptography", "pytest"} <= test_names | receipt_names


def test_version_locks_cover_current_and_full_test_import_closures() -> None:
    base_input = _requirements("requirements/base.in")
    base_lock = _requirements("requirements/base.lock")
    full_input = _requirements("requirements/optional-pymoo.in")
    full_lock = _requirements("requirements/optional-pymoo.lock")

    assert base_input["cryptography"] == "46.0.5"
    assert full_input["cryptography"] == "46.0.5"
    for locked in (base_lock, full_lock):
        assert locked["cryptography"] == "46.0.5"
        assert locked["cffi"] == "2.0.0"
        assert locked["pycparser"] == "2.23"
        assert "pytest" in locked
    assert full_lock["pymoo"] == "0.6.1.6"
    assert full_lock["moocore"] == "0.3.1"


def test_requirement_integrity_manifest_and_environment_spec_match_bytes() -> None:
    checksums: dict[str, str] = {}
    for line in (ROOT / "requirements/locks.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative_path = line.split("  ", maxsplit=1)
        checksums[relative_path] = digest

    spec = json.loads(
        (ROOT / "provenance/V9R2R1_ENVIRONMENT_SPEC.json").read_text(
            encoding="utf-8"
        )
    )
    spec_checksums = {
        f"requirements/{name}": digest
        for name, digest in spec["requirements"].items()
    }
    assert checksums == spec_checksums
    for relative_path, expected in checksums.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected


def test_public_live_workflow_materializes_observed_interpreter_identities() -> None:
    workflow = (
        ROOT / ".github/workflows/full-repository-contract.yml"
    ).read_text(encoding="utf-8")
    live = workflow.split(
        "  public-checkout-live-contract:\n", maxsplit=1
    )[1]

    assert "actions/setup-python" not in live
    for exact in (
        r"C:\Miniconda\Scripts\conda.exe",
        r"C:\miniconda3\python.exe",
        r"C:\miniconda3\envs\ssm_env\python.exe",
        "python=3.13.12=h39c999c_100_cp313",
        "python=3.11.15=h1044e36_0",
        "f77193cf0405ab440c39324bdb2f8864596321c1df888adbbe357f3d760f4716",
        "418228fb1417da15512fc44aba6f2e1d948786878ffd06fd661881c6d104c0f6",
        "V9R2R1_LIVE_PYTHON=C:\\miniconda3\\python.exe",
        "--basetemp=$baseTemp",
        "artifact-hashed environment lock remains false",
    ):
        assert exact in live
