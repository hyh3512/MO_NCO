from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "freeze_v21e3r1_development_snapshot.py"
)
PREFLIGHT_SCRIPT = (
    ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "preflight_v21e3r1_development_parity.py"
)
PROTOCOL = (
    ROOT
    / "ijoc_submission_v21e3"
    / "protocol"
    / "V21E3_C0_PARITY_PROTOCOL_V2.json"
)


def _freeze_module():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_snapshot_dynamic_dependencies", FREEZE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight_module():
    spec = importlib.util.spec_from_file_location(
        "v21e3r1_preflight_dynamic_dependencies", PREFLIGHT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prospective_snapshot_binds_v21e3_scripts_loaded_by_tests() -> None:
    freeze = _freeze_module()

    snapshot = freeze.compute_prospective_source_root(
        repo_root=ROOT,
        protocol_path=PROTOCOL,
    )
    bound_paths = {
        entry["path"] for entry in snapshot["prospective_source_files"]
    }

    assert {
        "ijoc_submission_v21e3/scripts/build_v21e3_code_release.py",
        "ijoc_submission_v21e3/scripts/verify_v21e3_clean_room.py",
    } <= bound_paths


def test_independent_preflight_reconstructs_the_same_prospective_path_set() -> None:
    freeze = _freeze_module()
    preflight = _preflight_module()
    prospective = freeze.compute_prospective_source_root(
        repo_root=ROOT,
        protocol_path=PROTOCOL,
    )
    snapshot_stub = {"protocol_path": PROTOCOL.relative_to(ROOT).as_posix()}

    producer_paths = [
        entry["path"] for entry in prospective["prospective_source_files"]
    ]
    verifier_paths = preflight._expected_prospective_paths(ROOT, snapshot_stub)

    assert verifier_paths == producer_paths

