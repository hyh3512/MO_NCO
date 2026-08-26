from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    REPOSITORY_ROOT
    / "ijoc_submission_v21e3r1"
    / "baselines"
    / "v7_reference_comparator_registry.json"
)
VERIFIER_PATH = (
    REPOSITORY_ROOT
    / "ijoc_submission_v21e3r1"
    / "scripts"
    / "verify_v21e3r1_reference_comparator_registry.py"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("v7_reference_registry_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _payload_digest(registry: dict) -> str:
    payload = copy.deepcopy(registry)
    payload.pop("registry_payload_sha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(canonical).hexdigest()


def _write_resigned(tmp_path: Path, registry: dict, name: str = "registry.json") -> Path:
    registry["registry_payload_sha256"] = _payload_digest(registry)
    target = tmp_path / name
    target.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return target


def _comparator(registry: dict, comparator_id: str) -> dict:
    return next(
        item for item in registry["comparators"] if item["comparator_id"] == comparator_id
    )


def test_real_registry_passes_strict_offline_verification() -> None:
    receipt = VERIFIER.verify_registry(REGISTRY_PATH, REPOSITORY_ROOT)

    assert receipt["status"] == "PASS_STRICT_OFFLINE_DEVELOPMENT_REFERENCE_FREEZE_ONLY"
    assert receipt["artifact_count"] == 30
    assert receipt["comparator_count"] == 9
    assert receipt["development_reference_eligible_count"] == 8
    assert receipt["external_family_native_strong_baseline_count"] == 0
    assert receipt["formal_primary_eligible_count"] == 0
    assert receipt["selection_execution"] == "NOT_AUTHORIZED"
    assert receipt["confirmation_execution"] == "NOT_AUTHORIZED"
    assert receipt["formal_materialization"] == "PROHIBITED"
    assert receipt["network_calls"] == 0
    assert receipt["ijoc_status"] == "IJOC_HOLD"


def test_registry_uses_honest_reference_classifications_and_budget_boundaries() -> None:
    registry = _registry()
    by_id = {item["comparator_id"]: item for item in registry["comparators"]}

    assert by_id["motsp-pymoo-nsga2-0.6.2-adapted-v1"]["classification"] == (
        "external_library_adapted_reference"
    )
    assert by_id["motsp-pymoo-moead-0.6.2-adapted-v1"]["license"]["expression"] == (
        "Apache-2.0"
    )
    assert by_id["motsp-lkh3-scalar-3.0.14-v1"]["classification"] == (
        "external_native_single_objective_solver_adapted_reference"
    )
    assert by_id["motsp-lkh3-scalar-3.0.14-v1"]["budget_semantics"][
        "matched_first_true_objective_evaluation_budget"
    ] is False
    assert by_id["motsp-lkh3-seeded-project-2opt-pls-v1"]["classification"] == (
        "project_hybrid_reference"
    )
    assert by_id["motsp-paquete-published-tpls-archive-v1"]["classification"] == (
        "published_result_archive_reference"
    )
    assert by_id["motsp-paquete-published-tpls-archive-v1"]["parameters"][
        "algorithm_execution"
    ] == "none"
    assert by_id["mokp-pls-native-v1"]["classification"] == "project_native_reference"
    assert by_id["platemo-mokp-candidate-v4.14-era"]["classification"] == (
        "candidate_external_platform_not_integrated"
    )
    assert by_id["platemo-mokp-candidate-v4.14-era"]["eligibility"][
        "development_reference_eligible"
    ] is False
    assert all(
        item["eligibility"]["external_family_native_strong_baseline"] is False
        for item in registry["comparators"]
    )


def test_resigned_formal_authorization_attempt_fails_closed(tmp_path: Path) -> None:
    registry = _registry()
    registry["gates"]["formal_primary_eligible_count"] = 1
    registry["gates"]["formal_primary_gate"] = "PASS"
    registry["gates"]["formal_materialization"] = "AUTHORIZED"
    registry["gates"]["ijoc_status"] = "IJOC_READY"
    registry["status"] = "FORMAL_READY"
    _comparator(registry, "motsp-pymoo-nsga2-0.6.2-adapted-v1")["eligibility"][
        "formal_primary_eligible"
    ] = True
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="frozen value|hard-coded"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_resigned_strong_external_claim_attempt_fails_closed(tmp_path: Path) -> None:
    registry = _registry()
    registry["gates"]["external_family_native_strong_baseline_count"] = 1
    comparator = _comparator(registry, "motsp-pymoo-moead-0.6.2-adapted-v1")
    comparator["classification"] = "external_family_native_strong_baseline"
    comparator["eligibility"]["external_family_native_strong_baseline"] = True
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="hard-coded"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_resigned_path_traversal_is_rejected(tmp_path: Path) -> None:
    registry = _registry()
    registry["artifacts"][0]["path"] = "../outside.py"
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="canonical relative path"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_resigned_artifact_digest_drift_is_rejected(tmp_path: Path) -> None:
    registry = _registry()
    registry["artifacts"][0]["sha256"] = "0" * 64
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="independently frozen value"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_bool_fields_reject_integer_substitutes_even_when_resigned(tmp_path: Path) -> None:
    registry = _registry()
    _comparator(registry, "mokp-pls-native-v1")["eligibility"][
        "formal_primary_eligible"
    ] = 0
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="exact type bool"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_resigned_network_authorization_is_rejected(tmp_path: Path) -> None:
    registry = _registry()
    _comparator(registry, "motsp-paquete-published-tpls-archive-v1")["invocation"][
        "network_access"
    ] = True
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="network access"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_resigned_command_or_budget_drift_is_rejected_by_comparator_seal(
    tmp_path: Path,
) -> None:
    registry = _registry()
    comparator = _comparator(registry, "motsp-lkh3-scalar-3.0.14-v1")
    comparator["invocation"]["environment"]["MO_NCO_OFFICIAL_LKH_RUNS"] = "2"
    comparator["budget_semantics"]["internal_solver_work_accounting"] = "claimed_charged"
    mutated = _write_resigned(tmp_path, registry)

    with pytest.raises(VERIFIER.RegistryVerificationError, match="comparator payload differs"):
        VERIFIER.verify_registry(mutated, REPOSITORY_ROOT)


def test_payload_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    registry = _registry()
    _comparator(registry, "mokp-pls-native-v1")["scientific_boundary"] += " tampered"
    target = tmp_path / "unresigned.json"
    target.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(VERIFIER.RegistryVerificationError, match="payload SHA-256"):
        VERIFIER.verify_registry(target, REPOSITORY_ROOT)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    raw = REGISTRY_PATH.read_text(encoding="utf-8")
    raw = raw.replace(
        '  "schema": "ijoc_v21e3r1_v7_reference_comparator_registry_v1",',
        '  "schema": "duplicate",\n'
        '  "schema": "ijoc_v21e3r1_v7_reference_comparator_registry_v1",',
        1,
    )
    assert '"schema": "duplicate"' in raw
    target = tmp_path / "duplicate.json"
    target.write_text(raw, encoding="utf-8")

    with pytest.raises(VERIFIER.RegistryVerificationError, match="duplicate JSON key"):
        VERIFIER.verify_registry(target, REPOSITORY_ROOT)


def test_verifier_has_no_network_or_process_execution_imports() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(
        {"ftplib", "http", "requests", "socket", "subprocess", "urllib"}
    )

