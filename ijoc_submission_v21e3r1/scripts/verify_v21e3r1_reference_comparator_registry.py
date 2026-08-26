"""Strict offline verifier for the V21e3r1 V7 reference comparator freeze.

This module deliberately performs no downloads and starts no comparator.  It
only verifies the fail-closed development-reference registry and the local
bytes to which that registry is bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn


SCHEMA = "ijoc_v21e3r1_v7_reference_comparator_registry_v1"
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    DEFAULT_REPOSITORY_ROOT
    / "ijoc_submission_v21e3r1"
    / "baselines"
    / "v7_reference_comparator_registry.json"
)

TOP_LEVEL_KEYS = {
    "schema",
    "primary_source_cutoff_date",
    "scope",
    "status",
    "network_policy",
    "gates",
    "artifacts",
    "comparators",
    "registry_payload_sha256",
}
GATE_KEYS = {
    "registry_entry_count",
    "development_reference_eligible_count",
    "external_family_native_strong_baseline_count",
    "formal_primary_eligible_count",
    "development_reference_freeze",
    "selection_execution",
    "confirmation_execution",
    "formal_primary_gate",
    "formal_materialization",
    "ijoc_status",
}
ARTIFACT_KEYS = {"artifact_id", "role", "path", "bytes", "sha256"}
COMPARATOR_KEYS = {
    "comparator_id",
    "problem_family",
    "classification",
    "claim_label",
    "source_identity",
    "license",
    "artifact_ids",
    "invocation",
    "parameters",
    "budget_semantics",
    "eligibility",
    "scientific_boundary",
}
SOURCE_IDENTITY_KEYS = {
    "upstream_name",
    "version",
    "release_date",
    "tag",
    "commit",
    "tree",
    "source_url",
    "binary_url",
    "cutoff_observation",
}
LICENSE_KEYS = {"expression", "evidence_artifact_id", "use_scope", "redistribution"}
INVOCATION_KEYS = {
    "kind",
    "argv",
    "python_api",
    "cwd",
    "environment",
    "network_access",
    "offline_cache_required",
}
BUDGET_KEYS = {
    "charged_unit",
    "initialization_accounting",
    "internal_solver_work_accounting",
    "matched_first_true_objective_evaluation_budget",
    "checkpoint_semantics",
}
ELIGIBILITY_KEYS = {
    "registry_entry_frozen",
    "development_reference_eligible",
    "selection_execution_authorized",
    "confirmation_execution_authorized",
    "formal_primary_eligible",
    "external_family_native_strong_baseline",
    "public_redistribution_authorized",
}

EXPECTED_GATES = {
    "registry_entry_count": 9,
    "development_reference_eligible_count": 8,
    "external_family_native_strong_baseline_count": 0,
    "formal_primary_eligible_count": 0,
    "development_reference_freeze": "PASS_ENGINEERING_IDENTITY_AND_SCOPE_ONLY",
    "selection_execution": "NOT_AUTHORIZED",
    "confirmation_execution": "NOT_AUTHORIZED",
    "formal_primary_gate": "FAIL_CLOSED_NO_EXTERNAL_FAMILY_NATIVE_STRONG_BASELINE",
    "formal_materialization": "PROHIBITED",
    "ijoc_status": "IJOC_HOLD",
}

# artifact_id -> (role, repository-relative POSIX path, bytes, sha256)
EXPECTED_ARTIFACTS = {
    "core.archive": (
        "transitive_project_source",
        "mo_nco/archive.py",
        13851,
        "edb715f7a02ff1934407d94d139d7f8d05bb8cde716a6539c260a04dfa94b07e",
    ),
    "core.evaluation": (
        "transitive_project_source",
        "mo_nco/evaluation.py",
        3536,
        "64bc0ec71930a12dbabcd706c7248fab00c2a2a1676ec8b7bbb79cfc8a230f7c",
    ),
    "core.instance": (
        "transitive_project_source",
        "mo_nco/instance.py",
        11863,
        "47b920592fd662cdff103df87f2b0efee6b9e3ee25ee42377a22ab0d49536ec4",
    ),
    "core.moves": (
        "transitive_project_source",
        "mo_nco/moves.py",
        2932,
        "bb5cba006da0a7e32b9691fe841dd1c5af489d9d77cc8a07c9559d3784151bf3",
    ),
    "core.pareto_ijoc_problem": (
        "transitive_project_source",
        "mo_nco/pareto_ijoc_problem.py",
        11801,
        "181299505f700689c6a3732b324de2b85e43b7b54b445e6f474c0599e77677f0",
    ),
    "core.pareto_smc_spec": (
        "transitive_project_source",
        "mo_nco/pareto_smc_spec.py",
        16642,
        "fa42d91101e86f7e529496699821e0b526b76c83afdc93fb43124646669ef5db",
    ),
    "core.potential": (
        "transitive_project_source",
        "mo_nco/potential.py",
        12519,
        "d9b63bb3cb8355ac0409fcaff41e2522b2a39b31277609bc30d92c63006ca765",
    ),
    "core.sampler": (
        "transitive_project_source",
        "mo_nco/sampler.py",
        15029,
        "94652ed837587465602aa806c39ffc634a5bb30ade3d1ea9db7d38890830ce2b",
    ),
    "core.types": (
        "transitive_project_source",
        "mo_nco/types.py",
        121,
        "07fe81bd9ab02a0165cdc5706c49a461057777f8cd9bed7340c45a9a7930cf1d",
    ),
    "lkh.binary": (
        "official_windows_executable",
        "external/LKH-3.0.14/LKH-3.exe",
        477184,
        "b44414b7c9aa111b4e782ba13fd2cf4684bf3ad505ba1f5d65822b8ad162c044",
    ),
    "lkh.local_readme": (
        "local_version_and_rights_evidence",
        "external/LKH-3.0.14/README.txt",
        6025,
        "8a21f486c3e12c97719ba85e99091c4b4ca80dbafc0cb377817052224ef829e7",
    ),
    "lkh.local_source_archive": (
        "local_february_2026_source_archive",
        "external/LKH-3.0.14.tgz",
        2321636,
        "2b08b78ac86e60c091d8bc7d1967e271766aecc16a5551fc5482a586dac37800",
    ),
    "mokp.adapter": (
        "project_native_reference_implementation",
        "mo_nco/ijoc_mokp_baselines.py",
        20539,
        "d87fe406ff89d98a96a73c808ef5691c10ec2a3664dc2ca45da3d6b4a28d5e09",
    ),
    "paquete.adapter": (
        "published_archive_reader_not_solver",
        "mo_nco/external_paquete_published_tpls_baseline.py",
        5624,
        "2071318f428a1273da3bf0f1a36b9cd60c492408c461f17c6b35d11b1cb08d9f",
    ),
    "paquete.result_archive": (
        "published_kroab100_result_archive",
        "benchmarks/paquete_published_tpls/TPLS__KROAB100__points.100.AB.a2000.3.first.ils.tgz",
        1031302,
        "4a6380d2d5f3b7c8b104d38506b5588e773d848778274f1cb94e1d39a319371a",
    ),
    "paquete.suite": (
        "supported_case_scope",
        "benchmarks/suite_paquete_kro_tpls.json",
        3423,
        "48f329c96093adba3231a41caeecb011a502d01d498e9cb4cda433b4a03ca5aa",
    ),
    "platemo.git_head": (
        "local_git_head_symbolic_binding",
        "external/PlatEMO/.git/HEAD",
        23,
        "f6f2b945f6c411b02ba3da9c7ace88dcf71b6af65ba2e0d89aa82900042b5a10",
    ),
    "platemo.git_master_ref": (
        "local_git_commit_binding",
        "external/PlatEMO/.git/refs/heads/master",
        41,
        "dfef63b29f2e10f7820b57b7644cc842da39d3a00a3e2da0dd9c84aefae60873",
    ),
    "platemo.mokp_problem": (
        "candidate_external_problem_definition",
        "external/PlatEMO/PlatEMO/Problems/Multi-objective optimization/Real-world MOPs/MOKP.m",
        2631,
        "e6008ae659afedf21d34a82fa215fd540f8dc7a7c49d1eb56bcdb053c192a17f",
    ),
    "platemo.readme": (
        "local_version_and_rights_evidence",
        "external/PlatEMO/README.md",
        5668,
        "ed11d7d88f4e7507ba8fcf06cdaa480d4f9a1968fadae15a5fa10bc6cbde4b31",
    ),
    "pymoo.adapter": (
        "project_motsp_adapter",
        "mo_nco/external_pymoo_baseline.py",
        8676,
        "a324e78abb6b92cd1f2e6fd24242c0c6df68a85c5a12a2cd62a579c27366a6ff",
    ),
    "pymoo.consumer_contract": (
        "runtime_authorization_hold_evidence",
        "artifacts/capacity_v3_durable_pymoo_runtime_v5/CONSUMER_CONTRACT.json",
        2088,
        "90d6678240ce95f5d3d7e0e67e0b698b601cd5134b250b391441bbbb003149f1",
    ),
    "pymoo.license": (
        "upstream_license_text",
        "artifacts/capacity_v3_durable_pymoo_runtime_v5/venv/Lib/site-packages/pymoo-0.6.2.dist-info/licenses/LICENSE",
        10956,
        "737c0b1c449cb08a5c58c66650ac623fa0104673b26d92af660ef52bcc5e7f27",
    ),
    "pymoo.requirement": (
        "hash_pinned_requirement",
        "artifacts/capacity_v3_durable_pymoo_runtime_v5/core_requirements/pymoo.txt",
        92,
        "ce3bdc3f14e45fbc6a4302e9c26d920e9648e094c1826bdf7c293d1406cdf3aa",
    ),
    "pymoo.wheel": (
        "upstream_distribution",
        "artifacts/capacity_v3_durable_pymoo_runtime_v5/wheelhouse/pymoo-0.6.2-cp311-cp311-win_amd64.whl",
        1872126,
        "5ee47aab8c8525ed927a204488f3297a38a0557b414c52d596c9a2f95524f255",
    ),
    "pymoo.wheelhouse_manifest": (
        "dependency_closure_manifest",
        "artifacts/capacity_v3_durable_pymoo_runtime_v5/WHEELHOUSE.json",
        27234,
        "7f28f2b1e4d3447ef85b9db8b6ae0b68f1cd8aeec35bf14ba46ade749ebcf0cc",
    ),
    "runtime.python311": (
        "pinned_python_executable",
        "artifacts/capacity_v3_durable_pymoo_runtime_v5/venv/Scripts/python.exe",
        274248,
        "f22b6439acbf7a459c24d21f515dc5a556dd58e144763bfb6b2dc6ed3839d11f",
    ),
    "shared.lkh_2ppls_adapter": (
        "project_hybrid_adapter",
        "mo_nco/external_lkh_2ppls_baseline.py",
        8708,
        "6019e2c74ef11371479fb02b85cd369136d3f6756a0eec208f15c0f0b1ca5218",
    ),
    "shared.lkh_official_adapter": (
        "project_scalarization_adapter",
        "mo_nco/external_official_lkh_baseline.py",
        7641,
        "776e14e17263f2af4127d8083f78bc01015bd15cfa961beae0781865a60c48c0",
    ),
    "shared.mature_adapter": (
        "project_external_process_protocol",
        "mo_nco/mature_baselines.py",
        23236,
        "6ea9ed5ddc002ec877451466157cf01399d7ad806784cea77b3be6a1c65f3560",
    ),
}

# The readable fields expose the scientific classification, while the payload
# digest seals every remaining command, parameter, license, artifact-reference,
# budget, and boundary field of each comparator object.
EXPECTED_COMPARATORS = {
    "mokp-binary-moead-native-v1": {
        "family": "MOKP",
        "classification": "project_native_reference",
        "license": "NO_REPOSITORY_ROOT_LICENSE_FOUND",
        "development": True,
        "matched_fe": True,
        "kind": "python_api",
        "fingerprint": "18c81286ff6c32bb70e8fa1ece14b15d25692a4f49d41ae23cd1c18c5d9c065a",
    },
    "mokp-binary-nsga2-native-v1": {
        "family": "MOKP",
        "classification": "project_native_reference",
        "license": "NO_REPOSITORY_ROOT_LICENSE_FOUND",
        "development": True,
        "matched_fe": True,
        "kind": "python_api",
        "fingerprint": "cd4fb0ce5183f65e4ce69f82ca6423d846414f8e842571815b2c77fce84f5a0b",
    },
    "mokp-pls-native-v1": {
        "family": "MOKP",
        "classification": "project_native_reference",
        "license": "NO_REPOSITORY_ROOT_LICENSE_FOUND",
        "development": True,
        "matched_fe": True,
        "kind": "python_api",
        "fingerprint": "38652cee24859b34867ea843623e55d5da0e33b09d86013cb1278d1364f87b0d",
    },
    "motsp-lkh3-scalar-3.0.14-v1": {
        "family": "MOTSP",
        "classification": "external_native_single_objective_solver_adapted_reference",
        "license": "RESEARCH_USE_ONLY_ALL_RIGHTS_RESERVED",
        "development": True,
        "matched_fe": False,
        "kind": "offline_command",
        "fingerprint": "34749d603bb34f1c6f1aea576ae4de925e78efe0b7e9f272507799fbb181d7fe",
    },
    "motsp-lkh3-seeded-project-2opt-pls-v1": {
        "family": "MOTSP",
        "classification": "project_hybrid_reference",
        "license": "MIXED_PROJECT_UNLICENSED_AND_LKH_RESEARCH_USE_ONLY",
        "development": True,
        "matched_fe": False,
        "kind": "offline_command",
        "fingerprint": "e6df40c9fd3354b39978b7525c12da97cf052ec9c877ad51a9dbf3467a32c489",
    },
    "motsp-paquete-published-tpls-archive-v1": {
        "family": "MOTSP",
        "classification": "published_result_archive_reference",
        "license": "NO_EXPLICIT_RESULT_ARCHIVE_LICENSE_OBSERVED",
        "development": True,
        "matched_fe": False,
        "kind": "offline_cached_archive_reader",
        "fingerprint": "e44ad244c321039867db435e546a9174cae63a2ecd59b51fa2959702a672ce14",
    },
    "motsp-pymoo-moead-0.6.2-adapted-v1": {
        "family": "MOTSP",
        "classification": "external_library_adapted_reference",
        "license": "Apache-2.0",
        "development": True,
        "matched_fe": True,
        "kind": "offline_command",
        "fingerprint": "ba0507815487e8caff43703866a71696882bb240a06b7473e67b62ba8cf83d0c",
    },
    "motsp-pymoo-nsga2-0.6.2-adapted-v1": {
        "family": "MOTSP",
        "classification": "external_library_adapted_reference",
        "license": "Apache-2.0",
        "development": True,
        "matched_fe": True,
        "kind": "offline_command",
        "fingerprint": "367f9aebab61e857ea1234a9894e2755b70f9c9bf5bb0bcc51e9833d77c1bad6",
    },
    "platemo-mokp-candidate-v4.14-era": {
        "family": "MOKP",
        "classification": "candidate_external_platform_not_integrated",
        "license": "RESEARCH_PURPOSES_WITH_ACKNOWLEDGMENT_NON_SPDX",
        "development": False,
        "matched_fe": False,
        "kind": "not_executable_in_current_environment",
        "fingerprint": "d154d73837a421696d86cd5ad051f67d0c36bfdf7ae378b284f12ea55b991f4f",
    },
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class RegistryVerificationError(ValueError):
    """The registry or one of its bound local artifacts failed closed."""


def _fail(message: str) -> NoReturn:
    raise RegistryVerificationError(message)


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is prohibited: {key!r}")
        result[key] = value
    return result


def _load_json_strict(path: Path) -> Any:
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"cannot read strict UTF-8 JSON {path}: {exc}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except RegistryVerificationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        _fail(f"invalid JSON {path}: {exc}")


def _expect_type(value: Any, expected: type, where: str) -> None:
    if type(value) is not expected:
        _fail(f"{where} must have exact type {expected.__name__}, got {type(value).__name__}")


def _expect_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    _expect_type(value, dict, where)
    actual = set(value)
    if actual != expected:
        _fail(
            f"{where} keys differ; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _expect_nonempty_string(value: Any, where: str) -> str:
    _expect_type(value, str, where)
    if not value:
        _fail(f"{where} must not be empty")
    return value


def _validate_json_value(value: Any, where: str) -> None:
    """Reject null, exotic values, non-string keys, and non-finite floats."""
    value_type = type(value)
    if value_type is str or value_type is bool:
        return
    if value_type is int:
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail(f"{where} contains a non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str or not key:
                _fail(f"{where} contains a non-string or empty object key")
            _validate_json_value(item, f"{where}.{key}")
        return
    _fail(f"{where} contains prohibited JSON value type {value_type.__name__}")


def _canonical_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if trailing_newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot hash bound artifact {path}: {exc}")
    return digest.hexdigest()


def _canonical_relative_path(value: Any, where: str) -> PurePosixPath:
    text = _expect_nonempty_string(value, where)
    if "\\" in text:
        _fail(f"{where} must use POSIX separators")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        _fail(f"{where} must be repository-relative")
    if text != posix.as_posix() or any(part in {"", ".", ".."} for part in posix.parts):
        _fail(f"{where} is not a canonical relative path: {text!r}")
    return posix


def _resolve_bound_file(repository_root: Path, relative: PurePosixPath, where: str) -> Path:
    unresolved = repository_root.joinpath(*relative.parts)
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        _fail(f"{where} does not resolve to an existing file: {exc}")
    try:
        resolved.relative_to(repository_root)
    except ValueError:
        _fail(f"{where} escapes repository root after resolution: {resolved}")
    if not resolved.is_file():
        _fail(f"{where} is not a regular file: {resolved}")
    return resolved


def _validate_artifacts(
    artifacts: Any, repository_root: Path
) -> tuple[dict[str, Path], set[str]]:
    _expect_type(artifacts, list, "artifacts")
    if len(artifacts) != len(EXPECTED_ARTIFACTS):
        _fail(f"artifacts count must be {len(EXPECTED_ARTIFACTS)}")
    artifact_ids: list[str] = []
    resolved_paths: dict[str, Path] = {}
    for index, artifact_value in enumerate(artifacts):
        where = f"artifacts[{index}]"
        artifact = _expect_keys(artifact_value, ARTIFACT_KEYS, where)
        artifact_id = _expect_nonempty_string(artifact["artifact_id"], f"{where}.artifact_id")
        if not IDENTIFIER.fullmatch(artifact_id):
            _fail(f"{where}.artifact_id is not canonical: {artifact_id!r}")
        artifact_ids.append(artifact_id)
        role = _expect_nonempty_string(artifact["role"], f"{where}.role")
        relative = _canonical_relative_path(artifact["path"], f"{where}.path")
        _expect_type(artifact["bytes"], int, f"{where}.bytes")
        if artifact["bytes"] <= 0:
            _fail(f"{where}.bytes must be positive")
        digest = _expect_nonempty_string(artifact["sha256"], f"{where}.sha256")
        if not HEX64.fullmatch(digest):
            _fail(f"{where}.sha256 must be lowercase SHA-256")

        expected = EXPECTED_ARTIFACTS.get(artifact_id)
        actual_metadata = (role, relative.as_posix(), artifact["bytes"], digest)
        if expected is None or actual_metadata != expected:
            _fail(f"{where} metadata is not the independently frozen value for {artifact_id!r}")

        resolved = _resolve_bound_file(repository_root, relative, where)
        try:
            actual_bytes = resolved.stat().st_size
        except OSError as exc:
            _fail(f"cannot stat bound artifact {resolved}: {exc}")
        if actual_bytes != artifact["bytes"]:
            _fail(
                f"{where} byte count mismatch for {relative.as_posix()}: "
                f"expected {artifact['bytes']}, got {actual_bytes}"
            )
        actual_digest = _sha256_file(resolved)
        if actual_digest != digest:
            _fail(
                f"{where} SHA-256 mismatch for {relative.as_posix()}: "
                f"expected {digest}, got {actual_digest}"
            )
        resolved_paths[artifact_id] = resolved

    if artifact_ids != sorted(artifact_ids) or len(artifact_ids) != len(set(artifact_ids)):
        _fail("artifacts must be strictly sorted by unique artifact_id")
    if set(artifact_ids) != set(EXPECTED_ARTIFACTS):
        _fail("artifact_id set differs from the independently frozen set")
    return resolved_paths, set(artifact_ids)


def _validate_string_object(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    result = _expect_keys(value, keys, where)
    for key in keys:
        _expect_nonempty_string(result[key], f"{where}.{key}")
    return result


def _validate_comparators(comparators: Any, artifact_ids: set[str]) -> list[dict[str, Any]]:
    _expect_type(comparators, list, "comparators")
    if len(comparators) != len(EXPECTED_COMPARATORS):
        _fail(f"comparators count must be {len(EXPECTED_COMPARATORS)}")
    ids: list[str] = []
    used_artifacts: set[str] = set()
    validated: list[dict[str, Any]] = []

    for index, comparator_value in enumerate(comparators):
        where = f"comparators[{index}]"
        comparator = _expect_keys(comparator_value, COMPARATOR_KEYS, where)
        comparator_id = _expect_nonempty_string(
            comparator["comparator_id"], f"{where}.comparator_id"
        )
        if not IDENTIFIER.fullmatch(comparator_id):
            _fail(f"{where}.comparator_id is not canonical: {comparator_id!r}")
        ids.append(comparator_id)
        for key in ("problem_family", "classification", "claim_label", "scientific_boundary"):
            _expect_nonempty_string(comparator[key], f"{where}.{key}")

        _validate_string_object(
            comparator["source_identity"], SOURCE_IDENTITY_KEYS, f"{where}.source_identity"
        )
        license_record = _validate_string_object(
            comparator["license"], LICENSE_KEYS, f"{where}.license"
        )

        references = comparator["artifact_ids"]
        _expect_type(references, list, f"{where}.artifact_ids")
        if not references:
            _fail(f"{where}.artifact_ids must not be empty")
        for ref_index, artifact_id in enumerate(references):
            _expect_nonempty_string(artifact_id, f"{where}.artifact_ids[{ref_index}]")
            if artifact_id not in artifact_ids:
                _fail(f"{where} references unknown artifact_id {artifact_id!r}")
        if references != sorted(references) or len(references) != len(set(references)):
            _fail(f"{where}.artifact_ids must be strictly sorted and unique")
        used_artifacts.update(references)

        evidence_id = license_record["evidence_artifact_id"]
        if evidence_id != "NOT_AVAILABLE" and evidence_id not in artifact_ids:
            _fail(f"{where}.license references unknown evidence artifact {evidence_id!r}")

        invocation = _expect_keys(comparator["invocation"], INVOCATION_KEYS, f"{where}.invocation")
        for key in ("kind", "python_api", "cwd"):
            _expect_nonempty_string(invocation[key], f"{where}.invocation.{key}")
        _expect_type(invocation["argv"], list, f"{where}.invocation.argv")
        for argv_index, argument in enumerate(invocation["argv"]):
            _expect_nonempty_string(argument, f"{where}.invocation.argv[{argv_index}]")
        environment = invocation["environment"]
        _expect_type(environment, dict, f"{where}.invocation.environment")
        for key, value in environment.items():
            _expect_nonempty_string(key, f"{where}.invocation.environment key")
            _expect_nonempty_string(value, f"{where}.invocation.environment.{key}")
        _expect_type(invocation["network_access"], bool, f"{where}.invocation.network_access")
        _expect_type(
            invocation["offline_cache_required"], bool, f"{where}.invocation.offline_cache_required"
        )
        if invocation["network_access"] is not False:
            _fail(f"{where} attempts to authorize network access")

        parameters = comparator["parameters"]
        _expect_type(parameters, dict, f"{where}.parameters")
        if not parameters:
            _fail(f"{where}.parameters must not be empty")
        _validate_json_value(parameters, f"{where}.parameters")

        budget = _expect_keys(
            comparator["budget_semantics"], BUDGET_KEYS, f"{where}.budget_semantics"
        )
        for key in BUDGET_KEYS - {"matched_first_true_objective_evaluation_budget"}:
            _expect_nonempty_string(budget[key], f"{where}.budget_semantics.{key}")
        _expect_type(
            budget["matched_first_true_objective_evaluation_budget"],
            bool,
            f"{where}.budget_semantics.matched_first_true_objective_evaluation_budget",
        )

        eligibility = _expect_keys(
            comparator["eligibility"], ELIGIBILITY_KEYS, f"{where}.eligibility"
        )
        for key in ELIGIBILITY_KEYS:
            _expect_type(eligibility[key], bool, f"{where}.eligibility.{key}")
        if eligibility["registry_entry_frozen"] is not True:
            _fail(f"{where} is not frozen")
        for prohibited in (
            "selection_execution_authorized",
            "confirmation_execution_authorized",
            "formal_primary_eligible",
            "external_family_native_strong_baseline",
            "public_redistribution_authorized",
        ):
            if eligibility[prohibited] is not False:
                _fail(f"{where}.eligibility.{prohibited} must fail closed")

        expected = EXPECTED_COMPARATORS.get(comparator_id)
        if expected is None:
            _fail(f"unexpected comparator_id {comparator_id!r}")
        visible_semantics = {
            "family": comparator["problem_family"],
            "classification": comparator["classification"],
            "license": license_record["expression"],
            "development": eligibility["development_reference_eligible"],
            "matched_fe": budget["matched_first_true_objective_evaluation_budget"],
            "kind": invocation["kind"],
        }
        for key, actual in visible_semantics.items():
            if actual != expected[key] or type(actual) is not type(expected[key]):
                _fail(f"{where} frozen semantic field {key!r} differs")
        fingerprint = _sha256_bytes(_canonical_bytes(comparator))
        if fingerprint != expected["fingerprint"]:
            _fail(
                f"{where} comparator payload differs from the independently frozen "
                f"command/license/budget/classification record"
            )
        validated.append(comparator)

    if ids != sorted(ids) or len(ids) != len(set(ids)):
        _fail("comparators must be strictly sorted by unique comparator_id")
    if set(ids) != set(EXPECTED_COMPARATORS):
        _fail("comparator_id set differs from the independently frozen set")
    if used_artifacts != artifact_ids:
        _fail(f"unreferenced artifact ids are prohibited: {sorted(artifact_ids - used_artifacts)!r}")
    return validated


def _expect_false_authority(value: Any, where: str, keys: tuple[str, ...]) -> None:
    _expect_type(value, dict, where)
    for key in keys:
        if key not in value:
            _fail(f"{where}.{key} is missing")
        _expect_type(value[key], bool, f"{where}.{key}")
        if value[key] is not False:
            _fail(f"{where}.{key} must remain false")


def _validate_bound_authority_evidence(paths: dict[str, Path]) -> None:
    consumer = _load_json_strict(paths["pymoo.consumer_contract"])
    _expect_type(consumer, dict, "pymoo consumer contract")
    if consumer.get("status") != (
        "PRE_AND_POST_FILES_DIRECTORIES_AND_PHYSICAL_IDENTITIES_REQUIRED__AUTHORIZATION_HOLD"
    ):
        _fail("pymoo consumer contract is not at AUTHORIZATION_HOLD")
    if consumer.get("consumer_use_authorized") is not False:
        _fail("pymoo consumer contract must not authorize consumer use")
    _expect_false_authority(
        consumer.get("authority"),
        "pymoo consumer contract authority",
        (
            "execution_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
            "selection_authorized",
        ),
    )

    wheelhouse = _load_json_strict(paths["pymoo.wheelhouse_manifest"])
    _expect_false_authority(
        wheelhouse,
        "pymoo wheelhouse manifest",
        (
            "execution_authorized",
            "formal_authorized",
            "ijoc_submission_authorized",
            "selection_authorized",
        ),
    )
    wheels = wheelhouse.get("wheels")
    _expect_type(wheels, list, "pymoo wheelhouse manifest.wheels")
    pymoo_rows = [
        row
        for row in wheels
        if type(row) is dict
        and row.get("normalized_name") == "pymoo"
        and row.get("version") == "0.6.2"
    ]
    if len(pymoo_rows) != 1:
        _fail("wheelhouse must contain exactly one pymoo 0.6.2 row")
    row = pymoo_rows[0]
    if (
        type(row.get("content_bytes")) is not int
        or row["content_bytes"] != EXPECTED_ARTIFACTS["pymoo.wheel"][2]
        or row.get("sha256") != EXPECTED_ARTIFACTS["pymoo.wheel"][3]
    ):
        _fail("wheelhouse pymoo row does not bind the frozen wheel bytes")

    try:
        head = paths["platemo.git_head"].read_bytes()
        master_ref = paths["platemo.git_master_ref"].read_bytes()
    except OSError as exc:
        _fail(f"cannot read PlatEMO Git binding: {exc}")
    if head != b"ref: refs/heads/master\n":
        _fail("PlatEMO HEAD is not the frozen master symbolic reference")
    if master_ref != b"ce1091953e19ca5650151c0d5b22cdaa12adeefd\n":
        _fail("PlatEMO master ref is not the frozen local commit")


def verify_registry(
    registry_path: str | Path = DEFAULT_REGISTRY,
    repository_root: str | Path = DEFAULT_REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Verify the registry and all local bindings without using the network."""
    root_input = Path(repository_root)
    try:
        root = root_input.resolve(strict=True)
    except OSError as exc:
        _fail(f"repository root does not exist: {exc}")
    if not root.is_dir():
        _fail(f"repository root is not a directory: {root}")

    registry_file = Path(registry_path)
    try:
        if not registry_file.resolve(strict=True).is_file():
            _fail(f"registry is not a file: {registry_file}")
    except OSError as exc:
        _fail(f"registry does not exist: {exc}")
    registry = _load_json_strict(registry_file)
    _validate_json_value(registry, "registry")
    registry = _expect_keys(registry, TOP_LEVEL_KEYS, "registry")

    expected_scalars = {
        "schema": SCHEMA,
        "primary_source_cutoff_date": "2026-08-22",
        "scope": "development_reference_freeze_only_not_strong_external_baseline_evidence",
        "status": "PASS_DEVELOPMENT_REFERENCE_FREEZE_ONLY__IJOC_HOLD",
        "network_policy": "PROHIBITED_OFFLINE_VERIFICATION_ONLY",
    }
    for key, expected in expected_scalars.items():
        _expect_nonempty_string(registry[key], f"registry.{key}")
        if registry[key] != expected:
            _fail(f"registry.{key} differs from the frozen value")

    gates = _expect_keys(registry["gates"], GATE_KEYS, "registry.gates")
    for key in (
        "registry_entry_count",
        "development_reference_eligible_count",
        "external_family_native_strong_baseline_count",
        "formal_primary_eligible_count",
    ):
        _expect_type(gates[key], int, f"registry.gates.{key}")
    for key in GATE_KEYS - {
        "registry_entry_count",
        "development_reference_eligible_count",
        "external_family_native_strong_baseline_count",
        "formal_primary_eligible_count",
    }:
        _expect_nonempty_string(gates[key], f"registry.gates.{key}")
    if gates != EXPECTED_GATES:
        _fail("registry.gates differs from the independently hard-coded fail-closed gates")

    supplied_payload_digest = _expect_nonempty_string(
        registry["registry_payload_sha256"], "registry.registry_payload_sha256"
    )
    if not HEX64.fullmatch(supplied_payload_digest):
        _fail("registry.registry_payload_sha256 must be lowercase SHA-256")
    payload = dict(registry)
    del payload["registry_payload_sha256"]
    actual_payload_digest = _sha256_bytes(_canonical_bytes(payload, trailing_newline=True))
    if actual_payload_digest != supplied_payload_digest:
        _fail(
            "registry payload SHA-256 mismatch: "
            f"expected {supplied_payload_digest}, got {actual_payload_digest}"
        )

    paths, artifact_ids = _validate_artifacts(registry["artifacts"], root)
    comparators = _validate_comparators(registry["comparators"], artifact_ids)

    derived = {
        "registry_entry_count": len(comparators),
        "development_reference_eligible_count": sum(
            comparator["eligibility"]["development_reference_eligible"]
            for comparator in comparators
        ),
        "external_family_native_strong_baseline_count": sum(
            comparator["eligibility"]["external_family_native_strong_baseline"]
            for comparator in comparators
        ),
        "formal_primary_eligible_count": sum(
            comparator["eligibility"]["formal_primary_eligible"]
            for comparator in comparators
        ),
    }
    for key, actual in derived.items():
        if type(actual) is not int or actual != gates[key]:
            _fail(f"derived gate {key!r} differs from registry.gates")
    if any("strong_baseline" in comparator["classification"] for comparator in comparators):
        _fail("a comparator classification attempts to claim strong-baseline status")

    _validate_bound_authority_evidence(paths)

    return {
        "schema": "ijoc_v21e3r1_v7_reference_comparator_registry_verification_v1",
        "status": "PASS_STRICT_OFFLINE_DEVELOPMENT_REFERENCE_FREEZE_ONLY",
        "registry_file_sha256": _sha256_file(registry_file),
        "registry_payload_sha256": actual_payload_digest,
        "artifact_count": len(paths),
        "comparator_count": len(comparators),
        "development_reference_eligible_count": derived[
            "development_reference_eligible_count"
        ],
        "external_family_native_strong_baseline_count": 0,
        "formal_primary_eligible_count": 0,
        "selection_execution": "NOT_AUTHORIZED",
        "confirmation_execution": "NOT_AUTHORIZED",
        "formal_materialization": "PROHIBITED",
        "formal_primary_gate": "FAIL_CLOSED_NO_EXTERNAL_FAMILY_NATIVE_STRONG_BASELINE",
        "network_calls": 0,
        "ijoc_status": "IJOC_HOLD",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    arguments = parser.parse_args(argv)
    try:
        receipt = verify_registry(arguments.registry, arguments.repository_root)
    except RegistryVerificationError as exc:
        print(
            json.dumps(
                {
                    "schema": "ijoc_v21e3r1_v7_reference_comparator_registry_verification_v1",
                    "status": "FAIL_CLOSED",
                    "error": str(exc),
                    "ijoc_status": "IJOC_HOLD",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
