#!/usr/bin/env python3
from __future__ import annotations

"""Freeze V21e3r1 prospective engineering boundaries without granting authority.

The freezer is deliberately offline.  It verifies and copies the existing
development-reference registry, independent replay design and golden corpus,
independent simultaneous-inference implementation, and targeted precedent
matrix into one exclusive directory.  Its receipts are engineering bindings;
they cannot authorize selection, confirmation, a formal study, or an IJOC
claim.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, NoReturn, Sequence


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
FAMILIES = ("MOKP", "MOTSP")
SIMULTANEOUS_METHOD = (
    "one_sided_observed_se_max_t_paired_case_cluster_bootstrap_v1"
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_823

REGISTRY_RELATIVE = PurePosixPath(
    "ijoc_submission_v21e3r1/baselines/v7_reference_comparator_registry.json"
)
REGISTRY_VERIFIER_RELATIVE = PurePosixPath(
    "ijoc_submission_v21e3r1/scripts/verify_v21e3r1_reference_comparator_registry.py"
)
REPLAY_SPEC_RELATIVE = PurePosixPath(
    "independent_reproduction/V21E3R1_ALGORITHM_REPLAY_SPEC_V1.md"
)
REPLAY_COMPARATOR_RELATIVE = PurePosixPath(
    "independent_reproduction/compare_v21e3r1_algorithm_events.py"
)
REPLAY_TEST_RELATIVE = PurePosixPath(
    "tests/test_v21e3r1_independent_algorithm_replay_design.py"
)
GOLDEN_RELATIVES = {
    "reference_valid": PurePosixPath("independent_reproduction/golden/reference_valid.jsonl"),
    "external_placeholder": PurePosixPath(
        "independent_reproduction/golden/external_valid.jsonl"
    ),
    "negative_decision_mismatch": PurePosixPath(
        "independent_reproduction/golden/negative_decision_mismatch.jsonl"
    ),
}
SIM_EVALUATOR_RELATIVE = PurePosixPath(
    "independent_reproduction/recompute_v21e3r1_simultaneous_bounds.py"
)
SIM_TEST_RELATIVE = PurePosixPath(
    "tests/test_v21e3r1_independent_simultaneous_inference.py"
)
PRODUCTION_METRIC_RELATIVE = PurePosixPath("mo_nco/pareto_ijoc_analysis.py")
INDEPENDENT_METRIC_RELATIVE = PurePosixPath(
    "independent_reproduction/recompute_v21e3r1_successor_metrics.py"
)
PRECEDENT_JSON_RELATIVE = PurePosixPath(
    "ijoc_submission_v21e3r1/novelty/precedent_mechanism_matrix.json"
)
PRECEDENT_RENDER_RELATIVES = {
    "csv": PurePosixPath(
        "ijoc_submission_v21e3r1/novelty/precedent_mechanism_matrix.csv"
    ),
    "markdown": PurePosixPath(
        "ijoc_submission_v21e3r1/novelty/precedent_mechanism_matrix.md"
    ),
    "tex": PurePosixPath(
        "ijoc_submission_v21e3r1/novelty/precedent_mechanism_matrix.tex"
    ),
}
PRECEDENT_RENDERER_RELATIVE = PurePosixPath("mo_nco/pareto_v21e3r1_precedent.py")
PRECEDENT_TEST_RELATIVE = PurePosixPath(
    "tests/test_pareto_v21e3r1_v7_theory_diagnostics.py"
)


class FreezeError(ValueError):
    """A prospective-boundary input or output violated the frozen contract."""


def _fail(message: str) -> NoReturn:
    raise FreezeError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is prohibited: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is prohibited: {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: object, *, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or not key:
                _fail(f"{label} contains an invalid object key")
            _validate_json_tree(item, label=f"{label}.{key}")
        return
    _fail(f"{label} contains prohibited type {type(value).__name__}")


def _parse_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{label} is not strict UTF-8: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except FreezeError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        _fail(f"{label} is malformed JSON: {error}")
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    _validate_json_tree(value, label=label)
    return value


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact JSON object")
    observed = set(value)
    if observed != expected:
        _fail(
            f"{label} key set drifted; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be an exact nonempty string")
    return value


def _identifier(value: object, *, label: str) -> str:
    result = _string(value, label=label)
    if IDENTIFIER_RE.fullmatch(result) is None:
        _fail(f"{label} must be a canonical identifier")
    return result


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an exact JSON boolean")
    return value


def _canonical_relative(value: object, *, label: str) -> PurePosixPath:
    raw = _string(value, label=label)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        "\\" in raw
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.as_posix() != raw
        or any(part in {"", ".", ".."} for part in posix.parts)
        or any(ord(character) < 32 for character in raw)
    ):
        _fail(f"{label} is not a canonical contained POSIX path")
    return posix


def _contained_input(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    try:
        path = root.joinpath(*relative.parts).resolve(strict=True)
    except OSError as error:
        _fail(f"{label} does not resolve: {error}")
    try:
        path.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes the repository root")
    if not path.is_file():
        _fail(f"{label} is not a regular file")
    return path


def _binding(relative: str, raw: bytes) -> dict[str, object]:
    _canonical_relative(relative, label="artifact binding path")
    return {"path": relative, "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _payload_receipt(core: Mapping[str, object]) -> dict[str, object]:
    result = dict(core)
    result["receipt_payload_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return result


def _write_bytes_exclusive(root: Path, relative: str, raw: bytes) -> dict[str, object]:
    rel = _canonical_relative(relative, label="output artifact path")
    path = root.joinpath(*rel.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _fail(f"output artifact already exists; exclusive create required: {relative}")
    return _binding(relative, raw)


def _write_json_exclusive(
    root: Path, relative: str, value: Mapping[str, object]
) -> dict[str, object]:
    return _write_bytes_exclusive(root, relative, _canonical_bytes(value))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load bound implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_inputs(
    root: Path, relatives: Sequence[PurePosixPath]
) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for relative in relatives:
        key = relative.as_posix()
        if key in snapshots:
            continue
        path = _contained_input(root, relative, label=f"input {key}")
        raw = path.read_bytes()
        snapshots[key] = {
            "path": key,
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
        }
    return snapshots


def _verify_snapshots_stable(
    root: Path, snapshots: Mapping[str, Mapping[str, object]]
) -> None:
    for key in sorted(snapshots, key=str.casefold):
        expected = snapshots[key]
        relative = _canonical_relative(key, label="snapshot path")
        path = _contained_input(root, relative, label=f"TOCTOU input {key}")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or _sha256_bytes(raw) != expected["sha256"]:
            _fail(f"input drifted during freeze (TOCTOU fail-closed): {key}")


def _copy_snapshot(
    root: Path,
    staging: Path,
    snapshots: Mapping[str, Mapping[str, object]],
    source_relative: PurePosixPath,
    output_relative: str,
) -> dict[str, object]:
    key = source_relative.as_posix()
    expected = snapshots[key]
    raw = _contained_input(root, source_relative, label=f"copy input {key}").read_bytes()
    if len(raw) != expected["bytes"] or _sha256_bytes(raw) != expected["sha256"]:
        _fail(f"input drifted before copy (TOCTOU fail-closed): {key}")
    return _write_bytes_exclusive(staging, output_relative, raw)


def _classification_flags(comparator_id: str) -> tuple[bool, bool, bool]:
    frozen = {
        "mokp-binary-moead-native-v1": (False, True, False),
        "mokp-binary-nsga2-native-v1": (False, True, False),
        "mokp-pls-native-v1": (False, True, False),
        "motsp-lkh3-scalar-3.0.14-v1": (True, False, False),
        "motsp-lkh3-seeded-project-2opt-pls-v1": (False, True, False),
        "motsp-paquete-published-tpls-archive-v1": (True, True, False),
        "motsp-pymoo-moead-0.6.2-adapted-v1": (True, False, False),
        "motsp-pymoo-nsga2-0.6.2-adapted-v1": (True, False, False),
        "platemo-mokp-candidate-v4.14-era": (True, True, False),
    }
    if comparator_id not in frozen:
        _fail(f"unrecognized comparator cannot be classified: {comparator_id}")
    return frozen[comparator_id]


def _freeze_baselines(
    *,
    root: Path,
    staging: Path,
    snapshots: Mapping[str, Mapping[str, object]],
    registry: Mapping[str, object],
    verification: Mapping[str, object],
    study_id: str,
    candidate_id: str,
    study_metric_spec_sha256: str,
) -> tuple[dict[str, object], dict[str, int]]:
    registry_copy = _copy_snapshot(
        root,
        staging,
        snapshots,
        REGISTRY_RELATIVE,
        "baseline-design/reference-comparator.registry.json",
    )
    verifier_copy = _copy_snapshot(
        root,
        staging,
        snapshots,
        REGISTRY_VERIFIER_RELATIVE,
        "baseline-design/reference-comparator.verifier.py",
    )
    verification_receipt = _payload_receipt(
        {
            **dict(verification),
            "registry_artifacts_verified": True,
            "network_policy": "PROHIBITED_OFFLINE_VERIFICATION_ONLY",
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
        }
    )
    verification_binding = _write_json_exclusive(
        staging,
        "baseline-design/reference-comparator.verification.receipt.json",
        verification_receipt,
    )

    artifacts = registry.get("artifacts")
    comparators = registry.get("comparators")
    if type(artifacts) is not list or type(comparators) is not list:
        _fail("verified registry artifacts/comparators are not exact arrays")
    artifacts_by_id: dict[str, dict[str, object]] = {}
    for index, item in enumerate(artifacts):
        artifact = _exact_keys(
            item,
            {"artifact_id", "role", "path", "bytes", "sha256"},
            label=f"registry artifacts[{index}]",
        )
        artifact_id = _identifier(artifact["artifact_id"], label="registry artifact id")
        if artifact_id in artifacts_by_id:
            _fail("registry contains duplicate artifact IDs")
        _string(artifact["role"], label=f"registry artifact {artifact_id}.role")
        relative = _canonical_relative(
            artifact["path"], label=f"registry artifact {artifact_id}.path"
        )
        snapshot = snapshots.get(relative.as_posix())
        if snapshot is None:
            _fail(f"registry artifact is missing from input snapshot: {artifact_id}")
        if (
            _integer(artifact["bytes"], label=f"registry artifact {artifact_id}.bytes", minimum=1)
            != snapshot["bytes"]
            or _sha256(artifact["sha256"], label=f"registry artifact {artifact_id}.sha256")
            != snapshot["sha256"]
        ):
            _fail(f"registry artifact binding disagrees with file: {artifact_id}")
        artifacts_by_id[artifact_id] = artifact

    families: dict[str, list[dict[str, object]]] = {family: [] for family in FAMILIES}
    comparator_ids: set[str] = set()
    for index, value in enumerate(comparators):
        if type(value) is not dict:
            _fail(f"registry comparators[{index}] must be an exact object")
        comparator_id = _identifier(
            value.get("comparator_id"), label=f"registry comparators[{index}].id"
        )
        if comparator_id in comparator_ids:
            _fail("registry contains duplicate comparator IDs")
        comparator_ids.add(comparator_id)
        family = _string(value.get("problem_family"), label=f"{comparator_id}.family")
        if family not in families:
            _fail(f"registry comparator has unsupported family: {comparator_id}")
        classification = _string(
            value.get("classification"), label=f"{comparator_id}.classification"
        )
        artifact_ids = value.get("artifact_ids")
        if type(artifact_ids) is not list or not artifact_ids:
            _fail(f"registry comparator {comparator_id} has no artifact IDs")
        if any(type(item) is not str or item not in artifacts_by_id for item in artifact_ids):
            _fail(f"registry comparator {comparator_id} has an unknown artifact ID")
        if len(set(artifact_ids)) != len(artifact_ids):
            _fail(f"registry comparator {comparator_id} repeats artifact IDs")
        source_entries = [
            {
                "artifact_id": artifact_id,
                "role": artifacts_by_id[artifact_id]["role"],
                "path": artifacts_by_id[artifact_id]["path"],
                "bytes": artifacts_by_id[artifact_id]["bytes"],
                "sha256": artifacts_by_id[artifact_id]["sha256"],
            }
            for artifact_id in artifact_ids
        ]
        source_root = _sha256_bytes(_canonical_bytes(source_entries))
        source_manifest = {
            "schema": "v21e3r1_reference_comparator_source_manifest_v1",
            "baseline_id": comparator_id,
            "problem_family": family,
            "source_identity": value.get("source_identity"),
            "artifact_bindings": source_entries,
            "source_root_sha256": source_root,
        }
        source_binding = _write_json_exclusive(
            staging,
            f"baseline-design/manifests/{comparator_id}.source-manifest.json",
            source_manifest,
        )
        eligibility = value.get("eligibility")
        if type(eligibility) is not dict:
            _fail(f"registry comparator {comparator_id} eligibility must be an object")
        registry_external_strong = _boolean(
            eligibility.get("external_family_native_strong_baseline"),
            label=f"{comparator_id}.external strong eligibility",
        )
        if registry_external_strong:
            _fail("verified development registry unexpectedly claims a strong baseline")
        development_eligible = _boolean(
            eligibility.get("development_reference_eligible"),
            label=f"{comparator_id}.development eligibility",
        )
        external, native, strong = _classification_flags(comparator_id)
        if strong:
            _fail("frozen development comparator classification cannot be strong")
        availability = _payload_receipt(
            {
                "schema": "v21e3r1_reference_comparator_availability_receipt_v1",
                "status": "HOLD_DEVELOPMENT_REFERENCE_ONLY_NO_FORMAL_EVALUATION",
                "baseline_id": comparator_id,
                "problem_family": family,
                "classification": classification,
                "registry_sha256": registry_copy["sha256"],
                "source_manifest_sha256": source_binding["sha256"],
                "study_metric_spec_sha256": study_metric_spec_sha256,
                "development_reference_eligible": development_eligible,
                "evaluation_executed": False,
                "external_family_native_strong_baseline_eligible": False,
                "selection_authorized": False,
                "confirmation_authorized": False,
                "formal_study_authorized": False,
                "scientific_claim_authorized": False,
                "ijoc_submission_status": "IJOC_HOLD",
            }
        )
        availability_binding = _write_json_exclusive(
            staging,
            f"baseline-design/evaluations/{comparator_id}.availability.receipt.json",
            availability,
        )
        families[family].append(
            {
                "baseline_id": comparator_id,
                "classification": classification,
                "external": external,
                "family_native": native,
                "strong": strong,
                "development_reference_eligible": development_eligible,
                "external_family_native_strong_baseline_eligible": False,
                "source_manifest_path": source_binding["path"],
                "source_manifest_sha256": source_binding["sha256"],
                "evaluation_receipt_path": availability_binding["path"],
                "evaluation_receipt_sha256": availability_binding["sha256"],
                "study_metric_spec_sha256": study_metric_spec_sha256,
            }
        )

    counts = {family: 0 for family in FAMILIES}
    family_rows = [
        {
            "family": family,
            "external_family_native_strong_baseline_count": counts[family],
            "baselines": sorted(families[family], key=lambda item: str(item["baseline_id"])),
        }
        for family in FAMILIES
    ]
    receipt = _payload_receipt(
        {
            "schema": "v21e3r1_external_family_native_strong_baseline_registry_receipt_v2",
            "status": "HOLD_NO_EXTERNAL_FAMILY_NATIVE_STRONG_BASELINES",
            "scope": "DEVELOPMENT_REFERENCE_FREEZE_ONLY_NOT_STRONG_EXTERNAL_BASELINE_EVIDENCE",
            "study_id": study_id,
            "candidate_id": candidate_id,
            "study_metric_spec_sha256": study_metric_spec_sha256,
            "primary_source_cutoff_date": registry.get("primary_source_cutoff_date"),
            "registry_path": registry_copy["path"],
            "registry_sha256": registry_copy["sha256"],
            "registry_payload_sha256": registry.get("registry_payload_sha256"),
            "verifier_source_path": verifier_copy["path"],
            "verifier_source_sha256": verifier_copy["sha256"],
            "verification_receipt_path": verification_binding["path"],
            "verification_receipt_sha256": verification_binding["sha256"],
            "artifact_count": len(artifacts_by_id),
            "comparator_count": len(comparator_ids),
            "all_registry_artifacts_verified": True,
            "families": family_rows,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    _write_json_exclusive(staging, "baseline-registry.receipt.json", receipt)
    return receipt, counts


def _freeze_external_replay(
    *,
    root: Path,
    staging: Path,
    snapshots: Mapping[str, Mapping[str, object]],
    study_id: str,
    candidate_id: str,
    successor_source_sha256: str,
    successor_config_sha256: str,
    study_metric_spec_sha256: str,
) -> dict[str, object]:
    design_binding = _copy_snapshot(
        root, staging, snapshots, REPLAY_SPEC_RELATIVE, "external-replay-design/algorithm-replay.spec.md"
    )
    comparator_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        REPLAY_COMPARATOR_RELATIVE,
        "external-replay-design/neutral-comparator.py",
    )
    test_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        REPLAY_TEST_RELATIVE,
        "external-replay-design/neutral-comparator.tests.py",
    )
    golden_bindings: dict[str, dict[str, object]] = {}
    for role, relative in GOLDEN_RELATIVES.items():
        golden_bindings[role] = _copy_snapshot(
            root,
            staging,
            snapshots,
            relative,
            f"external-replay-design/golden/{Path(relative.as_posix()).name}",
        )

    comparator_path = _contained_input(
        root, REPLAY_COMPARATOR_RELATIVE, label="neutral comparator source"
    )
    comparator = _load_module(comparator_path, "v21e3r1_bound_neutral_comparator")
    positive_path = staging / "external-replay-design" / "positive-comparison.receipt.json"
    negative_path = staging / "external-replay-design" / "negative-comparison.receipt.json"
    positive = comparator.compare_event_streams(
        reference_stream=_contained_input(
            root, GOLDEN_RELATIVES["reference_valid"], label="reference golden"
        ),
        candidate_stream=_contained_input(
            root, GOLDEN_RELATIVES["external_placeholder"], label="external placeholder golden"
        ),
        output_receipt=positive_path,
    )
    negative = comparator.compare_event_streams(
        reference_stream=_contained_input(
            root, GOLDEN_RELATIVES["reference_valid"], label="reference golden"
        ),
        candidate_stream=_contained_input(
            root,
            GOLDEN_RELATIVES["negative_decision_mismatch"],
            label="negative mismatch golden",
        ),
        output_receipt=negative_path,
    )
    if positive.get("status") != "PASS_NEUTRAL_EVENT_STREAM_COMPARISON":
        _fail("positive golden algorithm replay comparison did not pass")
    if negative.get("status") != "FAIL_ALGORITHM_EVENT_STREAM_MISMATCH":
        _fail("negative golden algorithm replay comparison did not fail as frozen")
    positive_raw = positive_path.read_bytes()
    negative_raw = negative_path.read_bytes()
    positive_binding = _binding(
        "external-replay-design/positive-comparison.receipt.json", positive_raw
    )
    negative_binding = _binding(
        "external-replay-design/negative-comparison.receipt.json", negative_raw
    )
    streams = positive.get("streams")
    if type(streams) is not dict:
        _fail("positive comparison receipt omitted stream bindings")
    reference_stream = streams.get("reference")
    candidate_stream = streams.get("candidate")
    if type(reference_stream) is not dict or type(candidate_stream) is not dict:
        _fail("positive comparison stream bindings are malformed")
    custody = _payload_receipt(
        {
            "schema": "v21e3r1_external_algorithm_replay_custody_receipt_v1",
            "status": "HOLD_DESIGN_ONLY_NO_EXTERNAL_CUSTODY",
            "reference_algorithm_producer_present": False,
            "external_producer_present": False,
            "producer_authorship_authenticated": False,
            "independent_custody_verified": False,
            "algorithm_execution_independence": False,
            "scientific_independence": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    custody_binding = _write_json_exclusive(
        staging, "external-replay-design/custody-unavailable.receipt.json", custody
    )
    receipt = _payload_receipt(
        {
            "schema": "v21e3r1_external_algorithm_replay_receipt_v2",
            "status": "HOLD_DESIGN_ONLY_NO_EXTERNAL_PRODUCER",
            "scope": "DESIGN_AND_GOLDEN_CORPUS_ONLY_NO_EXTERNAL_PRODUCER_OR_CUSTODY_CLAIM",
            "study_id": study_id,
            "candidate_id": candidate_id,
            "successor_source_sha256": successor_source_sha256,
            "successor_config_sha256": successor_config_sha256,
            "study_metric_spec_sha256": study_metric_spec_sha256,
            "design_spec_path": design_binding["path"],
            "design_spec_sha256": design_binding["sha256"],
            "comparator_source_path": comparator_binding["path"],
            "comparator_source_sha256": comparator_binding["sha256"],
            "comparator_test_path": test_binding["path"],
            "comparator_test_sha256": test_binding["sha256"],
            "golden_streams": golden_bindings,
            "positive_comparison_receipt_path": positive_binding["path"],
            "positive_comparison_receipt_sha256": positive_binding["sha256"],
            "negative_comparison_receipt_path": negative_binding["path"],
            "negative_comparison_receipt_sha256": negative_binding["sha256"],
            "reference_producer_id": reference_stream.get("producer_id"),
            "external_producer_id": candidate_stream.get("producer_id"),
            "reference_source_manifest_sha256": reference_stream.get(
                "producer_source_manifest_sha256"
            ),
            "external_source_manifest_sha256": candidate_stream.get(
                "producer_source_manifest_sha256"
            ),
            "event_streams_match": True,
            "reference_algorithm_producer_present": False,
            "external_producer_present": False,
            "producer_authorship_authenticated": False,
            "independent_producer": False,
            "independent_custody": False,
            "implementation_code_disjoint": False,
            "algorithm_execution_independence": False,
            "custody_receipt_path": custody_binding["path"],
            "custody_receipt_sha256": custody_binding["sha256"],
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    _write_json_exclusive(staging, "external-algorithm-replay.receipt.json", receipt)
    return receipt


def _selection_cells() -> list[dict[str, object]]:
    contrasts = (
        ("C1", "C0", ["primary", "adjacent"]),
        ("C2", "C0", ["primary"]),
        ("C2", "C1", ["adjacent"]),
        ("C3", "C0", ["primary"]),
        ("C3", "C2", ["adjacent"]),
    )
    return [
        {
            "hypothesis_id": f"{family}:{candidate}-{reference}",
            "family": family,
            "candidate": candidate,
            "reference": reference,
            "threshold_roles": roles,
        }
        for family in FAMILIES
        for candidate, reference, roles in contrasts
    ]


def _confirmation_cells() -> list[dict[str, object]]:
    return [
        {
            "hypothesis_template": f"{family}:SELECTED-{reference}",
            "family": family,
            "candidate": "SELECTED",
            "reference": reference,
            "threshold_role": role,
        }
        for family in FAMILIES
        for reference, role in (("C0", "primary"), ("PREDECESSOR", "adjacent"))
    ]


def _freeze_candidate_menu_contract(
    *, staging: Path, study_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    receipt = _payload_receipt(
        {
            "schema": "v21e3r1_successor_candidate_menu_contract_v1",
            "status": "HOLD_SUCCESSOR_CANDIDATE_ABSENT_FROM_SIMULTANEOUS_SPEC_V2",
            "scope": "ALGORITHM_IDENTITIES_AND_DECISION_SHAPES_ONLY_NO_RESULT",
            "study_id": study_id,
            "historical_candidate_order": ["C0", "C1", "C2", "C3"],
            "historical_candidate_menu": [
                {
                    "candidate_id": "C0",
                    "enabled_components": ["strong_native_backbone"],
                    "role": "LEGACY_COMPARATOR_BACKBONE",
                },
                {
                    "candidate_id": "C1",
                    "enabled_components": [
                        "strong_native_backbone",
                        "direction_conditioning",
                    ],
                    "role": "HISTORICAL_DIRECTION_CONDITIONING_CANDIDATE",
                },
                {
                    "candidate_id": "C2",
                    "enabled_components": [
                        "strong_native_backbone",
                        "direction_conditioning",
                        "typed_diversification",
                        "matched_exchange_control",
                    ],
                    "role": "HISTORICAL_MATCHED_EXCHANGE_CANDIDATE",
                },
                {
                    "candidate_id": "C3",
                    "enabled_components": [
                        "strong_native_backbone",
                        "direction_conditioning",
                        "typed_diversification",
                        "neighbor_path_relinking",
                    ],
                    "role": "HISTORICAL_PATH_RELINKING_CANDIDATE",
                },
            ],
            "successor_candidate_id": "V21E3R1_SUCCESSOR_SEARCH_NOVELTY_V1",
            "successor_candidate_contract": {
                "successor_candidate_id": "V21E3R1_SUCCESSOR_SEARCH_NOVELTY_V1",
                "backbone_candidate_id": "C0",
                "legacy_post_initialization_search_policy": "proposal_chain_v21e3r1_v1",
                "successor_post_initialization_search_policy": "post_commit_type_incumbent_anchor_development_v1",
                "legacy_mokp_novelty_generation_policy": "legacy_retry_and_local_v21e3r1_v1",
                "successor_mokp_novelty_generation_policy": "single_attempt_rotating_feasible_exchange_no_refill_development_v1",
            },
            "successor_present_in_simultaneous_spec_v2": False,
            "selection_prohibited_until_revised_simultaneous_spec": True,
            "legacy_c1_decision_shape": {
                "statistical_contrasts": ["MOKP:C1-C0", "MOTSP:C1-C0"],
                "statistical_contrast_count": 2,
                "decision_roles": [
                    "MOKP:C1-C0:primary",
                    "MOKP:C1-C0:adjacent",
                    "MOTSP:C1-C0:primary",
                    "MOTSP:C1-C0:adjacent",
                ],
                "decision_role_count": 4,
                "duplicate_statistic_per_role": False,
            },
            "successor_decision_shape_frozen": False,
            "selection_result_materialized": False,
            "confirmation_result_materialized": False,
            "formal_study_materialized": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    binding = _write_json_exclusive(
        staging, "successor-candidate-menu.contract.json", receipt
    )
    return receipt, binding


def _freeze_future_v3_evidence_contracts(
    *,
    staging: Path,
    study_id: str,
    candidate_id: str,
    successor_source_sha256: str,
    successor_config_sha256: str,
    study_metric_spec_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    common_identity = {
        "study_id": study_id,
        "candidate_id": candidate_id,
        "successor_source_sha256": successor_source_sha256,
        "successor_config_sha256": successor_config_sha256,
        "study_metric_spec_sha256": study_metric_spec_sha256,
    }
    external = _payload_receipt(
        {
            "schema": "v21e3r1_external_algorithm_replay_evidence_contract_v3",
            "status": "FROZEN_FUTURE_EXTERNAL_PRODUCER_CONTRACT_NO_LOCAL_EVIDENCE",
            "scope": "MACHINE_CHECKABLE_PATH_BOUND_REQUIREMENTS_ONLY_NO_INDEPENDENCE_CLAIM",
            **common_identity,
            "target_receipt_schema": "v21e3r1_external_algorithm_replay_receipt_v3",
            "required_path_bound_roles": [
                "reference_source_manifest",
                "external_source_manifest",
                "reference_event_stream",
                "external_event_stream",
                "neutral_comparison_receipt",
                "producer_authorship_authority_receipt",
                "independent_custody_authority_receipt",
                "external_execution_environment_receipt",
            ],
            "all_artifacts_require_path_sha256_and_payload_sha256": True,
            "producer_authorship_requires_external_authority": True,
            "custody_requires_external_authority": True,
            "external_producer_present": False,
            "independent_custody_authority_present": False,
            "implementation_code_disjoint_verified": False,
            "algorithm_execution_independence_verified": False,
            "gate_clearable_by_this_contract": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    external_binding = _write_json_exclusive(
        staging, "external-replay-v3.evidence-contract.json", external
    )
    phase = _payload_receipt(
        {
            "schema": "v21e3r1_path_bound_phase_evidence_contract_v3",
            "status": "FROZEN_FUTURE_PATH_BOUND_PHASE_CONTRACT_NO_LOCAL_EVIDENCE",
            "scope": "MACHINE_CHECKABLE_PHASE_REQUIREMENTS_ONLY_NO_PHASE_RESULT",
            **common_identity,
            "target_receipt_schema": "v21e3r1_path_bound_independent_phase_receipt_v3",
            "supported_phases": ["selection", "confirmation"],
            "required_common_path_bound_roles": [
                "study_freeze",
                "phase_manifest",
                "case_manifest",
                "matrix_receipt",
                "statistics_input",
                "statistics_source",
                "statistics_receipt",
                "execution_environment_receipt",
                "phase_producer_authority_receipt",
            ],
            "confirmation_additional_path_bound_roles": [
                "selection_receipt",
                "external_replay_receipt_v3",
                "independent_custody_authority_receipt",
            ],
            "selection_confirmation_case_disjointness_required": True,
            "prospective_chronology_required": True,
            "external_phase_producer_present": False,
            "independent_custody_authority_present": False,
            "gate_clearable_by_this_contract": False,
            "selection_result_materialized": False,
            "confirmation_result_materialized": False,
            "formal_study_materialized": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    phase_binding = _write_json_exclusive(
        staging, "path-bound-phase-v3.evidence-contract.json", phase
    )
    return external_binding, phase_binding


def _freeze_study_metric_spec(
    *, staging: Path, snapshots: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    def source_binding(relative: PurePosixPath) -> dict[str, object]:
        frozen = snapshots[relative.as_posix()]
        return {
            "path": relative.as_posix(),
            "bytes": frozen["bytes"],
            "sha256": frozen["sha256"],
        }

    receipt = _payload_receipt(
        {
            "schema": "v21e3r1_study_metric_spec_v1",
            "status": "FROZEN_BEFORE_SELECTION",
            "metric_id": "normalized_left_continuous_hypervolume_auc",
            "effect_direction": "LARGER_IS_BETTER",
            "evaluation_axis": "CHARGED_EVALUATIONS",
            "objective_dimension": 2,
            "normalization_contract": "CASE_FROZEN_LOWER_UPPER_AFFINE_TO_UNIT_SQUARE",
            "reference_point": [1.0, 1.0],
            "archive_contract": "ALL_CHARGED_EVALUATED_NONDOMINATED_ARCHIVE",
            "integration_contract": "EAUC=(1/B)*SUM_{b=1..B}HV(A_{b-1})",
            "primary_metric": "normalized_left_continuous_hypervolume_auc",
            "secondary_reporting_metrics": [
                "terminal_hypervolume",
                "attempt_count",
                "physical_start_count",
                "charged_evaluation_count",
                "wall_time_seconds",
                "peak_rss_bytes",
            ],
            "seed_within_case_aggregation": "ARITHMETIC_MEAN_WITHIN_CASE_ARM",
            "case_cluster_estimand": "MEAN_OF_PAIRED_CASE_DIFFERENCES",
            "row_crosscheck": {
                "required": True,
                "scope": "EVERY_FORMAL_STUDY_ROW",
                "tolerance": 0.0,
                "failure_policy": "HOLD_ON_ANY_MISMATCH",
            },
            "production_metric_source": source_binding(PRODUCTION_METRIC_RELATIVE),
            "independent_metric_source": source_binding(INDEPENDENT_METRIC_RELATIVE),
            "practical_thresholds_bound_in_simultaneous_spec": True,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    binding = _write_json_exclusive(staging, "study.metric-spec.json", receipt)
    return receipt, binding


def _freeze_simultaneous_spec(
    *,
    root: Path,
    staging: Path,
    snapshots: Mapping[str, Mapping[str, object]],
    study_id: str,
    candidate_id: str,
    successor_source_sha256: str,
    successor_config_sha256: str,
    study_metric_spec_sha256: str,
) -> dict[str, object]:
    source_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        SIM_EVALUATOR_RELATIVE,
        "simultaneous-inference-design/recompute-simultaneous-bounds.py",
    )
    test_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        SIM_TEST_RELATIVE,
        "simultaneous-inference-design/recompute-simultaneous-bounds.tests.py",
    )
    source_text = _contained_input(
        root, SIM_EVALUATOR_RELATIVE, label="simultaneous evaluator source"
    ).read_text(encoding="utf-8")
    required_source_tokens = (
        f'METHOD = "{SIMULTANEOUS_METHOD}"',
        'FAMILIES = ("MOKP", "MOTSP")',
        'CANDIDATES = ("C0", "C1", "C2", "C3")',
        '"domain": "v21e3r1-simultaneous-case-bootstrap-v1"',
        '"rng": "SHA256_COUNTER_U64_REJECTION_V1"',
        '"centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN"',
        '"studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR"',
    )
    missing = [token for token in required_source_tokens if token not in source_text]
    if missing:
        _fail(f"simultaneous evaluator source constants drifted: {missing}")
    receipt = _payload_receipt(
        {
            "schema": "v21e3r1_simultaneous_inference_spec_v2",
            "status": "PASS_FROZEN_BEFORE_SELECTION_ENGINEERING_ONLY",
            "scope": "FROZEN_PROSPECTIVE_DESIGN_ONLY_NO_CASE_MATERIALIZATION",
            "study_id": study_id,
            "candidate_id": candidate_id,
            "successor_source_sha256": successor_source_sha256,
            "successor_config_sha256": successor_config_sha256,
            "study_metric_spec_sha256": study_metric_spec_sha256,
            "evaluator_source_path": source_binding["path"],
            "evaluator_source_sha256": source_binding["sha256"],
            "evaluator_test_path": test_binding["path"],
            "evaluator_test_sha256": test_binding["sha256"],
            "method": SIMULTANEOUS_METHOD,
            "families": list(FAMILIES),
            "candidates": ["C0", "C1", "C2", "C3"],
            "familywise_alpha": 0.05,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "quantile_convention": "CEIL((1-ALPHA)*(B+1))_ORDER_STATISTIC",
            "critical_value_floor": 0.0,
            "rng_protocol": "SHA256_COUNTER_U64_REJECTION_V1",
            "rng_domain": "v21e3r1-simultaneous-case-bootstrap-v1",
            "cluster_unit": "PAIRED_CASE",
            "seed_aggregation": "MEAN_WITHIN_CASE_ARM",
            "resampling_rule": (
                "WITH_REPLACEMENT_INDEPENDENT_BY_FAMILY_"
                "SHARED_ACROSS_CELLS_WITHIN_FAMILY"
            ),
            "centering": "BOOTSTRAP_MEAN_MINUS_OBSERVED_MEAN",
            "studentization_denominator": "OBSERVED_CASE_CLUSTER_STANDARD_ERROR",
            "familywise_scope": "JOINT_ACROSS_BOTH_FAMILIES",
            "practical_thresholds": {
                "primary_effect": 0.0,
                "adjacent_mechanism_effect": 0.005,
            },
            "selection_cells": _selection_cells(),
            "selection_cell_count": 10,
            "confirmation_cells": _confirmation_cells(),
            "confirmation_cell_count": 4,
            "selection_and_confirmation_disjoint_by_construction": True,
            "frozen_before_selection": True,
            "selection_cases_materialized": False,
            "confirmation_cases_materialized": False,
            "formal_cases_materialized": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    _write_json_exclusive(staging, "simultaneous-inference.spec.json", receipt)
    return receipt


def _freeze_precedent(
    *,
    root: Path,
    staging: Path,
    snapshots: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    json_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        PRECEDENT_JSON_RELATIVE,
        "precedent-design/precedent-mechanism.matrix.json",
    )
    render_bindings = {
        role: _copy_snapshot(
            root,
            staging,
            snapshots,
            relative,
            f"precedent-design/precedent-mechanism.matrix.{Path(relative.as_posix()).suffix.lstrip('.')}",
        )
        for role, relative in PRECEDENT_RENDER_RELATIVES.items()
    }
    renderer_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        PRECEDENT_RENDERER_RELATIVE,
        "precedent-design/precedent-mechanism.renderer.py",
    )
    test_binding = _copy_snapshot(
        root,
        staging,
        snapshots,
        PRECEDENT_TEST_RELATIVE,
        "precedent-design/precedent-mechanism.tests.py",
    )
    matrix_path = _contained_input(root, PRECEDENT_JSON_RELATIVE, label="precedent matrix")
    matrix_raw = matrix_path.read_bytes()
    matrix = _parse_json(matrix_raw, label="precedent matrix")
    if matrix.get("schema") != "v21e3r1_precedent_mechanism_matrix_v1":
        _fail("precedent matrix schema drifted")
    if matrix.get("status") != "TARGETED_PRIMARY_SOURCE_MATRIX_NOT_SYSTEMATIC_REVIEW":
        _fail("precedent matrix is not explicitly targeted/not-systematic")
    components = matrix.get("components")
    methods = matrix.get("methods")
    positions = matrix.get("authorized_novelty_position")
    if type(components) is not list or len(components) != 16:
        _fail("precedent matrix must freeze exactly 16 components")
    if type(methods) is not list or len(methods) != 10:
        _fail("precedent matrix must freeze exactly 10 methods")
    if type(positions) is not list or len(positions) != 4:
        _fail("precedent matrix must freeze exactly four authorized positions")
    renderer_path = _contained_input(
        root, PRECEDENT_RENDERER_RELATIVE, label="precedent renderer source"
    )
    renderer = _load_module(renderer_path, "v21e3r1_bound_precedent_renderer")
    validated = renderer.load_precedent_matrix(matrix_path)
    expected_renderings = {
        "markdown": renderer.render_markdown(validated).encode("utf-8"),
        "csv": renderer.render_csv(validated).encode("utf-8"),
        "tex": renderer.render_latex(validated).encode("utf-8"),
    }
    byte_parity = True
    for role, raw in expected_renderings.items():
        source = _contained_input(
            root, PRECEDENT_RENDER_RELATIVES[role], label=f"precedent {role} rendering"
        ).read_bytes()
        byte_parity = byte_parity and source == raw
        if source.rstrip(b"\n") + b"\n" != raw.rstrip(b"\n") + b"\n":
            _fail(
                f"precedent {role} rendering content disagrees with the bound renderer"
            )
    receipt = _payload_receipt(
        {
            "schema": "v21e3r1_precedent_position_boundary_freeze_receipt_v1",
            "status": "PASS_TARGETED_PRIMARY_SOURCE_POSITION_FREEZE_ONLY__IJOC_HOLD",
            "scope": matrix.get("scope"),
            "review_scope": "TARGETED_NOT_SYSTEMATIC",
            "primary_source_cutoff_date": "2026-08-22",
            "matrix_json": json_binding,
            "matrix_json_payload_sha256": _sha256_bytes(_canonical_bytes(matrix)),
            "renderings": render_bindings,
            "renderer_source": renderer_binding,
            "renderer_test": test_binding,
            "component_count": len(components),
            "method_count": len(methods),
            "authorized_position_count": len(positions),
            "all_renderings_verified": True,
            "rendering_content_parity_after_trailing_newline_normalization": True,
            "cross_format_byte_parity_verified": byte_parity,
            "systematic_review": False,
            "novelty_priority_claim_authorized": False,
            "selection_authorized": False,
            "confirmation_authorized": False,
            "formal_study_authorized": False,
            "scientific_claim_authorized": False,
            "ijoc_submission_status": "IJOC_HOLD",
        }
    )
    _write_json_exclusive(staging, "precedent-mechanism.receipt.json", receipt)
    return receipt


def freeze_prospective_boundaries(
    *,
    repository_root: str | Path,
    output_directory: str | Path,
    study_id: str,
    candidate_id: str,
    successor_source_sha256: str,
    successor_config_sha256: str,
) -> dict[str, object]:
    """Create an exclusive, evaluator-consumable development boundary freeze."""

    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as error:
        _fail(f"repository root does not exist: {error}")
    if not root.is_dir():
        _fail("repository root is not a directory")
    study = _identifier(study_id, label="study_id")
    candidate = _identifier(candidate_id, label="candidate_id")
    successor_source = _sha256(
        successor_source_sha256, label="successor_source_sha256"
    )
    successor_config = _sha256(
        successor_config_sha256, label="successor_config_sha256"
    )
    output_input = Path(output_directory)
    if output_input.exists():
        _fail("output directory already exists; exclusive create required")
    try:
        output_parent = output_input.parent.resolve(strict=True)
    except OSError as error:
        _fail(f"output directory parent does not exist: {error}")
    output = output_parent / output_input.name
    if not output.name or output == output_parent:
        _fail("output directory must name a new child directory")
    staging = output_parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        staging.mkdir(exist_ok=False)
    except FileExistsError:
        _fail("exclusive staging directory collision")

    try:
        registry_path = _contained_input(root, REGISTRY_RELATIVE, label="registry")
        registry_raw = registry_path.read_bytes()
        registry = _parse_json(registry_raw, label="reference comparator registry")
        artifacts = registry.get("artifacts")
        if type(artifacts) is not list:
            _fail("reference comparator registry artifacts must be an array")
        registry_artifact_relatives: list[PurePosixPath] = []
        for index, value in enumerate(artifacts):
            if type(value) is not dict:
                _fail(f"registry artifacts[{index}] must be an object")
            registry_artifact_relatives.append(
                _canonical_relative(
                    value.get("path"), label=f"registry artifacts[{index}].path"
                )
            )
        fixed_relatives = [
            REGISTRY_RELATIVE,
            REGISTRY_VERIFIER_RELATIVE,
            REPLAY_SPEC_RELATIVE,
            REPLAY_COMPARATOR_RELATIVE,
            REPLAY_TEST_RELATIVE,
            *GOLDEN_RELATIVES.values(),
            SIM_EVALUATOR_RELATIVE,
            SIM_TEST_RELATIVE,
            PRODUCTION_METRIC_RELATIVE,
            INDEPENDENT_METRIC_RELATIVE,
            PRECEDENT_JSON_RELATIVE,
            *PRECEDENT_RENDER_RELATIVES.values(),
            PRECEDENT_RENDERER_RELATIVE,
            PRECEDENT_TEST_RELATIVE,
            *registry_artifact_relatives,
        ]
        snapshots = _snapshot_inputs(root, fixed_relatives)
        verifier_path = _contained_input(
            root, REGISTRY_VERIFIER_RELATIVE, label="registry verifier"
        )
        verifier = _load_module(verifier_path, "v21e3r1_bound_registry_verifier")
        try:
            verification = verifier.verify_registry(registry_path, root)
        except Exception as error:
            _fail(f"strict offline registry verification failed: {error}")
        if type(verification) is not dict:
            _fail("registry verifier did not return an exact receipt object")
        if verification.get("network_calls") != 0:
            _fail("registry verification did not prove zero network calls")
        if verification.get("external_family_native_strong_baseline_count") != 0:
            _fail("registry verifier unexpectedly found a strong external baseline")

        _study_metric_receipt, study_metric_binding = _freeze_study_metric_spec(
            staging=staging,
            snapshots=snapshots,
        )
        study_metric = str(study_metric_binding["sha256"])

        candidate_menu_receipt, candidate_menu_binding = (
            _freeze_candidate_menu_contract(staging=staging, study_id=study)
        )
        external_v3_contract_binding, phase_v3_contract_binding = (
            _freeze_future_v3_evidence_contracts(
                staging=staging,
                study_id=study,
                candidate_id=candidate,
                successor_source_sha256=successor_source,
                successor_config_sha256=successor_config,
                study_metric_spec_sha256=study_metric,
            )
        )

        baseline_receipt, baseline_counts = _freeze_baselines(
            root=root,
            staging=staging,
            snapshots=snapshots,
            registry=registry,
            verification=verification,
            study_id=study,
            candidate_id=candidate,
            study_metric_spec_sha256=study_metric,
        )
        external_receipt = _freeze_external_replay(
            root=root,
            staging=staging,
            snapshots=snapshots,
            study_id=study,
            candidate_id=candidate,
            successor_source_sha256=successor_source,
            successor_config_sha256=successor_config,
            study_metric_spec_sha256=study_metric,
        )
        simultaneous_receipt = _freeze_simultaneous_spec(
            root=root,
            staging=staging,
            snapshots=snapshots,
            study_id=study,
            candidate_id=candidate,
            successor_source_sha256=successor_source,
            successor_config_sha256=successor_config,
            study_metric_spec_sha256=study_metric,
        )
        precedent_receipt = _freeze_precedent(
            root=root, staging=staging, snapshots=snapshots
        )
        _verify_snapshots_stable(root, snapshots)
        snapshot_entries = [snapshots[key] for key in sorted(snapshots, key=str.casefold)]
        snapshot_receipt = _payload_receipt(
            {
                "schema": "v21e3r1_prospective_boundary_input_snapshot_v1",
                "repository_root_name": root.name,
                "entries": snapshot_entries,
                "entry_count": len(snapshot_entries),
                "snapshot_root_sha256": _sha256_bytes(_canonical_bytes(snapshot_entries)),
                "toctou_verification_passed": True,
                "network_calls": 0,
            }
        )
        snapshot_binding = _write_json_exclusive(
            staging, "input-snapshot.receipt.json", snapshot_receipt
        )
        boundary_bindings = {
            name: _binding(path, (staging / path).read_bytes())
            for name, path in (
                ("baseline_registry", "baseline-registry.receipt.json"),
                ("external_algorithm_replay", "external-algorithm-replay.receipt.json"),
                ("simultaneous_inference_spec", "simultaneous-inference.spec.json"),
                ("successor_candidate_menu", "successor-candidate-menu.contract.json"),
                ("external_replay_v3_contract", "external-replay-v3.evidence-contract.json"),
                ("path_bound_phase_v3_contract", "path-bound-phase-v3.evidence-contract.json"),
                ("precedent_mechanism", "precedent-mechanism.receipt.json"),
            )
        }
        master = _payload_receipt(
            {
                "schema": "v21e3r1_prospective_boundary_freeze_receipt_v1",
                "status": "PASS_PROSPECTIVE_BOUNDARIES_FROZEN_ENGINEERING_ONLY",
                "scope": "PROSPECTIVE_BOUNDARY_BINDINGS_ONLY_NO_CASE_GENERATION_OR_SCIENTIFIC_AUTHORITY",
                "study_id": study,
                "candidate_id": candidate,
                "successor_source_sha256": successor_source,
                "successor_config_sha256": successor_config,
                "study_metric_spec_sha256": study_metric,
                "study_metric_spec": study_metric_binding,
                "input_snapshot": snapshot_binding,
                "boundary_receipts": boundary_bindings,
                "input_artifact_count": len(snapshot_entries),
                "toctou_verification_passed": True,
                "external_family_native_strong_baseline_count_by_family": baseline_counts,
                "reference_snapshot_frozen": baseline_receipt[
                    "all_registry_artifacts_verified"
                ],
                "external_replay_design_frozen": external_receipt["event_streams_match"],
                "simultaneous_inference_spec_frozen": simultaneous_receipt[
                    "frozen_before_selection"
                ],
                "successor_candidate_menu_frozen": (
                    candidate_menu_receipt["status"]
                    == "HOLD_SUCCESSOR_CANDIDATE_ABSENT_FROM_SIMULTANEOUS_SPEC_V2"
                    and candidate_menu_receipt[
                        "selection_prohibited_until_revised_simultaneous_spec"
                    ]
                    is True
                    and candidate_menu_binding["sha256"]
                    == boundary_bindings["successor_candidate_menu"]["sha256"]
                ),
                "external_replay_v3_contract_frozen": (
                    external_v3_contract_binding["sha256"]
                    == boundary_bindings["external_replay_v3_contract"]["sha256"]
                ),
                "path_bound_phase_v3_contract_frozen": (
                    phase_v3_contract_binding["sha256"]
                    == boundary_bindings["path_bound_phase_v3_contract"]["sha256"]
                ),
                "precedent_position_frozen": precedent_receipt[
                    "all_renderings_verified"
                ],
                "independent_producer": False,
                "independent_custody": False,
                "implementation_code_disjoint": False,
                "algorithm_execution_independence": False,
                "selection_authorized": False,
                "confirmation_authorized": False,
                "formal_study_authorized": False,
                "scientific_claim_authorized": False,
                "case_generation_performed": False,
                "generated_case_count": 0,
                "network_calls": 0,
                "ijoc_submission_status": "IJOC_HOLD",
            }
        )
        _write_json_exclusive(
            staging, "prospective-boundary.freeze.receipt.json", master
        )
        _verify_snapshots_stable(root, snapshots)
        if output.exists():
            _fail("output directory appeared during freeze; exclusive create required")
        os.rename(staging, output)
        return master
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--successor-source-sha256", required=True)
    parser.add_argument("--successor-config-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        receipt = freeze_prospective_boundaries(
            repository_root=arguments.repository_root,
            output_directory=arguments.output_directory,
            study_id=arguments.study_id,
            candidate_id=arguments.candidate_id,
            successor_source_sha256=arguments.successor_source_sha256,
            successor_config_sha256=arguments.successor_config_sha256,
        )
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except (FreezeError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_prospective_boundary_freeze_error_v1",
                    "status": "HOLD_INTEGRITY_ERROR",
                    "error": str(error),
                    "selection_authorized": False,
                    "confirmation_authorized": False,
                    "formal_study_authorized": False,
                    "scientific_claim_authorized": False,
                    "ijoc_submission_status": "IJOC_HOLD",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
