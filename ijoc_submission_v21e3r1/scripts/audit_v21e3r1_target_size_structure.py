from __future__ import annotations

"""Validate target-size development parity inputs before source freezing.

This gate checks only frozen engineering structure.  It neither consumes a
source snapshot nor authorizes the matched matrix, selection, calibration, or
formal execution.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "ijoc_submission_v21e3"
    / "protocol"
    / "V21E3_C0_PARITY_PROTOCOL_V2.json"
)
DEFAULT_MANIFEST_ROOT = (
    REPO_ROOT / "ijoc_submission_v21e3" / "development_manifests_v1"
)
DEFAULT_CASE_MANIFEST = (
    REPO_ROOT
    / "ijoc_submission_v21e3"
    / "development_partitions_v1"
    / "case_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "ijoc_submission_v21e3r1"
    / "provenance"
    / "V21E3R1_TARGET_SIZE_INPUT_STRUCTURE_RECEIPT_V1.json"
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"Evidence path escapes repository root: {path}") from error
    return resolved


def _load(root: Path, path: Path) -> tuple[Path, bytes, dict[str, object]]:
    resolved = _inside(root, path)
    raw = resolved.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return resolved, raw, value


def _binding(root: Path, path: Path, raw: bytes) -> dict[str, object]:
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _verify_binding(root: Path, binding: object, expected_path: Path) -> None:
    if not isinstance(binding, Mapping):
        raise ValueError("A required manifest binding is absent.")
    path = _inside(root, root / str(binding.get("path", "")))
    if path != expected_path.resolve():
        raise ValueError("A manifest binding names the wrong path.")
    raw = path.read_bytes()
    if binding.get("bytes") != len(raw) or binding.get("sha256") != _sha256(raw):
        raise ValueError(f"A manifest binding drifted: {path}")


def _validate_protocol(protocol: Mapping[str, object]) -> None:
    if protocol.get("schema") != "pareto_v21e3_c0_parity_protocol_v2":
        raise ValueError("Unexpected parity protocol schema.")
    if protocol.get("status") != (
        "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING"
    ):
        raise ValueError("Structural audit requires the pending-snapshot protocol.")
    if protocol.get("successor_version") != "V21e3r1":
        raise ValueError("The protocol does not name V21e3r1.")
    if protocol.get("families") != ["MOTSP", "MOKP"]:
        raise ValueError("The protocol must cover MOTSP and MOKP.")
    common = protocol.get("common_execution_contract")
    if not isinstance(common, Mapping) or not (
        common.get("charged_evaluation_budget") == 2_000
        and common.get("checkpoint_period") == 200
    ):
        raise ValueError("The target-size budget/checkpoint contract drifted.")
    arms = protocol.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {
        "V21E3_C0",
        "NSGAII",
        "MOEAD",
    }:
        raise ValueError("The exact ordered three-arm development matrix is required.")
    if any(
        not isinstance(arms[arm], Mapping)
        or arms[arm].get("execution_adapter_status")
        != "DEVELOPMENT_ONLY_AVAILABLE"
        for arm in arms
    ):
        raise ValueError("Every development parity adapter must be available.")
    design = protocol.get("case_design")
    if not isinstance(design, Mapping) or not (
        design.get("case_count_per_family") == 6
        and design.get("sizes") == [100, 200, 500]
        and design.get("cases_per_size_per_family") == 2
        and design.get("seeds") == [31051, 31057, 31059]
    ):
        raise ValueError("The case/seed design drifted.")
    gates = protocol.get("preflight_gates")
    if not isinstance(gates, Mapping) or not (
        gates.get("successor_source_snapshot") == "PENDING"
        and gates.get("independent_protocol_preflight") == "NOT_RUN"
        and gates.get("matched_matrix") == "NOT_RUN"
        and gates.get("selection_entropy_release") == "PROHIBITED"
        and gates.get("calibration_execution") == "PROHIBITED"
        and gates.get("formal_execution") == "PROHIBITED"
        and gates.get("formal_authorized") is False
    ):
        raise ValueError("The pending protocol opened a later-stage gate.")


def audit_target_size_structure(
    *,
    repo_root: Path,
    protocol_path: Path,
    case_manifest_path: Path,
    reference_manifest_path: Path,
    config_manifest_path: Path,
    metric_manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    """Validate the pre-freeze target-size structure and write one receipt."""

    root = repo_root.resolve()
    destination = _inside(root, output)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace structural receipt: {destination}")
    protocol_path, protocol_raw, protocol = _load(root, protocol_path)
    case_path, case_raw, case_manifest = _load(root, case_manifest_path)
    reference_path, reference_raw, reference = _load(root, reference_manifest_path)
    config_path, config_raw, config = _load(root, config_manifest_path)
    metric_path, metric_raw, metric = _load(root, metric_manifest_path)

    _validate_protocol(protocol)
    design = protocol["case_design"]
    assert isinstance(design, Mapping)
    if _inside(root, root / str(design.get("manifest", ""))) != case_path:
        raise ValueError("The protocol binds another case manifest.")

    if not (
        case_manifest.get("schema") == "pareto_v21_partition_manifest_v1"
        and case_manifest.get("split") == "development"
        and case_manifest.get("formal_confirmatory_eligibility") is False
    ):
        raise ValueError("The case manifest is not development-only.")
    cases = case_manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("The target-size case manifest must contain 12 cases.")
    identifiers: list[str] = []
    distribution: Counter[tuple[object, object]] = Counter()
    for case in cases:
        if not isinstance(case, Mapping) or case.get("split") != "development":
            raise ValueError("Every target-size case must be development-only.")
        identifiers.append(str(case.get("case_id", "")))
        distribution[(case.get("family"), case.get("size"))] += 1
    if "" in identifiers or len(set(identifiers)) != 12:
        raise ValueError("Target-size case identifiers must be unique and nonempty.")
    expected_distribution = Counter(
        {(family, size): 2 for family in ("MOTSP", "MOKP") for size in (100, 200, 500)}
    )
    if distribution != expected_distribution:
        raise ValueError("The target-size family/size distribution drifted.")

    if not (
        reference.get("schema") == "pareto_v21e3_analytic_reference_manifest_v1"
        and reference.get("status") == "FROZEN_DEVELOPMENT_ONLY"
        and reference.get("split") == "development"
        and reference.get("formal_use") == "NOT_AUTHORIZED"
        and reference.get("case_count") == 12
    ):
        raise ValueError("The analytic reference manifest fails closed.")
    _verify_binding(root, reference.get("partition_manifest"), case_path)
    reference_cases = reference.get("cases")
    if not isinstance(reference_cases, list) or {
        (
            item.get("case_id"),
            item.get("family"),
            item.get("size"),
        )
        for item in reference_cases
        if isinstance(item, Mapping)
    } != {
        (case["case_id"], case["family"], case["size"])
        for case in cases
        if isinstance(case, Mapping)
    }:
        raise ValueError("Reference and case manifests disagree.")

    if not (
        config.get("schema") == "pareto_v21e3_development_config_manifest_v1"
        and config.get("status")
        == "FROZEN_DEVELOPMENT_INPUT_CALIBRATION_EXECUTION_BLOCKED"
        and config.get("selection_partition") == "NOT_GENERATED"
        and config.get("calibration_confirmation_partition") == "NOT_GENERATED"
        and config.get("calibration_execution_authorized") is False
        and config.get("formal_cases") == "NOT_MATERIALIZED"
        and config.get("formal_execution_authorized") is False
        and config.get("reference_manifest") == reference_path.name
        and config.get("metric_manifest") == metric_path.name
    ):
        raise ValueError("The development configuration manifest fails closed.")
    directions = config.get("reference_directions")
    if not isinstance(directions, list) or len(directions) != 21:
        raise ValueError("The candidate direction grid must contain 21 directions.")
    direction_contract = protocol.get("candidate_reference_directions")
    if not isinstance(direction_contract, Mapping) or not (
        direction_contract.get("count") == 21
        and direction_contract.get("source_field") == "reference_directions"
    ):
        raise ValueError("The candidate direction contract drifted.")
    _verify_binding(root, direction_contract.get("source_binding"), config_path)

    if not (
        metric.get("schema") == "pareto_v21e3_metric_manifest_v1"
        and metric.get("status")
        == "FROZEN_DEVELOPMENT_AND_FUTURE_CALIBRATION_INPUT"
        and metric.get("formal_use") == "NOT_AUTHORIZED"
        and metric.get("selection_grid")
        == {"charged_budget": 2_000, "checkpoint_period": 200}
    ):
        raise ValueError("The metric manifest changed the development grid.")

    bindings = {
        "protocol": _binding(root, protocol_path, protocol_raw),
        "case_manifest": _binding(root, case_path, case_raw),
        "reference_manifest": _binding(root, reference_path, reference_raw),
        "config_manifest": _binding(root, config_path, config_raw),
        "metric_manifest": _binding(root, metric_path, metric_raw),
    }
    receipt: dict[str, object] = {
        "schema": "pareto_v21e3r1_target_size_input_structure_receipt_v1",
        "status": "PASS_TARGET_SIZE_INPUT_STRUCTURE_ENGINEERING_ONLY",
        "scientific_scope": (
            "pre_freeze_input_structure_not_execution_or_performance_evidence"
        ),
        "bindings": bindings,
        "families": ["MOTSP", "MOKP"],
        "target_sizes": {"MOTSP": [100, 200, 500], "MOKP": [100, 200, 500]},
        "case_count": 12,
        "seeds": [31051, 31057, 31059],
        "arms": ["V21E3_C0", "NSGAII", "MOEAD"],
        "charged_evaluation_budget": 2_000,
        "checkpoint_period": 200,
        "development_parity_execution": "NOT_AUTHORIZED_BY_THIS_RECEIPT",
        "selection_entropy_release": "PROHIBITED",
        "calibration_execution": "PROHIBITED",
        "formal_execution": "PROHIBITED",
        "formal_authorized": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(_canonical_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT / "reference_manifest_development.json",
    )
    parser.add_argument(
        "--config-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT / "config_manifest_development.json",
    )
    parser.add_argument(
        "--metric-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT / "metric_manifest.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = audit_target_size_structure(
        repo_root=args.repo_root,
        protocol_path=args.protocol,
        case_manifest_path=args.case_manifest,
        reference_manifest_path=args.reference_manifest,
        config_manifest_path=args.config_manifest,
        metric_manifest_path=args.metric_manifest,
        output=args.output,
    )
    print(json.dumps(
        {
            "status": receipt["status"],
            "case_count": receipt["case_count"],
            "output": str(args.output.resolve()),
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
