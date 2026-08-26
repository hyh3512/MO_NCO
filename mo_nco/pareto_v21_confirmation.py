from __future__ import annotations

"""Fail-closed V21 independent-calibration confirmation gate.

This module is intentionally separate from candidate selection.  It consumes a
completed confirmation ``run_rows.jsonl`` and its matrix receipt, verifies the
exact prospective matrix and trace/partition bindings, and only then evaluates
the frozen case-cluster decision rules.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .pareto_v21_diagnostics import compare_paired_cluster_metric


_FAMILIES = ("MOKP", "MOTSP")
_HEX_DIGITS = frozenset("0123456789abcdef")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and set(text.lower()) <= _HEX_DIGITS


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _require_positive_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a positive integer.") from error
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{key} must be a positive integer.")
    return parsed


def _family_thresholds(
    raw: object,
    *,
    key: str,
    strictly_positive: bool,
) -> dict[str, float]:
    if isinstance(raw, Mapping):
        if set(raw) != set(_FAMILIES):
            raise ValueError(f"{key} must bind exactly MOKP and MOTSP.")
        values = {family: float(raw[family]) for family in _FAMILIES}
    else:
        try:
            scalar = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be numeric or family-specific.") from error
        values = {family: scalar for family in _FAMILIES}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"{key} must contain finite values.")
    if strictly_positive and any(value <= 0.0 for value in values.values()):
        raise ValueError(f"{key} must be strictly positive for each family.")
    return values


@dataclass(frozen=True)
class _FrozenGate:
    precommit_schema: str
    selected_candidate: str
    control_id: str
    predecessor_id: str
    manifest_sha256: str
    families: tuple[str, ...]
    cases_per_family: int
    seeds: tuple[int, ...]
    evaluation_budget: int
    checkpoint_period: int
    metric: str
    delta_min: dict[str, float]
    noninferiority_margin: dict[str, float]
    mechanism_noninferiority_margin: dict[str, float]
    bootstrap_samples: int
    randomization_seed: int
    tie_tolerance: float
    trim_fraction: float


def _load_frozen_gate(
    path: Path,
    *,
    selected_candidate: str,
    control_id: str,
    predecessor_id: str,
) -> tuple[_FrozenGate, bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("The confirmation precommit must be a JSON object.")
    schema = str(payload.get("schema", ""))
    if schema == "pareto_v21_confirmation_gate_precommit_v1":
        if payload.get("status") != "FROZEN_BEFORE_CONFIRMATION_RUNS":
            raise ValueError(
                "The V21 confirmation thresholds are not prospectively frozen."
            )
        normalized: Mapping[str, object] = payload
    elif schema == "pareto_v21_candidate_menu_precommit_v2":
        if payload.get("status") != "FROZEN_BEFORE_SELECTION_RUNS":
            raise ValueError("The V21e2 candidate menu is not prospectively frozen.")
        selection_gate = payload.get("selection_gate")
        confirmation = payload.get("confirmation_design_if_authorized")
        if not isinstance(selection_gate, Mapping) or not isinstance(
            confirmation, Mapping
        ):
            raise ValueError("The V21e2 precommit omits its frozen gate design.")
        if confirmation.get("gate_thresholds_identical_to_selection") is not True:
            raise ValueError("Confirmation thresholds are not bound to selection.")
        candidate_ids = tuple(
            str(entry.get("candidate_id", ""))
            for entry in payload.get("candidate_menu", ())
            if isinstance(entry, Mapping)
        )
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("The V21e2 candidate menu is invalid.")
        if selected_candidate not in candidate_ids or control_id not in candidate_ids:
            raise ValueError("Selected candidate/control is absent from the frozen menu.")
        if str(selection_gate.get("control_id", "")) != control_id:
            raise ValueError("The requested control differs from the frozen C0 control.")
        try:
            candidate_rank = int(selected_candidate.removeprefix("C"))
        except ValueError as error:
            raise ValueError("V21 adjacent gates require a Ck candidate ID.") from error
        expected_predecessor = control_id if candidate_rank == 1 else f"C{candidate_rank - 1}"
        if predecessor_id != expected_predecessor or predecessor_id not in candidate_ids:
            raise ValueError("The requested predecessor is not the frozen adjacent arm.")
        partition_binding = confirmation.get("partition_manifest")
        if not isinstance(partition_binding, Mapping):
            raise ValueError("The V21e2 confirmation partition is not hash-bound.")
        normalized = {
            "selected_candidate": selected_candidate,
            "control_id": control_id,
            "predecessor_id": predecessor_id,
            "confirmation_manifest_sha256": partition_binding.get("sha256"),
            "expected_families": _FAMILIES,
            "expected_cases_per_family": confirmation.get(
                "expected_cases_per_family"
            ),
            "expected_seeds": confirmation.get("seeds"),
            "expected_evaluation_budget": confirmation.get("evaluation_budget"),
            "expected_checkpoint_period": confirmation.get("checkpoint_period"),
            "primary_metric": "normalized_hv_auc",
            "delta_min": selection_gate.get("delta_min"),
            "noninferiority_margin": selection_gate.get(
                "noninferiority_margin"
            ),
            "mechanism_noninferiority_margin": selection_gate.get(
                "mechanism_noninferiority_margin"
            ),
            "bootstrap_samples": confirmation.get("bootstrap_samples"),
            "bootstrap_randomization_seed": confirmation.get(
                "bootstrap_randomization_seed"
            ),
            "tie_tolerance": selection_gate.get("tie_tolerance"),
            "trim_fraction_each_tail": selection_gate.get(
                "trim_fraction_each_tail"
            ),
        }
    else:
        raise ValueError("Unsupported V21 confirmation precommit schema.")
    families = tuple(str(value) for value in normalized.get("expected_families", ()))
    if families != _FAMILIES:
        raise ValueError("The confirmation precommit must bind MOKP and MOTSP.")
    seeds = tuple(int(value) for value in normalized.get("expected_seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("The confirmation seed list must be nonempty and unique.")
    manifest_sha = str(normalized.get("confirmation_manifest_sha256", ""))
    if not _is_sha256(manifest_sha):
        raise ValueError("The confirmation manifest SHA-256 is not frozen.")
    selected = str(normalized.get("selected_candidate", ""))
    control = str(normalized.get("control_id", ""))
    predecessor = str(normalized.get("predecessor_id", ""))
    if not selected or not control or not predecessor:
        raise ValueError("Candidate, control, and predecessor IDs must be frozen.")
    if selected == control or selected == predecessor:
        raise ValueError("The selected candidate must differ from both comparators.")
    metric = str(normalized.get("primary_metric", ""))
    if metric != "normalized_hv_auc":
        raise ValueError("V21 confirmation v1 requires normalized_hv_auc.")
    tie_tolerance = float(normalized.get("tie_tolerance"))
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and nonnegative.")
    trim_fraction = float(normalized.get("trim_fraction_each_tail"))
    if trim_fraction != 0.10:
        raise ValueError("V21 comparator v1 freezes a ten-percent trimmed mean.")
    budget = _require_positive_int(normalized, "expected_evaluation_budget")
    checkpoint = _require_positive_int(normalized, "expected_checkpoint_period")
    if budget % checkpoint:
        raise ValueError("The frozen checkpoint period must divide the budget.")
    randomization_seed = normalized.get("bootstrap_randomization_seed")
    if isinstance(randomization_seed, bool):
        raise ValueError("bootstrap_randomization_seed must be an integer.")
    try:
        randomization_seed = int(randomization_seed)
    except (TypeError, ValueError) as error:
        raise ValueError("bootstrap_randomization_seed must be an integer.") from error
    return (
        _FrozenGate(
            precommit_schema=schema,
            selected_candidate=selected,
            control_id=control,
            predecessor_id=predecessor,
            manifest_sha256=manifest_sha.lower(),
            families=families,
            cases_per_family=_require_positive_int(
                normalized, "expected_cases_per_family"
            ),
            seeds=seeds,
            evaluation_budget=budget,
            checkpoint_period=checkpoint,
            metric=metric,
            delta_min=_family_thresholds(
                normalized.get("delta_min"),
                key="delta_min",
                strictly_positive=True,
            ),
            noninferiority_margin=_family_thresholds(
                normalized.get("noninferiority_margin"),
                key="noninferiority_margin",
                strictly_positive=True,
            ),
            mechanism_noninferiority_margin=_family_thresholds(
                normalized.get("mechanism_noninferiority_margin"),
                key="mechanism_noninferiority_margin",
                strictly_positive=True,
            ),
            bootstrap_samples=_require_positive_int(
                normalized, "bootstrap_samples"
            ),
            randomization_seed=randomization_seed,
            tie_tolerance=tie_tolerance,
            trim_fraction=trim_fraction,
        ),
        raw,
    )


def _load_rows(path: Path) -> tuple[list[dict[str, object]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"run_rows line {line_number} is not a JSON object.")
        rows.append(payload)
    if not rows:
        raise ValueError("run_rows.jsonl is empty.")
    return rows, raw


def _resolve_bound_path(base: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _audit_inputs(
    *,
    rows_path: Path,
    rows: Sequence[Mapping[str, object]],
    rows_bytes: bytes,
    matrix_receipt_path: Path,
    frozen: _FrozenGate,
    selected_candidate: str,
    control_id: str,
    predecessor_id: str,
) -> dict[str, object]:
    identity_ok = (
        selected_candidate == frozen.selected_candidate
        and control_id == frozen.control_id
        and predecessor_id == frozen.predecessor_id
    )
    arms = tuple(dict.fromkeys((selected_candidate, control_id, predecessor_id)))
    expected_rows = (
        len(frozen.families)
        * frozen.cases_per_family
        * len(frozen.seeds)
        * len(arms)
    )

    matrix_errors: list[str] = []
    matrix_completeness_errors: list[str] = []
    matrix_trace_errors: list[str] = []
    matrix_partition_errors: list[str] = []
    matrix_run_context_errors: list[str] = []
    matrix: dict[str, object] = {}
    matrix_bytes = b""
    if not matrix_receipt_path.is_file():
        matrix_errors.append("matrix_receipt_missing")
        matrix_completeness_errors.append("matrix_receipt_missing")
        matrix_trace_errors.append("matrix_receipt_missing")
        matrix_partition_errors.append("matrix_receipt_missing")
        matrix_run_context_errors.append("matrix_receipt_missing")
    else:
        try:
            matrix_bytes = matrix_receipt_path.read_bytes()
            parsed = json.loads(matrix_bytes)
            if not isinstance(parsed, dict):
                raise ValueError("matrix receipt is not an object")
            matrix = parsed
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            matrix_errors.append("matrix_receipt_invalid_json")
            matrix_completeness_errors.append("matrix_receipt_invalid_json")
            matrix_trace_errors.append("matrix_receipt_invalid_json")
            matrix_partition_errors.append("matrix_receipt_invalid_json")
            matrix_run_context_errors.append("matrix_receipt_invalid_json")

    manifest: dict[str, object] = {}
    manifest_bytes = b""
    manifest_path: Path | None = None
    if matrix:
        required_matrix = {
            "schema": matrix.get("schema")
            == "pareto_v21_calibration_matrix_receipt_v1",
            "status": matrix.get("status") == "PASS",
            "split": matrix.get("split") == "confirmation",
            "rows_sha256": matrix.get("rows_sha256") == _sha256_bytes(rows_bytes),
            "expected_rows": matrix.get("expected_rows") == expected_rows,
            "completed_rows": matrix.get("completed_rows") == expected_rows,
            "candidate_ids": set(map(str, matrix.get("candidate_ids", ())))
            == set(arms),
            "seeds": set(map(int, matrix.get("seeds", ()))) == set(frozen.seeds),
            "evaluation_budget": matrix.get("evaluation_budget")
            == frozen.evaluation_budget,
            "checkpoint_period": matrix.get("checkpoint_period")
            == frozen.checkpoint_period,
            "all_trace_verifications_pass": matrix.get(
                "all_trace_verifications_pass"
            )
            is True,
            "full_partition_binding_gate": matrix.get(
                "full_partition_binding_gate"
            )
            == "PASS",
            "run_context_binding_gate": matrix.get("run_context_binding_gate")
            == "PASS",
            "run_context_sha256": _is_sha256(
                matrix.get("run_context_sha256")
            ),
            "manifest_sha256": str(matrix.get("manifest_sha256", "")).lower()
            == frozen.manifest_sha256,
        }
        for key, passed in required_matrix.items():
            if passed:
                continue
            error = f"matrix_{key}_mismatch"
            matrix_errors.append(error)
            if key in {
                "rows_sha256",
                "expected_rows",
                "completed_rows",
                "candidate_ids",
                "seeds",
                "evaluation_budget",
                "checkpoint_period",
            }:
                matrix_completeness_errors.append(error)
            if key == "all_trace_verifications_pass":
                matrix_trace_errors.append(error)
            if key in {"run_context_binding_gate", "run_context_sha256"}:
                matrix_run_context_errors.append(error)
            if key in {
                "schema",
                "split",
                "full_partition_binding_gate",
                "manifest_sha256",
            }:
                matrix_partition_errors.append(error)
        try:
            manifest_path = _resolve_bound_path(
                matrix_receipt_path.parent, matrix["manifest_path"]
            )
            manifest_bytes = manifest_path.read_bytes()
            parsed_manifest = json.loads(manifest_bytes)
            if not isinstance(parsed_manifest, dict):
                raise ValueError("manifest is not an object")
            manifest = parsed_manifest
        except (KeyError, OSError, UnicodeError, json.JSONDecodeError, ValueError):
            matrix_errors.append("bound_manifest_unreadable")
            matrix_partition_errors.append("bound_manifest_unreadable")

    manifest_cases: dict[str, str] = {}
    partition_errors: list[str] = list(matrix_partition_errors)
    if manifest:
        if _sha256_bytes(manifest_bytes) != frozen.manifest_sha256:
            partition_errors.append("bound_manifest_sha256_mismatch")
        if manifest.get("schema") != "pareto_v21_partition_manifest_v1":
            partition_errors.append("bound_manifest_schema_mismatch")
        if manifest.get("split") != "confirmation":
            partition_errors.append("bound_manifest_split_mismatch")
        for case in manifest.get("cases", ()):
            try:
                case_id = str(case["case_id"])
                family = str(case["family"])
            except (KeyError, TypeError):
                partition_errors.append("bound_manifest_case_invalid")
                continue
            if case_id in manifest_cases:
                partition_errors.append("bound_manifest_duplicate_case")
            manifest_cases[case_id] = family
        for family in frozen.families:
            if sum(value == family for value in manifest_cases.values()) != (
                frozen.cases_per_family
            ):
                partition_errors.append(
                    f"bound_manifest_{family.lower()}_case_count_mismatch"
                )

    row_errors: list[str] = list(matrix_completeness_errors)
    trace_errors: list[str] = list(matrix_trace_errors)
    run_context_errors: list[str] = list(matrix_run_context_errors)
    observed_keys: list[tuple[str, str, str, int]] = []
    observed_case_family: dict[str, str] = {}
    trace_paths: set[Path] = set()
    rows_root = rows_path.parent.resolve()
    matrix_context_sha256 = str(matrix.get("run_context_sha256", "")).lower()
    for row_index, row in enumerate(rows):
        prefix = f"row_{row_index + 1}"
        try:
            family = str(row["family"])
            case_id = str(row["case_id"])
            arm = str(row["candidate_id"])
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError):
            row_errors.append(f"{prefix}_identity_invalid")
            continue
        observed_keys.append((family, case_id, arm, seed))
        if case_id in observed_case_family and observed_case_family[case_id] != family:
            row_errors.append(f"{prefix}_case_family_inconsistent")
        observed_case_family[case_id] = family
        if row.get("schema") != "pareto_v21_calibration_run_row_v1":
            row_errors.append(f"{prefix}_schema_mismatch")
        if row.get("split") != "confirmation":
            row_errors.append(f"{prefix}_split_mismatch")
        if family not in frozen.families or arm not in arms or seed not in frozen.seeds:
            row_errors.append(f"{prefix}_unexpected_run_key")
        if row.get("evaluation_budget") != frozen.evaluation_budget:
            row_errors.append(f"{prefix}_budget_mismatch")
        if row.get("checkpoint_period") != frozen.checkpoint_period:
            row_errors.append(f"{prefix}_checkpoint_period_mismatch")
        try:
            metric = float(row[frozen.metric])
            if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            row_errors.append(f"{prefix}_metric_invalid")
        if row.get("trace_verification_status") != "PASS":
            trace_errors.append(f"{prefix}_trace_status_not_pass")
        if str(row.get("run_context_sha256", "")).lower() != matrix_context_sha256:
            run_context_errors.append(f"{prefix}_run_context_sha256_mismatch")
        for key in (
            "trace_database_sha256",
            "terminal_evaluation_chain_sha256",
            "terminal_decision_chain_sha256",
            "terminal_mechanism_chain_sha256",
        ):
            if not _is_sha256(row.get(key)):
                trace_errors.append(f"{prefix}_{key}_invalid")
        try:
            relative = Path(str(row["trace_relative_path"]))
            if relative.is_absolute():
                raise ValueError
            trace_path = (rows_root / relative).resolve()
            if not trace_path.is_relative_to(rows_root):
                raise ValueError
            if trace_path in trace_paths:
                trace_errors.append(f"{prefix}_trace_path_reused")
            trace_paths.add(trace_path)
            trace_bytes = trace_path.read_bytes()
            if _sha256_bytes(trace_bytes) != str(row["trace_database_sha256"]).lower():
                trace_errors.append(f"{prefix}_trace_file_sha256_mismatch")
        except (KeyError, OSError, ValueError):
            trace_errors.append(f"{prefix}_trace_file_unbound")

    expected_keys = {
        (family, case_id, arm, seed)
        for case_id, family in manifest_cases.items()
        for arm in arms
        for seed in frozen.seeds
    }
    observed_key_set = set(observed_keys)
    if len(observed_keys) != len(observed_key_set):
        row_errors.append("duplicate_run_key")
    if len(rows) != expected_rows:
        row_errors.append("row_count_mismatch")
    if expected_keys != observed_key_set:
        row_errors.append("expected_run_key_set_mismatch")
    if observed_case_family != manifest_cases:
        row_errors.append("row_case_partition_mismatch")

    completeness_pass = not row_errors
    trace_pass = not trace_errors
    partition_pass = not partition_errors
    run_context_pass = not run_context_errors
    return {
        "identity_binding_gate": _status(identity_ok),
        "completeness_gate": _status(completeness_pass),
        "trace_verified_gate": _status(trace_pass),
        "partition_binding_gate": _status(partition_pass),
        "run_context_binding_gate": _status(run_context_pass),
        "matrix_receipt_gate": _status(not matrix_errors),
        "expected_rows": expected_rows,
        "observed_rows": len(rows),
        "expected_arms": list(arms),
        "row_errors": sorted(set(row_errors)),
        "trace_errors": sorted(set(trace_errors)),
        "partition_errors": sorted(set(partition_errors)),
        "run_context_errors": sorted(set(run_context_errors)),
        "matrix_receipt_sha256": (
            None if not matrix_bytes else _sha256_bytes(matrix_bytes)
        ),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "manifest_sha256": (
            None if not manifest_bytes else _sha256_bytes(manifest_bytes)
        ),
        "integrity_pass": (
            identity_ok
            and completeness_pass
            and trace_pass
            and partition_pass
            and run_context_pass
            and not matrix_errors
        ),
    }


def _primary_checks(
    comparison: Mapping[str, object],
    *,
    delta_min: float,
    noninferiority_margin: float,
) -> dict[str, bool]:
    counts = comparison["wins_ties_losses"]
    ci = comparison["cluster_bootstrap_ci95"]
    assert isinstance(counts, Mapping)
    assert isinstance(ci, Mapping)
    return {
        "mean_at_least_delta_min": float(comparison["mean_difference"])
        >= delta_min,
        "median_strictly_positive": float(comparison["median_difference"]) > 0.0,
        "trimmed_mean_strictly_positive": float(
            comparison["trimmed_mean_difference"]
        )
        > 0.0,
        "wins_exceed_losses": int(counts["wins"]) > int(counts["losses"]),
        "ci95_lower_strictly_above_negative_noninferiority_margin": (
            ci.get("status") == "ESTABLISHED"
            and float(ci["lower"]) > -noninferiority_margin
        ),
    }


def _mechanism_checks(
    comparison: Mapping[str, object],
    *,
    noninferiority_margin: float,
) -> dict[str, bool]:
    counts = comparison["wins_ties_losses"]
    ci = comparison["cluster_bootstrap_ci95"]
    assert isinstance(counts, Mapping)
    assert isinstance(ci, Mapping)
    return {
        "mean_strictly_positive": float(comparison["mean_difference"]) > 0.0,
        "median_strictly_positive": float(comparison["median_difference"]) > 0.0,
        "trimmed_mean_strictly_positive": float(
            comparison["trimmed_mean_difference"]
        )
        > 0.0,
        "wins_exceed_losses": int(counts["wins"]) > int(counts["losses"]),
        "ci95_lower_strictly_above_negative_mechanism_margin": (
            ci.get("status") == "ESTABLISHED"
            and float(ci["lower"]) > -noninferiority_margin
        ),
    }


def _write_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(payload))


def evaluate_v21_confirmation_gate(
    *,
    rows_path: str | Path,
    thresholds_path: str | Path,
    output_path: str | Path,
    selected_candidate: str,
    control_id: str,
    predecessor_id: str,
    matrix_receipt_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate and exclusively record the prospective V21 confirmation gate."""

    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    rows_file = Path(rows_path).resolve()
    thresholds_file = Path(thresholds_path).resolve()
    frozen, thresholds_bytes = _load_frozen_gate(
        thresholds_file,
        selected_candidate=str(selected_candidate),
        control_id=str(control_id),
        predecessor_id=str(predecessor_id),
    )
    rows, rows_bytes = _load_rows(rows_file)
    matrix_file = (
        (rows_file.parent / "matrix_receipt.json").resolve()
        if matrix_receipt_path is None
        else Path(matrix_receipt_path).resolve()
    )
    audit = _audit_inputs(
        rows_path=rows_file,
        rows=rows,
        rows_bytes=rows_bytes,
        matrix_receipt_path=matrix_file,
        frozen=frozen,
        selected_candidate=str(selected_candidate),
        control_id=str(control_id),
        predecessor_id=str(predecessor_id),
    )
    base: dict[str, object] = {
        "schema": "pareto_v21_confirmation_gate_receipt_v1",
        "scientific_scope": (
            "independent_calibration_confirmation_not_formal_performance_evidence"
        ),
        "selected_candidate": str(selected_candidate),
        "control_id": str(control_id),
        "predecessor_id": str(predecessor_id),
        "rows_sha256": _sha256_bytes(rows_bytes),
        "thresholds_sha256": _sha256_bytes(thresholds_bytes),
        "thresholds_schema": frozen.precommit_schema,
        "matrix_receipt_path": str(matrix_file),
        "expected_rows": audit["expected_rows"],
        "observed_rows": audit["observed_rows"],
        "families": list(frozen.families),
        "inference_unit": "case_cluster",
        "replicate_aggregation": "seed_mean_within_case_and_arm",
        "frozen_thresholds": {
            "delta_min": frozen.delta_min,
            "noninferiority_margin": frozen.noninferiority_margin,
            "mechanism_noninferiority_margin": (
                frozen.mechanism_noninferiority_margin
            ),
            "bootstrap_samples": frozen.bootstrap_samples,
            "bootstrap_randomization_seed": frozen.randomization_seed,
            "tie_tolerance": frozen.tie_tolerance,
            "trim_fraction_each_tail": frozen.trim_fraction,
        },
        "integrity_checks": {
            key: value
            for key, value in audit.items()
            if key
            not in {
                "expected_rows",
                "observed_rows",
                "integrity_pass",
            }
        },
        "precommit_identity_binding_gate": audit["identity_binding_gate"],
        "completeness_gate": audit["completeness_gate"],
        "trace_verified_gate": audit["trace_verified_gate"],
        "partition_binding_gate": audit["partition_binding_gate"],
        "run_context_binding_gate": audit["run_context_binding_gate"],
        "matrix_receipt_gate": audit["matrix_receipt_gate"],
        "integrity_gate": "PASS" if audit["integrity_pass"] else "HOLD",
        "formal_status": "NOT_MATERIALIZED",
        "formal_performance_claim_authorized": False,
    }
    if not audit["integrity_pass"]:
        receipt = {
            **base,
            "status": "HOLD",
            "gate_status": "HOLD",
            "primary_gate": "NOT_RUN_DUE_TO_INTEGRITY_HOLD",
            "adjacent_mechanism_gate": "NOT_RUN_DUE_TO_INTEGRITY_HOLD",
            "primary_family_results": {},
            "mechanism_family_results": {},
            "action": "STOP_BEFORE_FORMAL_MATERIALIZATION",
        }
        _write_exclusive(destination, receipt)
        return receipt

    primary_results: dict[str, object] = {}
    mechanism_results: dict[str, object] = {}
    primary_gates: list[str] = []
    mechanism_gates: list[str] = []
    same_comparison = str(predecessor_id) == str(control_id)
    for family_index, family in enumerate(frozen.families):
        family_rows = [row for row in rows if row.get("family") == family]
        primary = compare_paired_cluster_metric(
            family_rows,
            cluster_keys=("case_id",),
            replicate_keys=("seed",),
            arm_key="candidate_id",
            treatment_arm=str(selected_candidate),
            control_arm=str(control_id),
            value_key=frozen.metric,
            bootstrap_samples=frozen.bootstrap_samples,
            randomization_seed=frozen.randomization_seed + 9173 * family_index,
            tie_tolerance=frozen.tie_tolerance,
        )
        primary_checks = _primary_checks(
            primary,
            delta_min=frozen.delta_min[family],
            noninferiority_margin=frozen.noninferiority_margin[family],
        )
        primary_gate = "PASS" if all(primary_checks.values()) else "FAIL"
        primary_results[family] = {
            "gate": primary_gate,
            "checks": primary_checks,
            "comparison": primary,
        }
        primary_gates.append(primary_gate)

        if same_comparison:
            mechanism = primary
            comparison_payload: dict[str, object] = {
                "comparison_source": "PRIMARY_CANDIDATE_VS_CONTROL_REUSED",
                "comparison_reference": f"primary_family_results.{family}.comparison",
            }
        else:
            mechanism = compare_paired_cluster_metric(
                family_rows,
                cluster_keys=("case_id",),
                replicate_keys=("seed",),
                arm_key="candidate_id",
                treatment_arm=str(selected_candidate),
                control_arm=str(predecessor_id),
                value_key=frozen.metric,
                bootstrap_samples=frozen.bootstrap_samples,
                randomization_seed=(
                    frozen.randomization_seed + 50_021 + 9173 * family_index
                ),
                tie_tolerance=frozen.tie_tolerance,
            )
            comparison_payload = {
                "comparison_source": "INDEPENDENT_ADJACENT_CONTRAST",
                "comparison": mechanism,
            }
        mechanism_checks = _mechanism_checks(
            mechanism,
            noninferiority_margin=(
                frozen.mechanism_noninferiority_margin[family]
            ),
        )
        mechanism_gate = "PASS" if all(mechanism_checks.values()) else "FAIL"
        mechanism_results[family] = {
            "gate": mechanism_gate,
            "checks": mechanism_checks,
            **comparison_payload,
        }
        mechanism_gates.append(mechanism_gate)

    primary_gate = "PASS" if all(value == "PASS" for value in primary_gates) else "FAIL"
    mechanism_gate = (
        "PASS" if all(value == "PASS" for value in mechanism_gates) else "FAIL"
    )
    gate_status = (
        "PASS" if primary_gate == mechanism_gate == "PASS" else "FAIL"
    )
    receipt = {
        **base,
        "status": gate_status,
        "gate_status": gate_status,
        "primary_gate": primary_gate,
        "adjacent_mechanism_gate": mechanism_gate,
        "adjacent_comparison_reused_primary": same_comparison,
        "primary_family_results": primary_results,
        "mechanism_family_results": mechanism_results,
        "action": (
            "AUTHORIZE_CANDIDATE_FREEZE_AND_FORMAL_ENTROPY_REVEAL"
            if gate_status == "PASS"
            else "STOP_BEFORE_FORMAL_MATERIALIZATION"
        ),
    }
    _write_exclusive(destination, receipt)
    return receipt


__all__ = ["evaluate_v21_confirmation_gate"]
