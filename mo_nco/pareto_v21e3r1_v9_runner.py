from __future__ import annotations

"""Fail-closed single-case V9R1 four-arm exposed-development runner."""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

from . import pareto_v21e3_hybrid as hybrid_module
from .pareto_ijoc_problem import (
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    problem_sha256,
)
from .pareto_v21e3_baselines import load_v21e3_development_problem
from .pareto_v21e3_hybrid import (
    V21E3HybridConfig,
    V21E3TypedHybridParetoSearch,
)
from .pareto_v21e3_trace_verify import verify_v21e3_trace_database
from .pareto_v21e3r1_branch_replay import reexecute_and_compare
from .pareto_v21e3r1_v9_diagnostics import analyze_v9_trace_database
from .pareto_v21e3r1_v9_protocol import (
    load_v9_predevelopment_protocol,
    validate_v9_resource_caps,
)


_SCREENING_POLICY = "bounded_cache_aware_structural_screen_development_v1"
_NONWORSE_REPLACEMENT = (
    "bounded_reference_neighborhood_nonworse_replacement_v1"
)
_LYAPUNOV_REPLACEMENT = (
    "archive_compensated_information_lyapunov_development_v1"
)
_ARM_ORDER = ("LEGACY", "SCREEN", "LYAP", "BOTH")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FROZEN_DEVELOPMENT_MANIFEST_SHA256 = (
    "1970361ba557aadd26de38aed008de11d11d158c797c00db1036cc4616cbdc8c"
)
_ARM_POLICIES: dict[str, tuple[str, str, bool, bool]] = {
    # arm -> (diagnostic stem, screening policy, uses Lyapunov, uses screening)
    "LEGACY": ("LEGACY", "disabled_v1", False, False),
    "SCREEN": ("INFORMATION_SCREEN", _SCREENING_POLICY, False, True),
    "LYAP": ("LYAPUNOV", "disabled_v1", True, False),
    "BOTH": ("INFORMATION_LYAPUNOV", _SCREENING_POLICY, True, True),
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exclusive_write_json(path: Path, payload: object) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_bytes(payload) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _positive_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _positive_real(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a real number") from error
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _validated_directions(payload: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(payload, (list, tuple)) or not payload:
        raise ValueError("directions must contain at least one direction")
    directions: list[tuple[float, ...]] = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(
                "every V9R1 reference direction must contain exactly two coordinates"
            )
        if any(type(value) not in {int, float} for value in row):
            raise TypeError(
                "reference-direction coordinates must be exact JSON numbers"
            )
        direction = tuple(float(value) for value in row)
        if any(not math.isfinite(value) or value <= 0.0 for value in direction):
            raise ValueError(
                "reference-direction coordinates must be finite and positive"
            )
        if not math.isclose(sum(direction), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "every reference direction must sum to one"
            )
        directions.append(direction)
    if len(set(directions)) != len(directions):
        raise ValueError("reference directions must be unique")
    return tuple(directions)


def _parse_directions(raw: str) -> tuple[tuple[float, ...], ...]:
    try:
        payload = json.loads(raw)
        return _validated_directions(payload)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "must be a JSON array of positive bi-objective directions"
        ) from error
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _validate_run_inputs(
    *,
    seed: int,
    charged_evaluations: int,
    attempt_cap: int,
    structural_screening_cap: int,
    wall_time_cap_seconds: float,
    candidate_screening_cap: int,
    archive_tradeoff_lambda: float,
    checkpoint_period: int,
    reference_directions: object,
) -> tuple[tuple[float, ...], ...]:
    if type(seed) is not int or seed < 0:
        raise TypeError("seed must be a nonnegative exact integer.")
    positive_integer_fields = {
        "charged_evaluations": charged_evaluations,
        "attempt_cap": attempt_cap,
        "candidate_screening_cap": candidate_screening_cap,
        "checkpoint_period": checkpoint_period,
    }
    for name, value in positive_integer_fields.items():
        if type(value) is not int or value <= 0:
            raise TypeError(f"{name} must be a positive exact integer.")
    if type(structural_screening_cap) is not int or structural_screening_cap < 0:
        raise TypeError(
            "structural_screening_cap must be a nonnegative exact integer."
        )
    for name, value in {
        "wall_time_cap_seconds": wall_time_cap_seconds,
        "archive_tradeoff_lambda": archive_tradeoff_lambda,
    }.items():
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise TypeError(f"{name} must be a finite positive exact real.")
    directions = _validated_directions(reference_directions)
    if charged_evaluations < len(directions):
        raise ValueError("B must initialize every reference direction.")
    if attempt_cap < charged_evaluations:
        raise ValueError("A must be at least B.")
    if structural_screening_cap < candidate_screening_cap:
        raise ValueError("S must cover one complete candidate screen.")
    return directions


def _artifact_binding(root: Path, path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _source_closure_manifest() -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    files: list[dict[str, object]] = []
    for path in sorted(package_root.rglob("*.py"), key=lambda item: item.as_posix()):
        raw = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(_PROJECT_ROOT).as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    core: dict[str, object] = {
        "schema": "pareto_v21e3r1_v9r1_python_source_closure_manifest_v1",
        "scope": "all_mo_nco_python_sources",
        "file_count": len(files),
        "files": files,
        "files_root_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
    }
    return {
        **core,
        "manifest_payload_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def _seal_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or int(checkpoint[0]) != 0:
            raise RuntimeError("The V9R1 SQLite WAL checkpoint failed.")
        journal_mode = str(
            connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        )
        if journal_mode.lower() != "delete":
            raise RuntimeError("The V9R1 SQLite trace did not enter DELETE mode.")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("The V9R1 SQLite trace failed integrity_check.")
        connection.commit()
    finally:
        connection.close()


def _problem_family(problem: object) -> str:
    if isinstance(problem, MultiObjectiveKnapsackInstance):
        return "MOKP"
    if isinstance(problem, MultiObjectiveTSPProblemAdapter):
        return "MOTSP"
    raise TypeError("The V9R1 runner supports only MOKP and MOTSP development cases.")


def _validate_exposed_development_case(
    *,
    case_path: Path,
    case_raw: bytes,
    problem: object,
    family: str,
) -> dict[str, object]:
    manifest_path = (case_path.parent.parent / "case_manifest.json").resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "The frozen exposed-development case manifest is unavailable: "
            f"{manifest_path}"
        )
    manifest_raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha256 != _FROZEN_DEVELOPMENT_MANIFEST_SHA256:
        raise ValueError("The frozen exposed-development manifest drifted.")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
        case_payload = json.loads(case_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("The case or development manifest is not UTF-8 JSON.") from error
    if not isinstance(manifest, dict) or not isinstance(case_payload, dict):
        raise ValueError("The case and development manifest must be JSON objects.")
    if not (
        manifest.get("schema") == "pareto_v21_partition_manifest_v1"
        and manifest.get("split") == "development"
        and manifest.get("formal_confirmatory_eligibility") is False
    ):
        raise ValueError("The frozen case manifest is not development-only.")
    case_id = case_payload.get("case_id")
    if type(case_id) is not str or not case_id:
        raise ValueError("The case omits an exact nonempty case_id.")
    entries = manifest.get("cases")
    if not isinstance(entries, list):
        raise ValueError("The frozen development manifest omits its cases.")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "The case is not an exact member of the frozen exposed-development set."
        )
    entry = matches[0]
    artifact = entry.get("artifact")
    fingerprints = entry.get("fingerprints")
    if not isinstance(artifact, dict) or not isinstance(fingerprints, dict):
        raise ValueError("The frozen development case entry is incomplete.")
    observed_artifact_sha256 = hashlib.sha256(case_raw).hexdigest()
    observed_problem_sha256 = problem_sha256(problem)  # type: ignore[arg-type]
    if not (
        entry.get("split") == "development"
        and entry.get("family") == family
        and entry.get("num_objectives") == 2
        and case_payload.get("family") == family
        and case_payload.get("num_objectives") == 2
        and artifact.get("bytes") == len(case_raw)
        and artifact.get("sha256") == observed_artifact_sha256
    ):
        raise ValueError(
            "The case bytes, family, or objective count drifted "
            "from the frozen exposed-development manifest."
        )
    return {
        "schema": "pareto_v21e3_exposed_development_case_binding_v1",
        "split": "development",
        "case_id": case_id,
        "manifest": {
            "path": str(manifest_path),
            "bytes": len(manifest_raw),
            "sha256": manifest_sha256,
        },
        "case_entry_sha256": hashlib.sha256(_canonical_bytes(entry)).hexdigest(),
        "artifact_sha256": observed_artifact_sha256,
        "manifest_declared_problem_sha256": fingerprints.get("problem_sha256"),
        "current_loader_problem_semantic_sha256": observed_problem_sha256,
    }


def _expected_run_context(
    problem: object,
    config: V21E3HybridConfig,
) -> dict[str, object]:
    semantic_config = config.semantic_payload()
    source_path = Path(hybrid_module.__file__).resolve()
    return {
        "schema": "v21e3r1_run_context_v2",
        "case_artifact_sha256": config.case_artifact_sha256,
        "case_artifact_binding_kind": "explicit_case_artifact_sha256_v1",
        "problem_semantic_sha256": problem_sha256(problem),  # type: ignore[arg-type]
        "candidate_id": "C0",
        "algorithm_config": semantic_config,
        "candidate_config_sha256": hashlib.sha256(
            _canonical_bytes(semantic_config)
        ).hexdigest(),
        "algorithm_source_sha256": (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            if config.source_snapshot_sha256 is None
            else config.source_snapshot_sha256
        ),
        "algorithm_source_binding_kind": (
            "explicit_source_snapshot_or_release_manifest_sha256_v1"
        ),
        "reference_directions": config.reference_directions,
        "seed": config.seed,
        "charged_evaluation_budget": config.charged_evaluations,
        "evidence_partition": "development",
        "objective_lower_bounds": tuple(  # type: ignore[attr-defined]
            float(value) for value in problem.objective_lower_bounds
        ),
        "objective_upper_bounds": tuple(  # type: ignore[attr-defined]
            float(value) for value in problem.objective_upper_bounds
        ),
        "v9_resource_contract_schema": "v21e3r1_v9_ast_resource_contract_v1",
    }


def _arm_config(
    *,
    arm: str,
    family: str,
    reference_directions: tuple[tuple[float, ...], ...],
    charged_evaluations: int,
    attempt_cap: int,
    structural_screening_cap: int,
    wall_time_cap_seconds: float,
    candidate_screening_cap: int,
    archive_tradeoff_lambda: float,
    checkpoint_period: int,
    seed: int,
    case_artifact_sha256: str,
    source_snapshot_sha256: str,
) -> V21E3HybridConfig:
    diagnostic_stem, screening_policy, uses_lyapunov, uses_screening = (
        _ARM_POLICIES[arm]
    )
    return V21E3HybridConfig(
        candidate_id="C0",
        reference_directions=reference_directions,
        charged_evaluations=charged_evaluations,
        checkpoint_period=checkpoint_period,
        seed=seed,
        phase="development",
        case_artifact_sha256=case_artifact_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        development_diagnostic_id=f"V21E3R1_V9_{diagnostic_stem}_{family}",
        candidate_screening_policy=screening_policy,
        candidate_screening_cap=candidate_screening_cap,
        replacement_policy=(
            _LYAPUNOV_REPLACEMENT if uses_lyapunov else _NONWORSE_REPLACEMENT
        ),
        archive_tradeoff_lambda=(
            archive_tradeoff_lambda if uses_lyapunov else 0.0
        ),
        attempt_cap=attempt_cap,
        structural_screening_cap=(
            structural_screening_cap if uses_screening else 0
        ),
        wall_time_cap_seconds=wall_time_cap_seconds,
    )


def _runner_report_failure(kind: str, arm: str, detail: str) -> RuntimeError:
    return RuntimeError(
        f"V9R2 {kind} did not fail closed for arm {arm}: {detail}."
    )


def _exact_nonnegative_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite_unit_interval(value: object) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_v9r2_trace_verification(
    report: object,
    *,
    arm: str,
    metadata: Mapping[str, object],
    population_size: int,
    candidate_screening_cap: int,
    archive_size: int,
    detached_terminal_receipt_sha256: str,
) -> None:
    """Require the complete V9 replay contract, not only its top-level PASS."""

    if not isinstance(report, Mapping):
        raise _runner_report_failure("trace verification", arm, "report is not a map")
    required_metadata_counts = {
        name: metadata.get(name)
        for name in (
            "attempt_count",
            "charged_evaluation_count",
            "physical_objective_call_count",
            "cache_hit_count",
            "candidate_screen_count",
            "archive_lyapunov_replacement_count",
            "archive_lyapunov_paid_worsening_count",
        )
    }
    if any(
        not _exact_nonnegative_count(value)
        for value in required_metadata_counts.values()
    ):
        raise _runner_report_failure(
            "trace verification", arm, "run accounting contains a non-exact count"
        )
    if (
        type(population_size) is not int
        or population_size <= 0
        or type(candidate_screening_cap) is not int
        or candidate_screening_cap <= 0
        or type(archive_size) is not int
        or archive_size < 0
    ):
        raise _runner_report_failure(
            "trace verification", arm, "runner expectations are invalid"
        )
    attempts = int(required_metadata_counts["attempt_count"])
    evaluations = int(required_metadata_counts["charged_evaluation_count"])
    physical_calls = int(
        required_metadata_counts["physical_objective_call_count"]
    )
    cache_hits = int(required_metadata_counts["cache_hit_count"])
    candidate_screens = int(required_metadata_counts["candidate_screen_count"])
    lyapunov_replacements = int(
        required_metadata_counts["archive_lyapunov_replacement_count"]
    )
    lyapunov_paid_worsening = int(
        required_metadata_counts["archive_lyapunov_paid_worsening_count"]
    )
    if population_size > evaluations:
        raise _runner_report_failure(
            "trace verification", arm, "population exceeds charged evaluations"
        )
    post_initialization_decisions = evaluations - population_size
    _, _, uses_lyapunov, uses_screening = _ARM_POLICIES[arm]
    expected_screen_state = (
        "PASS"
        if uses_screening and post_initialization_decisions > 0
        else "NOT_APPLICABLE"
    )
    expected_screen_witnesses = (
        post_initialization_decisions if uses_screening else 0
    )
    expected_lyapunov_state = (
        "PASS"
        if uses_lyapunov and post_initialization_decisions > 0
        else "NOT_APPLICABLE"
    )
    expected_lyapunov_witnesses = (
        post_initialization_decisions if uses_lyapunov else 0
    )
    exact_fields = {
        "schema": "v21e3r1_objective_archive_replay_receipt_v2",
        "status": "OBJECTIVE_AND_ARCHIVE_REPLAY_PASS",
        "verification_scope": (
            "objective_solution_chain_archive_and_terminal_replay_v1"
        ),
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "selection_authorization": "PROHIBITED",
        "detached_terminal_receipt_sha256": (
            detached_terminal_receipt_sha256
        ),
        "attempt_records": attempts,
        "physical_call_started_records": physical_calls,
        "evaluation_records": evaluations,
        "decision_records": evaluations,
        "unresolved_decision_records": 0,
        "cache_hit_records": cache_hits,
        "unique_solution_replays": physical_calls,
        "archive_reconstruction": "PASS",
        "archive_size": archive_size,
        "terminal_status": "SUCCESS",
        "v9_resource_contract_replay": "PASS",
        "v9_population_policy_replay": "PASS",
        "v9_population_policy_decisions_verified": evaluations,
        "v9_candidate_screen_witness_replay": expected_screen_state,
        "v9_candidate_screen_witnesses_verified": expected_screen_witnesses,
        "v9_lyapunov_policy_witness_replay": expected_lyapunov_state,
        "v9_lyapunov_policy_witnesses_verified": expected_lyapunov_witnesses,
    }
    mismatches = [
        name
        for name, expected in exact_fields.items()
        if report.get(name) != expected
    ]
    if mismatches:
        raise _runner_report_failure(
            "trace verification",
            arm,
            "mismatched fields " + ", ".join(mismatches),
        )
    if attempts != evaluations + cache_hits:
        raise _runner_report_failure(
            "trace verification", arm, "attempt/cache accounting is inconsistent"
        )
    if uses_screening:
        maximum_screens = candidate_screening_cap * expected_screen_witnesses
        if not expected_screen_witnesses <= candidate_screens <= maximum_screens:
            raise _runner_report_failure(
                "trace verification",
                arm,
                "candidate-screen accounting is inconsistent",
            )
    elif candidate_screens != 0:
        raise _runner_report_failure(
            "trace verification", arm, "disabled screen arm reports screen work"
        )
    if uses_lyapunov:
        if not (
            lyapunov_paid_worsening
            <= lyapunov_replacements
            <= expected_lyapunov_witnesses
        ):
            raise _runner_report_failure(
                "trace verification",
                arm,
                "Lyapunov replacement accounting is inconsistent",
            )
    elif lyapunov_replacements != 0 or lyapunov_paid_worsening != 0:
        raise _runner_report_failure(
            "trace verification", arm, "non-Lyapunov arm reports Lyapunov work"
        )


def _validate_v9r2_diagnostic_report(
    report: object,
    *,
    arm: str,
    family: str,
    development_diagnostic_id: str,
    metadata: Mapping[str, object],
    population_size: int,
    detached_terminal_receipt_sha256: str,
) -> None:
    """Validate every V9R2 diagnostic boundary consumed by the runner."""

    if not isinstance(report, Mapping):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "report is not a map"
        )
    required_metadata_counts = {
        name: metadata.get(name)
        for name in (
            "attempt_count",
            "charged_evaluation_count",
            "cache_hit_count",
            "candidate_screen_count",
            "candidate_screen_cache_skip_count",
        )
    }
    if any(
        not _exact_nonnegative_count(value)
        for value in required_metadata_counts.values()
    ):
        raise _runner_report_failure(
            "read-only diagnostic",
            arm,
            "run accounting contains a non-exact count",
        )
    attempts = int(required_metadata_counts["attempt_count"])
    evaluations = int(required_metadata_counts["charged_evaluation_count"])
    cache_hits = int(required_metadata_counts["cache_hit_count"])
    candidate_screens = int(required_metadata_counts["candidate_screen_count"])
    screen_cache_skips = int(
        required_metadata_counts["candidate_screen_cache_skip_count"]
    )
    if (
        type(population_size) is not int
        or population_size <= 0
        or population_size > evaluations
    ):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "population accounting is invalid"
        )
    post_initialization_decisions = evaluations - population_size
    _, _, uses_lyapunov, _ = _ARM_POLICIES[arm]
    lyapunov_replay = (
        "DURABLE_STATE_ARITHMETIC_REPLAY_PASS"
        if uses_lyapunov
        else "NOT_APPLICABLE_NON_LYAPUNOV_ARM"
    )
    expected_lyapunov_witnesses = (
        post_initialization_decisions if uses_lyapunov else 0
    )
    exact_fields = {
        "schema": "v21e3r1_v9_operator_productivity_diagnostic_v3",
        "status": "DEVELOPMENT_ONLY_NO_LATER_PHASE_AUTHORIZATION",
        "verification_scope": (
            "durable_semantic_chains_terminal_detached_and_arithmetic_"
            "reconstruction_v1"
        ),
        "objective_function_replay": "NOT_IMPLEMENTED_NO_PROBLEM_INPUT",
        "full_algorithm_decision_replay": "NOT_IMPLEMENTED",
        "detached_terminal_receipt_sha256": (
            detached_terminal_receipt_sha256
        ),
        "detached_terminal_receipt_externally_bound": True,
        "family": family,
        "development_diagnostic_id": development_diagnostic_id,
        "attempt_count": attempts,
        "first_evaluation_count": evaluations,
        "decision_count": evaluations,
        "population_size": population_size,
        "initialization_end_evaluation": population_size,
        "lyapunov_witness_count": expected_lyapunov_witnesses,
        "lyapunov_witness_violation_count": 0,
        "lyapunov_witness_replay": lyapunov_replay,
        "total_screenings": candidate_screens,
        "total_screen_cache_skips": screen_cache_skips,
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_independence": False,
        "policy_witness_independent_hv_reconstruction": True,
    }
    mismatches = [
        name
        for name, expected in exact_fields.items()
        if report.get(name) != expected
    ]
    if mismatches:
        raise _runner_report_failure(
            "read-only diagnostic",
            arm,
            "mismatched fields " + ", ".join(mismatches),
        )
    authorization = report.get("authorization")
    if not (
        isinstance(authorization, Mapping)
        and set(authorization) == {
            "selection",
            "confirmation",
            "formal",
            "submission",
        }
        and all(value is False for value in authorization.values())
    ):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "authorization is not exact deny-all"
        )
    validation = report.get("validation")
    expected_validation: dict[str, object] = {
        "sqlite_read_only_uri": True,
        "sqlite_query_only": True,
        "sqlite_integrity": "ok",
        "terminal_success": True,
        "contiguous_attempts": True,
        "contiguous_evaluations": True,
        "complete_decisions": True,
        "accounting_consistent": True,
        "attempt_semantic_hash_chain": True,
        "evaluation_semantic_hash_chain": True,
        "decision_semantic_hash_chain": True,
        "terminal_chain_bindings": True,
        "detached_terminal_receipt_exact_match": True,
        "detached_terminal_receipt_external_sha256_bound": True,
        "lyapunov_witness_durable_state_arithmetic": lyapunov_replay,
    }
    if not isinstance(validation, Mapping) or any(
        validation.get(name) != expected
        for name, expected in expected_validation.items()
    ):
        raise _runner_report_failure(
            "read-only diagnostic",
            arm,
            "semantic-chain, terminal, or detached validation is incomplete",
        )
    metric_names = (
        "initialization_terminal_hv",
        "exact_per_evaluation_left_continuous_hv_auc",
        "post_initialization_incremental_hv_gain",
        "final_normalized_hv",
        "total_reconstructed_hv_gain",
    )
    if any(not _finite_unit_interval(report.get(name)) for name in metric_names):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "HV metric is non-finite or out of range"
        )
    initialization_hv = float(report["initialization_terminal_hv"])
    hv_auc = float(report["exact_per_evaluation_left_continuous_hv_auc"])
    post_initialization_gain = float(
        report["post_initialization_incremental_hv_gain"]
    )
    final_hv = float(report["final_normalized_hv"])
    total_gain = float(report["total_reconstructed_hv_gain"])
    if not (
        initialization_hv <= final_hv + 1e-12
        and hv_auc <= final_hv + 1e-12
        and math.isclose(total_gain, final_hv, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            post_initialization_gain,
            final_hv - initialization_hv,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "HV arithmetic is inconsistent"
        )
    reconstruction = report.get("hv_reconstruction")
    if not (
        isinstance(reconstruction, Mapping)
        and reconstruction.get("schema")
        == "v21e3r1_v9_all_evaluated_2d_hv_reconstruction_v1"
        and reconstruction.get("normalized_reference") == [1.0, 1.0]
        and reconstruction.get("evaluation_order")
        == "charged_evaluation_index_ascending"
        and isinstance(reconstruction.get("trace_sha256"), str)
        and len(str(reconstruction["trace_sha256"])) == 64
    ):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "HV reconstruction binding is invalid"
        )
    for bounds_name in ("objective_lower_bounds", "objective_upper_bounds"):
        bounds = reconstruction.get(bounds_name)
        if not (
            isinstance(bounds, Sequence)
            and not isinstance(bounds, (str, bytes))
            and len(bounds) == 2
            and all(
                type(value) in {int, float} and math.isfinite(float(value))
                for value in bounds
            )
        ):
            raise _runner_report_failure(
                "read-only diagnostic", arm, f"{bounds_name} is invalid"
            )
    operators = report.get("operators")
    if not (
        isinstance(operators, list)
        and _exact_nonnegative_count(report.get("operator_count"))
        and report.get("operator_count") == len(operators)
    ):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "operator collection is invalid"
        )
    aggregate_counts = {
        "attempts": 0,
        "first_evaluations": 0,
        "cache_hits": 0,
        "screenings": 0,
        "screen_cache_skips": 0,
    }
    aggregate_gain = 0.0
    operator_names: set[str] = set()
    for index, row in enumerate(operators):
        if not isinstance(row, Mapping):
            raise _runner_report_failure(
                "read-only diagnostic", arm, f"operator {index} is not a map"
            )
        operator_name = row.get("operator")
        if (
            type(operator_name) is not str
            or not operator_name
            or operator_name in operator_names
        ):
            raise _runner_report_failure(
                "read-only diagnostic", arm, "operator identifiers are invalid"
            )
        operator_names.add(operator_name)
        count_names = (
            "attempts",
            "first_evaluations",
            "cache_hits",
            "screenings",
            "screen_cache_skips",
            "new_states",
        )
        if any(not _exact_nonnegative_count(row.get(name)) for name in count_names):
            raise _runner_report_failure(
                "read-only diagnostic", arm, f"operator {index} count is invalid"
            )
        row_attempts = int(row["attempts"])
        row_new_states = int(row["new_states"])
        row_first_evaluations = int(row["first_evaluations"])
        row_cache_hits = int(row["cache_hits"])
        if not (
            row_new_states == row_first_evaluations
            and row_attempts == row_first_evaluations + row_cache_hits
        ):
            raise _runner_report_failure(
                "read-only diagnostic",
                arm,
                f"operator {index} attempt accounting is inconsistent",
            )
        row_gain = row.get("hv_gain")
        if not _finite_unit_interval(row_gain):
            raise _runner_report_failure(
                "read-only diagnostic", arm, f"operator {index} gain is invalid"
            )
        gain = float(row_gain)
        expected_rate = row_new_states / row_attempts if row_attempts else 0.0
        expected_conditional = gain / row_new_states if row_new_states else 0.0
        expected_per_attempt = gain / row_attempts if row_attempts else 0.0
        productivity_fields = {
            "total_quality_gain": gain,
            "unseen_rate": expected_rate,
            "conditional_gain_per_new_state": expected_conditional,
            "gain_per_attempt": expected_per_attempt,
            "factorization_residual": (
                expected_per_attempt - expected_rate * expected_conditional
            ),
        }
        if any(
            type(row.get(name)) not in {int, float}
            or not math.isfinite(float(row[name]))
            or not math.isclose(
                float(row[name]), expected, rel_tol=0.0, abs_tol=1e-12
            )
            for name, expected in productivity_fields.items()
        ):
            raise _runner_report_failure(
                "read-only diagnostic",
                arm,
                f"operator {index} productivity arithmetic is inconsistent",
            )
        if row.get("elapsed_seconds") is not None or row.get("gain_per_second") is not None:
            raise _runner_report_failure(
                "read-only diagnostic", arm, f"operator {index} fabricates timing"
            )
        for name in aggregate_counts:
            aggregate_counts[name] += int(row[name])
        aggregate_gain += gain
    expected_aggregates = {
        "attempts": attempts,
        "first_evaluations": evaluations,
        "cache_hits": cache_hits,
        "screenings": candidate_screens,
        "screen_cache_skips": screen_cache_skips,
    }
    if aggregate_counts != expected_aggregates or not math.isclose(
        aggregate_gain, total_gain, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _runner_report_failure(
            "read-only diagnostic", arm, "operator aggregates are inconsistent"
        )


def _validate_v9r2_branch_replay_report(
    report: object,
    *,
    arm: str,
    metadata: Mapping[str, object],
    trace_path: Path,
    terminal_receipt_payload_sha256: str,
    case_artifact_sha256: str,
    problem_semantic_sha256: str,
    expected_run_context: Mapping[str, object],
    source_manifest: Mapping[str, object],
    source_manifest_path: Path,
    receipt_path: Path,
) -> None:
    """Validate the complete same-implementation branch receipt contract."""

    kind = "same-implementation branch replay"
    if not isinstance(report, Mapping):
        raise _runner_report_failure(kind, arm, "report is not a map")
    expected_report_keys = {
        "accounting",
        "artifacts",
        "checks",
        "first_mismatch",
        "implementation_independence",
        "original_counts",
        "problem_binding",
        "receipt_payload_sha256",
        "replay_counts",
        "run_context_binding",
        "schema",
        "scientific_independence",
        "scope",
        "source_binding",
        "status",
        "terminal_bindings",
        "third_party_replication",
    }
    if set(report) != expected_report_keys:
        raise _runner_report_failure(kind, arm, "receipt keys are not exact")
    exact_top_level = {
        "schema": "v21e3r1_same_implementation_branch_replay_v1",
        "status": "PASS_SAME_IMPLEMENTATION_FULL_BRANCH_REEXECUTION",
        "scope": (
            "same_source_stochastic_program_reexecution_not_independent_replication"
        ),
        "implementation_independence": False,
        "scientific_independence": False,
        "third_party_replication": False,
        "first_mismatch": None,
    }
    if any(report.get(name) != expected for name, expected in exact_top_level.items()):
        raise _runner_report_failure(kind, arm, "top-level contract is invalid")

    checks = report.get("checks")
    expected_check_names = {
        "accounting",
        "archive",
        "attempts",
        "decisions",
        "evaluations",
        "run_context",
        "solutions",
        "terminal",
    }
    if not (
        isinstance(checks, Mapping)
        and set(checks) == expected_check_names
        and all(checks[name] is True for name in expected_check_names)
    ):
        raise _runner_report_failure(kind, arm, "semantic checks are not exact all-true")

    required_metadata = {
        name: metadata.get(name)
        for name in (
            "attempt_count",
            "charged_evaluation_count",
            "physical_objective_call_count",
            "cache_hit_count",
        )
    }
    if any(not _exact_nonnegative_count(value) for value in required_metadata.values()):
        raise _runner_report_failure(kind, arm, "run accounting is not exact")
    attempts = int(required_metadata["attempt_count"])
    evaluations = int(required_metadata["charged_evaluation_count"])
    physical_calls = int(required_metadata["physical_objective_call_count"])
    cache_hits = int(required_metadata["cache_hit_count"])
    expected_counts = {
        "solutions": physical_calls,
        "attempts": attempts,
        "evaluations": evaluations,
        "decisions": evaluations,
    }
    if report.get("original_counts") != expected_counts or report.get(
        "replay_counts"
    ) != expected_counts:
        raise _runner_report_failure(kind, arm, "semantic row counts are inconsistent")
    expected_accounting = {
        "attempt_count": attempts,
        "physical_call_started_count": physical_calls,
        "charged_evaluation_count": evaluations,
        "decision_count": evaluations,
        "cache_hit_count": cache_hits,
        "unresolved_decision_count": 0,
    }
    accounting = report.get("accounting")
    if not (
        isinstance(accounting, Mapping)
        and set(accounting) == {"original", "replay"}
        and accounting["original"] == expected_accounting
        and accounting["replay"] == expected_accounting
    ):
        raise _runner_report_failure(kind, arm, "terminal accounting is inconsistent")

    terminal_bindings = report.get("terminal_bindings")
    if not (
        isinstance(terminal_bindings, Mapping)
        and set(terminal_bindings) == {"original", "replay"}
    ):
        raise _runner_report_failure(kind, arm, "terminal bindings are incomplete")
    for side in ("original", "replay"):
        binding = terminal_bindings[side]
        if not (
            isinstance(binding, Mapping)
            and set(binding)
            == {"run_status", "terminal_status", "failure_code", "receipt_sha256"}
            and binding["run_status"] == "SUCCESS"
            and binding["terminal_status"] == "SUCCESS"
            and binding["failure_code"] is None
            and _lower_sha256(binding["receipt_sha256"])
        ):
            raise _runner_report_failure(kind, arm, f"{side} terminal binding is invalid")
    if (
        terminal_bindings["original"]["receipt_sha256"]
        != terminal_receipt_payload_sha256
    ):
        raise _runner_report_failure(kind, arm, "original terminal digest is detached")

    expected_problem_binding = {
        "binding_kind": "explicit_case_artifact_sha256_v1",
        "context_case_artifact_sha256": case_artifact_sha256,
        "problem_artifact_sha256": case_artifact_sha256,
        "problem_semantic_sha256": problem_semantic_sha256,
    }
    if report.get("problem_binding") != expected_problem_binding:
        raise _runner_report_failure(kind, arm, "problem binding is invalid")

    expected_source_binding = {
        "binding_kind": "explicit_source_snapshot_or_release_manifest_sha256_v1",
        "context_source_sha256": source_manifest["files_root_sha256"],
        "executing_module": "mo_nco/pareto_v21e3_hybrid.py",
        "executing_module_sha256": hashlib.sha256(
            Path(hybrid_module.__file__).resolve().read_bytes()
        ).hexdigest(),
        "source_closure_scope": "all_live_mo_nco_python_sources",
        "source_closure_file_count": source_manifest["file_count"],
        "source_closure_verified": True,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            source_manifest_path.read_bytes()
        ).hexdigest(),
        "replay_verified": True,
    }
    if report.get("source_binding") != expected_source_binding:
        raise _runner_report_failure(kind, arm, "source closure binding is invalid")

    expected_context_binding = {
        "schema": expected_run_context["schema"],
        "digest_sha256": hashlib.sha256(
            _canonical_bytes(expected_run_context)
        ).hexdigest(),
        "candidate_config_sha256": expected_run_context[
            "candidate_config_sha256"
        ],
        "algorithm_source_sha256": expected_run_context[
            "algorithm_source_sha256"
        ],
    }
    if report.get("run_context_binding") != expected_context_binding:
        raise _runner_report_failure(kind, arm, "run-context binding is invalid")

    artifacts = report.get("artifacts")
    if not (
        isinstance(artifacts, Mapping)
        and set(artifacts)
        == {
            "original_database_bytes",
            "original_database_sha256",
            "replay_database_bytes",
            "replay_database_sha256",
        }
        and artifacts["original_database_bytes"] == trace_path.stat().st_size
        and artifacts["original_database_sha256"]
        == hashlib.sha256(trace_path.read_bytes()).hexdigest()
        and type(artifacts["replay_database_bytes"]) is int
        and artifacts["replay_database_bytes"] > 0
        and _lower_sha256(artifacts["replay_database_sha256"])
    ):
        raise _runner_report_failure(kind, arm, "database artifact binding is invalid")

    declared_receipt_sha256 = report.get("receipt_payload_sha256")
    receipt_core = dict(report)
    receipt_core.pop("receipt_payload_sha256", None)
    try:
        recomputed_receipt_sha256 = hashlib.sha256(
            _canonical_bytes(receipt_core)
        ).hexdigest()
        expected_receipt_bytes = (
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise _runner_report_failure(kind, arm, "receipt is not canonical JSON") from error
    if not (
        _lower_sha256(declared_receipt_sha256)
        and declared_receipt_sha256 == recomputed_receipt_sha256
        and receipt_path.read_bytes() == expected_receipt_bytes
    ):
        raise _runner_report_failure(kind, arm, "receipt self-binding is invalid")


def run_v9r1_development_case(
    *,
    case: str | Path,
    outdir: str | Path,
    seed: int,
    reference_directions: tuple[tuple[float, ...], ...],
    charged_evaluations: int,
    attempt_cap: int,
    structural_screening_cap: int,
    wall_time_cap_seconds: float,
    candidate_screening_cap: int,
    archive_tradeoff_lambda: float,
    checkpoint_period: int,
    acknowledge_exposed_development_only: bool,
    protocol_path: str | Path | None = None,
    expected_protocol_file_sha256: str | None = None,
) -> dict[str, object]:
    """Run the four hardened V9R2 arms on one acknowledged development case.

    The V9R1 function name remains as a compatibility alias only.
    """

    if acknowledge_exposed_development_only is not True:
        raise PermissionError(
            "Exact acknowledgement of exposed-development-only execution is required."
        )
    protocol = load_v9_predevelopment_protocol(
        protocol_path,
        expected_file_sha256=expected_protocol_file_sha256,
    )
    execution_authorization = protocol["execution_authorization"]
    later_authorization = protocol["later_phase_authorization"]
    protocol_payload = protocol["payload"]
    if not (
        protocol.get("status") == "PRE_DEVELOPMENT_HOLD"
        and isinstance(execution_authorization, dict)
        and execution_authorization.get("single_case_smoke") is True
        and execution_authorization.get("full_development_matrix") is False
        and execution_authorization.get("scientific_development_claims") is False
        and isinstance(later_authorization, dict)
        and all(value is False for value in later_authorization.values())
        and isinstance(protocol_payload, dict)
        and protocol_payload.get("execution_scope", {}).get(
            "case_manifest_sha256"
        )
        == _FROZEN_DEVELOPMENT_MANIFEST_SHA256
    ):
        raise PermissionError(
            "The V9R2 pre-development protocol does not authorize this "
            "single-case engineering smoke."
        )
    resource_caps = validate_v9_resource_caps(
        B=charged_evaluations,
        A=attempt_cap,
        S=structural_screening_cap,
        T=wall_time_cap_seconds,
    )
    reference_directions = _validate_run_inputs(
        seed=seed,
        charged_evaluations=charged_evaluations,
        attempt_cap=attempt_cap,
        structural_screening_cap=structural_screening_cap,
        wall_time_cap_seconds=wall_time_cap_seconds,
        candidate_screening_cap=candidate_screening_cap,
        archive_tradeoff_lambda=archive_tradeoff_lambda,
        checkpoint_period=checkpoint_period,
        reference_directions=reference_directions,
    )
    case_path = Path(case).resolve()
    output = Path(outdir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    case_raw = case_path.read_bytes()
    case_sha256 = hashlib.sha256(case_raw).hexdigest()
    problem = load_v21e3_development_problem(case_path)
    family = _problem_family(problem)
    if problem.num_objectives != 2:
        raise ValueError("V9R2 requires an exact bi-objective development case.")
    development_manifest_binding = _validate_exposed_development_case(
        case_path=case_path,
        case_raw=case_raw,
        problem=problem,
        family=family,
    )
    source_manifest = _source_closure_manifest()
    source_manifest_sha256 = str(source_manifest["files_root_sha256"])

    configs = {
        arm: _arm_config(
            arm=arm,
            family=family,
            reference_directions=reference_directions,
            charged_evaluations=charged_evaluations,
            attempt_cap=attempt_cap,
            structural_screening_cap=structural_screening_cap,
            wall_time_cap_seconds=wall_time_cap_seconds,
            candidate_screening_cap=candidate_screening_cap,
            archive_tradeoff_lambda=archive_tradeoff_lambda,
            checkpoint_period=checkpoint_period,
            seed=seed,
            case_artifact_sha256=case_sha256,
            source_snapshot_sha256=source_manifest_sha256,
        )
        for arm in _ARM_ORDER
    }

    output.mkdir(parents=True, exist_ok=False)
    protocol_path_in_output = output / "predevelopment_protocol.json"
    _exclusive_write_json(protocol_path_in_output, protocol_payload)
    source_manifest_path = output / "source_manifest.json"
    _exclusive_write_json(source_manifest_path, source_manifest)
    arm_summaries: dict[str, object] = {}
    for arm in _ARM_ORDER:
        arm_directory = output / arm
        arm_directory.mkdir(exist_ok=False)
        trace_path = arm_directory / "trace.sqlite3"
        terminal_path = arm_directory / "terminal.json"
        config = replace(
            configs[arm],
            trace_database=str(trace_path),
            terminal_receipt=str(terminal_path),
            receipt_database_path="trace.sqlite3",
            capture_trace=False,
        )
        run = V21E3TypedHybridParetoSearch(problem, config).run()
        result = run.optimization_result
        metadata = result.metadata
        _seal_sqlite(trace_path)
        terminal_receipt = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal_binding = _artifact_binding(output, terminal_path)
        expected_run_context = _expected_run_context(problem, config)
        trace_verification = verify_v21e3_trace_database(
            trace_path,
            problem,
            expected_run_context=expected_run_context,
            detached_terminal_receipt_path=terminal_path,
            expected_detached_terminal_receipt_sha256=str(
                terminal_binding["sha256"]
            ),
            expected_charged_evaluations=charged_evaluations,
        )
        _validate_v9r2_trace_verification(
            trace_verification,
            arm=arm,
            metadata=metadata,
            population_size=len(reference_directions),
            candidate_screening_cap=candidate_screening_cap,
            archive_size=len(result.archive.entries),
            detached_terminal_receipt_sha256=str(terminal_binding["sha256"]),
        )
        diagnostic = analyze_v9_trace_database(
            trace_path,
            detached_terminal_receipt_path=terminal_path,
            expected_detached_terminal_receipt_sha256=str(
                terminal_binding["sha256"]
            ),
        )
        _validate_v9r2_diagnostic_report(
            diagnostic,
            arm=arm,
            family=family,
            development_diagnostic_id=config.development_diagnostic_id,
            metadata=metadata,
            population_size=len(reference_directions),
            detached_terminal_receipt_sha256=str(terminal_binding["sha256"]),
        )
        portable_diagnostic = dict(diagnostic)
        portable_diagnostic.pop("database_path", None)
        portable_diagnostic.pop("detached_terminal_receipt_path", None)
        diagnostic_path = arm_directory / "diagnostic.json"
        _exclusive_write_json(diagnostic_path, portable_diagnostic)
        branch_replay_path = arm_directory / "branch_replay.json"
        branch_replay = reexecute_and_compare(
            original_database=trace_path,
            problem_artifact=case_path,
            output_receipt=branch_replay_path,
            source_manifest_path=source_manifest_path,
        )
        _validate_v9r2_branch_replay_report(
            branch_replay,
            arm=arm,
            metadata=metadata,
            trace_path=trace_path,
            terminal_receipt_payload_sha256=str(
                terminal_receipt["receipt_payload_sha256"]
            ),
            case_artifact_sha256=case_sha256,
            problem_semantic_sha256=problem_sha256(problem),
            expected_run_context=expected_run_context,
            source_manifest=source_manifest,
            source_manifest_path=source_manifest_path,
            receipt_path=branch_replay_path,
        )
        portable_trace_verification = dict(trace_verification)
        portable_trace_verification.pop("database_path", None)
        portable_trace_verification.pop("detached_terminal_receipt_path", None)
        arm_summaries[arm] = {
            "status": "SUCCESS_ENGINEERING_ONLY",
            "arm_id": arm,
            "development_diagnostic_id": config.development_diagnostic_id,
            "algorithm_config": config.semantic_payload(),
            "trace_database": _artifact_binding(output, trace_path),
            "terminal_receipt": terminal_binding,
            "diagnostic_report": _artifact_binding(output, diagnostic_path),
            "branch_replay_report": _artifact_binding(
                output, branch_replay_path
            ),
            "terminal_receipt_payload_sha256": terminal_receipt[
                "receipt_payload_sha256"
            ],
            "charged_evaluation_count": metadata["charged_evaluation_count"],
            "physical_objective_call_count": metadata[
                "physical_objective_call_count"
            ],
            "attempt_count": metadata["attempt_count"],
            "cache_hit_count": metadata["cache_hit_count"],
            "candidate_screen_count": metadata["candidate_screen_count"],
            "structural_screening_work_count": metadata[
                "structural_screening_work_count"
            ],
            "archive_lyapunov_replacement_count": metadata[
                "archive_lyapunov_replacement_count"
            ],
            "archive_lyapunov_paid_worsening_count": metadata[
                "archive_lyapunov_paid_worsening_count"
            ],
            "v9_resource_accounting": metadata["v9_resource_accounting"],
            "archive_size": len(result.archive.entries),
            "archive_objectives": [
                list(entry.objectives) for entry in result.archive.entries
            ],
            "trace_verification": portable_trace_verification,
            "read_only_operator_diagnostic": portable_diagnostic,
            "same_implementation_branch_replay": branch_replay,
        }

    summary_core: dict[str, object] = {
        "schema": "pareto_v21e3r1_v9r2_single_case_four_arm_summary_v2",
        "status": "SUCCESS_ENGINEERING_ONLY",
        "scientific_scope": (
            "same_implementation_exposed_development_diagnostic_only"
        ),
        "evidence_partition": "EXPOSED_DEVELOPMENT_ONLY",
        "case": {
            "path": str(case_path),
            "bytes": len(case_raw),
            "sha256": case_sha256,
            "case_id": problem.name,
            "family": family,
            "problem_semantic_sha256": problem_sha256(problem),
            "development_manifest_binding": development_manifest_binding,
        },
        "candidate_id": "C0",
        "phase": "development",
        "source_closure_manifest": _artifact_binding(output, source_manifest_path),
        "source_snapshot_sha256": source_manifest_sha256,
        "predevelopment_protocol": {
            "artifact": _artifact_binding(output, protocol_path_in_output),
            "status": protocol["status"],
            "canonical_sha256": protocol["canonical_sha256"],
            "protocol_payload_sha256": protocol["protocol_payload_sha256"],
            "resource_contract_sha256": protocol["resource_contract_sha256"],
            "unmet_required_artifacts": protocol["unmet_required_artifacts"],
            "execution_authorization": execution_authorization,
            "later_phase_authorization": later_authorization,
        },
        "validated_resource_caps": resource_caps,
        "common_execution_contract": {
            "seed": seed,
            "reference_directions": reference_directions,
            "charged_evaluation_budget": charged_evaluations,
            "attempt_cap": attempt_cap,
            "structural_screening_cap_for_screening_arms": (
                structural_screening_cap
            ),
            "structural_screening_cap_for_disabled_arms": 0,
            "wall_time_cap_seconds": wall_time_cap_seconds,
            "candidate_screening_cap": candidate_screening_cap,
            "archive_tradeoff_lambda_for_lyapunov_arms": (
                archive_tradeoff_lambda
            ),
            "archive_tradeoff_lambda_for_non_lyapunov_arms": 0.0,
            "checkpoint_period": checkpoint_period,
        },
        "arms": arm_summaries,
        "expected_arm_count": 4,
        "completed_arm_count": len(arm_summaries),
        "implementation_identity": "same_repository_same_implementation_v1",
        "implementation_independence": False,
        "third_party_independence": False,
        "scientific_independence": False,
        "scientific_claim_authorized": False,
        "selection_authorized": False,
        "selection_cases_materialized": False,
        "confirmation_authorized": False,
        "confirmation_cases_materialized": False,
        "formal_authorized": False,
        "formal_study_authorized": False,
        "formal_cases_materialized": False,
        "ijoc_submission_authorized": False,
        "ijoc_submission_status": "IJOC_HOLD_NO_SUBMIT",
        "authorized_next_phase": "NONE_STOP_AFTER_EXPOSED_DEVELOPMENT_RUN",
    }
    summary = {
        **summary_core,
        "summary_payload_sha256": hashlib.sha256(
            _canonical_bytes(summary_core)
        ).hexdigest(),
    }
    _exclusive_write_json(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V21e3r1 V9R1 four-arm matrix on one already exposed "
            "development case."
        )
    )
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=_nonnegative_integer, required=True)
    parser.add_argument("--directions", type=_parse_directions, required=True)
    parser.add_argument(
        "--charged-evaluations", type=_positive_integer, required=True
    )
    parser.add_argument("--attempt-cap", type=_positive_integer, required=True)
    parser.add_argument(
        "--structural-screening-cap",
        type=_nonnegative_integer,
        required=True,
    )
    parser.add_argument(
        "--wall-time-cap-seconds", type=_positive_real, required=True
    )
    parser.add_argument(
        "--candidate-screening-cap", type=_positive_integer, required=True
    )
    parser.add_argument(
        "--archive-tradeoff-lambda", type=_positive_real, required=True
    )
    parser.add_argument("--checkpoint-period", type=_positive_integer, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-file-sha256")
    parser.add_argument(
        "--acknowledge-exposed-development-only",
        action="store_true",
    )
    return parser


# V9R2 is the hardened public name.  Retain the V9R1 symbol so already-written
# development-only callers keep working while receiving the V9R2 checks.
run_v9r2_development_case = run_v9r1_development_case


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.acknowledge_exposed_development_only:
        parser.error(
            "--acknowledge-exposed-development-only is required; this runner "
            "is restricted to already exposed development cases"
        )
    summary = run_v9r2_development_case(
        case=args.case,
        outdir=args.outdir,
        seed=args.seed,
        reference_directions=args.directions,
        charged_evaluations=args.charged_evaluations,
        attempt_cap=args.attempt_cap,
        structural_screening_cap=args.structural_screening_cap,
        wall_time_cap_seconds=args.wall_time_cap_seconds,
        candidate_screening_cap=args.candidate_screening_cap,
        archive_tradeoff_lambda=args.archive_tradeoff_lambda,
        checkpoint_period=args.checkpoint_period,
        acknowledge_exposed_development_only=True,
        protocol_path=args.protocol,
        expected_protocol_file_sha256=args.expected_protocol_file_sha256,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(Path(args.outdir).resolve() / "summary.json"),
                "selection_authorized": False,
                "confirmation_authorized": False,
                "formal_authorized": False,
                "ijoc_submission_authorized": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_v9r1_development_case", "run_v9r2_development_case", "main"]
