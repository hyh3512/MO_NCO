from __future__ import annotations

"""Fail-closed four-arm common-runner preflight for prospective V21e3.

This module deliberately contains no formal-case generator and no execution
adapter.  It proves only that a prototype context has one artifact root and a
common four-arm contract.  Formal materialization remains prohibited.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .pareto_v21e3_artifacts import ArtifactRoot
from .pareto_v21e3_baselines import frozen_development_baseline_configs
from .pareto_v21e3_timing import load_timing_policy


_ARMS = ("V21E3_CANDIDATE", "V21E3_C0", "NSGAII", "MOEAD")
_FAMILIES = ("MOTSP", "MOKP")
_OBJECTIVE_SEMANTICS = "first_true_objective_evaluation_v1"
_ATTEMPT_SEMANTICS = "all_attempts_terminal_receipt_v1"


class FormalMaterializationProhibited(RuntimeError):
    """Raised whenever this prototype is asked to execute formal work."""


def preflight_development_parity_protocol_v2(
    artifact_root: ArtifactRoot,
    protocol_binding: Mapping[str, object],
) -> dict[str, object]:
    """Validate adapter parity protocol v2 without authorizing its matrix.

    Adapter implementation necessarily precedes the successor source freeze.
    This preflight therefore accepts only the explicit ``PENDING`` snapshot
    state and keeps the matched matrix, selection, calibration, and formal
    gates closed.
    """

    protocol_path = artifact_root.resolve_binding(protocol_binding)
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    if protocol.get("schema") != "pareto_v21e3_c0_parity_protocol_v2":
        raise ValueError("Unexpected V21e3 parity protocol-v2 schema.")
    if protocol.get("status") != (
        "ENGINEERING_ADAPTERS_AVAILABLE_SUCCESSOR_SNAPSHOT_PENDING"
    ):
        raise ValueError("Parity protocol v2 is not in its pre-snapshot state.")
    if protocol.get("successor_version") != "V21e3r1":
        raise ValueError("Parity protocol v2 does not name the successor object.")
    if tuple(protocol.get("families", ())) != _FAMILIES:
        raise ValueError("Parity protocol v2 must cover MOTSP and MOKP.")

    common = protocol.get("common_execution_contract")
    expected_common = {
        "charged_evaluation_budget": 2_000,
        "checkpoint_period": 200,
        "objective_call_semantics": _OBJECTIVE_SEMANTICS,
        "attempt_history_semantics": _ATTEMPT_SEMANTICS,
        "duplicate_policy": (
            "exact_solution_cache_zero_charge_retry_then_fallback_v1"
        ),
        "duplicate_retry_cap": 4,
        "retry_policy": "same_family_single_perturbation_v1",
        "fallback_attempt_cap": 16,
        "fallback_policy": "problem_native_exact_random_solution_v1",
        "archive_policy": (
            "unbounded_exact_nondominated_all_unique_evaluations_v1"
        ),
        "checkpoint_policy": (
            "genuine_archive_snapshot_on_charged_evaluation_grid_v1"
        ),
        "failure_policy": (
            "durable_terminal_failure_on_objective_contract_or_retry_exhaustion_v1"
        ),
        "duplicate_identity": "exact_integer_solution_tuple_v1",
        "algorithm_rng_policy": "repository_single_python_random_seed_stream_v1",
        "duplicate_rng_policy": "domain_separated_retry_and_fallback_streams_v1",
        "runtime_claim_authorized": False,
    }
    if common != expected_common:
        raise ValueError("Parity protocol v2 changed the common budget semantics.")

    direction_contract = protocol.get("candidate_reference_directions")
    if not isinstance(direction_contract, dict):
        raise ValueError("Parity protocol v2 omits its direction contract.")
    if not (
        direction_contract.get("count") == 21
        and direction_contract.get("policy")
        == "v21e3_candidate_positive_evenly_spaced_biobjective_v1"
        and direction_contract.get("source_field") == "reference_directions"
    ):
        raise ValueError("Parity protocol v2 changed the direction policy.")
    direction_path = artifact_root.resolve_binding(
        direction_contract.get("source_binding", {})
    )
    direction_raw = direction_path.read_bytes()
    direction_manifest = json.loads(direction_raw)
    directions = direction_manifest.get("reference_directions")
    if not isinstance(directions, list) or len(directions) != 21:
        raise ValueError("The bound development direction grid is not 21-point.")

    seeds = protocol.get("case_design", {}).get("seeds")
    if seeds != [31051, 31057, 31059]:
        raise ValueError("Parity protocol v2 changed the frozen paired seeds.")
    configs_by_family = {
        family: frozen_development_baseline_configs(
            family=family,
            charged_evaluations=2_000,
            checkpoint_period=200,
            seed=31051,
        )
        for family in _FAMILIES
    }
    arms = protocol.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != (
        "V21E3_C0",
        "NSGAII",
        "MOEAD",
    ):
        raise ValueError("Parity protocol v2 changed the ordered arm set.")
    if not (
        arms["V21E3_C0"].get("candidate_id") == "C0"
        and arms["V21E3_C0"].get("population_size") == 21
    ):
        raise ValueError("Parity protocol v2 changed the C0 identity.")
    adapter_status_by_arm: dict[str, str] = {}
    for arm_id in ("V21E3_C0", "NSGAII", "MOEAD"):
        arm = arms[arm_id]
        if arm.get("execution_adapter_status") != "DEVELOPMENT_ONLY_AVAILABLE":
            raise ValueError("Every parity-v2 arm must have a development adapter.")
        if arm.get("source_binding_status") != (
            "PENDING_SUCCESSOR_FULL_SNAPSHOT"
        ):
            raise ValueError("Adapters must be frozen only after implementation.")
        adapter_status_by_arm[arm_id] = str(arm["execution_adapter_status"])

    for arm_id in ("NSGAII", "MOEAD"):
        arm = arms[arm_id]
        expected_identity = configs_by_family["MOTSP"][arm_id].adaptation_identity
        if arm.get("adaptation_identity") != expected_identity:
            raise ValueError(f"Parity protocol v2 changed {arm_id} identity.")
        if arm.get("repository_baseline_deviation_scope") != (
            "first_true_budget_duplicate_trace_and_cumulative_measurement_"
            "archive_seam_only_v1"
        ):
            raise ValueError("Parity protocol v2 expanded baseline deviations.")
        family_contracts = arm.get("family_configurations")
        if not isinstance(family_contracts, dict) or tuple(family_contracts) != _FAMILIES:
            raise ValueError("Parity protocol v2 omits family-specific parameters.")
        for family in _FAMILIES:
            config = configs_by_family[family][arm_id]
            family_contract = family_contracts[family]
            expected_fields = {
                "population_size": config.population_size,
                "initialization_policy": config.initialization_policy,
                "crossover_policy": config.crossover_policy,
                "mutation_policy": config.mutation_policy,
                "repair_policy": config.repair_policy,
                "reference_direction_policy": config.reference_direction_policy,
            }
            if arm_id == "NSGAII":
                expected_fields["selection_policy"] = config.selection_policy
            else:
                expected_fields["parent_selection_policy"] = (
                    config.selection_policy
                )
            for field, expected in expected_fields.items():
                if family_contract.get(field) != expected:
                    raise ValueError(
                        f"Parity protocol v2 changed {arm_id}/{family} {field}."
                    )
            direction_count_key = (
                "reference_direction_count_for_context"
                if arm_id == "NSGAII"
                else "reference_direction_count"
            )
            if family_contract.get(direction_count_key) != config.population_size:
                raise ValueError("Parity context direction count is not population-bound.")
            if family == "MOTSP" and arm_id == "NSGAII" and (
                family_contract.get("mutation_probability")
                != config.motsp_mutation_probability
            ):
                raise ValueError("MOTSP NSGA-II mutation probability changed.")
            if family == "MOKP" and family_contract.get(
                "mutation_rate_policy"
            ) != config.mokp_mutation_rate_policy:
                raise ValueError("MOKP mutation-rate policy changed.")
            if arm_id == "MOEAD" and not (
                family_contract.get("neighborhood_size")
                == config.neighborhood_size
                and family_contract.get("neighborhood_policy")
                == config.neighborhood_policy
                and family_contract.get("scalar_weight_floor")
                == config.scalar_weight_floor
                and family_contract.get("maximum_replacements")
                == config.maximum_replacements
            ):
                raise ValueError("MOEA/D family parameters changed.")
        command = arms[arm_id].get("exact_command")
        if not isinstance(command, str) or not command.startswith(
            "python -m mo_nco.pareto_v21e3_baselines "
        ):
            raise ValueError(f"Parity protocol v2 omits the {arm_id} command.")
        if f"--arm {arm_id}" not in command:
            raise ValueError(f"Parity protocol v2 command names another arm.")
        if (
            "--source-snapshot-sha256 "
            "<pending-successor-source-snapshot-sha256>"
        ) not in command:
            raise ValueError(
                "Parity protocol v2 command cannot bind the successor source."
            )
    if not (
        arms["NSGAII"].get("survival_policy")
        == configs_by_family["MOTSP"]["NSGAII"].survival_policy
        and arms["NSGAII"].get("survival_schedule")
        == configs_by_family["MOTSP"]["NSGAII"].survival_schedule
        and arms["MOEAD"].get("scalarization")
        == configs_by_family["MOTSP"]["MOEAD"].scalarization_policy
        and arms["MOEAD"].get("replacement_policy")
        == configs_by_family["MOTSP"]["MOEAD"].replacement_policy
        and arms["MOEAD"].get("replacement_order_policy")
        == configs_by_family["MOTSP"]["MOEAD"].replacement_order_policy
        and arms["MOEAD"].get("survival_policy")
        == configs_by_family["MOTSP"]["MOEAD"].survival_policy
        and arms["MOEAD"].get("update_schedule")
        == configs_by_family["MOTSP"]["MOEAD"].survival_schedule
    ):
        raise ValueError("Parity protocol v2 changed an arm-specific policy.")

    runtime_contract = protocol.get("runtime_and_dependency_contract")
    if runtime_contract != {
        "python_implementation": "CPython",
        "minimum_python": "3.10",
        "external_algorithm_dependency": (
            "NONE_STDLIB_AND_BOUND_MO_NCO_SOURCE_ONLY"
        ),
        "executable_identity": (
            "absolute_path_version_bytes_sha256_recorded_per_run_v1"
        ),
        "argv_semantics": (
            "module_entrypoint_and_all_flags_frozen_in_each_arm_exact_command_v1"
        ),
        "successor_source_and_dependency_manifest": (
            "PENDING_AFTER_ADAPTER_IMPLEMENTATION"
        ),
        "runtime_efficiency_claim_authorized": False,
    }:
        raise ValueError("Parity protocol v2 runtime/dependency contract changed.")

    gates = protocol.get("preflight_gates")
    if not isinstance(gates, dict) or not (
        gates.get("successor_source_snapshot") == "PENDING"
        and gates.get("target_scale_structural_and_adversarial_tests") == "NOT_RUN"
        and gates.get("independent_protocol_preflight") == "NOT_RUN"
        and gates.get("matched_matrix") == "NOT_RUN"
        and gates.get("noninferiority") == "NOT_ESTABLISHED"
        and gates.get("selection_entropy_release") == "PROHIBITED"
        and gates.get("calibration_execution") == "PROHIBITED"
        and gates.get("formal_execution") == "PROHIBITED"
        and gates.get("formal_authorized") is False
    ):
        raise ValueError("Parity protocol v2 later-phase gates are not fail-closed.")

    baseline_source = Path(__file__).with_name("pareto_v21e3_baselines.py")
    return {
        "schema": "pareto_v21e3_c0_parity_protocol_preflight_receipt_v2",
        "status": "ADAPTER_PREFLIGHT_PASS_SUCCESSOR_SNAPSHOT_REQUIRED",
        "scientific_scope": "engineering_preflight_not_performance_evidence",
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "candidate_direction_manifest_sha256": hashlib.sha256(
            direction_raw
        ).hexdigest(),
        "development_adapter_source_observed_sha256": hashlib.sha256(
            baseline_source.read_bytes()
        ).hexdigest(),
        "development_adapter_source_hash_role": (
            "DIAGNOSTIC_ONLY_PENDING_SUCCESSOR_FULL_SNAPSHOT"
        ),
        "execution_adapter_status_by_arm": adapter_status_by_arm,
        "families": list(_FAMILIES),
        "charged_evaluation_budget": 2_000,
        "checkpoint_period": 200,
        "successor_source_snapshot": gates["successor_source_snapshot"],
        "target_scale_structural_and_adversarial_tests": gates[
            "target_scale_structural_and_adversarial_tests"
        ],
        "independent_protocol_preflight": gates[
            "independent_protocol_preflight"
        ],
        "matched_matrix": gates["matched_matrix"],
        "noninferiority": gates["noninferiority"],
        "parity_execution_authorized": False,
        "selection_entropy_release": gates["selection_entropy_release"],
        "calibration_execution": gates["calibration_execution"],
        "formal_execution": gates["formal_execution"],
        "formal_authorized": False,
    }


def preflight_formal_common_runner(
    artifact_root: ArtifactRoot,
    context_binding: Mapping[str, object],
) -> dict[str, object]:
    """Validate the frozen prototype contract without creating any cases."""

    context_path = artifact_root.resolve_binding(context_binding)
    context_raw = context_path.read_bytes()
    context = json.loads(context_raw)
    if context.get("schema") != "pareto_v21e3_formal_common_runner_context_v1":
        raise ValueError("Unexpected V21e3 common-runner context schema.")
    if context.get("status") != (
        "PROTOTYPE_ONLY_FORMAL_MATERIALIZATION_PROHIBITED"
    ):
        raise ValueError("The common-runner context is not prototype-only.")
    if not isinstance(context.get("artifact_root_id"), str) or not context[
        "artifact_root_id"
    ]:
        raise ValueError("The common-runner context omits artifact_root_id.")
    if not (
        context.get("formal_authorized") is False
        and context.get("formal_cases_status") == "NOT_MATERIALIZED"
        and context.get("formal_case_manifest") is None
        and context.get("future_external_entropy_status") == "NOT_ESTABLISHED"
    ):
        raise ValueError("Formal materialization boundaries are not fail-closed.")

    arms = context.get("arms")
    if not isinstance(arms, list) or tuple(
        arm.get("arm_id") if isinstance(arm, dict) else None for arm in arms
    ) != _ARMS:
        raise ValueError("The exact ordered four-arm contract is required.")
    source_hashes: dict[str, list[str]] = {}
    adapter_status_by_arm: dict[str, str] = {}
    for arm in arms:
        if not (
            tuple(arm.get("families", ())) == _FAMILIES
            and arm.get("objective_call_semantics") == _OBJECTIVE_SEMANTICS
            and arm.get("attempt_history_semantics") == _ATTEMPT_SEMANTICS
        ):
            raise ValueError("An arm differs from the common execution semantics.")
        adapter_status = str(arm.get("execution_adapter_status"))
        if adapter_status not in {
            "DEVELOPMENT_ONLY_AVAILABLE",
            "NOT_IMPLEMENTED",
            "NOT_AVAILABLE_CANDIDATE_NOT_SELECTED",
        }:
            raise ValueError("An arm has an invalid execution adapter status.")
        adapter_status_by_arm[str(arm["arm_id"])] = adapter_status
        source_bindings = arm.get("source_bindings")
        if not isinstance(source_bindings, list) or not source_bindings:
            raise ValueError("Every arm requires nonempty source bindings.")
        arm_hashes: list[str] = []
        for binding in source_bindings:
            source_path = artifact_root.resolve_binding(binding)
            arm_hashes.append(hashlib.sha256(source_path.read_bytes()).hexdigest())
        if len(arm_hashes) != len(set(arm_hashes)):
            raise ValueError("An arm repeats a source binding.")
        source_hashes[str(arm["arm_id"])] = arm_hashes

    budgets = tuple(context.get("evaluation_budgets", ()))
    if budgets != (10_000, 50_000, 100_000):
        raise ValueError("The formal budget grid is not the frozen V21e3 grid.")
    checkpoints = tuple(float(value) for value in context.get("checkpoint_fractions", ()))
    if (
        not checkpoints
        or checkpoints[-1] != 1.0
        or any(not 0.0 < value <= 1.0 for value in checkpoints)
        or any(left >= right for left, right in zip(checkpoints, checkpoints[1:]))
    ):
        raise ValueError("The common genuine checkpoint grid is invalid.")

    timing = load_timing_policy(
        artifact_root,
        context.get("timing_policy", {}),
    )
    trace_path = artifact_root.resolve_binding(
        context.get("trace_storage_policy", {})
    )
    trace_raw = trace_path.read_bytes()
    trace = json.loads(trace_raw)
    if trace != {
        "schema": "pareto_v21e3_trace_storage_policy_v1",
        "implementation_status": "PROTOTYPE_ONLY",
        "formal_authorized": False,
        "codec": "canonical_jsonl_zlib_chunk_v1",
        "chunk_hash_chain": "sha256_header_and_compressed_payload_v1",
        "restore_replay_gate": "REQUIRED_BEFORE_FORMAL",
    }:
        raise ValueError("Trace storage is not the frozen prototype policy.")

    return {
        "schema": "pareto_v21e3_formal_common_runner_preflight_receipt_v1",
        "status": "PROTOTYPE_PASS_FORMAL_PROHIBITED",
        "scientific_scope": "engineering_preflight_not_formal_evidence",
        "artifact_root_id": context["artifact_root_id"],
        "context_sha256": hashlib.sha256(context_raw).hexdigest(),
        "arm_ids": list(_ARMS),
        "families": list(_FAMILIES),
        "source_sha256_by_arm": source_hashes,
        "execution_adapter_status_by_arm": adapter_status_by_arm,
        "common_budget_adapter_complete": False,
        "common_budget_parity_status": "NOT_ESTABLISHED",
        "objective_call_semantics": _OBJECTIVE_SEMANTICS,
        "attempt_history_semantics": _ATTEMPT_SEMANTICS,
        "evaluation_budgets": list(budgets),
        "checkpoint_fractions": list(checkpoints),
        "timing_policy_sha256": timing.sha256,
        "timing_semantic_role": timing.semantic_role,
        "trace_storage_policy_sha256": hashlib.sha256(trace_raw).hexdigest(),
        "future_external_entropy_status": "NOT_ESTABLISHED",
        "formal_cases_status": "NOT_MATERIALIZED",
        "formal_authorized": False,
        "runner_execution_available": False,
    }


def run_formal_matrix(_preflight_receipt: Mapping[str, object]) -> None:
    """Never execute a formal matrix from this prototype revision."""

    raise FormalMaterializationProhibited(
        "V21e3 formal materialization is prohibited: the common runner is a "
        "preflight prototype, future entropy is not established, and no formal "
        "case manifest exists."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report the fail-closed V21e3 common-runner status."
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if not args.status:
        parser.error("only --status is available in the prototype")
    print(
        json.dumps(
            {
                "schema": "pareto_v21e3_common_runner_cli_status_v1",
                "status": "PROTOTYPE_ONLY_FORMAL_MATERIALIZATION_PROHIBITED",
                "formal_cases_status": "NOT_MATERIALIZED",
                "formal_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FormalMaterializationProhibited",
    "preflight_development_parity_protocol_v2",
    "preflight_formal_common_runner",
    "run_formal_matrix",
]
