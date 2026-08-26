"""Canonical Pareto-SMC v18 theorem packet.

The packet removes three unproved inputs accepted by earlier revisions:

* a caller may not supply ``epsilon``; minorization is derived from the raw
  independence-MH mixture and a frozen energy-span bound;
* simultaneous confirm risk is computed from a dominated categorical lower
  law by exact inclusion--exclusion, not only by a union bound;
* nonregular endpoint laws use an epsilon-PAC branch rather than pretending
  that a tied exact-best problem has positive characteristic information.

Optional exact small-state reference enumeration upgrades a frozen-reference
claim to a true-front-relative claim.  Large-state claims remain reference
relative unless such an artifact is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping, Sequence

from .pareto_v17_regeneration import (
    as_fraction,
    target_probability_lower_from_endpoint,
)
from .pareto_v17_track_and_stop import (
    exact_track_stop_decision,
    time_uniform_lower_matrix,
)
from .pareto_v18_minorization import (
    DerivedTypeMinorization,
    exact_fraction_payload,
    parse_type_minorization,
)
from .pareto_v18_nonregular import epsilon_pac_selection
from .pareto_v18_occupancy import (
    MultiTypeOccupancyProblem,
    exact_minimum_cost_occupancy_allocation,
)
from .pareto_v18_practicality import build_practicality_certificate
from .pareto_v18_kernel_perturbation import (
    build_kernel_perturbation_certificate,
    ideal_probability_lower_from_implementation,
    implementation_probability_lower,
)
from .pareto_v18_reference_completeness import (
    certify_exact_tsp_reference_completeness,
    compose_additive_error,
)
from .pareto_v18_reference_branch_bound import (
    certify_reference_cover_branch_and_bound,
)


class CanonicalV18PacketError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CanonicalV18PacketError("packet is not canonical-JSON serializable") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _fraction_matrix_strings(matrix: Sequence[Sequence[Fraction]]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _parse_categorical_matrix(raw: Sequence[Sequence[object]]) -> tuple[tuple[Fraction, ...], ...]:
    matrix = tuple(tuple(as_fraction(value) for value in row) for row in raw)
    if not matrix or len(matrix[0]) < 2 or any(len(row) != len(matrix[0]) for row in matrix):
        raise CanonicalV18PacketError("source endpoint model is malformed")
    if any(value < 0 or value > 1 for row in matrix for value in row):
        raise CanonicalV18PacketError("source endpoint probabilities must lie in [0,1]")
    if any(sum(row, Fraction(0, 1)) != 1 for row in matrix):
        raise CanonicalV18PacketError("source endpoint rows must sum to one")
    return matrix


@dataclass(frozen=True)
class CanonicalV18Result:
    packet_sha256: str
    context_sha256: str
    minorization_provenance_pass: bool
    selection_mode: str
    selection_pass: bool
    confirm_plan_optimal: bool
    confirm_counts: tuple[int, ...]
    confirm_exact_miss_upper: Fraction
    confirm_union_miss_upper: Fraction
    false_pass_upper: Fraction
    reference_scope: str
    practicality_verdict: str
    operational_randomness_status: str
    machine_formalization_status: str
    overall_theory_packet_pass: bool
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pareto_smc_v18_canonical_result_v1",
            "packet_sha256": self.packet_sha256,
            "context_sha256": self.context_sha256,
            "minorization_provenance_pass": self.minorization_provenance_pass,
            "selection_mode": self.selection_mode,
            "selection_pass": self.selection_pass,
            "confirm_plan_optimal": self.confirm_plan_optimal,
            "confirm_counts": list(self.confirm_counts),
            "confirm_exact_miss_upper": str(self.confirm_exact_miss_upper),
            "confirm_union_miss_upper": str(self.confirm_union_miss_upper),
            "false_pass_upper": str(self.false_pass_upper),
            "reference_scope": self.reference_scope,
            "practicality_verdict": self.practicality_verdict,
            "operational_randomness_status": self.operational_randomness_status,
            "machine_formalization_status": self.machine_formalization_status,
            "overall_theory_packet_pass": self.overall_theory_packet_pass,
            "report": dict(self.report),
        }


def build_canonical_v18_packet(raw: Mapping[str, Any]) -> CanonicalV18Result:
    if raw.get("schema") != "pareto_smc_v18_canonical_packet_v2":
        raise CanonicalV18PacketError("unsupported v18 packet schema")
    context = raw.get("context")
    if not isinstance(context, Mapping):
        raise CanonicalV18PacketError("context must be a mapping")
    context_sha = _sha256(context)
    if raw.get("context_sha256") != context_sha:
        raise CanonicalV18PacketError("context SHA-256 mismatch")

    objective_dimension = context.get("objective_dimension")
    if isinstance(objective_dimension, bool) or not isinstance(objective_dimension, int) or objective_dimension <= 0:
        raise CanonicalV18PacketError("context must declare a positive objective_dimension")

    type_ids = tuple(str(value) for value in raw.get("type_ids", ()))
    cell_ids = tuple(str(value) for value in raw.get("cell_ids", ()))
    if not type_ids or len(type_ids) != len(set(type_ids)):
        raise CanonicalV18PacketError("type IDs must be nonempty and unique")
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise CanonicalV18PacketError("cell IDs must be nonempty and unique")

    minorization_raw = raw.get("minorization")
    if not isinstance(minorization_raw, Sequence) or isinstance(minorization_raw, (str, bytes)):
        raise CanonicalV18PacketError("minorization section must be a sequence")
    if len(minorization_raw) != len(type_ids):
        raise CanonicalV18PacketError("one source-derived minorization entry is required per type")
    derived: tuple[DerivedTypeMinorization, ...] = tuple(
        parse_type_minorization(item, expected_type_id=type_id, context_sha256=context_sha)
        for type_id, item in zip(type_ids, minorization_raw, strict=True)
    )

    if any(len(item.potential.reference_weights) != objective_dimension for item in derived):
        raise CanonicalV18PacketError("reference-weight dimension differs from the objective dimension")

    perturbations = []
    for item in minorization_raw:
        if not isinstance(item, Mapping):
            raise CanonicalV18PacketError("minorization entry must be a mapping")
        perturbations.append((
            build_kernel_perturbation_certificate(item.get("pilot_kernel_arithmetic")),
            build_kernel_perturbation_certificate(item.get("confirm_kernel_arithmetic")),
        ))

    pilot = raw.get("pilot")
    if not isinstance(pilot, Mapping):
        raise CanonicalV18PacketError("pilot section is missing")
    pilot_sampling_model = str(pilot.get("sampling_model"))
    allowed_pilot_model = "independent_complete_interacting_run_replicas_one_endpoint_per_run"
    if pilot_sampling_model != allowed_pilot_model:
        raise CanonicalV18PacketError(
            "Track-and-Stop/mass inference requires independent complete-run replicas "
            "with one designated endpoint per run"
        )
    counts = tuple(tuple(int(value) for value in row) for row in pilot.get("counts", ()))
    if len(counts) != len(type_ids) or any(len(row) != len(cell_ids) + 1 for row in counts):
        raise CanonicalV18PacketError("pilot count dimensions differ from types/cells")
    if any(value < 0 for row in counts for value in row) or any(sum(row) <= 0 for row in counts):
        raise CanonicalV18PacketError("pilot counts must be nonnegative with one sample per type")
    mass_alpha = as_fraction(pilot.get("mass_alpha"))
    if not (Fraction(0, 1) < mass_alpha < Fraction(1, 1)):
        raise CanonicalV18PacketError("mass_alpha must lie in (0,1)")
    endpoint_lower = time_uniform_lower_matrix(
        counts,
        mass_alpha,
        denominator=int(pilot.get("rational_radius_denominator", 10**12)),
    )

    target_rows: list[tuple[Fraction, ...]] = []
    confirm_rows: list[tuple[Fraction, ...]] = []
    for r, item in enumerate(derived):
        pilot_perturbation, confirm_perturbation = perturbations[r]
        ideal_pilot_endpoint_lower = tuple(
            ideal_probability_lower_from_implementation(
                value, pilot_perturbation.horizon_tv_upper
            )
            for value in endpoint_lower[r]
        )
        target = tuple(
            target_probability_lower_from_endpoint(value, item.pilot_residual_upper)
            for value in ideal_pilot_endpoint_lower
        )
        ideal_confirm = tuple(
            (Fraction(1, 1) - item.confirm_residual_upper) * value
            for value in target
        )
        confirm = tuple(
            implementation_probability_lower(
                value, confirm_perturbation.horizon_tv_upper
            )
            for value in ideal_confirm
        )
        if sum(confirm, Fraction(0, 1)) > 1:
            raise CanonicalV18PacketError("derived confirm lower row is not categorical")
        target_rows.append(target)
        confirm_rows.append(confirm)
    target_lower = tuple(target_rows)
    confirm_lower = tuple(confirm_rows)

    selection = pilot.get("selection")
    if not isinstance(selection, Mapping):
        raise CanonicalV18PacketError("pilot selection contract is missing")
    selection_mode = str(selection.get("mode"))
    selection_report: dict[str, Any]
    if selection_mode == "exact_unique_track_and_stop":
        delta = as_fraction(selection.get("delta"))
        decision = exact_track_stop_decision(counts, delta)
        selection_pass = bool(decision["stopped"])
        selection_report = {
            "mode": selection_mode,
            "decision": decision,
            "claim": "finite_time_delta_correct_exact_best_at_unique_answer_models",
            "asymptotic_instance_optimality": "requires_separate_regular_model_proof",
        }
    elif selection_mode == "epsilon_pac":
        epsilons = selection.get("epsilon_by_cell")
        if not isinstance(epsilons, Sequence) or isinstance(epsilons, (str, bytes)):
            raise CanonicalV18PacketError("epsilon_pac mode requires epsilon_by_cell")
        certificate = epsilon_pac_selection(
            counts,
            epsilons,
            as_fraction(selection.get("alpha")),
            denominator=int(selection.get("rational_radius_denominator", 10**12)),
        )
        selection_pass = certificate.all_cells_certified
        selection_report = certificate.to_dict()
    else:
        raise CanonicalV18PacketError("unsupported pilot selection mode")

    confirm = raw.get("confirm")
    if not isinstance(confirm, Mapping):
        raise CanonicalV18PacketError("confirm section is missing")
    confirm_sampling_model = str(confirm.get("sampling_model"))
    if confirm_sampling_model not in {
        "independent_complete_interacting_run_replicas_one_endpoint_per_run",
        "conditionally_independent_private_final_tapes_given_preblock_history",
    }:
        raise CanonicalV18PacketError("unsupported confirm endpoint independence contract")
    confirm_delta = as_fraction(confirm.get("delta"))
    costs = tuple(as_fraction(value) for value in confirm.get("costs", ()))
    if len(costs) != len(type_ids):
        raise CanonicalV18PacketError("one confirm cost is required per type")
    problem = MultiTypeOccupancyProblem(
        endpoint_lower=confirm_lower,
        costs=costs,
        delta=confirm_delta,
        max_cells_exact=int(confirm.get("max_cells_exact", 14)),
        max_subset_terms=int(confirm.get("max_subset_terms", 1 << 14)),
    )
    plan = exact_minimum_cost_occupancy_allocation(
        problem,
        max_nodes=int(confirm.get("planner_max_nodes", 2_000_000)),
        max_greedy_steps=int(confirm.get("planner_max_greedy_steps", 1_000_000)),
    )

    source_model_raw = raw.get("source_endpoint_probability_model")
    practicality_report: dict[str, Any] = {
        "status": "UNRESOLVED_WITHOUT_SOURCE_ENDPOINT_UPPER_MODEL"
    }
    practicality_verdict = "UNRESOLVED"
    if source_model_raw is not None:
        if not isinstance(source_model_raw, Mapping):
            raise CanonicalV18PacketError(
                "source endpoint model must include probabilities and explicit provenance"
            )
        provenance = str(source_model_raw.get("provenance"))
        if provenance not in {
            "theorem_parameter_upper_bound",
            "exact_finite_state_enumeration",
            "independently_verified_upper_bound",
        }:
            raise CanonicalV18PacketError("unsupported source endpoint model provenance")
        proof_sha = source_model_raw.get("proof_sha256")
        if provenance != "theorem_parameter_upper_bound" and (
            not isinstance(proof_sha, str) or len(proof_sha) != 64
        ):
            raise CanonicalV18PacketError("verified source endpoint model requires proof_sha256")
        probabilities_raw = source_model_raw.get("probabilities")
        if not isinstance(probabilities_raw, Sequence) or isinstance(probabilities_raw, (str, bytes)):
            raise CanonicalV18PacketError("source endpoint probabilities are missing")
        source_model = _parse_categorical_matrix(probabilities_raw)
        if len(source_model) != len(type_ids) or len(source_model[0]) != len(cell_ids) + 1:
            raise CanonicalV18PacketError("source endpoint model dimensions differ")
        source_cell = tuple(tuple(row[j] for j in range(1, len(row))) for row in source_model)
        practicality = build_practicality_certificate(
            source_cell,
            costs,
            plan.counts,
            simultaneous_failure_target=confirm_delta,
            evaluation_budget=as_fraction(raw.get("evaluation_budget", plan.total_cost)),
            max_cells_exact=int(confirm.get("max_cells_exact", 14)),
        )
        practicality_report = practicality.to_dict() | {
            "source_model_provenance": provenance,
            "source_model_proof_sha256": proof_sha,
            "scientific_status": (
                practicality.verdict
                if provenance != "theorem_parameter_upper_bound"
                else "CONDITIONAL_ON_UNVERIFIED_THEOREM_PARAMETER_UPPER_BOUND"
            ),
        }
        practicality_verdict = (
            practicality.verdict
            if provenance != "theorem_parameter_upper_bound"
            else "CONDITIONAL"
        )

    reference_scope = "frozen_reference_relative_only"
    reference_report: dict[str, Any] = {
        "scope": reference_scope,
        "true_front_claim_authorized": False,
    }
    reference_raw = raw.get("reference_completeness")
    if reference_raw is not None:
        if not isinstance(reference_raw, Mapping):
            raise CanonicalV18PacketError("reference_completeness must be a mapping")
        reference_mode = reference_raw.get("mode")
        if reference_mode == "exact_fixed_origin_tsp_enumeration":
            certificate = certify_exact_tsp_reference_completeness(
                reference_raw["objective_matrices"],
                reference_raw["frozen_reference"],
                reference_raw["additive_eta"],
                max_tours=int(reference_raw.get("max_tours", 2_000_000)),
            )
            certified_reference_scope = "exact_enumerated_true_front_relative"
        elif reference_mode == "fixed_origin_tsp_min_outgoing_branch_and_bound":
            certificate = certify_reference_cover_branch_and_bound(
                reference_raw["objective_matrices"],
                reference_raw["reference_witnesses"],
                reference_raw["additive_eta"],
                max_nodes=int(reference_raw.get("max_nodes", 2_000_000)),
            )
            certified_reference_scope = "branch_and_bound_true_front_relative"
        else:
            raise CanonicalV18PacketError("unsupported executable reference-completeness mode")
        algorithm_epsilon = reference_raw.get("algorithm_epsilon")
        if not isinstance(algorithm_epsilon, Sequence) or isinstance(algorithm_epsilon, (str, bytes)):
            raise CanonicalV18PacketError("reference completeness requires algorithm_epsilon")
        composed = compose_additive_error(certificate.additive_eta, algorithm_epsilon)
        reference_scope = certified_reference_scope
        reference_report = certificate.to_dict() | {
            "algorithm_epsilon": [str(as_fraction(value)) for value in algorithm_epsilon],
            "composed_additive_error": [str(value) for value in composed],
            "true_front_claim_authorized": True,
        }

    false_pass_upper = min(Fraction(1, 1), mass_alpha + confirm_delta)
    overall = (
        selection_pass
        and plan.optimal
        and plan.exact_miss_upper <= confirm_delta
    )
    report: dict[str, Any] = {
        "minorization": {
            "provenance": "derived_from_independence_mh_mixture_and_energy_span",
            "types": [
                {
                    "type_id": item.type_id,
                    "final_target_sha256": item.target_sha256,
                    "potential_contract": item.potential.to_dict(),
                    "ideal_kernel_contract": item.ideal_kernel_contract.to_dict(),
                    "ideal_kernel_contract_sha256": item.ideal_kernel_contract.sha256,
                    "pilot_final_blocks": [block.to_dict() for block in item.pilot],
                    "confirm_final_blocks": [block.to_dict() for block in item.confirm],
                    "pilot_residual_upper": exact_fraction_payload(item.pilot_residual_upper),
                    "confirm_residual_upper": exact_fraction_payload(item.confirm_residual_upper),
                    "pilot_kernel_perturbation": perturbations[index][0].to_dict(),
                    "confirm_kernel_perturbation": perturbations[index][1].to_dict(),
                }
                for index, item in enumerate(derived)
            ],
        },
        "sampling_contract": {
            "pilot": pilot_sampling_model,
            "confirm": confirm_sampling_model,
        },
        "selection": selection_report,
        "pilot_endpoint_lower_matrix": _fraction_matrix_strings(endpoint_lower),
        "target_cell_lower_matrix": _fraction_matrix_strings(target_lower),
        "confirm_endpoint_lower_matrix": _fraction_matrix_strings(confirm_lower),
        "confirm_plan": plan.to_dict(),
        "false_pass_semantics": {
            "event": "PACKET_PASS_AND_SIMULTANEOUS_CONFIRM_CELL_MISS",
            "upper_bound": str(false_pass_upper),
            "conditional_probability_given_pass_claimed": False,
        },
        "practicality": practicality_report,
        "reference": reference_report,
        "operational_randomness": {
            "status": "SEPARATE_STUDY_LEVEL_EXTERNAL_CONTROL_REQUIRED",
            "local_packet_does_not_prove": [
                "future_beacon_unpredictability",
                "global_log_non_equivocation",
                "absence_of_unlogged_restarts",
            ],
        },
        "machine_formalization": {
            "status": "NOT_PERFORMED",
            "paper_and_python_tests_are_not_machine_proofs": True,
        },
    }
    return CanonicalV18Result(
        packet_sha256=_sha256(raw),
        context_sha256=context_sha,
        minorization_provenance_pass=True,
        selection_mode=selection_mode,
        selection_pass=selection_pass,
        confirm_plan_optimal=plan.optimal,
        confirm_counts=plan.counts,
        confirm_exact_miss_upper=plan.exact_miss_upper,
        confirm_union_miss_upper=plan.union_miss_upper,
        false_pass_upper=false_pass_upper,
        reference_scope=reference_scope,
        practicality_verdict=practicality_verdict,
        operational_randomness_status="CONDITIONAL_EXTERNAL_PROTOCOL_REQUIRED",
        machine_formalization_status="NOT_PERFORMED",
        overall_theory_packet_pass=overall,
        report=report,
    )


__all__ = [
    "CanonicalV18PacketError",
    "CanonicalV18Result",
    "build_canonical_v18_packet",
]
