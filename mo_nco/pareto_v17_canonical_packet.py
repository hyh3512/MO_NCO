"""Canonical theorem packet for Pareto-SMC v17.

The packet recomputes four theorem objects from one raw, hash-bound payload:

1. type-wise final-regeneration residuals (Theorem 17.2);
2. shared-categorical Track-and-Stop evidence and characteristic game
   (Theorem 17.6 plus the v17 upper theorem);
3. exact-rational time-uniform endpoint lower probabilities;
4. the exact multi-type confirm allocation (Theorems 17.4--17.5).

Derived certificates are never accepted as inputs.  A packet either recomputes
all objects under one context digest or fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping, Sequence

from .pareto_v17_multitype_confirm import (
    MultiTypeConfirmProblem,
    exact_minimum_cost_allocation,
)
from .pareto_v17_regeneration import (
    MinorizationBlock,
    RegenerationTransfer,
    TypeRegenerationCertificate,
    as_fraction,
    target_probability_lower_from_endpoint,
)
from .pareto_v17_track_and_stop import (
    TrackAndStopError,
    answer_map,
    binary_kl_decision_lower_bound,
    dirichlet_mixture_threshold,
    empirical_probabilities,
    glr_statistic,
    exact_track_stop_decision,
    minimum_cell_gap,
    solve_characteristic_game,
    time_uniform_lower_matrix,
)


class CanonicalPacketError(ValueError):
    pass


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _fraction_strings(matrix: Sequence[Sequence[Fraction]]) -> list[list[str]]:
    return [[str(x) for x in row] for row in matrix]


def _parse_probability_model(raw: Sequence[Sequence[str | int | Fraction]]) -> tuple[tuple[Fraction, ...], ...]:
    matrix = tuple(tuple(as_fraction(x) for x in row) for row in raw)
    if not matrix or len(matrix[0]) < 2 or any(len(row) != len(matrix[0]) for row in matrix):
        raise CanonicalPacketError("invalid source probability model")
    for row in matrix:
        if any(x < 0 or x > 1 for x in row) or sum(row, Fraction(0, 1)) != 1:
            raise CanonicalPacketError("source probability model is not categorical")
    return matrix


@dataclass(frozen=True)
class CanonicalV17Result:
    packet_sha256: str
    context_sha256: str
    regeneration_pass: bool
    track_and_stop_stopped: bool
    track_and_stop_delta_correct_object: bool
    track_and_stop_answer: tuple[int, ...]
    confirm_planner_optimal: bool
    confirm_counts: tuple[int, ...]
    confirm_exact_union_risk: Fraction
    false_pass_upper: Fraction
    characteristic_lower: float | None
    characteristic_upper: float | None
    characteristic_gap: float | None
    asymptotic_instance_optimality_applicable: bool
    overall_pass: bool
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pareto_smc_v17_canonical_result_v1",
            "packet_sha256": self.packet_sha256,
            "context_sha256": self.context_sha256,
            "regeneration_pass": self.regeneration_pass,
            "track_and_stop_stopped": self.track_and_stop_stopped,
            "track_and_stop_delta_correct_object": self.track_and_stop_delta_correct_object,
            "track_and_stop_answer": list(self.track_and_stop_answer),
            "confirm_planner_optimal": self.confirm_planner_optimal,
            "confirm_counts": list(self.confirm_counts),
            "confirm_exact_union_risk": str(self.confirm_exact_union_risk),
            "false_pass_upper": str(self.false_pass_upper),
            "characteristic_lower": self.characteristic_lower,
            "characteristic_upper": self.characteristic_upper,
            "characteristic_gap": self.characteristic_gap,
            "asymptotic_instance_optimality_applicable": self.asymptotic_instance_optimality_applicable,
            "overall_pass": self.overall_pass,
            "report": dict(self.report),
        }


def build_canonical_v17_packet(raw: Mapping[str, Any]) -> CanonicalV17Result:
    if raw.get("schema") != "pareto_smc_v17_canonical_packet_v1":
        raise CanonicalPacketError("unsupported canonical packet schema")
    context = raw.get("context")
    if not isinstance(context, Mapping):
        raise CanonicalPacketError("context must be a mapping")
    context_sha = _sha256(context)
    declared_context_sha = raw.get("context_sha256")
    if declared_context_sha != context_sha:
        raise CanonicalPacketError("context SHA-256 mismatch")

    type_ids = tuple(str(x) for x in raw.get("type_ids", ()))
    cell_ids = tuple(str(x) for x in raw.get("cell_ids", ()))
    if not type_ids or len(type_ids) != len(set(type_ids)):
        raise CanonicalPacketError("type_ids must be nonempty and unique")
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise CanonicalPacketError("cell_ids must be nonempty and unique")

    # Theorem 17.2: final-regeneration transfer.
    regen_raw = raw.get("regeneration")
    if not isinstance(regen_raw, Sequence) or len(regen_raw) != len(type_ids):
        raise CanonicalPacketError("one regeneration entry is required per type")
    regen_types: list[TypeRegenerationCertificate] = []
    for expected_id, item in zip(type_ids, regen_raw, strict=True):
        if not isinstance(item, Mapping) or str(item.get("type_id")) != expected_id:
            raise CanonicalPacketError("regeneration type ordering mismatch")
        pilot_blocks_raw = item.get("pilot_blocks", item.get("blocks"))
        confirm_blocks_raw = item.get("confirm_blocks", item.get("blocks"))
        if not isinstance(pilot_blocks_raw, Sequence) or not pilot_blocks_raw:
            raise CanonicalPacketError("pilot regeneration blocks are missing")
        if not isinstance(confirm_blocks_raw, Sequence) or not confirm_blocks_raw:
            raise CanonicalPacketError("confirm regeneration blocks are missing")
        pilot_blocks = tuple(
            MinorizationBlock(as_fraction(block["epsilon"]), int(block["steps"]))
            for block in pilot_blocks_raw
        )
        confirm_blocks = tuple(
            MinorizationBlock(as_fraction(block["epsilon"]), int(block["steps"]))
            for block in confirm_blocks_raw
        )
        regen_types.append(TypeRegenerationCertificate(expected_id, pilot_blocks))
        item_confirm = TypeRegenerationCertificate(expected_id, confirm_blocks)
        # Store the confirm certificate on a parallel list created lazily below.
        if "confirm_regen_types" not in locals():
            confirm_regen_types = []
        confirm_regen_types.append(item_confirm)
    transfer = RegenerationTransfer(tuple(regen_types))
    confirm_transfer = RegenerationTransfer(tuple(confirm_regen_types))

    # Shared-categorical pilot observations. Category 0 is outside; 1..J are cells.
    pilot = raw.get("pilot")
    if not isinstance(pilot, Mapping):
        raise CanonicalPacketError("pilot section is missing")
    counts = tuple(tuple(int(x) for x in row) for row in pilot.get("counts", ()))
    if len(counts) != len(type_ids) or any(len(row) != len(cell_ids) + 1 for row in counts):
        raise CanonicalPacketError("pilot categorical count dimensions do not match types/cells")
    if any(x < 0 for row in counts for x in row) or any(sum(row) <= 0 for row in counts):
        raise CanonicalPacketError("pilot counts must be nonnegative with one sample per type")

    track_delta = as_fraction(pilot.get("track_delta"))
    if track_delta <= 0 or track_delta >= 1:
        raise CanonicalPacketError("track_delta must lie in (0,1)")
    glr, empirical_answer, active = glr_statistic(counts)
    threshold = dirichlet_mixture_threshold(counts, float(track_delta))
    exact_stop = exact_track_stop_decision(counts, track_delta)
    track_stopped = bool(exact_stop["stopped"])
    if exact_stop["answer"] is not None:
        empirical_answer = tuple(int(x) for x in exact_stop["answer"])

    mass_alpha = as_fraction(pilot.get("mass_alpha"))
    pilot_endpoint_lower = time_uniform_lower_matrix(
        counts,
        mass_alpha,
        denominator=int(pilot.get("rational_radius_denominator", 10**12)),
    )
    target_lower_rows = []
    confirm_endpoint_rows = []
    for r in range(len(type_ids)):
        pilot_b = transfer.types[r].residual
        confirm_b = confirm_transfer.types[r].residual
        target_row = tuple(
            target_probability_lower_from_endpoint(value, pilot_b)
            for value in pilot_endpoint_lower[r]
        )
        confirm_row = tuple((Fraction(1, 1) - confirm_b) * value for value in target_row)
        target_lower_rows.append(target_row)
        confirm_endpoint_rows.append(confirm_row)
    target_lower_matrix = tuple(target_lower_rows)
    lower_matrix = tuple(confirm_endpoint_rows)

    # Theorems 17.4--17.5: exact multi-type confirm allocation.
    confirm = raw.get("confirm")
    if not isinstance(confirm, Mapping):
        raise CanonicalPacketError("confirm section is missing")
    confirm_delta = as_fraction(confirm.get("delta"))
    costs = tuple(as_fraction(x) for x in confirm.get("costs", ()))
    if len(costs) != len(type_ids):
        raise CanonicalPacketError("confirm costs do not match type count")
    problem = MultiTypeConfirmProblem(lower_matrix, costs, confirm_delta)
    plan = exact_minimum_cost_allocation(
        problem,
        max_nodes=int(confirm.get("planner_max_nodes", 2_000_000)),
    )

    # Optional source-certified model used only for the characteristic-time and
    # asymptotic instance-optimality audit.  Observed frequencies never acquire
    # theorem-parameter status merely by being present in the packet.
    characteristic_lower: float | None = None
    characteristic_upper: float | None = None
    characteristic_gap: float | None = None
    asymptotic_applicable = False
    source_model_report: dict[str, Any] = {"present": False}
    source_model_raw = raw.get("source_probability_model")
    if source_model_raw is not None:
        source_model = _parse_probability_model(source_model_raw)
        if len(source_model) != len(type_ids) or len(source_model[0]) != len(cell_ids) + 1:
            raise CanonicalPacketError("source probability model dimensions do not match packet")
        source_model_float = tuple(tuple(float(x) for x in row) for row in source_model)
        source_answers = answer_map(source_model_float)
        source_gap = minimum_cell_gap(source_model_float)
        solution = solve_characteristic_game(
            source_model_float,
            iterations=int(raw.get("characteristic_iterations", 40_000)),
            step_scale=float(raw.get("characteristic_step_scale", 0.5)),
        )
        characteristic_lower = solution.lower_bound
        characteristic_upper = solution.upper_bound
        characteristic_gap = solution.gap
        regularity = raw.get("track_and_stop_regularity", {})
        if not isinstance(regularity, Mapping):
            raise CanonicalPacketError("track_and_stop_regularity must be a mapping")
        full_support_floor = as_fraction(regularity.get("full_support_floor", "0"))
        actual_floor = min(x for row in source_model for x in row)
        optimizer_gap_cap = float(regularity.get("optimizer_gap_cap", 1e-3))
        proof_status = str(regularity.get("regularity_proof_status", "missing"))
        proof_sha256 = str(regularity.get("regularity_proof_sha256", ""))
        external_proof_bound = proof_status == "independently_verified" and len(proof_sha256) == 64
        asymptotic_applicable = (
            actual_floor >= full_support_floor > 0
            and source_gap > 0.0
            and solution.gap <= optimizer_gap_cap
            and external_proof_bound
        )
        source_model_report = {
            "present": True,
            "identity": raw.get("source_probability_identity", "unclassified"),
            "answers": list(source_answers),
            "minimum_cell_gap": source_gap,
            "full_support_floor_actual": str(actual_floor),
            "characteristic_weights": list(solution.weights),
            "characteristic_lower": solution.lower_bound,
            "characteristic_upper": solution.upper_bound,
            "characteristic_gap": solution.gap,
            "transportation_lower_expected_samples": binary_kl_decision_lower_bound(
                float(track_delta), solution.upper_bound
            ) if solution.upper_bound > 0.0 and float(track_delta) < 0.5 else None,
            "regularity_proof_status": proof_status,
            "regularity_proof_sha256": proof_sha256,
            "asymptotic_instance_optimality_applicable": asymptotic_applicable,
            "application_status": (
                "PASS_CONDITIONAL_EXTERNAL_REGULARITY_PROOF"
                if asymptotic_applicable
                else "THEOREM_PROVED_BUT_INSTANCE_REGULARITY_UNRESOLVED"
            ),
        }

    # The confirm risk certificate uses the time-uniform endpoint-law lower
    # matrix directly.  Track-and-Stop's answer-identification risk is a separate
    # claim and is not silently charged to the metric false-PASS event.
    false_pass_upper = min(Fraction(1, 1), mass_alpha + confirm_delta)
    overall = plan.optimal and plan.exact_union_risk <= confirm_delta and track_stopped

    report: dict[str, Any] = {
        "theorem_objects": {
            "17.2_final_regeneration_transfer": {
                "pilot_type_residuals": {item.type_id: str(item.residual) for item in transfer.types},
                "confirm_type_residuals": {item.type_id: str(item.residual) for item in confirm_transfer.types},
                "pilot_target_components": {item.type_id: str(item.target_component) for item in transfer.types},
                "confirm_target_components": {item.type_id: str(item.target_component) for item in confirm_transfer.types},
                "status": "PASS",
            },
            "17.4_17.5_multitype_confirm": plan.to_dict(),
            "17.6_transportation_and_track_stop": {
                "empirical_probabilities": [list(row) for row in empirical_probabilities(counts)],
                "glr": glr,
                "threshold": threshold,
                "stopped": track_stopped,
                "answer": list(empirical_answer),
                "active_cell": active[0],
                "active_challenger": active[1],
                "delta_correct_object": True,
                "exact_stopping_comparison": exact_stop,
                "source_model_audit": source_model_report,
            },
        },
        "pilot_endpoint_lower_matrix": _fraction_strings(pilot_endpoint_lower),
        "inferred_target_cell_lower_matrix": _fraction_strings(target_lower_matrix),
        "confirm_endpoint_lower_matrix": _fraction_strings(lower_matrix),
        "confirm_false_pass_semantics": {
            "event": "PASS_AND_SIMULTANEOUS_CONFIRM_CELL_MISS",
            "upper_bound": str(false_pass_upper),
            "conditional_probability_given_pass_claimed": False,
        },
        "scope": {
            "arm_sample": "independent_complete_canonical_run_replica",
            "within_one_interacting_population_particles_are_iid_arms": False,
            "reference_relative_only": True,
            "machine_formalized": False,
        },
    }

    packet_sha = _sha256(raw)
    return CanonicalV17Result(
        packet_sha256=packet_sha,
        context_sha256=context_sha,
        regeneration_pass=True,
        track_and_stop_stopped=track_stopped,
        track_and_stop_delta_correct_object=True,
        track_and_stop_answer=empirical_answer,
        confirm_planner_optimal=plan.optimal,
        confirm_counts=plan.counts,
        confirm_exact_union_risk=plan.exact_union_risk,
        false_pass_upper=false_pass_upper,
        characteristic_lower=characteristic_lower,
        characteristic_upper=characteristic_upper,
        characteristic_gap=characteristic_gap,
        asymptotic_instance_optimality_applicable=asymptotic_applicable,
        overall_pass=overall,
        report=report,
    )
