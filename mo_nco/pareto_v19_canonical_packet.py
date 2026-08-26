"""Canonical v19 extension packet for the Pareto-SMC theorem stack.

V19 never accepts caller-supplied child PASS booleans.  It first recomputes the
complete v18 canonical packet and then derives the selected v19 extension
certificates from raw arithmetic, bridge, occupancy, reference and
Track-and-Stop inputs.  The root hash binds every section.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .pareto_v18_canonical_packet import build_canonical_v18_packet
from .pareto_v17_regeneration import as_fraction
from .pareto_v19_blocked_occupancy import (
    BlockedOccupancyProblem,
    exact_minimum_cost_blocked_allocation,
)
from .pareto_v19_exact_mh import exp_neg_rational_interval
from .pareto_v19_kernel_perturbation import build_automatic_kernel_tv_certificate
from .pareto_v19_multilevel_bridge import (
    build_multilevel_bridge_certificate,
    exact_minimum_cost_bridge_plan,
)
from .pareto_v19_reference_grid_oracle import (
    GridOracleRecord,
    build_geometric_reference_certificate,
)
from .pareto_v19_track_stop_deadline import build_track_stop_deadline_certificate
from .pareto_v19_track_stop_regularized import (
    InformationRateCertificate,
    build_expected_stopping_certificate,
    solve_cost_aware_entropic_characteristic_game,
    solve_entropic_characteristic_game,
)


class CanonicalV19PacketError(ValueError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


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
        raise CanonicalV19PacketError("packet is not canonical-JSON serializable") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parse_fraction_matrix(raw: Sequence[Sequence[object]]) -> tuple[tuple[Fraction, ...], ...]:
    matrix = tuple(tuple(as_fraction(x) for x in row) for row in raw)
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise CanonicalV19PacketError("fraction matrix must be nonempty and rectangular")
    return matrix


def _derive_arithmetic_contract(base_raw: Mapping[str, Any]) -> dict[str, Any]:
    context = base_raw.get("context")
    minorization = base_raw.get("minorization")
    if not isinstance(context, Mapping) or not isinstance(minorization, Sequence):
        raise CanonicalV19PacketError("base packet lacks context/minorization arithmetic data")
    dimension = int(context.get("objective_dimension", 0))
    if dimension <= 0:
        raise CanonicalV19PacketError("base objective dimension must be positive")
    max_rho = Fraction(0, 1)
    max_beta = Fraction(0, 1)
    max_steps = 0
    type_contracts: list[dict[str, Any]] = []
    for item in minorization:
        if not isinstance(item, Mapping):
            raise CanonicalV19PacketError("base minorization record is malformed")
        potential = item.get("potential_contract")
        blocks = item.get("confirm_final_blocks")
        if not isinstance(potential, Mapping) or not isinstance(blocks, Sequence):
            raise CanonicalV19PacketError("base potential/final-block arithmetic contract is missing")
        weights = tuple(as_fraction(x) for x in potential.get("reference_weights", ()))
        rho = as_fraction(potential.get("rho"))
        if len(weights) != dimension or any(x <= 0 for x in weights):
            raise CanonicalV19PacketError("base reference weights must be positive and dimensionally aligned")
        if sum(weights, Fraction(0, 1)) != 1:
            raise CanonicalV19PacketError("base reference weights must sum exactly to one")
        if rho < 0:
            raise CanonicalV19PacketError("base rho must be nonnegative")
        total_steps = 0
        local_max_beta = Fraction(0, 1)
        for block in blocks:
            if not isinstance(block, Mapping):
                raise CanonicalV19PacketError("confirm final block is malformed")
            beta = as_fraction(block.get("beta"))
            steps = int(block.get("steps", -1))
            if beta < 0 or steps < 0:
                raise CanonicalV19PacketError("confirm beta/steps are invalid")
            total_steps += steps
            local_max_beta = max(local_max_beta, beta)
        max_rho = max(max_rho, rho)
        max_beta = max(max_beta, local_max_beta)
        max_steps = max(max_steps, total_steps)
        type_contracts.append({
            "type_id": str(item.get("type_id")),
            "weights": [str(x) for x in weights],
            "rho": str(rho),
            "maximum_confirm_beta": str(local_max_beta),
            "confirm_steps": total_steps,
        })
    return {
        "dimension": dimension,
        "rho": max_rho,
        "beta": max_beta,
        "steps": max_steps,
        "maximum_energy_span": Fraction(1, 1) + max_rho,
        "maximum_uphill_exponent": max_beta * (Fraction(1, 1) + max_rho),
        "type_contracts": type_contracts,
    }


def _derive_worst_confirm_step_dobrushin(
    base_report: Mapping[str, Any],
) -> Fraction:
    """Derive a common per-step Dobrushin bound from the v18 minorization.

    A caller-supplied contraction factor would allow the finite-horizon TV
    certificate to be made arbitrarily small.  V19 therefore recomputes the
    worst ``1-epsilon`` value from the already verified v18 final-target
    minorization records.
    """

    minorization = base_report.get("minorization")
    if not isinstance(minorization, Mapping):
        raise CanonicalV19PacketError("base result lacks verified minorization data")
    types = minorization.get("types")
    if not isinstance(types, Sequence) or isinstance(types, (str, bytes)):
        raise CanonicalV19PacketError("base minorization type list is malformed")
    worst = Fraction(0, 1)
    saw_step = False
    for type_record in types:
        if not isinstance(type_record, Mapping):
            raise CanonicalV19PacketError("base minorization type record is malformed")
        blocks = type_record.get("confirm_final_blocks")
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise CanonicalV19PacketError("base confirm final blocks are malformed")
        for block in blocks:
            if not isinstance(block, Mapping):
                raise CanonicalV19PacketError("base confirm final block is malformed")
            epsilon = as_fraction(block.get("epsilon_lower"))
            if epsilon < 0 or epsilon > 1:
                raise CanonicalV19PacketError("base epsilon lower escaped [0,1]")
            worst = max(worst, Fraction(1, 1) - epsilon)
            saw_step = True
    if not saw_step:
        return Fraction(1, 1)
    return worst


@dataclass(frozen=True)
class CanonicalV19Result:
    packet_sha256: str
    base_v18_packet_sha256: str
    base_v18_pass: bool
    arithmetic_pass: bool
    bridge_pass: bool | None
    blocked_occupancy_pass: bool | None
    reference_pass: bool | None
    track_stop_pass: bool | None
    expected_stopping_pass: bool | None
    deadline_stopping_pass: bool | None
    overall_v19_extension_pass: bool
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pareto_smc_v19_canonical_result_v1",
            "packet_sha256": self.packet_sha256,
            "base_v18_packet_sha256": self.base_v18_packet_sha256,
            "base_v18_pass": self.base_v18_pass,
            "arithmetic_pass": self.arithmetic_pass,
            "bridge_pass": self.bridge_pass,
            "blocked_occupancy_pass": self.blocked_occupancy_pass,
            "reference_pass": self.reference_pass,
            "track_stop_pass": self.track_stop_pass,
            "expected_stopping_pass": self.expected_stopping_pass,
            "deadline_stopping_pass": self.deadline_stopping_pass,
            "overall_v19_extension_pass": self.overall_v19_extension_pass,
            "report": dict(self.report),
        }


def build_canonical_v19_packet(raw: Mapping[str, Any]) -> CanonicalV19Result:
    if raw.get("schema") not in {"pareto_smc_v19_canonical_packet_v1", "pareto_smc_v19_canonical_packet_v1_1"}:
        raise CanonicalV19PacketError("unsupported v19 canonical schema")
    base_raw = raw.get("base_v18_packet")
    if not isinstance(base_raw, Mapping):
        raise CanonicalV19PacketError("base_v18_packet must be a raw mapping")
    base = build_canonical_v18_packet(base_raw)
    conditional_dependencies: list[str] = []

    derived_arithmetic = _derive_arithmetic_contract(base_raw)
    derived_step_dobrushin = _derive_worst_confirm_step_dobrushin(base.report)
    arithmetic_raw = raw.get("implementation_arithmetic")
    if not isinstance(arithmetic_raw, Mapping):
        raise CanonicalV19PacketError("implementation_arithmetic is required")
    arithmetic_mode = str(arithmetic_raw.get("mode"))
    arithmetic_report: dict[str, Any]
    if arithmetic_mode == "exact_rational_lazy_random_bits":
        exponent = derived_arithmetic["maximum_uphill_exponent"]
        supplied_exponent = arithmetic_raw.get("max_uphill_exponent")
        if supplied_exponent is not None and as_fraction(supplied_exponent) != exponent:
            raise CanonicalV19PacketError(
                "max_uphill_exponent must equal the value derived from the base packet"
            )
        order = int(arithmetic_raw.get("taylor_order", 32))
        lower, upper = exp_neg_rational_interval(exponent, order=order)
        width_cap = as_fraction(arithmetic_raw.get("maximum_probability_interval_width", 1))
        integration_provenance = str(
            arithmetic_raw.get(
                "exact_kernel_integration_provenance",
                "diagnostic_building_block_only",
            )
        )
        integration_proof_sha = str(
            arithmetic_raw.get("exact_kernel_integration_proof_sha256", "")
        )
        if integration_provenance not in {
            "diagnostic_building_block_only",
            "independently_verified_exact_final_kernel_integration",
            "theorem_parameter_conditional_exact_final_kernel",
        }:
            raise CanonicalV19PacketError(
                "unsupported exact-kernel integration provenance"
            )
        formal_integration = integration_provenance != "diagnostic_building_block_only"
        if formal_integration and _HEX64.fullmatch(integration_proof_sha) is None:
            raise CanonicalV19PacketError(
                "formal exact-kernel integration requires a proof SHA-256"
            )
        if formal_integration:
            conditional_dependencies.append(
                "exact final-kernel integration proof artifact"
            )
        building_block_ready = upper - lower <= width_cap
        arithmetic_pass = formal_integration and building_block_ready
        arithmetic_report = {
            "mode": arithmetic_mode,
            "max_uphill_exponent": str(exponent),
            "probability_lower": str(lower),
            "probability_upper": str(upper),
            "interval_width": str(upper - lower),
            "maximum_probability_interval_width": str(width_cap),
            "building_block_ready": building_block_ready,
            "exact_kernel_integration_provenance": integration_provenance,
            "exact_kernel_integration_proof_sha256": (
                integration_proof_sha or None
            ),
            "derived_from_base_v18": True,
            "derived_type_contracts": derived_arithmetic["type_contracts"],
            "ideal_randomness_assumption": "independent_infinite_uniform_bits",
            "whole_smc_exactness_claimed": False,
            "final_kernel_integration_claimed": formal_integration,
            "pass_gate": arithmetic_pass,
        }
    elif arithmetic_mode == "automatic_binary64_tv":
        for key in ("dimension", "rho", "beta", "steps"):
            if key in arithmetic_raw:
                expected = derived_arithmetic[key]
                observed = int(arithmetic_raw[key]) if key in {"dimension", "steps"} else as_fraction(arithmetic_raw[key])
                if observed != expected:
                    raise CanonicalV19PacketError(
                        f"{key} must equal the value derived from the base packet"
                    )
        transcendental_error = as_fraction(
            arithmetic_raw.get("transcendental_comparison_error_upper", 0)
        )
        provenance = str(arithmetic_raw.get("transcendental_error_provenance", ""))
        proof_sha = str(arithmetic_raw.get("transcendental_error_proof_sha256", ""))
        if derived_arithmetic["beta"] > 0:
            if provenance not in {
                "independently_verified_acceptance_comparison_error",
                "theorem_parameter_conditional_runtime_bound",
            } or _HEX64.fullmatch(proof_sha) is None:
                raise CanonicalV19PacketError(
                    "positive-beta binary64 arithmetic requires a provenance-tagged acceptance-comparison error proof"
                )
            conditional_dependencies.append(
                "binary64 acceptance-comparison error proof artifact"
            )
        supplied_proposal_tv = as_fraction(
            arithmetic_raw.get("proposal_row_tv_upper", 0)
        )
        if supplied_proposal_tv != 0:
            raise CanonicalV19PacketError(
                "the canonical automatic path derives zero proposal-row TV from the fixed v18 proposal contract; use a separate externally verified kernel certificate for a changed proposal"
            )
        if "ideal_step_dobrushin_upper" in arithmetic_raw:
            if as_fraction(arithmetic_raw["ideal_step_dobrushin_upper"]) != derived_step_dobrushin:
                raise CanonicalV19PacketError(
                    "ideal_step_dobrushin_upper must equal the value derived from the base minorization"
                )
        cert = build_automatic_kernel_tv_certificate(
            dimension=derived_arithmetic["dimension"],
            rho=derived_arithmetic["rho"],
            beta=derived_arithmetic["beta"],
            steps=derived_arithmetic["steps"],
            proposal_row_tv_upper=0,
            transcendental_comparison_error_upper=transcendental_error,
            ideal_step_dobrushin_upper=derived_step_dobrushin,
        )
        cap = as_fraction(arithmetic_raw.get("maximum_horizon_tv_upper", 1))
        arithmetic_pass = cert.horizon_tv_upper <= cap
        arithmetic_report = cert.to_dict() | {
            "mode": arithmetic_mode,
            "derived_from_base_v18": True,
            "derived_type_contracts": derived_arithmetic["type_contracts"],
            "ideal_step_dobrushin_upper_provenance": (
                "derived_from_v18_final_target_minorization"
            ),
            "transcendental_error_provenance": provenance or (
                "not_required_at_beta_zero" if derived_arithmetic["beta"] == 0 else None
            ),
            "transcendental_error_proof_sha256": proof_sha or None,
            "maximum_horizon_tv_upper": str(cap),
            "pass_gate": arithmetic_pass,
        }
    else:
        raise CanonicalV19PacketError("unsupported implementation arithmetic mode")

    bridge_report: dict[str, Any] | None = None
    bridge_pass: bool | None = None
    bridge_raw = raw.get("multilevel_bridge")
    if bridge_raw is not None:
        if not isinstance(bridge_raw, Mapping):
            raise CanonicalV19PacketError("multilevel_bridge must be a mapping")
        planner_mode = str(bridge_raw.get("planner_mode", "fixed_trials"))
        if planner_mode == "fixed_trials":
            bridge = build_multilevel_bridge_certificate(
                bridge_raw["conditional_success_lower"],
                bridge_raw["trials"],
                target_failure_budget=bridge_raw["target_failure_budget"],
                evaluation_cost_per_trial=bridge_raw.get("evaluation_cost_per_trial"),
                level_ids=bridge_raw.get("level_ids"),
                set_sha256_chain=bridge_raw.get("set_sha256_chain"),
                transition_proof_sha256=bridge_raw.get("transition_proof_sha256"),
                transition_proof_provenance=bridge_raw.get("transition_proof_provenance"),
                nesting_proof_provenance=bridge_raw.get("nesting_proof_provenance"),
                trial_contracts=bridge_raw.get("trial_contracts"),
            )
            bridge_report = bridge.to_dict() | {"planner_mode": planner_mode}
            bridge_pass = bridge.pass_gate
        elif planner_mode == "exact_minimum_cost":
            plan = exact_minimum_cost_bridge_plan(
                bridge_raw["conditional_success_lower"],
                target_failure_budget=bridge_raw["target_failure_budget"],
                evaluation_cost_per_trial=bridge_raw.get("evaluation_cost_per_trial"),
                level_ids=bridge_raw.get("level_ids"),
                set_sha256_chain=bridge_raw.get("set_sha256_chain"),
                transition_proof_sha256=bridge_raw.get("transition_proof_sha256"),
                transition_proof_provenance=bridge_raw.get("transition_proof_provenance"),
                nesting_proof_provenance=bridge_raw.get("nesting_proof_provenance"),
                trial_contracts=bridge_raw.get("trial_contracts"),
                max_nodes=int(bridge_raw.get("planner_max_nodes", 2_000_000)),
            )
            bridge_report = plan.to_dict() | {"planner_mode": planner_mode}
            bridge_pass = plan.optimal and plan.certificate.pass_gate
        else:
            raise CanonicalV19PacketError("unsupported multilevel bridge planner mode")
        conditional_dependencies.append(
            "multilevel bridge transition-probability proof artifacts"
        )

    confirm_matrix = _parse_fraction_matrix(
        base.report["confirm_endpoint_lower_matrix"]  # type: ignore[index]
    )
    occupancy_report: dict[str, Any] | None = None
    occupancy_pass: bool | None = None
    occupancy_raw = raw.get("blocked_occupancy")
    if occupancy_raw is not None:
        if not isinstance(occupancy_raw, Mapping):
            raise CanonicalV19PacketError("blocked_occupancy must be a mapping")
        base_confirm = base_raw.get("confirm")
        if not isinstance(base_confirm, Mapping):
            raise CanonicalV19PacketError("base confirm section is malformed")
        problem = BlockedOccupancyProblem(
            endpoint_lower=confirm_matrix,
            costs=tuple(as_fraction(x) for x in base_confirm["costs"]),
            delta=as_fraction(base_confirm["delta"]),
            blocks=tuple(tuple(int(j) for j in block) for block in occupancy_raw["blocks"]),
            max_block_size=int(occupancy_raw.get("max_block_size", 12)),
            risk_cache_size=int(occupancy_raw.get("risk_cache_size", 100_000)),
            combination_mode=str(occupancy_raw.get("combination_mode", "hunter_pairwise")),
            max_pair_union_size=(
                None
                if occupancy_raw.get("max_pair_union_size") is None
                else int(occupancy_raw["max_pair_union_size"])
            ),
        )
        plan = exact_minimum_cost_blocked_allocation(
            problem,
            max_nodes=int(occupancy_raw.get("planner_max_nodes", 2_000_000)),
            max_greedy_steps=int(
                occupancy_raw.get("planner_max_greedy_steps", 1_000_000)
            ),
        )
        occupancy_report = plan.to_dict() | {
            "blocks": [list(block) for block in problem.blocks],
            "combination_mode": problem.combination_mode,
            "max_pair_union_size": problem.max_pair_union_size,
        }
        occupancy_pass = plan.optimal_for_blocked_surrogate and (
            plan.blocked_miss_upper <= problem.delta
        )

    reference_report: dict[str, Any] | None = None
    reference_pass: bool | None = None
    reference_raw = raw.get("geometric_reference")
    if reference_raw is not None:
        if not isinstance(reference_raw, Mapping):
            raise CanonicalV19PacketError("geometric_reference must be a mapping")
        records_raw = reference_raw.get("records")
        if not isinstance(records_raw, Sequence) or isinstance(records_raw, (str, bytes)):
            raise CanonicalV19PacketError("geometric reference records are missing")
        records: list[GridOracleRecord] = []
        for item in records_raw:
            if not isinstance(item, Mapping):
                raise CanonicalV19PacketError("geometric reference record is malformed")
            witness_raw = item.get("witness_objective")
            lower_raw = item.get("constrained_lower_bound")
            records.append(
                GridOracleRecord(
                    thresholds=tuple(as_fraction(x) for x in item["thresholds"]),
                    feasible=bool(item["feasible"]),
                    witness_objective=(
                        None
                        if witness_raw is None
                        else tuple(as_fraction(x) for x in witness_raw)
                    ),
                    constrained_lower_bound=(
                        None if lower_raw is None else as_fraction(lower_raw)
                    ),
                    approximation_factor=as_fraction(item["approximation_factor"]),
                    proof_sha256=str(item["proof_sha256"]),
                    provenance=str(item["provenance"]),
                    pivot_index=int(item.get("pivot_index", reference_raw.get("pivot_index", -1))),
                )
            )
        cert = build_geometric_reference_certificate(
            lower=reference_raw["lower"],
            upper=reference_raw["upper"],
            eta=reference_raw["eta"],
            approximation_factor=reference_raw["approximation_factor"],
            records=records,
            pivot_index=(
                None
                if reference_raw.get("pivot_index") is None
                else int(reference_raw["pivot_index"])
            ),
            max_grid_points=int(reference_raw.get("max_grid_points", 2_000_000)),
        )
        reference_report = cert.to_dict() | {
            "formal_claim_status": cert.proof_status,
            "unconditional_local_recomputation_pass": (
                cert.proof_status == "LOCALLY_RECOMPUTED_EXACT_ENUMERATION"
            ),
        }
        # Structural validity is enough to include the conditional theorem in the
        # canonical packet.  The report separately records whether the proof was
        # locally recomputed or depends on external constrained-oracle artifacts.
        reference_pass = True
        if cert.external_proof_record_count:
            conditional_dependencies.append(
                "external constrained-reference oracle proof artifacts"
            )

    track_report: dict[str, Any] | None = None
    track_pass: bool | None = None
    expected_pass: bool | None = None
    deadline_pass: bool | None = None
    track_raw = raw.get("regularized_track_and_stop")
    if track_raw is not None:
        if not isinstance(track_raw, Mapping):
            raise CanonicalV19PacketError("regularized_track_and_stop must be a mapping")
        proof_sha = str(track_raw.get("probability_model_proof_sha256", ""))
        provenance = str(track_raw.get("probability_model_provenance", ""))
        if provenance not in {
            "exact_finite_state_enumeration",
            "independently_verified_endpoint_law",
            "theorem_parameter",
        }:
            raise CanonicalV19PacketError("unsupported endpoint-law provenance")
        if provenance != "theorem_parameter" and _HEX64.fullmatch(proof_sha) is None:
            raise CanonicalV19PacketError("verified endpoint law requires a proof SHA-256")
        conditional_dependencies.append(
            "shared-categorical endpoint-law identity or theorem parameter"
        )
        costs_raw = track_raw.get("arm_costs")
        if costs_raw is None:
            cert = solve_entropic_characteristic_game(
                track_raw["probability_matrix"],
                regularization=float(track_raw["regularization"]),
                smoothing=float(track_raw.get("smoothing", 0.0)),
                iterations=int(track_raw.get("iterations", 30_000)),
                step_scale=float(track_raw.get("step_scale", 0.25)),
            )
            gap_value = cert.total_characteristic_gap_upper
            objective_kind = "pull_count_characteristic"
        else:
            cert = solve_cost_aware_entropic_characteristic_game(
                track_raw["probability_matrix"],
                costs_raw,
                regularization=float(track_raw["regularization"]),
                smoothing=float(track_raw.get("smoothing", 0.0)),
                iterations=int(track_raw.get("iterations", 30_000)),
                step_scale=float(track_raw.get("step_scale", 0.25)),
            )
            gap_value = cert.total_cost_characteristic_gap_upper
            objective_kind = "expected_evaluation_cost_characteristic"
        gap_cap = float(track_raw.get("maximum_characteristic_gap", float("inf")))
        optimization_provenance = str(
            track_raw.get("optimization_bound_provenance", "diagnostic_only")
        )
        optimization_proof_sha = str(
            track_raw.get("optimization_bound_proof_sha256", "")
        )
        if optimization_provenance not in {
            "diagnostic_only",
            "independently_verified_interval_characteristic_optimization",
            "theorem_parameter_conditional_optimization_bound",
        }:
            raise CanonicalV19PacketError("unsupported characteristic optimization provenance")
        formal_optimization_bound = optimization_provenance != "diagnostic_only"
        if formal_optimization_bound and _HEX64.fullmatch(optimization_proof_sha) is None:
            raise CanonicalV19PacketError(
                "a formal characteristic optimization gate requires a proof SHA-256"
            )
        if formal_optimization_bound:
            conditional_dependencies.append(
                "characteristic-game optimization proof artifact"
            )
        track_pass = formal_optimization_bound and gap_value <= gap_cap
        track_report = cert.to_dict() | {
            "objective_kind": objective_kind,
            "probability_model_provenance": provenance,
            "probability_model_proof_sha256": proof_sha or None,
            "optimization_bound_provenance": optimization_provenance,
            "optimization_bound_proof_sha256": optimization_proof_sha or None,
            "maximum_characteristic_gap": gap_cap,
            "numerical_solver_status": (
                "FLOATING OPTIMIZATION WITNESS; THE FORMAL GATE IS OPEN ONLY WHEN "
                "A SEPARATE INTERVAL/EXACT OR THEOREM-PARAMETER PROOF ARTIFACT IS BOUND"
            ),
            "pass_gate": track_pass,
        }
        rate_raw = track_raw.get("expected_stopping_rate_certificate")
        if rate_raw is not None:
            if not isinstance(rate_raw, Mapping):
                raise CanonicalV19PacketError("expected stopping rate certificate is malformed")
            rate = InformationRateCertificate(
                gamma_lower=as_fraction(rate_raw["gamma_lower"]),
                deficit_scale=as_fraction(rate_raw["deficit_scale"]),
                deficit_power=int(rate_raw["deficit_power"]),
                threshold_log_inv_delta_upper=as_fraction(
                    rate_raw["threshold_log_inv_delta_upper"]
                ),
                threshold_log2_time_coefficient=as_fraction(
                    rate_raw["threshold_log2_time_coefficient"]
                ),
                threshold_constant=as_fraction(rate_raw["threshold_constant"]),
                tail_prefactor=as_fraction(rate_raw["tail_prefactor"]),
                tail_ratio=as_fraction(rate_raw["tail_ratio"]),
                proof_sha256=str(rate_raw["proof_sha256"]),
            )
            expected = build_expected_stopping_certificate(
                rate,
                max_time=int(rate_raw.get("max_time", 100_000_000)),
            )
            expected_cap = as_fraction(
                rate_raw.get("maximum_expected_stopping_time", expected.expected_stopping_time_upper)
            )
            expected_pass = expected.expected_stopping_time_upper <= expected_cap
            conditional_dependencies.append(
                "quantitative information-rate tail proof artifact"
            )
            track_report["expected_stopping"] = expected.to_dict() | {
                "maximum_expected_stopping_time": str(expected_cap),
                "pass_gate": expected_pass,
            }
        deadline_raw = track_raw.get("deadline_fallback")
        if deadline_raw is not None:
            if not isinstance(deadline_raw, Mapping):
                raise CanonicalV19PacketError("deadline_fallback must be a mapping")
            gap_provenance = str(deadline_raw.get("gap_lower_provenance", ""))
            gap_proof_sha = str(deadline_raw.get("gap_lower_proof_sha256", ""))
            tracking_provenance = str(deadline_raw.get("tracking_share_provenance", ""))
            tracking_proof_sha = str(deadline_raw.get("tracking_share_proof_sha256", ""))
            allowed_deadline_provenance = {
                "exact_finite_state_enumeration",
                "independently_verified_endpoint_gap_bound",
                "theorem_parameter_conditional_bound",
            }
            allowed_tracking_provenance = {
                "deterministic_tracking_lemma",
                "independently_verified_tracking_bound",
                "theorem_parameter_conditional_bound",
            }
            if gap_provenance not in allowed_deadline_provenance or _HEX64.fullmatch(gap_proof_sha) is None:
                raise CanonicalV19PacketError("deadline gap lower bounds need a provenance-tagged proof artifact")
            if tracking_provenance not in allowed_tracking_provenance or _HEX64.fullmatch(tracking_proof_sha) is None:
                raise CanonicalV19PacketError("deadline tracking-share bounds need a provenance-tagged proof artifact")
            deadline = build_track_stop_deadline_certificate(
                best_types=deadline_raw["best_types"],
                gap_lower=deadline_raw["gap_lower"],
                allocation_share_lower=deadline_raw["allocation_share_lower"],
                tracking_deficit=deadline_raw["tracking_deficit"],
                alpha_fallback=deadline_raw["alpha_fallback"],
                delta_glr=deadline_raw.get("delta_glr", 0),
                arm_costs=deadline_raw.get("arm_costs"),
                max_deadline=int(deadline_raw.get("max_deadline", 10_000_000)),
            )
            total_error_cap = as_fraction(
                deadline_raw.get("maximum_total_error_upper", deadline.total_error_upper)
            )
            cost_cap = as_fraction(
                deadline_raw.get("maximum_evaluation_cost_upper", deadline.evaluation_cost_upper)
            )
            deadline_pass = (
                deadline.pass_gate
                and deadline.total_error_upper <= total_error_cap
                and deadline.evaluation_cost_upper <= cost_cap
            )
            conditional_dependencies.extend((
                "endpoint winner-gap lower-bound proof artifact",
                "tracking allocation-share lower-bound proof artifact",
            ))
            track_report["deadline_fallback"] = deadline.to_dict() | {
                "gap_lower_provenance": gap_provenance,
                "gap_lower_proof_sha256": gap_proof_sha,
                "tracking_share_provenance": tracking_provenance,
                "tracking_share_proof_sha256": tracking_proof_sha,
                "maximum_total_error_upper": str(total_error_cap),
                "maximum_evaluation_cost_upper": str(cost_cap),
                "pass_gate": deadline_pass,
            }

    enabled_passes = [arithmetic_pass]
    for value in (bridge_pass, occupancy_pass, reference_pass, track_pass, expected_pass, deadline_pass):
        if value is not None:
            enabled_passes.append(value)
    overall = base.overall_theory_packet_pass and all(enabled_passes)
    report = {
        "base_v18": base.to_dict(),
        "implementation_arithmetic": arithmetic_report,
        "multilevel_bridge": bridge_report,
        "blocked_occupancy": occupancy_report,
        "geometric_reference": reference_report,
        "regularized_track_and_stop": track_report,
        "overall_claim_status": (
            "PASS_CONDITIONAL_ON_BOUND_EXTERNAL_PROOF_ARTIFACTS"
            if overall and conditional_dependencies
            else "PASS_LOCALLY_RECOMPUTED"
            if overall
            else "FAIL"
        ),
        "conditional_external_dependencies": sorted(
            set(conditional_dependencies)
        ),
        "claim_boundary": {
            "small_mass_barrier_removed_unconditionally": False,
            "many_objective_scalability_claimed": False,
            "finite_budget_instance_optimality_claimed": False,
            "floating_characteristic_solver_machine_certified": False,
            "external_oracle_hash_is_itself_a_mathematical_proof": False,
            "external_randomness_unconditionally_proved": False,
            "shared_beacon_v1_provides_rowwise_independence": False,
            "rowwise_beacon_v2_requires_product_uniform_external_source": True,
            "machine_formalization": "NOT_PERFORMED_UNLESS_A_SEPARATE_COMPILED_ARTIFACT_IS_ATTACHED",
        },
    }
    return CanonicalV19Result(
        packet_sha256=_sha256(raw),
        base_v18_packet_sha256=base.packet_sha256,
        base_v18_pass=base.overall_theory_packet_pass,
        arithmetic_pass=arithmetic_pass,
        bridge_pass=bridge_pass,
        blocked_occupancy_pass=occupancy_pass,
        reference_pass=reference_pass,
        track_stop_pass=track_pass,
        expected_stopping_pass=expected_pass,
        deadline_stopping_pass=deadline_pass,
        overall_v19_extension_pass=overall,
        report=report,
    )


__all__ = [
    "CanonicalV19PacketError",
    "CanonicalV19Result",
    "build_canonical_v19_packet",
]
