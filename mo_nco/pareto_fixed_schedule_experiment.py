from __future__ import annotations

"""End-to-end runner for a predeclared fixed-schedule pilot-confirm pair."""

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from .instance import MultiObjectiveTSPInstance, instance_sha256
from .pareto_fixed_reference_spec import (
    FixedReferenceCertificateSpecification,
)
from .pareto_fixed_schedule_certificate import (
    build_regeneration_pilot_plan_commitment_from_spec,
    certify_fixed_schedule_reference_metrics_from_spec,
)
from .pareto_execution_contract import (
    DOMAIN_SEPARATED_SEED_SCHEMA_V1,
    PARETO_SMC_V13_ALGORITHM_ROLE,
    PARETO_SMC_V13_ALGORITHM_VERSION,
    DomainSeparatedSeed,
    derive_domain_separated_seed,
)
from .pareto_fk_certificate import make_contraction_aware_fk_plan
from .pareto_smc import AnnealedParetoSMCOptimizer
from .pareto_v18_minorization import (
    Binary64AugmentedTchebycheffPotentialSpec,
    IdealFinalKernelContract,
    IndependenceMHMinorizationSpec,
)
from .pareto_smc_spec import (
    EXACT_INCREMENTAL_TWO_OPT_CONTRACT,
    ParetoSMCSpecification,
    original_unit_cell_widths,
)
from .sampler import OptimizationResult


@dataclass(frozen=True)
class FixedSchedulePilotConfirmResult:
    pilot: OptimizationResult
    confirm: OptimizationResult
    certificate: dict[str, object]


@dataclass(frozen=True)
class FixedScheduleExecutionPlan:
    """Validated immutable inputs shared by pilot and confirm streams."""

    pareto_smc_specification: ParetoSMCSpecification
    certificate_specification: FixedReferenceCertificateSpecification
    particles_per_reference: int
    run_seed: int
    anytime_checkpoint_period: int | None
    certificate_mode: str
    pilot_seed: int
    confirm_seed: int
    pilot_seed_contract: DomainSeparatedSeed | None
    confirm_seed_contract: DomainSeparatedSeed | None
    reference_count: int
    total_particles_per_stream: int
    evaluations_per_stream: int
    optimizer_arguments: Mapping[str, object]
    v18_source_derived_minorization: tuple[Mapping[str, object], ...]


def _bind_external_specification(
    result: OptimizationResult,
    specification: ParetoSMCSpecification,
) -> None:
    payload = {
        "specification_sha256": specification.sha256,
        "target_context_hash": result.metadata.get("context_hash"),
        "reporting_context_hash": result.metadata.get(
            "reporting_context_hash"
        ),
        "run_contract_hash": result.metadata.get("run_contract_hash"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    result.metadata.update(
        {
            "external_specification_schema": (
                "annealed_pareto_smc_spec_v1"
            ),
            "external_specification_path": str(specification.path),
            "external_specification_sha256": specification.sha256,
            "specification_run_binding_sha256": hashlib.sha256(
                encoded
            ).hexdigest(),
        }
    )


def prepare_fixed_schedule_execution(
    instance: MultiObjectiveTSPInstance,
    *,
    pareto_smc_specification: ParetoSMCSpecification,
    certificate_specification: FixedReferenceCertificateSpecification,
    particles_per_reference: int,
    run_seed: int = 0,
    anytime_checkpoint_period: int | None = None,
    certificate_mode: str = "published",
    v13_case_identity: str | None = None,
) -> FixedScheduleExecutionPlan:
    """Validate and freeze the common pilot-confirm execution contract."""

    smc = pareto_smc_specification
    certificate_spec = certificate_specification
    allowed_certificate_modes = {
        "published",
        "regeneration",
        "published_or_regeneration",
    }
    if certificate_mode not in allowed_certificate_modes:
        raise ValueError(
            "certificate_mode must be one of: "
            + ", ".join(sorted(allowed_certificate_modes))
        )
    if smc.mutation_steps_by_stage is None:
        raise ValueError(
            "The Pareto-SMC specification must freeze mutation.steps_per_stage."
        )
    if smc.global_refresh_probability <= 0.0:
        raise ValueError(
            "The fixed-schedule concentration theorem requires positive "
            "global refresh."
        )
    current_instance_hash = instance_sha256(instance)
    if current_instance_hash != certificate_spec.instance_sha256:
        raise ValueError(
            "The instance hash does not match the certificate specification."
        )
    if smc.sha256 != certificate_spec.pareto_smc_specification_sha256:
        raise ValueError(
            "The Pareto-SMC specification hash does not match the certificate "
            "specification."
        )
    reference_count = len(smc.reference_directions)
    if (
        isinstance(particles_per_reference, bool)
        or not isinstance(particles_per_reference, int)
        or particles_per_reference <= 0
    ):
        raise ValueError("particles_per_reference must be a positive integer.")
    total_particles = particles_per_reference * reference_count
    pilot_seed, confirm_seed = certificate_spec.stream_seeds(run_seed)
    pilot_seed_contract = None
    confirm_seed_contract = None
    if v13_case_identity is not None:
        pilot_seed_contract = derive_domain_separated_seed(
            case_identity=v13_case_identity,
            instance_sha256=current_instance_hash,
            paired_seed=run_seed,
            algorithm_role=PARETO_SMC_V13_ALGORITHM_ROLE,
            algorithm_version=PARETO_SMC_V13_ALGORITHM_VERSION,
            stream_role="pilot",
            schema=DOMAIN_SEPARATED_SEED_SCHEMA_V1,
        )
        confirm_seed_contract = derive_domain_separated_seed(
            case_identity=v13_case_identity,
            instance_sha256=current_instance_hash,
            paired_seed=run_seed,
            algorithm_role=PARETO_SMC_V13_ALGORITHM_ROLE,
            algorithm_version=PARETO_SMC_V13_ALGORITHM_VERSION,
            stream_role="confirm",
            schema=DOMAIN_SEPARATED_SEED_SCHEMA_V1,
        )
        if (
            pilot_seed != pilot_seed_contract.seed
            or confirm_seed != confirm_seed_contract.seed
        ):
            raise ValueError(
                "The frozen pilot-confirm seeds do not match the v13 "
                "case/algorithm/version/stream domain-separated derivation."
            )
    evaluations_per_stream = total_particles * (
        1 + sum(smc.mutation_steps_by_stage)
    )
    if (
        anytime_checkpoint_period is not None
        and (
            anytime_checkpoint_period <= 0
            or anytime_checkpoint_period > evaluations_per_stream
            or evaluations_per_stream % anytime_checkpoint_period != 0
        )
    ):
        raise ValueError(
            "A pilot-confirm anytime checkpoint period must be positive, "
            "no larger than one stream, and divide the per-stream budget "
            "so both streams lie on one global grid."
        )
    preflight_plan = make_contraction_aware_fk_plan(
        smc.beta_schedule,
        potential_upper_bound=1.0 + smc.chebyshev_rho,
        global_refresh_probability=smc.global_refresh_probability,
        mutation_steps_by_stage=smc.mutation_steps_by_stage,
        particle_count=particles_per_reference,
        observable_count=1,
        failure_budget=certificate_spec.pilot_failure_budget,
    )
    if (
        certificate_mode == "published"
        and not preflight_plan.published_concentration_gate
    ):
        raise ValueError(
            "The fixed schedule fails the published concentration regularity "
            "gate G_star*max_l b_l < 1/2."
        )
    if certificate_mode in {
        "regeneration",
        "published_or_regeneration",
    } and (
        smc.global_refresh_probability <= 0.0
        or smc.mutation_steps_by_stage[-1] <= 0
    ):
        raise ValueError(
            "The direct regeneration certificate requires positive global "
            "refresh and at least one final-stage mutation step."
        )

    # Only the terminal stage preserves the final target used by the
    # regeneration certificate.  Earlier stages have different beta targets
    # and therefore cannot be multiplied into the final-target residual.
    final_beta = smc.beta_schedule[-1]
    final_steps = smc.mutation_steps_by_stage[-1]
    v18_minorization_items = []
    for reference_index, weights in enumerate(smc.reference_directions):
        potential = Binary64AugmentedTchebycheffPotentialSpec(
            reference_weights=tuple(float(value) for value in weights),
            rho=float(smc.chebyshev_rho),
        )
        block = IndependenceMHMinorizationSpec(
            gamma=Fraction(str(smc.global_refresh_probability)),
            beta=Fraction(str(final_beta)),
            energy_span_upper=potential.energy_span_upper,
            steps=int(final_steps),
            subdivisions=max(
                256,
                int(16 * final_beta * float(potential.energy_span_upper)) + 2,
            ),
        )
        v18_minorization_items.append(
            {
                "type_id": f"r{reference_index}",
                "scope": "terminal_final_target_regeneration_only",
                "potential_contract": potential.to_dict(),
                "ideal_kernel_contract": IdealFinalKernelContract().to_dict(),
                "final_beta": str(Fraction(str(final_beta))),
                "final_block": block.to_dict(),
            }
        )
    v18_minorization = tuple(v18_minorization_items)


    common: dict[str, object] = {
        "instance": instance,
        "particles_per_reference": particles_per_reference,
        "evaluations": evaluations_per_stream,
        "beta_schedule": smc.beta_schedule,
        "reference_directions": smc.reference_directions,
        "num_reference_types": reference_count,
        "epsilon": original_unit_cell_widths(instance, smc),
        "ess_threshold": smc.ess_threshold_fraction,
        "resampling_policy": "always",
        "mutation_steps_by_stage": smc.mutation_steps_by_stage,
        "chebyshev_rho": smc.chebyshev_rho,
        "global_refresh_probability": smc.global_refresh_probability,
        "enable_exact_incremental_two_opt": (
            smc.mutation_objective_evaluation
            == EXACT_INCREMENTAL_TWO_OPT_CONTRACT
        ),
        "archive_max_size": smc.archive_max_size,
        "audit_trace_level": "summary",
        "anytime_checkpoint_period": anytime_checkpoint_period,
    }
    return FixedScheduleExecutionPlan(
        pareto_smc_specification=smc,
        certificate_specification=certificate_spec,
        particles_per_reference=particles_per_reference,
        run_seed=run_seed,
        anytime_checkpoint_period=anytime_checkpoint_period,
        certificate_mode=certificate_mode,
        pilot_seed=pilot_seed,
        confirm_seed=confirm_seed,
        pilot_seed_contract=pilot_seed_contract,
        confirm_seed_contract=confirm_seed_contract,
        reference_count=reference_count,
        total_particles_per_stream=total_particles,
        evaluations_per_stream=evaluations_per_stream,
        optimizer_arguments=common,
        v18_source_derived_minorization=v18_minorization,
    )


def run_fixed_schedule_stream(
    execution_plan: FixedScheduleExecutionPlan,
    *,
    stream: str,
) -> OptimizationResult:
    """Run exactly one validated stream.

    This seam lets a caller stop after the pilot, obtain an external
    authorization receipt, and only then launch confirm.
    """

    if stream == "pilot":
        seed = execution_plan.pilot_seed
        seed_contract = execution_plan.pilot_seed_contract
    elif stream == "confirm":
        seed = execution_plan.confirm_seed
        seed_contract = execution_plan.confirm_seed_contract
    else:
        raise ValueError("stream must be either 'pilot' or 'confirm'.")
    result = AnnealedParetoSMCOptimizer(
        seed=seed,
        domain_separated_seed=seed_contract,
        **dict(execution_plan.optimizer_arguments),
    ).run()
    _bind_external_specification(
        result,
        execution_plan.pareto_smc_specification,
    )
    result.metadata["v18_source_derived_minorization"] = [
        dict(item) for item in execution_plan.v18_source_derived_minorization
    ]
    result.metadata["v18_minorization_provenance"] = (
        "derived_from_actual_uniform_global_refresh_mixture_and_frozen_energy_span"
    )
    return result


def run_fixed_schedule_pilot_confirm(
    instance: MultiObjectiveTSPInstance,
    *,
    pareto_smc_specification: ParetoSMCSpecification,
    certificate_specification: FixedReferenceCertificateSpecification,
    particles_per_reference: int,
    run_seed: int = 0,
    anytime_checkpoint_period: int | None = None,
    certificate_mode: str = "published",
    v13_case_identity: str | None = None,
) -> FixedSchedulePilotConfirmResult:
    """Run both charged streams and return their direct metric certificate."""

    plan = prepare_fixed_schedule_execution(
        instance,
        pareto_smc_specification=pareto_smc_specification,
        certificate_specification=certificate_specification,
        particles_per_reference=particles_per_reference,
        run_seed=run_seed,
        anytime_checkpoint_period=anytime_checkpoint_period,
        certificate_mode=certificate_mode,
        v13_case_identity=v13_case_identity,
    )
    pilot = run_fixed_schedule_stream(plan, stream="pilot")
    pilot_plan_commitment = None
    if certificate_mode == "regeneration":
        pilot_plan_commitment = (
            build_regeneration_pilot_plan_commitment_from_spec(
                pilot,
                plan.certificate_specification,
                confirm_particles_per_reference=(
                    plan.particles_per_reference
                ),
                run_seed=run_seed,
            )
        )
    confirm = run_fixed_schedule_stream(plan, stream="confirm")
    certificate = certify_fixed_schedule_reference_metrics_from_spec(
        pilot,
        confirm,
        plan.certificate_specification,
        run_seed=run_seed,
        certificate_mode=certificate_mode,
        pilot_plan_commitment=pilot_plan_commitment,
        pilot_plan_commitment_preconfirm_order_attested_by_runner=(
            certificate_mode == "regeneration"
            and pilot_plan_commitment is not None
        ),
    )
    certificate["run_seed"] = run_seed
    certificate["requested_certificate_mode"] = certificate_mode
    certificate["particles_per_reference"] = (
        plan.particles_per_reference
    )
    certificate["total_particles_per_stream"] = (
        plan.total_particles_per_stream
    )
    certificate["evaluations_per_stream"] = (
        plan.evaluations_per_stream
    )
    certificate["v18_source_derived_minorization"] = [
        dict(item) for item in plan.v18_source_derived_minorization
    ]
    certificate["v13_domain_separated_seed_gate"] = (
        "PASS" if v13_case_identity is not None else "NOT_RUN"
    )
    certificate["v13_domain_separated_seed_schema"] = (
        DOMAIN_SEPARATED_SEED_SCHEMA_V1
        if v13_case_identity is not None
        else None
    )
    pilot_sweep_gate = pilot.metadata.get(
        "formal_full_type_sweep_checkpoint_gate"
    )
    confirm_sweep_gate = confirm.metadata.get(
        "formal_full_type_sweep_checkpoint_gate"
    )
    certificate["formal_full_type_sweep_checkpoint_gate"] = (
        "NOT_RUN"
        if anytime_checkpoint_period is None
        else (
            "PASS"
            if pilot_sweep_gate == "PASS" and confirm_sweep_gate == "PASS"
            else "FAIL"
        )
    )
    certificate["pilot_full_type_sweep_checkpoint_gate"] = pilot_sweep_gate
    certificate["confirm_full_type_sweep_checkpoint_gate"] = (
        confirm_sweep_gate
    )
    return FixedSchedulePilotConfirmResult(
        pilot=pilot,
        confirm=confirm,
        certificate=certificate,
    )
