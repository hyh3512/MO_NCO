from __future__ import annotations

import csv
import functools
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .archive import ArchiveEntry, ParetoArchive
from .baselines import MOEADOptimizer, MOTSPParetoLocalSearchOptimizer, NSGAIIOptimizer, RandomTwoOptOptimizer
from .evaluation import CountingTSPInstance, evaluation_count
from .ips_efficient import EfficientIPSOptimizer, TheoryAlignedIPSOptimizer
from .ips_certified import CertifiedSingleSiteIPSOptimizer
from .instance import MultiObjectiveTSPInstance, instance_sha256
from .metrics import additive_epsilon, empirical_reference_front, ideal_nadir, igd_plus, normalize_points, spacing
from .mature_baselines import (
    ExternalBaselineOptimizer,
    builtin_pymoo_baseline_configuration,
    external_baseline_configuration_sha256,
    external_baseline_provenance,
    load_external_baseline_from_env,
)
from .neural_potential import NeuralScalarPotential
from .potential import HypervolumeArchivePotential, ScalarArchivePotential
from .pareto_smc import AnnealedParetoSMCOptimizer
from .pareto_cell_certification import CellCertifiedParetoSampler
from .pareto_cell_spec import load_pareto_cell_certification_specification
from .pareto_smc_spec import (
    EXACT_INCREMENTAL_TWO_OPT_CONTRACT,
    load_pareto_smc_specification,
    original_unit_cell_widths,
)
from .pareto_ijoc_spec import load_ijoc_pareto_smc_specification
from .sampler import IPSMetropolisOptimizer, OptimizationResult
from .types import ObjectiveVector


def _neural_prior_path_for_backend(backend: str) -> str:
    """Resolve a backend-specific neural prior path with generic fallback."""

    key = "".join(ch if ch.isalnum() else "_" for ch in backend.upper())
    return os.environ.get(f"MO_NCO_NEURAL_PRIOR_PATH_{key}", os.environ.get("MO_NCO_NEURAL_PRIOR_PATH", ""))


@dataclass(frozen=True)
class RunRecord:
    algorithm: str
    seed: int
    population: int
    algorithm_configuration_sha256: str
    search_evaluations: int
    pilot_evaluations: int
    confirm_evaluations: int
    archive_size: int
    hypervolume_2d: float
    runtime_seconds: float
    python_peak_traced_memory_bytes: int
    output_objective_equivalence_gate: str
    output_objective_max_abs_error: float
    output_objective_equivalence_contract: str
    anytime_objective_equivalence_gate: str
    anytime_objective_equivalence_contract: str
    evaluation_evidence_gate: str
    evaluation_evidence_contract: str
    native_archive_completeness_gate: str
    native_archive_completeness_contract: str
    anytime_front_semantics: str
    anytime_checkpoint_gate: str
    anytime_checkpoint_contract: str
    anytime_checkpoint_period: int
    anytime_checkpoint_count: int
    anytime_auc_integration_contract: str
    anytime_time_auc_status: str
    max_diagnostic_archive_size: int
    diagnostic_archive_limit_gate: str
    diagnostic_archive_limit_contract: str
    acceptance_rate: float
    empirical_energy: float
    evaluations: int
    hypervolume_per_second: float
    hypervolume_per_evaluation: float
    anytime_hv_eval_auc: float
    anytime_hv_time_auc: float
    anytime_hv_auc_per_second: float
    igd_plus: float
    additive_epsilon: float
    spacing: float
    rejection_rate: float
    max_rejection_streak: int
    current_rejection_streak: int
    archive_csv: str
    publication_certificate_packet_gate: str = "NOT_APPLICABLE"


@functools.lru_cache(maxsize=1)
def _implementation_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def _runtime_environment_fingerprint() -> Dict[str, object]:
    distributions: Dict[str, object] = {}
    for name in ("numpy", "pymoo", "scipy", "numba", "torch"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = {"installed": False}
            continue
        record = distribution.read_text("RECORD")
        package_metadata = distribution.read_text("METADATA")
        direct_url = distribution.read_text("direct_url.json")
        distributions[name] = {
            "installed": True,
            "version": distribution.version,
            "record_sha256": (
                hashlib.sha256(record.encode("utf-8")).hexdigest()
                if record is not None
                else None
            ),
            "metadata_sha256": (
                hashlib.sha256(
                    package_metadata.encode("utf-8")
                ).hexdigest()
                if package_metadata is not None
                else None
            ),
            "direct_url_sha256": (
                hashlib.sha256(direct_url.encode("utf-8")).hexdigest()
                if direct_url is not None
                else None
            ),
        }
    executable = Path(sys.executable).resolve()
    return {
        "schema": "mo_nco_runtime_environment_fingerprint_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_sha256": hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "determinism_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "distributions": distributions,
    }


@dataclass(frozen=True)
class FrozenAlgorithmConfiguration:
    payload: Dict[str, object]
    sha256: str
    search_evaluations: int
    pilot_evaluations: int
    confirm_evaluations: int


def _canonical_payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _resolved_certified_configuration() -> Dict[str, float]:
    return {
        "temperature": float(
            os.environ.get("MO_NCO_CERTIFIED_TEMPERATURE", "0.05")
        ),
        "chebyshev_rho": float(
            os.environ.get("MO_NCO_CERTIFIED_CHEBYSHEV_RHO", "0.03")
        ),
        "uniformization_rate": float(
            os.environ.get(
                "MO_NCO_CERTIFIED_UNIFORMIZATION_RATE",
                "1.0",
            )
        ),
        "lazy_probability": float(
            os.environ.get(
                "MO_NCO_CERTIFIED_LAZY_PROBABILITY",
                "0.05",
            )
        ),
    }


def _pure_certificate_binding(
    *,
    instance_digest: str,
    smc_digest: str,
    run_seed: int,
) -> Dict[str, object]:
    from .pareto_fixed_reference_spec import (
        resolve_fixed_reference_certificate_specification,
    )

    direct_path = os.environ.get(
        "MO_NCO_PARETO_FIXED_REFERENCE_SPEC",
        "",
    )
    manifest_path = os.environ.get(
        "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST",
        "",
    )
    if bool(direct_path) == bool(manifest_path):
        raise ValueError(
            "Prelaunch configuration requires exactly one fixed-reference "
            "specification or case manifest."
        )
    manifest_digest = None
    if manifest_path:
        resolved = resolve_fixed_reference_certificate_specification(
            manifest_path,
            expected_instance_sha256=instance_digest,
            expected_pareto_smc_specification_sha256=smc_digest,
        )
        certificate_path = resolved.specification_path
        certificate_digest = resolved.specification_sha256
        manifest_digest = resolved.manifest_sha256
    else:
        certificate_path = Path(direct_path).expanduser().resolve()
        if not certificate_path.is_file():
            raise ValueError(
                "Fixed-reference certificate specification is missing."
            )
        certificate_digest = hashlib.sha256(
            certificate_path.read_bytes()
        ).hexdigest()
    try:
        certificate_payload = json.loads(
            certificate_path.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Fixed-reference certificate specification is not valid JSON."
        ) from error
    if (
        not isinstance(certificate_payload, dict)
        or certificate_payload.get("schema")
        != "pareto_smc_fixed_reference_certificate_spec_v2"
        or certificate_payload.get("instance_sha256") != instance_digest
        or certificate_payload.get("pareto_smc_specification_sha256")
        != smc_digest
    ):
        raise ValueError(
            "The prelaunch certificate header is not the expected "
            "witness-bound v2 instance/spec binding."
        )
    streams = certificate_payload.get("streams")
    if not isinstance(streams, dict):
        raise ValueError("The certificate streams block is malformed.")
    seed_pairs = streams.get("seed_pairs")
    if seed_pairs is None:
        if run_seed != 0:
            raise ValueError(
                "A certificate without seed_pairs can only bind run seed 0."
            )
        pilot_seed = streams.get("pilot_seed")
        confirm_seed = streams.get("confirm_seed")
    else:
        if not isinstance(seed_pairs, list):
            raise ValueError("certificate streams.seed_pairs is malformed.")
        matches = [
            pair
            for pair in seed_pairs
            if isinstance(pair, dict)
            and pair.get("run_seed") == run_seed
        ]
        if len(matches) != 1:
            raise ValueError(
                "No unique certificate stream pair is frozen for run seed "
                f"{run_seed}."
            )
        pilot_seed = matches[0].get("pilot_seed")
        confirm_seed = matches[0].get("confirm_seed")
    if (
        isinstance(pilot_seed, bool)
        or not isinstance(pilot_seed, int)
        or isinstance(confirm_seed, bool)
        or not isinstance(confirm_seed, int)
        or pilot_seed == confirm_seed
    ):
        raise ValueError("The frozen pilot-confirm stream seeds are invalid.")
    return {
        "certificate_specification_sha256": certificate_digest,
        "certificate_manifest_sha256": manifest_digest,
        "pilot_stream_seed": pilot_seed,
        "confirm_stream_seed": confirm_seed,
    }


def resolve_predeclared_algorithm_configuration(
    *,
    case_name: str,
    instance: MultiObjectiveTSPInstance,
    algorithm: str,
    seed: int,
    population: int,
    iterations: int,
    log_period: int,
    archive_update_period: int,
    output_archive_limit: Optional[int],
    certified_traces: bool,
    anytime_checkpoint_period: Optional[int] = None,
) -> FrozenAlgorithmConfiguration:
    """Resolve a canonical configuration without evaluating a tour."""

    if not case_name:
        raise ValueError("case_name must be nonempty.")
    if (
        anytime_checkpoint_period is not None
        and (
            isinstance(anytime_checkpoint_period, bool)
            or anytime_checkpoint_period <= 0
            or anytime_checkpoint_period > iterations
        )
    ):
        raise ValueError(
            "anytime_checkpoint_period must be a positive integer no "
            "larger than the evaluation budget."
        )
    if (
        anytime_checkpoint_period is not None
        and population > anytime_checkpoint_period
    ):
        raise ValueError(
            "The common anytime grid must begin no earlier than completion "
            "of every arm's initial population."
        )
    name = algorithm.lower()
    instance_digest = instance_sha256(instance)
    search_evaluations = iterations
    pilot_evaluations = 0
    confirm_evaluations = 0
    algorithm_specific: Dict[str, object] = {}
    external_command_digest = None

    if name in {
        "ijoc-pareto-smc",
        "typed-pareto-smc-ijoc",
    }:
        ijoc_specification_path = os.environ.get(
            "MO_NCO_IJOC_PARETO_SMC_SPEC",
            "",
        )
        if not ijoc_specification_path:
            raise ValueError(
                "Prelaunch configuration requires "
                "MO_NCO_IJOC_PARETO_SMC_SPEC."
            )
        ijoc_specification = load_ijoc_pareto_smc_specification(
            ijoc_specification_path,
            objective_dimension=instance.num_objectives,
            total_evaluations=iterations,
        )
        base_specification = ijoc_specification.base_smc_specification
        reference_count = len(base_specification.reference_directions)
        if population < reference_count or population % reference_count != 0:
            raise ValueError(
                "The IJOC population must be a positive multiple of the "
                "frozen reference-type count."
            )
        core_budget = iterations - ijoc_specification.adaptive_search_evaluations
        if (
            anytime_checkpoint_period is not None
            and iterations % anytime_checkpoint_period != 0
        ):
            raise ValueError(
                "The IJOC common anytime checkpoint period must divide the "
                "total evaluation budget exactly."
            )
        minimum_core_budget = population * len(base_specification.beta_schedule)
        if core_budget < minimum_core_budget:
            raise ValueError(
                "The IJOC adaptive tail leaves fewer evaluations than the "
                "minimum typed SMC core budget."
            )
        algorithm_specific = {
            "ijoc_specification_sha256": ijoc_specification.sha256,
            "base_pareto_smc_specification_sha256": base_specification.sha256,
            "reference_type_count": reference_count,
            "smc_core_evaluations": core_budget,
            "adaptive_search_evaluations": (
                ijoc_specification.adaptive_search_evaluations
            ),
            "adaptive_allocation_policy": ijoc_specification.allocation_policy,
            "adaptive_minimum_pulls_per_type": (
                ijoc_specification.minimum_pulls_per_type
            ),
            "exp3_exploration": ijoc_specification.exp3_exploration,
            "search_reward_weights": {
                "hypervolume": ijoc_specification.reward_weights.hypervolume,
                "new_cell": ijoc_specification.reward_weights.new_cell,
                "scalar_improvement": (
                    ijoc_specification.reward_weights.scalar_improvement
                ),
            },
            "competitive_archive_contract": (
                ijoc_specification.competitive_archive_contract
            ),
            "deployment_archive_max_size": (
                ijoc_specification.deployment_archive_max_size
            ),
            "certificate_scope": "pre_tail_typed_smc_snapshot_only",
            "anytime_checkpoint_emission": (
                "per_evaluation_passive_unbounded_search_archive_snapshot_v1"
            ),
        }
    elif name in {
        "pareto-smc-pilot-confirm-v11",
        "pareto-smc-pilot-confirm-v12",
    }:
        certificate_mode = (
            "published"
            if name == "pareto-smc-pilot-confirm-v11"
            else "regeneration"
        )
        specification_path = os.environ.get(
            "MO_NCO_PARETO_SMC_SPEC",
            "",
        )
        if not specification_path:
            raise ValueError(
                "Prelaunch configuration requires MO_NCO_PARETO_SMC_SPEC."
            )
        specification = load_pareto_smc_specification(
            specification_path,
            objective_dimension=instance.num_objectives,
        )
        if specification.archive_max_size is not None:
            raise ValueError(
                "The pilot-confirm competitive SMC specification must keep an "
                "unbounded native archive."
            )
        reference_count = len(specification.reference_directions)
        if (
            population < reference_count
            or population % reference_count != 0
        ):
            raise ValueError(
                "The population must be a positive multiple of the frozen "
                "reference-type count."
            )
        if specification.mutation_steps_by_stage is None:
            raise ValueError(
                "The pilot-confirm configuration must freeze mutation steps."
            )
        per_stream = population * (
            1 + sum(specification.mutation_steps_by_stage)
        )
        if iterations != 2 * per_stream:
            raise ValueError(
                "The requested budget does not exactly equal both frozen "
                "pilot-confirm streams."
            )
        if (
            anytime_checkpoint_period is not None
            and per_stream % anytime_checkpoint_period != 0
        ):
            raise ValueError(
                "The per-stream pilot-confirm budget must be divisible by "
                "the common anytime checkpoint period."
            )
        certificate_binding = _pure_certificate_binding(
            instance_digest=instance_digest,
            smc_digest=specification.sha256,
            run_seed=seed,
        )
        search_evaluations = 0
        pilot_evaluations = per_stream
        confirm_evaluations = per_stream
        algorithm_specific = {
            "pilot_confirm_protocol_version": (
                "v11_published"
                if certificate_mode == "published"
                else "v12_regeneration"
            ),
            "certificate_mode": certificate_mode,
            "pareto_smc_specification_sha256": specification.sha256,
            "reference_type_count": reference_count,
            "mutation_steps_by_stage": (
                specification.mutation_steps_by_stage
            ),
            "mutation_objective_evaluation": (
                specification.mutation_objective_evaluation
            ),
            "anytime_checkpoint_emission": (
                "per_evaluation_passive_archive_snapshot_v1"
            ),
            **certificate_binding,
        }
    elif name in {
        "ips-theory-certified",
        "ips-certified-mh",
        "ips-typed-mh",
    }:
        algorithm_specific = {
            **_resolved_certified_configuration(),
            "native_archive_max_size": None,
            "effective_diagnostic_log_period": (
                math.gcd(log_period, anytime_checkpoint_period)
                if anytime_checkpoint_period is not None
                else log_period
            ),
        }
    elif name in {"pymoo-nsga2", "pymoo-moead"}:
        if population <= 0 or iterations % population != 0:
            raise ValueError(
                "The pymoo exact-budget contract requires a positive "
                "population that divides the evaluation budget."
            )
        if name == "pymoo-moead" and population < 2:
            raise ValueError(
                "The pymoo MOEA/D prelaunch contract requires population "
                "size at least two."
            )
        external_config = builtin_pymoo_baseline_configuration(name)
        external_command_digest = (
            external_baseline_configuration_sha256(external_config)
        )
        algorithm_specific = {
            "native_archive_max_size": None,
            "pymoo_save_history": False,
            "all_evaluated_archive_instrumentation": (
                "elementwise_problem_pending_batch_callback_v1"
            ),
            "anytime_checkpoint_emission": (
                "elementwise_exact_evaluation_snapshot_v1"
            ),
            "formal_baseline_identity_contract": (
                "non_overridable_builtin_pymoo_module_v1"
            ),
            "external_baseline_provenance": (
                external_baseline_provenance(external_config)
            ),
        }
    elif name in {"motsp-pls", "pls", "tpls", "mogls"}:
        algorithm_specific = {
            "native_archive_max_size": None,
            "neighborhood_sample": max(8, min(48, population)),
            "scalar_guided": name
            in {"tpls", "mogls", "motsp-pls"},
            "anytime_checkpoint_emission": (
                "per_evaluation_passive_archive_snapshot_v1"
            ),
        }

    payload = {
        "schema": "mo_nco_predeclared_algorithm_configuration_v2",
        "case": case_name,
        "instance_sha256": instance_digest,
        "objective_dimension": instance.num_objectives,
        "algorithm": name,
        "seed": seed,
        "population": population,
        "iterations": iterations,
        "log_period": log_period,
        "archive_update_period": archive_update_period,
        "anytime_checkpoint_period": anytime_checkpoint_period,
        "output_archive_limit": output_archive_limit,
        "certified_traces_enabled": bool(certified_traces),
        "implementation_tree_sha256": _implementation_tree_sha256(),
        "runtime_environment_fingerprint": (
            _runtime_environment_fingerprint()
        ),
        "external_command_configuration_sha256": (
            external_command_digest
        ),
        "algorithm_specific": algorithm_specific,
        "budget_split": {
            "search_evaluations": search_evaluations,
            "pilot_evaluations": pilot_evaluations,
            "confirm_evaluations": confirm_evaluations,
        },
    }
    return FrozenAlgorithmConfiguration(
        payload=payload,
        sha256=_canonical_payload_sha256(payload),
        search_evaluations=search_evaluations,
        pilot_evaluations=pilot_evaluations,
        confirm_evaluations=confirm_evaluations,
    )


def _verify_result_objectives(
    instance: MultiObjectiveTSPInstance,
    result: OptimizationResult,
) -> None:
    """Re-evaluate every returned tour outside the measured search runtime."""

    if len(result.particles) != len(result.objectives):
        raise ValueError(
            "Optimization result has different particle and objective counts."
        )
    exact_required = bool(
        getattr(instance, "exact_two_opt_delta_in_binary64", False)
    )
    cache: dict[tuple[int, ...], ObjectiveVector] = {}
    maximum_error = 0.0

    def check(tour: tuple[int, ...], reported: ObjectiveVector) -> None:
        nonlocal maximum_error
        if len(reported) != instance.num_objectives:
            raise ValueError("Returned objective vector has the wrong dimension.")
        local = cache.get(tour)
        if local is None:
            local = instance.evaluate(tour)
            cache[tour] = local
        for observed, expected in zip(reported, local):
            error = abs(float(observed) - float(expected))
            maximum_error = max(maximum_error, error)
            equivalent = (
                observed == expected
                if exact_required
                else math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            if not equivalent:
                raise ValueError(
                    "Returned objective value does not match local full-tour "
                    f"evaluation for tour {tour}: reported={reported}, "
                    f"local={local}."
                )

    for tour, objectives in zip(result.particles, result.objectives):
        check(tour, objectives)
    for entry in result.archive.entries:
        check(entry.tour, entry.objectives)
    result.metadata.update(
        {
            "output_objective_equivalence_gate": "PASS",
            "output_objective_max_abs_error": maximum_error,
            "output_objective_equivalence_contract": (
                "local_full_tour_exact_on_integer_domain_else_"
                "rel1e-12_abs1e-12_v1"
            ),
            "output_objective_replayed_unique_tours": len(cache),
            "anytime_objective_equivalence_gate": "PASS",
            "anytime_objective_equivalence_contract": (
                result.metadata.get(
                    "external_anytime_objective_equivalence_contract",
                    "internal_diagnostic_front_from_local_evaluations_v1",
                )
            ),
        }
    )


def run_algorithm(
    algorithm: str,
    instance: MultiObjectiveTSPInstance,
    seed: int,
    population: int,
    iterations: int,
    log_period: int,
    archive_update_period: int,
    certified_trace_path: Optional[Path] = None,
    anytime_checkpoint_period: Optional[int] = None,
) -> OptimizationResult:
    name = algorithm.lower()
    if name in {
        "ijoc-pareto-smc",
        "typed-pareto-smc-ijoc",
    }:
        specification_path = os.environ.get(
            "MO_NCO_IJOC_PARETO_SMC_SPEC",
            "",
        )
        if not specification_path:
            raise ValueError(
                "Set MO_NCO_IJOC_PARETO_SMC_SPEC to a frozen IJOC "
                "configuration before using this alias."
            )
        specification = load_ijoc_pareto_smc_specification(
            specification_path,
            objective_dimension=instance.num_objectives,
            total_evaluations=iterations,
        )
        base = specification.base_smc_specification
        reference_count = len(base.reference_directions)
        if population < reference_count or population % reference_count != 0:
            raise ValueError(
                "The CLI population must be a positive multiple of the "
                f"{reference_count} frozen IJOC reference types."
            )
        if (
            anytime_checkpoint_period is not None
            and iterations % anytime_checkpoint_period != 0
        ):
            raise ValueError(
                "The IJOC common anytime checkpoint period must divide the "
                "total evaluation budget exactly."
            )
        if base.mutation_steps_by_stage is not None:
            raise ValueError(
                "The IJOC search alias uses the ESS SMC core and an adaptive "
                "post-certificate tail; mutation.steps_per_stage must be absent."
            )
        result = AnnealedParetoSMCOptimizer(
            instance=instance,
            particles_per_reference=population // reference_count,
            evaluations=iterations,
            seed=seed,
            beta_schedule=base.beta_schedule,
            reference_directions=base.reference_directions,
            num_reference_types=reference_count,
            epsilon=original_unit_cell_widths(instance, base),
            ess_threshold=base.ess_threshold_fraction,
            resampling_policy="ess",
            finite_particle_delta=0.05,
            chebyshev_rho=base.chebyshev_rho,
            global_refresh_probability=base.global_refresh_probability,
            adaptive_search_evaluations=(
                specification.adaptive_search_evaluations
            ),
            adaptive_allocation_policy=specification.allocation_policy,
            adaptive_minimum_pulls_per_type=(
                specification.minimum_pulls_per_type
            ),
            exp3_exploration=specification.exp3_exploration,
            search_reward_weights=specification.reward_weights,
            enable_exact_incremental_two_opt=(
                base.mutation_objective_evaluation
                == EXACT_INCREMENTAL_TWO_OPT_CONTRACT
            ),
            archive_tolerance=0.0,
            archive_max_size=specification.deployment_archive_max_size,
            audit_trace_level="summary",
            anytime_checkpoint_period=anytime_checkpoint_period,
        ).run()
        result.metadata.update(
            {
                "requested_algorithm_alias": name,
                "ijoc_algorithm_contract": (
                    "typed_smc_core_plus_stratified_exp3_archive_search_tail_v2"
                ),
                "ijoc_specification_path": str(specification.path),
                "ijoc_specification_sha256": specification.sha256,
                "base_pareto_smc_specification_path": str(base.path),
                "base_pareto_smc_specification_sha256": base.sha256,
                "native_archive_completeness_gate": "PASS",
                "native_archive_completeness_contract": (
                    "unbounded_exact_nondominated_all_evaluated_candidates_v2"
                ),
                "publication_certificate_packet_gate": "NOT_APPLICABLE",
                "theoretical_claim_scope": (
                    "smc_core_mechanics_and_exp3_observable_reward_regret_only"
                ),
            }
        )
        return result
    if name in {
        "pareto-smc-pilot-confirm-v11",
        "pareto-smc-pilot-confirm-v12",
    }:
        certificate_mode = (
            "published"
            if name == "pareto-smc-pilot-confirm-v11"
            else "regeneration"
        )
        smc_specification_path = os.environ.get(
            "MO_NCO_PARETO_SMC_SPEC",
            "",
        )
        certificate_specification_path = os.environ.get(
            "MO_NCO_PARETO_FIXED_REFERENCE_SPEC",
            "",
        )
        certificate_manifest_path = os.environ.get(
            "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST",
            "",
        )
        if (
            not smc_specification_path
            or bool(certificate_specification_path)
            == bool(certificate_manifest_path)
        ):
            raise ValueError(
                "Set MO_NCO_PARETO_SMC_SPEC and exactly one of "
                "MO_NCO_PARETO_FIXED_REFERENCE_SPEC (single-case smoke) or "
                "MO_NCO_PARETO_FIXED_REFERENCE_MANIFEST (multi-case study)."
            )
        from .pareto_fixed_reference_spec import (
            load_fixed_reference_certificate_specification,
            resolve_fixed_reference_certificate_specification,
        )
        from .pareto_fixed_schedule_experiment import (
            run_fixed_schedule_pilot_confirm,
        )

        smc_specification = load_pareto_smc_specification(
            smc_specification_path,
            objective_dimension=instance.num_objectives,
        )
        base_instance = getattr(instance, "base", instance)
        certificate_manifest_sha256 = None
        if certificate_manifest_path:
            resolved_certificate = (
                resolve_fixed_reference_certificate_specification(
                    certificate_manifest_path,
                    expected_instance_sha256=instance_sha256(
                        base_instance
                    ),
                    expected_pareto_smc_specification_sha256=(
                        smc_specification.sha256
                    ),
                )
            )
            certificate_specification_path = str(
                resolved_certificate.specification_path
            )
            certificate_manifest_sha256 = (
                resolved_certificate.manifest_sha256
            )
        certificate_specification = (
            load_fixed_reference_certificate_specification(
                certificate_specification_path,
                objective_dimension=instance.num_objectives,
                instance=base_instance,
            )
        )
        if (
            certificate_specification.schema
            != "pareto_smc_fixed_reference_certificate_spec_v2"
        ):
            raise ValueError(
                "The pilot-confirm competitive alias requires a witness-bound v2 "
                "certificate specification."
            )
        if smc_specification.archive_max_size is not None:
            raise ValueError(
                "The pilot-confirm competitive alias requires an "
                "unbounded native reporting archive before the common "
                "postprocessing cap."
            )
        reference_count = len(smc_specification.reference_directions)
        if population < reference_count or population % reference_count != 0:
            raise ValueError(
                "The CLI population must be a positive multiple of the "
                "predeclared reference-type count."
            )
        mutation_steps = smc_specification.mutation_steps_by_stage
        if mutation_steps is None:
            raise ValueError(
                "The pilot-confirm competitive alias requires a fixed per-stage "
                "mutation schedule."
            )
        per_stream_evaluations = population * (
            1 + sum(mutation_steps)
        )
        exact_total_evaluations = 2 * per_stream_evaluations
        if iterations != exact_total_evaluations:
            raise ValueError(
                "The CLI evaluation budget must exactly equal both charged "
                "pilot-confirm streams: "
                f"expected={exact_total_evaluations}, observed={iterations}."
            )
        pair = run_fixed_schedule_pilot_confirm(
            instance,  # type: ignore[arg-type]
            pareto_smc_specification=smc_specification,
            certificate_specification=certificate_specification,
            particles_per_reference=population // reference_count,
            run_seed=seed,
            anytime_checkpoint_period=anytime_checkpoint_period,
            certificate_mode=certificate_mode,
        )
        if pair.certificate.get("formal_packet_gate") != "PASS":
            raise ValueError(
                "The pilot-confirm run failed its witness, concentration, "
                "metric, or bounded-archive formal packet gate."
            )
        pilot_elapsed = (
            pair.pilot.diagnostics[-1].elapsed_seconds
            if pair.pilot.diagnostics
            else 0.0
        )
        combined_archive = ParetoArchive(max_size=None)
        combined_archive.update(pair.pilot.archive.entries)
        combined_archive.update(pair.confirm.archive.entries)
        cumulative_front = ParetoArchive(max_size=None)
        combined_diagnostics = []
        point_serial = 0
        for stream_index, diagnostics in enumerate(
            (pair.pilot.diagnostics, pair.confirm.diagnostics)
        ):
            for diagnostic in diagnostics:
                snapshot_entries = []
                for point in diagnostic.front:
                    snapshot_entries.append(
                        ArchiveEntry(
                            tour=(point_serial,),
                            objectives=tuple(
                                float(value) for value in point
                            ),
                        )
                    )
                    point_serial += 1
                cumulative_front.update(snapshot_entries)
                front = tuple(
                    entry.objectives
                    for entry in cumulative_front.entries
                )
                combined_diagnostics.append(
                    replace(
                        diagnostic,
                        iteration=(
                            diagnostic.iteration
                            + (
                                per_stream_evaluations
                                if stream_index == 1
                                else 0
                            )
                        ),
                        elapsed_seconds=(
                            diagnostic.elapsed_seconds
                            + (pilot_elapsed if stream_index == 1 else 0.0)
                        ),
                        archive_size=len(front),
                        hypervolume_2d=(
                            cumulative_front.hypervolume_2d()
                            if front and len(front[0]) == 2
                            else 0.0
                        ),
                        front=front,
                    )
                )
        if not combined_diagnostics:
            raise RuntimeError(
                "The pilot-confirm run did not emit anytime diagnostics."
            )
        if {
            entry.objectives for entry in combined_archive.entries
        } != set(combined_diagnostics[-1].front):
            raise RuntimeError(
                "The pilot-confirm cumulative diagnostic endpoint does not "
                "match the union of both unbounded stream archives."
            )
        result = OptimizationResult(
            particles=pair.confirm.particles,
            objectives=pair.confirm.objectives,
            archive=combined_archive,
            diagnostics=tuple(combined_diagnostics),
            metadata=dict(pair.confirm.metadata),
        )
        result.metadata.update(
            {
                "requested_algorithm_alias": name,
                "pilot_confirm_protocol_version": (
                    "v11_published"
                    if certificate_mode == "published"
                    else "v12_regeneration"
                ),
                "certificate_mode": certificate_mode,
                "pilot_plan_commitment_sha256": pair.certificate.get(
                    "pilot_plan_commitment_sha256"
                ),
                "pilot_plan_commitment_gate": pair.certificate.get(
                    "pilot_plan_commitment_gate"
                ),
                "pilot_plan_commitment_preconfirm_order_gate": (
                    pair.certificate.get(
                        "pilot_plan_commitment_preconfirm_order_gate"
                    )
                ),
                "pilot_plan_commitment_preconfirm_timing_independently_verified": (
                    pair.certificate.get(
                        "pilot_plan_commitment_preconfirm_timing_"
                        "independently_verified"
                    )
                ),
                "publication_certificate_packet_gate": (
                    pair.certificate.get(
                        "publication_certificate_packet_gate"
                    )
                ),
                "pilot_evaluations": per_stream_evaluations,
                "confirm_evaluations": per_stream_evaluations,
                "evaluations_used": exact_total_evaluations,
                "pilot_confirm_total_evaluations": (
                    exact_total_evaluations
                ),
                "pilot_confirm_budget_split_gate": "PASS",
                "pilot_confirm_run_seed": seed,
                "certificate_specification_path": str(
                    certificate_specification.path
                ),
                "certificate_specification_sha256": (
                    certificate_specification.sha256
                ),
                "certificate_manifest_sha256": (
                    certificate_manifest_sha256
                ),
                "external_specification_sha256": (
                    smc_specification.sha256
                ),
                "formal_packet_gate": "PASS",
                "fixed_cell_cover_archive": pair.certificate[
                    "cell_cover_archive_entries"
                ],
                "fixed_cell_cover_archive_size": pair.certificate[
                    "cell_cover_archive_size"
                ],
                "fixed_cell_cover_archive_max_size": (
                    certificate_specification.certified_archive_max_size
                ),
                "fixed_cell_cover_archive_gate": pair.certificate[
                    "certified_archive_gate"
                ],
                "reporting_archive_semantics": (
                    "pilot_confirm_union_competitive_archive_separate_from_"
                    "bounded_certificate_cell_cover"
                ),
                "certificate_support_semantics": (
                    "independent_confirm_terminal_support_only"
                ),
                "native_archive_completeness_gate": "PASS",
                "native_archive_completeness_contract": (
                    "unbounded_exact_nondominated_all_evaluated_candidates_v2"
                ),
                "pilot_confirm_certificate": pair.certificate,
            }
        )
        return result
    if name in {
        "pareto-cell-certified",
        "cell-certified-pareto-smc",
        "pareto-smc-cell-certificate",
    }:
        specification_path = os.environ.get("MO_NCO_PARETO_CELL_SPEC", "")
        if not specification_path:
            raise ValueError(
                "Set MO_NCO_PARETO_CELL_SPEC to a source-bound cell "
                "certification manifest before using this alias."
            )
        specification = load_pareto_cell_certification_specification(
            specification_path,
            objective_dimension=instance.num_objectives,
            num_cities=instance.num_cities,
            expected_instance_sha256=instance_sha256(instance),
        )
        probe_budget = sum(
            contract.particle_count * (1 + contract.mutation_steps)
            for contract in specification.cell_types
        )
        if probe_budget > iterations:
            raise ValueError(
                "The source-bound cell certification plan exceeds the CLI "
                "evaluation budget."
            )
        result = CellCertifiedParetoSampler(
            instance,
            cell_types=specification.cell_types,
            objective_lower_bounds=specification.target_safety_lower_bounds,
            objective_upper_bounds=specification.target_safety_upper_bounds,
            metric_lower_bounds=specification.metric_lower_bounds,
            metric_upper_bounds=specification.metric_upper_bounds,
            cell_widths=specification.cell_widths,
            beta=specification.beta,
            chebyshev_rho=specification.chebyshev_rho,
            seed=seed,
            confidence_delta=specification.confidence_delta,
            cell_completeness_proof_sha256=(
                specification.cell_completeness_proof_sha256
            ),
            objective_box_proof_sha256=(
                specification.target_safety_box_proof_sha256
            ),
            metric_box_proof_sha256=(
                specification.metric_box_proof_sha256
            ),
            metric_igd_p=specification.metric_igd_p,
            max_igd_bound=specification.max_igd_bound,
            hv_reference=specification.hv_reference,
            max_hv_deficit_bound=(
                specification.max_hv_deficit_bound
            ),
            archive_max_size=specification.archive_max_size,
            anytime_checkpoint_period=anytime_checkpoint_period,
        ).run()
        result.metadata.update(
            {
                "requested_algorithm_alias": name,
                "external_specification_schema": "pareto_cell_source_bound_spec_v4",
                "external_specification_path": str(specification.path),
                "external_specification_sha256": specification.sha256,
                "metric_box_proof_sha256": specification.metric_box_proof_sha256,
                "cli_evaluation_budget": iterations,
                "planned_probe_evaluations": probe_budget,
                "unused_cli_evaluations": iterations - probe_budget,
                "exact_cli_budget_match": probe_budget == iterations,
                "formal_equal_budget_comparison_gate": (
                    "PASS" if probe_budget == iterations else "FAIL"
                ),
            }
        )
        return result
    if name in {
        "annealed-pareto-smc",
        "pareto-smc-feynman-kac",
        "pareto-smc-certified-mechanics",
        "annealed-pareto-smc-bootstrap-bound",
        "pareto-smc-bootstrap-bound",
    }:
        specification_path = os.environ.get("MO_NCO_PARETO_SMC_SPEC", "")
        if not specification_path:
            raise ValueError(
                "Set MO_NCO_PARETO_SMC_SPEC to a predeclared Pareto-SMC "
                "specification before using the formal Pareto-SMC aliases."
            )
        specification = load_pareto_smc_specification(
            specification_path,
            objective_dimension=instance.num_objectives,
        )
        reference_count = len(specification.reference_directions)
        if population < reference_count or population % reference_count != 0:
            raise ValueError(
                "The CLI population must be a positive multiple of the "
                f"{reference_count} predeclared Pareto-SMC reference types."
            )
        deterministic_bootstrap = name in {
            "annealed-pareto-smc-bootstrap-bound",
            "pareto-smc-bootstrap-bound",
        }
        if (
            not deterministic_bootstrap
            and specification.mutation_steps_by_stage is not None
        ):
            raise ValueError(
                "A specification with mutation.steps_per_stage requires the "
                "deterministic bootstrap alias."
            )
        result = AnnealedParetoSMCOptimizer(
            instance=instance,
            particles_per_reference=population // reference_count,
            evaluations=iterations,
            seed=seed,
            beta_schedule=specification.beta_schedule,
            reference_directions=specification.reference_directions,
            num_reference_types=reference_count,
            epsilon=original_unit_cell_widths(instance, specification),
            ess_threshold=specification.ess_threshold_fraction,
            resampling_policy=(
                "always" if deterministic_bootstrap else "ess"
            ),
            mutation_steps_by_stage=(
                specification.mutation_steps_by_stage
                if deterministic_bootstrap
                else None
            ),
            finite_particle_delta=0.05,
            chebyshev_rho=specification.chebyshev_rho,
            global_refresh_probability=specification.global_refresh_probability,
            enable_exact_incremental_two_opt=(
                specification.mutation_objective_evaluation
                == EXACT_INCREMENTAL_TWO_OPT_CONTRACT
            ),
            archive_max_size=specification.archive_max_size,
        ).run()
        binding_payload = {
            "specification_sha256": specification.sha256,
            "target_context_hash": result.metadata.get("context_hash"),
            "reporting_context_hash": result.metadata.get(
                "reporting_context_hash"
            ),
            "run_contract_hash": result.metadata.get("run_contract_hash"),
        }
        binding_bytes = json.dumps(
            binding_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        result.metadata.update(
            {
                "requested_algorithm_alias": name,
                "external_specification_schema": "annealed_pareto_smc_spec_v1",
                "external_specification_path": str(specification.path),
                "external_specification_sha256": specification.sha256,
                "specification_run_binding_sha256": hashlib.sha256(
                    binding_bytes
                ).hexdigest(),
                "normalized_epsilon_cell_widths": (
                    specification.normalized_cell_widths
                ),
                "cli_population": population,
                "deterministic_bootstrap_bound_branch": (
                    deterministic_bootstrap
                ),
                "particles_split_equally_across_reference_types": True,
            }
        )
        return result
    if name in {"ips-theory-certified", "ips-certified-mh", "ips-typed-mh"}:
        resolved_configuration = _resolved_certified_configuration()
        result = CertifiedSingleSiteIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            temperature=resolved_configuration["temperature"],
            chebyshev_rho=resolved_configuration["chebyshev_rho"],
            log_period=(
                math.gcd(log_period, anytime_checkpoint_period)
                if anytime_checkpoint_period is not None
                else log_period
            ),
            uniformization_rate=resolved_configuration[
                "uniformization_rate"
            ],
            lazy_probability=resolved_configuration[
                "lazy_probability"
            ],
            archive_max_size=None,
            trace_path=certified_trace_path,
        ).run()
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            metadata["resolved_algorithm_configuration"] = (
                resolved_configuration
            )
            metadata["native_archive_completeness_gate"] = "PASS"
            metadata["native_archive_completeness_contract"] = (
                "unbounded_exact_nondominated_all_evaluated_candidates_v2"
            )
            metadata["anytime_checkpoint_emission_contract"] = (
                "diagnostic_period_gcd_exact_checkpoint_v1"
                if anytime_checkpoint_period is not None
                else "disabled"
            )
        return result
    if name in {"ips-theory-legacy", "theory-ips-legacy"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=8,
            crossover_probability=0.0,
            archive_parent_probability=0.10,
            archive_parent_sample=4,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.05,
            neural_training_epochs=3,
            neural_archive_repeats=4,
        ).run()
    if name in {
        "ips-heuristic-adaptive",
        "ips-theory",
        "theory-ips",
        "ips-neural-policy",
    }:
        result = TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=8,
            crossover_probability=0.0,
            archive_parent_probability=0.12,
            archive_parent_sample=6,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.08,
            neural_training_epochs=4,
            neural_archive_repeats=4,
            neural_proposal_probability=0.55,
            neural_proposal_weight=0.35,
            neural_candidate_pool=4,
            neural_proposal_min_samples=64,
            initialization="scalar_greedy",
            greedy_candidate_pool=3,
            initial_2opt_passes=3,
            proposal_2opt_passes=1,
        ).run()
        result.metadata["requested_algorithm_alias"] = name
        result.metadata["alias_claim_boundary"] = (
            "explicit_heuristic_descent"
            if name == "ips-heuristic-adaptive"
            else "deprecated_theory_named_alias_for_heuristic_descent"
        )
        return result
    if name in {"ips-offline-neural", "ips-frozen-neural"}:
        prior_path = _neural_prior_path_for_backend("tiny")
        if not prior_path:
            raise ValueError("Set MO_NCO_NEURAL_PRIOR_PATH for ips-offline-neural.")
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=8,
            crossover_probability=0.0,
            archive_parent_probability=0.12,
            archive_parent_sample=6,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_training_epochs=0,
            neural_online_training=False,
            neural_prior_path=prior_path,
            neural_proposal_probability=0.55,
            neural_proposal_weight=0.35,
            neural_candidate_pool=4,
            neural_proposal_min_samples=1,
            initialization="scalar_greedy",
            greedy_candidate_pool=3,
            initial_2opt_passes=3,
            proposal_2opt_passes=1,
        ).run()
    if name in {"ips-scalar-greedy", "ips-no-neural-policy"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=8,
            crossover_probability=0.0,
            archive_parent_probability=0.12,
            archive_parent_sample=6,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            initialization="mixed_scalar_greedy",
            greedy_candidate_pool=3,
            initial_2opt_passes=3,
            proposal_2opt_passes=1,
        ).run()
    if name in {"ips-descent", "ips-scalar-descent"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=8,
            crossover_probability=0.0,
            archive_parent_probability=0.12,
            archive_parent_sample=6,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            initialization="mixed_scalar_greedy",
            greedy_candidate_pool=3,
            initial_2opt_passes=3,
            proposal_2opt_passes=1,
        ).run()
    if name in {"ips-descent-deep", "ips-theory-deep"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=10,
            crossover_probability=0.0,
            archive_parent_probability=0.18,
            archive_parent_sample=8,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            initialization="scalar_greedy",
            greedy_candidate_pool=2,
            initial_2opt_passes=12,
            proposal_2opt_passes=2,
        ).run()
    if name in {"ips-neural-deep", "ips-theory-neural-deep"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=10,
            crossover_probability=0.0,
            archive_parent_probability=0.18,
            archive_parent_sample=8,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.08,
            neural_training_epochs=4,
            neural_archive_repeats=4,
            neural_proposal_probability=0.65,
            neural_proposal_weight=0.35,
            neural_candidate_pool=8,
            neural_proposal_min_samples=64,
            initialization="scalar_greedy",
            greedy_candidate_pool=2,
            initial_2opt_passes=12,
            proposal_2opt_passes=2,
        ).run()
    if name in {"ips-quality", "ips-descent-quality"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=12,
            crossover_probability=0.0,
            archive_parent_probability=0.22,
            archive_parent_sample=10,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            initialization="scalar_greedy",
            greedy_candidate_pool=1,
            initial_2opt_passes=60,
            proposal_2opt_passes=3,
        ).run()
    if name in {"ips-neural-quality", "ips-theory-neural-quality"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=12,
            crossover_probability=0.0,
            archive_parent_probability=0.22,
            archive_parent_sample=10,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.08,
            neural_hidden_units=10,
            neural_training_epochs=1,
            neural_archive_repeats=2,
            neural_proposal_probability=0.45,
            neural_proposal_weight=0.35,
            neural_candidate_pool=12,
            neural_proposal_min_samples=64,
            neural_prior_path=_neural_prior_path_for_backend("tiny"),
            neural_directional_coverage_weight=0.20,
            neural_extreme_progress_weight=0.25,
            neural_gap_fill_weight=0.12,
            neural_hv_center_bias=0.70,
            neural_extreme_repeats=3,
            neural_action_sample_pool=8,
            neural_mean_field_features=True,
            neural_prefilter_pool=12,
            neural_refine_top_k=3,
            neural_flow_pair_samples=12,
            neural_flow_residual_weight=0.7,
            neural_ranking_weight=0.12,
            neural_hypercone_loss_weight=0.18,
            neural_weight_norm_bound=3.0,
            neural_mean_field_update_period=128,
            neural_mean_field_target_weight=0.20,
            initialization="scalar_greedy",
            greedy_candidate_pool=1,
            initial_2opt_passes=60,
            proposal_2opt_passes=3,
        ).run()
    if name in {"ips-quality-relocate", "ips-relocate-quality"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=12,
            crossover_probability=0.0,
            archive_parent_probability=0.22,
            archive_parent_sample=10,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            initialization="scalar_greedy",
            greedy_candidate_pool=1,
            initial_2opt_passes=60,
            proposal_2opt_passes=3,
            initial_relocate_passes=20,
            proposal_relocate_passes=1,
        ).run()
    if name in {"ips-jitgreedy-scalar-polish", "ips-random-jit-scalar-polish"}:
        random_init = name == "ips-random-jit-scalar-polish"
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=12,
            crossover_probability=0.0,
            archive_parent_probability=0.0,
            archive_parent_sample=1,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=False,
            archive_conditioning_weight=0.0,
            neural_scalar_weight=0.0,
            neural_proposal_probability=0.0,
            neural_proposal_weight=0.0,
            neural_mean_field_features=False,
            initialization="random" if random_init else "scalar_greedy",
            greedy_candidate_pool=1 if random_init else 2,
            greedy_start_pool=1 if random_init else 8,
            initial_2opt_passes=0 if random_init else 120,
            initial_relocate_passes=0 if random_init else 45,
            proposal_2opt_passes=16,
            proposal_relocate_passes=6,
            jit_polish_fraction=0.0,
        ).run()
    if name in {
        "ips-neural-mv-jitgreedy-sota",
        "ips-neural-mv-jitgreedy-sota-no-mf",
        "ips-neural-mv-jitgreedy-paretoflow",
        "ips-neural-mv-jitgreedy-paretoflow-no-mf",
        "ips-neural-mv-jitgreedy-pcd",
        "ips-neural-mv-jitgreedy-pcd-no-mf",
        "ips-neural-mv-jitgreedy-targetflow",
        "ips-neural-mv-jitgreedy-targetflow-no-mf",
        "ips-neural-mv-jitgreedy-targetflow-no-cfg",
        "ips-neural-mv-jitgreedy-targetflow-no-flow-consistency",
        "ips-neural-mv-jitgreedy-targetflow-efficient",
        "ips-neural-mv-jitgreedy-targetflow-theory-optimized",
        "ips-theory-heavy-no-prior",
        "ips-theory-endpoint-only",
        "ips-theory-move-only",
    }:
        theory_ablation_mode = {
            "ips-neural-mv-jitgreedy-targetflow-theory-optimized": "full",
            "ips-theory-heavy-no-prior": "none",
            "ips-theory-endpoint-only": "scalar",
            "ips-theory-move-only": "move",
        }.get(name, "")
        targetflow = "-targetflow" in name or bool(theory_ablation_mode)
        # The held-out certificate showed no reproducible final-HV gain from
        # mean-field/CFG/flow-consistency, while every one of those paths added
        # wall-clock cost.  Keep the original variants intact for ablations and
        # expose a conservative, prior-only proposal branch for confirmation.
        theory_optimized_targetflow = theory_ablation_mode == "full"
        efficient_targetflow = name.endswith("-targetflow-efficient") or bool(theory_ablation_mode)
        use_mean_field = "-no-mf" not in name and not efficient_targetflow
        no_cfg = "-no-cfg" in name or efficient_targetflow
        no_flow_consistency = "-no-flow-consistency" in name or efficient_targetflow
        neural_backend = "pcd" if ("-pcd" in name or targetflow) else "paretoflow"
        scalar_prior_path = _neural_prior_path_for_backend(neural_backend)
        move_prior_path = os.environ.get("MO_NCO_LEARNED_MOVE_PRIOR_PATH", "")
        scalar_file_ready = bool(scalar_prior_path) and Path(scalar_prior_path).is_file()
        move_file_ready = bool(move_prior_path) and Path(move_prior_path).is_file()
        scalar_enabled = theory_ablation_mode in {"full", "scalar"} if theory_ablation_mode else scalar_file_ready
        move_enabled = theory_ablation_mode in {"full", "move"} if theory_ablation_mode else move_file_ready
        efficient_scalar_ready = scalar_enabled and scalar_file_ready
        efficient_move_ready = move_enabled and move_file_ready
        if theory_ablation_mode in {"full", "scalar"} and not efficient_scalar_ready:
            raise FileNotFoundError(
                f"{name} requires MO_NCO_NEURAL_PRIOR_PATH_PCD to point to an endpoint_state_v1 prior."
            )
        if theory_ablation_mode in {"full", "move"} and not efficient_move_ready:
            raise FileNotFoundError(
                f"{name} requires MO_NCO_LEARNED_MOVE_PRIOR_PATH to point to a target-only move prior."
            )
        learned_move_probability = (
            (0.45 if efficient_move_ready else 0.0)
            if efficient_targetflow
            else (0.90 if targetflow else (0.85 if neural_backend == "pcd" else 0.0))
        )
        flow_pair_samples = 6 if (targetflow and use_mean_field and not no_flow_consistency) else (
            4 if use_mean_field else 0
        )
        flow_residual_weight = 0.32 if (targetflow and use_mean_field and not no_flow_consistency) else (
            0.25 if use_mean_field else 0.0
        )
        coverage_pair_weight = 0.18 if (targetflow and use_mean_field and not no_flow_consistency) else (
            0.10 if use_mean_field else 0.0
        )
        expert_pair_weight = 0.14 if (targetflow and use_mean_field and not no_flow_consistency) else (
            0.10 if use_mean_field else 0.0
        )
        guidance_scale = 1.55 if (targetflow and use_mean_field and not no_cfg) else (
            1.6 if neural_backend == "pcd" and use_mean_field and not no_cfg else 1.0
        )
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=14,
            crossover_probability=0.0,
            archive_parent_probability=0.24,
            archive_parent_sample=14,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.2,
            neural_scalar_weight=(0.06 if efficient_scalar_ready else 0.0) if efficient_targetflow else 0.10,
            enable_neural_scalar=efficient_scalar_ready if efficient_targetflow else True,
            neural_backend=neural_backend,
            neural_hidden_units=64 if targetflow else (48 if neural_backend == "pcd" else 96),
            neural_training_epochs=1,
            neural_online_training=not efficient_targetflow,
            neural_fit_period=16 if efficient_targetflow else 8,
            neural_archive_repeats=1 if efficient_targetflow else 2,
            neural_proposal_probability=(0.14 if (efficient_scalar_ready or efficient_move_ready) else 0.0)
            if efficient_targetflow
            else 0.24,
            neural_proposal_weight=0.30 if efficient_targetflow else 0.45,
            neural_candidate_pool=2 if neural_backend == "pcd" else 16,
            neural_proposal_min_samples=64,
            neural_prior_path=(scalar_prior_path if efficient_scalar_ready else "")
            if efficient_targetflow
            else scalar_prior_path,
            require_endpoint_only_prior=theory_ablation_mode in {"full", "scalar"},
            neural_directional_coverage_weight=0.24,
            neural_extreme_progress_weight=0.34,
            neural_gap_fill_weight=0.20,
            neural_hv_center_bias=0.80,
            neural_extreme_repeats=5,
            neural_action_sample_pool=1 if neural_backend == "pcd" else 32,
            neural_mean_field_features=use_mean_field,
            neural_prefilter_pool=1 if neural_backend == "pcd" else 64,
            neural_refine_top_k=1 if neural_backend == "pcd" else 12,
            neural_exact_two_opt_prefilter=False if neural_backend == "pcd" else True,
            neural_flow_pair_samples=flow_pair_samples,
            neural_flow_residual_weight=flow_residual_weight,
            neural_ranking_weight=0.08 if (targetflow and use_mean_field and not no_flow_consistency) else (
                0.06 if use_mean_field else 0.0
            ),
            neural_hypercone_loss_weight=0.10 if (targetflow and use_mean_field and not no_flow_consistency) else (
                0.08 if use_mean_field else 0.0
            ),
            neural_coverage_pair_weight=coverage_pair_weight,
            neural_expert_pair_weight=expert_pair_weight,
            neural_expert_pair_samples=8 if (targetflow and use_mean_field and not no_flow_consistency) else (
                6 if use_mean_field else 0
            ),
            neural_weight_norm_bound=3.0,
            neural_mean_field_update_period=96,
            neural_mean_field_target_weight=0.16 if (targetflow and use_mean_field) else (0.10 if use_mean_field else 0.0),
            neural_active_fraction=0.08 if efficient_targetflow else 0.22,
            neural_stagnation_patience=log_period,
            neural_stagnation_epsilon=0.0,
            neural_stagnation_wake_steps=max(64, log_period) if efficient_targetflow else max(96, log_period),
            neural_late_repair_fraction=0.04 if efficient_targetflow else 0.08,
            neural_rank_fusion_weight=0.0 if efficient_targetflow else (
                0.72 if (targetflow and use_mean_field) else (0.65 if use_mean_field else 0.35)
            ),
            neural_mean_field_guidance_weight=0.18 if (targetflow and use_mean_field) else 0.0,
            neural_gap_direction_probability=0.08 if (targetflow and use_mean_field) else 0.0,
            neural_learned_move_probability=learned_move_probability,
            neural_learned_move_sparse_nodes=14 if efficient_targetflow else (22 if targetflow else 18),
            neural_learned_move_sparse_partners=14 if efficient_targetflow else (22 if targetflow else 18),
            neural_learned_move_samples=4 if efficient_move_ready and efficient_targetflow else (
                3 if targetflow else (2 if neural_backend == "pcd" else 1)
            ),
            neural_learned_move_learning_rate=0.02 if efficient_targetflow else 0.035,
            neural_learned_move_prior_path=(move_prior_path if efficient_move_ready else "")
            if efficient_targetflow
            else move_prior_path,
            require_target_only_move_prior=theory_ablation_mode in {"full", "move"},
            allow_move_without_scalar=theory_ablation_mode == "move",
            ablation_contract=(f"theory_search_v2:{theory_ablation_mode}" if theory_ablation_mode else ""),
            isolate_prior_loading_rng=bool(theory_ablation_mode),
            enable_mechanism_diagnostics=bool(theory_ablation_mode),
            neural_condition_guidance_scale=guidance_scale,
            neural_front_reweighting_strength=0.40 if (targetflow and use_mean_field) else (
                0.35 if neural_backend == "pcd" and use_mean_field else 0.0
            ),
            extreme_anchor_fraction=0.08 if efficient_targetflow else 0.12,
            extreme_anchor_period=8 if efficient_targetflow else 4,
            initialization="scalar_greedy",
            greedy_candidate_pool=2,
            greedy_start_pool=8,
            initial_2opt_passes=120,
            proposal_2opt_passes=16,
            initial_relocate_passes=45,
            proposal_relocate_passes=6,
            jit_polish_fraction=0.16 if efficient_targetflow else 0.12,
            jit_polish_chunk_size=log_period,
        ).run()
    if name in {
        "ips-neural-quality-relocate",
        "ips-neural-relocate-quality",
        "ips-neural-mv-lite",
        "ips-neural-mv-lite-no-mf",
        "ips-neural-mv-lite-no-topk",
        "ips-neural-mv-scheduled",
        "ips-neural-mv-scheduled-no-mf",
        "ips-neural-mv-exacttopk",
        "ips-neural-mv-exacttopk-no-mf",
        "ips-neural-mv-fast",
        "ips-neural-mv-fast-no-mf",
        "ips-neural-mv-polish",
        "ips-neural-mv-polish-no-mf",
        "ips-neural-mv-compact",
        "ips-neural-mv-compact-no-mf",
        "ips-neural-mv-jitpolish",
        "ips-neural-mv-jitpolish-no-mf",
        "ips-neural-mv-jitgreedy",
        "ips-neural-mv-jitgreedy-no-mf",
        "ips-neural-mv-jitgreedy-fast",
        "ips-neural-mv-jitgreedy-fast-no-mf",
        "ips-neural-mv-jitgreedy-balanced",
        "ips-neural-mv-jitgreedy-balanced-no-mf",
        "ips-neural-mv-jitgreedy-sprint",
        "ips-neural-mv-jitgreedy-sprint-no-mf",
        "ips-neural-mv-jitgreedy-sprint-adaptive",
        "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
    }:
        use_mean_field = name not in {
            "ips-neural-mv-lite-no-mf",
            "ips-neural-mv-scheduled-no-mf",
            "ips-neural-mv-exacttopk-no-mf",
            "ips-neural-mv-fast-no-mf",
            "ips-neural-mv-polish-no-mf",
            "ips-neural-mv-compact-no-mf",
            "ips-neural-mv-jitpolish-no-mf",
            "ips-neural-mv-jitgreedy-no-mf",
            "ips-neural-mv-jitgreedy-fast-no-mf",
            "ips-neural-mv-jitgreedy-balanced-no-mf",
            "ips-neural-mv-jitgreedy-sprint-no-mf",
            "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
        }
        use_topk = name != "ips-neural-mv-lite-no-topk"
        scheduled = name in {"ips-neural-mv-scheduled", "ips-neural-mv-scheduled-no-mf"}
        exact_topk = name in {"ips-neural-mv-exacttopk", "ips-neural-mv-exacttopk-no-mf"}
        fast = name in {
            "ips-neural-mv-fast",
            "ips-neural-mv-fast-no-mf",
            "ips-neural-mv-polish",
            "ips-neural-mv-polish-no-mf",
            "ips-neural-mv-compact",
            "ips-neural-mv-compact-no-mf",
            "ips-neural-mv-jitpolish",
            "ips-neural-mv-jitpolish-no-mf",
            "ips-neural-mv-jitgreedy",
            "ips-neural-mv-jitgreedy-no-mf",
            "ips-neural-mv-jitgreedy-fast",
            "ips-neural-mv-jitgreedy-fast-no-mf",
            "ips-neural-mv-jitgreedy-balanced",
            "ips-neural-mv-jitgreedy-balanced-no-mf",
            "ips-neural-mv-jitgreedy-sprint",
            "ips-neural-mv-jitgreedy-sprint-no-mf",
            "ips-neural-mv-jitgreedy-sprint-adaptive",
            "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
        }
        polish = name in {
            "ips-neural-mv-polish",
            "ips-neural-mv-polish-no-mf",
            "ips-neural-mv-jitpolish",
            "ips-neural-mv-jitpolish-no-mf",
            "ips-neural-mv-jitgreedy",
            "ips-neural-mv-jitgreedy-no-mf",
            "ips-neural-mv-jitgreedy-fast",
            "ips-neural-mv-jitgreedy-fast-no-mf",
            "ips-neural-mv-jitgreedy-balanced",
            "ips-neural-mv-jitgreedy-balanced-no-mf",
            "ips-neural-mv-jitgreedy-sprint",
            "ips-neural-mv-jitgreedy-sprint-no-mf",
            "ips-neural-mv-jitgreedy-sprint-adaptive",
            "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
        }
        jit_polish = name in {
            "ips-neural-mv-jitpolish",
            "ips-neural-mv-jitpolish-no-mf",
            "ips-neural-mv-jitgreedy",
            "ips-neural-mv-jitgreedy-no-mf",
            "ips-neural-mv-jitgreedy-fast",
            "ips-neural-mv-jitgreedy-fast-no-mf",
            "ips-neural-mv-jitgreedy-balanced",
            "ips-neural-mv-jitgreedy-balanced-no-mf",
            "ips-neural-mv-jitgreedy-sprint",
            "ips-neural-mv-jitgreedy-sprint-no-mf",
            "ips-neural-mv-jitgreedy-sprint-adaptive",
            "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
        }
        jit_greedy = name in {
            "ips-neural-mv-jitgreedy",
            "ips-neural-mv-jitgreedy-no-mf",
            "ips-neural-mv-jitgreedy-fast",
            "ips-neural-mv-jitgreedy-fast-no-mf",
            "ips-neural-mv-jitgreedy-balanced",
            "ips-neural-mv-jitgreedy-balanced-no-mf",
            "ips-neural-mv-jitgreedy-sprint",
            "ips-neural-mv-jitgreedy-sprint-no-mf",
            "ips-neural-mv-jitgreedy-sprint-adaptive",
            "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
        }
        jit_greedy_fast = name in {"ips-neural-mv-jitgreedy-fast", "ips-neural-mv-jitgreedy-fast-no-mf"}
        jit_greedy_balanced = name in {
            "ips-neural-mv-jitgreedy-balanced",
            "ips-neural-mv-jitgreedy-balanced-no-mf",
        }
        jit_greedy_sprint = name in {
            "ips-neural-mv-jitgreedy-sprint",
            "ips-neural-mv-jitgreedy-sprint-no-mf",
        }
        jit_greedy_sprint_adaptive = name in {
            "ips-neural-mv-jitgreedy-sprint-adaptive",
            "ips-neural-mv-jitgreedy-sprint-adaptive-no-mf",
        }
        jit_greedy_sprint_any = jit_greedy_sprint or jit_greedy_sprint_adaptive
        compact = name in {"ips-neural-mv-compact", "ips-neural-mv-compact-no-mf"}
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=min(population, 24) if compact else population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=8 if compact else 12,
            crossover_probability=0.0,
            archive_parent_probability=0.22,
            archive_parent_sample=10,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.08,
            neural_hidden_units=10,
            neural_training_epochs=1,
            neural_fit_period=12
            if jit_greedy_sprint_any
            else (6 if jit_greedy_balanced else (8 if jit_greedy_fast else (4 if fast else 1))),
            neural_archive_repeats=2,
            neural_proposal_probability=0.16
            if jit_greedy_sprint_adaptive
            else 0.14
            if jit_greedy_sprint
            else (
                0.24
                if jit_greedy_balanced
                else (
                    0.18
                    if jit_greedy_fast
                    else (0.30 if polish else (0.36 if fast else (0.32 if exact_topk else (0.40 if scheduled else 0.45))))
                )
            ),
            neural_proposal_weight=0.35,
            neural_candidate_pool=16 if exact_topk else (10 if scheduled else 12),
            neural_proposal_min_samples=64,
            neural_prior_path=_neural_prior_path_for_backend("tiny"),
            neural_directional_coverage_weight=0.20,
            neural_extreme_progress_weight=0.25,
            neural_gap_fill_weight=0.12,
            neural_hv_center_bias=0.70,
            neural_extreme_repeats=3,
            neural_action_sample_pool=16 if exact_topk else (8 if use_topk else 1),
            neural_mean_field_features=use_mean_field,
            neural_prefilter_pool=24 if exact_topk else (10 if scheduled and use_topk else (12 if use_topk else 1)),
            neural_refine_top_k=4 if exact_topk else (2 if scheduled and use_topk else (3 if use_topk else 1)),
            neural_exact_two_opt_prefilter=exact_topk,
            neural_flow_pair_samples=3
            if jit_greedy_sprint_adaptive and use_mean_field
            else 2
            if jit_greedy_sprint and use_mean_field
            else (
                6
                if jit_greedy_balanced and use_mean_field
                else (4 if jit_greedy_fast and use_mean_field else (8 if fast and use_mean_field else (12 if use_mean_field else 0)))
            ),
            neural_flow_residual_weight=0.7 if use_mean_field else 0.0,
            neural_ranking_weight=0.12 if use_mean_field else 0.0,
            neural_hypercone_loss_weight=0.18 if use_mean_field else 0.0,
            neural_weight_norm_bound=3.0,
            neural_mean_field_update_period=128,
            neural_mean_field_target_weight=0.20 if use_mean_field else 0.0,
            neural_active_fraction=0.16
            if jit_greedy_sprint_adaptive
            else 0.14
            if jit_greedy_sprint
            else (0.32 if jit_greedy_balanced else (0.20 if jit_greedy_fast else (0.55 if scheduled else 1.0))),
            neural_stagnation_patience=log_period if jit_greedy_sprint_adaptive else 0,
            neural_stagnation_epsilon=0.0,
            neural_stagnation_wake_steps=max(64, log_period // 2) if jit_greedy_sprint_adaptive else 0,
            extreme_anchor_fraction=0.10 if jit_greedy_sprint_adaptive else 0.0,
            extreme_anchor_period=4 if jit_greedy_sprint_adaptive else 0,
            initialization="scalar_greedy",
            greedy_candidate_pool=2 if jit_greedy else 1,
            greedy_start_pool=8 if jit_greedy else 1,
            initial_2opt_passes=120 if jit_greedy else (90 if polish else 60),
            proposal_2opt_passes=16 if jit_greedy_sprint_any else (10 if jit_greedy else (8 if polish else 3)),
            initial_relocate_passes=45 if jit_greedy else (35 if polish else 20),
            proposal_relocate_passes=6 if jit_greedy_sprint_any else (4 if jit_greedy else (3 if polish else 1)),
            jit_polish_fraction=0.12
            if jit_greedy_sprint_any
            else (
                0.32
                if jit_greedy_balanced
                else (0.18 if jit_greedy_fast else (0.55 if jit_greedy else (0.45 if jit_polish else 1.1)))
            ),
            jit_polish_chunk_size=log_period if jit_greedy_sprint_adaptive else 0,
        ).run()
    if name in {"ips-neural-mv-strong", "ips-neural-mv-strong-no-mf"}:
        use_mean_field = name != "ips-neural-mv-strong-no-mf"
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=16,
            crossover_probability=0.0,
            archive_parent_probability=0.26,
            archive_parent_sample=16,
            archive_update_period=64,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=3.0,
            neural_scalar_weight=0.08,
            neural_hidden_units=10,
            neural_training_epochs=1,
            neural_archive_repeats=2,
            neural_proposal_probability=0.55,
            neural_proposal_weight=0.35,
            neural_candidate_pool=16,
            neural_proposal_min_samples=64,
            neural_prior_path=_neural_prior_path_for_backend("tiny"),
            neural_directional_coverage_weight=0.24,
            neural_extreme_progress_weight=0.30,
            neural_gap_fill_weight=0.16,
            neural_hv_center_bias=0.75,
            neural_extreme_repeats=4,
            neural_action_sample_pool=12,
            neural_mean_field_features=use_mean_field,
            neural_prefilter_pool=16,
            neural_refine_top_k=4,
            neural_flow_pair_samples=16 if use_mean_field else 0,
            neural_flow_residual_weight=0.75 if use_mean_field else 0.0,
            neural_ranking_weight=0.16 if use_mean_field else 0.0,
            neural_hypercone_loss_weight=0.22 if use_mean_field else 0.0,
            neural_weight_norm_bound=3.0,
            neural_mean_field_update_period=96,
            neural_mean_field_target_weight=0.28 if use_mean_field else 0.0,
            initialization="scalar_greedy",
            greedy_candidate_pool=1,
            initial_2opt_passes=80,
            proposal_2opt_passes=4,
            initial_relocate_passes=30,
            proposal_relocate_passes=2,
        ).run()
    if name in {"ips-theory-core", "ips-efficient", "ips-fast", "ips-zero"}:
        return TheoryAlignedIPSOptimizer(
            instance=instance,
            num_particles=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            neighbor_size=6,
            crossover_probability=0.0,
            archive_parent_probability=0.0,
            archive_update_period=max(iterations + 1, 1),
            archive_parent_sample=1,
            proposal="two_opt",
            extra_two_opt_probability=0.0,
            initial_temperature=0.0,
            final_temperature=0.0,
            archive_conditioning=True,
            archive_conditioning_weight=0.0,
        ).run()
    if name == "ips":
        return IPSMetropolisOptimizer(
            instance=instance,
            num_particles=population,
            iterations=iterations,
            seed=seed,
            initial_temperature=0.18,
            final_temperature=0.006,
            log_period=log_period,
            archive_update_period=archive_update_period,
            potential=HypervolumeArchivePotential(
                single_weight=0.08,
                coverage_weight=5.0,
                diversity_weight=0.02,
                diversity_sigma=0.12,
            ),
            candidate_trials=3,
            selection_tournament=4,
            resample_fraction=0.25,
            crossover_probability=0.35,
            local_search_steps=12,
            local_search_directions=7,
            directional_probability=0.65,
        ).run()
    if name in {"ips-neural", "neural"}:
        return IPSMetropolisOptimizer(
            instance=instance,
            num_particles=population,
            iterations=iterations,
            seed=seed,
            initial_temperature=0.18,
            final_temperature=0.006,
            log_period=log_period,
            archive_update_period=archive_update_period,
            potential=NeuralScalarPotential(
                seed=seed,
                single_weight=0.08,
                coverage_weight=5.0,
                diversity_weight=0.02,
                diversity_sigma=0.12,
                training_epochs=120,
            ),
            candidate_trials=3,
            selection_tournament=4,
            resample_fraction=0.25,
            crossover_probability=0.35,
            local_search_steps=12,
            local_search_directions=7,
            directional_probability=0.65,
        ).run()
    if name == "nsga2":
        return NSGAIIOptimizer(
            instance=instance,
            population_size=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
        ).run()
    if name == "moead":
        return MOEADOptimizer(
            instance=instance,
            population_size=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
        ).run()
    if name in {"motsp-pls", "pls", "tpls", "mogls"}:
        result = MOTSPParetoLocalSearchOptimizer(
            instance=instance,
            population_size=population,
            evaluations=iterations,
            seed=seed,
            log_period=log_period,
            archive_max_size=None,
            neighborhood_sample=max(8, min(48, population)),
            scalar_guided=name in {"tpls", "mogls", "motsp-pls"},
            anytime_checkpoint_period=anytime_checkpoint_period,
        ).run()
        result.metadata["native_archive_completeness_gate"] = "PASS"
        result.metadata["native_archive_completeness_contract"] = (
            "unbounded_exact_nondominated_all_evaluated_candidates_v2"
        )
        result.metadata["anytime_checkpoint_emission_contract"] = (
            "per_evaluation_passive_archive_snapshot_v1"
            if anytime_checkpoint_period is not None
            else "disabled"
        )
        return result
    if name in {"random", "random2opt", "random-2opt"}:
        return RandomTwoOptOptimizer(
            instance=instance,
            num_particles=population,
            iterations=iterations,
            seed=seed,
            log_period=log_period,
            archive_update_period=archive_update_period,
        ).run()
    if (
        name.startswith("external-")
        or name.startswith("pymoo-")
        or name.startswith("jmetal-")
        or name.startswith("platemo-")
        or name
        in {
            "lkh-scalar",
            "elkai-lkh",
            "lkh-derived",
            "lkh-official",
            "lkh3-official",
            "official-lkh",
            "lkh-2ppls",
            "official-lkh-2ppls",
            "tpls-lkh-official",
            "paquete-published-tpls",
            "tpls-published",
            "published-tpls",
            "paquete",
            "tpls-external",
            "mogls-external",
        }
    ):
        config = (
            builtin_pymoo_baseline_configuration(name)
            if name in {"pymoo-nsga2", "pymoo-moead"}
            else load_external_baseline_from_env(name)
        )
        return ExternalBaselineOptimizer(
            instance=instance,
            config=config,
            population_size=population,
            evaluations=iterations,
            seed=seed,
            archive_max_size=None,
        ).run()
    raise ValueError(f"Unknown algorithm: {algorithm}")


def build_execution_schedule(
    algorithms: Sequence[str],
    seeds: Sequence[int],
    execution_order: str = "algorithm-major",
    rotation_offset: int = 0,
) -> Tuple[Tuple[str, int], ...]:
    algorithm_list = tuple(algorithms)
    seed_list = tuple(seeds)
    if execution_order == "algorithm-major":
        return tuple((algorithm, seed) for algorithm in algorithm_list for seed in seed_list)
    if execution_order != "seed-major-balanced-v1":
        raise ValueError(
            "execution_order must be 'algorithm-major' or 'seed-major-balanced-v1'."
        )
    if not algorithm_list:
        return ()
    schedule: List[Tuple[str, int]] = []
    for seed_position, seed in enumerate(seed_list):
        offset = (int(rotation_offset) + seed_position) % len(algorithm_list)
        rotated = algorithm_list[offset:] + algorithm_list[:offset]
        schedule.extend((algorithm, seed) for algorithm in rotated)
    return tuple(schedule)


def _validated_metric_reference(payload: Dict[str, object]) -> Tuple[
    ObjectiveVector,
    Tuple[ObjectiveVector, ...],
    ObjectiveVector,
    ObjectiveVector,
    str,
]:
    contract = str(payload.get("contract", ""))
    if contract != "frozen_external_v1":
        raise ValueError("Metric reference contract must be 'frozen_external_v1'.")
    try:
        reference = tuple(float(value) for value in payload["hypervolume_reference"])  # type: ignore[index]
        ideal = tuple(float(value) for value in payload["ideal"])  # type: ignore[index]
        nadir = tuple(float(value) for value in payload["nadir"])  # type: ignore[index]
        reference_front = tuple(
            tuple(float(value) for value in point)
            for point in payload["reference_front"]  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed frozen metric reference payload.") from exc
    if len(reference) != 2 or len(ideal) != 2 or len(nadir) != 2:
        raise ValueError("Frozen metric reference currently requires two objectives.")
    if not reference_front or any(len(point) != 2 for point in reference_front):
        raise ValueError("Frozen metric reference_front must contain two-dimensional points.")
    if any(not math.isfinite(value) for value in (*reference, *ideal, *nadir)):
        raise ValueError("Frozen metric reference contains non-finite bounds.")
    if any(
        not math.isfinite(value)
        for point in reference_front
        for value in point
    ):
        raise ValueError(
            "Frozen metric reference_front contains a non-finite value."
        )
    if any(nadir[idx] <= ideal[idx] for idx in range(2)):
        raise ValueError("Frozen metric nadir must be strictly larger than ideal on every axis.")
    if any(reference[idx] < nadir[idx] for idx in range(2)):
        raise ValueError(
            "Hypervolume reference must weakly dominate the frozen nadir."
        )
    if any(
        point[idx] < ideal[idx] or point[idx] > nadir[idx]
        for point in reference_front
        for idx in range(2)
    ):
        raise ValueError(
            "Every reference-front point must lie inside the frozen "
            "ideal-nadir box."
        )
    if any(point[idx] > reference[idx] for point in reference_front for idx in range(2)):
        raise ValueError("Hypervolume reference must weakly dominate every reference-front point.")
    if len(set(reference_front)) != len(reference_front):
        raise ValueError(
            "Frozen metric reference_front must contain unique points."
        )
    for left_index, left in enumerate(reference_front):
        for right_index, right in enumerate(reference_front):
            if left_index == right_index:
                continue
            if all(
                left[axis] <= right[axis]
                for axis in range(2)
            ) and any(
                left[axis] < right[axis]
                for axis in range(2)
            ):
                raise ValueError(
                    "Frozen metric reference_front must be mutually "
                    "nondominated."
                )
    return reference, reference_front, ideal, nadir, contract


def run_benchmark(
    algorithms: Sequence[str],
    seeds: Sequence[int],
    cities: int,
    population: int,
    iterations: int,
    instance_seed: int,
    output_dir: Path,
    log_period: int,
    archive_update_period: int,
    instance: Optional[MultiObjectiveTSPInstance] = None,
    execution_order: str = "algorithm-major",
    execution_order_offset: int = 0,
    metric_reference: Optional[Dict[str, object]] = None,
    metric_reference_manifest_sha256: str = "",
    certified_traces: bool = False,
    measure_python_memory: bool = False,
    output_archive_limit: Optional[int] = None,
    case_name: Optional[str] = None,
    expected_algorithm_configurations: Optional[
        Dict[Tuple[str, int], str]
    ] = None,
    anytime_checkpoint_period: Optional[int] = None,
) -> Tuple[List[RunRecord], Dict[str, Dict[str, float]]]:
    if output_archive_limit is not None and output_archive_limit <= 0:
        raise ValueError("output_archive_limit must be positive when provided.")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = output_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if instance is None:
        instance = MultiObjectiveTSPInstance.random_biobjective(cities, seed=instance_seed)
    resolved_case_name = case_name or instance.name
    schedule = build_execution_schedule(
        algorithms,
        seeds,
        execution_order=execution_order,
        rotation_offset=execution_order_offset,
    )
    predeclared_configurations = {
        (algorithm, seed): resolve_predeclared_algorithm_configuration(
            case_name=resolved_case_name,
            instance=instance,
            algorithm=algorithm,
            seed=seed,
            population=population,
            iterations=iterations,
            log_period=log_period,
            archive_update_period=archive_update_period,
            output_archive_limit=output_archive_limit,
            certified_traces=certified_traces,
            anytime_checkpoint_period=anytime_checkpoint_period,
        )
        for algorithm, seed in schedule
    }
    if expected_algorithm_configurations is not None:
        if set(expected_algorithm_configurations) != set(
            predeclared_configurations
        ):
            raise ValueError(
                "The frozen configuration manifest does not contain the "
                "exact algorithm x seed matrix for this case."
            )
        for key, configuration in predeclared_configurations.items():
            if expected_algorithm_configurations[key] != configuration.sha256:
                raise ValueError(
                    "Prelaunch algorithm configuration mismatch for "
                    f"case={resolved_case_name}, algorithm={key[0]}, "
                    f"seed={key[1]}."
                )
    raw_results: List[Tuple[str, int, OptimizationResult, float, Path, int]] = []
    try:
        from .numba_kernels import warmup_numba_kernels

        warmup_numba_kernels()
    except Exception:
        pass

    if measure_python_memory and tracemalloc.is_tracing():
        raise RuntimeError(
            "Python memory measurement requires exclusive tracemalloc "
            "ownership; tracing is already active."
        )

    def apply_common_archive_limit(
        candidate: OptimizationResult,
    ) -> OptimizationResult:
        candidate.metadata.setdefault(
            "native_archive_completeness_gate",
            "NOT_RUN",
        )
        candidate.metadata.setdefault(
            "native_archive_completeness_contract",
            "not_declared",
        )
        candidate.metadata["anytime_front_semantics"] = (
            "cumulative_nondominated_best_so_far_v1"
        )
        bounded_archive = ParetoArchive(max_size=output_archive_limit)
        bounded_archive.update(candidate.archive.entries)
        bounded_diagnostics = []
        cumulative_snapshot = ParetoArchive(max_size=None)
        point_serial = 0
        for diagnostic in candidate.diagnostics:
            canonical_points = tuple(
                sorted(
                    {
                        tuple(float(value) for value in point)
                        for point in diagnostic.front
                    }
                )
            )
            snapshot_entries = []
            for point in canonical_points:
                snapshot_entries.append(
                    ArchiveEntry(
                        tour=(point_serial,),
                        objectives=point,
                    )
                )
                point_serial += 1
            cumulative_snapshot.update(snapshot_entries)
            snapshot = ParetoArchive(max_size=output_archive_limit)
            snapshot.update(cumulative_snapshot.entries)
            bounded_front = tuple(
                entry.objectives for entry in snapshot.entries
            )
            bounded_diagnostics.append(
                replace(
                    diagnostic,
                    archive_size=len(bounded_front),
                    hypervolume_2d=(
                        snapshot.hypervolume_2d()
                        if bounded_front
                        and len(bounded_front[0]) == 2
                        else 0.0
                    ),
                    front=bounded_front,
                )
            )
        if (
            candidate.metadata["native_archive_completeness_gate"]
            == "PASS"
        ):
            final_snapshot = {
                entry.objectives
                for entry in cumulative_snapshot.entries
            }
            final_archive = {
                entry.objectives for entry in candidate.archive.entries
            }
            if not bounded_diagnostics or final_snapshot != final_archive:
                raise RuntimeError(
                    "A declared complete native archive does not match the "
                    "last cumulative pre-cap diagnostic front."
                )
        bounded = OptimizationResult(
            particles=candidate.particles,
            objectives=candidate.objectives,
            archive=bounded_archive,
            diagnostics=tuple(bounded_diagnostics),
            metadata=candidate.metadata,
        )
        bounded.metadata.update(
            {
                "common_output_archive_limit": output_archive_limit,
                "common_output_archive_policy": (
                    "nondominated_crowding_truncation_v1"
                    if output_archive_limit is not None
                    else "unbounded"
                ),
                "common_output_archive_postprocessing": (
                    output_archive_limit is not None
                ),
                "max_diagnostic_archive_size": max(
                    (
                        len(diagnostic.front)
                        for diagnostic in bounded_diagnostics
                    ),
                    default=0,
                ),
                "diagnostic_archive_limit_gate": (
                    "PASS"
                    if output_archive_limit is not None
                    else "NOT_RUN"
                ),
                "diagnostic_archive_limit_contract": (
                    "deterministic_nondominated_crowding_"
                    "truncation_per_snapshot_v1"
                    if output_archive_limit is not None
                    else "disabled"
                ),
            }
        )
        return bounded

    for algorithm, seed in schedule:
        predeclared_configuration = predeclared_configurations[
            (algorithm, seed)
        ]
        start = time.perf_counter()
        counted_instance = CountingTSPInstance(instance, max_evaluations=iterations)
        setattr(
            counted_instance,
            "anytime_checkpoint_period",
            anytime_checkpoint_period,
        )
        old_spectral_log = os.environ.get("MO_NCO_NEURAL_SPECTRAL_LOG")
        spectral_log = output_dir / "neural_spectral" / f"{algorithm}_seed{seed}.jsonl"
        os.environ["MO_NCO_NEURAL_SPECTRAL_LOG"] = str(spectral_log)
        try:
            certified_trace_path = None
            if certified_traces and algorithm.lower() in {
                "ips-theory-certified",
                "ips-certified-mh",
                "ips-typed-mh",
            }:
                trace_dir = output_dir / "kernel_traces"
                trace_dir.mkdir(parents=True, exist_ok=True)
                certified_trace_path = trace_dir / f"{algorithm}_seed{seed}.jsonl"
            result = apply_common_archive_limit(
                run_algorithm(
                    algorithm=algorithm,
                    instance=counted_instance,  # type: ignore[arg-type]
                    seed=seed,
                    population=population,
                    iterations=iterations,
                    log_period=log_period,
                    archive_update_period=archive_update_period,
                    certified_trace_path=certified_trace_path,
                    anytime_checkpoint_period=(
                        anytime_checkpoint_period
                    ),
                )
            )
        finally:
            if old_spectral_log is None:
                os.environ.pop("MO_NCO_NEURAL_SPECTRAL_LOG", None)
            else:
                os.environ["MO_NCO_NEURAL_SPECTRAL_LOG"] = old_spectral_log
        runtime = time.perf_counter() - start
        observed_evaluations = evaluation_count(counted_instance)
        pilot_evaluations = int(
            result.metadata.get("pilot_evaluations", 0)
        )
        confirm_evaluations = int(
            result.metadata.get("confirm_evaluations", 0)
        )
        is_pilot_confirm = (
            pilot_evaluations > 0 or confirm_evaluations > 0
        )
        search_evaluations = (
            0 if is_pilot_confirm else observed_evaluations
        )
        if (
            search_evaluations
            != predeclared_configuration.search_evaluations
            or pilot_evaluations
            != predeclared_configuration.pilot_evaluations
            or confirm_evaluations
            != predeclared_configuration.confirm_evaluations
        ):
            raise RuntimeError(
                "The realized search/pilot/confirm budget split does not "
                "match the predeclared configuration."
            )
        external_evaluation_gate = result.metadata.get(
            "external_evaluation_evidence_gate"
        )
        if external_evaluation_gate is None:
            evaluation_evidence_contract = (
                "inprocess_counting_instance_exact_budget_v1"
            )
            evaluation_evidence_gate = (
                "PASS" if observed_evaluations == iterations else "FAIL"
            )
        else:
            evaluation_evidence_contract = str(
                result.metadata.get(
                    "external_evaluation_evidence_contract",
                    "missing",
                )
            )
            evaluation_evidence_gate = (
                "PASS"
                if external_evaluation_gate == "PASS"
                and observed_evaluations == iterations
                else "FAIL"
            )

        result.metadata["runtime_environment_fingerprint"] = (
            _runtime_environment_fingerprint()
        )
        result.metadata.update(
            {
                "python_peak_traced_memory_bytes": -1,
                "memory_measurement_contract": (
                    "python_tracemalloc_separate_replay_peak_increment_v1"
                    if measure_python_memory
                    else "disabled"
                ),
                "memory_measurement_scope": (
                    "python_allocator_only_excludes_native_and_accelerator_memory"
                    if measure_python_memory
                    else "not_measured"
                ),
                "runtime_measurement_contract": (
                    "uninstrumented_wall_clock_inprocess_v1"
                ),
                "memory_replay_state_equivalence_gate": (
                    "PENDING" if measure_python_memory else "NOT_RUN"
                ),
                "memory_replay_order_contract": (
                    "all_case_timed_runs_before_case_memory_replays_v1"
                    if measure_python_memory
                    else "not_applicable"
                ),
                "evaluation_evidence_gate": evaluation_evidence_gate,
                "evaluation_evidence_contract": (
                    evaluation_evidence_contract
                ),
                "algorithm_configuration_sha256": (
                    predeclared_configuration.sha256
                ),
                "predeclared_algorithm_configuration": (
                    predeclared_configuration.payload
                ),
                "population": population,
                "search_evaluations": search_evaluations,
                "pilot_evaluations": pilot_evaluations,
                "confirm_evaluations": confirm_evaluations,
            }
        )
        archive_csv = archive_dir / f"{algorithm}_seed{seed}.csv"
        raw_results.append((algorithm, seed, result, runtime, archive_csv, evaluation_count(counted_instance)))

    for _, _, result, _, _, _ in raw_results:
        _verify_result_objectives(instance, result)

    # Memory instrumentation is intentionally performed only after every timed
    # arm in this matched case has completed.  This prevents deterministic
    # replay from warming caches or otherwise perturbing a later arm of the
    # same case.  The suite-level contract intentionally makes no stronger
    # claim about later cases.
    if measure_python_memory:
        for algorithm, seed, result, _, _, timed_evaluations in raw_results:
            replay_log = (
                output_dir
                / "neural_spectral_memory_replay"
                / f"{algorithm}_seed{seed}.jsonl"
            )
            replay_old_spectral_log = os.environ.get(
                "MO_NCO_NEURAL_SPECTRAL_LOG"
            )
            os.environ["MO_NCO_NEURAL_SPECTRAL_LOG"] = str(replay_log)
            tracemalloc.start()
            memory_baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            try:
                replay_counted_instance = CountingTSPInstance(
                    instance,
                    max_evaluations=iterations,
                )
                replay_trace_path = None
                if certified_traces and algorithm.lower() in {
                    "ips-theory-certified",
                    "ips-certified-mh",
                    "ips-typed-mh",
                }:
                    replay_trace_dir = (
                        output_dir / "kernel_traces_memory_replay"
                    )
                    replay_trace_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    replay_trace_path = (
                        replay_trace_dir
                        / f"{algorithm}_seed{seed}.jsonl"
                    )
                replay = apply_common_archive_limit(
                    run_algorithm(
                        algorithm=algorithm,
                        instance=replay_counted_instance,  # type: ignore[arg-type]
                        seed=seed,
                        population=population,
                        iterations=iterations,
                        log_period=log_period,
                        archive_update_period=archive_update_period,
                        certified_trace_path=replay_trace_path,
                        anytime_checkpoint_period=(
                            anytime_checkpoint_period
                        ),
                    )
                )
                _, memory_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
                if replay_old_spectral_log is None:
                    os.environ.pop("MO_NCO_NEURAL_SPECTRAL_LOG", None)
                else:
                    os.environ["MO_NCO_NEURAL_SPECTRAL_LOG"] = (
                        replay_old_spectral_log
                    )
            if (
                replay.particles != result.particles
                or replay.objectives != result.objectives
                or replay.archive.entries != result.archive.entries
                or evaluation_count(replay_counted_instance)
                != timed_evaluations
            ):
                raise RuntimeError(
                    "The separate memory replay did not reproduce the "
                    "timed algorithm state exactly."
                )
            result.metadata.update(
                {
                    "python_peak_traced_memory_bytes": max(
                        0,
                        int(memory_peak) - int(memory_baseline),
                    ),
                    "memory_replay_state_equivalence_gate": "PASS",
                }
            )

    for _, _, result, _, archive_csv, _ in raw_results:
        result.write_archive_csv(archive_csv)

    archive_fronts = [
        tuple(entry.objectives for entry in result.archive.entries)
        for _, _, result, _, _, _ in raw_results
    ]
    if metric_reference is None:
        reference = common_reference([result for _, _, result, _, _, _ in raw_results])
        empirical_front = empirical_reference_front(archive_fronts)
        all_points = [point for front in archive_fronts for point in front]
        metric_ideal, metric_nadir = ideal_nadir(all_points) if all_points else ((), ())
        metric_reference_contract = "batch_derived_v1"
    else:
        (
            reference,
            empirical_front,
            metric_ideal,
            metric_nadir,
            metric_reference_contract,
        ) = _validated_metric_reference(metric_reference)
    normalized_reference = (
        normalize_points(empirical_front, metric_ideal, metric_nadir)
        if empirical_front and metric_ideal and metric_nadir
        else ()
    )
    records: List[RunRecord] = []
    anytime_rows: List[Dict[str, object]] = []
    for idx, (algorithm, seed, result, runtime, archive_csv, evaluations) in enumerate(raw_results):
        final = result.diagnostics[-1] if result.diagnostics else None
        hv = result.archive.hypervolume_2d(reference=reference) if reference is not None else 0.0
        eval_auc, time_auc, curve_rows = calibrated_anytime_auc(
            algorithm=algorithm,
            seed=seed,
            result=result,
            final_hypervolume=hv,
            reference=reference,
            runtime_seconds=runtime,
            evaluations=evaluations,
            checkpoint_period=anytime_checkpoint_period,
        )
        anytime_rows.extend(curve_rows)
        normalized_archive = (
            normalize_points(archive_fronts[idx], metric_ideal, metric_nadir)
            if archive_fronts[idx] and metric_ideal and metric_nadir
            else ()
        )
        records.append(
            RunRecord(
                algorithm=algorithm,
                seed=seed,
                population=population,
                algorithm_configuration_sha256=str(
                    result.metadata[
                        "algorithm_configuration_sha256"
                    ]
                ),
                search_evaluations=int(
                    result.metadata["search_evaluations"]
                ),
                pilot_evaluations=int(
                    result.metadata["pilot_evaluations"]
                ),
                confirm_evaluations=int(
                    result.metadata["confirm_evaluations"]
                ),
                publication_certificate_packet_gate=(
                    str(
                        result.metadata[
                            "publication_certificate_packet_gate"
                        ]
                    )
                    if algorithm.lower()
                    == "pareto-smc-pilot-confirm-v12"
                    else "NOT_APPLICABLE"
                ),
                archive_size=len(result.archive),
                hypervolume_2d=hv,
                runtime_seconds=runtime,
                python_peak_traced_memory_bytes=int(
                    result.metadata["python_peak_traced_memory_bytes"]
                ),
                output_objective_equivalence_gate=str(
                    result.metadata["output_objective_equivalence_gate"]
                ),
                output_objective_max_abs_error=float(
                    result.metadata["output_objective_max_abs_error"]
                ),
                output_objective_equivalence_contract=str(
                    result.metadata[
                        "output_objective_equivalence_contract"
                    ]
                ),
                anytime_objective_equivalence_gate=str(
                    result.metadata["anytime_objective_equivalence_gate"]
                ),
                anytime_objective_equivalence_contract=str(
                    result.metadata[
                        "anytime_objective_equivalence_contract"
                    ]
                ),
                evaluation_evidence_gate=str(
                    result.metadata["evaluation_evidence_gate"]
                ),
                evaluation_evidence_contract=str(
                    result.metadata["evaluation_evidence_contract"]
                ),
                native_archive_completeness_gate=str(
                    result.metadata[
                        "native_archive_completeness_gate"
                    ]
                ),
                native_archive_completeness_contract=str(
                    result.metadata[
                        "native_archive_completeness_contract"
                    ]
                ),
                anytime_front_semantics=str(
                    result.metadata["anytime_front_semantics"]
                ),
                anytime_checkpoint_gate=str(
                    result.metadata["anytime_checkpoint_gate"]
                ),
                anytime_checkpoint_contract=str(
                    result.metadata["anytime_checkpoint_contract"]
                ),
                anytime_checkpoint_period=int(
                    result.metadata["anytime_checkpoint_period"]
                ),
                anytime_checkpoint_count=int(
                    result.metadata["anytime_checkpoint_count"]
                ),
                anytime_auc_integration_contract=str(
                    result.metadata[
                        "anytime_auc_integration_contract"
                    ]
                ),
                anytime_time_auc_status=str(
                    result.metadata["anytime_time_auc_status"]
                ),
                max_diagnostic_archive_size=int(
                    result.metadata["max_diagnostic_archive_size"]
                ),
                diagnostic_archive_limit_gate=str(
                    result.metadata["diagnostic_archive_limit_gate"]
                ),
                diagnostic_archive_limit_contract=str(
                    result.metadata[
                        "diagnostic_archive_limit_contract"
                    ]
                ),
                acceptance_rate=final.acceptance_rate if final else 0.0,
                empirical_energy=final.empirical_energy if final else 0.0,
                evaluations=evaluations,
                hypervolume_per_second=hv / max(runtime, 1e-12),
                hypervolume_per_evaluation=hv / max(evaluations, 1),
                anytime_hv_eval_auc=eval_auc,
                anytime_hv_time_auc=time_auc,
                anytime_hv_auc_per_second=time_auc / max(runtime, 1e-12),
                igd_plus=igd_plus(normalized_archive, normalized_reference),
                additive_epsilon=additive_epsilon(normalized_archive, normalized_reference),
                spacing=spacing(normalized_archive),
                rejection_rate=final.rejection_rate if final else 0.0,
                max_rejection_streak=final.max_rejection_streak if final else 0,
                current_rejection_streak=final.current_rejection_streak if final else 0,
                archive_csv=str(archive_csv),
            )
        )

    write_runs_csv(output_dir / "runs.csv", records)
    write_anytime_csv(output_dir / "anytime.csv", anytime_rows)
    summary = summarize(records)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_comparison_markdown(output_dir / "comparison.md", summary)
    write_pairwise_markdown(output_dir / "paired_comparison.md", records, summary)
    write_front_svg(output_dir / "pareto_fronts.svg", raw_results, title="Pareto fronts by algorithm")
    write_run_metadata_jsonl(
        output_dir / "run_metadata.jsonl",
        raw_results,
        execution_order_contract=execution_order,
        metric_reference_contract=metric_reference_contract,
        metric_reference_manifest_sha256=metric_reference_manifest_sha256,
        metric_hypervolume_reference=reference,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return records, summary


def write_run_metadata_jsonl(
    path: Path,
    raw_results: Sequence[Tuple[str, int, OptimizationResult, float, Path, int]],
    execution_order_contract: str = "algorithm-major",
    metric_reference_contract: str = "batch_derived_v1",
    metric_reference_manifest_sha256: str = "",
    metric_hypervolume_reference: Optional[ObjectiveVector] = None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for execution_order_index, (algorithm, seed, result, runtime, archive_csv, evaluations) in enumerate(raw_results):
            payload = {
                "algorithm": algorithm,
                "seed": seed,
                "execution_order_index": execution_order_index,
                "execution_order_contract": execution_order_contract,
                "runtime_seconds": runtime,
                "evaluations": evaluations,
                "archive_csv": str(archive_csv),
                "metric_reference_contract": metric_reference_contract,
                "metric_reference_manifest_sha256": metric_reference_manifest_sha256,
                "metric_hypervolume_reference": (
                    list(metric_hypervolume_reference)
                    if metric_hypervolume_reference is not None
                    else None
                ),
                "metadata": result.metadata,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def calibrated_anytime_auc(
    algorithm: str,
    seed: int,
    result: OptimizationResult,
    final_hypervolume: float,
    reference: Optional[ObjectiveVector],
    runtime_seconds: float,
    evaluations: int,
    checkpoint_period: Optional[int] = None,
) -> Tuple[float, float, List[Dict[str, object]]]:
    """Build a common-reference, left-continuous anytime curve.

    A formal curve is admitted only when every arm exposes a genuine archive
    snapshot at the same frozen evaluation checkpoints.  Extra algorithm-
    specific diagnostics are ignored.  Left-continuous step integration
    prevents sparse logging from inventing gradual improvements between two
    observations.
    """

    max_evals = max(1, evaluations)
    diagnostics = sorted(
        result.diagnostics,
        key=lambda item: (item.iteration, item.elapsed_seconds),
    )
    by_evaluation = {
        int(diagnostic.iteration): diagnostic
        for diagnostic in diagnostics
    }
    if checkpoint_period is not None:
        if (
            isinstance(checkpoint_period, bool)
            or checkpoint_period <= 0
            or checkpoint_period > max_evals
        ):
            raise ValueError(
                "checkpoint_period must be a positive integer no larger "
                "than the evaluation budget."
            )
        required_checkpoints = tuple(
            range(checkpoint_period, max_evals, checkpoint_period)
        ) + (max_evals,)
        missing = tuple(
            checkpoint
            for checkpoint in required_checkpoints
            if checkpoint not in by_evaluation
        )
        if missing:
            preview = ", ".join(str(value) for value in missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise RuntimeError(
                "Formal anytime AUC is unavailable because genuine archive "
                "snapshots are missing at common evaluation checkpoints: "
                f"{preview}{suffix}"
            )
        diagnostics = [
            by_evaluation[checkpoint]
            for checkpoint in required_checkpoints
        ]
        if reference is None:
            raise RuntimeError(
                "Formal anytime AUC requires a frozen common hypervolume "
                "reference."
            )
        for diagnostic in diagnostics:
            if not diagnostic.front:
                raise RuntimeError(
                    "Formal anytime AUC requires a genuine nonempty archive "
                    "front at every common evaluation checkpoint."
                )
            if diagnostic.archive_size != len(diagnostic.front):
                raise RuntimeError(
                    "Formal anytime checkpoint archive_size does not match "
                    "the attached front."
                )
            if any(
                len(point) != len(reference)
                or any(not math.isfinite(float(value)) for value in point)
                for point in diagnostic.front
            ):
                raise RuntimeError(
                    "Formal anytime checkpoint front has a non-finite or "
                    "wrong-dimensional objective vector."
                )
        elapsed_values = [
            float(diagnostic.elapsed_seconds)
            for diagnostic in diagnostics
        ]
        if (
            runtime_seconds <= 0.0
            or any(
                not math.isfinite(value)
                or value <= 0.0
                or value > runtime_seconds
                for value in elapsed_values
            )
            or any(
                right < left
                for left, right in zip(
                    elapsed_values,
                    elapsed_values[1:],
                )
            )
        ):
            raise RuntimeError(
                "Formal anytime snapshots require positive, finite, "
                "nondecreasing elapsed-time evidence."
            )
        result.metadata.update(
            {
                "anytime_checkpoint_gate": "PASS",
                "anytime_checkpoint_contract": (
                    "exact_common_evaluation_checkpoint_"
                    "archive_snapshot_v1"
                ),
                "anytime_checkpoint_period": checkpoint_period,
                "anytime_checkpoint_count": len(
                    required_checkpoints
                ),
            }
        )
    else:
        result.metadata.update(
            {
                "anytime_checkpoint_gate": "NOT_RUN",
                "anytime_checkpoint_contract": "not_predeclared",
                "anytime_checkpoint_period": 0,
                "anytime_checkpoint_count": len(diagnostics),
            }
        )
    final_diag_hv = max((diag.hypervolume_2d for diag in diagnostics), default=0.0)
    points: List[Tuple[int, float, float, float]] = [(0, 0.0, 0.0, 0.0)]
    for diag in diagnostics:
        evals = min(max_evals, max(0, int(diag.iteration)))
        frac = evals / max_evals
        elapsed = diag.elapsed_seconds if diag.elapsed_seconds > 0.0 else runtime_seconds * frac
        if checkpoint_period is not None:
            assert reference is not None
            calibrated_hv = hypervolume_2d_points(
                diag.front,
                reference,
            )
        elif diag.front and reference is not None:
            calibrated_hv = hypervolume_2d_points(diag.front, reference)
        elif final_diag_hv > 0.0:
            calibrated_hv = final_hypervolume * max(0.0, min(1.0, diag.hypervolume_2d / final_diag_hv))
        else:
            calibrated_hv = final_hypervolume * frac
        points.append((evals, elapsed, calibrated_hv, diag.hypervolume_2d))
    points.append((max_evals, runtime_seconds, final_hypervolume, final_diag_hv))

    deduped: List[Tuple[int, float, float, float]] = []
    for point in sorted(points, key=lambda item: (item[0], item[1])):
        if deduped and point[0] == deduped[-1][0]:
            deduped[-1] = point
        else:
            deduped.append(point)

    eval_auc = 0.0
    time_auc = 0.0
    for left, right in zip(deduped, deduped[1:]):
        eval_auc += (
            left[2]
            * max(0.0, right[0] - left[0])
            / max_evals
        )
        time_auc += (
            left[2]
            * max(0.0, right[1] - left[1])
            / max(runtime_seconds, 1e-12)
        )

    result.metadata.update(
        {
            "anytime_auc_integration_contract": (
                "left_continuous_step_on_evaluation_snapshots_v1"
            ),
            "anytime_time_auc_status": (
                "descriptive_only_not_formal_quality_gate_v1"
            ),
        }
    )

    rows = [
        {
            "algorithm": algorithm,
            "seed": seed,
            "evaluations": evals,
            "elapsed_seconds": elapsed,
            "common_reference_hypervolume_2d": calibrated_hv,
            "raw_diagnostic_hypervolume_2d": raw_hv,
        }
        for evals, elapsed, calibrated_hv, raw_hv in deduped
    ]
    return eval_auc, time_auc, rows


def hypervolume_2d_points(points: Sequence[ObjectiveVector], reference: ObjectiveVector) -> float:
    if not points:
        return 0.0
    sorted_points = sorted(points, key=lambda item: (item[0], item[1]))
    nondominated: List[ObjectiveVector] = []
    best_y = float("inf")
    for x, y in sorted_points:
        if y < best_y:
            nondominated.append((x, y))
            best_y = y
    ref_x, ref_y = reference
    hv = 0.0
    prev_y = ref_y
    for x, y in nondominated:
        hv += max(0.0, ref_x - x) * max(0.0, prev_y - y)
        prev_y = min(prev_y, y)
    return hv


def common_reference(results: Sequence[OptimizationResult]) -> Optional[ObjectiveVector]:
    points = [entry.objectives for result in results for entry in result.archive.entries]
    if not points or len(points[0]) != 2:
        return None
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    return (max_x + 0.1 * max(1e-9, max_x - min_x), max_y + 0.1 * max(1e-9, max_y - min_y))


def summarize(records: Sequence[RunRecord]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.algorithm, []).append(record)

    summary: Dict[str, Dict[str, float]] = {}
    for algorithm, group in grouped.items():
        summary[algorithm] = {}
        for metric in [
            "archive_size",
            "hypervolume_2d",
            "runtime_seconds",
            "python_peak_traced_memory_bytes",
            "acceptance_rate",
            "rejection_rate",
            "max_rejection_streak",
            "current_rejection_streak",
            "empirical_energy",
            "anytime_hv_eval_auc",
            "anytime_hv_time_auc",
            "anytime_hv_auc_per_second",
            "igd_plus",
            "additive_epsilon",
            "spacing",
        ]:
            values = [float(getattr(record, metric)) for record in group]
            summary[algorithm][f"{metric}_mean"] = mean(values)
            summary[algorithm][f"{metric}_std"] = std(values)
            summary[algorithm][f"{metric}_min"] = min(values)
            summary[algorithm][f"{metric}_max"] = max(values)
        for metric in ["evaluations", "hypervolume_per_second", "hypervolume_per_evaluation"]:
            values = [float(getattr(record, metric)) for record in group]
            summary[algorithm][f"{metric}_mean"] = mean(values)
            summary[algorithm][f"{metric}_std"] = std(values)
            summary[algorithm][f"{metric}_min"] = min(values)
            summary[algorithm][f"{metric}_max"] = max(values)
    return summary


def write_runs_csv(path: Path, records: Sequence[RunRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "algorithm",
                "seed",
                "population",
                "algorithm_configuration_sha256",
                "search_evaluations",
                "pilot_evaluations",
                "confirm_evaluations",
                "publication_certificate_packet_gate",
                "archive_size",
                "hypervolume_2d",
                "runtime_seconds",
                "python_peak_traced_memory_bytes",
                "output_objective_equivalence_gate",
                "output_objective_max_abs_error",
                "output_objective_equivalence_contract",
                "anytime_objective_equivalence_gate",
                "anytime_objective_equivalence_contract",
                "evaluation_evidence_gate",
                "evaluation_evidence_contract",
                "native_archive_completeness_gate",
                "native_archive_completeness_contract",
                "anytime_front_semantics",
                "anytime_checkpoint_gate",
                "anytime_checkpoint_contract",
                "anytime_checkpoint_period",
                "anytime_checkpoint_count",
                "anytime_auc_integration_contract",
                "anytime_time_auc_status",
                "max_diagnostic_archive_size",
                "diagnostic_archive_limit_gate",
                "diagnostic_archive_limit_contract",
                "acceptance_rate",
                "empirical_energy",
                "evaluations",
                "hypervolume_per_second",
                "hypervolume_per_evaluation",
                "anytime_hv_eval_auc",
                "anytime_hv_time_auc",
                "anytime_hv_auc_per_second",
                "igd_plus",
                "additive_epsilon",
                "spacing",
                "rejection_rate",
                "max_rejection_streak",
                "current_rejection_streak",
                "archive_csv",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.algorithm,
                    record.seed,
                    record.population,
                    record.algorithm_configuration_sha256,
                    record.search_evaluations,
                    record.pilot_evaluations,
                    record.confirm_evaluations,
                    record.publication_certificate_packet_gate,
                    record.archive_size,
                    record.hypervolume_2d,
                    record.runtime_seconds,
                    record.python_peak_traced_memory_bytes,
                    record.output_objective_equivalence_gate,
                    record.output_objective_max_abs_error,
                    record.output_objective_equivalence_contract,
                    record.anytime_objective_equivalence_gate,
                    record.anytime_objective_equivalence_contract,
                    record.evaluation_evidence_gate,
                    record.evaluation_evidence_contract,
                    record.native_archive_completeness_gate,
                    record.native_archive_completeness_contract,
                    record.anytime_front_semantics,
                    record.anytime_checkpoint_gate,
                    record.anytime_checkpoint_contract,
                    record.anytime_checkpoint_period,
                    record.anytime_checkpoint_count,
                    record.anytime_auc_integration_contract,
                    record.anytime_time_auc_status,
                    record.max_diagnostic_archive_size,
                    record.diagnostic_archive_limit_gate,
                    record.diagnostic_archive_limit_contract,
                    record.acceptance_rate,
                    record.empirical_energy,
                    record.evaluations,
                    record.hypervolume_per_second,
                    record.hypervolume_per_evaluation,
                    record.anytime_hv_eval_auc,
                    record.anytime_hv_time_auc,
                    record.anytime_hv_auc_per_second,
                    record.igd_plus,
                    record.additive_epsilon,
                    record.spacing,
                    record.rejection_rate,
                    record.max_rejection_streak,
                    record.current_rejection_streak,
                    record.archive_csv,
                ]
            )


def write_anytime_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "algorithm",
        "seed",
        "evaluations",
        "elapsed_seconds",
        "common_reference_hypervolume_2d",
        "raw_diagnostic_hypervolume_2d",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    metrics = sorted({metric for values in summary.values() for metric in values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", *metrics])
        for algorithm, values in sorted(summary.items()):
            writer.writerow([algorithm, *[values.get(metric, 0.0) for metric in metrics]])


def write_comparison_markdown(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    ordered = sorted(
        summary.items(),
        key=lambda item: item[1].get("hypervolume_2d_mean", 0.0),
        reverse=True,
    )
    lines = [
        "# Benchmark Comparison",
        "",
        "Higher hypervolume is better. Runtime is wall-clock seconds for the current Python implementation.",
        "Anytime AUC uses common-reference archive snapshots when available; external one-shot baselines only expose final snapshots.",
        "",
        "| rank | algorithm | HV mean ± std | AUC(eval) mean ± std | AUC/sec mean ± std | IGD+ mean ± std | eps mean ± std | HV/sec mean ± std | HV/eval mean ± std | evals mean ± std | runtime mean ± std |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (algorithm, values) in enumerate(ordered, start=1):
        lines.append(
            "| "
            f"{rank} | {algorithm} | "
            f"{_fmt_pm(values, 'hypervolume_2d')} | "
            f"{_fmt_pm(values, 'anytime_hv_eval_auc')} | "
            f"{_fmt_pm(values, 'anytime_hv_auc_per_second')} | "
            f"{_fmt_pm(values, 'igd_plus')} | "
            f"{_fmt_pm(values, 'additive_epsilon')} | "
            f"{_fmt_pm(values, 'hypervolume_per_second')} | "
            f"{_fmt_pm(values, 'hypervolume_per_evaluation')} | "
            f"{_fmt_pm(values, 'evaluations')} | "
            f"{_fmt_pm(values, 'runtime_seconds')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pairwise_markdown(
    path: Path,
    records: Sequence[RunRecord],
    summary: Dict[str, Dict[str, float]],
) -> None:
    ordered = sorted(
        summary,
        key=lambda algorithm: summary[algorithm].get("hypervolume_2d_mean", 0.0),
        reverse=True,
    )
    if len(ordered) < 2:
        path.write_text("# Paired Comparison\n\nNot enough algorithms to compare.\n", encoding="utf-8")
        return

    anchor = ordered[0]
    by_algorithm: Dict[str, Dict[int, RunRecord]] = {}
    for record in records:
        by_algorithm.setdefault(record.algorithm, {})[record.seed] = record

    lines = [
        "# Paired Comparison",
        "",
        f"Anchor algorithm: `{anchor}`. Deltas are anchor minus comparator over matched seeds.",
        "The p-values use an exact two-sided sign test after dropping ties.",
        "",
        "| comparator | matched seeds | ΔHV mean | HV wins-losses | HV sign p | ΔAUC(eval) mean | AUC wins-losses | AUC sign p | ΔHV/sec mean | HV/sec wins-losses | HV/sec sign p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    anchor_records = by_algorithm[anchor]
    for comparator in ordered[1:]:
        common_seeds = sorted(set(anchor_records).intersection(by_algorithm.get(comparator, {})))
        hv_deltas = [
            anchor_records[seed].hypervolume_2d - by_algorithm[comparator][seed].hypervolume_2d
            for seed in common_seeds
        ]
        hvs_deltas = [
            anchor_records[seed].hypervolume_per_second
            - by_algorithm[comparator][seed].hypervolume_per_second
            for seed in common_seeds
        ]
        auc_deltas = [
            anchor_records[seed].anytime_hv_eval_auc - by_algorithm[comparator][seed].anytime_hv_eval_auc
            for seed in common_seeds
        ]
        hv_wins, hv_losses, hv_p = paired_sign_summary(hv_deltas)
        hvs_wins, hvs_losses, hvs_p = paired_sign_summary(hvs_deltas)
        auc_wins, auc_losses, auc_p = paired_sign_summary(auc_deltas)
        lines.append(
            "| "
            f"{comparator} | {len(common_seeds)} | "
            f"{mean(hv_deltas):.6g} | {hv_wins}-{hv_losses} | {hv_p:.4g} | "
            f"{mean(auc_deltas):.6g} | {auc_wins}-{auc_losses} | {auc_p:.4g} | "
            f"{mean(hvs_deltas):.6g} | {hvs_wins}-{hvs_losses} | {hvs_p:.4g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_front_svg(
    path: Path,
    raw_results: Sequence[Tuple[str, int, OptimizationResult, float, Path, int]],
    title: str,
) -> None:
    points_by_algorithm: Dict[str, List[ObjectiveVector]] = {}
    for algorithm, _, result, _, _, _ in raw_results:
        for entry in result.archive.entries:
            if len(entry.objectives) == 2:
                points_by_algorithm.setdefault(algorithm, []).append(entry.objectives)

    all_points = [point for points in points_by_algorithm.values() for point in points]
    if not all_points:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"500\"></svg>", encoding="utf-8")
        return

    width, height = 900, 620
    left, right, top, bottom = 80, 220, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x, max_x = min(p[0] for p in all_points), max(p[0] for p in all_points)
    min_y, max_y = min(p[1] for p in all_points), max(p[1] for p in all_points)
    span_x = max(1e-9, max_x - min_x)
    span_y = max(1e-9, max_y - min_y)
    min_x -= 0.05 * span_x
    max_x += 0.05 * span_x
    min_y -= 0.05 * span_y
    max_y += 0.05 * span_y

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    elements = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        f"<text x=\"{left}\" y=\"28\" font-family=\"Arial\" font-size=\"20\" font-weight=\"700\">{html.escape(title)}</text>",
        f"<line x1=\"{left}\" y1=\"{top + plot_h}\" x2=\"{left + plot_w}\" y2=\"{top + plot_h}\" stroke=\"#111827\"/>",
        f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + plot_h}\" stroke=\"#111827\"/>",
        f"<text x=\"{left + plot_w / 2}\" y=\"{height - 28}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">objective 1</text>",
        f"<text x=\"24\" y=\"{top + plot_h / 2}\" transform=\"rotate(-90 24 {top + plot_h / 2})\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">objective 2</text>",
    ]

    for tick in range(6):
        tx = min_x + (max_x - min_x) * tick / 5
        ty = min_y + (max_y - min_y) * tick / 5
        px = sx(tx)
        py = sy(ty)
        elements.append(f"<line x1=\"{px:.2f}\" y1=\"{top + plot_h}\" x2=\"{px:.2f}\" y2=\"{top + plot_h + 5}\" stroke=\"#111827\"/>")
        elements.append(f"<text x=\"{px:.2f}\" y=\"{top + plot_h + 22}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"11\">{tx:.2f}</text>")
        elements.append(f"<line x1=\"{left - 5}\" y1=\"{py:.2f}\" x2=\"{left}\" y2=\"{py:.2f}\" stroke=\"#111827\"/>")
        elements.append(f"<text x=\"{left - 8}\" y=\"{py + 4:.2f}\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"11\">{ty:.2f}</text>")

    for idx, (algorithm, points) in enumerate(sorted(points_by_algorithm.items())):
        color = colors[idx % len(colors)]
        for x, y in points:
            elements.append(f"<circle cx=\"{sx(x):.2f}\" cy=\"{sy(y):.2f}\" r=\"3.5\" fill=\"{color}\" fill-opacity=\"0.78\"/>")
        legend_y = top + 24 + idx * 24
        legend_x = left + plot_w + 35
        elements.append(f"<circle cx=\"{legend_x}\" cy=\"{legend_y}\" r=\"5\" fill=\"{color}\"/>")
        elements.append(f"<text x=\"{legend_x + 14}\" y=\"{legend_y + 4}\" font-family=\"Arial\" font-size=\"13\">{html.escape(algorithm)}</text>")

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def paired_sign_summary(deltas: Sequence[float], tol: float = 1e-12) -> Tuple[int, int, float]:
    wins = sum(1 for value in deltas if value > tol)
    losses = sum(1 for value in deltas if value < -tol)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    tail = min(wins, losses)
    # Exact two-sided sign-test p-value under Binomial(n, 1/2).  The
    # log-domain form avoids converting enormous integers to floats when
    # formal suites aggregate hundreds or thousands of matched pairs.
    log_half = math.log(0.5)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + n * log_half
        for k in range(tail + 1)
    ]
    max_log = max(log_terms)
    p_value = 2.0 * math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return wins, losses, min(1.0, p_value)


def _fmt_pm(values: Dict[str, float], metric: str) -> str:
    return f"{values.get(metric + '_mean', 0.0):.4g} ± {values.get(metric + '_std', 0.0):.3g}"
