"""Theory-guided population optimizer for multi-objective CO problems."""

__version__ = "0.21.3.14"

from .archive import ArchiveEntry, ParetoArchive
from .contracts import ClaimLevel, EvidenceLevel
from .instance import MultiObjectiveTSPInstance
from .ips_certified import CertifiedSingleSiteIPSOptimizer
from .ips_efficient import EfficientIPSOptimizer, TheoryAlignedIPSOptimizer
from .kernel_trace import TraceVerificationResult, verify_certified_trace
from .neural_potential import NeuralScalarPotential
from .pareto_bounds import certify_pareto_bounds
from .pareto_smc import AnnealedParetoSMCOptimizer, ObjectiveBoundsViolation
from .pareto_ijoc_allocation import (
    Exp3Snapshot,
    Exp3TypeAllocator,
    SearchRewardWeights,
    derive_domain_separated_seed,
    normalized_hypervolume_gain,
)
from .pareto_ijoc_generic_search import GenericTypedArchiveSearch
from .pareto_ijoc_generic_smc import GenericAnnealedParetoSMCOptimizer
from .pareto_ijoc_preflight import (
    IJOCPreflightResult,
    audit_ijoc_competitive_study,
)
from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    problem_sha256,
)
from .pareto_ijoc_spec import (
    IJOCParetoSMCSpecification,
    load_ijoc_pareto_smc_specification,
)
from .pareto_archive_cap_certificate import (
    ArchiveCapMetricCertificate,
    canonical_gonzalez_cap,
    certify_archive_cap,
)
from .pareto_dyadic_objective import (
    EXACT_EDGE_SUM_CONTRACT,
    DyadicObjectiveEncoding,
)
from .pareto_frozen_cells import (
    FrozenCellManifest,
    load_frozen_cell_manifest,
)
from .pareto_independent_replica_certificate import (
    build_false_pass_certificate,
    certify_pilot_power,
    clopper_pearson_lower_bracket,
    mutually_exclusive_cell_occupancy_lower_bound,
    plan_replica_count,
)
from .pareto_independent_replica_runner import (
    IndependentReplicaBatchResult,
    ReplicaTypeConfiguration,
    replica_configuration_sha256,
    replica_stream_plan_sha256,
    run_independent_replica_batch,
)
from .pareto_kernel_perturbation import (
    KernelPerturbationBound,
    RationalInterval,
    certify_kernel_perturbation_bound,
    decide_strict_less,
)
from .pareto_reference_fidelity import (
    ReferenceFidelityCertificate,
    certify_reference_fidelity_composition,
)
from .pareto_v15_publication_gate import (
    V15PublicationGate,
    evaluate_v15_publication_gate,
)
from .pareto_v15_context import (
    V15CertificateContext,
    V15CertificateContextError,
    verify_v15_context_sha256,
)
from .pareto_adaptive_type_cell import (
    AdaptiveIdentificationCertificate,
    ConfirmAllocationCertificate,
    TransportLowerBound,
    certify_balanced_successive_elimination_upper_bound,
    exact_confirm_risk_allocation,
    pairwise_transportation_lower_bound,
)
from .pareto_adaptive_replica_experiment import (
    AdaptivePilotResult,
    run_cell_separated_successive_elimination,
)
from .pareto_finite_menu_generalization import (
    FiniteMenuGeneralizationCertificate,
    certify_finite_menu_generalization,
)
from .pareto_intrinsic_dimension import (
    IntrinsicDimensionCertificate,
    canonical_maximal_tau_net,
    certify_ordered_bilipschitz_reference_family,
)
from .pareto_v16_artifact_bundle import (
    V16ComposedArtifactCertificate,
    verify_v16_composed_bundle,
    write_canonical_v16_bundle,
)
from .pareto_shared_categorical_design import (
    RationalTransportationLowerBound,
    SharedCategoricalIdentificationCertificate,
    SharedConfirmAllocationCertificate,
    certify_shared_categorical_identification_upper_bound,
    exact_shared_confirm_allocation,
    rational_pairwise_transportation_lower_bound,
)
from .pareto_shared_categorical_experiment import (
    SharedCategoricalPilotResult,
    pilot_lower_bound_matrix,
    plan_shared_confirm_from_pilot,
    run_shared_categorical_successive_elimination,
)
from .pareto_v16_theory_packet import (
    V16TheoryParameterCertificate,
    verify_v16_theory_packet,
    write_canonical_v16_theory_packet,
)
from .pareto_v16_theory_gate import V16TheoryGate, evaluate_v16_theory_gate
from .pareto_execution_contract import (
    DOMAIN_SEPARATED_SEED_SCHEMA_V1,
    FULL_TYPE_SWEEP_CHECKPOINT_SCHEMA_V1,
    PARETO_SMC_V13_ALGORITHM_ROLE,
    PARETO_SMC_V13_ALGORITHM_VERSION,
    DomainSeparatedSeed,
    FullTypeSweepCheckpointVerification,
    derive_domain_separated_seed,
    verify_domain_separated_seed,
    verify_full_type_sweep_checkpoints,
)
from .pareto_cell_certification import (
    CertifiedCellType,
    CellCertifiedParetoSampler,
    CellTypePlan,
    plan_cell_type,
)
from .pareto_cell_spec import (
    ParetoCellCertificationSpecification,
    load_pareto_cell_certification_specification,
)
from .pareto_fk_certificate import (
    BootstrapFeynmanKacPlan,
    ContractionAwareFeynmanKacPlan,
    bootstrap_fk_stability_constants,
    make_bootstrap_fk_plan,
    make_contraction_aware_fk_plan,
    recommend_mutation_steps_for_stage_contraction,
)
from .pareto_fixed_schedule_certificate import (
    FixedScheduleCertificateError,
    build_regeneration_pilot_plan_commitment_from_spec,
    certify_fixed_schedule_reference_metrics,
    certify_fixed_schedule_reference_metrics_from_spec,
)
from .pareto_fixed_reference_spec import (
    FixedReferenceCertificateSpecification,
    load_fixed_reference_certificate_specification,
)
from .pareto_fixed_schedule_experiment import (
    FixedScheduleExecutionPlan,
    FixedSchedulePilotConfirmResult,
    prepare_fixed_schedule_execution,
    run_fixed_schedule_pilot_confirm,
    run_fixed_schedule_stream,
)
from .pareto_preconfirm_receipt import (
    PreconfirmReceiptBindings,
    PreconfirmReceiptError,
    PreconfirmReceiptVerificationError,
    VerifiedPreconfirmReceipt,
    create_unsigned_preconfirm_receipt_request,
    sign_preconfirm_receipt_request,
    verify_preconfirm_receipt,
)
from .pareto_smc_spec import ParetoSMCSpecification, load_pareto_smc_specification
from .pareto_regeneration_certificate import (
    AssignmentPilotNonemptinessPreflight,
    ConfirmCellCertificate,
    EqualDualStreamSchedule,
    HeterogeneousPilotConfirmBudget,
    HoeffdingSuiteRequirement,
    JointCertificateDesign,
    PilotMassCertificate,
    PilotObservationRequirement,
    RefreshRequirement,
    assignment_pilot_nonemptiness_preflight,
    confirm_cell_certificate,
    deterministic_target_mass_lower_bound,
    enumerate_equal_dual_stream_schedules,
    evaluate_joint_certificate_design,
    finite_suite_hoeffding_half_width,
    heterogeneous_pilot_confirm_budget,
    minimum_independent_units_for_hoeffding_half_width,
    minimum_pilot_empirical_mass_for_target_bound,
    minimum_refresh_for_assigned_cells,
    pareto_minimal_designs,
    pilot_target_mass_lower_bound,
    regeneration_exposure,
    subset_normalizer_lower_bound,
    target_normalizer_lower_bound,
    terminal_residual_weight,
)
from .pareto_sparse_reference import (
    ManyObjectiveCapacityLowerBound,
    SparseMetricBounds,
    SparseReferenceCover,
    SpernerLowerBound,
    doubling_cover_cardinality_bound,
    greedy_maximal_reference_net,
    sparse_reference_metric_bounds,
    sperner_capacity_lower_bound,
    sperner_many_objective_lower_bound,
)
from .pareto_sparse_compression_certificate import (
    SparseCompressionCertificateError,
    SparseFiniteReferenceCompressionCertificate,
    certify_sparse_finite_reference_compression,
)
from .pareto_v13_publication import (
    V13PilotConfirmResult,
    V13PilotFreezeResult,
    V13PublicationProtocolError,
    V13RefreshCostCertificate,
    load_v13_pilot_artifact,
    run_v13_confirm_from_signed_receipt,
    run_v13_pilot_freeze,
    write_v13_pilot_artifact,
)
from .pareto_v13_spec import (
    V13ProtocolSpecification,
    load_v13_protocol_specification,
)
from .potential import HypervolumeArchivePotential, PotentialContext, ScalarArchivePotential
from .sampler import IPSMetropolisOptimizer, OptimizationResult
from .tsplib import TSPLIBProblem, load_bitsp, load_multiobjective_tsplib, parse_tsplib

__all__ = [
    "ArchiveEntry",
    "ParetoArchive",
    "ClaimLevel",
    "EvidenceLevel",
    "MultiObjectiveTSPInstance",
    "CertifiedSingleSiteIPSOptimizer",
    "EfficientIPSOptimizer",
    "TheoryAlignedIPSOptimizer",
    "TraceVerificationResult",
    "verify_certified_trace",
    "NeuralScalarPotential",
    "AnnealedParetoSMCOptimizer",
    "ObjectiveBoundsViolation",
    "ArchiveCapMetricCertificate",
    "canonical_gonzalez_cap",
    "certify_archive_cap",
    "EXACT_EDGE_SUM_CONTRACT",
    "DyadicObjectiveEncoding",
    "FrozenCellManifest",
    "load_frozen_cell_manifest",
    "build_false_pass_certificate",
    "certify_pilot_power",
    "clopper_pearson_lower_bracket",
    "mutually_exclusive_cell_occupancy_lower_bound",
    "plan_replica_count",
    "IndependentReplicaBatchResult",
    "ReplicaTypeConfiguration",
    "replica_configuration_sha256",
    "replica_stream_plan_sha256",
    "run_independent_replica_batch",
    "KernelPerturbationBound",
    "RationalInterval",
    "certify_kernel_perturbation_bound",
    "decide_strict_less",
    "ReferenceFidelityCertificate",
    "certify_reference_fidelity_composition",
    "V15PublicationGate",
    "evaluate_v15_publication_gate",
    "V15CertificateContext",
    "V15CertificateContextError",
    "verify_v15_context_sha256",
    "AdaptiveIdentificationCertificate",
    "ConfirmAllocationCertificate",
    "TransportLowerBound",
    "certify_balanced_successive_elimination_upper_bound",
    "exact_confirm_risk_allocation",
    "pairwise_transportation_lower_bound",
    "AdaptivePilotResult",
    "run_cell_separated_successive_elimination",
    "FiniteMenuGeneralizationCertificate",
    "certify_finite_menu_generalization",
    "IntrinsicDimensionCertificate",
    "canonical_maximal_tau_net",
    "certify_ordered_bilipschitz_reference_family",
    "V16ComposedArtifactCertificate",
    "verify_v16_composed_bundle",
    "write_canonical_v16_bundle",
    "RationalTransportationLowerBound",
    "SharedCategoricalIdentificationCertificate",
    "SharedConfirmAllocationCertificate",
    "certify_shared_categorical_identification_upper_bound",
    "exact_shared_confirm_allocation",
    "rational_pairwise_transportation_lower_bound",
    "SharedCategoricalPilotResult",
    "pilot_lower_bound_matrix",
    "plan_shared_confirm_from_pilot",
    "run_shared_categorical_successive_elimination",
    "V16TheoryParameterCertificate",
    "verify_v16_theory_packet",
    "write_canonical_v16_theory_packet",
    "V16TheoryGate",
    "evaluate_v16_theory_gate",
    "DOMAIN_SEPARATED_SEED_SCHEMA_V1",
    "FULL_TYPE_SWEEP_CHECKPOINT_SCHEMA_V1",
    "PARETO_SMC_V13_ALGORITHM_ROLE",
    "PARETO_SMC_V13_ALGORITHM_VERSION",
    "DomainSeparatedSeed",
    "FullTypeSweepCheckpointVerification",
    "derive_domain_separated_seed",
    "verify_domain_separated_seed",
    "verify_full_type_sweep_checkpoints",
    "CertifiedCellType",
    "CellCertifiedParetoSampler",
    "CellTypePlan",
    "plan_cell_type",
    "ParetoCellCertificationSpecification",
    "load_pareto_cell_certification_specification",
    "BootstrapFeynmanKacPlan",
    "ContractionAwareFeynmanKacPlan",
    "bootstrap_fk_stability_constants",
    "make_bootstrap_fk_plan",
    "make_contraction_aware_fk_plan",
    "recommend_mutation_steps_for_stage_contraction",
    "FixedScheduleCertificateError",
    "build_regeneration_pilot_plan_commitment_from_spec",
    "certify_fixed_schedule_reference_metrics",
    "certify_fixed_schedule_reference_metrics_from_spec",
    "FixedReferenceCertificateSpecification",
    "load_fixed_reference_certificate_specification",
    "FixedScheduleExecutionPlan",
    "FixedSchedulePilotConfirmResult",
    "prepare_fixed_schedule_execution",
    "run_fixed_schedule_pilot_confirm",
    "run_fixed_schedule_stream",
    "PreconfirmReceiptBindings",
    "PreconfirmReceiptError",
    "PreconfirmReceiptVerificationError",
    "VerifiedPreconfirmReceipt",
    "create_unsigned_preconfirm_receipt_request",
    "sign_preconfirm_receipt_request",
    "verify_preconfirm_receipt",
    "ParetoSMCSpecification",
    "load_pareto_smc_specification",
    "AssignmentPilotNonemptinessPreflight",
    "ConfirmCellCertificate",
    "EqualDualStreamSchedule",
    "HeterogeneousPilotConfirmBudget",
    "HoeffdingSuiteRequirement",
    "JointCertificateDesign",
    "PilotMassCertificate",
    "PilotObservationRequirement",
    "RefreshRequirement",
    "assignment_pilot_nonemptiness_preflight",
    "confirm_cell_certificate",
    "deterministic_target_mass_lower_bound",
    "enumerate_equal_dual_stream_schedules",
    "evaluate_joint_certificate_design",
    "finite_suite_hoeffding_half_width",
    "heterogeneous_pilot_confirm_budget",
    "minimum_independent_units_for_hoeffding_half_width",
    "minimum_pilot_empirical_mass_for_target_bound",
    "minimum_refresh_for_assigned_cells",
    "pareto_minimal_designs",
    "pilot_target_mass_lower_bound",
    "regeneration_exposure",
    "subset_normalizer_lower_bound",
    "target_normalizer_lower_bound",
    "terminal_residual_weight",
    "ManyObjectiveCapacityLowerBound",
    "SparseMetricBounds",
    "SparseReferenceCover",
    "SpernerLowerBound",
    "doubling_cover_cardinality_bound",
    "greedy_maximal_reference_net",
    "sparse_reference_metric_bounds",
    "sperner_capacity_lower_bound",
    "sperner_many_objective_lower_bound",
    "SparseCompressionCertificateError",
    "SparseFiniteReferenceCompressionCertificate",
    "certify_sparse_finite_reference_compression",
    "V13PilotConfirmResult",
    "V13PilotFreezeResult",
    "V13PublicationProtocolError",
    "V13RefreshCostCertificate",
    "load_v13_pilot_artifact",
    "run_v13_confirm_from_signed_receipt",
    "run_v13_pilot_freeze",
    "write_v13_pilot_artifact",
    "V13ProtocolSpecification",
    "load_v13_protocol_specification",
    "certify_pareto_bounds",
    "HypervolumeArchivePotential",
    "PotentialContext",
    "ScalarArchivePotential",
    "IPSMetropolisOptimizer",
    "OptimizationResult",
    "TSPLIBProblem",
    "load_bitsp",
    "load_multiobjective_tsplib",
    "parse_tsplib",
    "Exp3Snapshot",
    "Exp3TypeAllocator",
    "SearchRewardWeights",
    "derive_domain_separated_seed",
    "normalized_hypervolume_gain",
    "GenericTypedArchiveSearch",
    "GenericAnnealedParetoSMCOptimizer",
    "IJOCPreflightResult",
    "audit_ijoc_competitive_study",
    "MultiObjectiveCombinatorialProblem",
    "MultiObjectiveKnapsackInstance",
    "MultiObjectiveTSPProblemAdapter",
    "problem_sha256",
    "IJOCParetoSMCSpecification",
    "load_ijoc_pareto_smc_specification",
]

# Pareto-SMC v17 canonical theorem packet
from .pareto_v17_regeneration import (
    MinorizationBlock,
    RegenerationTransfer,
    TypeRegenerationCertificate,
)
from .pareto_v17_multitype_confirm import (
    MultiTypeConfirmProblem,
    ConfirmPlanCertificate,
    exact_minimum_cost_allocation,
)
from .pareto_v17_track_and_stop import (
    TrackAndStopConfig,
    TrackAndStopResult,
    run_track_and_stop,
    solve_characteristic_game,
)
from .pareto_v17_canonical_packet import (
    CanonicalV17Result,
    build_canonical_v17_packet,
)
from .pareto_v17_final_regeneration_runtime import (
    RuntimeBlock,
    FinalRegenerationRuntimeResult,
    run_final_regeneration_block,
)

# Pareto-SMC v18 source-derived and nonregular certificate packet
from .pareto_v18_minorization import (
    DerivedTypeMinorization,
    IdealFinalKernelContract,
    IndependenceMHMinorizationSpec,
    rational_exp_neg_lower,
)
from .pareto_v18_occupancy import (
    MultiTypeOccupancyProblem,
    OccupancyPlanCertificate,
    exact_all_cells_hit_lower,
    exact_minimum_cost_occupancy_allocation,
)
from .pareto_v18_nonregular import (
    EpsilonPACSelectionCertificate,
    epsilon_pac_selection,
)
from .pareto_v18_practicality import (
    PracticalityCertificate,
    build_practicality_certificate,
)
from .pareto_v18_reference_completeness import (
    ExactTSPReferenceCompletenessCertificate,
    certify_exact_tsp_reference_completeness,
)
from .pareto_v18_study_commitment import (
    StudyExecutionAudit,
    StudyRow,
    verify_study_execution,
)
from .pareto_v18_canonical_packet import (
    CanonicalV18Result,
    build_canonical_v18_packet,
)
from .pareto_v18_minorization import (
    Binary64AugmentedTchebycheffPotentialSpec,
    binary64_augmented_tchebycheff_span_upper,
)
from .pareto_v18_practicality import (
    binomial_upper_tail,
    minimum_endpoints_for_subset_hits,
    subset_coverage_endpoint_lower,
)
from .pareto_v18_reference_branch_bound import (
    BranchAndBoundReferenceCertificate,
    certify_reference_cover_branch_and_bound,
)
from .pareto_v18_formalization_gate import (
    FormalizationGateResult,
    run_formalization_gate,
)

# V21e3r1 V9 development-only information-time successor helpers
from .pareto_v21e3r1_v9_theory import (
    ArchiveCompensatedReplacementDecision,
    CandidateScreenDecision,
    DualResourceBudget,
    InformationTimePath,
    OperatorProductivity,
    archive_compensated_replacement,
    composite_potential,
    information_time_equivalent,
    information_time_path,
    operator_productivity,
    select_first_unseen,
)

# Keep the public V9 convenience imports without pre-importing executable
# modules.  Eager imports here caused ``python -m <module>`` to find its target
# already present in ``sys.modules`` and emit a runpy RuntimeWarning.  PEP 562
# lazy attributes preserve ``from mo_nco import <name>`` compatibility while
# letting gate/diagnostic modules execute from a clean module state.
_V9_LAZY_EXPORTS = {
    "analyze_v9_trace_database": (
        "pareto_v21e3r1_v9_diagnostics",
        "analyze_v9_trace_database",
    ),
    "V9PredevelopmentProtocolError": (
        "pareto_v21e3r1_v9_protocol",
        "V9PredevelopmentProtocolError",
    ),
    "load_v9_predevelopment_protocol": (
        "pareto_v21e3r1_v9_protocol",
        "load_v9_predevelopment_protocol",
    ),
    "validate_v9_predevelopment_protocol": (
        "pareto_v21e3r1_v9_protocol",
        "validate_v9_predevelopment_protocol",
    ),
    "validate_v9_resource_caps": (
        "pareto_v21e3r1_v9_protocol",
        "validate_v9_resource_caps",
    ),
    "evaluate_v9_predevelopment_readiness": (
        "pareto_v21e3r1_v9_gate",
        "evaluate_v9_predevelopment_readiness",
    ),
    "write_v9_predevelopment_readiness_receipt": (
        "pareto_v21e3r1_v9_gate",
        "write_v9_predevelopment_readiness_receipt",
    ),
}


def __getattr__(name: str):
    target = _V9_LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute_name = target
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_V9_LAZY_EXPORTS))
