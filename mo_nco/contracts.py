from __future__ import annotations

"""Static claim levels for algorithm and certificate reporting.

The values are deliberately independent of CLI aliases. A fast optimizer
cannot acquire a stronger theoretical identity merely because it is registered
under a name containing ``theory`` or ``certified``.
"""

from enum import Enum


class ClaimLevel(str, Enum):
    """Strongest claim a component is allowed to emit."""

    CERTIFIED_MH = "certified_mh"
    # The implementation replays the typed Feynman--Kac mechanics but does
    # not claim a finite-particle coverage theorem without external constants.
    PARETO_SMC_MECHANICAL = "pareto_smc_mechanical"
    # Deterministic bootstrap-resampling branch with an explicit source-level
    # cellwise FK MSE/radius constant.  A Pareto coverage claim still requires
    # an independently certified positive cell-mass floor.
    PARETO_SMC_BOOTSTRAP_BOUND = "pareto_smc_bootstrap_bound"
    # Two independent deterministic-bootstrap Pareto-SMC streams certify
    # coverage and practical metrics for a reference front frozen before both
    # streams. This is not a complete-unknown-Pareto-front claim.
    PARETO_SMC_FIXED_REFERENCE_BOUND = (
        "pareto_smc_fixed_reference_bound"
    )
    # Direct terminal-regeneration pilot-confirm certificate. This level uses
    # target minorization and a no-hit bound instead of the general published
    # Feynman--Kac empirical-measure radius.
    PARETO_SMC_REGENERATION_REFERENCE_BOUND = (
        "pareto_smc_regeneration_reference_bound"
    )
    # Runtime-validated source-bound cell contract. External proof artifacts
    # are hash-bound but their mathematical truth is not established by runtime.
    PARETO_CELL_SOURCE_BOUND = "pareto_cell_source_bound"
    # Reserved for an independent exact/mechanized audit that verifies those
    # source-bound proof artifacts as well as the runtime contract.
    PARETO_CELL_CERTIFIED = "pareto_cell_certified"
    # Deprecated compatibility value. Replay is evidence about an
    # implementation record, not an algorithmic claim.
    SOURCE_REPLAYED = "source_replayed"
    HEURISTIC_DESCENT = "heuristic_descent"
    BASELINE = "baseline"


class EvidenceLevel(str, Enum):
    """Strength of the artifact evidence used to audit a runtime claim."""

    SELF_REPORTED_METADATA = "self_reported_metadata"
    INTERNAL_TRACE_REPLAY = "internal_trace_replay"
    SOURCE_REPLAYED = "source_replayed"
    NO_SUITE_WIDE_LEVEL_PASSED = "no_suite_wide_level_passed"
