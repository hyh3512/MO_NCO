from __future__ import annotations

"""Fail-closed specification for the IJOC-oriented Pareto-SMC branch."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .pareto_ijoc_allocation import SearchRewardWeights
from .pareto_smc_spec import ParetoSMCSpecification, load_pareto_smc_specification


IJOC_SPEC_SCHEMA = "ijoc_typed_pareto_smc_spec_v2"


@dataclass(frozen=True)
class IJOCParetoSMCSpecification:
    path: Path
    sha256: str
    base_smc_specification: ParetoSMCSpecification
    adaptive_search_evaluations: int
    allocation_policy: str
    minimum_pulls_per_type: int
    exp3_exploration: float | None
    reward_weights: SearchRewardWeights
    deployment_archive_max_size: int | None
    competitive_archive_contract: str


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def load_ijoc_pareto_smc_specification(
    path: str | Path,
    *,
    objective_dimension: int,
    total_evaluations: int | None = None,
) -> IJOCParetoSMCSpecification:
    if objective_dimension != 2:
        raise ValueError(
            f"{IJOC_SPEC_SCHEMA} is restricted to biobjective "
            "studies because its adaptive reward uses exact 2D hypervolume."
        )
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"IJOC Pareto-SMC specification is missing: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"IJOC Pareto-SMC specification is not strict UTF-8 JSON: {error}"
        ) from error
    root = _mapping(payload, "IJOC Pareto-SMC specification")
    _exact_keys(root, {"schema", "base_smc", "adaptive_search", "output"}, "root")
    if root.get("schema") != IJOC_SPEC_SCHEMA:
        raise ValueError(f"schema must be {IJOC_SPEC_SCHEMA!r}.")

    base = _mapping(root.get("base_smc"), "base_smc")
    _exact_keys(base, {"path", "sha256"}, "base_smc")
    raw_base_path = base.get("path")
    raw_base_sha = base.get("sha256")
    if not isinstance(raw_base_path, str) or not raw_base_path:
        raise ValueError("base_smc.path must be a nonempty string.")
    if not isinstance(raw_base_sha, str) or len(raw_base_sha) != 64:
        raise ValueError("base_smc.sha256 must be a 64-character hex digest.")
    try:
        int(raw_base_sha, 16)
    except ValueError as error:
        raise ValueError("base_smc.sha256 must be hexadecimal.") from error
    candidate_base_path = Path(raw_base_path)
    if candidate_base_path.is_absolute():
        raise ValueError("base_smc.path must be relative to the IJOC specification.")
    specification_root = resolved.parent.resolve()
    base_path = (specification_root / candidate_base_path).resolve()
    try:
        base_path.relative_to(specification_root)
    except ValueError as error:
        raise ValueError(
            "base_smc.path escapes the IJOC specification directory."
        ) from error
    if not base_path.is_file():
        raise ValueError(f"The bound base SMC specification is missing: {base_path}")
    actual_base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if actual_base_sha != raw_base_sha:
        raise ValueError("The base SMC specification SHA-256 does not match.")
    base_specification = load_pareto_smc_specification(
        base_path,
        objective_dimension=objective_dimension,
    )

    adaptive = _mapping(root.get("adaptive_search"), "adaptive_search")
    _exact_keys(
        adaptive,
        {
            "evaluations",
            "allocation_policy",
            "minimum_pulls_per_type",
            "exp3_exploration",
            "reward_weights",
        },
        "adaptive_search",
    )
    raw_evaluations = adaptive.get("evaluations")
    if (
        isinstance(raw_evaluations, bool)
        or not isinstance(raw_evaluations, int)
        or raw_evaluations < 0
    ):
        raise ValueError("adaptive_search.evaluations must be a nonnegative integer.")
    if total_evaluations is not None and raw_evaluations >= total_evaluations:
        raise ValueError(
            "adaptive_search.evaluations must leave a positive SMC-core budget."
        )
    policy = adaptive.get("allocation_policy")
    if policy not in {"exp3", "uniform"}:
        raise ValueError("adaptive_search.allocation_policy must be 'exp3' or 'uniform'.")
    raw_minimum_pulls = adaptive.get("minimum_pulls_per_type")
    if (
        isinstance(raw_minimum_pulls, bool)
        or not isinstance(raw_minimum_pulls, int)
        or raw_minimum_pulls < 0
    ):
        raise ValueError(
            "adaptive_search.minimum_pulls_per_type must be a nonnegative integer."
        )
    if policy == "uniform" and raw_minimum_pulls != 0:
        raise ValueError(
            "The uniform allocation policy must set minimum_pulls_per_type to zero."
        )
    uniform_prefix = (
        len(base_specification.reference_directions) * raw_minimum_pulls
    )
    if uniform_prefix > raw_evaluations:
        raise ValueError(
            "minimum_pulls_per_type requires more uniform-prefix evaluations "
            "than the adaptive-search budget provides."
        )
    raw_exploration = adaptive.get("exp3_exploration")
    if raw_exploration is None:
        exploration = None
    else:
        if isinstance(raw_exploration, bool) or not isinstance(raw_exploration, (int, float)):
            raise ValueError("adaptive_search.exp3_exploration must be null or numeric.")
        exploration = float(raw_exploration)
        if not math.isfinite(exploration) or not 0.0 < exploration <= 1.0:
            raise ValueError("adaptive_search.exp3_exploration must lie in (0, 1].")
    if policy == "uniform" and exploration is not None:
        raise ValueError("The uniform allocation policy must set exp3_exploration to null.")

    reward = _mapping(adaptive.get("reward_weights"), "adaptive_search.reward_weights")
    _exact_keys(reward, {"hypervolume", "new_cell", "scalar_improvement"}, "reward_weights")
    raw_reward_values = []
    for key in ("hypervolume", "new_cell", "scalar_improvement"):
        value = reward.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reward_weights.{key} must be numeric.")
        raw_reward_values.append(float(value))
    reward_weights = SearchRewardWeights(*raw_reward_values)

    output = _mapping(root.get("output"), "output")
    _exact_keys(
        output,
        {"competitive_archive", "deployment_archive_max_size"},
        "output",
    )
    competitive_archive = output.get("competitive_archive")
    if competitive_archive != "unbounded_all_evaluated_nondominated":
        raise ValueError(
            "output.competitive_archive must be "
            "'unbounded_all_evaluated_nondominated'."
        )
    raw_cap = output.get("deployment_archive_max_size")
    if raw_cap is None:
        archive_cap = None
    else:
        if isinstance(raw_cap, bool) or not isinstance(raw_cap, int) or raw_cap <= 0:
            raise ValueError("deployment_archive_max_size must be null or positive.")
        archive_cap = raw_cap

    return IJOCParetoSMCSpecification(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        base_smc_specification=base_specification,
        adaptive_search_evaluations=raw_evaluations,
        allocation_policy=str(policy),
        minimum_pulls_per_type=raw_minimum_pulls,
        exp3_exploration=exploration,
        reward_weights=reward_weights,
        deployment_archive_max_size=archive_cap,
        competitive_archive_contract=str(competitive_archive),
    )
