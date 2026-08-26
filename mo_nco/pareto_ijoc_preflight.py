from __future__ import annotations

"""Fail-closed preflight for the IJOC computational study.

The preflight validates that the study is frozen before the first optimizer
run.  It does not turn a planned study into empirical evidence: a valid
prelaunch packet still reports ``NOT_RUN`` until every declared row has a
mechanically valid result record.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


STUDY_SCHEMA = "ijoc_competitive_study_v3"
REPRO_SCHEMA = "ijoc_reproducibility_manifest_v2"
METRIC_SCHEMA = "ijoc_metric_reference_manifest_v2"
CONFIG_SCHEMA = "ijoc_algorithm_configuration_matrix_v1"
TREATMENT_ID = "ijoc-pareto-smc"
MIN_CASES_PER_FAMILY = 10
MIN_TOTAL_CASES = 30


@dataclass(frozen=True)
class IJOCPreflightResult:
    study_sha256: str
    metric_reference_sha256: str
    configuration_matrix_sha256: str
    reproducibility_manifest_sha256: str
    problem_family_count: int
    case_count: int
    algorithm_count: int
    seed_count: int
    budget_count: int
    expected_run_count: int
    generality_gate: str
    baseline_gate: str
    exact_matrix_gate: str
    artifact_release_gate: str
    submission_preflight_gate: str
    evidence_status: str
    submission_verdict: str
    reasons: Tuple[str, ...]

    def metadata(self) -> Dict[str, object]:
        return {
            "schema": "ijoc_competitive_preflight_result_v3",
            "study_sha256": self.study_sha256,
            "metric_reference_sha256": self.metric_reference_sha256,
            "configuration_matrix_sha256": self.configuration_matrix_sha256,
            "reproducibility_manifest_sha256": (
                self.reproducibility_manifest_sha256
            ),
            "problem_family_count": self.problem_family_count,
            "case_count": self.case_count,
            "algorithm_count": self.algorithm_count,
            "seed_count": self.seed_count,
            "budget_count": self.budget_count,
            "expected_run_count": self.expected_run_count,
            "generality_gate": self.generality_gate,
            "baseline_gate": self.baseline_gate,
            "exact_matrix_gate": self.exact_matrix_gate,
            "artifact_release_gate": self.artifact_release_gate,
            "submission_preflight_gate": self.submission_preflight_gate,
            "evidence_status": self.evidence_status,
            "submission_verdict": self.submission_verdict,
            "reasons": self.reasons,
        }


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON field is forbidden: {key!r}.")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _strict_json(path: Path) -> tuple[Mapping[str, Any], bytes, str]:
    if not path.is_file():
        raise ValueError(f"Required IJOC artifact is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Artifact is not strict UTF-8 JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact root must be a JSON object: {path}")
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} has an unexpected shape; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )


def _string_list(value: object, label: str, *, nonempty: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{label} must be a nonempty JSON array of strings.")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{label} must contain nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates.")
    return result


def _integer_list(
    value: object,
    label: str,
    *,
    minimum_count: int,
    minimum_value: int,
) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) < minimum_count:
        raise ValueError(f"{label} must contain at least {minimum_count} integers.")
    result = tuple(value)
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < minimum_value
        for item in result
    ):
        raise ValueError(
            f"{label} must contain integers no smaller than {minimum_value}."
        )
    if len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError(f"{label} must be strictly increasing and duplicate-free.")
    return result



def _resolve_bound_child(parent: Path, raw_path: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a nonempty relative string.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{label}.path must be relative to its manifest.")
    root = parent.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{label}.path escapes the bound artifact directory."
        ) from error
    return resolved

def _bound_artifact(
    parent: Path,
    value: object,
    *,
    label: str,
    expected_schema: str,
) -> tuple[Path, Mapping[str, Any], str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    _exact_keys(value, {"path", "sha256"}, label)
    raw_path = value.get("path")
    declared_sha = value.get("sha256")
    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        raise ValueError(f"{label}.sha256 must be a 64-character digest.")
    try:
        int(declared_sha, 16)
    except ValueError as error:
        raise ValueError(f"{label}.sha256 must be hexadecimal.") from error
    path = _resolve_bound_child(parent, raw_path, label=label)
    payload, _, actual_sha = _strict_json(path)
    if actual_sha != declared_sha:
        raise ValueError(f"{label} SHA-256 does not match the bound artifact.")
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{label} must use schema {expected_schema!r}.")
    return path, payload, actual_sha


def _bound_file(
    parent: Path,
    value: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    _exact_keys(value, {"path", "sha256"}, label)
    raw_path = value.get("path")
    declared_sha = value.get("sha256")
    if not isinstance(declared_sha, str) or len(declared_sha) != 64:
        raise ValueError(f"{label}.sha256 must be a 64-character digest.")
    try:
        int(declared_sha, 16)
    except ValueError as error:
        raise ValueError(f"{label}.sha256 must be hexadecimal.") from error
    path = _resolve_bound_child(parent, raw_path, label=label)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != declared_sha:
        raise ValueError(f"{label} SHA-256 does not match the bound file.")
    return path, actual


def _configuration_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    _exact_keys(
        row,
        {
            "case_id",
            "algorithm",
            "seed",
            "budget",
            "configuration",
            "configuration_sha256",
        },
        "configuration row",
    )
    case_id = row.get("case_id")
    algorithm = row.get("algorithm")
    seed = row.get("seed")
    budget = row.get("budget")
    configuration = row.get("configuration")
    digest = row.get("configuration_sha256")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("configuration row case_id must be nonempty.")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("configuration row algorithm must be nonempty.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("configuration row seed must be a nonnegative integer.")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("configuration row budget must be a positive integer.")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("configuration_sha256 must be a 64-character digest.")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError("configuration_sha256 must be hexadecimal.") from error
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be a readable JSON object.")
    for key, expected in (
        ("case_id", case_id),
        ("algorithm", algorithm),
        ("seed", seed),
        ("budget", budget),
    ):
        if configuration.get(key) != expected:
            raise ValueError(
                f"Readable configuration field {key!r} does not match its row."
            )
    actual_digest = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if actual_digest != digest:
        raise ValueError("Readable configuration bytes do not match configuration_sha256.")
    return case_id, algorithm, seed, budget


def audit_ijoc_competitive_study(path: str | Path) -> IJOCPreflightResult:
    """Validate a frozen IJOC study packet without evaluating a solution."""

    study_path = Path(path).expanduser().resolve()
    study, _, study_sha = _strict_json(study_path)
    study_keys = {
        "schema",
        "study_id",
        "problem_families",
        "seeds",
        "budgets",
        "anytime_checkpoint_period",
        "metric_reference_manifest",
        "algorithm_configuration_matrix",
        "artifact_release",
    }
    if set(study) not in (study_keys, study_keys | {"formal_analysis_plan"}):
        raise ValueError("study root has an unexpected shape.")
    if study.get("schema") != STUDY_SCHEMA:
        raise ValueError(f"study schema must be {STUDY_SCHEMA!r}.")
    if not isinstance(study.get("study_id"), str) or not study.get("study_id"):
        raise ValueError("study_id must be a nonempty string.")

    seeds = _integer_list(
        study.get("seeds"),
        "seeds",
        minimum_count=10,
        minimum_value=0,
    )
    budgets = _integer_list(
        study.get("budgets"),
        "budgets",
        minimum_count=3,
        minimum_value=1,
    )
    checkpoint = study.get("anytime_checkpoint_period")
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint <= 0:
        raise ValueError("anytime_checkpoint_period must be a positive integer.")
    if any(budget % checkpoint != 0 for budget in budgets):
        raise ValueError("The common checkpoint period must divide every budget.")
    if "formal_analysis_plan" in study:
        _bound_artifact(
            study_path.parent,
            study.get("formal_analysis_plan"),
            label="formal_analysis_plan",
            expected_schema="ijoc_formal_analysis_plan_v1",
        )

    families_raw = study.get("problem_families")
    if not isinstance(families_raw, list) or not families_raw:
        raise ValueError("problem_families must be a nonempty array.")
    family_ids: list[str] = []
    all_cases: list[str] = []
    all_algorithms: set[str] = set()
    expected_rows: set[tuple[str, str, int, int]] = set()
    baseline_gate = "PASS"
    required_baseline_algorithms: set[str] = set()
    per_family_case_gate = "PASS"
    for index, raw_family in enumerate(families_raw):
        if not isinstance(raw_family, dict):
            raise ValueError("Each problem family must be a JSON object.")
        _exact_keys(raw_family, {"id", "cases", "algorithms", "required_baselines"}, f"family {index}")
        family_id = raw_family.get("id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("Every family id must be nonempty.")
        family_ids.append(family_id)
        cases = _string_list(raw_family.get("cases"), f"family {family_id} cases")
        algorithms = _string_list(raw_family.get("algorithms"), f"family {family_id} algorithms")
        baselines = _string_list(raw_family.get("required_baselines"), f"family {family_id} required_baselines")
        if len(cases) < MIN_CASES_PER_FAMILY:
            per_family_case_gate = "FAIL"
        if TREATMENT_ID not in algorithms:
            raise ValueError(f"Family {family_id} omits the frozen treatment {TREATMENT_ID!r}.")
        if (
            len(baselines) < 3
            or TREATMENT_ID in baselines
            or not set(baselines).issubset(algorithms)
        ):
            baseline_gate = "FAIL"
        required_baseline_algorithms.update(baselines)
        all_cases.extend(cases)
        all_algorithms.update(algorithms)
        for case in cases:
            for algorithm in algorithms:
                for seed in seeds:
                    for budget in budgets:
                        expected_rows.add((case, algorithm, seed, budget))
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("Problem-family ids must be unique.")
    if len(set(all_cases)) != len(all_cases):
        raise ValueError("Case ids must be globally unique across problem families.")

    metric_path, metric, metric_sha = _bound_artifact(
        study_path.parent,
        study.get("metric_reference_manifest"),
        label="metric_reference_manifest",
        expected_schema=METRIC_SCHEMA,
    )
    _exact_keys(metric, {"schema", "cases"}, "metric reference manifest")
    metric_cases = metric.get("cases")
    if not isinstance(metric_cases, dict):
        raise ValueError("metric reference cases must be a JSON object.")
    if set(metric_cases) != set(all_cases):
        raise ValueError("Metric-reference cases do not exactly match the frozen study cases.")
    for case_id, payload in metric_cases.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Metric reference for {case_id} must be an object.")
        required = {
            "source_artifact",
            "source_role",
            "reference_sha256",
            "reference_points",
            "ideal",
            "nadir",
            "hv_reference",
        }
        if set(payload) != required:
            raise ValueError(f"Metric reference for {case_id} has an unexpected shape.")
        digest = payload["reference_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                f"Metric reference_sha256 for {case_id} must be a SHA-256 digest."
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                f"Metric reference_sha256 for {case_id} must be hexadecimal."
            ) from error
        _bound_file(
            metric_path.parent,
            payload["source_artifact"],
            label=f"metric source artifact for {case_id}",
        )
        if payload["source_role"] not in {
            "calibration_only_disjoint_from_current_arms",
            "reference_calibration_precommitted_disjoint_arms_and_seeds",
        }:
            raise ValueError(
                f"Metric reference for {case_id} lacks precommitted "
                "reference-calibration provenance."
            )
        points_raw = payload["reference_points"]
        ideal_raw = payload["ideal"]
        nadir_raw = payload["nadir"]
        hv_raw = payload["hv_reference"]
        if not isinstance(points_raw, list) or not points_raw:
            raise ValueError(f"Metric reference for {case_id} must contain points.")
        if not all(isinstance(value, list) for value in (ideal_raw, nadir_raw, hv_raw)):
            raise ValueError(f"Metric box vectors for {case_id} must be arrays.")
        ideal = tuple(float(value) for value in ideal_raw)
        nadir = tuple(float(value) for value in nadir_raw)
        hv_reference = tuple(float(value) for value in hv_raw)
        dimension = len(ideal)
        if dimension < 2 or len(nadir) != dimension or len(hv_reference) != dimension:
            raise ValueError(f"Metric dimensions for {case_id} are inconsistent.")
        points = []
        for raw_point in points_raw:
            if not isinstance(raw_point, list) or len(raw_point) != dimension:
                raise ValueError(f"A metric point for {case_id} has the wrong dimension.")
            point = tuple(float(value) for value in raw_point)
            if any(not math.isfinite(value) for value in point):
                raise ValueError(f"Metric points for {case_id} must be finite.")
            points.append(point)
        if len(set(points)) != len(points):
            raise ValueError(f"Metric points for {case_id} must be unique.")
        if any(
            not math.isfinite(value)
            for vector in (ideal, nadir, hv_reference)
            for value in vector
        ):
            raise ValueError(f"Metric box for {case_id} must be finite.")
        if any(not lower < upper for lower, upper in zip(ideal, nadir)):
            raise ValueError(f"Metric ideal/nadir box for {case_id} is invalid.")
        if any(reference < upper for reference, upper in zip(hv_reference, nadir)):
            raise ValueError(f"HV reference for {case_id} must dominate the nadir.")
        for point in points:
            if any(
                value < lower or value > upper
                for value, lower, upper in zip(point, ideal, nadir)
            ):
                raise ValueError(f"Metric point for {case_id} leaves the ideal/nadir box.")
            if any(value > reference for value, reference in zip(point, hv_reference)):
                raise ValueError(f"HV reference for {case_id} does not dominate every point.")
        for left_index, left in enumerate(points):
            for right_index, right in enumerate(points):
                if left_index == right_index:
                    continue
                if all(a <= b for a, b in zip(left, right)) and any(
                    a < b for a, b in zip(left, right)
                ):
                    raise ValueError(f"Metric reference for {case_id} contains a dominated point.")
        reference_digest = hashlib.sha256(
            json.dumps(
                points,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if reference_digest != payload["reference_sha256"]:
            raise ValueError(f"Metric reference point hash for {case_id} does not match.")

    _, configuration, configuration_sha = _bound_artifact(
        study_path.parent,
        study.get("algorithm_configuration_matrix"),
        label="algorithm_configuration_matrix",
        expected_schema=CONFIG_SCHEMA,
    )
    _exact_keys(configuration, {"schema", "rows"}, "configuration matrix")
    rows = configuration.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Configuration rows must be a JSON array.")
    actual_rows: set[tuple[str, str, int, int]] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise ValueError("Every configuration row must be a JSON object.")
        key = _configuration_key(raw_row)
        if key in actual_rows:
            raise ValueError(f"Duplicate configuration row: {key!r}.")
        actual_rows.add(key)
    exact_matrix_gate = "PASS" if actual_rows == expected_rows else "FAIL"

    reproducibility_path, reproducibility, reproducibility_sha = _bound_artifact(
        study_path.parent,
        study.get("artifact_release"),
        label="artifact_release",
        expected_schema=REPRO_SCHEMA,
    )
    reproducibility_keys = {
        "schema",
        "source_archive",
        "instance_files",
        "reproduction_commands",
        "baseline_bindings",
        "license",
        "environment",
    }
    if set(reproducibility) not in (
        reproducibility_keys,
        reproducibility_keys | {"formal_analysis_plan"},
    ):
        raise ValueError("reproducibility manifest has an unexpected shape.")
    _bound_file(
        reproducibility_path.parent,
        reproducibility.get("source_archive"),
        label="reproducibility source_archive",
    )
    if "formal_analysis_plan" in reproducibility:
        _bound_file(
            reproducibility_path.parent,
            reproducibility.get("formal_analysis_plan"),
            label="reproducibility formal_analysis_plan",
        )
    instance_files = reproducibility.get("instance_files")
    if not isinstance(instance_files, list) or not instance_files:
        raise ValueError("reproducibility instance_files must be nonempty.")
    bound_instance_cases: set[str] = set()
    for index, raw_file in enumerate(instance_files):
        if not isinstance(raw_file, dict):
            raise ValueError("Every reproducibility instance file must be an object.")
        _exact_keys(
            raw_file,
            {"case_id", "path", "sha256"},
            f"reproducibility instance_files[{index}]",
        )
        case_id = raw_file.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every reproducibility instance file needs a case_id.")
        if case_id in bound_instance_cases:
            raise ValueError(f"Duplicate reproducibility instance case: {case_id!r}.")
        bound_instance_cases.add(case_id)
        _bound_file(
            reproducibility_path.parent,
            {"path": raw_file.get("path"), "sha256": raw_file.get("sha256")},
            label=f"reproducibility instance_files[{index}]",
        )
    if bound_instance_cases != set(all_cases):
        raise ValueError(
            "Reproducibility instance files do not exactly match the study cases."
        )
    commands = _string_list(
        reproducibility.get("reproduction_commands"),
        "reproducibility_commands",
    )
    forbidden_command_markers = ("TBD", "TODO", "<PATH>", "PLACEHOLDER")
    if any(
        marker in command.upper()
        for command in commands
        for marker in forbidden_command_markers
    ):
        raise ValueError("Reproduction commands contain an unresolved placeholder.")
    baseline_bindings = reproducibility.get("baseline_bindings")
    if not isinstance(baseline_bindings, list):
        raise ValueError("baseline_bindings must be a JSON array.")
    bound_baselines: set[str] = set()
    for index, binding in enumerate(baseline_bindings):
        if not isinstance(binding, dict):
            raise ValueError("Every baseline binding must be a JSON object.")
        _exact_keys(
            binding,
            {"algorithm", "kind", "version", "command", "artifact"},
            f"baseline_bindings[{index}]",
        )
        algorithm = binding.get("algorithm")
        kind = binding.get("kind")
        version = binding.get("version")
        command = binding.get("command")
        if not isinstance(algorithm, str) or not algorithm:
            raise ValueError("Baseline binding algorithm must be nonempty.")
        if algorithm in bound_baselines:
            raise ValueError(f"Duplicate baseline binding: {algorithm!r}.")
        bound_baselines.add(algorithm)
        if kind not in {"executable", "python_module", "wrapper_script"}:
            raise ValueError(f"Unsupported baseline binding kind for {algorithm}.")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Baseline binding version for {algorithm} is empty.")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"Baseline binding command for {algorithm} is empty.")
        if any(marker in command.upper() for marker in forbidden_command_markers):
            raise ValueError(f"Baseline command for {algorithm} contains a placeholder.")
        _bound_file(
            reproducibility_path.parent,
            binding.get("artifact"),
            label=f"baseline binding artifact for {algorithm}",
        )
    if bound_baselines != required_baseline_algorithms:
        raise ValueError(
            "Baseline bindings do not exactly match the required baseline algorithms."
        )

    license_name = reproducibility.get("license")
    environment = reproducibility.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("reproducibility environment must be a JSON object.")
    _exact_keys(
        environment,
        {"python_version", "dependency_lock"},
        "reproducibility environment",
    )
    python_version = environment.get("python_version")
    if not isinstance(python_version, str) or not python_version.strip():
        raise ValueError("reproducibility python_version must be nonempty.")
    _bound_file(
        reproducibility_path.parent,
        environment.get("dependency_lock"),
        label="reproducibility dependency_lock",
    )
    artifact_release_gate = (
        "PASS"
        if isinstance(license_name, str)
        and bool(license_name.strip())
        and license_name.strip().upper()
        not in {"TBD", "TO_BE_SELECTED", "NONE", "NO LICENSE"}
        else "FAIL"
    )

    generality_gate = (
        "PASS"
        if len(families_raw) >= 2
        and per_family_case_gate == "PASS"
        and len(all_cases) >= MIN_TOTAL_CASES
        else "FAIL"
    )
    reasons: list[str] = []
    for gate, reason in (
        (generality_gate, "study lacks two families, ten cases per family, or thirty total cases"),
        (baseline_gate, "a family has fewer than three frozen required baselines"),
        (exact_matrix_gate, "configuration matrix is not the exact Cartesian study matrix"),
        (artifact_release_gate, "source/data/commands/license artifact contract is incomplete"),
    ):
        if gate != "PASS":
            reasons.append(reason)
    preflight = "PASS" if not reasons else "FAIL"
    return IJOCPreflightResult(
        study_sha256=study_sha,
        metric_reference_sha256=metric_sha,
        configuration_matrix_sha256=configuration_sha,
        reproducibility_manifest_sha256=reproducibility_sha,
        problem_family_count=len(families_raw),
        case_count=len(all_cases),
        algorithm_count=len(all_algorithms),
        seed_count=len(seeds),
        budget_count=len(budgets),
        expected_run_count=len(expected_rows),
        generality_gate=generality_gate,
        baseline_gate=baseline_gate,
        exact_matrix_gate=exact_matrix_gate,
        artifact_release_gate=artifact_release_gate,
        submission_preflight_gate=preflight,
        evidence_status="NOT_RUN",
        submission_verdict="HOLD",
        reasons=tuple(reasons),
    )
