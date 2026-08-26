from __future__ import annotations

"""Build the real 30-case IJOC MOTSP+MOKP freeze request."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
REQUEST_PATH = SUBMISSION_ROOT / "freeze_request.json"
FORMAL_ROOT = SUBMISSION_ROOT / "formal_study"
RELEASE_ROOT = SUBMISSION_ROOT / "release"
CASE_MANIFEST_PATH = FORMAL_ROOT / "case_manifest.json"
PACKET_MANIFEST_PATH = FORMAL_ROOT / "instance_packet_manifest.json"
REFERENCE_ROOT = FORMAL_ROOT / "metric_references"
TAIL_RECEIPT_PATH = (
    SUBMISSION_ROOT / "calibration" / "frozen" / "calibration_suite_receipt.json"
)
TAIL_POLICY_PATH = (
    SUBMISSION_ROOT / "calibration" / "frozen" / "tail_policy_freeze.json"
)
ANALYSIS_PLAN_PATH = SUBMISSION_ROOT / "protocol" / "formal_analysis_plan.json"
SUCCESSOR_ANALYSIS_PLAN_PATH = (
    SUBMISSION_ROOT
    / "protocol"
    / "formal_analysis_plan_v20_1_pls_restart.json"
)
SOURCE_ARCHIVE_PATH = (
    RELEASE_ROOT / "mo_nco_pareto_smc_v20_ijoc_source.tar.gz"
)
DEPENDENCY_LOCK_PATH = (
    RELEASE_ROOT / "requirements-formal-lock.txt"
)
ADAPTER_PATH = SUBMISSION_ROOT / "scripts" / "ijoc_algorithm_adapter.py"
REPLAY_PATH = SUBMISSION_ROOT / "scripts" / "ijoc_replay_verifier.py"
DEFAULT_STUDY_ID = (
    "pareto_smc_v20_ijoc_motsp_mokp_30case_10seed_3budget_v1"
)
DEFAULT_MOTSP_PLS_VERSION = "native-v1-population-40-neighborhood-40"
DEFAULT_MOTSP_PLS_ALGORITHM_ID = "motsp-pls-native-v1"
RESTART_MOTSP_PLS_ALGORITHM_ID = "motsp-pls-restart-native-v2"
RESTART_MOTSP_PLS_VERSION = (
    "restart-native-v2-population-40-neighborhood-40-"
    "exact-archive-retry-64"
)
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class FreezeRequestBuildOptions:
    __slots__ = (
        "tag",
        "request_path",
        "source_archive_path",
        "study_id",
        "frozen_output_directory",
        "results_directory",
        "formal_analysis_plan_path",
        "motsp_pls_algorithm_id",
        "motsp_pls_version",
    )

    def __init__(
        self,
        *,
        tag: str | None,
        request_path: Path,
        source_archive_path: Path,
        study_id: str,
        frozen_output_directory: Path,
        results_directory: Path,
        formal_analysis_plan_path: Path,
        motsp_pls_algorithm_id: str,
        motsp_pls_version: str,
    ) -> None:
        self.tag = tag
        self.request_path = request_path
        self.source_archive_path = source_archive_path
        self.study_id = study_id
        self.frozen_output_directory = frozen_output_directory
        self.results_directory = results_directory
        self.formal_analysis_plan_path = formal_analysis_plan_path
        self.motsp_pls_algorithm_id = motsp_pls_algorithm_id
        self.motsp_pls_version = motsp_pls_version


def resolve_request_build_options(
    *,
    tag: str | None,
    request_path: Path | None,
    source_archive_path: Path | None,
    study_id: str | None,
    frozen_output_directory: Path | None,
    results_directory: Path | None,
    formal_analysis_plan_path: Path | None,
    motsp_pls_algorithm_id: str | None,
    motsp_pls_version: str | None,
) -> FreezeRequestBuildOptions:
    """Resolve canonical defaults or an isolated version-tagged successor."""

    if tag is not None and TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(
            "tag must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    tagged = tag is not None
    resolved_request = request_path or (
        REQUEST_PATH
        if not tagged
        else SUBMISSION_ROOT / f"freeze_request_{tag}.json"
    )
    resolved_archive = source_archive_path or (
        SOURCE_ARCHIVE_PATH
        if not tagged
        else RELEASE_ROOT / f"mo_nco_pareto_smc_{tag}_source.tar.gz"
    )
    resolved_study_id = study_id or (
        DEFAULT_STUDY_ID
        if not tagged
        else (
            f"pareto_smc_{tag}_ijoc_motsp_mokp_"
            "30case_10seed_3budget_v1"
        )
    )
    resolved_frozen = frozen_output_directory or (
        FORMAL_ROOT / "frozen"
        if not tagged
        else FORMAL_ROOT / f"frozen_{tag}"
    )
    resolved_results = results_directory or (
        FORMAL_ROOT / "results"
        if not tagged
        else FORMAL_ROOT / f"formal_results_{tag}"
    )
    resolved_analysis_plan = formal_analysis_plan_path or (
        SUCCESSOR_ANALYSIS_PLAN_PATH
        if tag == "v20_1_pls_restart"
        else ANALYSIS_PLAN_PATH
    )
    automatic_pls_algorithm_id = (
        RESTART_MOTSP_PLS_ALGORITHM_ID
        if tag == "v20_1_pls_restart"
        else DEFAULT_MOTSP_PLS_ALGORITHM_ID
    )
    automatic_pls_version = (
        RESTART_MOTSP_PLS_VERSION
        if tag == "v20_1_pls_restart"
        else DEFAULT_MOTSP_PLS_VERSION
    )
    return FreezeRequestBuildOptions(
        tag=tag,
        request_path=Path(resolved_request).resolve(),
        source_archive_path=Path(resolved_archive).resolve(),
        study_id=resolved_study_id,
        frozen_output_directory=Path(resolved_frozen).resolve(),
        results_directory=Path(resolved_results).resolve(),
        formal_analysis_plan_path=Path(resolved_analysis_plan).resolve(),
        motsp_pls_algorithm_id=(
            motsp_pls_algorithm_id or automatic_pls_algorithm_id
        ),
        motsp_pls_version=motsp_pls_version or automatic_pls_version,
    )


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite constant {value!r}: {path}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def relative(path: Path) -> str:
    resolved = path.resolve()
    resolved.relative_to(SUBMISSION_ROOT.resolve())
    return resolved.relative_to(SUBMISSION_ROOT.resolve()).as_posix()


def repo_command_path(path: Path) -> str:
    """Render a submission artifact for commands launched at the repo root."""

    resolved = path.resolve()
    repo_root = SUBMISSION_ROOT.parent.resolve()
    resolved.relative_to(repo_root)
    return resolved.relative_to(repo_root).as_posix().replace("/", "\\")


def formal_python_version() -> str:
    """Return the exact version token consumed by the cold-runner gate."""

    return platform.python_version()


def algorithm_entry(
    *,
    role: str,
    families: list[str],
    version: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "families": families,
        "kind": "wrapper_script",
        "version": version,
        "adapter_artifact_path": relative(ADAPTER_PATH),
        "command_argv": [
            "{python_executable}",
            "{adapter_path}",
            "--input",
            "{input_path}",
            "--output",
            "{result_path}",
        ],
        "replay_verifier_artifact_path": relative(REPLAY_PATH),
        "replay_verifier_argv": [
            "{python_executable}",
            "{replay_verifier_path}",
            "--input",
            "{input_path}",
            "--result",
            "{result_path}",
            "--output",
            "{replay_result_path}",
        ],
        "configuration": configuration,
    }


def build_freeze_request(
    options: FreezeRequestBuildOptions,
) -> dict[str, object]:
    """Build one immutable formal request for a canonical or successor study."""

    relative(options.request_path)
    relative(options.source_archive_path)
    relative(options.frozen_output_directory)
    relative(options.results_directory)
    relative(options.formal_analysis_plan_path)
    if options.request_path.exists():
        raise FileExistsError(
            "Refusing to replace existing freeze request: "
            f"{options.request_path}"
        )
    occupied_outputs = [
        path
        for path in (
            options.frozen_output_directory,
            options.results_directory,
        )
        if path.exists()
    ]
    if occupied_outputs:
        raise FileExistsError(
            "Refusing to bind a successor command to an existing frozen/results "
            "directory: "
            + ", ".join(str(path) for path in occupied_outputs)
        )
    required_files = (
        CASE_MANIFEST_PATH,
        PACKET_MANIFEST_PATH,
        REFERENCE_ROOT / "reference_calibration_precommit.json",
        REFERENCE_ROOT / "reference_calibration_completion_receipt.json",
        TAIL_RECEIPT_PATH,
        TAIL_POLICY_PATH,
        options.formal_analysis_plan_path,
        options.source_archive_path,
        DEPENDENCY_LOCK_PATH,
        ADAPTER_PATH,
        REPLAY_PATH,
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Formal freeze inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )
    cases_manifest = strict_json(CASE_MANIFEST_PATH)
    packet_manifest = strict_json(PACKET_MANIFEST_PATH)
    packet_by_case = {
        str(item["case_id"]): item for item in packet_manifest["packets"]
    }
    reference_precommit = strict_json(
        REFERENCE_ROOT / "reference_calibration_precommit.json"
    )
    reference_cases = {
        str(item["case_id"]) for item in reference_precommit["cases"]
    }
    formal_cases = {str(item["case_id"]) for item in cases_manifest["cases"]}
    if reference_cases != formal_cases or set(packet_by_case) != formal_cases:
        raise ValueError(
            "Formal cases, packets, and reference precommit do not match."
        )
    if (
        reference_precommit["metric_contract"]["evaluation_code_sha256"]
        != file_sha256(REPLAY_PATH)
    ):
        raise ValueError("Reference calibration used another replay verifier.")
    tail_policy = strict_json(TAIL_POLICY_PATH)
    if (
        tail_policy.get("status") != "FROZEN"
        or tail_policy.get("policy_id") != "uniform_t30"
        or tail_policy.get("selection_gate") != "FALLBACK"
        or not tail_policy.get("fallback_applied")
    ):
        raise ValueError("The expected fail-closed uniform_t30 policy is not frozen.")
    analysis_plan = strict_json(options.formal_analysis_plan_path)
    seeds = [int(value) for value in analysis_plan["formal_seeds"]]
    budgets = [int(value) for value in analysis_plan["evaluation_budgets"]]
    checkpoint = int(analysis_plan["anytime_checkpoint_period"])

    fixed_core = {
        "reference_directions": [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
        "particles_per_reference": 8,
        "beta_schedule": [0.0, 0.5, 1.0, 2.0],
        "ess_threshold": 0.5,
        "chebyshev_rho": 0.03,
        "archive_tolerance": 0.0,
        "deployment_archive_max_size": 100,
        "normalized_cell_width": 0.05,
        "global_refresh_probability": 0.0,
    }
    frozen_tail = dict(tail_policy["configuration"]["candidate"])
    common_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    treatment_configuration = {
        "families": ["MOKP", "MOTSP"],
        "environment": common_environment,
        "treatment": {
            "fixed_core": fixed_core,
            "frozen_tail_policy": frozen_tail,
            "tail_policy_artifact_sha256": file_sha256(TAIL_POLICY_PATH),
        },
    }
    algorithms = {
        "ijoc-pareto-smc": algorithm_entry(
            role="treatment",
            families=["MOKP", "MOTSP"],
            version="0.20.0-uniform_t30-fallback-frozen",
            configuration=treatment_configuration,
        ),
        "mokp-binary-moead-native-v1": algorithm_entry(
            role="baseline",
            families=["MOKP"],
            version="native-v1-population-40",
            configuration={
                "families": ["MOKP"],
                "environment": common_environment,
                "population_size": 40,
                "scope": "transparent_in_repo_native_not_external_replication",
            },
        ),
        "mokp-binary-nsga2-native-v1": algorithm_entry(
            role="baseline",
            families=["MOKP"],
            version="native-v1-population-40",
            configuration={
                "families": ["MOKP"],
                "environment": common_environment,
                "population_size": 40,
                "scope": "transparent_in_repo_native_not_external_replication",
            },
        ),
        "mokp-pls-native-v1": algorithm_entry(
            role="baseline",
            families=["MOKP"],
            version="native-v1",
            configuration={
                "families": ["MOKP"],
                "environment": common_environment,
                "scope": "transparent_in_repo_native_not_external_replication",
            },
        ),
        options.motsp_pls_algorithm_id: algorithm_entry(
            role="baseline",
            families=["MOTSP"],
            version=options.motsp_pls_version,
            configuration={
                "families": ["MOTSP"],
                "environment": common_environment,
                "population_size": 40,
                "neighborhood_sample": 40,
                "scope": "transparent_in_repo_native_not_external_replication",
                **(
                    {
                        "archive_tolerance": 0.0,
                        "stalled_expansion_policy": (
                            "uniform-random-unvisited-v1"
                        ),
                        "restart_random_attempts": 64,
                        "liveness_contract": (
                            "each_nonterminal_step_adds_evaluation_or_fails_v1"
                        ),
                        "successor_provenance": {
                            "successor_tag": options.tag,
                            "matched_design_change": (
                                "MOTSP_PLS_baseline_implementation_replacement_only"
                            ),
                            "exact_archive_equivalence_domain": (
                                "integer_valued_motsp_objectives"
                            ),
                            "restart_trajectory_equivalence_claim": (
                                "NOT_CLAIMED"
                            ),
                            "v1_liveness_defect": (
                                "visited_neighborhood_exhaustion_can_create_"
                                "zero_evaluation_progress"
                            ),
                            "formal_estimand_unchanged": True,
                            "cases_seeds_budgets_metrics_unchanged": True,
                        },
                    }
                    if options.motsp_pls_algorithm_id
                    == RESTART_MOTSP_PLS_ALGORITHM_ID
                    else {}
                ),
            },
        ),
        "pymoo-moead": algorithm_entry(
            role="baseline",
            families=["MOTSP"],
            version="pymoo-0.6.1.6-population-40",
            configuration={
                "families": ["MOTSP"],
                "environment": common_environment,
                "population_size": 40,
            },
        ),
        "pymoo-nsga2": algorithm_entry(
            role="baseline",
            families=["MOTSP"],
            version="pymoo-0.6.1.6-population-40",
            configuration={
                "families": ["MOTSP"],
                "environment": common_environment,
                "population_size": 40,
            },
        ),
    }
    baselines = {
        "MOKP": [
            "mokp-binary-nsga2-native-v1",
            "mokp-binary-moead-native-v1",
            "mokp-pls-native-v1",
        ],
        "MOTSP": [
            "pymoo-nsga2",
            "pymoo-moead",
            options.motsp_pls_algorithm_id,
        ],
    }
    families = []
    for family in ("MOKP", "MOTSP"):
        family_cases = []
        for case in sorted(
            (
                item
                for item in cases_manifest["cases"]
                if str(item["family"]) == family
            ),
            key=lambda item: str(item["case_id"]),
        ):
            case_id = str(case["case_id"])
            reference_path = REFERENCE_ROOT / "cases" / f"{case_id}.json"
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            family_cases.append(
                {
                    "id": case_id,
                    "instance_path": relative(
                        FORMAL_ROOT / str(packet_by_case[case_id]["path"])
                    ),
                    "metric_reference": {
                        "source_artifact_path": relative(reference_path)
                    },
                }
            )
        family_algorithms = ["ijoc-pareto-smc", *baselines[family]]
        families.append(
            {
                "id": family,
                "cases": family_cases,
                "algorithms": family_algorithms,
                "required_baselines": baselines[family],
            }
        )
    request = {
        "schema": "ijoc_manifest_freeze_request_v1",
        "study_id": options.study_id,
        "evidence_status": "NOT_RUN",
        "problem_families": families,
        "algorithms": algorithms,
        "seeds": seeds,
        "budgets": budgets,
        "anytime_checkpoint_period": checkpoint,
        "source_archive_path": relative(options.source_archive_path),
        "dependency_lock_path": relative(DEPENDENCY_LOCK_PATH),
        "tail_calibration_suite_receipt_path": relative(TAIL_RECEIPT_PATH),
        "reference_calibration_precommit_path": relative(
            REFERENCE_ROOT / "reference_calibration_precommit.json"
        ),
        "reference_calibration_completion_receipt_path": relative(
            REFERENCE_ROOT / "reference_calibration_completion_receipt.json"
        ),
        "tail_policy_artifact_path": relative(TAIL_POLICY_PATH),
        "formal_analysis_plan_path": relative(
            options.formal_analysis_plan_path
        ),
        "python_version": formal_python_version(),
        "license": (
            "internal-evaluation-only; public license pending author approval"
        ),
        "reproduction_commands": [
            (
                r"C:\miniconda3\python.exe scripts\freeze_ijoc_manifests.py "
                "--request "
                + repo_command_path(options.request_path)
                + " --output-directory "
                + repo_command_path(options.frozen_output_directory)
            ),
            (
                r"C:\miniconda3\python.exe scripts\run_ijoc_cold_matrix.py "
                "--study "
                + repo_command_path(
                    options.frozen_output_directory / "study.json"
                )
                + " "
                r"--execution-plan "
                + repo_command_path(
                    options.frozen_output_directory / "execution_plan.json"
                )
                + " --results-directory "
                + repo_command_path(options.results_directory)
                + " "
                r"--timeout-seconds 1800 --workers 4"
            ),
            (
                r"C:\miniconda3\python.exe scripts\audit_ijoc_postrun.py "
                "--study "
                + repo_command_path(
                    options.frozen_output_directory / "study.json"
                )
                + " "
                r"--execution-plan "
                + repo_command_path(
                    options.frozen_output_directory / "execution_plan.json"
                )
                + " --results-directory "
                + repo_command_path(options.results_directory)
            ),
        ],
    }
    options.request_path.parent.mkdir(parents=True, exist_ok=True)
    with options.request_path.open("xb") as handle:
        handle.write(canonical_bytes(request))
    return {
        "schema": "ijoc_formal_freeze_request_build_result_v1",
        "version_tag": options.tag,
        "request_path": relative(options.request_path),
        "request_sha256": file_sha256(options.request_path),
        "source_archive_path": relative(options.source_archive_path),
        "source_archive_sha256": file_sha256(options.source_archive_path),
        "formal_analysis_plan_path": relative(
            options.formal_analysis_plan_path
        ),
        "formal_analysis_plan_sha256": file_sha256(
            options.formal_analysis_plan_path
        ),
        "study_id": options.study_id,
        "motsp_pls_algorithm_id": options.motsp_pls_algorithm_id,
        "motsp_pls_version": options.motsp_pls_version,
        "case_count": sum(len(family["cases"]) for family in families),
        "algorithm_count": len(algorithms),
        "expected_run_count": (
            sum(
                len(family["cases"]) * len(family["algorithms"])
                for family in families
            )
            * len(seeds)
            * len(budgets)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable IJOC formal freeze request. A version tag "
            "derives isolated request, source, frozen, and results paths."
        )
    )
    parser.add_argument("--tag", "--version-tag", dest="tag")
    parser.add_argument("--request-path", type=Path)
    parser.add_argument("--source-archive-path", type=Path)
    parser.add_argument("--study-id")
    parser.add_argument("--frozen-output-directory", type=Path)
    parser.add_argument("--results-directory", type=Path)
    parser.add_argument("--formal-analysis-plan-path", type=Path)
    parser.add_argument("--motsp-pls-algorithm-id")
    parser.add_argument("--motsp-pls-version")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = resolve_request_build_options(
        tag=args.tag,
        request_path=args.request_path,
        source_archive_path=args.source_archive_path,
        study_id=args.study_id,
        frozen_output_directory=args.frozen_output_directory,
        results_directory=args.results_directory,
        formal_analysis_plan_path=args.formal_analysis_plan_path,
        motsp_pls_algorithm_id=args.motsp_pls_algorithm_id,
        motsp_pls_version=args.motsp_pls_version,
    )
    result = build_freeze_request(options)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
