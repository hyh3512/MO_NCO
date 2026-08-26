from __future__ import annotations

"""Fail-closed, crash-recoverable evaluation ledger for prospective V21e3 runs."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sqlite3
import struct
import time
from typing import Callable, Mapping, Sequence, Tuple

from .pareto_ijoc_problem import (
    MultiObjectiveCombinatorialProblem,
    MultiObjectiveKnapsackInstance,
    MultiObjectiveTSPProblemAdapter,
    Solution,
    problem_sha256,
)
from .types import ObjectiveVector


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _portable_receipt_path(value: str, *, field: str) -> str:
    """Validate a repository-artifact display path without resolving it.

    The SQLite connection still uses an absolute host path.  This separate
    display value is committed into the terminal receipt so a copied evidence
    directory does not retain a machine-specific drive path.
    """

    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or pure.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{field} must be a canonical relative POSIX path.")
    return text


def _solution_sha256(solution: Sequence[int]) -> str:
    return hashlib.sha256(
        _canonical_bytes([int(value) for value in solution])
    ).hexdigest()


def _qualified_type(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception as error:
        return f"<repr-failed:{_qualified_type(error)}:{error}>"


def _raw_value_payload(value: object) -> object:
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": bool(value)}
    if type(value) is int:
        return {"kind": "int", "decimal": str(value)}
    if type(value) is float:
        return {
            "kind": "float64",
            "ieee754_be_hex": struct.pack(">d", value).hex(),
        }
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if type(value) in {tuple, list}:
        return {
            "kind": "sequence",
            "container_type": _qualified_type(value),
            "items": [_raw_value_payload(item) for item in value],
        }
    return {
        "kind": "unsupported",
        "qualified_type": _qualified_type(value),
        "repr": _safe_repr(value),
    }


def _canonical_raw_proposal(proposal: object) -> tuple[str, str]:
    try:
        items = list(proposal)  # type: ignore[arg-type]
    except Exception as error:
        payload: object = {
            "schema": "v21e3_raw_proposal_v1",
            "container_type": _qualified_type(proposal),
            "iterable": False,
            "repr": _safe_repr(proposal),
            "iteration_error": {
                "qualified_type": _qualified_type(error),
                "message": str(error),
            },
        }
    else:
        payload = {
            "schema": "v21e3_raw_proposal_v1",
            "container_type": _qualified_type(proposal),
            "iterable": True,
            "items": [_raw_value_payload(item) for item in items],
        }
    raw = _canonical_bytes(payload)
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    raw = _canonical_bytes(payload)
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ObjectiveContractError(RuntimeError):
    """The raw evaluator violated the prospectively frozen objective contract."""

    def __init__(self, code: str, message: str, *, detail: object) -> None:
        super().__init__(message)
        self.code = str(code)
        self.detail = detail


class ObjectiveEvaluationError(RuntimeError):
    """The raw evaluator raised after the attempt start was durably recorded."""

    code = "OBJECTIVE_EVALUATOR_EXCEPTION"


class SolutionValidationError(RuntimeError):
    """The proposed solution failed validation after its attempt was recorded."""

    code = "SOLUTION_VALIDATION_FAILED"


class EvaluationContextError(RuntimeError):
    """An attempt context disagreed with the prospectively frozen run context."""

    code = "EVALUATION_CONTEXT_MISMATCH"


@dataclass(frozen=True, init=False)
class V21E3RunContext:
    """Canonical binding for one prospective algorithm run.

    ``v21e3_run_context_v1`` remains readable for legacy development and
    generic ledger tests.  The successor ``v21e3r1_run_context_v2`` is the
    prospective algorithm root and requires every duplicated authoritative
    field to agree with its value inside ``algorithm_config``.
    """

    canonical_json: str
    digest_sha256: str

    _REQUIRED = frozenset(
        {
            "schema",
            "case_artifact_sha256",
            "problem_semantic_sha256",
            "candidate_id",
            "algorithm_config",
            "candidate_config_sha256",
            "algorithm_source_sha256",
            "reference_directions",
            "seed",
            "charged_evaluation_budget",
            "evidence_partition",
        }
    )

    def __init__(self, payload: Mapping[str, object]) -> None:
        materialized = dict(payload)
        missing = sorted(self._REQUIRED - set(materialized))
        if missing:
            raise ValueError(f"V21e3 run context misses required fields: {missing}")
        schema = materialized.get("schema")
        if schema not in {
            "v21e3_run_context_v1",
            "v21e3r1_run_context_v2",
        }:
            raise ValueError("Unsupported V21e3 run-context schema.")
        for key in (
            "case_artifact_sha256",
            "problem_semantic_sha256",
            "candidate_config_sha256",
            "algorithm_source_sha256",
        ):
            value = str(materialized[key])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"V21e3 run-context field {key} is not lowercase SHA-256.")
        algorithm_config = materialized["algorithm_config"]
        if not isinstance(algorithm_config, Mapping):
            raise ValueError("V21e3 algorithm_config must be a mapping.")
        config_hash = hashlib.sha256(_canonical_bytes(algorithm_config)).hexdigest()
        if config_hash != materialized["candidate_config_sha256"]:
            raise ValueError("V21e3 candidate config digest does not match its payload.")
        mirror_fields = (
            ("candidate_id", "candidate_id"),
            ("seed", "seed"),
            ("charged_evaluation_budget", "charged_evaluations"),
            ("evidence_partition", "phase"),
            ("reference_directions", "reference_directions"),
        )
        for outer_key, config_key in mirror_fields:
            if config_key not in algorithm_config:
                if schema == "v21e3r1_run_context_v2":
                    raise ValueError(
                        "V21e3r1 run context is missing mirrored field: "
                        f"algorithm_config.{config_key}."
                    )
                continue
            if _canonical_bytes(materialized[outer_key]) != _canonical_bytes(
                algorithm_config[config_key]
            ):
                raise ValueError(
                    "V21e3 run context has inconsistent mirrored field: "
                    f"{outer_key} != algorithm_config.{config_key}."
                )
        if (
            isinstance(materialized["charged_evaluation_budget"], bool)
            or not isinstance(materialized["charged_evaluation_budget"], int)
            or int(materialized["charged_evaluation_budget"]) < 0
        ):
            raise ValueError("V21e3 charged evaluation budget is invalid.")
        if isinstance(materialized["seed"], bool) or not isinstance(
            materialized["seed"], int
        ):
            raise ValueError("V21e3 seed is invalid.")
        diagnostic_id = algorithm_config.get("development_diagnostic_id")
        if type(diagnostic_id) is str and diagnostic_id.startswith("V21E3R1_V9_"):
            if materialized.get("v9_resource_contract_schema") != (
                "v21e3r1_v9_ast_resource_contract_v1"
            ):
                raise ValueError("V9 run context misses its exact A/S/T contract schema.")
            try:
                lower = tuple(materialized["objective_lower_bounds"])  # type: ignore[arg-type]
                upper = tuple(materialized["objective_upper_bounds"])  # type: ignore[arg-type]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "V9 run context must bind the normalized-HV objective box."
                ) from error
            if len(lower) != 2 or len(upper) != 2:
                raise ValueError("V9 normalized-HV objective box must be two-dimensional.")
            if any(
                type(value) not in {int, float} or not math.isfinite(float(value))
                for value in (*lower, *upper)
            ):
                raise TypeError(
                    "V9 normalized-HV objective bounds must be finite exact reals."
                )
            if any(float(lo) >= float(hi) for lo, hi in zip(lower, upper)):
                raise ValueError("V9 normalized-HV objective bounds must be nondegenerate.")
        raw = _canonical_bytes(materialized)
        object.__setattr__(self, "canonical_json", raw.decode("utf-8"))
        object.__setattr__(self, "digest_sha256", hashlib.sha256(raw).hexdigest())

    @property
    def payload(self) -> dict[str, object]:
        return dict(json.loads(self.canonical_json))


def build_v21e3_run_context(
    problem: MultiObjectiveCombinatorialProblem,
    *,
    case_artifact_sha256: str,
    candidate_id: str,
    algorithm_config: Mapping[str, object],
    algorithm_source_sha256: str,
    reference_directions: Sequence[Sequence[float]],
    seed: int,
    charged_evaluation_budget: int,
    evidence_partition: str,
    case_artifact_binding_kind: str = "explicit_case_artifact_sha256_v1",
) -> V21E3RunContext:
    config_payload = dict(algorithm_config)
    return V21E3RunContext(
        {
            "schema": "v21e3_run_context_v1",
            "case_artifact_sha256": str(case_artifact_sha256),
            "case_artifact_binding_kind": str(case_artifact_binding_kind),
            "problem_semantic_sha256": problem_sha256(problem),
            "candidate_id": str(candidate_id),
            "algorithm_config": config_payload,
            "candidate_config_sha256": hashlib.sha256(
                _canonical_bytes(config_payload)
            ).hexdigest(),
            "algorithm_source_sha256": str(algorithm_source_sha256),
            "reference_directions": tuple(
                tuple(float(value) for value in direction)
                for direction in reference_directions
            ),
            "seed": int(seed),
            "charged_evaluation_budget": int(charged_evaluation_budget),
            "evidence_partition": str(evidence_partition),
        }
    )


@dataclass(frozen=True)
class EvaluationContext:
    evidence_partition: str
    search_phase_id: str
    stage_id: str
    type_id: int | None
    operator_id: str
    operator_call_id: int
    parent_solutions: Tuple[Solution, ...] = ()
    parent_type_ids: Tuple[int, ...] = ()
    objective_evaluation_kind: str = "full_objective_vector_v1"
    feasible_before_repair: bool | None = None
    repair_applied: bool = False
    repair_operator_id: str | None = None
    local_search_block_id: int | None = None
    local_search_depth: int = 0
    operator_witness: object | None = None


@dataclass(frozen=True)
class AttemptOutcome:
    attempt_index: int
    proposal: Solution
    proposal_solution_ref: int
    proposal_sha256: str
    status: str
    cache_hit: bool
    charged_evaluation_index: int | None
    objectives: ObjectiveVector
    duplicate_of_evaluation_index: int | None
    elapsed_monotonic_ns: int

    @property
    def evaluation_index(self) -> int:
        if self.charged_evaluation_index is None:
            raise RuntimeError("A cache-hit attempt has no charged evaluation index.")
        return self.charged_evaluation_index


@dataclass(frozen=True)
class DecisionInput:
    accepted_into_population: bool
    population_replacement_count: int
    population_target_type_ids: Tuple[int, ...]
    decision_reason: str
    archive_changed: bool
    retained_after_update: bool
    archive_size_after: int
    scalarization_id: str | None = None
    scalar_parent: float | None = None
    scalar_candidate: float | None = None
    scalar_advantage: float | None = None
    cell_id: str | None = None
    new_evaluated_cell: bool | None = None
    new_nondominated_cell: bool | None = None
    policy_witness: object | None = None


def _attempt_semantic_payload(
    *,
    attempt_index: int,
    proposal_solution_ref: int | None,
    proposal_sha256: str | None,
    proposal_raw_json: str,
    proposal_raw_sha256: str,
    evaluation_context_json: str,
    status: str,
    physical_call_started: int,
    charged_evaluation_index: int | None,
    cache_source_evaluation_index: int | None,
    failure_code: str | None,
    failure_detail_json: str | None,
    run_context_digest_sha256: str,
    prev_attempt_sha256: str,
) -> dict[str, object]:
    return {
        "attempt_index": int(attempt_index),
        "proposal_solution_ref": proposal_solution_ref,
        "proposal_sha256": proposal_sha256,
        "proposal_raw": json.loads(proposal_raw_json),
        "proposal_raw_sha256": str(proposal_raw_sha256),
        "evaluation_context": json.loads(evaluation_context_json),
        "status": str(status),
        "physical_call_started": int(physical_call_started),
        "charged_evaluation_index": charged_evaluation_index,
        "cache_source_evaluation_index": cache_source_evaluation_index,
        "failure_code": failure_code,
        "failure_detail": (
            None if failure_detail_json is None else json.loads(failure_detail_json)
        ),
        "run_context_digest_sha256": str(run_context_digest_sha256),
        "prev_attempt_sha256": str(prev_attempt_sha256),
    }


class V21E3SQLiteLedger:
    """Own the raw evaluator and durably commit each returned objective vector."""

    def __init__(
        self,
        problem: MultiObjectiveCombinatorialProblem,
        *,
        run_context: V21E3RunContext | Mapping[str, object] | None,
        database_path: str | Path | None,
        receipt_path: str | Path | None = None,
        receipt_database_path: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self._problem_name = str(problem.name)
        self._family = self._infer_family(problem)
        self._solution_size = int(problem.solution_size)
        self._num_objectives = int(problem.num_objectives)
        self._objective_lower_bounds = tuple(
            float(value) for value in problem.objective_lower_bounds
        )
        self._objective_upper_bounds = tuple(
            float(value) for value in problem.objective_upper_bounds
        )
        self._validate_frozen_box()
        if run_context is None:
            raise ValueError("V21e3 requires an explicit canonical run context.")
        self._run_context = (
            run_context
            if isinstance(run_context, V21E3RunContext)
            else V21E3RunContext(run_context)
        )
        if self._run_context.payload["problem_semantic_sha256"] != problem_sha256(
            problem
        ):
            raise ValueError("V21e3 run context binds another problem semantic payload.")
        self._raw_evaluator: Callable[[Solution], ObjectiveVector] = problem.evaluate
        self._solution_validator = problem.validate_solution
        self._database_path = (
            None if database_path is None else Path(database_path).resolve()
        )
        self._receipt_path = (
            None if receipt_path is None else Path(receipt_path).resolve()
        )
        if receipt_database_path is not None and self._database_path is None:
            raise ValueError(
                "receipt_database_path requires a durable SQLite database."
            )
        self._receipt_database_path = (
            None
            if self._database_path is None
            else (
                str(self._database_path)
                if receipt_database_path is None
                else _portable_receipt_path(
                    receipt_database_path, field="receipt_database_path"
                )
            )
        )
        self._fault_injector = fault_injector
        if self._database_path is None:
            location = ":memory:"
        else:
            if self._database_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite V21e3 trace database: {self._database_path}"
                )
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            location = str(self._database_path)
        self._connection = sqlite3.connect(location)
        self._connection.execute("PRAGMA foreign_keys=ON")
        if self._database_path is not None:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self._connection.execute(
            """
            INSERT INTO run_attempt(
                run_id,problem,family,run_context_json,run_context_digest_sha256,status
            ) VALUES (1,?,?,?,?,'STARTED')
            """,
            (
                self._problem_name,
                self._family,
                self._run_context.canonical_json,
                self._run_context.digest_sha256,
            ),
        )
        self._connection.commit()
        self._attempt_count = 0
        self._physical_call_count = 0
        self._evaluation_count = 0
        self._decision_count = 0
        self._solution_ref_by_sha: dict[str, int] = {}
        self._solution_by_sha: dict[str, Solution] = {}
        self._evaluation_cache_by_solution_sha: dict[
            str, tuple[int, ObjectiveVector]
        ] = {}
        self._previous_attempt_hash = "0" * 64
        self._previous_evaluation_hash = "0" * 64
        self._previous_decision_hash = "0" * 64
        self._start_ns = time.monotonic_ns()
        self._terminal = False
        self._fault("after_run_start_commit")

    @classmethod
    def from_problem(
        cls,
        problem: MultiObjectiveCombinatorialProblem,
        *,
        run_context: V21E3RunContext | Mapping[str, object] | None,
        database_path: str | Path | None,
        receipt_path: str | Path | None = None,
        receipt_database_path: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> "V21E3SQLiteLedger":
        return cls(
            problem,
            run_context=run_context,
            database_path=database_path,
            receipt_path=receipt_path,
            receipt_database_path=receipt_database_path,
            fault_injector=fault_injector,
        )

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def physical_call_count(self) -> int:
        return self._physical_call_count

    @property
    def database_path(self) -> Path | None:
        return self._database_path

    def has_evaluated_solution(self, solution: Solution) -> bool:
        """Whether an exact canonical solution already has a charged evaluation.

        The query does not register the solution, mutate the database, or consume an
        objective call.  It is intended for bounded development-only candidate
        screening.
        """

        exact_solution = tuple(int(value) for value in solution)
        self._solution_validator(exact_solution)
        digest = _solution_sha256(exact_solution)
        cached_solution = self._solution_by_sha.get(digest)
        if cached_solution is not None and cached_solution != exact_solution:
            raise RuntimeError(
                "V21e3 detected a SHA-256 collision between distinct exact solutions."
            )
        return digest in self._evaluation_cache_by_solution_sha

    def attempt(
        self,
        proposal: Solution,
        context: EvaluationContext,
    ) -> AttemptOutcome:
        if self._terminal:
            raise RuntimeError("A terminal V21e3 ledger is immutable.")
        proposal_raw_json, proposal_raw_sha = _canonical_raw_proposal(proposal)
        parent_refs = tuple(
            self._register_solution(parent)[0] for parent in context.parent_solutions
        )
        context_payload = {
            "evidence_partition": context.evidence_partition,
            "search_phase_id": context.search_phase_id,
            "stage_id": context.stage_id,
            "type_id": context.type_id,
            "operator_id": context.operator_id,
            "operator_call_id": context.operator_call_id,
            "parent_solution_refs": parent_refs,
            "parent_type_ids": context.parent_type_ids,
            "objective_evaluation_kind": context.objective_evaluation_kind,
            "feasible_before_repair": context.feasible_before_repair,
            "repair_applied": context.repair_applied,
            "repair_operator_id": context.repair_operator_id,
            "local_search_block_id": context.local_search_block_id,
            "local_search_depth": context.local_search_depth,
            "operator_witness": context.operator_witness,
        }
        context_json = _canonical_bytes(context_payload).decode("utf-8")
        self._attempt_count += 1
        attempt_index = self._attempt_count
        elapsed_ns = time.monotonic_ns() - self._start_ns
        self._connection.execute(
            """
            INSERT INTO attempts(
                attempt_index,proposal_solution_ref,proposal_sha256,proposal_json,
                proposal_raw_sha256,context_json,status,physical_call_started,
                cache_source_evaluation_index,elapsed_monotonic_ns
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                attempt_index,
                None,
                None,
                proposal_raw_json,
                proposal_raw_sha,
                context_json,
                "PROPOSED",
                0,
                None,
                elapsed_ns,
            ),
        )
        self._connection.commit()
        self._fault("after_proposal_commit")
        expected_partition = str(
            self._run_context.payload["evidence_partition"]
        )
        if context.evidence_partition != expected_partition:
            detail = {
                "field": "evidence_partition",
                "expected": expected_partition,
                "observed": context.evidence_partition,
            }
            self._record_terminal_failure(
                failure_code=EvaluationContextError.code,
                failure_detail=detail,
                failed_attempt_index=attempt_index,
            )
            raise EvaluationContextError(
                "The attempt evidence partition disagrees with the frozen run context."
            )
        try:
            self._solution_validator(proposal)
        except Exception as error:
            detail = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "proposal_raw_sha256": proposal_raw_sha,
            }
            self._record_terminal_failure(
                failure_code=SolutionValidationError.code,
                failure_detail=detail,
                failed_attempt_index=attempt_index,
            )
            raise SolutionValidationError(
                f"The proposed solution failed validation: {error}"
            ) from error
        proposal_ref, registered_sha = self._register_solution(proposal)
        proposal_sha = registered_sha
        cached = self._evaluation_cache_by_solution_sha.get(proposal_sha)
        if cached is not None:
            cached_evaluation_index, objectives = cached
            attempt_semantic = _attempt_semantic_payload(
                attempt_index=attempt_index,
                proposal_solution_ref=proposal_ref,
                proposal_sha256=proposal_sha,
                proposal_raw_json=proposal_raw_json,
                proposal_raw_sha256=proposal_raw_sha,
                evaluation_context_json=context_json,
                status="CACHE_HIT",
                physical_call_started=0,
                charged_evaluation_index=None,
                cache_source_evaluation_index=cached_evaluation_index,
                failure_code=None,
                failure_detail_json=None,
                run_context_digest_sha256=self._run_context.digest_sha256,
                prev_attempt_sha256=self._previous_attempt_hash,
            )
            attempt_hash = hashlib.sha256(
                _canonical_bytes(attempt_semantic)
            ).hexdigest()
            self._connection.execute(
                """
                UPDATE attempts
                SET proposal_solution_ref=?,proposal_sha256=?,status='CACHE_HIT',
                    cache_source_evaluation_index=?,prev_attempt_sha256=?,
                    attempt_sha256=?
                WHERE attempt_index=?
                """,
                (
                    proposal_ref,
                    proposal_sha,
                    cached_evaluation_index,
                    self._previous_attempt_hash,
                    attempt_hash,
                    attempt_index,
                ),
            )
            self._connection.commit()
            self._previous_attempt_hash = attempt_hash
            self._fault("after_cache_hit_commit")
            return AttemptOutcome(
                attempt_index=attempt_index,
                proposal=tuple(proposal),
                proposal_solution_ref=proposal_ref,
                proposal_sha256=proposal_sha,
                status="CACHE_HIT",
                cache_hit=True,
                charged_evaluation_index=None,
                objectives=objectives,
                duplicate_of_evaluation_index=cached_evaluation_index,
                elapsed_monotonic_ns=elapsed_ns,
            )
        self._connection.execute(
            """
            UPDATE attempts
            SET proposal_solution_ref=?,proposal_sha256=?,status='OBJECTIVE_STARTED',
                physical_call_started=1
            WHERE attempt_index=?
            """,
            (proposal_ref, proposal_sha, attempt_index),
        )
        self._connection.commit()
        self._fault("after_objective_start_commit")
        self._physical_call_count += 1
        try:
            raw_objectives = self._raw_evaluator(proposal)
        except Exception as error:
            detail = {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
            self._record_terminal_failure(
                failure_code=ObjectiveEvaluationError.code,
                failure_detail=detail,
                failed_attempt_index=attempt_index,
            )
            raise ObjectiveEvaluationError(
                f"The objective evaluator raised: {error}"
            ) from error
        self._fault("after_raw_objective_return")
        try:
            objectives = self._validate_objectives(raw_objectives)
        except ObjectiveContractError as error:
            self._record_terminal_failure(
                failure_code=error.code,
                failure_detail=error.detail,
                failed_attempt_index=attempt_index,
            )
            raise
        self._evaluation_count += 1
        evaluation_index = self._evaluation_count
        semantic_payload = {
            "evaluation_index": evaluation_index,
            "attempt_index": attempt_index,
            "context": context_payload,
            "proposal_solution_ref": proposal_ref,
            "proposal_sha256": proposal_sha,
            "objectives": objectives,
            "run_context_digest_sha256": self._run_context.digest_sha256,
            "prev_record_sha256": self._previous_evaluation_hash,
        }
        record_hash = hashlib.sha256(_canonical_bytes(semantic_payload)).hexdigest()
        self._connection.execute(
            """
            INSERT INTO evaluations(
                evaluation_index,attempt_index,evidence_partition,search_phase_id,
                stage_id,type_id,operator_id,operator_call_id,
                proposal_solution_ref,proposal_sha256,objectives_json,
                prev_record_sha256,record_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                evaluation_index,
                attempt_index,
                context.evidence_partition,
                context.search_phase_id,
                context.stage_id,
                context.type_id,
                context.operator_id,
                context.operator_call_id,
                proposal_ref,
                proposal_sha,
                json.dumps(objectives, separators=(",", ":"), allow_nan=False),
                self._previous_evaluation_hash,
                record_hash,
            ),
        )
        attempt_semantic = _attempt_semantic_payload(
            attempt_index=attempt_index,
            proposal_solution_ref=proposal_ref,
            proposal_sha256=proposal_sha,
            proposal_raw_json=proposal_raw_json,
            proposal_raw_sha256=proposal_raw_sha,
            evaluation_context_json=context_json,
            status="EVALUATED",
            physical_call_started=1,
            charged_evaluation_index=evaluation_index,
            cache_source_evaluation_index=None,
            failure_code=None,
            failure_detail_json=None,
            run_context_digest_sha256=self._run_context.digest_sha256,
            prev_attempt_sha256=self._previous_attempt_hash,
        )
        attempt_hash = hashlib.sha256(_canonical_bytes(attempt_semantic)).hexdigest()
        self._connection.execute(
            "UPDATE attempts SET status='EVALUATED',charged_evaluation_index=? "
            ",prev_attempt_sha256=?,attempt_sha256=? WHERE attempt_index=?",
            (
                evaluation_index,
                self._previous_attempt_hash,
                attempt_hash,
                attempt_index,
            ),
        )
        self._fault("before_evaluation_commit")
        self._connection.commit()
        self._previous_attempt_hash = attempt_hash
        self._previous_evaluation_hash = record_hash
        self._evaluation_cache_by_solution_sha[proposal_sha] = (
            evaluation_index,
            objectives,
        )
        self._fault("after_evaluation_commit")
        return AttemptOutcome(
            attempt_index=attempt_index,
            proposal=tuple(proposal),
            proposal_solution_ref=proposal_ref,
            proposal_sha256=proposal_sha,
            status="EVALUATED",
            cache_hit=False,
            charged_evaluation_index=evaluation_index,
            objectives=objectives,
            duplicate_of_evaluation_index=None,
            elapsed_monotonic_ns=elapsed_ns,
        )

    def commit_decision(
        self,
        evaluation_index: int,
        decision: DecisionInput,
    ) -> None:
        if self._terminal:
            raise RuntimeError("A terminal V21e3 ledger is immutable.")
        if evaluation_index != self._decision_count + 1:
            raise RuntimeError(
                "V21e3 decisions must be durably committed in charged-evaluation order."
            )
        payload = {
            "evaluation_index": int(evaluation_index),
            "accepted_into_population": bool(decision.accepted_into_population),
            "population_replacement_count": int(
                decision.population_replacement_count
            ),
            "population_target_type_ids": tuple(
                int(value) for value in decision.population_target_type_ids
            ),
            "decision_reason": str(decision.decision_reason),
            "archive_changed": bool(decision.archive_changed),
            "retained_after_update": bool(decision.retained_after_update),
            "archive_size_after": int(decision.archive_size_after),
            "scalarization_id": decision.scalarization_id,
            "scalar_parent": decision.scalar_parent,
            "scalar_candidate": decision.scalar_candidate,
            "scalar_advantage": decision.scalar_advantage,
            "cell_id": decision.cell_id,
            "new_evaluated_cell": decision.new_evaluated_cell,
            "new_nondominated_cell": decision.new_nondominated_cell,
            "policy_witness": decision.policy_witness,
            "run_context_digest_sha256": self._run_context.digest_sha256,
            "prev_decision_sha256": self._previous_decision_hash,
        }
        decision_bytes = _canonical_bytes(payload)
        decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
        self._connection.execute(
            """
            INSERT INTO decisions(
                evaluation_index,decision_json,prev_decision_sha256,decision_sha256
            ) VALUES (?,?,?,?)
            """,
            (
                evaluation_index,
                decision_bytes.decode("utf-8"),
                self._previous_decision_hash,
                decision_sha256,
            ),
        )
        self._fault("before_decision_commit")
        self._connection.commit()
        self._decision_count += 1
        self._previous_decision_hash = decision_sha256
        self._fault("after_decision_commit")

    def finalize(
        self,
        *,
        expected_charged_evaluations: int,
        expected_decisions: int | None = None,
        resource_accounting: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if self._terminal:
            raise RuntimeError("The V21e3 ledger already has a terminal receipt.")
        expected_evaluations = int(expected_charged_evaluations)
        expected_decision_count = (
            expected_evaluations
            if expected_decisions is None
            else int(expected_decisions)
        )
        if expected_evaluations < 0 or expected_decision_count < 0:
            raise ValueError("Expected V21e3 terminal counts must be nonnegative.")
        context_budget = int(
            self._run_context.payload["charged_evaluation_budget"]
        )
        persisted_attempts = int(
            self._connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        )
        persisted_evaluations = int(
            self._connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        )
        persisted_decisions = int(
            self._connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        physical_starts = int(
            self._connection.execute(
                "SELECT COALESCE(SUM(physical_call_started),0) FROM attempts"
            ).fetchone()[0]
        )
        cache_hits = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE status='CACHE_HIT'"
            ).fetchone()[0]
        )
        nonterminal_attempts = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM attempts "
                "WHERE status NOT IN ('EVALUATED','CACHE_HIT') "
                "OR prev_attempt_sha256 IS NULL OR attempt_sha256 IS NULL"
            ).fetchone()[0]
        )
        evaluation_bounds = self._connection.execute(
            "SELECT MIN(evaluation_index),MAX(evaluation_index) FROM evaluations"
        ).fetchone()
        expected_bounds = (
            (None, None)
            if expected_evaluations == 0
            else (1, expected_evaluations)
        )
        integrity = str(
            self._connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        materialized_resources: dict[str, object] | None = None
        resource_gate_passed = True
        if resource_accounting is not None:
            if not isinstance(resource_accounting, Mapping):
                raise TypeError("V9 resource_accounting must be a mapping.")
            materialized_resources = dict(resource_accounting)
            required_resource_fields = {
                "schema",
                "first_evaluations",
                "first_evaluation_cap",
                "attempts",
                "attempt_cap",
                "structural_candidate_generations",
                "cache_membership_probes",
                "structural_screening_work",
                "structural_screening_cap",
                "elapsed_seconds",
                "wall_time_cap_seconds",
                "all_configured_caps_satisfied",
            }
            if set(materialized_resources) != required_resource_fields:
                raise ValueError(
                    "V9 resource_accounting has an incomplete or extended schema."
                )
            if materialized_resources["schema"] != (
                "v21e3r1_v9_ast_resource_accounting_v1"
            ):
                raise ValueError("Unsupported V9 resource-accounting schema.")
            integer_fields = (
                "first_evaluations",
                "first_evaluation_cap",
                "attempts",
                "attempt_cap",
                "structural_candidate_generations",
                "cache_membership_probes",
                "structural_screening_work",
                "structural_screening_cap",
            )
            if any(
                type(materialized_resources[field]) is not int
                or int(materialized_resources[field]) < 0
                for field in integer_fields
            ):
                raise TypeError(
                    "V9 resource counts and caps must be nonnegative exact integers."
                )
            elapsed = materialized_resources["elapsed_seconds"]
            if type(elapsed) not in {int, float} or not math.isfinite(float(elapsed)):
                raise TypeError("V9 resource elapsed_seconds must be a finite exact real.")
            if float(elapsed) < 0.0:
                raise ValueError("V9 resource elapsed_seconds must be nonnegative.")
            wall_cap = materialized_resources["wall_time_cap_seconds"]
            if wall_cap is not None and (
                type(wall_cap) not in {int, float}
                or not math.isfinite(float(wall_cap))
                or float(wall_cap) <= 0.0
            ):
                raise ValueError(
                    "V9 resource wall_time_cap_seconds must be null or a finite "
                    "positive exact real."
                )
            generated = int(
                materialized_resources["structural_candidate_generations"]
            )
            probes = int(materialized_resources["cache_membership_probes"])
            structural = int(materialized_resources["structural_screening_work"])
            resource_gate_passed = (
                int(materialized_resources["first_evaluations"])
                == persisted_evaluations
                and int(materialized_resources["attempts"]) == persisted_attempts
                and structural == generated + probes
                and int(materialized_resources["first_evaluations"])
                <= int(materialized_resources["first_evaluation_cap"])
                and int(materialized_resources["attempts"])
                <= int(materialized_resources["attempt_cap"])
                and structural
                <= int(materialized_resources["structural_screening_cap"])
                and (
                    wall_cap is None or float(elapsed) <= float(wall_cap)
                )
                and materialized_resources["all_configured_caps_satisfied"] is True
            )
        gates = {
            "expected_charged_evaluations": expected_evaluations,
            "expected_decisions": expected_decision_count,
            "run_context_charged_evaluation_budget": context_budget,
            "persisted_attempts": persisted_attempts,
            "persisted_evaluations": persisted_evaluations,
            "persisted_decisions": persisted_decisions,
            "physical_call_starts": physical_starts,
            "cache_hits": cache_hits,
            "nonterminal_attempts": nonterminal_attempts,
            "evaluation_index_bounds": list(evaluation_bounds),
            "expected_evaluation_index_bounds": list(expected_bounds),
            "sqlite_integrity": integrity,
        }
        if materialized_resources is not None:
            gates["resource_accounting"] = materialized_resources
            gates["resource_accounting_gate"] = (
                "PASS" if resource_gate_passed else "FAIL"
            )
        gate_passed = (
            expected_evaluations == context_budget
            and
            persisted_attempts == self._attempt_count
            and persisted_evaluations
            == self._evaluation_count
            == self._physical_call_count
            == physical_starts
            == expected_evaluations
            and persisted_decisions
            == self._decision_count
            == expected_decision_count
            and cache_hits == persisted_attempts - persisted_evaluations
            and nonterminal_attempts == 0
            and evaluation_bounds == expected_bounds
            and integrity == "ok"
            and resource_gate_passed
        )
        if not gate_passed:
            self._record_terminal_failure(
                failure_code="FINALIZATION_GATE_FAILED",
                failure_detail=gates,
                failed_attempt_index=None,
            )
            raise RuntimeError(f"V21e3 trace finalization gate failed: {gates}")
        receipt_core: dict[str, object] = {
            "schema": "v21e3_terminal_receipt_v1",
            "status": "SUCCESS",
            "problem": self._problem_name,
            "family": self._family,
            "failure_code": None,
            "failure_detail": None,
            "attempt_count": persisted_attempts,
            "physical_call_started_count": physical_starts,
            "charged_evaluation_count": persisted_evaluations,
            "decision_count": persisted_decisions,
            "cache_hit_count": cache_hits,
            "unresolved_decision_count": 0,
            "terminal_evaluation_chain_sha256": self._previous_evaluation_hash,
            "terminal_decision_chain_sha256": self._previous_decision_hash,
            "terminal_attempt_chain_sha256": self._previous_attempt_hash,
            "run_context_digest_sha256": self._run_context.digest_sha256,
            "database_path": self._receipt_database_path,
            "durability_mode": (
                "MEMORY_ONLY_NONFORMAL"
                if self._database_path is None
                else "SQLITE_WAL_SYNCHRONOUS_FULL"
            ),
            "finalization_gates": gates,
        }
        if materialized_resources is not None:
            receipt_core["resource_accounting"] = materialized_resources
        receipt_sha256 = hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest()
        receipt = {**receipt_core, "receipt_payload_sha256": receipt_sha256}
        receipt_json = _canonical_bytes(receipt).decode("utf-8")
        self._connection.execute(
            """
            INSERT INTO terminal_receipts(
                run_id,status,failure_code,receipt_json,receipt_sha256
            ) VALUES (1,'SUCCESS',NULL,?,?)
            """,
            (receipt_json, receipt_sha256),
        )
        self._connection.execute(
            """
            UPDATE run_attempt
            SET status='SUCCESS',terminal_receipt_sha256=?
            WHERE run_id=1
            """,
            (receipt_sha256,),
        )
        self._fault("before_terminal_commit")
        self._connection.commit()
        self._terminal = True
        self._fault("after_terminal_commit")
        self._write_receipt_sidecar(receipt)
        self._connection.close()
        return receipt

    def finalize_failure(
        self,
        *,
        failure_code: str,
        failure_detail: object,
    ) -> dict[str, object]:
        """Durably terminate an active ledger for an external fail-closed gate."""

        if type(failure_code) is not str or not failure_code:
            raise TypeError("failure_code must be a nonempty exact string.")
        return self._record_terminal_failure(
            failure_code=failure_code,
            failure_detail=failure_detail,
            failed_attempt_index=None,
        )

    def _validate_frozen_box(self) -> None:
        if self._num_objectives <= 0:
            raise ValueError("The frozen objective dimension must be positive.")
        if not (
            len(self._objective_lower_bounds)
            == len(self._objective_upper_bounds)
            == self._num_objectives
        ):
            raise ValueError("The frozen objective box has the wrong dimension.")
        for index, (lower, upper) in enumerate(
            zip(self._objective_lower_bounds, self._objective_upper_bounds)
        ):
            if not (math.isfinite(lower) and math.isfinite(upper)):
                raise ValueError(
                    f"Frozen objective box coordinate {index} is not finite."
                )
            if lower > upper:
                raise ValueError(
                    f"Frozen objective box coordinate {index} has lower > upper."
                )

    def _validate_objectives(self, raw_objectives: object) -> ObjectiveVector:
        try:
            raw_values = tuple(raw_objectives)  # type: ignore[arg-type]
        except TypeError as error:
            raise ObjectiveContractError(
                "OBJECTIVE_NOT_A_VECTOR",
                "The raw objective result is not an iterable vector.",
                detail={"raw_type": type(raw_objectives).__name__},
            ) from error
        if len(raw_values) != self._num_objectives:
            raise ObjectiveContractError(
                "OBJECTIVE_DIMENSION_MISMATCH",
                (
                    "The raw objective vector dimension does not match the "
                    "prospectively frozen dimension."
                ),
                detail={
                    "expected_dimension": self._num_objectives,
                    "observed_dimension": len(raw_values),
                    "raw_repr": repr(raw_values),
                },
            )
        if any(type(value) not in (int, float) for value in raw_values):
            raise ObjectiveContractError(
                "OBJECTIVE_NONNUMERIC",
                (
                    "At least one raw objective coordinate is not an exact "
                    "built-in integer or floating-point number."
                ),
                detail={"raw_repr": repr(raw_values)},
            )
        try:
            objectives = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ObjectiveContractError(
                "OBJECTIVE_NONNUMERIC",
                "At least one raw objective coordinate is not numeric.",
                detail={"raw_repr": repr(raw_values)},
            ) from error
        nonfinite = [
            index for index, value in enumerate(objectives) if not math.isfinite(value)
        ]
        if nonfinite:
            raise ObjectiveContractError(
                "OBJECTIVE_NONFINITE",
                "At least one raw objective coordinate is NaN or infinite.",
                detail={
                    "coordinate_indices": nonfinite,
                    "raw_repr": repr(raw_values),
                },
            )
        violations = [
            {
                "coordinate_index": index,
                "value": value,
                "lower": lower,
                "upper": upper,
            }
            for index, (value, lower, upper) in enumerate(
                zip(
                    objectives,
                    self._objective_lower_bounds,
                    self._objective_upper_bounds,
                )
            )
            if value < lower or value > upper
        ]
        if violations:
            raise ObjectiveContractError(
                "OBJECTIVE_OUT_OF_BOUNDS",
                "At least one raw objective coordinate is outside the frozen box.",
                detail={"violations": violations},
            )
        return objectives

    def _record_terminal_failure(
        self,
        *,
        failure_code: str,
        failure_detail: object,
        failed_attempt_index: int | None,
    ) -> dict[str, object]:
        if self._terminal:
            raise RuntimeError("The V21e3 ledger already has a terminal receipt.")
        terminal_attempt_hash = self._previous_attempt_hash
        if failed_attempt_index is not None:
            row = self._connection.execute(
                """
                SELECT proposal_solution_ref,proposal_sha256,proposal_json,
                       proposal_raw_sha256,context_json,physical_call_started,
                       charged_evaluation_index,cache_source_evaluation_index
                FROM attempts WHERE attempt_index=?
                """,
                (int(failed_attempt_index),),
            ).fetchone()
            if row is None:
                raise RuntimeError("The failed V21e3 attempt row is missing.")
            failure_detail_json = _canonical_bytes(failure_detail).decode("utf-8")
            semantic = _attempt_semantic_payload(
                attempt_index=int(failed_attempt_index),
                proposal_solution_ref=(None if row[0] is None else int(row[0])),
                proposal_sha256=(None if row[1] is None else str(row[1])),
                proposal_raw_json=str(row[2]),
                proposal_raw_sha256=str(row[3]),
                evaluation_context_json=str(row[4]),
                status="FAILED",
                physical_call_started=int(row[5]),
                charged_evaluation_index=(None if row[6] is None else int(row[6])),
                cache_source_evaluation_index=(
                    None if row[7] is None else int(row[7])
                ),
                failure_code=str(failure_code),
                failure_detail_json=failure_detail_json,
                run_context_digest_sha256=self._run_context.digest_sha256,
                prev_attempt_sha256=self._previous_attempt_hash,
            )
            terminal_attempt_hash = hashlib.sha256(
                _canonical_bytes(semantic)
            ).hexdigest()
            self._connection.execute(
                """
                UPDATE attempts
                SET status='FAILED',failure_code=?,failure_detail_json=?,
                    prev_attempt_sha256=?,attempt_sha256=?
                WHERE attempt_index=?
                """,
                (
                    str(failure_code),
                    failure_detail_json,
                    self._previous_attempt_hash,
                    terminal_attempt_hash,
                    int(failed_attempt_index),
                ),
            )
        cache_hit_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE status='CACHE_HIT'"
            ).fetchone()[0]
        )
        receipt_core: dict[str, object] = {
            "schema": "v21e3_terminal_receipt_v1",
            "status": "FAILURE",
            "problem": self._problem_name,
            "family": self._family,
            "failure_code": str(failure_code),
            "failure_detail": failure_detail,
            "attempt_count": self._attempt_count,
            "physical_call_started_count": self._physical_call_count,
            "charged_evaluation_count": self._evaluation_count,
            "decision_count": self._decision_count,
            "cache_hit_count": cache_hit_count,
            "unresolved_decision_count": self._evaluation_count
            - self._decision_count,
            "terminal_evaluation_chain_sha256": self._previous_evaluation_hash,
            "terminal_decision_chain_sha256": self._previous_decision_hash,
            "terminal_attempt_chain_sha256": terminal_attempt_hash,
            "run_context_digest_sha256": self._run_context.digest_sha256,
            "database_path": self._receipt_database_path,
        }
        receipt_sha256 = hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest()
        receipt = {**receipt_core, "receipt_payload_sha256": receipt_sha256}
        receipt_json = _canonical_bytes(receipt).decode("utf-8")
        self._connection.execute(
            """
            INSERT INTO terminal_receipts(
                run_id,status,failure_code,receipt_json,receipt_sha256
            ) VALUES (1,'FAILURE',?,?,?)
            """,
            (str(failure_code), receipt_json, receipt_sha256),
        )
        self._connection.execute(
            """
            UPDATE run_attempt
            SET status='FAILURE',terminal_receipt_sha256=?
            WHERE run_id=1
            """,
            (receipt_sha256,),
        )
        self._fault("before_terminal_commit")
        self._connection.commit()
        self._previous_attempt_hash = terminal_attempt_hash
        self._terminal = True
        self._fault("after_terminal_commit")
        self._write_receipt_sidecar(receipt)
        return receipt

    def _fault(self, boundary: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(str(boundary))

    def _write_receipt_sidecar(self, receipt: object) -> None:
        if self._receipt_path is None:
            return
        _atomic_write_json(self._receipt_path, receipt)

    def _register_solution(self, solution: Solution) -> tuple[int, str]:
        exact_solution = tuple(int(value) for value in solution)
        digest = _solution_sha256(exact_solution)
        cached = self._solution_ref_by_sha.get(digest)
        if cached is not None:
            if self._solution_by_sha[digest] != exact_solution:
                raise RuntimeError(
                    "V21e3 detected a SHA-256 collision between distinct "
                    "exact solution tuples."
                )
            return cached, digest
        codec, payload = self._encode_solution(exact_solution)
        cursor = self._connection.execute(
            """
            INSERT INTO solutions(solution_sha256,family,codec,solution_size,payload)
            VALUES (?,?,?,?,?)
            """,
            (digest, self._family, codec, len(solution), payload),
        )
        solution_ref = int(cursor.lastrowid)
        self._solution_ref_by_sha[digest] = solution_ref
        self._solution_by_sha[digest] = exact_solution
        return solution_ref, digest

    def _encode_solution(self, solution: Sequence[int]) -> tuple[str, bytes]:
        if self._family == "MOKP":
            payload = bytearray((len(solution) + 7) // 8)
            for index, value in enumerate(solution):
                if int(value):
                    payload[index // 8] |= 1 << (index % 8)
            return "mokp-bitpack-lsb-v1", bytes(payload)
        if self._family == "MOTSP" and max(solution, default=0) < 2**16:
            return "motsp-uint16le-v1", struct.pack(f"<{len(solution)}H", *solution)
        if self._family == "MOTSP":
            return "motsp-uint32le-v1", struct.pack(f"<{len(solution)}I", *solution)
        return "generic-json-v1", _canonical_bytes([int(value) for value in solution])

    @staticmethod
    def _infer_family(problem: MultiObjectiveCombinatorialProblem) -> str:
        if isinstance(problem, MultiObjectiveKnapsackInstance):
            return "MOKP"
        if isinstance(problem, MultiObjectiveTSPProblemAdapter):
            return "MOTSP"
        return "GENERIC"

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE solutions(
                solution_ref INTEGER PRIMARY KEY,
                solution_sha256 TEXT NOT NULL UNIQUE,
                family TEXT NOT NULL,
                codec TEXT NOT NULL,
                solution_size INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE attempts(
                attempt_index INTEGER PRIMARY KEY,
                proposal_solution_ref INTEGER REFERENCES solutions(solution_ref),
                proposal_sha256 TEXT,
                proposal_json TEXT NOT NULL,
                proposal_raw_sha256 TEXT NOT NULL,
                context_json TEXT NOT NULL,
                status TEXT NOT NULL,
                physical_call_started INTEGER NOT NULL,
                charged_evaluation_index INTEGER,
                cache_source_evaluation_index INTEGER,
                failure_code TEXT,
                failure_detail_json TEXT,
                elapsed_monotonic_ns INTEGER NOT NULL,
                prev_attempt_sha256 TEXT,
                attempt_sha256 TEXT UNIQUE
            );
            CREATE TABLE evaluations(
                evaluation_index INTEGER PRIMARY KEY,
                attempt_index INTEGER NOT NULL UNIQUE REFERENCES attempts(attempt_index),
                evidence_partition TEXT NOT NULL,
                search_phase_id TEXT NOT NULL,
                stage_id TEXT NOT NULL,
                type_id INTEGER,
                operator_id TEXT NOT NULL,
                operator_call_id INTEGER NOT NULL,
                proposal_solution_ref INTEGER NOT NULL REFERENCES solutions(solution_ref),
                proposal_sha256 TEXT NOT NULL,
                objectives_json TEXT NOT NULL,
                prev_record_sha256 TEXT NOT NULL,
                record_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE UNIQUE INDEX evaluations_unique_proposal_sha256
            ON evaluations(proposal_sha256);
            CREATE TABLE decisions(
                evaluation_index INTEGER PRIMARY KEY REFERENCES evaluations(evaluation_index),
                decision_json TEXT NOT NULL,
                prev_decision_sha256 TEXT NOT NULL,
                decision_sha256 TEXT NOT NULL UNIQUE
            );
            CREATE TABLE run_attempt(
                run_id INTEGER PRIMARY KEY CHECK(run_id=1),
                problem TEXT NOT NULL,
                family TEXT NOT NULL,
                run_context_json TEXT NOT NULL,
                run_context_digest_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                terminal_receipt_sha256 TEXT
            );
            CREATE TABLE terminal_receipts(
                run_id INTEGER PRIMARY KEY REFERENCES run_attempt(run_id),
                status TEXT NOT NULL,
                failure_code TEXT,
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL UNIQUE
            );
            """
        )


def recover_v21e3_terminal_receipt(
    database_path: str | Path,
    receipt_path: str | Path | None = None,
    *,
    failure_code: str = "PROCESS_INTERRUPTION",
    receipt_database_path: str | None = None,
) -> dict[str, object]:
    """Idempotently terminalize an interrupted V21e3 run as a failure."""

    resolved_database = Path(database_path).resolve()
    if not resolved_database.is_file():
        raise FileNotFoundError(
            f"Cannot recover missing V21e3 trace database: {resolved_database}"
        )
    resolved_receipt = (
        None if receipt_path is None else Path(receipt_path).resolve()
    )
    displayed_database_path = (
        str(resolved_database)
        if receipt_database_path is None
        else _portable_receipt_path(
            receipt_database_path, field="receipt_database_path"
        )
    )
    with sqlite3.connect(resolved_database) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        existing = connection.execute(
            "SELECT receipt_json,receipt_sha256 FROM terminal_receipts WHERE run_id=1"
        ).fetchone()
        if existing is not None:
            receipt = json.loads(str(existing[0]))
            claimed = str(existing[1])
            receipt_core = dict(receipt)
            embedded = receipt_core.pop("receipt_payload_sha256", None)
            observed = hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest()
            if embedded != claimed or observed != claimed:
                raise ValueError("Persisted V21e3 terminal receipt hash mismatch.")
            if (
                receipt_database_path is not None
                and receipt.get("database_path") != displayed_database_path
            ):
                raise ValueError(
                    "Persisted V21e3 terminal receipt uses another database display path."
                )
            if resolved_receipt is not None:
                _atomic_write_json(resolved_receipt, receipt)
            return receipt

        run_row = connection.execute(
            """
            SELECT problem,family,run_context_json,run_context_digest_sha256,status
            FROM run_attempt WHERE run_id=1
            """
        ).fetchone()
        if run_row is None:
            raise ValueError("The database is not a V21e3 run-attempt ledger.")
        run_context_json = str(run_row["run_context_json"])
        run_context_digest = str(run_row["run_context_digest_sha256"])
        if hashlib.sha256(run_context_json.encode("utf-8")).hexdigest() != run_context_digest:
            raise ValueError("Interrupted V21e3 run-context digest mismatch.")
        if _canonical_bytes(json.loads(run_context_json)).decode("utf-8") != run_context_json:
            raise ValueError("Interrupted V21e3 run context is not canonical.")
        if str(run_row["status"]) != "STARTED":
            raise ValueError(
                "V21e3 run status is terminal but its receipt row is missing."
            )
        attempt_count = int(
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        )
        physical_starts = int(
            connection.execute(
                "SELECT COALESCE(SUM(physical_call_started),0) FROM attempts"
            ).fetchone()[0]
        )
        evaluation_count = int(
            connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        )
        decision_count = int(
            connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        cache_hits = int(
            connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE status='CACHE_HIT'"
            ).fetchone()[0]
        )
        last_evaluation = connection.execute(
            "SELECT record_sha256 FROM evaluations ORDER BY evaluation_index DESC LIMIT 1"
        ).fetchone()
        last_decision = connection.execute(
            "SELECT decision_sha256 FROM decisions ORDER BY evaluation_index DESC LIMIT 1"
        ).fetchone()
        terminal_evaluation_hash = (
            "0" * 64 if last_evaluation is None else str(last_evaluation[0])
        )
        terminal_decision_hash = (
            "0" * 64 if last_decision is None else str(last_decision[0])
        )
        attempt_rows = list(
            connection.execute("SELECT * FROM attempts ORDER BY attempt_index")
        )
        incomplete_attempts = [
            int(row["attempt_index"])
            for row in attempt_rows
            if str(row["status"]) not in {"EVALUATED", "CACHE_HIT", "FAILED"}
        ]
        recovery_detail = {
            "recovered_after_interruption": True,
            "incomplete_attempt_indices": incomplete_attempts,
        }
        previous_attempt_hash = "0" * 64
        recovery_detail_json = _canonical_bytes(recovery_detail).decode("utf-8")
        for expected_index, row in enumerate(attempt_rows, start=1):
            if int(row["attempt_index"]) != expected_index:
                raise ValueError("Interrupted V21e3 attempt indices are not contiguous.")
            incomplete = int(row["attempt_index"]) in incomplete_attempts
            semantic = _attempt_semantic_payload(
                attempt_index=int(row["attempt_index"]),
                proposal_solution_ref=(
                    None
                    if row["proposal_solution_ref"] is None
                    else int(row["proposal_solution_ref"])
                ),
                proposal_sha256=(
                    None
                    if row["proposal_sha256"] is None
                    else str(row["proposal_sha256"])
                ),
                proposal_raw_json=str(row["proposal_json"]),
                proposal_raw_sha256=str(row["proposal_raw_sha256"]),
                evaluation_context_json=str(row["context_json"]),
                status=("FAILED" if incomplete else str(row["status"])),
                physical_call_started=int(row["physical_call_started"]),
                charged_evaluation_index=(
                    None
                    if row["charged_evaluation_index"] is None
                    else int(row["charged_evaluation_index"])
                ),
                cache_source_evaluation_index=(
                    None
                    if row["cache_source_evaluation_index"] is None
                    else int(row["cache_source_evaluation_index"])
                ),
                failure_code=(
                    str(failure_code)
                    if incomplete
                    else (
                        None if row["failure_code"] is None else str(row["failure_code"])
                    )
                ),
                failure_detail_json=(
                    recovery_detail_json
                    if incomplete
                    else (
                        None
                        if row["failure_detail_json"] is None
                        else str(row["failure_detail_json"])
                    )
                ),
                run_context_digest_sha256=run_context_digest,
                prev_attempt_sha256=previous_attempt_hash,
            )
            observed_attempt_hash = hashlib.sha256(
                _canonical_bytes(semantic)
            ).hexdigest()
            if incomplete:
                connection.execute(
                    """
                    UPDATE attempts
                    SET status='FAILED',failure_code=?,failure_detail_json=?,
                        prev_attempt_sha256=?,attempt_sha256=?
                    WHERE attempt_index=?
                    """,
                    (
                        str(failure_code),
                        recovery_detail_json,
                        previous_attempt_hash,
                        observed_attempt_hash,
                        int(row["attempt_index"]),
                    ),
                )
            elif (
                str(row["prev_attempt_sha256"]) != previous_attempt_hash
                or str(row["attempt_sha256"]) != observed_attempt_hash
            ):
                raise ValueError("Interrupted V21e3 attempt hash chain mismatch.")
            previous_attempt_hash = observed_attempt_hash
        receipt_core: dict[str, object] = {
            "schema": "v21e3_terminal_receipt_v1",
            "status": "FAILURE",
            "problem": str(run_row["problem"]),
            "family": str(run_row["family"]),
            "failure_code": str(failure_code),
            "failure_detail": recovery_detail,
            "attempt_count": attempt_count,
            "physical_call_started_count": physical_starts,
            "charged_evaluation_count": evaluation_count,
            "decision_count": decision_count,
            "cache_hit_count": cache_hits,
            "unresolved_decision_count": evaluation_count - decision_count,
            "terminal_evaluation_chain_sha256": terminal_evaluation_hash,
            "terminal_decision_chain_sha256": terminal_decision_hash,
            "terminal_attempt_chain_sha256": previous_attempt_hash,
            "run_context_digest_sha256": run_context_digest,
            "recovered_after_interruption": True,
            "database_path": displayed_database_path,
            "durability_mode": "SQLITE_WAL_SYNCHRONOUS_FULL",
        }
        receipt_sha256 = hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest()
        receipt = {**receipt_core, "receipt_payload_sha256": receipt_sha256}
        receipt_json = _canonical_bytes(receipt).decode("utf-8")
        connection.execute(
            """
            INSERT INTO terminal_receipts(
                run_id,status,failure_code,receipt_json,receipt_sha256
            ) VALUES (1,'FAILURE',?,?,?)
            """,
            (str(failure_code), receipt_json, receipt_sha256),
        )
        connection.execute(
            """
            UPDATE run_attempt
            SET status='FAILURE',terminal_receipt_sha256=?
            WHERE run_id=1
            """,
            (receipt_sha256,),
        )
        connection.commit()
    if resolved_receipt is not None:
        _atomic_write_json(resolved_receipt, receipt)
    return receipt


SQLiteEvaluationLedger = V21E3SQLiteLedger
LedgerEvaluation = AttemptOutcome


__all__ = [
    "AttemptOutcome",
    "DecisionInput",
    "EvaluationContextError",
    "EvaluationContext",
    "LedgerEvaluation",
    "ObjectiveContractError",
    "ObjectiveEvaluationError",
    "SQLiteEvaluationLedger",
    "SolutionValidationError",
    "V21E3SQLiteLedger",
    "V21E3RunContext",
    "build_v21e3_run_context",
    "recover_v21e3_terminal_receipt",
]
